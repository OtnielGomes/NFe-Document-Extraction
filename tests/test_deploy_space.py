"""Deploy script: a Docker Space on free CPU, secrets only on the Hub."""

from pathlib import Path

import pytest

from scripts import deploy_space

_SECRET_NAMES = (
    "HF_TOKEN",
    "HF_SPACE_ID",
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
)


@pytest.mark.parametrize("missing", _SECRET_NAMES)
def test_deploy_stops_when_a_repository_secret_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    monkeypatch.setattr(deploy_space, "load_env_file", lambda path: None)
    for name in _SECRET_NAMES:
        monkeypatch.setenv(name, "present")
    monkeypatch.setenv(missing, "  ")

    def boom(token: str) -> object:
        raise AssertionError(token)

    monkeypatch.setattr(deploy_space, "HuggingFaceSpaceHub", boom)

    code = deploy_space.main()

    assert code == 2
    assert missing in capsys.readouterr().err


def test_main_publishes_this_repository_with_the_repository_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deploy_space, "load_env_file", lambda path: None)
    monkeypatch.setenv("HF_TOKEN", "hf-token-value")
    monkeypatch.setenv("HF_SPACE_ID", "user/nfe-document-extraction")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret-value")
    monkeypatch.setenv("LANGSMITH_API_KEY", "langsmith-secret-value")
    seen: dict[str, object] = {}

    class _Hub:
        def __init__(self, token: str) -> None:
            seen["token"] = token

    def fake_publish(
        hub: object,
        *,
        repo_id: str,
        root: Path,
        openai_api_key: str,
        langsmith_api_key: str,
    ) -> int:
        seen["hub"] = hub
        seen["repo_id"] = repo_id
        seen["root"] = root
        seen["openai"] = openai_api_key
        seen["langsmith"] = langsmith_api_key
        return 0

    monkeypatch.setattr(deploy_space, "HuggingFaceSpaceHub", _Hub)
    monkeypatch.setattr(deploy_space, "publish", fake_publish)

    code = deploy_space.main()

    root = seen["root"]
    assert code == 0
    assert seen["token"] == "hf-token-value"
    assert isinstance(seen["hub"], _Hub)
    assert seen["repo_id"] == "user/nfe-document-extraction"
    assert isinstance(root, Path)
    assert (root / "Dockerfile").is_file()
    assert (root / "data" / "synthetic").is_dir()
    assert seen["openai"] == "openai-secret-value"
    assert seen["langsmith"] == "langsmith-secret-value"


class _RecordingHub:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def create_docker_space(self, repo_id: str) -> None:
        self.events.append(("create", repo_id))

    def add_secret(self, repo_id: str, key: str, value: str) -> None:
        self.events.append(("secret", repo_id, key, value))

    def use_free_cpu(self, repo_id: str) -> None:
        self.events.append(("cpu-basic", repo_id))

    def upload(self, repo_id: str, folder: Path) -> None:
        self.events.append(("upload", repo_id, _snapshot(folder)))


def test_publish_uploads_a_docker_space_that_sleeps_on_free_cpu(tmp_path: Path) -> None:
    root = _image_root(tmp_path)
    (root / ".env").write_text("OPENAI_API_KEY=openai-secret-value\n", encoding="utf-8")
    (root / "notes.txt").write_text("openai-secret-value", encoding="utf-8")
    (root / "src" / "cached.pyc").write_text("openai-secret-value", encoding="utf-8")
    cache = root / "src" / "__pycache__"
    cache.mkdir()
    (cache / "app.pyc").write_text("openai-secret-value", encoding="utf-8")
    hub = _RecordingHub()

    code = deploy_space.publish(
        hub,
        repo_id="user/nfe-document-extraction",
        root=root,
        openai_api_key="openai-secret-value",
        langsmith_api_key="langsmith-secret-value",
    )

    assert code == 0
    assert [event[0] for event in hub.events] == [
        "create",
        "secret",
        "secret",
        "cpu-basic",
        "upload",
    ]
    assert hub.events[0] == ("create", "user/nfe-document-extraction")
    assert hub.events[1] == (
        "secret",
        "user/nfe-document-extraction",
        "OPENAI_API_KEY",
        "openai-secret-value",
    )
    assert hub.events[2] == (
        "secret",
        "user/nfe-document-extraction",
        "LANGSMITH_API_KEY",
        "langsmith-secret-value",
    )
    assert hub.events[3] == ("cpu-basic", "user/nfe-document-extraction")
    uploaded = hub.events[4][2]
    assert isinstance(uploaded, dict)
    assert "sdk: docker" in uploaded["README.md"]
    assert "app_port: 7860" in uploaded["README.md"]
    assert uploaded["Dockerfile"] == "FROM gallery\n"
    assert ".dockerignore" in uploaded
    assert "pyproject.toml" in uploaded
    assert "uv.lock" in uploaded
    assert uploaded["src/app.py"] == "cached-gallery\n"
    assert uploaded["data/synthetic/synthetic-001/marker.txt"] == "gallery\n"
    assert ".env" not in uploaded
    assert "notes.txt" not in uploaded
    assert "src/cached.pyc" not in uploaded
    assert "src/__pycache__/app.pyc" not in uploaded
    blob = "\n".join(uploaded.values())
    assert "openai-secret-value" not in blob
    assert "langsmith-secret-value" not in blob


def test_hub_adapter_requests_a_docker_space_on_free_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _RecordingApi()
    seen: dict[str, str] = {}

    def factory(token: str) -> _RecordingApi:
        seen["token"] = token
        return api

    monkeypatch.setattr(deploy_space, "_hf_api", factory)
    folder = tmp_path / "stage"
    folder.mkdir()

    hub = deploy_space.HuggingFaceSpaceHub("hf-token-value")
    hub.create_docker_space("user/nfe-document-extraction")
    hub.add_secret(
        "user/nfe-document-extraction", "OPENAI_API_KEY", "openai-secret-value"
    )
    hub.use_free_cpu("user/nfe-document-extraction")
    hub.upload("user/nfe-document-extraction", folder)

    assert api.calls == [
        (
            "create_repo",
            {
                "repo_id": "user/nfe-document-extraction",
                "repo_type": "space",
                "space_sdk": "docker",
                "space_hardware": "cpu-basic",
                "exist_ok": True,
            },
        ),
        (
            "add_space_secret",
            {
                "repo_id": "user/nfe-document-extraction",
                "key": "OPENAI_API_KEY",
                "value": "openai-secret-value",
            },
        ),
        (
            "request_space_hardware",
            {
                "repo_id": "user/nfe-document-extraction",
                "hardware": "cpu-basic",
            },
        ),
        (
            "upload_folder",
            {
                "repo_id": "user/nfe-document-extraction",
                "folder_path": folder,
                "repo_type": "space",
                "commit_message": "Deploy Space",
            },
        ),
    ]
    assert seen["token"] == "hf-token-value"
    assert "hf-token-value" not in str(api.calls)
    assert "sleep_time" not in api.calls[2][1]


class _RecordingApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_repo(self, **kwargs: object) -> None:
        self.calls.append(("create_repo", kwargs))

    def add_space_secret(self, **kwargs: object) -> None:
        self.calls.append(("add_space_secret", kwargs))

    def request_space_hardware(self, **kwargs: object) -> None:
        self.calls.append(("request_space_hardware", kwargs))

    def upload_folder(self, **kwargs: object) -> None:
        self.calls.append(("upload_folder", kwargs))


def test_publish_stops_when_the_image_inputs_are_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = _RecordingHub()

    code = deploy_space.publish(
        hub,
        repo_id="user/nfe-document-extraction",
        root=tmp_path,
        openai_api_key="openai-secret-value",
        langsmith_api_key="langsmith-secret-value",
    )

    assert code == 2
    assert hub.events == []
    assert "Dockerfile" in capsys.readouterr().err


def _image_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "Dockerfile").write_text("FROM gallery\n", encoding="utf-8")
    (root / ".dockerignore").write_text("*.md\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='nfe'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    source = root / "src"
    source.mkdir()
    (source / "app.py").write_text("cached-gallery\n", encoding="utf-8")
    marker = root / "data" / "synthetic" / "synthetic-001"
    marker.mkdir(parents=True)
    (marker / "marker.txt").write_text("gallery\n", encoding="utf-8")
    return root


def _snapshot(folder: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in folder.rglob("*"):
        if path.is_file():
            files[path.relative_to(folder).as_posix()] = path.read_text(
                encoding="utf-8"
            )
    return files
