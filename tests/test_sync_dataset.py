"""Sync script: the repository dataset is what gets uploaded."""

from test_langsmith import _dataset

from nfe_document_extraction.evaluation.langsmith import LangSmithDatasetStore
from scripts import sync_dataset


def test_sync_script_requires_a_langsmith_key(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sync_dataset, "load_env_file", lambda path: None)

    code = sync_dataset.main([])

    assert code == 2
    assert "LANGSMITH_API_KEY" in capsys.readouterr().err


def test_sync_script_uploads_every_rendition(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(sync_dataset, "load_env_file", lambda path: None)
    monkeypatch.setattr(sync_dataset, "Client", lambda: object())
    seen: dict[str, object] = {}

    def fake_sync(store: object, documents: tuple[object, ...], *, name: str) -> str:
        seen["store"] = store
        seen["count"] = len(documents)
        seen["name"] = name
        return "2026-10-05T12:00:00+00:00"

    monkeypatch.setattr(sync_dataset, "sync_dataset", fake_sync)

    code = sync_dataset.main(
        ["--dataset", str(_dataset(tmp_path)), "--name", "nfe-synthetic"]
    )

    assert code == 0
    assert isinstance(seen["store"], LangSmithDatasetStore)
    assert seen["count"] == 3
    assert seen["name"] == "nfe-synthetic"
    assert "2026-10-05T12:00:00+00:00" in capsys.readouterr().out
