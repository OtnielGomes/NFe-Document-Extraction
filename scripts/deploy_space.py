"""Publish the demo container to a Hugging Face Docker Space.

Repository secrets stay in the environment. The Space receives the OpenAI and
LangSmith keys as Space secrets. The Hugging Face token only authenticates
this publish. Free CPU sleeps after inactivity; the cached gallery is in the
image, so a wake does not call a model.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Protocol

from nfe_document_extraction.env import load_env_file

_ROOT = Path(__file__).resolve().parent.parent
_REQUIRED = (
    "HF_TOKEN",
    "HF_SPACE_ID",
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
)
_IMAGE_FILES = ("Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock")
_IMAGE_DIRS = ("src", "data/synthetic")
_SPACE_README = """\
---
title: NFe Document Extraction
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

The gallery is cached synthetic invoices. It is ready when this Space wakes on free CPU, before any model call.
"""


class SpaceHub(Protocol):
    def create_docker_space(self, repo_id: str) -> None: ...

    def add_secret(self, repo_id: str, key: str, value: str) -> None: ...

    def use_free_cpu(self, repo_id: str) -> None: ...

    def upload(self, repo_id: str, folder: Path) -> None: ...


class HuggingFaceSpaceHub:
    """Hugging Face Hub calls for one publish."""

    def __init__(self, token: str) -> None:
        self._api = _hf_api(token)

    def create_docker_space(self, repo_id: str) -> None:
        self._api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            space_hardware="cpu-basic",
            exist_ok=True,
        )

    def add_secret(self, repo_id: str, key: str, value: str) -> None:
        self._api.add_space_secret(repo_id=repo_id, key=key, value=value)

    def use_free_cpu(self, repo_id: str) -> None:
        """Free CPU sleeps after inactivity. Sleep time cannot be set there."""
        self._api.request_space_hardware(repo_id=repo_id, hardware="cpu-basic")

    def upload(self, repo_id: str, folder: Path) -> None:
        self._api.upload_folder(
            repo_id=repo_id,
            folder_path=folder,
            repo_type="space",
            commit_message="Deploy Space",
        )


def _hf_api(token: str) -> object:
    from huggingface_hub import HfApi

    return HfApi(token=token)


def main() -> int:
    load_env_file(_ROOT / ".env")
    missing = tuple(name for name in _REQUIRED if not os.environ.get(name, "").strip())
    if missing:
        print(
            "Missing repository secrets: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    hub = HuggingFaceSpaceHub(os.environ["HF_TOKEN"].strip())
    return publish(
        hub,
        repo_id=os.environ["HF_SPACE_ID"].strip(),
        root=_ROOT,
        openai_api_key=os.environ["OPENAI_API_KEY"].strip(),
        langsmith_api_key=os.environ["LANGSMITH_API_KEY"].strip(),
    )


def publish(
    hub: SpaceHub,
    *,
    repo_id: str,
    root: Path,
    openai_api_key: str,
    langsmith_api_key: str,
) -> int:
    """Create or update the Docker Space and upload the image inputs."""
    missing = _missing_inputs(root)
    if missing:
        print(
            "Missing Space image inputs: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    with tempfile.TemporaryDirectory() as raw:
        destination = Path(raw)
        _stage(root, destination)
        hub.create_docker_space(repo_id)
        hub.add_secret(repo_id, "OPENAI_API_KEY", openai_api_key)
        hub.add_secret(repo_id, "LANGSMITH_API_KEY", langsmith_api_key)
        hub.use_free_cpu(repo_id)
        hub.upload(repo_id, destination)
    print(f"Deployed https://huggingface.co/spaces/{repo_id}", flush=True)
    return 0


def _missing_inputs(root: Path) -> tuple[str, ...]:
    missing: list[str] = []
    for name in _IMAGE_FILES:
        if not (root / name).is_file():
            missing.append(name)
    for name in _IMAGE_DIRS:
        if not (root / name).is_dir():
            missing.append(name)
    return tuple(missing)


def _stage(root: Path, destination: Path) -> None:
    for name in _IMAGE_FILES:
        shutil.copy2(root / name, destination / name)
    for name in _IMAGE_DIRS:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            root / name,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env"),
        )
    (destination / "README.md").write_text(_SPACE_README, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
