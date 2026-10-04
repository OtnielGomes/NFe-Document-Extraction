"""Load a local .env file without overriding the process environment."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Set missing environment variables from ``path``.

    Blank lines and ``#`` comments are ignored. ``export KEY=value`` is
    accepted. Empty values are skipped. A variable already set in the
    environment is left unchanged.
    """
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        if separator == "":
            continue
        name = key.strip()
        if name == "" or name in os.environ:
            continue
        parsed = _unquote(value.strip())
        if parsed:
            os.environ[name] = parsed


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
