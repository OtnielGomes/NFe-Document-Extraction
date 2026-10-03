"""Cross-platform quality gate: format, lint, strict type check, tests."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckStep:
    name: str
    command: tuple[str, ...]


Runner = Callable[[CheckStep], int]


def run_checks(steps: Sequence[CheckStep], *, runner: Runner | None = None) -> int:
    execute = runner or _run_command
    for step in steps:
        print(f"==> {step.name}", flush=True)
        code = execute(step)
        if code != 0:
            print(f"Quality check failed: {step.name}", file=sys.stderr)
            return code
    return 0


def _run_command(step: CheckStep) -> int:
    completed = subprocess.run(step.command, check=False)
    return completed.returncode


STEPS: tuple[CheckStep, ...] = (
    CheckStep("format check", (sys.executable, "-m", "ruff", "format", "--check", ".")),
    CheckStep("lint", (sys.executable, "-m", "ruff", "check", ".")),
    CheckStep(
        "type check",
        (
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "src/nfe_document_extraction/domain",
            "src/nfe_document_extraction/validation",
        ),
    ),
    CheckStep(
        "tests",
        (
            sys.executable,
            "-m",
            "pytest",
            "--cov=nfe_document_extraction.validation",
            "--cov-fail-under=90",
            "--cov-report=term-missing",
        ),
    ),
)


def main() -> int:
    os.chdir(Path(__file__).resolve().parent.parent)
    return run_checks(STEPS)


if __name__ == "__main__":
    sys.exit(main())
