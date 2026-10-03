import sys
from pathlib import Path

from scripts import check
from scripts.check import STEPS, CheckStep


def test_quality_script_stops_at_the_first_failing_step(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        check,
        "STEPS",
        (
            CheckStep("format check", (sys.executable, "-c", "raise SystemExit(0)")),
            CheckStep("lint", (sys.executable, "-c", "raise SystemExit(5)")),
            CheckStep("type check", (sys.executable, "-c", "print('should-not-run')")),
        ),
    )

    code = check.main()

    captured = capsys.readouterr()
    assert code == 5
    assert "Quality check failed: lint" in captured.err
    assert "should-not-run" not in captured.out
    assert "should-not-run" not in captured.err


def test_quality_steps_run_format_lint_types_then_tests() -> None:
    assert [step.name for step in STEPS] == [
        "format check",
        "lint",
        "type check",
        "tests",
    ]
    assert all(step.command[0] == sys.executable for step in STEPS)

    format_check, lint, type_check, tests = STEPS
    assert format_check.command[1:] == ("-m", "ruff", "format", "--check", ".")
    assert lint.command[1:] == ("-m", "ruff", "check", ".")
    assert type_check.command[1:] == (
        "-m",
        "mypy",
        "--strict",
        "src/nfe_document_extraction/domain",
        "src/nfe_document_extraction/validation",
    )
    assert tests.command[1:] == (
        "-m",
        "pytest",
        "--cov=nfe_document_extraction.validation",
        "--cov-fail-under=90",
        "--cov-report=term-missing",
    )


def test_ci_workflow_runs_only_the_quality_script() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    workflow = (repository_root / ".github" / "workflows" / "checks.yml").read_text(
        encoding="utf-8"
    )
    assert "push:" in workflow
    assert "pull_request:" in workflow
    run_lines = []
    for line in workflow.splitlines():
        stripped = line.strip()
        if stripped.startswith("- run:"):
            run_lines.append(stripped.removeprefix("- run:").strip())
        elif stripped.startswith("run:"):
            run_lines.append(stripped.removeprefix("run:").strip())
    assert run_lines == ["uv run --locked python scripts/check.py"]
