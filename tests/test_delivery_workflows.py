"""Delivery workflows: Space deploy, manual LLM eval, and local compose."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_llm_eval_runs_only_on_manual_dispatch() -> None:
    text = _workflow("llm-eval.yml")
    triggers = _on_block(text)

    assert "workflow_dispatch" in triggers
    assert "push" not in triggers
    assert "pull_request" not in triggers
    assert "scripts/sync_dataset.py" in text
    assert "scripts/run_eval.py" in text
    assert text.index("scripts/sync_dataset.py") < text.index("scripts/run_eval.py")
    assert "--limit" not in text
    assert "--trace" not in text
    assert "--no-trace" not in text
    assert "run_real_eval.py" not in text
    assert "docs/results.md" in text
    assert "secrets.OPENAI_API_KEY" in text
    assert "secrets.LANGSMITH_API_KEY" in text
    assert "timeout-minutes: 360" in text


def test_space_deploys_only_after_checks_succeed_on_main() -> None:
    text = _workflow("deploy-space.yml")
    triggers = _on_block(text)

    assert "workflow_run" in triggers
    assert "checks" in triggers
    assert "push:" not in triggers
    assert "pull_request" not in triggers
    assert "github.event.workflow_run.conclusion == 'success'" in text
    assert "github.event.workflow_run.event == 'push'" in text
    assert "github.event.workflow_run.head_branch == 'main'" in text
    assert (
        "github.event.workflow_run.head_repository.full_name == github.repository"
        in text
    )
    assert "github.event.workflow_run.head_sha" in text
    assert "scripts/deploy_space.py" in text
    assert "cancel-in-progress: false" in text
    assert "git fetch origin main" in text
    assert "git rev-parse origin/main" in text
    assert "steps.tip.outputs.current == 'true'" in text
    assert "secrets.HF_TOKEN" in text
    assert "secrets.HF_SPACE_ID" in text
    assert "secrets.OPENAI_API_KEY" in text
    assert "secrets.LANGSMITH_API_KEY" in text


def test_push_checks_do_not_run_the_llm_eval() -> None:
    text = _workflow("checks.yml")

    assert "run_eval.py" not in text
    assert "sync_dataset.py" not in text
    assert "deploy_space.py" not in text


def test_openai_budget_is_a_documented_manual_step() -> None:
    text = (_ROOT / "docs" / "deploy.md").read_text(encoding="utf-8")

    assert "US$ 5" in text
    assert "manual step" in text
    assert "month" in text.lower()


def test_local_compose_serves_the_space_image_on_port_7860() -> None:
    text = (_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "dockerfile: Dockerfile" in text
    assert '"7860:7860"' in text
    assert "env_file: .env" in text
    assert "volumes:" not in text
    assert text.count("build:") == 1


def _workflow(name: str) -> str:
    return (_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def _on_block(text: str) -> str:
    lines = text.splitlines()
    start = lines.index("on:")
    block = ["on:"]
    for line in lines[start + 1 :]:
        if line and not line.startswith(" "):
            break
        block.append(line)
    return "\n".join(block)
