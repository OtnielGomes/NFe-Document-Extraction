"""Local eval script: three approaches, no model calls in the test."""

import os
from datetime import date
from pathlib import Path

from test_identifier_rules import _extraction

from nfe_document_extraction.agents.baseline import BaselineReading
from nfe_document_extraction.agents.config import ApproachConfig
from nfe_document_extraction.env import load_env_file
from nfe_document_extraction.graph import GraphRun
from nfe_document_extraction.validation import DocumentOutcome, validate
from scripts import run_eval

_ROOT = Path(__file__).resolve().parents[1]
_AS_OF = date(2026, 10, 3)


class _RecordingReader:
    def __init__(self, client: object, config: ApproachConfig) -> None:
        self.config = config


def test_missing_api_key_stops_before_a_model_call(monkeypatch, capsys) -> None:
    monkeypatch.setattr(run_eval, "load_env_file", lambda path: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = run_eval.main([])

    assert code == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_eval_compares_baseline_with_cheap_and_strong_vision(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(run_eval, "load_env_file", lambda path: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "OpenAI", lambda: object())
    monkeypatch.setattr(run_eval, "VisionSectionReader", _RecordingReader)
    monkeypatch.setattr(run_eval, "compile_document_graph", lambda reader: reader)
    monkeypatch.setattr(run_eval, "extract_baseline", _baseline)
    monkeypatch.setattr(run_eval, "run_document", _graph_run)
    output = tmp_path / "results.md"

    code = run_eval.main(
        [
            "--dataset",
            str(_ROOT / "data" / "synthetic"),
            "--output",
            str(output),
            "--limit",
            "1",
        ]
    )

    assert code == 0
    text = output.read_text(encoding="utf-8")
    assert "single-call baseline" in text
    assert "cheap vision" in text
    assert "strong vision" in text
    assert "gpt-4.1-mini-2025-04-14" in text
    assert "gpt-4.1-2025-04-14" in text
    assert "Errors fixed by repair" in text
    assert "Errors reaching Human review" in text
    assert "| single-call baseline | digital PDF | synthetic | 1 |" in text
    assert "| cheap vision | digital PDF | synthetic | 1 |" in text
    assert "| strong vision | digital PDF | synthetic | 1 |" in text
    assert text.count("| 1 | 1 | 0 | n/a | 0 |") == 2
    assert "| 1 | 0 | 0 | n/a | 0 |" in text
    assert text.count("| 0.400000 |") == 2
    assert "| 2.000000 |" in text
    captured = capsys.readouterr()
    assert "single-call baseline: synthetic-001 (1/1)" in captured.out
    assert "cheap vision: synthetic-001 (1/1)" in captured.out
    assert "strong vision: synthetic-001 (1/1)" in captured.out


def test_env_file_sets_missing_keys_without_overriding(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """\
# comment
export OPENAI_API_KEY=from-file
ALREADY=from-file
EMPTY=
QUOTED="abc"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALREADY", "from-env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)

    load_env_file(env_file)

    assert os.environ["OPENAI_API_KEY"] == "from-file"
    assert os.environ["ALREADY"] == "from-env"
    assert os.environ["QUOTED"] == "abc"
    assert "EMPTY" not in os.environ


def _baseline(client: object, pdf: bytes, *, config: ApproachConfig) -> BaselineReading:
    assert config.approach == "single-call baseline"
    return BaselineReading(
        extraction=_extraction(invoice_total=1),
        failure=None,
        input_tokens=1_000_000,
        cached_input_tokens=0,
        output_tokens=0,
        latency_seconds=1,
    )


def _graph_run(
    reader: _RecordingReader,
    pdf: bytes,
    *,
    today: object = None,
    graph: object = None,
) -> GraphRun:
    assert graph is reader
    assert reader.config.approach in {"cheap vision", "strong vision"}
    initial = validate(_extraction(invoice_total=1), today=_AS_OF)
    accepted = validate(_extraction(), today=_AS_OF)
    return GraphRun(
        outcome=DocumentOutcome.ACCEPTED,
        extraction=_extraction(),
        report=accepted,
        initial_report=initial,
        extraction_failure=None,
        repair_attempted=True,
        flagged_fields=(),
        input_tokens=1_000_000,
        cached_input_tokens=0,
        output_tokens=0,
        latency_seconds=2,
    )
