"""Real-set runner: tracing forced off, aggregated results only."""

from datetime import date
from pathlib import Path

import langsmith
import pytest
from test_identifier_rules import _extraction

from nfe_document_extraction.agents.baseline import BaselineReading
from nfe_document_extraction.agents.config import ApproachConfig
from nfe_document_extraction.evaluation.tracing import tracing_is_disabled
from nfe_document_extraction.graph import GraphRun
from nfe_document_extraction.validation import DocumentOutcome, validate
from scripts import run_eval, run_real_eval

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "data" / "synthetic" / "synthetic-001"
_AS_OF = date(2026, 10, 3)
_ACCESS_KEY = "35240983606369122950550010000000011645576624"


class _RecordingReader:
    def __init__(self, client: object, config: ApproachConfig) -> None:
        self.config = config


def test_real_run_forces_tracing_off_and_writes_only_aggregates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _real_invoice(tmp_path)
    output = tmp_path / "results.md"
    seen: list[tuple[str, bytes, tuple[bytes, ...]]] = []
    _patch_models(monkeypatch, seen)
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    langsmith.configure(enabled=True)
    monkeypatch.setattr(langsmith, "Client", _client_must_not_be_created)

    code = run_real_eval.main(["--dataset", str(dataset), "--output", str(output)])

    assert code == 0
    assert tracing_is_disabled()
    text = output.read_text(encoding="utf-8")
    assert "| single-call baseline | digital PDF | real | 1 |" in text
    assert "| single-call baseline | phone photo | real | 1 |" in text
    assert "| cheap vision | digital PDF | real | 1 |" in text
    assert "| cheap vision | phone photo | real | 1 |" in text
    assert "| strong vision | digital PDF | real | 1 |" in text
    assert "| strong vision | phone photo | real | 1 |" in text
    assert "| text layer + text model | digital PDF | real | 1 |" in text
    assert _not_applicable("phone photo") in text
    assert "| simulated photo |" not in text
    assert _ACCESS_KEY not in text
    assert "Parafuso" not in text
    assert "197.7105" not in text
    assert "ls-secret" not in text
    assert list(dataset.rglob("*.json")) == []
    captured = capsys.readouterr()
    assert _ACCESS_KEY not in captured.out
    assert _ACCESS_KEY not in captured.err
    photo_pages = (b"page-1", b"page-2")
    assert ("single-call baseline", b"%PDF-real", ()) in seen
    assert ("single-call baseline", b"", photo_pages) in seen
    assert ("cheap vision", b"", photo_pages) in seen
    assert ("strong vision", b"", photo_pages) in seen
    assert ("text layer + text model", b"%PDF-real", ()) in seen
    assert ("text layer + text model", b"", photo_pages) not in seen


def test_real_rows_replace_previous_real_rows_and_keep_synthetic_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _real_invoice(tmp_path)
    output = tmp_path / "results.md"
    output.write_text(_existing_results(), encoding="utf-8")
    _patch_models(monkeypatch, [])

    code = run_real_eval.main(["--dataset", str(dataset), "--output", str(output)])

    text = output.read_text(encoding="utf-8")
    assert code == 0
    assert "| single-call baseline | digital PDF | synthetic | 7 |" in text
    assert "real | 99 |" not in text
    assert "| single-call baseline | phone photo | real | 1 |" in text
    assert _ACCESS_KEY not in text


def test_runner_refuses_to_start_when_tracing_cannot_be_confirmed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "real"
    invoice = dataset / "secret"
    invoice.mkdir(parents=True)
    (invoice / "nfe.xml").write_text("<not-an-nfe>", encoding="utf-8")
    output = tmp_path / "results.md"
    monkeypatch.setattr(run_real_eval, "load_env_file", lambda path: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(run_real_eval, "tracing_is_disabled", lambda: False)
    monkeypatch.setattr(run_real_eval, "OpenAI", _openai_must_not_be_called)

    code = run_real_eval.main(["--dataset", str(dataset), "--output", str(output)])

    assert code == 2
    assert not output.exists()
    assert "Tracing" in capsys.readouterr().err


def test_empty_real_set_does_not_rewrite_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "real"
    dataset.mkdir()
    output = tmp_path / "results.md"
    output.write_text("synthetic sentinel\n", encoding="utf-8")
    _patch_models(monkeypatch, [])
    monkeypatch.setattr(run_real_eval, "OpenAI", _openai_must_not_be_called)

    code = run_real_eval.main(["--dataset", str(dataset), "--output", str(output)])

    assert code == 1
    assert output.read_text(encoding="utf-8") == "synthetic sentinel\n"


def _patch_models(
    monkeypatch: pytest.MonkeyPatch,
    seen: list[tuple[str, bytes, tuple[bytes, ...]]],
) -> None:
    def baseline(
        client: object,
        pdf: bytes,
        *,
        config: ApproachConfig,
        pages: tuple[bytes, ...] = (),
    ) -> BaselineReading:
        seen.append((config.approach, pdf, pages))
        return _baseline(client, pdf, config=config)

    def run_document(
        reader: _RecordingReader,
        pdf: bytes,
        *,
        today: object = None,
        graph: object = None,
        pages: tuple[bytes, ...] = (),
    ) -> GraphRun:
        seen.append((reader.config.approach, pdf, pages))
        return _graph_run(reader, pdf, today=today, graph=graph)

    monkeypatch.setattr(run_real_eval, "load_env_file", lambda path: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(run_real_eval, "OpenAI", lambda: object())
    monkeypatch.setattr(run_eval, "VisionSectionReader", _RecordingReader)
    monkeypatch.setattr(run_eval, "TextLayerSectionReader", _RecordingReader)
    monkeypatch.setattr(run_eval, "compile_document_graph", lambda reader: reader)
    monkeypatch.setattr(run_eval, "extract_baseline", baseline)
    monkeypatch.setattr(run_eval, "run_document", run_document)


def _real_invoice(tmp_path: Path) -> Path:
    dataset = tmp_path / "real"
    invoice = dataset / _ACCESS_KEY
    photos = invoice / "phone-photo"
    photos.mkdir(parents=True)
    (invoice / "nfe.xml").write_bytes((_SOURCE / "nfe.xml").read_bytes())
    (invoice / "danfe.pdf").write_bytes(b"%PDF-real")
    (photos / "page-002.jpg").write_bytes(b"page-2")
    (photos / "page-001.jpg").write_bytes(b"page-1")
    return dataset


def _existing_results() -> str:
    return """\
# Evaluation results

Generated by `scripts/run_eval.py`. Do not edit by hand.

| Extraction approach | Rendition | Origin | Documents |
| --- | --- | --- | --- |
| single-call baseline | digital PDF | synthetic | 7 |
| single-call baseline | phone photo | real | 99 |
"""


def _not_applicable(rendition: str) -> str:
    gaps = " | ".join(["n/a"] * 12)
    return f"| text layer + text model | {rendition} | real | {gaps} |"


def _client_must_not_be_created(*args: object, **kwargs: object) -> object:
    raise AssertionError("LangSmith client was created")


def _openai_must_not_be_called() -> object:
    raise AssertionError("OpenAI client was created")


def _baseline(
    client: object,
    pdf: bytes,
    *,
    config: ApproachConfig,
    pages: tuple[bytes, ...] = (),
) -> BaselineReading:
    del client, pdf, pages
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
    pages: tuple[bytes, ...] = (),
) -> GraphRun:
    del pdf, today, pages
    assert graph is reader
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
