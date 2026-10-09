"""Local eval script: four approaches, no model calls in the test."""

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from test_identifier_rules import _extraction

from nfe_document_extraction.agents.baseline import BaselineReading
from nfe_document_extraction.agents.config import ApproachConfig
from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.env import load_env_file
from nfe_document_extraction.evaluation.langsmith import score_extraction
from nfe_document_extraction.evaluation.tracing import tracing_is_disabled
from nfe_document_extraction.graph import GraphRun
from nfe_document_extraction.synthetic import SyntheticDocument, write_synthetic_set
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
    monkeypatch.setattr(run_eval, "TextLayerSectionReader", _RecordingReader)
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
    assert "text layer + text model" in text
    assert "not applicable" in text
    assert "gpt-4.1-mini-2025-04-14" in text
    assert "gpt-4.1-2025-04-14" in text
    assert "Errors fixed by repair" in text
    assert "Errors reaching Human review" in text
    assert "| single-call baseline | digital PDF | synthetic | 1 |" in text
    assert "| cheap vision | digital PDF | synthetic | 1 |" in text
    assert "| strong vision | digital PDF | synthetic | 1 |" in text
    assert "| text layer + text model | digital PDF | synthetic | 1 |" in text
    assert text.count("| 1 | 1 | 0 | n/a | 0 |") == 3
    assert "| 1 | 0 | 0 | n/a | 0 |" in text
    assert text.count("| 0.400000 |") == 2
    assert "| 0.100000 |" in text
    assert "| 2.000000 |" in text
    assert _not_applicable("simulated photo") in text
    assert _not_applicable("phone photo") in text
    captured = capsys.readouterr()
    assert "single-call baseline: synthetic-001 (1/1)" in captured.out
    assert "cheap vision: synthetic-001 (1/1)" in captured.out
    assert "strong vision: synthetic-001 (1/1)" in captured.out
    assert "text layer + text model: synthetic-001 (1/1)" in captured.out


def test_simulated_photo_is_scored_separately_and_sent_as_page_images(
    monkeypatch, tmp_path, capsys
) -> None:
    dataset = _written_invoice(tmp_path)
    photo_pages = tuple(
        path.read_bytes()
        for path in sorted(
            (dataset / "synthetic-001" / "simulated-photo").glob("page-*.jpg")
        )
    )
    seen: list[tuple[str, tuple[bytes, ...]]] = []

    def baseline(
        client: object,
        pdf: bytes,
        *,
        config: ApproachConfig,
        pages: tuple[bytes, ...] = (),
    ) -> BaselineReading:
        seen.append((config.approach, pages))
        return _baseline(client, pdf, config=config)

    def run_document(
        reader: _RecordingReader,
        pdf: bytes,
        *,
        today: object = None,
        graph: object = None,
        pages: tuple[bytes, ...] = (),
    ) -> GraphRun:
        seen.append((reader.config.approach, pages))
        return _graph_run(reader, pdf, today=today, graph=graph)

    monkeypatch.setattr(run_eval, "load_env_file", lambda path: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "OpenAI", lambda: object())
    monkeypatch.setattr(run_eval, "VisionSectionReader", _RecordingReader)
    monkeypatch.setattr(run_eval, "TextLayerSectionReader", _RecordingReader)
    monkeypatch.setattr(run_eval, "compile_document_graph", lambda reader: reader)
    monkeypatch.setattr(run_eval, "extract_baseline", baseline)
    monkeypatch.setattr(run_eval, "run_document", run_document)
    output = tmp_path / "results.md"

    code = run_eval.main(["--dataset", str(dataset), "--output", str(output)])

    assert code == 0
    assert photo_pages
    assert ("single-call baseline", ()) in seen
    assert ("single-call baseline", photo_pages) in seen
    assert ("cheap vision", photo_pages) in seen
    assert ("strong vision", photo_pages) in seen
    assert ("text layer + text model", ()) in seen
    assert ("text layer + text model", photo_pages) not in seen
    text = output.read_text(encoding="utf-8")
    assert "| single-call baseline | digital PDF | synthetic | 1 |" in text
    assert "| single-call baseline | simulated photo | synthetic | 1 |" in text
    assert "| cheap vision | digital PDF | synthetic | 1 |" in text
    assert "| cheap vision | simulated photo | synthetic | 1 |" in text
    assert "| strong vision | digital PDF | synthetic | 1 |" in text
    assert "| strong vision | simulated photo | synthetic | 1 |" in text
    assert "| text layer + text model | digital PDF | synthetic | 1 |" in text
    assert _not_applicable("simulated photo") in text
    assert "accuracy drop from PDF to photo" in text
    captured = capsys.readouterr()
    assert "text layer + text model: synthetic-001 (1/1)" in captured.out


def test_text_layer_skips_photo_renditions(monkeypatch, tmp_path, capsys) -> None:
    dataset = _dataset_with_photos(tmp_path)
    calls: list[tuple[str, bytes]] = []

    def run_document(
        reader: _RecordingReader,
        pdf: bytes,
        *,
        today: object = None,
        graph: object = None,
        pages: tuple[bytes, ...] = (),
    ) -> GraphRun:
        calls.append((reader.config.approach, pdf))
        return _graph_run(reader, pdf, today=today, graph=graph)

    monkeypatch.setattr(run_eval, "load_env_file", lambda path: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "OpenAI", lambda: object())
    monkeypatch.setattr(run_eval, "VisionSectionReader", _RecordingReader)
    monkeypatch.setattr(run_eval, "TextLayerSectionReader", _RecordingReader)
    monkeypatch.setattr(run_eval, "compile_document_graph", lambda reader: reader)
    monkeypatch.setattr(run_eval, "extract_baseline", _baseline)
    monkeypatch.setattr(run_eval, "run_document", run_document)
    output = tmp_path / "results.md"

    code = run_eval.main(
        ["--dataset", str(dataset), "--output", str(output), "--limit", "10"]
    )

    assert code == 0
    digital_pdf = (dataset / "digital" / "danfe.pdf").read_bytes()
    simulated_pdf = (dataset / "simulated" / "danfe.pdf").read_bytes()
    phone_pdf = (dataset / "phone" / "danfe.pdf").read_bytes()
    text_calls = [
        pdf for approach, pdf in calls if approach == "text layer + text model"
    ]
    assert text_calls == [digital_pdf]
    assert simulated_pdf not in text_calls
    assert phone_pdf not in text_calls
    text = output.read_text(encoding="utf-8")
    assert "| single-call baseline | digital PDF | synthetic | 1 |" in text
    assert "| single-call baseline | simulated photo | synthetic | 1 |" in text
    assert "| single-call baseline | phone photo | synthetic | 1 |" in text
    assert "| cheap vision | simulated photo | synthetic | 1 |" in text
    assert "| strong vision | phone photo | synthetic | 1 |" in text
    assert "| text layer + text model | digital PDF | synthetic | 1 |" in text
    assert "| text layer + text model | simulated photo | synthetic | 1 |" not in text
    assert "| text layer + text model | phone photo | synthetic | 1 |" not in text
    assert _not_applicable("simulated photo") in text
    assert _not_applicable("phone photo") in text
    captured = capsys.readouterr()
    assert "text layer + text model: synthetic-001 (1/1)" in captured.out
    assert "text layer + text model: synthetic-photo" not in captured.out
    assert "text layer + text model: synthetic-phone" not in captured.out


def test_unknown_model_stops_before_a_model_call(monkeypatch, capsys) -> None:
    monkeypatch.setattr(run_eval, "load_env_file", lambda path: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("NFE_BASELINE_MODEL", "gpt-not-priced")
    monkeypatch.setattr(run_eval, "OpenAI", _openai_must_not_be_called)

    code = run_eval.main(["--limit", "1"])

    assert code == 2
    assert "pricing table" in capsys.readouterr().err


def test_without_a_langsmith_key_the_eval_only_writes_results(
    monkeypatch, tmp_path
) -> None:
    _patch_models(monkeypatch)
    monkeypatch.setattr(run_eval, "Client", _langsmith_must_not_be_called)
    monkeypatch.setattr(run_eval, "wrap_openai", _langsmith_must_not_be_called)
    output = tmp_path / "results.md"

    code = run_eval.main(
        ["--dataset", str(_one_invoice(tmp_path)), "--output", str(output)]
    )

    assert code == 0
    assert output.is_file()
    assert "| single-call baseline | digital PDF | synthetic | 1 |" in output.read_text(
        encoding="utf-8"
    )
    assert tracing_is_disabled() is False


def test_a_langsmith_key_uploads_the_commit_and_dataset_version(
    monkeypatch, tmp_path
) -> None:
    _patch_models(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    wrapped: list[object] = []
    monkeypatch.setattr(
        run_eval, "wrap_openai", lambda client: wrapped.append(client) or client
    )
    monkeypatch.setattr(run_eval, "repository_commit", lambda root: "abc123")
    client = _UploadClient()
    monkeypatch.setattr(run_eval, "Client", lambda: client)
    output = tmp_path / "results.md"

    code = run_eval.main(
        ["--dataset", str(_one_invoice(tmp_path)), "--output", str(output)]
    )

    assert code == 0
    assert output.is_file()
    assert wrapped == []
    assert len(client.calls) == 4
    assert {call["metadata"]["commit"] for call in client.calls} == {"abc123"}
    assert {call["metadata"]["dataset_version"] for call in client.calls} == {
        "2026-10-05T12:00:00+00:00"
    }
    assert {call["metadata"]["approach"] for call in client.calls} == {
        "single-call baseline",
        "cheap vision",
        "strong vision",
        "text layer + text model",
    }
    assert all(call["evaluators"] == [score_extraction] for call in client.calls)


def test_trace_flag_wraps_the_model_client(monkeypatch, tmp_path) -> None:
    _patch_models(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    wrapped: list[object] = []
    monkeypatch.setattr(
        run_eval, "wrap_openai", lambda client: wrapped.append(client) or client
    )
    monkeypatch.setattr(run_eval, "repository_commit", lambda root: "abc123")
    monkeypatch.setattr(run_eval, "Client", lambda: _UploadClient())
    output = tmp_path / "results.md"

    code = run_eval.main(
        [
            "--dataset",
            str(_one_invoice(tmp_path)),
            "--output",
            str(output),
            "--trace",
        ]
    )

    assert code == 0
    assert len(wrapped) == 1
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_no_trace_writes_results_and_sends_nothing(monkeypatch, tmp_path) -> None:
    _patch_models(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(run_eval, "Client", _langsmith_must_not_be_called)
    monkeypatch.setattr(run_eval, "wrap_openai", _langsmith_must_not_be_called)
    output = tmp_path / "results.md"

    code = run_eval.main(
        [
            "--dataset",
            str(_one_invoice(tmp_path)),
            "--output",
            str(output),
            "--no-trace",
        ]
    )

    assert code == 0
    assert "| cheap vision | digital PDF | synthetic | 1 |" in output.read_text(
        encoding="utf-8"
    )
    assert tracing_is_disabled()


def test_missing_langsmith_dataset_keeps_the_results_file(
    monkeypatch, tmp_path, capsys
) -> None:
    _patch_models(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "wrap_openai", lambda client: client)
    monkeypatch.setattr(run_eval, "repository_commit", lambda root: "abc123")
    monkeypatch.setattr(run_eval, "Client", _MissingDataset)
    output = tmp_path / "results.md"

    code = run_eval.main(
        ["--dataset", str(_one_invoice(tmp_path)), "--output", str(output)]
    )

    assert code == 1
    assert output.is_file()
    assert "sync_dataset.py" in capsys.readouterr().err


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


def _patch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "load_env_file", lambda path: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "OpenAI", lambda: object())
    monkeypatch.setattr(run_eval, "VisionSectionReader", _RecordingReader)
    monkeypatch.setattr(run_eval, "TextLayerSectionReader", _RecordingReader)
    monkeypatch.setattr(run_eval, "compile_document_graph", lambda reader: reader)
    monkeypatch.setattr(run_eval, "extract_baseline", _baseline)
    monkeypatch.setattr(run_eval, "run_document", _graph_run)


def _one_invoice(tmp_path: Path) -> Path:
    folder = tmp_path / "synthetic" / "synthetic-001"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "nfe_id": "synthetic-001",
                "has_icms_st": False,
                "origin": "synthetic",
                "split": "clean",
            }
        ),
        encoding="utf-8",
    )
    (folder / "ground_truth.json").write_text(
        _extraction().model_dump_json(),
        encoding="utf-8",
    )
    (folder / "danfe.pdf").write_bytes(b"%PDF-1.4")
    (folder / "renditions.json").write_text(
        json.dumps([{"input_type": "digital PDF"}]),
        encoding="utf-8",
    )
    return folder.parent


def _langsmith_must_not_be_called(*args: object, **kwargs: object) -> object:
    raise AssertionError("LangSmith was contacted")


@dataclass(frozen=True)
class _Revision:
    id: uuid.UUID
    modified_at: datetime
    created_at: datetime | None = None


class _UploadClient:
    def __init__(self) -> None:
        self.dataset_id = uuid.uuid4()
        self.calls: list[dict[str, object]] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return True

    def read_dataset(self, *, dataset_name: str) -> _Revision:
        return _Revision(self.dataset_id, datetime(2026, 10, 5, 12, 0, tzinfo=UTC))

    def evaluate(self, target: object, /, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _MissingDataset:
    def has_dataset(self, *, dataset_name: str) -> bool:
        return False

    def read_dataset(self, *, dataset_name: str) -> object:
        raise AssertionError("dataset was read")

    def evaluate(self, target: object, /, **kwargs: object) -> None:
        raise AssertionError("experiment was uploaded")


def test_approach_runs_only_cheap_vision_and_keeps_the_other_rows(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(run_eval, "load_env_file", lambda path: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "OpenAI", lambda: object())
    monkeypatch.setattr(run_eval, "VisionSectionReader", _RecordingReader)
    monkeypatch.setattr(run_eval, "TextLayerSectionReader", _RecordingReader)
    monkeypatch.setattr(run_eval, "compile_document_graph", lambda reader: reader)
    monkeypatch.setattr(run_eval, "extract_baseline", _baseline_must_not_run)
    monkeypatch.setattr(run_eval, "run_document", _graph_run)
    output = tmp_path / "results.md"
    output.write_text(_results_before_rejected_column(), encoding="utf-8")

    code = run_eval.main(
        [
            "--dataset",
            str(_ROOT / "data" / "synthetic"),
            "--output",
            str(output),
            "--limit",
            "1",
            "--approach",
            "cheap-vision",
        ]
    )

    assert code == 0
    text = output.read_text(encoding="utf-8")
    assert "Clean documents rejected" in text
    assert (
        "| single-call baseline | digital PDF | synthetic | 10 | 0.9322 | 1.0000 | "
        "0.0000 | 0.003345 | 5.931 | 30.466 | 9 | 0 | 0 | 1.0000 | 0 | 0 |"
    ) in text
    assert (
        "| strong vision | digital PDF | synthetic | 10 | 0.6907 | 1.0000 | 0.0000 | "
        "0.035147 | 8.467 | 40.786 | 9 | 0 | 9 | 1.0000 | 0 | 0 |"
    ) in text
    assert (
        "| text layer + text model | simulated photo | synthetic | "
        + " | ".join(["n/a"] * 13)
        + " |"
    ) in text
    assert "| cheap vision | digital PDF | synthetic | 1 |" in text
    assert "0.1111" not in text
    assert text.count("| cheap vision | digital PDF | synthetic |") == 1
    assert "gpt-4.1-2025-04-14 (strong vision)" in text
    assert (
        "gpt-4.1-mini-2025-04-14 (single-call baseline, text layer + text model)"
    ) in text
    assert "gpt-6-luna (cheap vision, run " in text
    assert "reasoning effort none, Responses API)" in text
    captured = capsys.readouterr()
    assert "cheap vision: synthetic-001 (1/1)" in captured.out
    assert "strong vision:" not in captured.out
    assert "single-call baseline:" not in captured.out
    assert "text layer + text model:" not in captured.out


def _results_before_rejected_column() -> str:
    return """\
# Evaluation results

Generated by `scripts/run_eval.py`. Do not edit by hand.

Model snapshot: `gpt-4.1-mini-2025-04-14 (single-call baseline, cheap vision, text layer + text model); gpt-4.1-2025-04-14 (strong vision)`. Pricing table date: 2026-10-03.

| Extraction approach | Rendition | Origin | Documents | Field accuracy | Description similarity | Perfect-invoice rate | Cost per invoice (USD) | Latency median (s) | Latency p95 (s) | Errors caught by the validator | Errors fixed by repair | Errors reaching Human review | Planted-error detection rate | Extraction errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single-call baseline | digital PDF | synthetic | 10 | 0.9322 | 1.0000 | 0.0000 | 0.003345 | 5.931 | 30.466 | 9 | 0 | 0 | 1.0000 | 0 |
| strong vision | digital PDF | synthetic | 10 | 0.6907 | 1.0000 | 0.0000 | 0.035147 | 8.467 | 40.786 | 9 | 0 | 9 | 1.0000 | 0 |
| text layer + text model | simulated photo | synthetic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cheap vision | digital PDF | synthetic | 10 | 0.1111 | 1.0000 | 0.0000 | 0.007105 | 7.903 | 45.377 | 9 | 0 | 9 | 1.0000 | 0 |
"""


def _baseline_must_not_run(
    client: object,
    pdf: bytes,
    *,
    config: ApproachConfig,
    pages: tuple[bytes, ...] = (),
) -> BaselineReading:
    del client, pdf, config, pages
    raise AssertionError("baseline ran")


def _not_applicable(rendition: str) -> str:
    gaps = " | ".join(["n/a"] * 13)
    return f"| text layer + text model | {rendition} | synthetic | {gaps} |"


def _written_invoice(tmp_path: Path) -> Path:
    folder = _ROOT / "data" / "synthetic" / "synthetic-001"
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    document = SyntheticDocument(
        nfe_id=metadata["nfe_id"],
        has_icms_st=metadata["has_icms_st"],
        origin=metadata["origin"],
        split=metadata["split"],
        planted_error=None,
        xml=(folder / "nfe.xml").read_text(encoding="utf-8"),
        pdf=(folder / "danfe.pdf").read_bytes(),
        ground_truth=Extraction.model_validate_json(
            (folder / "ground_truth.json").read_text(encoding="utf-8")
        ),
    )
    destination = tmp_path / "dataset"
    write_synthetic_set(destination, (document,))
    return destination


def _dataset_with_photos(tmp_path: Path) -> Path:
    source = _ROOT / "data" / "synthetic" / "synthetic-001"
    dataset = tmp_path / "dataset"
    copies = (
        ("digital", "synthetic-001", None),
        ("simulated", "synthetic-photo", "simulated photo"),
        ("phone", "synthetic-phone", "phone photo"),
    )
    for folder_name, nfe_id, rendition in copies:
        folder = dataset / folder_name
        shutil.copytree(source, folder)
        metadata_path = folder / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["nfe_id"] = nfe_id
        if rendition is not None:
            metadata["rendition"] = rendition
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        (folder / "danfe.pdf").write_bytes(nfe_id.encode())
        photo_dir = folder / "simulated-photo"
        if photo_dir.exists():
            shutil.rmtree(photo_dir)
        renditions = folder / "renditions.json"
        if renditions.exists():
            renditions.unlink()
    return dataset


def _openai_must_not_be_called() -> object:
    raise AssertionError("OpenAI client was created")


def _baseline(
    client: object,
    pdf: bytes,
    *,
    config: ApproachConfig,
    pages: tuple[bytes, ...] = (),
) -> BaselineReading:
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
    assert graph is reader
    assert reader.config.approach in {
        "cheap vision",
        "strong vision",
        "text layer + text model",
    }
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
