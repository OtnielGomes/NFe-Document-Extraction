"""Local eval script: four approaches, no model calls in the test."""

import json
import os
import shutil
from datetime import date
from pathlib import Path

from test_identifier_rules import _extraction

from nfe_document_extraction.agents.baseline import BaselineReading
from nfe_document_extraction.agents.config import ApproachConfig
from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.env import load_env_file
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
    assert text.count("| 0.400000 |") == 3
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


def _not_applicable(rendition: str) -> str:
    gaps = " | ".join(["n/a"] * 12)
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
