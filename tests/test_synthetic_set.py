"""Synthetic set seam: generate the fixed-seed NF-e, DANFE, and ground truth."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pypdf import PdfReader

from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.evaluation.dataset import (
    DIGITAL_PDF,
    SIMULATED_PHOTO,
    load_synthetic_pdfs,
)
from nfe_document_extraction.synthetic import (
    SYNTHETIC_SEED,
    PlantedError,
    SyntheticDocument,
    simulate_photo_pages,
    write_synthetic_set,
)
from nfe_document_extraction.validation import ValidationReport, validate

_NFE_NS = "{http://www.portalfiscal.inf.br/nfe}"
_REPO_SET = Path(__file__).resolve().parents[1] / "data" / "synthetic"
_AS_OF = date(2026, 10, 3)


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("synthetic")
    _generate(destination)
    return destination


def test_fixed_seed_regenerates_identical_simulated_photos() -> None:
    pdf = _multipage_pdf()

    first = simulate_photo_pages(pdf, seed=SYNTHETIC_SEED)
    second = simulate_photo_pages(pdf, seed=SYNTHETIC_SEED)

    assert first == second
    assert simulate_photo_pages(pdf, seed=SYNTHETIC_SEED + 1) != first
    assert len(first) == _page_count(pdf)
    assert _page_count(pdf) > 1
    assert all(
        page.startswith(b"\xff\xd8\xff") and len(page) > 10_000 for page in first
    )


def test_each_nfe_is_written_with_both_renditions_and_one_ground_truth(
    tmp_path: Path,
) -> None:
    document = next(item for item in _load(_REPO_SET) if _page_count(item.pdf) > 1)
    destination = tmp_path / "synthetic"
    write_synthetic_set(destination, (document,))
    folder = destination / document.nfe_id
    renditions = json.loads((folder / "renditions.json").read_text(encoding="utf-8"))
    pages = sorted((folder / "simulated-photo").glob("page-*.jpg"))

    assert [entry["input_type"] for entry in renditions] == [
        DIGITAL_PDF,
        SIMULATED_PHOTO,
    ]
    assert len(pages) == _page_count(document.pdf)
    assert (folder / "danfe.pdf").read_bytes() == document.pdf
    assert json.loads((folder / "ground_truth.json").read_text(encoding="utf-8")) == (
        document.ground_truth.model_dump(mode="json")
    )
    assert [path.name for path in folder.rglob("ground_truth.json")] == [
        "ground_truth.json"
    ]
    assert "input_type" not in json.loads(
        (folder / "metadata.json").read_text(encoding="utf-8")
    )

    again = tmp_path / "again"
    write_synthetic_set(again, (document,))
    assert _tree(again) == _tree(destination)


def test_dataset_examples_are_one_per_rendition_and_share_ground_truth(
    tmp_path: Path,
) -> None:
    document = next(item for item in _load(_REPO_SET) if _page_count(item.pdf) > 1)
    destination = tmp_path / "synthetic"
    write_synthetic_set(destination, (document,))

    pdf_example, photo_example = load_synthetic_pdfs(destination)

    assert pdf_example.input_type == DIGITAL_PDF
    assert photo_example.input_type == SIMULATED_PHOTO
    assert pdf_example.nfe_id == photo_example.nfe_id == document.nfe_id
    assert pdf_example.origin == photo_example.origin == document.origin
    assert pdf_example.ground_truth == photo_example.ground_truth
    assert pdf_example.planted_rule == photo_example.planted_rule
    assert pdf_example.pdf.startswith(b"%PDF")
    assert len(photo_example.pages) == _page_count(document.pdf)
    assert all(page.startswith(b"\xff\xd8\xff") for page in photo_example.pages)


def test_fixed_seed_regenerates_a_byte_identical_set(
    synthetic_dir: Path,
    tmp_path: Path,
) -> None:
    regenerated = tmp_path / "regenerated"
    _generate(regenerated)

    assert _tree(regenerated) == _tree(synthetic_dir)


def test_repository_set_matches_the_fixed_seed(synthetic_dir: Path) -> None:
    # pdfium's raster is not byte-stable across operating systems, so committed
    # Simulated photos are checked by path. Byte identity is the same-process test.
    assert _tree(synthetic_dir, include_photos=False) == _tree(
        _REPO_SET, include_photos=False
    )
    assert _photo_paths(synthetic_dir) == _photo_paths(_REPO_SET)
    assert _photo_paths(_REPO_SET)
    for relative in _photo_paths(_REPO_SET):
        payload = (_REPO_SET / relative).read_bytes()
        assert payload.startswith(b"\xff\xd8\xff")
        assert len(payload) > 10_000
    metadata = json.loads(
        (synthetic_dir / "synthetic-001" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["origin"] == "synthetic"
    assert metadata["nfe_id"] == "synthetic-001"
    assert isinstance(metadata["has_icms_st"], bool)
    assert metadata["split"] in {"clean", "planted_errors"}


def test_synthetic_set_covers_the_acceptance_criteria(synthetic_dir: Path) -> None:
    documents = _load(synthetic_dir)

    assert len(documents) == 100
    assert {document.nfe_id for document in documents} == {
        f"synthetic-{index:03d}" for index in range(1, 101)
    }
    assert {document.origin for document in documents} == {"synthetic"}

    item_counts = [len(document.ground_truth.line_items) for document in documents]
    assert min(item_counts) == 1
    assert max(item_counts) == 40
    assert all(1 <= count <= 40 for count in item_counts)

    cpf_recipients = [
        document
        for document in documents
        if len(document.ground_truth.header.recipient_cnpj_cpf) == 11
    ]
    assert len(cpf_recipients) == 20
    assert sum(1 for document in documents if document.has_icms_st) == 30
    assert all(
        (document.ground_truth.header.icms_st_total > 0) is document.has_icms_st
        for document in documents
    )
    assert sum(_page_count(document.pdf) > 1 for document in documents) == 20
    assert any(document.ground_truth.header.freight > 0 for document in documents)
    assert any(document.ground_truth.header.discount > 0 for document in documents)
    assert any(document.ground_truth.header.ipi_total > 0 for document in documents)

    for document in documents:
        if document.split == "clean":
            report = validate(document.ground_truth, today=_AS_OF)
            assert report.accepted, _failure_messages(document.nfe_id, report)
        _assert_printed_unit_price(document)


def _generate(destination: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nfe_document_extraction.synthetic",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _load(root: Path) -> tuple[SyntheticDocument, ...]:
    documents: list[SyntheticDocument] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        planted = metadata.get("planted_error")
        planted_error = None
        if planted is not None:
            planted_error = PlantedError(rule=planted["rule"], field=planted["field"])
        documents.append(
            SyntheticDocument(
                nfe_id=metadata["nfe_id"],
                has_icms_st=metadata["has_icms_st"],
                origin=metadata["origin"],
                split=metadata["split"],
                planted_error=planted_error,
                xml=(folder / "nfe.xml").read_text(encoding="utf-8"),
                pdf=(folder / "danfe.pdf").read_bytes(),
                ground_truth=Extraction.model_validate_json(
                    (folder / "ground_truth.json").read_text(encoding="utf-8")
                ),
            )
        )
    return tuple(documents)


def _assert_printed_unit_price(document: SyntheticDocument) -> None:
    root = ET.fromstring(document.xml)
    xml_prices = [
        Decimal(element.text or "0") for element in root.iter(f"{_NFE_NS}vUnCom")
    ]
    printed_prices = [item.unit_price for item in document.ground_truth.line_items]
    assert xml_prices
    assert all(
        xml_price != printed and abs(xml_price - printed) < Decimal("0.0001")
        for xml_price, printed in zip(xml_prices, printed_prices, strict=True)
    )
    pdf_text = _pdf_text(document.pdf)
    for printed in printed_prices:
        assert _brazilian(printed, 4) in pdf_text


def _failure_messages(nfe_id: str, report: ValidationReport) -> str:
    messages = [failure.message for rule in report.rules for failure in rule.failures]
    return f"{nfe_id}: {messages}"


def _multipage_pdf() -> bytes:
    for folder in sorted(path for path in _REPO_SET.iterdir() if path.is_dir()):
        pdf = (folder / "danfe.pdf").read_bytes()
        if _page_count(pdf) > 1:
            return pdf
    raise AssertionError("synthetic set has no multi-page DANFE")


def _page_count(pdf: bytes) -> int:
    return len(PdfReader(io.BytesIO(pdf)).pages)


def _pdf_text(pdf: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _brazilian(value: Decimal, precision: int) -> str:
    text = f"{value:.{precision}f}"
    whole, fraction = text.split(".")
    sign = "-" if whole.startswith("-") else ""
    digits = whole.removeprefix("-")
    groups: list[str] = []
    while digits:
        groups.append(digits[-3:])
        digits = digits[:-3]
    return sign + ".".join(reversed(groups)) + "," + fraction


def _photo_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        path.relative_to(root).as_posix() for path in sorted(root.rglob("*.jpg"))
    )


def _tree(root: Path, *, include_photos: bool = True) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and (include_photos or path.suffix != ".jpg")
    }
