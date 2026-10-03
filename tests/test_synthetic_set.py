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
from nfe_document_extraction.synthetic import SyntheticDocument
from nfe_document_extraction.validation import ValidationReport, validate

_NFE_NS = "{http://www.portalfiscal.inf.br/nfe}"
_REPO_SET = Path(__file__).resolve().parents[1] / "data" / "synthetic"
_AS_OF = date(2026, 10, 3)


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("synthetic")
    _generate(destination)
    return destination


def test_fixed_seed_regenerates_a_byte_identical_set(
    synthetic_dir: Path,
    tmp_path: Path,
) -> None:
    regenerated = tmp_path / "regenerated"
    _generate(regenerated)

    assert _tree(regenerated) == _tree(synthetic_dir)


def test_repository_set_matches_the_fixed_seed(synthetic_dir: Path) -> None:
    assert _tree(synthetic_dir) == _tree(_REPO_SET)
    metadata = json.loads(
        (synthetic_dir / "synthetic-001" / "metadata.json").read_text(encoding="utf-8")
    )
    assert set(metadata) == {"nfe_id", "has_icms_st", "origin"}
    assert metadata["origin"] == "synthetic"
    assert metadata["nfe_id"] == "synthetic-001"
    assert isinstance(metadata["has_icms_st"], bool)


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
        documents.append(
            SyntheticDocument(
                nfe_id=metadata["nfe_id"],
                has_icms_st=metadata["has_icms_st"],
                origin=metadata["origin"],
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


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
