"""Committed synthetic set: planted errors fail in the validator, clean ones pass."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from pypdf import PdfReader

from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.validation import FieldStatus, ValidationReport, validate

_AS_OF = date(2026, 10, 3)
_REPO_SET = Path(__file__).resolve().parents[1] / "data" / "synthetic"
_RULES = frozenset(
    {
        "cnpj_cpf_check_digit",
        "access_key_check_digit",
        "access_key_cross_consistency",
        "quantity_times_unit_price",
        "totals",
        "ncm_cfop_issue_date",
    }
)
_CLEAN_METADATA = frozenset({"nfe_id", "has_icms_st", "origin", "split"})
_PLANTED_METADATA = _CLEAN_METADATA | {"planted_error"}


@dataclass(frozen=True)
class _PlantedError:
    rule: str
    field: str


@dataclass(frozen=True)
class _Document:
    nfe_id: str
    split: str
    planted_error: _PlantedError | None
    ground_truth: Extraction
    pdf: bytes


def test_about_fifteen_percent_carry_one_planted_error_across_every_rule() -> None:
    documents = _load(_REPO_SET)
    planted = [document for document in documents if document.split == "planted_errors"]
    clean = [document for document in documents if document.split == "clean"]

    assert len(documents) == 100
    assert {document.split for document in documents} == {"clean", "planted_errors"}
    assert len(planted) == 15
    assert len(clean) == 85
    assert all(document.planted_error is None for document in clean)
    assert all(document.planted_error is not None for document in planted)
    assert {document.planted_error.rule for document in planted} == _RULES


def test_planted_ground_truth_fails_only_the_recorded_rule_and_flags_the_field() -> (
    None
):
    planted = [
        document
        for document in _load(_REPO_SET)
        if document.split == "planted_errors" and document.planted_error is not None
    ]

    assert planted
    for document in planted:
        planted_error = document.planted_error
        assert planted_error is not None
        report = validate(document.ground_truth, today=_AS_OF)
        failing = [rule.rule_id for rule in report.rules if rule.failures]
        assert failing == [planted_error.rule], _failure_detail(document.nfe_id, report)
        flagged = {
            field.field
            for field in report.fields
            if field.status is FieldStatus.FLAGGED
        }
        assert planted_error.field in flagged


def test_clean_ground_truth_is_accepted() -> None:
    clean = [document for document in _load(_REPO_SET) if document.split == "clean"]

    assert clean
    for document in clean:
        report = validate(document.ground_truth, today=_AS_OF)
        assert report.accepted, _failure_detail(document.nfe_id, report)


def test_planted_error_is_the_value_printed_on_the_danfe() -> None:
    planted = [
        document
        for document in _load(_REPO_SET)
        if document.split == "planted_errors" and document.planted_error is not None
    ]

    assert planted
    for document in planted:
        planted_error = document.planted_error
        assert planted_error is not None
        printed = _printed_value(document.ground_truth, planted_error.field)
        assert printed in _pdf_text(document.pdf)


def _load(root: Path) -> tuple[_Document, ...]:
    documents: list[_Document] = []
    folders = sorted(path for path in root.iterdir() if path.is_dir())
    assert len(folders) == 100
    for folder in folders:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["origin"] == "synthetic"
        assert metadata["nfe_id"] == folder.name
        assert isinstance(metadata["has_icms_st"], bool)
        planted_error = _planted_error(metadata)
        expected_keys = _PLANTED_METADATA if planted_error else _CLEAN_METADATA
        assert set(metadata) == expected_keys
        documents.append(
            _Document(
                nfe_id=metadata["nfe_id"],
                split=metadata["split"],
                planted_error=planted_error,
                ground_truth=Extraction.model_validate_json(
                    (folder / "ground_truth.json").read_text(encoding="utf-8")
                ),
                pdf=(folder / "danfe.pdf").read_bytes(),
            )
        )
    return tuple(documents)


def _planted_error(metadata: dict[str, object]) -> _PlantedError | None:
    if metadata["split"] == "clean":
        assert "planted_error" not in metadata
        return None
    assert metadata["split"] == "planted_errors"
    planted = metadata["planted_error"]
    assert isinstance(planted, dict)
    assert set(planted) == {"rule", "field"}
    rule = planted["rule"]
    field = planted["field"]
    assert isinstance(rule, str) and rule in _RULES
    assert isinstance(field, str) and field
    return _PlantedError(rule=rule, field=field)


def _printed_value(extraction: Extraction, field: str) -> str:
    value = _field_value(extraction, field)
    if field == "header.recipient_cnpj_cpf":
        return _format_cnpj_cpf(str(value))
    if field == "header.access_key":
        text = str(value)
        return " ".join(text[index : index + 4] for index in range(0, len(text), 4))
    if field == "header.series":
        return f"SÉRIE {value}"
    if field == "header.invoice_total" or field.endswith(".item_total"):
        return _brazilian_cents(int(value))
    if field.endswith(".unit_price"):
        return _brazilian(Decimal(value), 4)
    if field.endswith(".ncm"):
        return str(value)
    raise AssertionError(f"no DANFE rendering for {field}")


def _field_value(extraction: Extraction, field: str) -> object:
    if field.startswith("header."):
        return getattr(extraction.header, field.removeprefix("header."))
    index_text, _, name = field.removeprefix("line_items[").partition("].")
    return getattr(extraction.line_items[int(index_text)], name)


def _format_cnpj_cpf(digits: str) -> str:
    if len(digits) == 11:
        return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"
    if len(digits) == 14:
        return f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:]}"
    raise AssertionError(digits)


def _brazilian_cents(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    amount = abs(cents)
    return sign + _brazilian(Decimal(amount) / Decimal(100), 2)


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


def _pdf_text(pdf: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _failure_detail(nfe_id: str, report: ValidationReport) -> str:
    messages = [failure.message for rule in report.rules for failure in rule.failures]
    return f"{nfe_id}: {messages}"
