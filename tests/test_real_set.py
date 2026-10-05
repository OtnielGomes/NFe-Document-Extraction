"""The Real set stays outside the repository and is read from a local path."""

import subprocess
from pathlib import Path

import pytest

from nfe_document_extraction.evaluation.real_set import (
    ground_truth_from_xml,
    load_real_documents,
)

_ROOT = Path(__file__).resolve().parents[1]
_SYNTHETIC = _ROOT / "data" / "synthetic" / "synthetic-001"


def test_repository_ignores_the_real_set_path() -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--", "data/real/invoice/nfe.xml"],
        cwd=_ROOT,
        check=False,
    )

    assert completed.returncode == 0


def test_real_documents_are_read_from_a_local_path(tmp_path: Path) -> None:
    xml = (_SYNTHETIC / "nfe.xml").read_text(encoding="utf-8")
    invoice = tmp_path / "market"
    invoice.mkdir()
    (invoice / "nfe.xml").write_text(xml, encoding="utf-8")
    (invoice / "danfe.pdf").write_bytes(b"%PDF-real")
    photos = invoice / "phone-photo"
    photos.mkdir()
    (photos / "page-002.jpg").write_bytes(b"page-2")
    (photos / "page-001.jpg").write_bytes(b"page-1")

    documents = load_real_documents(tmp_path)

    assert [document.rendition for document in documents] == [
        "digital PDF",
        "phone photo",
    ]
    assert {document.origin for document in documents} == {"real"}
    assert documents[0].pdf == b"%PDF-real"
    assert documents[0].pages == ()
    assert documents[1].pdf == b""
    assert documents[1].pages == (b"page-1", b"page-2")
    assert documents[0].ground_truth == documents[1].ground_truth
    assert documents[0].ground_truth == ground_truth_from_xml(xml)
    assert documents[0].planted_rule is None
    assert list(tmp_path.rglob("ground_truth.json")) == []


def test_phone_photo_is_enough_when_there_is_no_digital_pdf(tmp_path: Path) -> None:
    invoice = tmp_path / "counter"
    invoice.mkdir()
    (invoice / "nfe.xml").write_bytes((_SYNTHETIC / "nfe.xml").read_bytes())
    photos = invoice / "phone-photo"
    photos.mkdir()
    (photos / "page-001.jpg").write_bytes(b"page-1")

    documents = load_real_documents(tmp_path)

    assert [document.rendition for document in documents] == ["phone photo"]
    assert documents[0].pages == (b"page-1",)


def test_an_invoice_without_a_rendition_is_refused(tmp_path: Path) -> None:
    invoice = tmp_path / "empty"
    invoice.mkdir()
    (invoice / "nfe.xml").write_bytes((_SYNTHETIC / "nfe.xml").read_bytes())

    with pytest.raises(FileNotFoundError, match="empty"):
        load_real_documents(tmp_path)
