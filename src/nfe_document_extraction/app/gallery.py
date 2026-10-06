"""Cached synthetic invoices for the demo gallery. No model call, no trace."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from PIL import Image

from nfe_document_extraction.app.service import document_payload
from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.evaluation.dataset import (
    DIGITAL_PDF,
    SIMULATED_PHOTO,
    SIMULATED_PHOTO_DIRECTORY,
)
from nfe_document_extraction.validation import validate
from nfe_document_extraction.validation.report import DocumentOutcome, ValidationReport

# A few published invoices, including planted errors, so the page stays small.
GALLERY_SELECTION: tuple[tuple[str, str], ...] = (
    ("synthetic-001", DIGITAL_PDF),
    ("synthetic-001", SIMULATED_PHOTO),
    ("synthetic-007", DIGITAL_PDF),
    ("synthetic-014", DIGITAL_PDF),
    ("synthetic-015", DIGITAL_PDF),
    ("synthetic-021", DIGITAL_PDF),
)

_PREVIEW_SCALE = 1


@dataclass(frozen=True)
class GalleryItem:
    """One preloaded rendition and the validator's reading of what it prints."""

    caption: str
    image: Image.Image
    payload: dict[str, object]


def load_gallery(root: Path, *, today: date) -> tuple[GalleryItem, ...]:
    """Validate the printed values once. Opening an item only returns this cache."""
    return tuple(
        _item(root, nfe_id, rendition, today) for nfe_id, rendition in GALLERY_SELECTION
    )


def _item(root: Path, nfe_id: str, rendition: str, today: date) -> GalleryItem:
    folder = root / nfe_id
    if not folder.is_dir():
        raise FileNotFoundError(f"{nfe_id} is not in {root}")
    extraction = Extraction.model_validate_json(
        (folder / "ground_truth.json").read_text(encoding="utf-8")
    )
    report = validate(extraction, today=today)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return GalleryItem(
        caption=_caption(nfe_id, rendition, metadata),
        image=_preview(folder, rendition),
        payload=document_payload(
            extraction,
            report,
            outcome=_shown_outcome(report),
            thread_id=None,
        ),
    )


def _caption(nfe_id: str, rendition: str, metadata: dict[str, object]) -> str:
    caption = f"{nfe_id} · {rendition}"
    if metadata.get("planted_error") is not None:
        return f"{caption} · planted error"
    return caption


def _shown_outcome(report: ValidationReport) -> DocumentOutcome:
    if report.accepted:
        return DocumentOutcome.ACCEPTED
    return DocumentOutcome.HUMAN_REVIEW


def _preview(folder: Path, rendition: str) -> Image.Image:
    if rendition == SIMULATED_PHOTO:
        pages = sorted((folder / SIMULATED_PHOTO_DIRECTORY).glob("page-*.jpg"))
        if not pages:
            raise FileNotFoundError(f"{folder.name} is missing simulated photo pages")
        return _jpeg(pages[0].read_bytes())
    if rendition != DIGITAL_PDF:
        raise ValueError(f"{folder.name} has no gallery preview for {rendition}")
    pdf = folder / "danfe.pdf"
    if not pdf.is_file():
        raise FileNotFoundError(f"{folder.name} is missing danfe.pdf")
    return _pdf_preview(pdf.read_bytes())


def _jpeg(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def _pdf_preview(pdf: bytes) -> Image.Image:
    document = pdfium.PdfDocument(pdf)
    try:
        page = document[0]
        bitmap = page.render(scale=_PREVIEW_SCALE)
        try:
            return bitmap.to_pil().convert("RGB")
        finally:
            bitmap.close()
            page.close()
    finally:
        document.close()
