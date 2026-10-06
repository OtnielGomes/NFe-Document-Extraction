"""Decide whether an upload may be read, before any model call."""

from __future__ import annotations

from dataclasses import dataclass

from nfe_document_extraction.app.errors import RequestRejected
from nfe_document_extraction.app.limits import CostLimits
from nfe_document_extraction.pdfium import open_pdf

TOO_MANY_PAGES = "This document has {count} pages, which is over the limit of {limit}."
HEIC_NOT_ACCEPTED = (
    "HEIC photos are not accepted. Export the photo as JPG and upload that file."
)
UNSUPPORTED_TYPE = "Upload a PDF, JPG, or PNG."
UNREADABLE_PDF = "This PDF could not be read."

_JPEG = b"\xff\xd8\xff"
_PNG = b"\x89PNG\r\n\x1a\n"
_HEIC_BRANDS = frozenset(
    {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"heif", b"mif1", b"msf1"}
)


@dataclass(frozen=True)
class PreparedUpload:
    """Bytes the graph should see. A photo leaves ``pdf`` empty."""

    pdf: bytes
    pages: tuple[bytes, ...]


def prepare_upload(
    data: bytes,
    *,
    filename: str | None,
    content_type: str | None,
    limits: CostLimits,
) -> PreparedUpload:
    """Accept a PDF, JPG, or PNG within the page and size limits."""
    if len(data) > limits.max_bytes:
        raise RequestRejected(_too_large(limits.max_bytes), 413)
    if data.startswith(b"%PDF"):
        _reject_long_pdf(data, limits.max_pages)
        return PreparedUpload(pdf=data, pages=())
    if data.startswith((_JPEG, _PNG)):
        return PreparedUpload(pdf=b"", pages=(data,))
    if _is_heic(data, filename, content_type):
        raise RequestRejected(HEIC_NOT_ACCEPTED, 400)
    raise RequestRejected(UNSUPPORTED_TYPE, 400)


def _too_large(max_bytes: int) -> str:
    if max_bytes % (1024 * 1024) == 0:
        return f"This file is larger than {max_bytes // (1024 * 1024)} MB."
    return f"This file is larger than {max_bytes} bytes."


def _is_heic(data: bytes, filename: str | None, content_type: str | None) -> bool:
    if _declared_heic(filename, content_type):
        return True
    if len(data) < 12 or data[4:8] != b"ftyp":
        return False
    if data[8:12] in _HEIC_BRANDS:
        return True
    compatible = data[16:]
    return any(
        compatible[offset : offset + 4] in _HEIC_BRANDS
        for offset in range(0, len(compatible) - 3, 4)
    )


def _declared_heic(filename: str | None, content_type: str | None) -> bool:
    if filename is not None and filename.lower().endswith((".heic", ".heif")):
        return True
    if content_type is None:
        return False
    declared = content_type.lower()
    return "heic" in declared or "heif" in declared


def _reject_long_pdf(data: bytes, max_pages: int) -> None:
    try:
        with open_pdf(data) as document:
            count = len(document)
    except Exception as exc:
        raise RequestRejected(UNREADABLE_PDF, 400) from exc
    if count < 1:
        raise RequestRejected(UNREADABLE_PDF, 400)
    if count > max_pages:
        raise RequestRejected(
            TOO_MANY_PAGES.format(count=count, limit=max_pages),
            400,
        )
