"""One process-wide lock around pypdfium2.

PDFium is not thread-safe. A repair runs both Section agents at once, and each
one opens the PDF, so every open holds this lock until the document is closed.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pypdfium2 as pdfium  # type: ignore[import-untyped]

_PDFIUM = threading.Lock()


@contextmanager
def open_pdf(data: bytes) -> Iterator[Any]:
    """Open a PDF while every other pypdfium2 call in this process waits."""
    with _PDFIUM:
        document = pdfium.PdfDocument(data)
        try:
            yield document
        finally:
            document.close()
