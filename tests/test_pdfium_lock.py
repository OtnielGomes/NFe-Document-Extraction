"""pypdfium2 is not thread-safe. Parallel section reads must still agree."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pypdfium2

from nfe_document_extraction.agents.text_layer import extract_pdf_text
from nfe_document_extraction.agents.vision import render_pdf_pages

_PDF = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic"
    / "synthetic-001"
    / "danfe.pdf"
)
_ACCESS_KEY_TEXT = "3524 0983 6063 6912 2950 5500 1000 0000 0116 4557 6624"


def test_parallel_renders_and_text_reads_match_one_read(monkeypatch) -> None:
    pdf = _PDF.read_bytes()
    expected_text = extract_pdf_text(pdf)
    expected_pages = render_pdf_pages(pdf)
    assert "CAM-01" in expected_text
    assert _ACCESS_KEY_TEXT in expected_text
    assert expected_pages
    assert expected_pages[0].startswith(b"\x89PNG")

    real_document = pypdfium2.PdfDocument
    active = 0
    peak = 0
    guard = threading.Lock()

    class _TrackingDocument:
        def __init__(self, data: bytes) -> None:
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.05)
                self._inner = real_document(data)
            except BaseException:
                with guard:
                    active -= 1
                raise

        def close(self) -> None:
            nonlocal active
            try:
                self._inner.close()
            finally:
                with guard:
                    active -= 1

        def __len__(self) -> int:
            return len(self._inner)

        def __getitem__(self, index: int) -> object:
            return self._inner[index]

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    monkeypatch.setattr(pypdfium2, "PdfDocument", _TrackingDocument)
    barrier = threading.Barrier(4)

    def _read(index: int) -> str | tuple[bytes, ...]:
        barrier.wait(timeout=5)
        if index < 2:
            return render_pdf_pages(pdf)
        return extract_pdf_text(pdf)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_read, index) for index in range(4)]
        results = [future.result(timeout=60) for future in futures]

    assert peak == 1
    assert results[0] == expected_pages
    assert results[1] == expected_pages
    assert results[2] == expected_text
    assert results[3] == expected_text


def test_pypdfium2_is_opened_only_through_the_process_lock() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "nfe_document_extraction"
    importers = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "pypdfium2" in path.read_text(encoding="utf-8")
    )
    assert importers == ["pdfium.py"]
