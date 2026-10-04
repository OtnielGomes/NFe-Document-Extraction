"""PDF text layer. No page is rasterized."""

from __future__ import annotations

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from openai.types.responses import EasyInputMessageParam
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputContentParam,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam


def extract_pdf_text(pdf: bytes) -> str:
    """Read each page's text layer, in order."""
    document = pdfium.PdfDocument(pdf)
    pages: list[str] = []
    try:
        for index in range(len(document)):
            page = document[index]
            text_page = page.get_textpage()
            try:
                pages.append(str(text_page.get_text_bounded()))
            finally:
                text_page.close()
                page.close()
    finally:
        document.close()
    return "\n\n".join(pages)


def pdf_text_message(instruction: str, document_text: str) -> EasyInputMessageParam:
    """One user message: a short instruction plus the PDF text layer."""
    prompt: ResponseInputTextParam = {"type": "input_text", "text": instruction}
    document: ResponseInputTextParam = {"type": "input_text", "text": document_text}
    content: list[ResponseInputContentParam] = [prompt, document]
    return {"role": "user", "content": content}
