"""PDF text layer. No page is rasterized."""

from __future__ import annotations

from openai.types.responses import EasyInputMessageParam
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputContentParam,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam

from nfe_document_extraction.pdfium import open_pdf


def extract_pdf_text(pdf: bytes) -> str:
    """Read each page's text layer, in order."""
    pages: list[str] = []
    with open_pdf(pdf) as document:
        for index in range(len(document)):
            page = document[index]
            text_page = page.get_textpage()
            try:
                pages.append(str(text_page.get_text_bounded()))
            finally:
                text_page.close()
                page.close()
    return "\n\n".join(pages)


def pdf_text_message(instruction: str, document_text: str) -> EasyInputMessageParam:
    """One user message: a short instruction plus the PDF text layer."""
    prompt: ResponseInputTextParam = {"type": "input_text", "text": instruction}
    document: ResponseInputTextParam = {"type": "input_text", "text": document_text}
    content: list[ResponseInputContentParam] = [prompt, document]
    return {"role": "user", "content": content}
