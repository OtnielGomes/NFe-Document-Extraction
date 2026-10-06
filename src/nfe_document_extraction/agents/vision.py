"""Shared vision-request helpers for the baseline and the section agents."""

from __future__ import annotations

import base64
import io
from typing import cast

from openai.lib._parsing import type_to_response_format_param
from openai.types.responses import EasyInputMessageParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputContentParam,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)
from PIL import Image
from pydantic import BaseModel

from nfe_document_extraction.pdfium import open_pdf

MAX_OUTPUT_TOKENS = 16_000
RENDER_SCALE = 2


def render_pdf_pages(pdf: bytes) -> tuple[bytes, ...]:
    """Rasterize each PDF page to a PNG so the model sees the page, not the text layer."""
    images: list[Image.Image] = []
    with open_pdf(pdf) as document:
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=RENDER_SCALE)
            try:
                images.append(bitmap.to_pil().convert("RGB"))
            finally:
                bitmap.close()
                page.close()
    pages: list[bytes] = []
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        pages.append(buffer.getvalue())
    return tuple(pages)


def strict_text_format(model: type[BaseModel]) -> ResponseTextConfigParam:
    """OpenAI structured output for one strict schema."""
    raw = type_to_response_format_param(model)
    if not isinstance(raw, dict) or raw.get("type") != "json_schema":
        raise RuntimeError(f"{model.__name__} schema must be json_schema")
    json_schema = cast(ResponseFormatJSONSchema, raw)["json_schema"]
    schema = json_schema.get("schema")
    if schema is None:
        raise RuntimeError(f"{model.__name__} schema is empty")
    format_config: ResponseFormatTextJSONSchemaConfigParam = {
        "type": "json_schema",
        "name": json_schema["name"],
        "schema": schema,
        "strict": True,
    }
    return {"format": format_config}


def document_message(text: str, images: tuple[bytes, ...]) -> EasyInputMessageParam:
    """One user message: a short instruction plus every page image."""
    prompt: ResponseInputTextParam = {"type": "input_text", "text": text}
    content: list[ResponseInputContentParam] = [
        prompt,
        *(_image_input(image) for image in images),
    ]
    return {"role": "user", "content": content}


def response_usage(response: object) -> tuple[int, int, int]:
    """Input, cached input, and output tokens recorded on one response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    details = getattr(usage, "input_tokens_details", None)
    cached = 0 if details is None else int(getattr(details, "cached_tokens", 0) or 0)
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        cached,
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def response_refusal(response: object) -> str | None:
    """The refusal text, when the model refused instead of returning a schema."""
    for item in getattr(response, "output", ()) or ():
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", ()) or ():
            if getattr(content, "type", None) == "refusal":
                return str(getattr(content, "refusal", "") or "")
    return None


def _image_input(image: bytes) -> ResponseInputImageParam:
    encoded = base64.b64encode(image).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:{_image_mime(image)};base64,{encoded}",
        "detail": "high",
    }


def _image_mime(image: bytes) -> str:
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "image/png"
