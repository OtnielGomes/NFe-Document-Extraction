"""Single-call baseline: one vision request reads every section."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from openai import OpenAI
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
from pydantic import ValidationError

from nfe_document_extraction.agents.config import BaselineConfig, load_baseline_config
from nfe_document_extraction.agents.schema import BaselinePayload
from nfe_document_extraction.domain import Extraction, Header, LineItem
from nfe_document_extraction.evaluation.metrics import ExtractionFailure

_PROMPT = (
    Path(__file__)
    .with_name("prompts")
    .joinpath("single_call_baseline.md")
    .read_text(encoding="utf-8")
)
_MAX_OUTPUT_TOKENS = 16_000
_RENDER_SCALE = 2


@dataclass(frozen=True)
class BaselineReading:
    """What one baseline call returned, or the extraction error it produced."""

    extraction: Extraction | None
    failure: ExtractionFailure | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    latency_seconds: float


def extract_baseline(
    client: OpenAI,
    pdf: bytes,
    *,
    config: BaselineConfig | None = None,
) -> BaselineReading:
    """Read a digital DANFE with one structured vision call.

    A parse failure or a refusal is an extraction error. It does not raise.
    """
    resolved = load_baseline_config() if config is None else config
    images = render_pdf_pages(pdf)
    if not images:
        return _failed(ExtractionFailure.PARSE_FAILURE, latency_seconds=0.0)
    started = time.perf_counter()
    response = client.responses.create(
        model=resolved.model,
        instructions=_PROMPT,
        input=[_message(images)],
        text=_strict_text_format(),
        temperature=0,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        store=False,
    )
    latency = time.perf_counter() - started
    input_tokens, cached_input_tokens, output_tokens = _usage(response)
    extraction, failure = interpret_baseline_output(
        _parse_output(getattr(response, "output_text", None)),
        _refusal(response),
    )
    return BaselineReading(
        extraction=extraction,
        failure=failure,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
    )


def interpret_baseline_output(
    parsed: BaselinePayload | None,
    refusal: str | None,
) -> tuple[Extraction | None, ExtractionFailure | None]:
    """Turn a structured response into an Extraction, or record why it failed."""
    if refusal:
        return None, ExtractionFailure.REFUSAL
    if not isinstance(parsed, BaselinePayload):
        return None, ExtractionFailure.PARSE_FAILURE
    try:
        return _to_extraction(parsed), None
    except ValidationError:
        return None, ExtractionFailure.PARSE_FAILURE


def render_pdf_pages(pdf: bytes) -> tuple[bytes, ...]:
    """Rasterize each PDF page to a PNG so the model sees the page, not the text layer."""
    document = pdfium.PdfDocument(pdf)
    pages: list[bytes] = []
    try:
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=_RENDER_SCALE)
            image = bitmap.to_pil()
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            pages.append(buffer.getvalue())
            bitmap.close()
            page.close()
    finally:
        document.close()
    return tuple(pages)


def _to_extraction(payload: BaselinePayload) -> Extraction:
    return Extraction(
        header=Header.model_validate(payload.header.model_dump()),
        line_items=tuple(
            LineItem.model_validate(item.model_dump()) for item in payload.line_items
        ),
    )


def _strict_text_format() -> ResponseTextConfigParam:
    raw = type_to_response_format_param(BaselinePayload)
    if not isinstance(raw, dict) or raw.get("type") != "json_schema":
        raise RuntimeError("baseline schema must be json_schema")
    json_schema = cast(ResponseFormatJSONSchema, raw)["json_schema"]
    schema = json_schema.get("schema")
    if schema is None:
        raise RuntimeError("baseline schema is empty")
    format_config: ResponseFormatTextJSONSchemaConfigParam = {
        "type": "json_schema",
        "name": json_schema["name"],
        "schema": schema,
        "strict": True,
    }
    return {"format": format_config}


def _parse_output(text: object) -> BaselinePayload | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return BaselinePayload.model_validate_json(text)
    except ValidationError:
        return None


def _message(images: tuple[bytes, ...]) -> EasyInputMessageParam:
    text: ResponseInputTextParam = {
        "type": "input_text",
        "text": "Extract every printed field of this DANFE.",
    }
    content: list[ResponseInputContentParam] = [
        text,
        *(_image_input(image) for image in images),
    ]
    return {"role": "user", "content": content}


def _image_input(png: bytes) -> ResponseInputImageParam:
    encoded = base64.b64encode(png).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{encoded}",
        "detail": "high",
    }


def _failed(failure: ExtractionFailure, *, latency_seconds: float) -> BaselineReading:
    return BaselineReading(
        extraction=None,
        failure=failure,
        input_tokens=0,
        cached_input_tokens=0,
        output_tokens=0,
        latency_seconds=latency_seconds,
    )


def _usage(response: object) -> tuple[int, int, int]:
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


def _refusal(response: object) -> str | None:
    for item in getattr(response, "output", ()) or ():
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", ()) or ():
            if getattr(content, "type", None) == "refusal":
                return str(getattr(content, "refusal", "") or "")
    return None
