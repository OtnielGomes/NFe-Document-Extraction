"""Single-call baseline: one vision request reads every section."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from nfe_document_extraction.agents.config import ApproachConfig, load_baseline_config
from nfe_document_extraction.agents.schema import BaselinePayload
from nfe_document_extraction.agents.vision import (
    MAX_OUTPUT_TOKENS,
    document_message,
    render_pdf_pages,
    response_refusal,
    response_usage,
    strict_text_format,
)
from nfe_document_extraction.domain import Extraction, Header, LineItem
from nfe_document_extraction.evaluation.metrics import ExtractionFailure

_PROMPT = (
    Path(__file__)
    .with_name("prompts")
    .joinpath("single_call_baseline.md")
    .read_text(encoding="utf-8")
)


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
    config: ApproachConfig | None = None,
    pages: tuple[bytes, ...] = (),
) -> BaselineReading:
    """Read a DANFE with one structured vision call.

    ``pages`` are the Simulated photo JPEGs. A digital PDF is rasterized when
    ``pages`` is empty. A parse failure or a refusal is an extraction error.
    It does not raise.
    """
    resolved = load_baseline_config() if config is None else config
    images = pages or render_pdf_pages(pdf)
    if not images:
        return _failed(ExtractionFailure.PARSE_FAILURE, latency_seconds=0.0)
    started = time.perf_counter()
    response = client.responses.create(
        model=resolved.model,
        instructions=_PROMPT,
        input=[document_message("Extract every printed field of this DANFE.", images)],
        text=strict_text_format(BaselinePayload),
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
    )
    latency = time.perf_counter() - started
    input_tokens, cached_input_tokens, output_tokens = response_usage(response)
    extraction, failure = interpret_baseline_output(
        _parse_output(getattr(response, "output_text", None)),
        response_refusal(response),
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


def _to_extraction(payload: BaselinePayload) -> Extraction:
    return Extraction(
        header=Header.model_validate(payload.header.model_dump()),
        line_items=tuple(
            LineItem.model_validate(item.model_dump()) for item in payload.line_items
        ),
    )


def _parse_output(text: object) -> BaselinePayload | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return BaselinePayload.model_validate_json(text)
    except ValidationError:
        return None


def _failed(failure: ExtractionFailure, *, latency_seconds: float) -> BaselineReading:
    return BaselineReading(
        extraction=None,
        failure=failure,
        input_tokens=0,
        cached_input_tokens=0,
        output_tokens=0,
        latency_seconds=latency_seconds,
    )
