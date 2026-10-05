"""Header and Line items agents. Each owns one section and sees the whole document."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from openai import OpenAI
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel, ValidationError

from nfe_document_extraction.agents.config import ApproachConfig
from nfe_document_extraction.agents.schema import (
    HeaderPayload,
    LineItemsSectionPayload,
)
from nfe_document_extraction.agents.text_layer import extract_pdf_text, pdf_text_message
from nfe_document_extraction.agents.vision import (
    MAX_OUTPUT_TOKENS,
    document_message,
    render_pdf_pages,
    response_refusal,
    response_usage,
    strict_text_format,
)
from nfe_document_extraction.domain import Header, LineItem
from nfe_document_extraction.evaluation.metrics import ExtractionFailure

_PROMPTS = Path(__file__).with_name("prompts")
_HEADER_PROMPT = (_PROMPTS / "header.md").read_text(encoding="utf-8")
_LINE_ITEMS_PROMPT = (_PROMPTS / "line_items.md").read_text(encoding="utf-8")
_HEADER_TEXT_PROMPT = (_PROMPTS / "header_text.md").read_text(encoding="utf-8")
_LINE_ITEMS_TEXT_PROMPT = (_PROMPTS / "line_items_text.md").read_text(encoding="utf-8")
_USER_TEXT = {
    "header": "Extract the header of this DANFE.",
    "line_items": "Extract the line items of this DANFE.",
}


class Section(StrEnum):
    """The section of the DANFE an agent owns."""

    HEADER = "header"
    LINE_ITEMS = "line_items"


@dataclass(frozen=True)
class RepairFieldNote:
    """One failing rule handed to the agent that owns those fields."""

    message: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class SectionRequest:
    """What one section agent is asked to read."""

    section: Section
    document: bytes
    notes: tuple[RepairFieldNote, ...] = ()
    pages: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class SectionReading:
    """One section agent's reading, or the extraction error it produced."""

    section: Section
    header: Header | None = None
    line_items: tuple[LineItem, ...] | None = None
    failure: ExtractionFailure | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


class SectionReader(Protocol):
    """The model boundary. Tests pass a scripted reader; the eval passes one."""

    def read(self, request: SectionRequest) -> SectionReading: ...


class VisionSectionReader:
    """One pinned vision snapshot reading a section through a strict schema."""

    def __init__(self, client: OpenAI, config: ApproachConfig) -> None:
        self._client = client
        self._config = config

    def read(self, request: SectionRequest) -> SectionReading:
        """Read one section. A parse failure or a refusal is an extraction error.

        Simulated photos pass their page images. A digital PDF is rasterized.
        """
        images = request.pages or render_pdf_pages(request.document)
        if not images:
            return SectionReading(
                section=request.section,
                failure=ExtractionFailure.PARSE_FAILURE,
            )
        return _complete(
            self._client,
            self._config,
            request,
            document_message(_USER_TEXT[request.section.value], images),
        )


class TextLayerSectionReader:
    """One pinned text snapshot reading a section from the PDF text layer."""

    def __init__(self, client: OpenAI, config: ApproachConfig) -> None:
        self._client = client
        self._config = config

    def read(self, request: SectionRequest) -> SectionReading:
        """Read one section from the text layer. No page image is sent."""
        document_text = extract_pdf_text(request.document)
        if not document_text.strip():
            return SectionReading(
                section=request.section,
                failure=ExtractionFailure.PARSE_FAILURE,
            )
        return _complete(
            self._client,
            self._config,
            request,
            pdf_text_message(_USER_TEXT[request.section.value], document_text),
            text_layer=True,
        )


def interpret_header_output(
    parsed: HeaderPayload | None,
    refusal: str | None,
) -> tuple[Header | None, ExtractionFailure | None]:
    """Turn a header response into a Header, or record why it failed."""
    if refusal:
        return None, ExtractionFailure.REFUSAL
    if not isinstance(parsed, HeaderPayload):
        return None, ExtractionFailure.PARSE_FAILURE
    try:
        return Header.model_validate(parsed.model_dump()), None
    except ValidationError:
        return None, ExtractionFailure.PARSE_FAILURE


def interpret_line_items_output(
    parsed: LineItemsSectionPayload | None,
    refusal: str | None,
) -> tuple[tuple[LineItem, ...] | None, ExtractionFailure | None]:
    """Turn a line-items response into rows, or record why it failed."""
    if refusal:
        return None, ExtractionFailure.REFUSAL
    if not isinstance(parsed, LineItemsSectionPayload):
        return None, ExtractionFailure.PARSE_FAILURE
    try:
        return tuple(
            LineItem.model_validate(item.model_dump()) for item in parsed.line_items
        ), None
    except ValidationError:
        return None, ExtractionFailure.PARSE_FAILURE


def _complete(
    client: OpenAI,
    config: ApproachConfig,
    request: SectionRequest,
    message: EasyInputMessageParam,
    *,
    text_layer: bool = False,
) -> SectionReading:
    schema = _schema(request.section)
    response = client.responses.create(
        model=config.model,
        instructions=_instructions(
            request.section, request.notes, text_layer=text_layer
        ),
        input=[message],
        text=strict_text_format(schema),
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
    )
    input_tokens, cached_input_tokens, output_tokens = response_usage(response)
    parsed = _parse(schema, getattr(response, "output_text", None))
    refusal = response_refusal(response)
    header, line_items, failure = _interpret(request.section, parsed, refusal)
    return SectionReading(
        section=request.section,
        header=header,
        line_items=line_items,
        failure=failure,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )


def _schema(section: Section) -> type[BaseModel]:
    if section is Section.HEADER:
        return HeaderPayload
    return LineItemsSectionPayload


def _prompt(section: Section, *, text_layer: bool) -> str:
    if text_layer and section is Section.HEADER:
        return _HEADER_TEXT_PROMPT
    if text_layer:
        return _LINE_ITEMS_TEXT_PROMPT
    if section is Section.HEADER:
        return _HEADER_PROMPT
    return _LINE_ITEMS_PROMPT


def _instructions(
    section: Section,
    notes: tuple[RepairFieldNote, ...],
    *,
    text_layer: bool = False,
) -> str:
    prompt = _prompt(section, text_layer=text_layer)
    if not notes:
        return prompt
    lines = [
        prompt.rstrip(),
        "",
        "Read these fields again. A deterministic check failed.",
        "Return the value printed on the page.",
        "Do not calculate a replacement and do not change a number so the arithmetic closes.",
        "",
    ]
    lines.extend(f"- {', '.join(note.fields)}: {note.message}" for note in notes)
    return "\n".join(lines)


def _interpret(
    section: Section,
    parsed: BaseModel | None,
    refusal: str | None,
) -> tuple[Header | None, tuple[LineItem, ...] | None, ExtractionFailure | None]:
    if section is Section.HEADER:
        header, failure = interpret_header_output(
            parsed if isinstance(parsed, HeaderPayload) else None,
            refusal,
        )
        return header, None, failure
    line_items, failure = interpret_line_items_output(
        parsed if isinstance(parsed, LineItemsSectionPayload) else None,
        refusal,
    )
    return None, line_items, failure


def _parse(model: type[BaseModel], text: object) -> BaseModel | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return model.model_validate_json(text)
    except ValidationError:
        return None
