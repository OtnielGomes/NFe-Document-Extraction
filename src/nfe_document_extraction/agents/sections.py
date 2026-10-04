"""Header and Line items agents. Each owns one section and sees the whole document."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from nfe_document_extraction.agents.config import ApproachConfig
from nfe_document_extraction.agents.schema import (
    HeaderPayload,
    LineItemsSectionPayload,
)
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
    """The model boundary. Tests pass a scripted reader; the eval passes vision."""

    def read(self, request: SectionRequest) -> SectionReading: ...


class VisionSectionReader:
    """One pinned vision snapshot reading a section through a strict schema."""

    def __init__(self, client: OpenAI, config: ApproachConfig) -> None:
        self._client = client
        self._config = config

    def read(self, request: SectionRequest) -> SectionReading:
        """Read one section. A parse failure or a refusal is an extraction error."""
        images = render_pdf_pages(request.document)
        if not images:
            return SectionReading(
                section=request.section,
                failure=ExtractionFailure.PARSE_FAILURE,
            )
        schema = _schema(request.section)
        response = self._client.responses.create(
            model=self._config.model,
            instructions=_instructions(request.section, request.notes),
            input=[document_message(_USER_TEXT[request.section.value], images)],
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


def _schema(section: Section) -> type[BaseModel]:
    if section is Section.HEADER:
        return HeaderPayload
    return LineItemsSectionPayload


def _instructions(section: Section, notes: tuple[RepairFieldNote, ...]) -> str:
    prompt = _HEADER_PROMPT if section is Section.HEADER else _LINE_ITEMS_PROMPT
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
