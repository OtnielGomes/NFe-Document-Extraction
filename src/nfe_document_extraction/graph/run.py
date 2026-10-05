"""LangGraph fan-out to the section agents, then one repair."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from nfe_document_extraction.agents.sections import (
    RepairFieldNote,
    Section,
    SectionReader,
    SectionReading,
    SectionRequest,
)
from nfe_document_extraction.domain import Extraction, Header, LineItem
from nfe_document_extraction.evaluation.metrics import ExtractionFailure
from nfe_document_extraction.validation import (
    DocumentOutcome,
    FieldStatus,
    ValidationReport,
    validate,
)


@dataclass(frozen=True)
class TokenUsage:
    """Tokens recorded for one section call."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


class _CompiledGraph(Protocol):
    def invoke(self, state: Mapping[str, object]) -> Mapping[str, object]: ...

    @property
    def nodes(self) -> Mapping[str, object]: ...


class DocumentState(TypedDict, total=False):
    """Shared graph state. Section nodes write disjoint keys."""

    pdf: bytes
    pages: tuple[bytes, ...]
    header: Header
    line_items: tuple[LineItem, ...]
    header_failure: ExtractionFailure | None
    line_items_failure: ExtractionFailure | None
    header_usage: TokenUsage
    line_items_usage: TokenUsage
    header_repair_usage: TokenUsage
    line_items_repair_usage: TokenUsage
    extraction: Extraction | None
    extraction_failure: ExtractionFailure | None
    report: ValidationReport | None
    initial_report: ValidationReport | None
    repair_attempted: bool
    header_notes: tuple[RepairFieldNote, ...]
    line_item_notes: tuple[RepairFieldNote, ...]


@dataclass(frozen=True)
class GraphRun:
    """The document after at most one repair."""

    outcome: DocumentOutcome | None
    extraction: Extraction | None
    report: ValidationReport | None
    initial_report: ValidationReport | None
    extraction_failure: ExtractionFailure | None
    repair_attempted: bool
    flagged_fields: tuple[str, ...]
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    latency_seconds: float


def run_document(
    reader: SectionReader,
    pdf: bytes,
    *,
    today: date | None = None,
    graph: _CompiledGraph | None = None,
    pages: tuple[bytes, ...] = (),
) -> GraphRun:
    """Read a DANFE with the Header and Line items agents.

    Both agents see the whole document. ``pages`` are Simulated photo images;
    a digital PDF leaves ``pages`` empty and is rasterized by the vision reader.
    Routing is code: a failing document is sent once to the agents that own
    the flagged fields, and a document that still fails is ready for human
    review. ``graph`` is a compiled graph from :func:`compile_document_graph`
    for the same reader.
    """
    compiled = compile_document_graph(reader, today=today) if graph is None else graph
    started = time.perf_counter()
    state = compiled.invoke({"pdf": pdf, "pages": pages})
    elapsed = time.perf_counter() - started
    return _graph_run(state, elapsed)


def compile_document_graph(
    reader: SectionReader, *, today: date | None = None
) -> _CompiledGraph:
    """Fan out to both section agents, validate, and repair once."""

    def read_header(state: DocumentState) -> dict[str, object]:
        return _read(reader, state, Section.HEADER)

    def read_line_items(state: DocumentState) -> dict[str, object]:
        return _read(reader, state, Section.LINE_ITEMS)

    def validate_first(state: DocumentState) -> dict[str, object]:
        return _validate(state, today=today, record_initial=True)

    def repair_header(state: DocumentState) -> dict[str, object]:
        return _repair(reader, state, Section.HEADER)

    def repair_line_items(state: DocumentState) -> dict[str, object]:
        return _repair(reader, state, Section.LINE_ITEMS)

    def validate_after_repair(state: DocumentState) -> dict[str, object]:
        return _validate(state, today=today, record_initial=False)

    builder = StateGraph(DocumentState)
    builder.add_node("read_header", read_header)
    builder.add_node("read_line_items", read_line_items)
    builder.add_node("merge", _merge)
    builder.add_node("validate", validate_first)
    builder.add_node("route_repair", _route_repair)
    builder.add_node("repair_header", repair_header)
    builder.add_node("repair_line_items", repair_line_items)
    builder.add_node("merge_repair", _merge)
    builder.add_node("validate_after_repair", validate_after_repair)
    builder.add_edge(START, "read_header")
    builder.add_edge(START, "read_line_items")
    builder.add_edge(["read_header", "read_line_items"], "merge")
    builder.add_edge("merge", "validate")
    builder.add_conditional_edges("validate", _route_after_validation)
    builder.add_edge("route_repair", "repair_header")
    builder.add_edge("route_repair", "repair_line_items")
    builder.add_edge(["repair_header", "repair_line_items"], "merge_repair")
    builder.add_edge("merge_repair", "validate_after_repair")
    builder.add_edge("validate_after_repair", END)
    return cast(_CompiledGraph, builder.compile())


def document_graph_nodes(reader: SectionReader) -> frozenset[str]:
    """Node names of the compiled graph."""
    compiled = compile_document_graph(reader)
    return frozenset(compiled.nodes)


def _repair_plan(
    report: ValidationReport,
) -> dict[Section, tuple[RepairFieldNote, ...]]:
    """Split every failing rule's fields by the agent that owns them.

    A failure whose fields span both sections produces one note per section,
    with the same message and only that section's fields.
    """
    grouped: dict[Section, list[RepairFieldNote]] = {
        Section.HEADER: [],
        Section.LINE_ITEMS: [],
    }
    for rule in report.rules:
        for failure in rule.failures:
            owned: dict[Section, list[str]] = {
                Section.HEADER: [],
                Section.LINE_ITEMS: [],
            }
            for field in failure.fields:
                owner = _section_owning(field)
                if owner is not None:
                    owned[owner].append(field)
            for section, fields in owned.items():
                if fields:
                    grouped[section].append(
                        RepairFieldNote(message=failure.message, fields=tuple(fields))
                    )
    return {section: tuple(notes) for section, notes in grouped.items()}


def _read(
    reader: SectionReader,
    state: DocumentState,
    section: Section,
) -> dict[str, object]:
    reading = reader.read(
        SectionRequest(
            section=section,
            document=state["pdf"],
            pages=_pages(state),
        )
    )
    return _apply_reading(section, reading, repair=False)


def _repair(
    reader: SectionReader,
    state: DocumentState,
    section: Section,
) -> dict[str, object]:
    notes = _notes(state, section)
    if not notes:
        return {}
    reading = reader.read(
        SectionRequest(
            section=section,
            document=state["pdf"],
            notes=notes,
            pages=_pages(state),
        )
    )
    return _apply_reading(section, reading, repair=True)


def _apply_reading(
    section: Section,
    reading: SectionReading,
    *,
    repair: bool,
) -> dict[str, object]:
    usage_key = _usage_key(section, repair=repair)
    update: dict[str, object] = {usage_key: _usage(reading)}
    value = _section_value(reading)
    if reading.failure is not None or value is None:
        if not repair:
            update[_failure_key(section)] = (
                reading.failure or ExtractionFailure.PARSE_FAILURE
            )
        return update
    update[_section_key(section)] = value
    return update


def _merge(state: DocumentState) -> dict[str, object]:
    failure = _prefer_failure(
        state.get("header_failure"),
        state.get("line_items_failure"),
    )
    header = state.get("header")
    items = state.get("line_items")
    if (
        failure is not None
        or not isinstance(header, Header)
        or not isinstance(items, tuple)
    ):
        return {
            "extraction": None,
            "extraction_failure": failure or ExtractionFailure.PARSE_FAILURE,
        }
    return {
        "extraction": Extraction(header=header, line_items=items),
        "extraction_failure": None,
    }


def _validate(
    state: DocumentState,
    *,
    today: date | None,
    record_initial: bool,
) -> dict[str, object]:
    extraction = state.get("extraction")
    if (
        not isinstance(extraction, Extraction)
        or state.get("extraction_failure") is not None
    ):
        if record_initial:
            return {"report": None, "initial_report": None}
        return {}
    report = validate(extraction, today=today)
    if record_initial:
        return {"report": report, "initial_report": report}
    return {"report": report}


def _route_after_validation(state: DocumentState) -> str:
    report = state.get("report")
    if (
        isinstance(report, ValidationReport)
        and state.get("extraction_failure") is None
        and isinstance(state.get("extraction"), Extraction)
        and report.outcome is DocumentOutcome.NEEDS_REPAIR
    ):
        return "route_repair"
    return END


def _route_repair(state: DocumentState) -> dict[str, object]:
    report = state.get("report")
    if not isinstance(report, ValidationReport):
        return {"repair_attempted": True, "header_notes": (), "line_item_notes": ()}
    plan = _repair_plan(report)
    return {
        "repair_attempted": True,
        "header_notes": plan[Section.HEADER],
        "line_item_notes": plan[Section.LINE_ITEMS],
    }


def _graph_run(state: Mapping[str, object], latency_seconds: float) -> GraphRun:
    extraction = state.get("extraction")
    report = state.get("report")
    initial = state.get("initial_report")
    failure = state.get("extraction_failure")
    resolved_extraction = extraction if isinstance(extraction, Extraction) else None
    resolved_report = report if isinstance(report, ValidationReport) else None
    resolved_initial = initial if isinstance(initial, ValidationReport) else None
    resolved_failure = failure if isinstance(failure, ExtractionFailure) else None
    input_tokens, cached_input_tokens, output_tokens = _tokens(state)
    return GraphRun(
        outcome=_public_outcome(
            resolved_extraction,
            resolved_failure,
            resolved_report,
        ),
        extraction=resolved_extraction,
        report=resolved_report,
        initial_report=resolved_initial,
        extraction_failure=resolved_failure,
        repair_attempted=bool(state.get("repair_attempted", False)),
        flagged_fields=_flagged(resolved_report),
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency_seconds,
    )


def _public_outcome(
    extraction: Extraction | None,
    failure: ExtractionFailure | None,
    report: ValidationReport | None,
) -> DocumentOutcome | None:
    if extraction is None or failure is not None or report is None:
        return None
    if report.outcome is DocumentOutcome.ACCEPTED:
        return DocumentOutcome.ACCEPTED
    return DocumentOutcome.HUMAN_REVIEW


def _flagged(report: ValidationReport | None) -> tuple[str, ...]:
    if report is None:
        return ()
    return tuple(
        field.field for field in report.fields if field.status is FieldStatus.FLAGGED
    )


def _tokens(state: Mapping[str, object]) -> tuple[int, int, int]:
    incoming = 0
    cached = 0
    outgoing = 0
    for key in (
        "header_usage",
        "line_items_usage",
        "header_repair_usage",
        "line_items_repair_usage",
    ):
        usage = state.get(key)
        if isinstance(usage, TokenUsage):
            incoming += usage.input_tokens
            cached += usage.cached_input_tokens
            outgoing += usage.output_tokens
    return incoming, cached, outgoing


def _prefer_failure(
    header_failure: object,
    items_failure: object,
) -> ExtractionFailure | None:
    failures = [
        failure
        for failure in (header_failure, items_failure)
        if isinstance(failure, ExtractionFailure)
    ]
    if any(failure is ExtractionFailure.REFUSAL for failure in failures):
        return ExtractionFailure.REFUSAL
    if failures:
        return ExtractionFailure.PARSE_FAILURE
    return None


def _section_owning(field: str) -> Section | None:
    if field.startswith("header."):
        return Section.HEADER
    if field.startswith("line_items["):
        return Section.LINE_ITEMS
    return None


def _pages(state: DocumentState) -> tuple[bytes, ...]:
    pages = state.get("pages", ())
    if isinstance(pages, tuple):
        return pages
    return ()


def _notes(state: DocumentState, section: Section) -> tuple[RepairFieldNote, ...]:
    key = "header_notes" if section is Section.HEADER else "line_item_notes"
    notes = state.get(key, ())
    if isinstance(notes, tuple):
        return notes
    return ()


def _section_value(reading: SectionReading) -> Header | tuple[LineItem, ...] | None:
    if reading.section is Section.HEADER:
        return reading.header
    return reading.line_items


def _section_key(section: Section) -> str:
    if section is Section.HEADER:
        return "header"
    return "line_items"


def _failure_key(section: Section) -> str:
    if section is Section.HEADER:
        return "header_failure"
    return "line_items_failure"


def _usage_key(section: Section, *, repair: bool) -> str:
    if section is Section.HEADER:
        return "header_repair_usage" if repair else "header_usage"
    return "line_items_repair_usage" if repair else "line_items_usage"


def _usage(reading: SectionReading) -> TokenUsage:
    return TokenUsage(
        input_tokens=reading.input_tokens,
        cached_input_tokens=reading.cached_input_tokens,
        output_tokens=reading.output_tokens,
    )
