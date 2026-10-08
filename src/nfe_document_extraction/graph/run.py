"""LangGraph fan-out to the section agents, one repair, then human review."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

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
    cache_write_tokens: int = 0


class _CompiledGraph(Protocol):
    def invoke(
        self,
        state: Mapping[str, object] | Command,
        config: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]: ...

    @property
    def nodes(self) -> Mapping[str, object]: ...


_CHECKPOINT_TYPES: tuple[tuple[str, str], ...] = (
    ("nfe_document_extraction.domain.extraction", "Extraction"),
    ("nfe_document_extraction.domain.extraction", "Header"),
    ("nfe_document_extraction.domain.extraction", "LineItem"),
    ("nfe_document_extraction.domain.extraction", "UnsupportedComponents"),
    ("nfe_document_extraction.validation.report", "DocumentOutcome"),
    ("nfe_document_extraction.validation.report", "FieldReport"),
    ("nfe_document_extraction.validation.report", "FieldStatus"),
    ("nfe_document_extraction.validation.report", "RuleFailure"),
    ("nfe_document_extraction.validation.report", "RuleReport"),
    ("nfe_document_extraction.validation.report", "ValidationReport"),
    ("nfe_document_extraction.graph.run", "TokenUsage"),
    ("nfe_document_extraction.agents.sections", "RepairFieldNote"),
    ("nfe_document_extraction.evaluation.metrics", "ExtractionFailure"),
)
_HEADER_FIELD = re.compile(r"^header\.([a-z0-9_]+)$")
_ITEM_FIELD = re.compile(r"^line_items\[(\d+)\]\.([a-z0-9_]+)$")


class NonFlaggedFieldEdit(Exception):
    """A human review submission named a field that is not flagged."""

    def __init__(self, fields: tuple[str, ...]) -> None:
        self.fields = fields
        listed = ", ".join(fields)
        super().__init__(f"Only flagged fields can be edited: {listed}.")


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
    pending_edits: dict[str, object]
    rejected_fields: tuple[str, ...]
    human_fields: tuple[str, ...]
    document_rejected: bool


@dataclass(frozen=True)
class GraphRun:
    """The document after at most one repair, including human review."""

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
    thread_id: str | None = None
    awaiting_human_review: bool = False
    human_fields: tuple[str, ...] = ()
    rejected: bool = False
    cache_write_tokens: int = 0


def run_document(
    reader: SectionReader,
    pdf: bytes,
    *,
    today: date | None = None,
    graph: _CompiledGraph | None = None,
    pages: tuple[bytes, ...] = (),
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    thread_id: str | None = None,
) -> GraphRun:
    """Read a DANFE with the Header and Line items agents.

    Both agents see the whole document. ``pages`` are Simulated photo images;
    a digital PDF leaves ``pages`` empty and is rasterized by the vision reader.
    Routing is code: a failing document is sent once to the agents that own
    the flagged fields. If it still fails, the graph pauses for human review.
    The paused checkpoint is identified by ``thread_id``. Pass the same
    checkpointer to :func:`submit_human_review` to resume it. When ``graph``
    is supplied it must already be compiled with a checkpointer; otherwise
    ``checkpointer`` is attached here (an in-memory saver if omitted).
    """
    thread = thread_id if thread_id is not None else str(uuid.uuid4())
    compiled = (
        compile_document_graph(reader, today=today, checkpointer=checkpointer)
        if graph is None
        else graph
    )
    started = time.perf_counter()
    state = compiled.invoke(
        {"pdf": pdf, "pages": pages},
        _thread_config(thread),
    )
    elapsed = time.perf_counter() - started
    return _graph_run(state, elapsed, thread_id=thread)


def submit_human_review(
    reader: SectionReader,
    thread_id: str,
    edits: Mapping[str, object],
    *,
    checkpointer: BaseCheckpointSaver[Any],
    today: date | None = None,
) -> GraphRun:
    """Resume a paused human review and validate the edited fields.

    ``thread_id`` selects the checkpoint. Edits are applied in code and
    checked with the same rules as model output. This does not call the model
    and does not start another repair.
    """
    compiled = compile_document_graph(reader, today=today, checkpointer=checkpointer)
    started = time.perf_counter()
    state = compiled.invoke(
        # LangGraph reads an empty dict as an interrupt-id map, so the
        # field map travels under "edits" even when the person changes nothing.
        Command(resume={"edits": dict(edits)}),
        _thread_config(thread_id),
    )
    elapsed = time.perf_counter() - started
    rejected = state.get("rejected_fields") or ()
    if isinstance(rejected, tuple) and rejected:
        raise NonFlaggedFieldEdit(rejected)
    return _graph_run(state, elapsed, thread_id=thread_id)


def compile_document_graph(
    reader: SectionReader,
    *,
    today: date | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> _CompiledGraph:
    """Fan out to both section agents, validate, repair once, then review."""

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

    def apply_human_edits(state: DocumentState) -> dict[str, object]:
        return _apply_human_edits(state, today=today)

    builder = StateGraph(DocumentState)
    builder.add_node("read_header", read_header)
    builder.add_node("read_line_items", read_line_items)
    builder.add_node("merge", _merge)
    builder.add_node("reject_document", _reject_document)
    builder.add_node("validate", validate_first)
    builder.add_node("route_repair", _route_repair)
    builder.add_node("repair_header", repair_header)
    builder.add_node("repair_line_items", repair_line_items)
    builder.add_node("merge_repair", _merge)
    builder.add_node("validate_after_repair", validate_after_repair)
    builder.add_node("human_review", _human_review)
    builder.add_node("apply_human_edits", apply_human_edits)
    builder.add_edge(START, "read_header")
    builder.add_edge(START, "read_line_items")
    builder.add_edge(["read_header", "read_line_items"], "merge")
    builder.add_conditional_edges("merge", _route_after_first_merge)
    builder.add_edge("reject_document", END)
    builder.add_conditional_edges("validate", _route_after_validation)
    builder.add_edge("route_repair", "repair_header")
    builder.add_edge("route_repair", "repair_line_items")
    builder.add_edge(["repair_header", "repair_line_items"], "merge_repair")
    builder.add_edge("merge_repair", "validate_after_repair")
    builder.add_conditional_edges("validate_after_repair", _route_after_repair)
    builder.add_edge("human_review", "apply_human_edits")
    builder.add_conditional_edges("apply_human_edits", _route_after_human_edits)
    saver = InMemorySaver() if checkpointer is None else checkpointer
    _allow_checkpoint_types(saver)
    return cast(_CompiledGraph, builder.compile(checkpointer=saver))


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


def _missing_access_key(state: DocumentState) -> bool:
    """True when the header agent read the document and found no access key."""
    if state.get("header_failure") is not None:
        return False
    header = state.get("header")
    return isinstance(header, Header) and header.access_key == ""


def _route_after_first_merge(state: DocumentState) -> str:
    if _missing_access_key(state):
        return "reject_document"
    return "validate"


def _reject_document(_state: DocumentState) -> dict[str, object]:
    return {"document_rejected": True}


def _route_after_validation(state: DocumentState) -> str:
    if _needs_repair(state):
        return "route_repair"
    if _needs_human_review(state):
        return "human_review"
    return END


def _route_after_repair(state: DocumentState) -> str:
    if _needs_human_review(state):
        return "human_review"
    return END


def _route_after_human_edits(state: DocumentState) -> str:
    rejected = state.get("rejected_fields") or ()
    if isinstance(rejected, tuple) and rejected:
        return "human_review"
    report = state.get("report")
    if isinstance(report, ValidationReport) and _flagged(report):
        return "human_review"
    return END


def _needs_repair(state: DocumentState) -> bool:
    report = state.get("report")
    return (
        isinstance(report, ValidationReport)
        and state.get("extraction_failure") is None
        and isinstance(state.get("extraction"), Extraction)
        and report.outcome is DocumentOutcome.NEEDS_REPAIR
    )


def _needs_human_review(state: DocumentState) -> bool:
    report = state.get("report")
    return (
        isinstance(report, ValidationReport)
        and state.get("extraction_failure") is None
        and isinstance(state.get("extraction"), Extraction)
        and report.outcome is not DocumentOutcome.ACCEPTED
    )


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


def _human_review(state: DocumentState) -> dict[str, object]:
    report = state.get("report")
    flagged = _flagged(report) if isinstance(report, ValidationReport) else ()
    messages = (
        list(report.review_messages) if isinstance(report, ValidationReport) else []
    )
    payload = interrupt(
        {
            "flagged_fields": list(flagged),
            "review_messages": messages,
        }
    )
    edits = payload.get("edits") if isinstance(payload, dict) else None
    if not isinstance(edits, dict):
        raise TypeError("Human review edits must map field paths to values.")
    rejected = tuple(field for field in edits if field not in flagged)
    if rejected:
        return {"pending_edits": {}, "rejected_fields": rejected}
    return {"pending_edits": dict(edits), "rejected_fields": ()}


def _apply_human_edits(
    state: DocumentState, *, today: date | None
) -> dict[str, object]:
    extraction = state.get("extraction")
    edits = state.get("pending_edits") or {}
    if (
        not isinstance(extraction, Extraction)
        or not isinstance(edits, dict)
        or not edits
    ):
        return {}
    updated = _apply_field_edits(extraction, edits)
    previous = state.get("human_fields") or ()
    if not isinstance(previous, tuple):
        previous = ()
    return {
        "extraction": updated,
        "header": updated.header,
        "line_items": updated.line_items,
        "report": validate(updated, today=today),
        "human_fields": tuple(dict.fromkeys((*previous, *edits))),
    }


def apply_extraction_edits(
    extraction: Extraction, edits: Mapping[str, object]
) -> Extraction:
    """Return a copy with these field edits applied.

    Raises ``ValidationError`` or ``ValueError`` when a value does not fit
    the field. Does not check whether the field is flagged.
    """
    return _apply_field_edits(extraction, edits)


def _apply_field_edits(
    extraction: Extraction, edits: Mapping[str, object]
) -> Extraction:
    header_updates: dict[str, object] = {}
    item_updates: dict[int, dict[str, object]] = {}
    for path, value in edits.items():
        header_field = _HEADER_FIELD.fullmatch(path)
        if header_field is not None:
            header_updates[header_field.group(1)] = value
            continue
        item_field = _ITEM_FIELD.fullmatch(path)
        if item_field is not None:
            item_updates.setdefault(int(item_field.group(1)), {})[
                item_field.group(2)
            ] = value
            continue
        raise ValueError(f"Unknown extraction field: {path}")
    header = Header.model_validate({**extraction.header.model_dump(), **header_updates})
    items = list(extraction.line_items)
    for index, updates in item_updates.items():
        items[index] = LineItem.model_validate({**items[index].model_dump(), **updates})
    return Extraction(header=header, line_items=tuple(items))


def _graph_run(
    state: Mapping[str, object],
    latency_seconds: float,
    *,
    thread_id: str | None,
) -> GraphRun:
    extraction = state.get("extraction")
    report = state.get("report")
    initial = state.get("initial_report")
    failure = state.get("extraction_failure")
    resolved_extraction = extraction if isinstance(extraction, Extraction) else None
    resolved_report = report if isinstance(report, ValidationReport) else None
    resolved_initial = initial if isinstance(initial, ValidationReport) else None
    resolved_failure = failure if isinstance(failure, ExtractionFailure) else None
    input_tokens, cached_input_tokens, output_tokens, cache_write_tokens = _tokens(
        state
    )
    human_fields = state.get("human_fields", ())
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
        thread_id=thread_id,
        awaiting_human_review=bool(state.get("__interrupt__")),
        human_fields=human_fields if isinstance(human_fields, tuple) else (),
        rejected=bool(state.get("document_rejected", False)),
        cache_write_tokens=cache_write_tokens,
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


def _tokens(state: Mapping[str, object]) -> tuple[int, int, int, int]:
    incoming = 0
    cached = 0
    outgoing = 0
    cache_writes = 0
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
            cache_writes += usage.cache_write_tokens
    return incoming, cached, outgoing, cache_writes


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


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _allow_checkpoint_types(checkpointer: BaseCheckpointSaver[Any]) -> None:
    serde = getattr(checkpointer, "serde", None)
    if not isinstance(serde, JsonPlusSerializer):
        return
    checkpointer.serde = JsonPlusSerializer(
        allowed_msgpack_modules=_CHECKPOINT_TYPES,
        pickle_fallback=serde.pickle_fallback,
    )


def _usage(reading: SectionReading) -> TokenUsage:
    return TokenUsage(
        input_tokens=reading.input_tokens,
        cached_input_tokens=reading.cached_input_tokens,
        output_tokens=reading.output_tokens,
        cache_write_tokens=reading.cache_write_tokens,
    )
