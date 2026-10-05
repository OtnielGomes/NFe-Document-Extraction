"""Graph seam: section agents, validator, one repair, human review."""

import threading
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from test_identifier_rules import ISSUER_CNPJ, _extraction, _line_item

from nfe_document_extraction.agents.sections import (
    RepairFieldNote,
    Section,
    SectionReading,
    SectionRequest,
)
from nfe_document_extraction.domain import Extraction, Header, LineItem
from nfe_document_extraction.evaluation.metrics import ExtractionFailure
from nfe_document_extraction.graph import (
    NonFlaggedFieldEdit,
    compile_document_graph,
    document_graph_nodes,
    run_document,
    submit_human_review,
)
from nfe_document_extraction.validation import DocumentOutcome, FieldStatus

_PDF = b"%PDF-1.4 scripted"
_TODAY = date(2026, 10, 3)
_ITEM_SUM_MESSAGE = "Sum of item totals does not equal total products."


class ScriptedReader:
    """Returns queued section readings and records every request."""

    def __init__(
        self,
        readings: dict[Section, list[SectionReading]],
        *,
        synchronize_repair: bool = False,
    ) -> None:
        self._readings = readings
        self.requests: list[SectionRequest] = []
        self._lock = threading.Lock()
        self._repair_barrier = threading.Barrier(2) if synchronize_repair else None

    def read(self, request: SectionRequest) -> SectionReading:
        with self._lock:
            self.requests.append(request)
            reading = self._readings[request.section].pop(0)
        if request.notes and self._repair_barrier is not None:
            self._repair_barrier.wait(timeout=2)
        return reading


def test_the_graph_routes_in_code_without_a_supervisor() -> None:
    names = document_graph_nodes(_reader(_queues(_extraction())))

    assert {
        "read_header",
        "read_line_items",
        "merge",
        "validate",
        "route_repair",
        "repair_header",
        "repair_line_items",
        "merge_repair",
        "validate_after_repair",
        "human_review",
        "apply_human_edits",
    } <= names
    assert not any("supervisor" in name or "reconcil" in name for name in names)


def test_a_valid_reading_is_accepted_without_repair() -> None:
    invoice = _extraction()
    reader = _reader(_queues(invoice))

    run = run_document(reader, _PDF, today=_TODAY)

    assert run.outcome is DocumentOutcome.ACCEPTED
    assert run.awaiting_human_review is False
    assert run.human_fields == ()
    assert run.extraction == invoice
    assert run.repair_attempted is False
    assert run.flagged_fields == ()
    assert run.extraction_failure is None
    assert _sections(reader) == [Section.HEADER, Section.LINE_ITEMS]
    assert all(request.notes == () for request in reader.requests)
    assert all(request.document == _PDF for request in reader.requests)


def test_simulated_photo_pages_reach_the_first_read_and_the_repair() -> None:
    pages = (b"\xff\xd8\xffpage-one", b"\xff\xd8\xffpage-two")
    invoice = _extraction()
    broken = _extraction(issuer_cnpj="00000000000181")
    reader = _reader(
        {
            Section.HEADER: [_header(broken.header), _header(invoice.header)],
            Section.LINE_ITEMS: [_items(invoice.line_items)],
        }
    )

    run = run_document(reader, b"", pages=pages, today=_TODAY)

    assert run.outcome is DocumentOutcome.ACCEPTED
    assert len(reader.requests) == 3
    assert all(request.pages == pages for request in reader.requests)
    assert any(request.notes for request in reader.requests)


def test_a_header_failure_is_repaired_only_by_the_header_agent() -> None:
    invoice = _extraction()
    broken = _extraction(issuer_cnpj="00000000000181")
    reader = _reader(
        {
            Section.HEADER: [
                _header(broken.header),
                _header(invoice.header, input_tokens=5, output_tokens=6),
            ],
            Section.LINE_ITEMS: [_items(invoice.line_items, input_tokens=20)],
        }
    )

    run = run_document(reader, _PDF, today=_TODAY)

    assert run.outcome is DocumentOutcome.ACCEPTED
    assert run.repair_attempted is True
    assert run.extraction == invoice
    repairs = _repairs(reader, Section.HEADER)
    assert len(repairs) == 1
    assert _repairs(reader, Section.LINE_ITEMS) == []
    assert [request.section for request in reader.requests].count(Section.HEADER) == 2
    assert [request.section for request in reader.requests].count(
        Section.LINE_ITEMS
    ) == 1
    messages = [note.message for note in repairs[0].notes]
    assert "Issuer CNPJ check digit does not match." in messages
    assert "Access key issuer CNPJ does not match issuer CNPJ." in messages
    assert any("header.issuer_cnpj" in note.fields for note in repairs[0].notes)
    assert run.input_tokens == 5 + 20
    assert run.output_tokens == 6


def test_a_line_item_failure_is_repaired_only_by_the_line_items_agent() -> None:
    invoice = _extraction()
    broken_item = _line_item(
        quantity=Decimal(2),
        unit_price=Decimal("8.00"),
        item_total=1_000,
    )
    broken = _extraction(line_items=(broken_item,))
    reader = _reader(
        {
            Section.HEADER: [_header(invoice.header)],
            Section.LINE_ITEMS: [
                _items(broken.line_items),
                _items(invoice.line_items),
            ],
        }
    )

    run = run_document(reader, _PDF, today=_TODAY)

    assert run.outcome is DocumentOutcome.ACCEPTED
    assert run.repair_attempted is True
    repairs = _repairs(reader, Section.LINE_ITEMS)
    assert len(repairs) == 1
    assert _repairs(reader, Section.HEADER) == []
    note = repairs[0].notes[0]
    assert note.fields == (
        "line_items[0].quantity",
        "line_items[0].unit_price",
        "line_items[0].item_total",
    )
    assert "quantity" in note.message
    assert "unit price" in note.message


def test_a_cross_section_sum_sends_the_same_message_to_both_agents() -> None:
    invoice = _extraction()
    broken = _extraction(total_products=900, invoice_total=1_080)
    reader = _reader(
        {
            Section.HEADER: [_header(broken.header), _header(invoice.header)],
            Section.LINE_ITEMS: [
                _items(invoice.line_items),
                _items(invoice.line_items),
            ],
        },
        synchronize_repair=True,
    )

    run = run_document(reader, _PDF, today=_TODAY)

    assert run.outcome is DocumentOutcome.ACCEPTED
    assert run.repair_attempted is True
    header_notes = _repairs(reader, Section.HEADER)[0].notes
    item_notes = _repairs(reader, Section.LINE_ITEMS)[0].notes
    assert header_notes == (
        RepairFieldNote(
            message=_ITEM_SUM_MESSAGE,
            fields=("header.total_products",),
        ),
    )
    assert item_notes == (
        RepairFieldNote(
            message=_ITEM_SUM_MESSAGE,
            fields=("line_items[0].item_total",),
        ),
    )


def test_a_failed_repair_is_not_retried() -> None:
    broken = _extraction(issuer_cnpj="00000000000181")
    reader = _reader(
        {
            Section.HEADER: [_header(broken.header), _header(broken.header)],
            Section.LINE_ITEMS: [_items(broken.line_items)],
        }
    )

    run = run_document(reader, _PDF, today=_TODAY)

    assert run.repair_attempted is True
    assert [request.section for request in reader.requests].count(Section.HEADER) == 2
    assert _repairs(reader, Section.LINE_ITEMS) == []
    assert run.outcome is DocumentOutcome.HUMAN_REVIEW
    assert run.awaiting_human_review is True
    assert run.thread_id


def test_a_document_that_still_fails_goes_to_human_review_with_flagged_fields() -> None:
    broken = _extraction(issuer_cnpj="00000000000181")
    reader = _reader(
        {
            Section.HEADER: [_header(broken.header), _header(broken.header)],
            Section.LINE_ITEMS: [_items(broken.line_items)],
        }
    )

    run = run_document(reader, _PDF, today=_TODAY)

    assert run.outcome is DocumentOutcome.HUMAN_REVIEW
    assert run.awaiting_human_review is True
    assert run.extraction is not None
    assert "header.issuer_cnpj" in run.flagged_fields
    assert "header.access_key" in run.flagged_fields
    assert run.report is not None
    assert any(
        "Issuer CNPJ check digit does not match." in field.failures[0].message
        for field in run.report.fields
        if field.field == "header.issuer_cnpj"
    )


def test_an_unsupported_component_goes_straight_to_human_review() -> None:
    reviewed = _extraction(
        total_products=900,
        invoice_total=1,
        unsupported_components={
            "import_tax": True,
            "returned_ipi": False,
            "icms_relief": False,
            "fcp_st": False,
        },
    )
    reader = _reader(_queues(reviewed))

    run = run_document(reader, _PDF, today=_TODAY)

    assert run.outcome is DocumentOutcome.HUMAN_REVIEW
    assert run.awaiting_human_review is True
    assert run.repair_attempted is False
    assert "header.total_products" in run.flagged_fields
    assert "header.invoice_total" not in run.flagged_fields
    assert all(request.notes == () for request in reader.requests)
    assert len(reader.requests) == 2
    assert run.report is not None
    assert any("import tax" in message for message in run.report.review_messages)


def test_human_review_pauses_on_the_thread_and_a_valid_edit_is_accepted() -> None:
    invoice = _extraction()
    broken = _extraction(issuer_cnpj="00000000000181")
    reader = _reader(
        {
            Section.HEADER: [_header(broken.header), _header(broken.header)],
            Section.LINE_ITEMS: [_items(broken.line_items)],
        }
    )
    checkpointer = InMemorySaver()
    thread_id = "review-thread-1"

    paused = run_document(
        reader,
        _PDF,
        today=_TODAY,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )

    assert paused.thread_id == thread_id
    assert paused.awaiting_human_review is True
    assert paused.outcome is DocumentOutcome.HUMAN_REVIEW
    assert paused.human_fields == ()
    assert "header.issuer_cnpj" in paused.flagged_fields
    calls_before_review = len(reader.requests)

    accepted = submit_human_review(
        reader,
        thread_id,
        {"header.issuer_cnpj": ISSUER_CNPJ},
        checkpointer=checkpointer,
        today=_TODAY,
    )

    assert accepted.thread_id == thread_id
    assert accepted.awaiting_human_review is False
    assert accepted.outcome is DocumentOutcome.ACCEPTED
    assert accepted.extraction == invoice
    assert accepted.human_fields == ("header.issuer_cnpj",)
    assert accepted.repair_attempted is True
    assert len(reader.requests) == calls_before_review


def test_an_invalid_human_edit_stays_flagged_with_its_message() -> None:
    broken = _extraction(issuer_cnpj="00000000000181")
    reader = _reader(
        {
            Section.HEADER: [_header(broken.header), _header(broken.header)],
            Section.LINE_ITEMS: [_items(broken.line_items)],
        }
    )
    checkpointer = InMemorySaver()
    thread_id = "review-thread-invalid"
    run_document(
        reader,
        _PDF,
        today=_TODAY,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )
    calls_before_review = len(reader.requests)

    reviewed = submit_human_review(
        reader,
        thread_id,
        {"header.issuer_cnpj": "11222333000182"},
        checkpointer=checkpointer,
        today=_TODAY,
    )

    assert reviewed.outcome is DocumentOutcome.HUMAN_REVIEW
    assert reviewed.awaiting_human_review is True
    assert reviewed.extraction is not None
    assert reviewed.extraction.header.issuer_cnpj == "11222333000182"
    assert reviewed.human_fields == ("header.issuer_cnpj",)
    assert reviewed.report is not None
    issuer = next(
        field for field in reviewed.report.fields if field.field == "header.issuer_cnpj"
    )
    assert issuer.status is FieldStatus.FLAGGED
    assert issuer.failures[0].message == "Issuer CNPJ check digit does not match."
    assert reviewed.repair_attempted is True
    assert len(reader.requests) == calls_before_review

    corrected = submit_human_review(
        reader,
        thread_id,
        {"header.issuer_cnpj": ISSUER_CNPJ},
        checkpointer=checkpointer,
        today=_TODAY,
    )

    assert corrected.outcome is DocumentOutcome.ACCEPTED
    assert corrected.awaiting_human_review is False
    assert corrected.human_fields == ("header.issuer_cnpj",)
    assert corrected.extraction is not None
    assert corrected.extraction.header.issuer_cnpj == ISSUER_CNPJ
    assert len(reader.requests) == calls_before_review


def test_an_edit_to_a_field_that_is_not_flagged_is_rejected() -> None:
    broken = _extraction(issuer_cnpj="00000000000181")
    reader = _reader(
        {
            Section.HEADER: [_header(broken.header), _header(broken.header)],
            Section.LINE_ITEMS: [_items(broken.line_items)],
        }
    )
    checkpointer = InMemorySaver()
    thread_id = "review-thread-rejected"
    paused = run_document(
        reader,
        _PDF,
        today=_TODAY,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )
    assert paused.extraction is not None
    calls_before_review = len(reader.requests)

    with pytest.raises(NonFlaggedFieldEdit) as raised:
        submit_human_review(
            reader,
            thread_id,
            {"header.number": "999"},
            checkpointer=checkpointer,
            today=_TODAY,
        )

    assert raised.value.fields == ("header.number",)
    accepted = submit_human_review(
        reader,
        thread_id,
        {"header.issuer_cnpj": ISSUER_CNPJ},
        checkpointer=checkpointer,
        today=_TODAY,
    )
    assert accepted.outcome is DocumentOutcome.ACCEPTED
    assert accepted.extraction is not None
    assert accepted.extraction.header.number == "123"
    assert accepted.extraction.header.issuer_cnpj == ISSUER_CNPJ
    assert len(reader.requests) == calls_before_review


def test_a_paused_review_resumes_from_the_sqlite_thread_id(tmp_path: Path) -> None:
    invoice = _extraction()
    broken = _extraction(issuer_cnpj="00000000000181")
    reader = _reader(
        {
            Section.HEADER: [_header(broken.header), _header(broken.header)],
            Section.LINE_ITEMS: [_items(broken.line_items)],
        }
    )
    database = tmp_path / "reviews.sqlite"
    thread_id = "sqlite-review"

    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        paused = run_document(
            reader,
            _PDF,
            today=_TODAY,
            checkpointer=checkpointer,
            thread_id=thread_id,
        )
    calls_before_review = len(reader.requests)

    assert database.is_file()
    assert paused.thread_id == thread_id
    assert paused.awaiting_human_review is True

    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        accepted = submit_human_review(
            reader,
            thread_id,
            {"header.issuer_cnpj": ISSUER_CNPJ},
            checkpointer=checkpointer,
            today=_TODAY,
        )

    assert accepted.outcome is DocumentOutcome.ACCEPTED
    assert accepted.thread_id == thread_id
    assert accepted.extraction == invoice
    assert accepted.human_fields == ("header.issuer_cnpj",)
    assert len(reader.requests) == calls_before_review


def test_fixing_a_flagged_field_does_not_accept_an_unsupported_document() -> None:
    reviewed = _extraction(
        total_products=900,
        invoice_total=1,
        unsupported_components={
            "import_tax": True,
            "returned_ipi": False,
            "icms_relief": False,
            "fcp_st": False,
        },
    )
    reader = _reader(_queues(reviewed))
    checkpointer = InMemorySaver()
    thread_id = "review-unsupported"
    paused = run_document(
        reader,
        _PDF,
        today=_TODAY,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )
    calls_before_review = len(reader.requests)
    assert paused.repair_attempted is False
    assert paused.awaiting_human_review is True
    assert "header.total_products" in paused.flagged_fields

    result = submit_human_review(
        reader,
        thread_id,
        {"header.total_products": 1_000},
        checkpointer=checkpointer,
        today=_TODAY,
    )

    assert result.outcome is DocumentOutcome.HUMAN_REVIEW
    assert result.awaiting_human_review is False
    assert result.repair_attempted is False
    assert result.human_fields == ("header.total_products",)
    assert result.extraction is not None
    assert result.extraction.header.total_products == 1_000
    assert "header.total_products" not in result.flagged_fields
    assert result.report is not None
    assert any("import tax" in message for message in result.report.review_messages)
    assert len(reader.requests) == calls_before_review


def test_an_unsupported_document_with_nothing_flagged_still_pauses() -> None:
    reviewed = _extraction(
        unsupported_components={
            "import_tax": True,
            "returned_ipi": False,
            "icms_relief": False,
            "fcp_st": False,
        },
    )
    reader = _reader(_queues(reviewed))
    checkpointer = InMemorySaver()
    thread_id = "review-unsupported-only"
    paused = run_document(
        reader,
        _PDF,
        today=_TODAY,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )
    calls_before_review = len(reader.requests)

    assert paused.thread_id == thread_id
    assert paused.awaiting_human_review is True
    assert paused.outcome is DocumentOutcome.HUMAN_REVIEW
    assert paused.flagged_fields == ()
    assert paused.repair_attempted is False
    assert paused.report is not None
    assert any("import tax" in message for message in paused.report.review_messages)

    with pytest.raises(NonFlaggedFieldEdit) as raised:
        submit_human_review(
            reader,
            thread_id,
            {"header.number": "999"},
            checkpointer=checkpointer,
            today=_TODAY,
        )
    assert raised.value.fields == ("header.number",)

    finished = submit_human_review(
        reader,
        thread_id,
        {},
        checkpointer=checkpointer,
        today=_TODAY,
    )

    assert finished.outcome is DocumentOutcome.HUMAN_REVIEW
    assert finished.awaiting_human_review is False
    assert finished.human_fields == ()
    assert finished.thread_id == thread_id
    assert len(reader.requests) == calls_before_review


def test_a_valid_line_item_edit_is_accepted_without_another_model_call() -> None:
    invoice = _extraction()
    broken = _extraction(
        line_items=(
            _line_item(
                quantity=Decimal(2),
                unit_price=Decimal("8.00"),
                item_total=1_000,
            ),
        )
    )
    reader = _reader(
        {
            Section.HEADER: [_header(invoice.header)],
            Section.LINE_ITEMS: [
                _items(broken.line_items),
                _items(broken.line_items),
            ],
        }
    )
    checkpointer = InMemorySaver()
    thread_id = "review-line-item"
    paused = run_document(
        reader,
        _PDF,
        today=_TODAY,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )
    calls_before_review = len(reader.requests)
    assert paused.awaiting_human_review is True
    assert "line_items[0].quantity" in paused.flagged_fields

    accepted = submit_human_review(
        reader,
        thread_id,
        {"line_items[0].quantity": "1.2500"},
        checkpointer=checkpointer,
        today=_TODAY,
    )

    assert accepted.outcome is DocumentOutcome.ACCEPTED
    assert accepted.extraction == invoice
    assert accepted.human_fields == ("line_items[0].quantity",)
    assert len(reader.requests) == calls_before_review


def test_each_thread_reviews_its_own_document() -> None:
    invoice = _extraction()
    broken = _extraction(issuer_cnpj="00000000000181")
    reader = _reader(
        {
            Section.HEADER: [
                _header(invoice.header),
                _header(broken.header),
                _header(broken.header),
            ],
            Section.LINE_ITEMS: [
                _items(invoice.line_items),
                _items(broken.line_items),
            ],
        }
    )
    checkpointer = InMemorySaver()
    graph = compile_document_graph(reader, today=_TODAY, checkpointer=checkpointer)

    accepted = run_document(
        reader,
        _PDF,
        today=_TODAY,
        graph=graph,
        thread_id="accepted-doc",
    )
    paused = run_document(
        reader,
        _PDF,
        today=_TODAY,
        graph=graph,
        thread_id="review-doc",
    )

    assert accepted.outcome is DocumentOutcome.ACCEPTED
    assert accepted.thread_id == "accepted-doc"
    assert accepted.awaiting_human_review is False
    assert paused.thread_id == "review-doc"
    assert paused.awaiting_human_review is True
    assert paused.extraction is not None
    assert paused.extraction.header.issuer_cnpj == "00000000000181"
    calls_before_review = len(reader.requests)

    fixed = submit_human_review(
        reader,
        "review-doc",
        {"header.issuer_cnpj": ISSUER_CNPJ},
        checkpointer=checkpointer,
        today=_TODAY,
    )

    assert fixed.outcome is DocumentOutcome.ACCEPTED
    assert fixed.thread_id == "review-doc"
    assert fixed.extraction == invoice
    assert len(reader.requests) == calls_before_review


def test_a_section_parse_failure_is_an_extraction_error_without_repair() -> None:
    invoice = _extraction()
    reader = _reader(
        {
            Section.HEADER: [
                SectionReading(
                    section=Section.HEADER,
                    failure=ExtractionFailure.PARSE_FAILURE,
                )
            ],
            Section.LINE_ITEMS: [_items(invoice.line_items)],
        }
    )

    run = run_document(reader, _PDF, today=_TODAY)

    assert run.outcome is None
    assert run.extraction is None
    assert run.extraction_failure is ExtractionFailure.PARSE_FAILURE
    assert run.repair_attempted is False
    assert len(reader.requests) == 2


def test_a_repair_that_does_not_parse_keeps_the_original_reading() -> None:
    broken = _extraction(issuer_cnpj="00000000000181")
    reader = _reader(
        {
            Section.HEADER: [
                _header(broken.header),
                SectionReading(
                    section=Section.HEADER,
                    failure=ExtractionFailure.REFUSAL,
                ),
            ],
            Section.LINE_ITEMS: [_items(broken.line_items)],
        }
    )

    run = run_document(reader, _PDF, today=_TODAY)

    assert run.outcome is DocumentOutcome.HUMAN_REVIEW
    assert run.awaiting_human_review is True
    assert run.extraction == broken
    assert run.extraction_failure is None
    assert "header.issuer_cnpj" in run.flagged_fields


def _reader(
    readings: dict[Section, list[SectionReading]],
    *,
    synchronize_repair: bool = False,
) -> ScriptedReader:
    return ScriptedReader(readings, synchronize_repair=synchronize_repair)


def _queues(invoice: Extraction) -> dict[Section, list[SectionReading]]:
    return {
        Section.HEADER: [_header(invoice.header)],
        Section.LINE_ITEMS: [_items(invoice.line_items)],
    }


def _header(
    header: Header,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> SectionReading:
    return SectionReading(
        section=Section.HEADER,
        header=header,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _items(
    items: tuple[LineItem, ...],
    *,
    input_tokens: int = 0,
) -> SectionReading:
    return SectionReading(
        section=Section.LINE_ITEMS,
        line_items=items,
        input_tokens=input_tokens,
    )


def _sections(reader: ScriptedReader) -> list[Section]:
    return sorted(request.section for request in reader.requests)


def _repairs(reader: ScriptedReader, section: Section) -> list[SectionRequest]:
    return [
        request
        for request in reader.requests
        if request.section is section and request.notes
    ]
