"""Graph seam: section agents, validator, one repair. The model is scripted."""

import threading
from datetime import date
from decimal import Decimal

from test_identifier_rules import _extraction, _line_item

from nfe_document_extraction.agents.sections import (
    RepairFieldNote,
    Section,
    SectionReading,
    SectionRequest,
)
from nfe_document_extraction.domain import Extraction, Header, LineItem
from nfe_document_extraction.evaluation.metrics import ExtractionFailure
from nfe_document_extraction.graph import document_graph_nodes, run_document
from nfe_document_extraction.validation import DocumentOutcome

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
    } <= names
    assert not any("supervisor" in name or "reconcil" in name for name in names)


def test_a_valid_reading_is_accepted_without_repair() -> None:
    invoice = _extraction()
    reader = _reader(_queues(invoice))

    run = run_document(reader, _PDF, today=_TODAY)

    assert run.outcome is DocumentOutcome.ACCEPTED
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
    assert run.repair_attempted is False
    assert "header.total_products" in run.flagged_fields
    assert "header.invoice_total" not in run.flagged_fields
    assert all(request.notes == () for request in reader.requests)
    assert len(reader.requests) == 2
    assert run.report is not None
    assert any("import tax" in message for message in run.report.review_messages)


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
