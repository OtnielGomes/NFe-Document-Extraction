"""Evaluation seam: score an Extraction against ground truth. No model calls."""

from datetime import date
from decimal import Decimal

from test_identifier_rules import _extraction, _line_item

from nfe_document_extraction.agents.config import load_baseline_config
from nfe_document_extraction.evaluation import (
    DocumentScore,
    ExtractionFailure,
    InvoiceEvaluation,
    render_results,
    score_document,
    summarize,
    token_cost_usd,
)


def test_identical_invoice_is_a_perfect_invoice() -> None:
    invoice = _extraction(line_items=(_line_item(description="Café  torrado"),))
    reading = _extraction(line_items=(_line_item(description="cafe TORRADO"),))

    score = score_document(reading, invoice)

    assert score.perfect
    assert score.field_accuracy == Decimal(1)
    assert score.missing_items == 0
    assert score.spurious_items == 0
    assert score.description_similarities == (Decimal(1),)


def test_one_mismatched_field_is_one_error() -> None:
    invoice = _extraction()
    reading = _extraction(invoice_total=1_181)

    score = score_document(reading, invoice)

    assert score.field_correct == 28
    assert score.field_opportunities == 29
    assert score.field_accuracy == Decimal(28) / Decimal(29)
    assert score.perfect is False


def test_description_at_the_threshold_is_a_perfect_invoice_but_not_an_exact_field() -> (
    None
):
    invoice = _extraction(line_items=(_line_item(description="abcdefghijklmnopqrst"),))
    reading = _extraction(line_items=(_line_item(description="abcdefghijklmnopqrsu"),))

    score = score_document(reading, invoice)

    assert score.description_similarities == (Decimal("0.95"),)
    assert score.perfect
    assert score.field_correct == 28
    assert score.field_opportunities == 29


def test_description_below_the_threshold_is_not_a_perfect_invoice() -> None:
    invoice = _extraction(line_items=(_line_item(description="abcdefghijklmnopqrs"),))
    reading = _extraction(line_items=(_line_item(description="abcdefghijklmnopqrt"),))

    score = score_document(reading, invoice)

    assert score.description_similarities == (Decimal(18) / Decimal(19),)
    assert score.perfect is False
    assert score.field_correct == 28


def test_skipped_line_item_is_one_missing_item_and_does_not_cascade() -> None:
    first = _line_item(product_code="A", description="Alpha", item_total=100)
    second = _line_item(
        printed_order=2,
        product_code="B",
        description="Beta",
        item_total=200,
    )
    third = _line_item(
        printed_order=3,
        product_code="C",
        description="Gamma",
        item_total=300,
    )
    invoice = _extraction(line_items=(first, second, third))
    reading = _extraction(line_items=(first, third))

    score = score_document(reading, invoice)

    assert score.missing_items == 1
    assert score.spurious_items == 0
    assert score.field_correct == 37
    assert score.field_opportunities == 38
    assert score.description_similarities == (Decimal(1), Decimal(1))
    assert score.perfect is False


def test_extra_line_item_is_one_spurious_item() -> None:
    kept = _line_item(product_code="A", description="Alpha", item_total=100)
    extra = _line_item(
        printed_order=2,
        product_code="Z",
        description="Not on the invoice",
        item_total=999,
    )
    invoice = _extraction(line_items=(kept,))
    reading = _extraction(line_items=(kept, extra))

    score = score_document(reading, invoice)

    assert score.spurious_items == 1
    assert score.missing_items == 0
    assert score.field_correct == 29
    assert score.field_opportunities == 30
    assert score.description_similarities == (Decimal(1),)


def test_same_product_code_aligns_even_when_the_description_is_wrong() -> None:
    invoice = _extraction(line_items=(_line_item(description="Alpha"),))
    reading = _extraction(
        line_items=(_line_item(description="completely different text"),)
    )

    score = score_document(reading, invoice)

    assert score.missing_items == 0
    assert score.spurious_items == 0
    assert score.field_correct == 28
    assert score.field_opportunities == 29
    assert score.perfect is False


def test_similar_description_aligns_even_when_the_product_code_is_wrong() -> None:
    description = "abcdefghijklmnopqrst"
    invoice = _extraction(
        line_items=(_line_item(product_code="SKU-1", description=description),)
    )
    reading = _extraction(
        line_items=(_line_item(product_code="OTHER", description=description),)
    )

    score = score_document(reading, invoice)

    assert score.missing_items == 0
    assert score.spurious_items == 0
    assert score.description_similarities == (Decimal(1),)
    assert score.field_correct == 28
    assert score.perfect is False


def test_unrelated_line_item_is_one_missing_and_one_spurious() -> None:
    invoice = _extraction(
        line_items=(_line_item(product_code="A", description="Alpha"),)
    )
    reading = _extraction(
        line_items=(_line_item(product_code="Z", description="Not on the invoice"),)
    )

    score = score_document(reading, invoice)

    assert score.missing_items == 1
    assert score.spurious_items == 1
    assert score.description_similarities == ()
    assert score.field_correct == 21
    assert score.field_opportunities == 23


def test_product_code_matches_after_case_and_accent_folding() -> None:
    invoice = _extraction(line_items=(_line_item(product_code="Café-01"),))
    reading = _extraction(line_items=(_line_item(product_code="cafe-01"),))

    score = score_document(reading, invoice)

    assert score.perfect
    assert score.field_accuracy == Decimal(1)


def _invoice(
    *,
    approach: str = "single-call baseline",
    rendition: str = "digital PDF",
    origin: str = "synthetic",
    planted_error: bool = False,
    score: DocumentScore | None = None,
    use_score: bool = True,
    extraction_failure: ExtractionFailure | None = None,
    validator_caught_error: bool = False,
    planted_error_detected: bool = False,
    cost_usd: Decimal = Decimal("0.01"),
    latency_seconds: float = 1.0,
) -> InvoiceEvaluation:
    resolved = score
    if use_score and resolved is None:
        resolved = score_document(_extraction(), _extraction())
    return InvoiceEvaluation(
        approach=approach,
        rendition=rendition,
        origin=origin,
        planted_error=planted_error,
        score=resolved,
        extraction_failure=extraction_failure,
        validator_caught_error=validator_caught_error,
        planted_error_detected=planted_error_detected,
        cost_usd=cost_usd,
        latency_seconds=latency_seconds,
    )


def test_planted_error_documents_are_excluded_from_field_accuracy() -> None:
    clean = _invoice(
        score=score_document(_extraction(invoice_total=1_181), _extraction())
    )
    planted = _invoice(
        planted_error=True,
        score=score_document(_extraction(), _extraction()),
        planted_error_detected=True,
    )

    row = summarize((clean, planted))[0]

    assert row.field_accuracy == Decimal(28) / Decimal(29)
    assert row.description_similarity == Decimal(1)
    assert row.perfect_invoice_rate == Decimal("0.5")
    assert row.planted_detection_rate == Decimal(1)
    assert row.documents == 2


def test_extraction_failure_is_counted_and_excluded_from_field_accuracy() -> None:
    read = _invoice()
    failed = _invoice(
        use_score=False,
        extraction_failure=ExtractionFailure.REFUSAL,
    )

    row = summarize((read, failed))[0]

    assert row.field_accuracy == Decimal(1)
    assert row.perfect_invoice_rate == Decimal("0.5")
    assert row.extraction_failures == 1


def test_reported_metrics_cover_cost_latency_and_validator_catches() -> None:
    invoices = (
        _invoice(
            cost_usd=Decimal("0.10"),
            latency_seconds=10,
            validator_caught_error=True,
            score=score_document(_extraction(), _extraction()),
        ),
        _invoice(cost_usd=Decimal("0.30"), latency_seconds=20),
        _invoice(cost_usd=Decimal("0.30"), latency_seconds=30),
        _invoice(cost_usd=Decimal("0.30"), latency_seconds=40),
        _invoice(
            cost_usd=Decimal("0.00"),
            latency_seconds=100,
            planted_error=True,
            planted_error_detected=False,
            validator_caught_error=True,
        ),
    )

    row = summarize(invoices)[0]

    assert row.cost_per_invoice_usd == Decimal("0.20")
    assert row.latency_median_seconds == 30
    assert row.latency_p95_seconds == 88
    assert row.validator_errors_caught == 1
    assert row.planted_detection_rate == Decimal(0)
    assert row.description_similarity == Decimal(1)


def test_results_table_has_one_row_per_approach_rendition_and_origin() -> None:
    rows = summarize(
        (
            _invoice(rendition="digital PDF", latency_seconds=2),
            _invoice(rendition="simulated photo", latency_seconds=4),
        )
    )

    text = render_results(
        rows,
        model_snapshot="gpt-4.1-mini-2025-04-14",
        pricing_as_of="2026-10-03",
    )

    assert "| Extraction approach | Rendition | Origin |" in text
    assert "| single-call baseline | digital PDF | synthetic |" in text
    assert "| single-call baseline | simulated photo | synthetic |" in text
    assert "0.95" in text
    assert "gpt-4.1-mini-2025-04-14" in text
    assert "2026-10-03" in text
    assert "Do not edit by hand." in text


def test_dated_pricing_table_converts_recorded_tokens_to_usd() -> None:
    price = load_baseline_config()

    assert price.model == "gpt-4.1-mini-2025-04-14"
    assert price.pricing_as_of == date(2026, 10, 3)
    assert token_cost_usd(1_000_000, 0, 1_000_000, price) == Decimal("2.00")
    assert token_cost_usd(1_000_000, 500_000, 0, price) == Decimal("0.25")
