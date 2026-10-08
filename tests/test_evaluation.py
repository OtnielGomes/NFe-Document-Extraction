"""Evaluation seam: score an Extraction against ground truth. No model calls."""

from datetime import date
from decimal import Decimal

import pytest
from test_identifier_rules import _extraction, _line_item

from nfe_document_extraction.agents.config import (
    BASELINE_MODEL_ENV,
    CHEAP_VISION,
    CHEAP_VISION_MODEL_ENV,
    STRONG_VISION,
    STRONG_VISION_MODEL_ENV,
    TEXT_LAYER,
    TEXT_LAYER_MODEL_ENV,
    load_approach_config,
    load_baseline_config,
)
from nfe_document_extraction.evaluation import (
    DocumentScore,
    ExtractionFailure,
    InvoiceEvaluation,
    render_results,
    score_document,
    summarize,
    token_cost_usd,
)
from nfe_document_extraction.evaluation.judge import judge_reading
from nfe_document_extraction.validation import validate


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
    repaired: bool = False,
    reached_human_review: bool = False,
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
        repaired=repaired,
        reached_human_review=reached_human_review,
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


def test_repair_and_human_review_are_counted_on_clean_documents_only() -> None:
    invoices = (
        _invoice(repaired=True, validator_caught_error=True),
        _invoice(reached_human_review=True, validator_caught_error=True),
        _invoice(
            planted_error=True,
            repaired=True,
            reached_human_review=True,
            planted_error_detected=True,
        ),
    )

    row = summarize(invoices)[0]
    text = render_results(
        (row,),
        model_snapshot="gpt-4.1-mini-2025-04-14",
        pricing_as_of="2026-10-03",
    )

    assert row.errors_fixed_by_repair == 1
    assert row.errors_reaching_human_review == 1
    assert row.validator_errors_caught == 2
    assert "Errors fixed by repair" in text
    assert "Errors reaching Human review" in text
    assert "single Repair attempt" in text


def test_a_repaired_document_stays_caught_by_the_first_validation() -> None:
    initial = validate(_extraction(invoice_total=1), today=date(2026, 10, 3))

    judged = judge_reading(
        approach="cheap vision",
        rendition="digital PDF",
        origin="synthetic",
        ground_truth=_extraction(),
        extraction=_extraction(),
        extraction_failure=None,
        planted_rule=None,
        cost_usd=Decimal("0.01"),
        latency_seconds=1,
        today=date(2026, 10, 3),
        repaired=True,
        initial_rules=initial.rules,
    )

    assert judged.validator_caught_error
    assert judged.repaired
    assert judged.reached_human_review is False
    assert judged.score is not None
    assert judged.score.perfect


def test_dated_pricing_table_converts_recorded_tokens_to_usd(monkeypatch) -> None:
    for name in (
        BASELINE_MODEL_ENV,
        CHEAP_VISION_MODEL_ENV,
        STRONG_VISION_MODEL_ENV,
        TEXT_LAYER_MODEL_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    price = load_baseline_config()
    cheap = load_approach_config(CHEAP_VISION)
    strong = load_approach_config(STRONG_VISION)
    text_layer = load_approach_config(TEXT_LAYER)

    assert price.model == "gpt-4.1-mini-2025-04-14"
    assert price.pricing_as_of == date(2026, 10, 8)
    assert cheap.model == price.model
    assert cheap.pricing_as_of == price.pricing_as_of
    assert strong.model == "gpt-4.1-2025-04-14"
    assert strong.pricing_as_of == date(2026, 10, 8)
    assert token_cost_usd(1_000_000, 0, 1_000_000, price) == Decimal("2.00")
    assert token_cost_usd(1_000_000, 500_000, 0, price) == Decimal("0.25")
    assert token_cost_usd(1_000_000, 0, 1_000_000, cheap) == Decimal("2.00")
    assert token_cost_usd(1_000_000, 0, 1_000_000, strong) == Decimal(10)
    assert token_cost_usd(1_000_000, 500_000, 0, strong) == Decimal("1.25")
    assert text_layer.approach == "text layer + text model"
    assert text_layer.model == "gpt-4.1-mini-2025-04-14"
    assert text_layer.pricing_as_of == date(2026, 10, 8)
    assert token_cost_usd(1_000_000, 0, 1_000_000, text_layer) == Decimal("2.00")


def test_gpt_6_luna_cache_writes_are_priced_and_unpriced_rows_bill_them_at_zero(
    monkeypatch,
) -> None:
    monkeypatch.setenv(CHEAP_VISION_MODEL_ENV, "gpt-6-luna")
    for name in (BASELINE_MODEL_ENV, STRONG_VISION_MODEL_ENV, TEXT_LAYER_MODEL_ENV):
        monkeypatch.delenv(name, raising=False)

    luna = load_approach_config(CHEAP_VISION)
    mini = load_baseline_config()

    assert luna.model == "gpt-6-luna"
    assert luna.approach == "cheap vision"
    assert luna.cache_write_usd_per_million == Decimal("0.125")
    assert token_cost_usd(
        1_000_000, 0, 0, luna, cache_write_tokens=1_000_000
    ) == Decimal("0.125")
    assert mini.cache_write_usd_per_million == Decimal(0)
    assert token_cost_usd(
        1_000_000, 0, 0, mini, cache_write_tokens=1_000_000
    ) == Decimal(0)


def test_clean_documents_rejected_exclude_planted_errors_and_validator_catches() -> (
    None
):
    rejected = judge_reading(
        approach="cheap vision",
        rendition="digital PDF",
        origin="synthetic",
        ground_truth=_extraction(),
        extraction=_extraction(access_key=""),
        extraction_failure=None,
        planted_rule=None,
        cost_usd=Decimal("0.01"),
        latency_seconds=1,
        rejected=True,
    )
    planted = judge_reading(
        approach="cheap vision",
        rendition="digital PDF",
        origin="synthetic",
        ground_truth=_extraction(),
        extraction=_extraction(access_key=""),
        extraction_failure=None,
        planted_rule="access_key_check_digit",
        cost_usd=Decimal("0.01"),
        latency_seconds=1,
        rejected=True,
    )

    row = summarize((rejected, planted))[0]
    text = render_results(
        (row,),
        model_snapshot="gpt-4.1-mini-2025-04-14",
        pricing_as_of="2026-10-08",
    )

    assert rejected.rejected is True
    assert rejected.validator_caught_error is False
    assert rejected.reached_human_review is False
    assert rejected.score is not None
    assert planted.planted_error_detected is False
    assert row.clean_documents_rejected == 1
    assert row.validator_errors_caught == 0
    assert row.errors_reaching_human_review == 0
    assert row.planted_detection_rate == Decimal(0)
    assert "Clean documents rejected" in text
    assert "| 0 | 0 | 0.0000 | 0 | 1 |" in text


def test_env_model_override_keeps_the_approach_and_uses_the_snapshot_price(
    monkeypatch,
) -> None:
    monkeypatch.setenv(STRONG_VISION_MODEL_ENV, "gpt-4.1-mini-2025-04-14")

    strong = load_approach_config(STRONG_VISION)

    assert strong.approach == "strong vision"
    assert strong.model == "gpt-4.1-mini-2025-04-14"
    assert token_cost_usd(1_000_000, 0, 1_000_000, strong) == Decimal("2.00")


def test_an_unpriced_env_model_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv(BASELINE_MODEL_ENV, "gpt-not-priced")

    with pytest.raises(ValueError, match="pricing table"):
        load_baseline_config()


def test_a_blank_env_model_keeps_the_pin(monkeypatch) -> None:
    monkeypatch.setenv(STRONG_VISION_MODEL_ENV, "   ")

    strong = load_approach_config(STRONG_VISION)

    assert strong.model == "gpt-4.1-2025-04-14"
