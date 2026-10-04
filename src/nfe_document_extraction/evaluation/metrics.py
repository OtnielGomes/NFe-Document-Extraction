"""Aggregate eval scores into one row per approach, rendition, and origin."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from nfe_document_extraction.evaluation.scoring import DocumentScore


class ExtractionFailure(StrEnum):
    PARSE_FAILURE = "parse_failure"
    REFUSAL = "refusal"


@dataclass(frozen=True)
class InvoiceEvaluation:
    """One document after extraction, validation, and scoring."""

    approach: str
    rendition: str
    origin: str
    planted_error: bool
    score: DocumentScore | None
    extraction_failure: ExtractionFailure | None
    validator_caught_error: bool
    planted_error_detected: bool
    cost_usd: Decimal
    latency_seconds: float


@dataclass(frozen=True)
class ResultRow:
    """Metrics for one Extraction approach, Rendition, and origin."""

    approach: str
    rendition: str
    origin: str
    documents: int
    field_accuracy: Decimal | None
    description_similarity: Decimal | None
    perfect_invoice_rate: Decimal | None
    cost_per_invoice_usd: Decimal
    latency_median_seconds: float
    latency_p95_seconds: float
    validator_errors_caught: int
    planted_detection_rate: Decimal | None
    extraction_failures: int


def summarize(invoices: Sequence[InvoiceEvaluation]) -> tuple[ResultRow, ...]:
    """Group invoices. Planted-error documents stay out of field accuracy."""
    groups: dict[tuple[str, str, str], list[InvoiceEvaluation]] = {}
    for invoice in invoices:
        key = (invoice.approach, invoice.rendition, invoice.origin)
        groups.setdefault(key, []).append(invoice)
    return tuple(_row(key, group) for key, group in groups.items())


def _row(key: tuple[str, str, str], invoices: Sequence[InvoiceEvaluation]) -> ResultRow:
    approach, rendition, origin = key
    clean = tuple(invoice for invoice in invoices if not invoice.planted_error)
    planted = tuple(invoice for invoice in invoices if invoice.planted_error)
    accurate = tuple(invoice.score for invoice in clean if invoice.score is not None)
    readable = tuple(invoice.score for invoice in invoices if invoice.score is not None)
    field_correct = sum(score.field_correct for score in accurate)
    field_opportunities = sum(score.field_opportunities for score in accurate)
    similarity_total = sum(
        (sum(score.description_similarities, Decimal(0)) for score in readable),
        Decimal(0),
    )
    aligned_items = sum(len(score.description_similarities) for score in readable)
    perfect_count = sum(score.perfect for score in readable)
    return ResultRow(
        approach=approach,
        rendition=rendition,
        origin=origin,
        documents=len(invoices),
        field_accuracy=_ratio(field_correct, field_opportunities),
        description_similarity=_ratio(similarity_total, aligned_items),
        perfect_invoice_rate=_ratio(perfect_count, len(invoices)),
        cost_per_invoice_usd=_mean_cost(invoices),
        latency_median_seconds=_percentile(
            [invoice.latency_seconds for invoice in invoices],
            Decimal("0.50"),
        ),
        latency_p95_seconds=_percentile(
            [invoice.latency_seconds for invoice in invoices],
            Decimal("0.95"),
        ),
        validator_errors_caught=sum(
            invoice.validator_caught_error for invoice in clean
        ),
        planted_detection_rate=_ratio(
            sum(invoice.planted_error_detected for invoice in planted),
            len(planted),
        ),
        extraction_failures=sum(
            invoice.extraction_failure is not None for invoice in invoices
        ),
    )


def _ratio(numerator: int | Decimal, denominator: int) -> Decimal | None:
    if denominator == 0:
        return None
    return Decimal(numerator) / Decimal(denominator)


def _mean_cost(invoices: Sequence[InvoiceEvaluation]) -> Decimal:
    if not invoices:
        return Decimal(0)
    return sum((invoice.cost_usd for invoice in invoices), Decimal(0)) / Decimal(
        len(invoices)
    )


def _percentile(values: Sequence[float], percentile: Decimal) -> float:
    """Linear interpolation between closest ranks (percentile type 7)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = 1 + (Decimal(len(ordered) - 1) * percentile)
    lower_index = int(rank.to_integral_value(rounding="ROUND_FLOOR")) - 1
    upper_index = int(rank.to_integral_value(rounding="ROUND_CEILING")) - 1
    fraction = rank - Decimal(lower_index + 1)
    lower = Decimal(str(ordered[lower_index]))
    upper = Decimal(str(ordered[upper_index]))
    return float(lower + (upper - lower) * fraction)
