"""Combine a reading, the validator, and the score for one document."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.evaluation.metrics import (
    ExtractionFailure,
    InvoiceEvaluation,
)
from nfe_document_extraction.evaluation.scoring import score_document
from nfe_document_extraction.validation import validate


def judge_reading(
    *,
    approach: str,
    rendition: str,
    origin: str,
    ground_truth: Extraction,
    extraction: Extraction | None,
    extraction_failure: ExtractionFailure | None,
    planted_rule: str | None,
    cost_usd: Decimal,
    latency_seconds: float,
    today: date | None = None,
) -> InvoiceEvaluation:
    """Score a reading and record whether the validator caught an error.

    Planted-error documents are judged by detection of their planted rule.
    They do not count as validator catches on the clean set.
    """
    score = None
    validator_caught_error = False
    planted_error_detected = False
    if extraction is not None and extraction_failure is None:
        score = score_document(extraction, ground_truth)
        report = validate(extraction, today=today)
        if planted_rule is None:
            validator_caught_error = not report.accepted
        else:
            planted_error_detected = any(
                rule.rule_id == planted_rule and rule.failures for rule in report.rules
            )
    return InvoiceEvaluation(
        approach=approach,
        rendition=rendition,
        origin=origin,
        planted_error=planted_rule is not None,
        score=score,
        extraction_failure=extraction_failure,
        validator_caught_error=validator_caught_error,
        planted_error_detected=planted_error_detected,
        cost_usd=cost_usd,
        latency_seconds=latency_seconds,
    )
