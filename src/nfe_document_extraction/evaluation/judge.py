"""Combine a reading, the validator, and the score for one document."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.evaluation.metrics import (
    ExtractionFailure,
    InvoiceEvaluation,
)
from nfe_document_extraction.evaluation.scoring import score_document
from nfe_document_extraction.validation import RuleReport, validate


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
    repaired: bool = False,
    reached_human_review: bool = False,
    rejected: bool = False,
    initial_rules: Sequence[RuleReport] | None = None,
) -> InvoiceEvaluation:
    """Score a reading and record whether the validator caught an error.

    Planted-error documents are judged by detection of their planted rule.
    They do not count as validator catches on the clean set. ``initial_rules``
    is the first validation, before a repair; detection uses it when the
    caller repaired the reading that is being scored.
    """
    score = None
    validator_caught_error = False
    planted_error_detected = False
    if extraction is not None and extraction_failure is None:
        score = score_document(extraction, ground_truth)
        if not rejected:
            rules = (
                tuple(initial_rules)
                if initial_rules is not None
                else validate(extraction, today=today).rules
            )
            if planted_rule is None:
                validator_caught_error = any(rule.failures for rule in rules)
            else:
                planted_error_detected = any(
                    rule.rule_id == planted_rule and rule.failures for rule in rules
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
        repaired=repaired,
        reached_human_review=reached_human_review,
        rejected=rejected,
    )
