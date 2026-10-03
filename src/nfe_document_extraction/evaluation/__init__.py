"""Scoring, item alignment, and eval results."""

from nfe_document_extraction.evaluation.metrics import (
    ExtractionFailure,
    InvoiceEvaluation,
    ResultRow,
    summarize,
)
from nfe_document_extraction.evaluation.pricing import token_cost_usd
from nfe_document_extraction.evaluation.results import render_results
from nfe_document_extraction.evaluation.scoring import (
    DESCRIPTION_CORRECT_THRESHOLD,
    DocumentScore,
    score_document,
)

__all__ = [
    "DESCRIPTION_CORRECT_THRESHOLD",
    "DocumentScore",
    "ExtractionFailure",
    "InvoiceEvaluation",
    "ResultRow",
    "render_results",
    "score_document",
    "summarize",
    "token_cost_usd",
]
