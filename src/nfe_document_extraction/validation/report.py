"""Report returned by the validation seam."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FieldStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIABLE = "unverifiable"
    FLAGGED = "flagged"


class DocumentOutcome(StrEnum):
    ACCEPTED = "accepted"
    NEEDS_REPAIR = "needs_repair"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class RuleFailure:
    rule_id: str
    fields: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class RuleReport:
    rule_id: str
    covered_fields: tuple[str, ...]
    failures: tuple[RuleFailure, ...]


@dataclass(frozen=True)
class FieldReport:
    field: str
    status: FieldStatus
    failures: tuple[RuleFailure, ...]


@dataclass(frozen=True)
class ValidationReport:
    outcome: DocumentOutcome
    rules: tuple[RuleReport, ...]
    fields: tuple[FieldReport, ...]
    review_messages: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.outcome is DocumentOutcome.ACCEPTED
