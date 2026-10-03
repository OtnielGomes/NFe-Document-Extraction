"""Report returned by the validation seam."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FieldStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIABLE = "unverifiable"
    FLAGGED = "flagged"


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
    accepted: bool
    rules: tuple[RuleReport, ...]
    fields: tuple[FieldReport, ...]
