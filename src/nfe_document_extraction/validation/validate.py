"""Validation seam: an Extraction in, a report out."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

from nfe_document_extraction.domain import (
    Extraction,
    Header,
    LineItem,
    UnsupportedComponents,
)
from nfe_document_extraction.validation.identifiers import (
    access_key_check_digit,
    cnpj_check_digits,
    cpf_check_digits,
    parse_access_key,
    same_numeric_identifier,
    states_match,
)
from nfe_document_extraction.validation.report import (
    FieldReport,
    FieldStatus,
    RuleFailure,
    RuleReport,
    ValidationReport,
)

CNPJ_CPF_CHECK_DIGIT = "cnpj_cpf_check_digit"
ACCESS_KEY_CHECK_DIGIT = "access_key_check_digit"
ACCESS_KEY_CROSS_CONSISTENCY = "access_key_cross_consistency"

ISSUER_CNPJ = "header.issuer_cnpj"
RECIPIENT_CNPJ_CPF = "header.recipient_cnpj_cpf"
ACCESS_KEY = "header.access_key"
STATE = "header.state"
ISSUE_DATE = "header.issue_date"
MODEL = "header.model"
SERIES = "header.series"
NUMBER = "header.number"

_CNPJ_CPF_FIELDS = (ISSUER_CNPJ, RECIPIENT_CNPJ_CPF)
_ACCESS_KEY_FIELDS = (ACCESS_KEY,)
_CROSS_FIELDS = (
    ACCESS_KEY,
    STATE,
    ISSUE_DATE,
    ISSUER_CNPJ,
    MODEL,
    SERIES,
    NUMBER,
)

_Rule = Callable[[Extraction], RuleReport]


def validate(extraction: Extraction) -> ValidationReport:
    """Apply the identifier rules and report a status for every field."""
    rules = tuple(rule(extraction) for rule in _RULES)
    fields = _field_reports(extraction, rules)
    accepted = all(field.status is not FieldStatus.FLAGGED for field in fields)
    return ValidationReport(accepted=accepted, rules=rules, fields=fields)


def _cnpj_cpf_check_digit(extraction: Extraction) -> RuleReport:
    failures = (
        failure
        for failure in (
            _issuer_cnpj_failure(extraction.header.issuer_cnpj),
            _recipient_failure(extraction.header.recipient_cnpj_cpf),
        )
        if failure is not None
    )
    return RuleReport(CNPJ_CPF_CHECK_DIGIT, _CNPJ_CPF_FIELDS, tuple(failures))


def _issuer_cnpj_failure(cnpj: str) -> RuleFailure | None:
    if len(cnpj) != 14:
        return _failure(
            CNPJ_CPF_CHECK_DIGIT,
            (ISSUER_CNPJ,),
            "Issuer CNPJ must contain 14 digits.",
        )
    if _repeated_digits(cnpj):
        return _failure(
            CNPJ_CPF_CHECK_DIGIT,
            (ISSUER_CNPJ,),
            "Issuer CNPJ cannot be a sequence of repeated digits.",
        )
    if cnpj[12:] != cnpj_check_digits(cnpj):
        return _failure(
            CNPJ_CPF_CHECK_DIGIT,
            (ISSUER_CNPJ,),
            "Issuer CNPJ check digit does not match.",
        )
    return None


def _recipient_failure(document: str) -> RuleFailure | None:
    if len(document) == 11:
        return _person_document_failure(
            document,
            kind="CPF",
            expected_digits=cpf_check_digits(document),
        )
    if len(document) == 14:
        return _person_document_failure(
            document,
            kind="CNPJ",
            expected_digits=cnpj_check_digits(document),
        )
    return _failure(
        CNPJ_CPF_CHECK_DIGIT,
        (RECIPIENT_CNPJ_CPF,),
        "Recipient CNPJ/CPF must contain 11 digits (CPF) or 14 digits (CNPJ).",
    )


def _person_document_failure(
    document: str,
    *,
    kind: str,
    expected_digits: str,
) -> RuleFailure | None:
    if _repeated_digits(document):
        message = f"Recipient {kind} cannot be a sequence of repeated digits."
    elif document[-len(expected_digits) :] != expected_digits:
        message = f"Recipient {kind} check digit does not match."
    else:
        return None
    return _failure(CNPJ_CPF_CHECK_DIGIT, (RECIPIENT_CNPJ_CPF,), message)


def _access_key_check_digit(extraction: Extraction) -> RuleReport:
    access_key = extraction.header.access_key
    expected = access_key_check_digit(access_key)
    if expected is None:
        message = "Access key must contain 44 digits."
    elif access_key[43] != expected:
        message = "Access key check digit does not match."
    else:
        message = None
    failures: tuple[RuleFailure, ...] = ()
    if message is not None:
        failures = (_failure(ACCESS_KEY_CHECK_DIGIT, _ACCESS_KEY_FIELDS, message),)
    return RuleReport(ACCESS_KEY_CHECK_DIGIT, _ACCESS_KEY_FIELDS, failures)


def _access_key_cross_consistency(extraction: Extraction) -> RuleReport:
    access_key = extraction.header.access_key
    parts = parse_access_key(access_key)
    if parts is None:
        return RuleReport(
            ACCESS_KEY_CROSS_CONSISTENCY,
            _CROSS_FIELDS,
            (
                _failure(
                    ACCESS_KEY_CROSS_CONSISTENCY,
                    _CROSS_FIELDS,
                    "Access key must contain 44 digits to compare state, year/month, "
                    "issuer CNPJ, model, series, and number.",
                ),
            ),
        )

    header = extraction.header
    failures: list[RuleFailure] = []
    if not states_match(header.state, parts.state):
        failures.append(_mismatch(STATE, "Access key state does not match state."))
    if parts.year_month != _year_month(header.issue_date):
        failures.append(
            _mismatch(
                ISSUE_DATE,
                "Access key year/month does not match issue date.",
            )
        )
    if parts.issuer_cnpj != header.issuer_cnpj:
        failures.append(
            _mismatch(
                ISSUER_CNPJ,
                "Access key issuer CNPJ does not match issuer CNPJ.",
            )
        )
    if parts.model != header.model:
        failures.append(_mismatch(MODEL, "Access key model does not match model."))
    if not same_numeric_identifier(parts.series, header.series):
        failures.append(_mismatch(SERIES, "Access key series does not match series."))
    if not same_numeric_identifier(parts.number, header.number):
        failures.append(_mismatch(NUMBER, "Access key number does not match number."))
    return RuleReport(ACCESS_KEY_CROSS_CONSISTENCY, _CROSS_FIELDS, tuple(failures))


def _mismatch(field: str, message: str) -> RuleFailure:
    return _failure(ACCESS_KEY_CROSS_CONSISTENCY, (ACCESS_KEY, field), message)


def _year_month(issue_date: date) -> str:
    return f"{issue_date.year % 100:02d}{issue_date.month:02d}"


def _repeated_digits(value: str) -> bool:
    return len(value) > 0 and all(character == value[0] for character in value)


def _failure(rule_id: str, fields: tuple[str, ...], message: str) -> RuleFailure:
    return RuleFailure(rule_id=rule_id, fields=fields, message=message)


def _field_reports(
    extraction: Extraction,
    rules: Sequence[RuleReport],
) -> tuple[FieldReport, ...]:
    return tuple(_field_report(path, rules) for path in _field_paths(extraction))


def _field_report(path: str, rules: Sequence[RuleReport]) -> FieldReport:
    failures = tuple(
        failure for rule in rules for failure in rule.failures if path in failure.fields
    )
    covered = any(path in rule.covered_fields for rule in rules)
    if failures:
        status = FieldStatus.FLAGGED
    elif covered:
        status = FieldStatus.VERIFIED
    else:
        status = FieldStatus.UNVERIFIABLE
    return FieldReport(field=path, status=status, failures=failures)


def _field_paths(extraction: Extraction) -> tuple[str, ...]:
    paths: list[str] = []
    for name in Header.model_fields:
        if name == "unsupported_components":
            paths.extend(
                f"header.unsupported_components.{component}"
                for component in UnsupportedComponents.model_fields
            )
        else:
            paths.append(f"header.{name}")
    for index in range(len(extraction.line_items)):
        paths.extend(f"line_items[{index}].{name}" for name in LineItem.model_fields)
    return tuple(paths)


_RULES: tuple[_Rule, ...] = (
    _cnpj_cpf_check_digit,
    _access_key_check_digit,
    _access_key_cross_consistency,
)
