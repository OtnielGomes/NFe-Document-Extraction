"""Validation seam: an Extraction in, a report out."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

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
    DocumentOutcome,
    FieldReport,
    FieldStatus,
    RuleFailure,
    RuleReport,
    ValidationReport,
)

CNPJ_CPF_CHECK_DIGIT = "cnpj_cpf_check_digit"
ACCESS_KEY_CHECK_DIGIT = "access_key_check_digit"
ACCESS_KEY_CROSS_CONSISTENCY = "access_key_cross_consistency"
QUANTITY_TIMES_UNIT_PRICE = "quantity_times_unit_price"
TOTALS = "totals"
NCM_CFOP_ISSUE_DATE = "ncm_cfop_issue_date"

_INVOICE_TOTAL_FIELDS = (
    "header.invoice_total",
    "header.total_products",
    "header.discount",
    "header.icms_st_total",
    "header.freight",
    "header.insurance",
    "header.other_expenses",
    "header.ipi_total",
)
_ITEM_SUM_MESSAGE = "Sum of item totals does not equal total products."
_INVOICE_TOTAL_MESSAGE = (
    "Invoice total does not equal total products - discount + ICMS-ST total "
    "+ freight + insurance + other expenses + IPI total."
)

_ITEM_TOTAL_TOLERANCE = Decimal("0.01")
# Brasilia has stayed at UTC-3 since daylight saving ended in 2019.
_BRASILIA = timezone(timedelta(hours=-3))

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


def validate(extraction: Extraction, *, today: date | None = None) -> ValidationReport:
    """Apply every validation rule and report a status for every field.

    ``today`` is the civil date compared with the issue date. It defaults to
    the current date in Brasília.
    """
    as_of = _today() if today is None else today
    rules = tuple(rule(extraction) for rule in _rules(as_of))
    fields = _field_reports(extraction, rules)
    review_messages = _review_messages(extraction.header)
    return ValidationReport(
        outcome=_outcome(fields, review_messages),
        rules=rules,
        fields=fields,
        review_messages=review_messages,
    )


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
    parts = parse_access_key(access_key)
    expected = access_key_check_digit(access_key)
    if parts is None or expected is None:
        message = "Access key must contain 44 digits."
    elif parts.check_digit != expected:
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
                    _ACCESS_KEY_FIELDS,
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


def _quantity_times_unit_price(extraction: Extraction) -> RuleReport:
    covered: list[str] = []
    failures: list[RuleFailure] = []
    for index, item in enumerate(extraction.line_items):
        fields = (
            f"line_items[{index}].quantity",
            f"line_items[{index}].unit_price",
            f"line_items[{index}].item_total",
        )
        covered.extend(fields)
        if not _item_product_matches(item):
            failures.append(
                _failure(
                    QUANTITY_TIMES_UNIT_PRICE,
                    fields,
                    f"Item {item.printed_order}: quantity × unit price does not equal "
                    "the item total within R$ 0.01.",
                )
            )
    return RuleReport(QUANTITY_TIMES_UNIT_PRICE, tuple(covered), tuple(failures))


def _totals(extraction: Extraction) -> RuleReport:
    header = extraction.header
    item_total_fields = tuple(
        f"line_items[{index}].item_total" for index in range(len(extraction.line_items))
    )
    item_sum_fields = (*item_total_fields, "header.total_products")
    failures: list[RuleFailure] = []
    item_sum = sum(item.item_total for item in extraction.line_items)
    if item_sum != header.total_products:
        failures.append(_failure(TOTALS, item_sum_fields, _ITEM_SUM_MESSAGE))
    if _review_messages(header):
        return RuleReport(TOTALS, item_sum_fields, tuple(failures))
    covered = item_sum_fields + tuple(
        field for field in _INVOICE_TOTAL_FIELDS if field not in item_sum_fields
    )
    if header.invoice_total != _expected_invoice_total(header):
        failures.append(_failure(TOTALS, _INVOICE_TOTAL_FIELDS, _INVOICE_TOTAL_MESSAGE))
    return RuleReport(TOTALS, covered, tuple(failures))


def _review_messages(header: Header) -> tuple[str, ...]:
    components = header.unsupported_components
    present = (
        ("import tax", components.import_tax),
        ("returned IPI", components.returned_ipi),
        ("ICMS relief", components.icms_relief),
        ("FCP-ST", components.fcp_st),
    )
    return tuple(
        f"{label} is an unsupported component; the invoice total rule is skipped "
        "and the document goes straight to human review."
        for label, is_present in present
        if is_present
    )


def _outcome(
    fields: Sequence[FieldReport],
    review_messages: Sequence[str],
) -> DocumentOutcome:
    if review_messages:
        return DocumentOutcome.HUMAN_REVIEW
    if any(field.status is FieldStatus.FLAGGED for field in fields):
        return DocumentOutcome.NEEDS_REPAIR
    return DocumentOutcome.ACCEPTED


def _expected_invoice_total(header: Header) -> int:
    return (
        header.total_products
        - header.discount
        + header.icms_st_total
        + header.freight
        + header.insurance
        + header.other_expenses
        + header.ipi_total
    )


def _ncm_cfop_issue_date(extraction: Extraction, today: date) -> RuleReport:
    covered: list[str] = []
    failures: list[RuleFailure] = []
    for index, item in enumerate(extraction.line_items):
        ncm = f"line_items[{index}].ncm"
        cfop = f"line_items[{index}].cfop"
        covered.extend((ncm, cfop))
        if not _digits(item.ncm, 8):
            failures.append(
                _failure(
                    NCM_CFOP_ISSUE_DATE,
                    (ncm,),
                    "NCM must contain 8 digits.",
                )
            )
        if not _digits(item.cfop, 4):
            failures.append(
                _failure(
                    NCM_CFOP_ISSUE_DATE,
                    (cfop,),
                    "CFOP must contain 4 digits.",
                )
            )
    covered.append(ISSUE_DATE)
    if extraction.header.issue_date > today:
        failures.append(
            _failure(
                NCM_CFOP_ISSUE_DATE,
                (ISSUE_DATE,),
                "Issue date is in the future.",
            )
        )
    return RuleReport(NCM_CFOP_ISSUE_DATE, tuple(covered), tuple(failures))


def _digits(value: str, length: int) -> bool:
    return len(value) == length and value.isdigit()


def _item_product_matches(item: LineItem) -> bool:
    actual = Decimal(item.item_total) / Decimal(100)
    difference = abs(item.quantity * item.unit_price - actual)
    return difference <= _ITEM_TOTAL_TOLERANCE


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


def _rules(today: date) -> tuple[_Rule, ...]:
    def ncm_cfop_issue_date(extraction: Extraction) -> RuleReport:
        return _ncm_cfop_issue_date(extraction, today)

    return (
        _cnpj_cpf_check_digit,
        _access_key_check_digit,
        _access_key_cross_consistency,
        _quantity_times_unit_price,
        _totals,
        ncm_cfop_issue_date,
    )


def _today() -> date:
    return datetime.now(_BRASILIA).date()
