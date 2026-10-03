from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from nfe_document_extraction.domain import Extraction, Header, LineItem
from nfe_document_extraction.validation import (
    FieldReport,
    FieldStatus,
    RuleReport,
    ValidationReport,
    validate,
)

# Banco do Brasil CNPJ 00.000.000/0001-91, CPF 529.982.247-25, and CNPJ
# 11.222.333/0001-81 are published check-digit examples. The access key is:
# 35 2410 00000000000191 55 001 000000123 1 00000001, check digit 0.
ACCESS_KEY = "35241000000000000191550010000001231000000010"
ISSUER_CNPJ = "00000000000191"
RECIPIENT_CPF = "52998224725"
RECIPIENT_CNPJ = "11222333000181"


def test_valid_identifiers_are_accepted_and_rules_report_coverage() -> None:
    report = validate(_extraction())

    assert report.accepted
    rules = {rule.rule_id: rule for rule in report.rules}
    assert tuple(rules) == (
        "cnpj_cpf_check_digit",
        "access_key_check_digit",
        "access_key_cross_consistency",
    )
    assert rules["cnpj_cpf_check_digit"].failures == ()
    assert rules["cnpj_cpf_check_digit"].covered_fields == (
        "header.issuer_cnpj",
        "header.recipient_cnpj_cpf",
    )
    assert rules["access_key_check_digit"].covered_fields == ("header.access_key",)
    assert rules["access_key_cross_consistency"].covered_fields == (
        "header.access_key",
        "header.state",
        "header.issue_date",
        "header.issuer_cnpj",
        "header.model",
        "header.series",
        "header.number",
    )
    verified = {
        field.field for field in report.fields if field.status is FieldStatus.VERIFIED
    }
    assert verified == {
        "header.access_key",
        "header.state",
        "header.model",
        "header.series",
        "header.number",
        "header.issue_date",
        "header.issuer_cnpj",
        "header.recipient_cnpj_cpf",
    }
    assert all(
        field.status is FieldStatus.UNVERIFIABLE
        for field in report.fields
        if field.field not in verified
    )
    present = {field.field for field in report.fields}
    assert "header.icms_st_total" in present
    assert "header.unsupported_components.import_tax" in present
    assert "header.unsupported_components.returned_ipi" in present
    assert "header.unsupported_components.icms_relief" in present
    assert "header.unsupported_components.fcp_st" in present
    assert "line_items[0].product_code" in present
    assert "line_items[0].description" in present


def test_misread_issuer_cnpj_digit_fails_check_digit_and_cross_consistency() -> None:
    report = validate(_extraction(issuer_cnpj="00000000000181"))

    assert not report.accepted
    issuer = _field(report, "header.issuer_cnpj")
    assert issuer.status is FieldStatus.FLAGGED
    assert [failure.message for failure in issuer.failures] == [
        "Issuer CNPJ check digit does not match.",
        "Access key issuer CNPJ does not match issuer CNPJ.",
    ]
    assert [failure.rule_id for failure in issuer.failures] == [
        "cnpj_cpf_check_digit",
        "access_key_cross_consistency",
    ]
    assert _rule(report, "cnpj_cpf_check_digit").covered_fields == (
        "header.issuer_cnpj",
        "header.recipient_cnpj_cpf",
    )
    assert _rule(report, "access_key_check_digit").failures == ()
    assert _field(report, "header.recipient_cnpj_cpf").status is FieldStatus.VERIFIED
    assert _field(report, "header.access_key").status is FieldStatus.FLAGGED


def test_recipient_cnpj_is_accepted() -> None:
    document = _extraction(recipient_cnpj_cpf="11.222.333/0001-81")

    assert document.header.recipient_cnpj_cpf == RECIPIENT_CNPJ
    report = validate(document)
    assert report.accepted
    assert _field(report, "header.recipient_cnpj_cpf").status is FieldStatus.VERIFIED


def test_recipient_cpf_with_bad_check_digit_is_flagged() -> None:
    report = validate(_extraction(recipient_cnpj_cpf="529.982.247-26"))

    assert not report.accepted
    recipient = _field(report, "header.recipient_cnpj_cpf")
    assert recipient.status is FieldStatus.FLAGGED
    assert recipient.failures[0].message == "Recipient CPF check digit does not match."
    assert _field(report, "header.issuer_cnpj").status is FieldStatus.VERIFIED


def test_recipient_cnpj_with_bad_check_digit_is_flagged() -> None:
    report = validate(_extraction(recipient_cnpj_cpf="11.222.333/0001-82"))

    recipient = _field(report, "header.recipient_cnpj_cpf")
    assert recipient.status is FieldStatus.FLAGGED
    assert recipient.failures[0].message == "Recipient CNPJ check digit does not match."


def test_repeated_digit_issuer_cnpj_is_flagged() -> None:
    report = validate(_extraction(issuer_cnpj="00.000.000/0000-00"))

    issuer = _field(report, "header.issuer_cnpj")
    assert issuer.failures[0].message == (
        "Issuer CNPJ cannot be a sequence of repeated digits."
    )


def test_repeated_digit_recipient_cpf_is_flagged() -> None:
    report = validate(_extraction(recipient_cnpj_cpf="000.000.000-00"))

    recipient = _field(report, "header.recipient_cnpj_cpf")
    assert recipient.failures[0].message == (
        "Recipient CPF cannot be a sequence of repeated digits."
    )


def test_issuer_cpf_is_not_accepted_as_cnpj() -> None:
    report = validate(_extraction(issuer_cnpj="529.982.247-25"))

    issuer = _field(report, "header.issuer_cnpj")
    assert issuer.failures[0].message == "Issuer CNPJ must contain 14 digits."


def test_recipient_with_neither_cnpj_nor_cpf_length_is_flagged() -> None:
    report = validate(_extraction(recipient_cnpj_cpf="12345"))

    recipient = _field(report, "header.recipient_cnpj_cpf")
    assert recipient.failures[0].message == (
        "Recipient CNPJ/CPF must contain 11 digits (CPF) or 14 digits (CNPJ)."
    )
    assert recipient.failures[0].fields == ("header.recipient_cnpj_cpf",)


def test_access_key_with_bad_check_digit_is_flagged() -> None:
    report = validate(_extraction(access_key=ACCESS_KEY[:-1] + "1"))

    assert not report.accepted
    access_key = _field(report, "header.access_key")
    assert access_key.status is FieldStatus.FLAGGED
    assert access_key.failures[0].message == "Access key check digit does not match."
    assert _rule(report, "access_key_cross_consistency").failures == ()
    assert _field(report, "header.series").status is FieldStatus.VERIFIED


def test_short_access_key_flags_every_cross_checked_field() -> None:
    report = validate(_extraction(access_key="123"))

    assert not report.accepted
    messages = {
        failure.message for failure in _field(report, "header.access_key").failures
    }
    assert "Access key must contain 44 digits." in messages
    assert any("44 digits to compare" in message for message in messages)
    assert _field(report, "header.series").status is FieldStatus.FLAGGED
    assert _field(report, "header.issuer_cnpj").status is FieldStatus.FLAGGED


@pytest.mark.parametrize(
    ("overrides", "field", "message"),
    [
        ({"state": "RJ"}, "header.state", "Access key state does not match state."),
        ({"model": "65"}, "header.model", "Access key model does not match model."),
        ({"series": "2"}, "header.series", "Access key series does not match series."),
        (
            {"number": "124"},
            "header.number",
            "Access key number does not match number.",
        ),
        (
            {"issue_date": date(2024, 11, 3)},
            "header.issue_date",
            "Access key year/month does not match issue date.",
        ),
    ],
)
def test_access_key_component_mismatch_flags_that_field_and_the_key(
    overrides: dict[str, object],
    field: str,
    message: str,
) -> None:
    report = validate(_extraction(**overrides))

    assert not report.accepted
    flagged = _field(report, field)
    assert flagged.status is FieldStatus.FLAGGED
    assert flagged.failures[0].message == message
    assert flagged.failures[0].fields == ("header.access_key", field)
    assert _field(report, "header.access_key").status is FieldStatus.FLAGGED
    assert _field(report, "header.issuer_cnpj").status is FieldStatus.VERIFIED


def test_state_ibge_code_matches_the_access_key() -> None:
    report = validate(_extraction(state="35"))

    assert report.accepted
    assert _field(report, "header.state").status is FieldStatus.VERIFIED


def test_masked_identifiers_dates_and_money_are_normalized() -> None:
    document = _extraction(
        access_key="3524 1000 0000 0000 0191 5500 1000 0001 2310 0000 0010",
        state="sp",
        issuer_cnpj="00.000.000/0001-91",
        recipient_cnpj_cpf="529.982.247-25",
        issue_date="03/10/2024",
        series="001",
        number=123,
        total_products="10,50",
        freight="1.234,56",
        insurance=Decimal("0.01"),
        discount="R$ 2,00",
    )

    assert document.header.access_key == ACCESS_KEY
    assert document.header.state == "SP"
    assert document.header.issuer_cnpj == ISSUER_CNPJ
    assert document.header.recipient_cnpj_cpf == RECIPIENT_CPF
    assert document.header.issue_date == date(2024, 10, 3)
    assert document.header.issue_date.isoformat() == "2024-10-03"
    assert document.header.series == "001"
    assert document.header.number == "123"
    assert document.header.total_products == 1050
    assert document.header.freight == 123456
    assert document.header.insurance == 1
    assert document.header.discount == 200
    assert validate(document).accepted


def test_iso_issue_date_matches_the_access_key_year_and_month() -> None:
    document = _extraction(issue_date="2024-10-03T15:04:05")

    assert document.header.issue_date == date(2024, 10, 3)
    assert validate(document).accepted

    stamped = _extraction(issue_date=datetime(2024, 10, 3, 8, 30, tzinfo=UTC))
    assert stamped.header.issue_date == date(2024, 10, 3)
    assert validate(stamped).accepted


def test_quantity_and_unit_price_keep_printed_precision() -> None:
    item = LineItem(
        printed_order=1,
        product_code=" SKU-1 ",
        description="Plain bearing",
        ncm="84833090",
        cfop="5102",
        quantity="1,2500",
        unit_price="0.3333333333",
        item_total="10,00",
    )

    assert item.product_code == "SKU-1"
    assert str(item.quantity) == "1.2500"
    assert str(item.unit_price) == "0.3333333333"
    assert item.item_total == 1000

    whole = LineItem(
        printed_order=2,
        product_code="SKU-2",
        description="Bolt",
        ncm="73181500",
        cfop="5102",
        quantity=3,
        unit_price=Decimal("1.10"),
        item_total=330,
    )
    assert whole.quantity == Decimal(3)


def test_money_rejects_binary_floats() -> None:
    with pytest.raises(ValidationError):
        _extraction(total_products=10.5)


def _extraction(**overrides: object) -> Extraction:
    values: dict[str, object] = {
        "access_key": ACCESS_KEY,
        "state": "SP",
        "model": "55",
        "series": "1",
        "number": "123",
        "issue_date": date(2024, 10, 3),
        "issuer_cnpj": ISSUER_CNPJ,
        "recipient_cnpj_cpf": RECIPIENT_CPF,
        "total_products": 1_000,
        "freight": 0,
        "insurance": 0,
        "discount": 0,
        "ipi_total": 0,
        "other_expenses": 0,
        "icms_total": 0,
        "icms_st_total": 180,
        "invoice_total": 1_000,
        "unsupported_components": {
            "import_tax": False,
            "returned_ipi": False,
            "icms_relief": False,
            "fcp_st": False,
        },
    }
    values.update(overrides)
    return Extraction(
        header=Header.model_validate(values),
        line_items=(
            LineItem(
                printed_order=1,
                product_code="SKU-1",
                description="Plain bearing",
                ncm="84833090",
                cfop="5102",
                quantity=Decimal("1.2500"),
                unit_price=Decimal("0.3333333333"),
                item_total=1_000,
            ),
        ),
    )


def _field(report: ValidationReport, name: str) -> FieldReport:
    for field in report.fields:
        if field.field == name:
            return field
    raise AssertionError(name)


def _rule(report: ValidationReport, rule_id: str) -> RuleReport:
    for rule in report.rules:
        if rule.rule_id == rule_id:
            return rule
    raise AssertionError(rule_id)
