from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError
from test_identifier_rules import _extraction, _field, _line_item, _rule

from nfe_document_extraction.validation import DocumentOutcome, FieldStatus, validate


def _human_review_message(label: str) -> str:
    return (
        f"{label} is an unsupported component; the invoice total rule is skipped "
        "and the document goes straight to human review."
    )


def _components(**overrides: bool) -> dict[str, bool]:
    values = {
        "import_tax": False,
        "returned_ipi": False,
        "icms_relief": False,
        "fcp_st": False,
    }
    values.update(overrides)
    return values


def test_fractional_unit_price_matches_item_total_within_one_cent() -> None:
    report = validate(
        _extraction(
            total_products=100,
            icms_st_total=0,
            invoice_total=100,
            line_items=(
                _line_item(
                    quantity=Decimal(3),
                    unit_price=Decimal("0.3333333333"),
                    item_total=100,
                ),
            ),
        )
    )

    rule = _rule(report, "quantity_times_unit_price")
    assert rule.failures == ()
    assert rule.covered_fields == (
        "line_items[0].quantity",
        "line_items[0].unit_price",
        "line_items[0].item_total",
    )
    assert _field(report, "line_items[0].unit_price").status is FieldStatus.VERIFIED


def test_item_product_mismatch_flags_quantity_unit_price_and_item_total() -> None:
    report = validate(
        _extraction(
            total_products=1_300,
            invoice_total=1_480,
            line_items=(
                _line_item(),
                _line_item(
                    printed_order=2,
                    product_code="SKU-2",
                    description="Bolt",
                    quantity=Decimal(2),
                    unit_price=Decimal("1.00"),
                    item_total=300,
                ),
            ),
        )
    )

    failure = _rule(report, "quantity_times_unit_price").failures[0]
    assert failure.fields == (
        "line_items[1].quantity",
        "line_items[1].unit_price",
        "line_items[1].item_total",
    )
    assert failure.message == (
        "Item 2: quantity × unit price does not equal the item total within R$ 0.01."
    )
    for name in failure.fields:
        assert _field(report, name).status is FieldStatus.FLAGGED
    assert _field(report, "line_items[0].item_total").status is FieldStatus.VERIFIED
    assert (
        _field(report, "line_items[1].description").status is FieldStatus.UNVERIFIABLE
    )


@pytest.mark.parametrize(
    ("item_total", "matches"),
    [
        (99, True),
        (101, True),
        (98, False),
        (102, False),
    ],
)
def test_item_total_tolerance_is_one_cent_inclusive(
    item_total: int,
    matches: bool,
) -> None:
    report = validate(
        _extraction(
            total_products=item_total,
            icms_st_total=0,
            invoice_total=item_total,
            line_items=(
                _line_item(
                    quantity=Decimal(1),
                    unit_price=Decimal("1.00"),
                    item_total=item_total,
                ),
            ),
        )
    )

    failures = _rule(report, "quantity_times_unit_price").failures
    assert (failures == ()) is matches


def test_invoice_total_matches_products_minus_discount_plus_components() -> None:
    report = validate(
        _extraction(
            total_products=10_000,
            discount=500,
            icms_st_total=250,
            freight=1_500,
            insurance=100,
            other_expenses=300,
            ipi_total=700,
            icms_total=9_999,
            invoice_total=12_350,
            line_items=(
                _line_item(
                    quantity=Decimal(1),
                    unit_price=Decimal(100),
                    item_total=10_000,
                ),
            ),
        )
    )

    rule = _rule(report, "totals")
    assert rule.failures == ()
    assert rule.covered_fields == (
        "line_items[0].item_total",
        "header.total_products",
        "header.invoice_total",
        "header.discount",
        "header.icms_st_total",
        "header.freight",
        "header.insurance",
        "header.other_expenses",
        "header.ipi_total",
    )
    assert _field(report, "header.invoice_total").status is FieldStatus.VERIFIED
    assert _field(report, "header.icms_total").status is FieldStatus.UNVERIFIABLE


def test_item_sum_failure_flags_every_item_total_and_total_products() -> None:
    report = validate(
        _extraction(
            total_products=1_400,
            invoice_total=1_580,
            line_items=(
                _line_item(),
                _line_item(
                    printed_order=2,
                    product_code="SKU-2",
                    description="Bolt",
                    quantity=Decimal(1),
                    unit_price=Decimal("5.00"),
                    item_total=500,
                ),
            ),
        )
    )

    failure = _rule(report, "totals").failures[0]
    assert failure.message == "Sum of item totals does not equal total products."
    assert failure.fields == (
        "line_items[0].item_total",
        "line_items[1].item_total",
        "header.total_products",
    )
    for name in failure.fields:
        flagged = _field(report, name)
        assert flagged.status is FieldStatus.FLAGGED
        assert flagged.failures == (failure,)
    assert _field(report, "line_items[0].quantity").status is FieldStatus.VERIFIED
    assert _field(report, "header.freight").status is FieldStatus.VERIFIED
    assert _field(report, "header.invoice_total").status is FieldStatus.VERIFIED


def test_invoice_total_failure_flags_every_formula_term() -> None:
    report = validate(_extraction(invoice_total=1))

    assert report.outcome is DocumentOutcome.NEEDS_REPAIR
    assert report.review_messages == ()
    failure = _rule(report, "totals").failures[0]
    assert failure.message == (
        "Invoice total does not equal total products - discount + ICMS-ST total "
        "+ freight + insurance + other expenses + IPI total."
    )
    assert failure.fields == (
        "header.invoice_total",
        "header.total_products",
        "header.discount",
        "header.icms_st_total",
        "header.freight",
        "header.insurance",
        "header.other_expenses",
        "header.ipi_total",
    )
    for name in failure.fields:
        flagged = _field(report, name)
        assert flagged.status is FieldStatus.FLAGGED
        assert flagged.failures == (failure,)
    assert _field(report, "header.icms_total").status is FieldStatus.UNVERIFIABLE
    assert _field(report, "line_items[0].item_total").status is FieldStatus.VERIFIED


def test_both_total_failures_are_listed_on_total_products() -> None:
    report = validate(_extraction(total_products=1, invoice_total=1))

    total_products = _field(report, "header.total_products")
    assert [failure.message for failure in total_products.failures] == [
        "Sum of item totals does not equal total products.",
        (
            "Invoice total does not equal total products - discount + ICMS-ST total "
            "+ freight + insurance + other expenses + IPI total."
        ),
    ]
    assert [failure.rule_id for failure in total_products.failures] == [
        "totals",
        "totals",
    ]


def test_import_tax_skips_the_invoice_total_rule_and_requires_human_review() -> None:
    report = validate(
        _extraction(
            invoice_total=1,
            unsupported_components=_components(import_tax=True),
        )
    )

    assert report.outcome is DocumentOutcome.HUMAN_REVIEW
    assert not report.accepted
    assert report.review_messages == (_human_review_message("import tax"),)
    totals = _rule(report, "totals")
    assert totals.failures == ()
    assert "header.invoice_total" not in totals.covered_fields
    assert "header.total_products" in totals.covered_fields
    assert _field(report, "header.invoice_total").status is FieldStatus.UNVERIFIABLE
    assert _field(report, "header.discount").status is FieldStatus.UNVERIFIABLE
    assert _field(report, "header.total_products").status is FieldStatus.VERIFIED
    assert (
        _field(report, "header.unsupported_components.import_tax").status
        is FieldStatus.UNVERIFIABLE
    )


@pytest.mark.parametrize(
    ("component", "label"),
    [
        ("returned_ipi", "returned IPI"),
        ("icms_relief", "ICMS relief"),
        ("fcp_st", "FCP-ST"),
    ],
)
def test_each_unsupported_component_is_named_in_the_human_review_message(
    component: str,
    label: str,
) -> None:
    report = validate(
        _extraction(unsupported_components=_components(**{component: True}))
    )

    assert report.outcome is DocumentOutcome.HUMAN_REVIEW
    assert report.review_messages == (_human_review_message(label),)


def test_several_unsupported_components_are_named_in_schema_order() -> None:
    report = validate(
        _extraction(
            unsupported_components=_components(
                returned_ipi=True,
                fcp_st=True,
                import_tax=True,
            )
        )
    )

    assert report.review_messages == (
        _human_review_message("import tax"),
        _human_review_message("returned IPI"),
        _human_review_message("FCP-ST"),
    )


def test_unsupported_component_still_flags_a_failing_item_sum() -> None:
    report = validate(
        _extraction(
            total_products=1,
            unsupported_components=_components(import_tax=True),
        )
    )

    assert report.outcome is DocumentOutcome.HUMAN_REVIEW
    assert _field(report, "header.total_products").status is FieldStatus.FLAGGED
    assert _field(report, "line_items[0].item_total").status is FieldStatus.FLAGGED
    assert _field(report, "header.freight").status is FieldStatus.UNVERIFIABLE
    assert _field(report, "header.invoice_total").status is FieldStatus.UNVERIFIABLE


def test_consistent_document_is_accepted_and_leaves_uncovered_fields_unverifiable() -> (
    None
):
    report = validate(_extraction())

    assert report.outcome is DocumentOutcome.ACCEPTED
    assert report.accepted
    assert report.review_messages == ()
    for name in (
        "header.icms_total",
        "header.unsupported_components.import_tax",
        "header.unsupported_components.returned_ipi",
        "header.unsupported_components.icms_relief",
        "header.unsupported_components.fcp_st",
        "line_items[0].printed_order",
        "line_items[0].product_code",
        "line_items[0].description",
    ):
        field = _field(report, name)
        assert field.status is FieldStatus.UNVERIFIABLE
        assert field.failures == ()


def test_human_edited_invoice_total_uses_the_same_validator() -> None:
    extracted = _extraction(invoice_total=1)
    before = validate(extracted)
    corrected = extracted.model_copy(
        update={"header": extracted.header.model_copy(update={"invoice_total": 1_180})}
    )

    after = validate(corrected)

    assert before.outcome is DocumentOutcome.NEEDS_REPAIR
    assert after.outcome is DocumentOutcome.ACCEPTED
    assert _field(after, "header.invoice_total").status is FieldStatus.VERIFIED


def test_valid_ncm_cfop_and_past_issue_date_are_verified() -> None:
    report = validate(_extraction())

    rule = _rule(report, "ncm_cfop_issue_date")
    assert rule.failures == ()
    assert rule.covered_fields == (
        "line_items[0].ncm",
        "line_items[0].cfop",
        "header.issue_date",
    )
    assert _field(report, "line_items[0].ncm").status is FieldStatus.VERIFIED
    assert _field(report, "line_items[0].cfop").status is FieldStatus.VERIFIED


def test_malformed_ncm_and_cfop_are_flagged_on_that_item_only() -> None:
    report = validate(
        _extraction(
            total_products=1_500,
            invoice_total=1_680,
            line_items=(
                _line_item(),
                _line_item(
                    printed_order=2,
                    product_code="SKU-2",
                    description="Bolt",
                    ncm="8483.30.90",
                    cfop="510",
                    quantity=Decimal(1),
                    unit_price=Decimal("5.00"),
                    item_total=500,
                ),
            ),
        )
    )

    ncm = _field(report, "line_items[1].ncm")
    cfop = _field(report, "line_items[1].cfop")
    assert ncm.status is FieldStatus.FLAGGED
    assert ncm.failures[0].rule_id == "ncm_cfop_issue_date"
    assert ncm.failures[0].message == "NCM must contain 8 digits."
    assert ncm.failures[0].fields == ("line_items[1].ncm",)
    assert cfop.status is FieldStatus.FLAGGED
    assert cfop.failures[0].message == "CFOP must contain 4 digits."
    assert cfop.failures[0].fields == ("line_items[1].cfop",)
    assert _field(report, "line_items[0].ncm").status is FieldStatus.VERIFIED
    assert _field(report, "line_items[0].cfop").status is FieldStatus.VERIFIED
    assert report.outcome is DocumentOutcome.NEEDS_REPAIR


def test_future_issue_date_is_flagged() -> None:
    report = validate(
        _extraction(issue_date=date(2099, 10, 3)),
        today=date(2026, 10, 3),
    )

    issue_date = _field(report, "header.issue_date")
    assert issue_date.status is FieldStatus.FLAGGED
    future = [
        failure
        for failure in issue_date.failures
        if failure.rule_id == "ncm_cfop_issue_date"
    ]
    assert len(future) == 1
    assert future[0].message == "Issue date is in the future."
    assert future[0].fields == ("header.issue_date",)


def test_issue_date_on_the_clock_date_is_not_treated_as_future() -> None:
    clock = date(2026, 10, 3)
    report = validate(_extraction(issue_date=clock), today=clock)

    assert _rule(report, "ncm_cfop_issue_date").failures == ()


def test_issue_date_tomorrow_is_in_the_future() -> None:
    report = validate(
        _extraction(issue_date=date(2026, 10, 4)),
        today=date(2026, 10, 3),
    )

    issue_date = _field(report, "header.issue_date")
    assert any(
        failure.message == "Issue date is in the future."
        for failure in issue_date.failures
    )


def test_impossible_issue_date_is_rejected_before_validation() -> None:
    with pytest.raises(ValidationError):
        _extraction(issue_date="31/02/2024")
