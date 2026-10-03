"""Schema of an Extraction."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict

from nfe_document_extraction.domain.normalization import (
    normalize_decimal,
    normalize_digits,
    normalize_issue_date,
    normalize_money,
    normalize_numeric_identifier,
    normalize_state,
    normalize_text,
)

_CONFIG = ConfigDict(extra="forbid", frozen=True)


class UnsupportedComponents(BaseModel):
    """Presence of invoice-total components this project does not calculate."""

    model_config = _CONFIG

    import_tax: bool
    returned_ipi: bool
    icms_relief: bool
    fcp_st: bool


class Header(BaseModel):
    """Invoice-level fields.

    Money is integer cents. CNPJ and CPF are digits. ``state`` is a UF
    abbreviation or the IBGE code carried in the access key. ``model`` is the
    two-digit document model, 55 for an NF-e.
    """

    model_config = _CONFIG

    access_key: Annotated[str, BeforeValidator(normalize_digits)]
    state: Annotated[str, BeforeValidator(normalize_state)]
    model: Annotated[str, BeforeValidator(normalize_digits)]
    series: Annotated[str, BeforeValidator(normalize_numeric_identifier)]
    number: Annotated[str, BeforeValidator(normalize_numeric_identifier)]
    issue_date: Annotated[date, BeforeValidator(normalize_issue_date)]
    issuer_cnpj: Annotated[str, BeforeValidator(normalize_digits)]
    recipient_cnpj_cpf: Annotated[str, BeforeValidator(normalize_digits)]
    total_products: Annotated[int, BeforeValidator(normalize_money)]
    freight: Annotated[int, BeforeValidator(normalize_money)]
    insurance: Annotated[int, BeforeValidator(normalize_money)]
    discount: Annotated[int, BeforeValidator(normalize_money)]
    ipi_total: Annotated[int, BeforeValidator(normalize_money)]
    other_expenses: Annotated[int, BeforeValidator(normalize_money)]
    icms_total: Annotated[int, BeforeValidator(normalize_money)]
    icms_st_total: Annotated[int, BeforeValidator(normalize_money)]
    invoice_total: Annotated[int, BeforeValidator(normalize_money)]
    unsupported_components: UnsupportedComponents


class LineItem(BaseModel):
    """One product row. Quantity and unit price keep the precision as printed."""

    model_config = _CONFIG

    printed_order: int
    product_code: Annotated[str, BeforeValidator(normalize_text)]
    description: Annotated[str, BeforeValidator(normalize_text)]
    ncm: Annotated[str, BeforeValidator(normalize_text)]
    cfop: Annotated[str, BeforeValidator(normalize_text)]
    quantity: Annotated[Decimal, BeforeValidator(normalize_decimal)]
    unit_price: Annotated[Decimal, BeforeValidator(normalize_decimal)]
    item_total: Annotated[int, BeforeValidator(normalize_money)]


class Extraction(BaseModel):
    """A reading of one DANFE: header plus line items in printed order."""

    model_config = _CONFIG

    header: Header
    line_items: tuple[LineItem, ...]
