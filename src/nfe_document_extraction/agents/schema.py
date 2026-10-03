"""Strict response schema for the single-call baseline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_CONFIG = ConfigDict(extra="forbid")


class UnsupportedPayload(BaseModel):
    """Unsupported invoice-total components, present or absent."""

    model_config = _CONFIG

    import_tax: bool = Field(
        description="True when import tax (II) is printed and non-zero."
    )
    returned_ipi: bool = Field(
        description="True when returned IPI is printed and non-zero."
    )
    icms_relief: bool = Field(
        description="True when ICMS relief (desoneração) is printed and non-zero."
    )
    fcp_st: bool = Field(description="True when FCP-ST is printed and non-zero.")


class HeaderPayload(BaseModel):
    """Header fields as printed on the DANFE."""

    model_config = _CONFIG

    access_key: str = Field(description="44-digit access key, digits only.")
    state: str = Field(description="Issuer UF abbreviation, such as SP.")
    model: str = Field(description="Document model. 55 is an NF-e.")
    series: str = Field(description="Invoice series, digits only.")
    number: str = Field(description="Invoice number, digits only.")
    issue_date: str = Field(description="Issue date as printed.")
    issuer_cnpj: str = Field(description="Issuer CNPJ, 14 digits.")
    recipient_cnpj_cpf: str = Field(
        description="Recipient CNPJ (14 digits) or CPF (11 digits)."
    )
    total_products: str = Field(description="Total products in reais, as printed.")
    freight: str = Field(description="Freight in reais, as printed. 0 when absent.")
    insurance: str = Field(description="Insurance in reais, as printed. 0 when absent.")
    discount: str = Field(description="Discount in reais, as printed. 0 when absent.")
    ipi_total: str = Field(description="IPI total in reais, as printed. 0 when absent.")
    other_expenses: str = Field(
        description="Other expenses in reais, as printed. 0 when absent."
    )
    icms_total: str = Field(
        description="ICMS total in reais, as printed. 0 when absent."
    )
    icms_st_total: str = Field(
        description="ICMS-ST total in reais, as printed. 0 when absent."
    )
    invoice_total: str = Field(description="Invoice total in reais, as printed.")
    unsupported_components: UnsupportedPayload


class LineItemPayload(BaseModel):
    """One product row as printed."""

    model_config = _CONFIG

    printed_order: int = Field(description="1-based row order on the DANFE.")
    product_code: str = Field(description="Product code as printed.")
    description: str = Field(description="Product description as printed.")
    ncm: str = Field(description="NCM, 8 digits.")
    cfop: str = Field(description="CFOP, 4 digits.")
    quantity: str = Field(description="Quantity as printed, keeping decimals.")
    unit_price: str = Field(description="Unit price as printed, keeping decimals.")
    item_total: str = Field(description="Line total in reais, as printed.")


class BaselinePayload(BaseModel):
    """One model call reading the whole DANFE."""

    model_config = _CONFIG

    header: HeaderPayload
    line_items: list[LineItemPayload]
