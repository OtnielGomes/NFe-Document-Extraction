"""Generate the synthetic NF-e set from a fixed seed.

Unit prices are stored in the XML with extra decimals. Ground truth keeps the
four decimal places BrazilFiscalReport prints on the DANFE.

About 15% of documents carry one planted error. The DANFE prints that error,
and ground truth keeps the printed value.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from brazilfiscalreport.danfe import Danfe

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
)
from nfe_document_extraction.validation.validate import (
    ACCESS_KEY,
    ACCESS_KEY_CHECK_DIGIT,
    ACCESS_KEY_CROSS_CONSISTENCY,
    CNPJ_CPF_CHECK_DIGIT,
    NCM_CFOP_ISSUE_DATE,
    QUANTITY_TIMES_UNIT_PRICE,
    RECIPIENT_CNPJ_CPF,
    SERIES,
    TOTALS,
)

SYNTHETIC_SEED = 5
SYNTHETIC_COUNT = 100
_CPF_RECIPIENTS = 20
_ICMS_ST_DOCUMENTS = 30
_MULTIPAGE_DOCUMENTS = 20
_FREIGHT_DOCUMENTS = 25
_DISCOUNT_DOCUMENTS = 20
_IPI_DOCUMENTS = 15
# ~15% of the set, one document per planted error, spread across every rule.
_PLANTED_COUNT = 15
_PLANTED_RULES = (
    CNPJ_CPF_CHECK_DIGIT,
    ACCESS_KEY_CHECK_DIGIT,
    ACCESS_KEY_CROSS_CONSISTENCY,
    QUANTITY_TIMES_UNIT_PRICE,
    TOTALS,
    NCM_CFOP_ISSUE_DATE,
)
_CLEAN_SPLIT = "clean"
_PLANTED_SPLIT = "planted_errors"
_PLANTED_SERIES = "9"
_PLANTED_NCM = "9999999"
_PLANTED_CENTS = 100
_UNIT_PRICE_SHIFT = Decimal("200.0000")
_FIRST_UNIT_PRICE = "line_items[0].unit_price"
_INVOICE_TOTAL = "header.invoice_total"
_FIRST_NCM = "line_items[0].ncm"
# Long descriptions at this item count spill onto a second DANFE page.
_MULTIPAGE_MIN_ITEMS = 24
_PRICE_PRECISION = 4
_QUANTITY_PRECISION = 4
_NFE_NAMESPACE = "http://www.portalfiscal.inf.br/nfe"
# fpdf2 freezes both XMP timestamps only for this sentinel, which is what
# makes two renders of the same DANFE byte-identical.
_PDF_EPOCH = datetime(1969, 12, 31, 19, 0, tzinfo=UTC)

_PLACES = (
    ("35", "SP", "3550308", "Sao Paulo", "01001000"),
    ("33", "RJ", "3304557", "Rio de Janeiro", "20040020"),
    ("31", "MG", "3106200", "Belo Horizonte", "30130000"),
    ("41", "PR", "4106902", "Curitiba", "80010000"),
    ("43", "RS", "4314902", "Porto Alegre", "90010000"),
    ("29", "BA", "2927408", "Salvador", "40020000"),
    ("26", "PE", "2611606", "Recife", "50010000"),
    ("23", "CE", "2304400", "Fortaleza", "60010000"),
    ("42", "SC", "4205407", "Florianopolis", "88010000"),
    ("52", "GO", "5208707", "Goiania", "74010000"),
)
_PRODUCTS = (
    ("CAM", "Camiseta algodao", "61091000"),
    ("CAF", "Cafe torrado", "09012100"),
    ("CAD", "Caderno brochura", "48202000"),
    ("CAN", "Caneta azul", "96081000"),
    ("DET", "Detergente neutro", "34025000"),
    ("ARR", "Arroz tipo 1", "10063021"),
    ("FEI", "Feijao carioca", "07133399"),
    ("LEI", "Leite integral", "04012010"),
    ("SAB", "Sabonete barra", "34011190"),
    ("PAP", "Papel sulfite", "48025610"),
)
_LONG_DESCRIPTION = (
    "Parafuso sextavado zincado com arruela de pressao e porca "
    "para fixacao de estrutura metalica"
)
_NO_UNSUPPORTED = UnsupportedComponents(
    import_tax=False,
    returned_ipi=False,
    icms_relief=False,
    fcp_st=False,
)


@dataclass(frozen=True)
class PlantedError:
    """The single validation rule and field deliberately broken on a DANFE."""

    rule: str
    field: str


@dataclass(frozen=True)
class SyntheticDocument:
    """One fictitious NF-e and the DANFE values used as ground truth."""

    nfe_id: str
    has_icms_st: bool
    origin: str
    split: str
    planted_error: PlantedError | None
    xml: str
    pdf: bytes
    ground_truth: Extraction


@dataclass(frozen=True)
class _ItemDraft:
    printed_order: int
    product_code: str
    description: str
    ncm: str
    cfop: str
    quantity: Decimal
    quantity_xml: str
    unit_price: Decimal
    unit_price_xml: str
    item_total: int
    icms: int
    icms_st: int
    ipi: int


def generate_synthetic_set(
    *,
    seed: int = SYNTHETIC_SEED,
) -> tuple[SyntheticDocument, ...]:
    """Build the 100 fictitious NF-e documents from ``seed``."""
    rng = random.Random(seed)
    count = SYNTHETIC_COUNT
    cpf_recipients = _pick(rng, count, _CPF_RECIPIENTS)
    icms_st = _pick(rng, count, _ICMS_ST_DOCUMENTS)
    multipage = _pick(rng, count, _MULTIPAGE_DOCUMENTS)
    freight = _pick(rng, count, _FREIGHT_DOCUMENTS)
    discount = _pick(rng, count, _DISCOUNT_DOCUMENTS)
    ipi = _pick(rng, count, _IPI_DOCUMENTS)
    item_counts = _item_counts(multipage, count)
    documents = tuple(
        _document(
            rng,
            index=index,
            item_count=item_counts[index],
            recipient_is_cpf=index in cpf_recipients,
            has_icms_st=index in icms_st,
            has_freight=index in freight,
            has_discount=index in discount,
            has_ipi=index in ipi,
        )
        for index in range(count)
    )
    planted_rules = _planted_rules(seed)
    return tuple(
        _plant_error(document, planted_rules[index])
        if index in planted_rules
        else document
        for index, document in enumerate(documents)
    )


def write_synthetic_set(
    destination: Path,
    documents: Sequence[SyntheticDocument],
) -> None:
    """Write XML, DANFE PDF, ground truth, and metadata for each document."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for document in documents:
        folder = destination / document.nfe_id
        folder.mkdir()
        _write(folder / "nfe.xml", document.xml.encode("utf-8"))
        _write(folder / "danfe.pdf", document.pdf)
        _write(
            folder / "ground_truth.json",
            _json_bytes(document.ground_truth.model_dump(mode="json")),
        )
        _write(folder / "metadata.json", _json_bytes(_metadata(document)))


def _document(
    rng: random.Random,
    *,
    index: int,
    item_count: int,
    recipient_is_cpf: bool,
    has_icms_st: bool,
    has_freight: bool,
    has_discount: bool,
    has_ipi: bool,
) -> SyntheticDocument:
    place = _PLACES[index % len(_PLACES)]
    state_code, state, city_code, city, cep = place
    issue_date = date(2024, 1, 1) + timedelta(days=rng.randrange(366))
    series = (index % 3) + 1
    number = index + 1
    invoice_code = f"{rng.randrange(100_000_000):08d}"
    issuer_cnpj = _cnpj(rng)
    recipient = _cpf(rng) if recipient_is_cpf else _cnpj(rng)
    items = tuple(
        _item(
            rng,
            printed_order=order,
            long_description=item_count >= _MULTIPAGE_MIN_ITEMS,
            has_icms_st=has_icms_st,
            has_ipi=has_ipi,
        )
        for order in range(1, item_count + 1)
    )
    total_products = sum(item.item_total for item in items)
    freight = rng.randrange(100, 5_000) if has_freight else 0
    discount = rng.randrange(50, 2_000) if has_discount else 0
    if discount >= total_products:
        discount = total_products - 1 if total_products > 1 else 0
    icms_total = sum(item.icms for item in items)
    icms_st_total = sum(item.icms_st for item in items)
    ipi_total = sum(item.ipi for item in items)
    invoice_total = total_products - discount + icms_st_total + freight + ipi_total
    access_key = _access_key(
        state_code=state_code,
        issue_date=issue_date,
        issuer_cnpj=issuer_cnpj,
        series=series,
        number=number,
        invoice_code=invoice_code,
    )
    nfe_id = f"synthetic-{index + 1:03d}"
    ground_truth = Extraction(
        header=Header(
            access_key=access_key,
            state=state,
            model="55",
            series=str(series),
            number=str(number),
            issue_date=issue_date,
            issuer_cnpj=issuer_cnpj,
            recipient_cnpj_cpf=recipient,
            total_products=total_products,
            freight=freight,
            insurance=0,
            discount=discount,
            ipi_total=ipi_total,
            other_expenses=0,
            icms_total=icms_total,
            icms_st_total=icms_st_total,
            invoice_total=invoice_total,
            unsupported_components=_NO_UNSUPPORTED,
        ),
        line_items=tuple(_line_item(item) for item in items),
    )
    xml = _xml(
        access_key=access_key,
        state_code=state_code,
        state=state,
        city_code=city_code,
        city=city,
        cep=cep,
        issue_date=issue_date,
        series=series,
        number=number,
        invoice_code=invoice_code,
        issuer_cnpj=issuer_cnpj,
        recipient=recipient,
        recipient_is_cpf=recipient_is_cpf,
        issuer_name=f"Comercio Sintetico {index + 1:03d} Ltda",
        recipient_name=f"Destinatario Sintetico {index + 1:03d}",
        items=items,
        total_products=total_products,
        freight=freight,
        discount=discount,
        icms_total=icms_total,
        icms_st_total=icms_st_total,
        ipi_total=ipi_total,
        invoice_total=invoice_total,
    )
    return SyntheticDocument(
        nfe_id=nfe_id,
        has_icms_st=has_icms_st,
        origin="synthetic",
        split=_CLEAN_SPLIT,
        planted_error=None,
        xml=xml,
        pdf=_render_pdf(xml),
        ground_truth=ground_truth,
    )


def _item(
    rng: random.Random,
    *,
    printed_order: int,
    long_description: bool,
    has_icms_st: bool,
    has_ipi: bool,
) -> _ItemDraft:
    sku, short_name, ncm = _PRODUCTS[(printed_order - 1) % len(_PRODUCTS)]
    quantity_text = f"{Decimal(rng.randrange(1, 200_000)) / Decimal(10000):.4f}"
    quantity = _displayed(quantity_text, _QUANTITY_PRECISION)
    unit_price = (Decimal(rng.randrange(5_000, 2_500_000)) / Decimal(10000)).quantize(
        Decimal("0.0001")
    )
    unit_price_xml = _unit_price_xml(unit_price)
    unit_price = _displayed(unit_price_xml, _PRICE_PRECISION)
    item_total = _cents(quantity * unit_price)
    return _ItemDraft(
        printed_order=printed_order,
        product_code=f"{sku}-{printed_order:02d}",
        description=_LONG_DESCRIPTION if long_description else short_name,
        ncm=ncm,
        cfop="5405" if has_icms_st else "5102",
        quantity=quantity,
        quantity_xml=f"{quantity:.4f}",
        unit_price=unit_price,
        unit_price_xml=unit_price_xml,
        item_total=item_total,
        icms=_cents(Decimal(item_total) / Decimal(100) * Decimal("0.18")),
        icms_st=(
            max(1, _cents(Decimal(item_total) / Decimal(100) * Decimal("0.12")))
            if has_icms_st
            else 0
        ),
        ipi=(
            max(1, _cents(Decimal(item_total) / Decimal(100) * Decimal("0.05")))
            if has_ipi
            else 0
        ),
    )


def _line_item(item: _ItemDraft) -> LineItem:
    return LineItem(
        printed_order=item.printed_order,
        product_code=item.product_code,
        description=item.description,
        ncm=item.ncm,
        cfop=item.cfop,
        quantity=item.quantity,
        unit_price=item.unit_price,
        item_total=item.item_total,
    )


def _xml(
    *,
    access_key: str,
    state_code: str,
    state: str,
    city_code: str,
    city: str,
    cep: str,
    issue_date: date,
    series: int,
    number: int,
    invoice_code: str,
    issuer_cnpj: str,
    recipient: str,
    recipient_is_cpf: bool,
    issuer_name: str,
    recipient_name: str,
    items: Sequence[_ItemDraft],
    total_products: int,
    freight: int,
    discount: int,
    icms_total: int,
    icms_st_total: int,
    ipi_total: int,
    invoice_total: int,
) -> str:
    issued_at = f"{issue_date.isoformat()}T10:00:00-03:00"
    recipient_tag = "CPF" if recipient_is_cpf else "CNPJ"
    freight_mode = "0" if freight else "9"
    rows = "\n".join(_item_xml(item) for item in items)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<nfeProc versao="4.00" xmlns="{_NFE_NAMESPACE}">
  <NFe xmlns="{_NFE_NAMESPACE}">
    <infNFe versao="4.00" Id="NFe{access_key}">
      <ide>
        <cUF>{state_code}</cUF>
        <cNF>{invoice_code}</cNF>
        <natOp>VENDA</natOp>
        <mod>55</mod>
        <serie>{series}</serie>
        <nNF>{number}</nNF>
        <dhEmi>{issued_at}</dhEmi>
        <tpNF>1</tpNF>
        <idDest>1</idDest>
        <cMunFG>{city_code}</cMunFG>
        <tpImp>1</tpImp>
        <tpEmis>1</tpEmis>
        <cDV>{access_key[-1]}</cDV>
        <tpAmb>2</tpAmb>
        <finNFe>1</finNFe>
        <indFinal>1</indFinal>
        <indPres>1</indPres>
        <procEmi>0</procEmi>
        <verProc>synthetic</verProc>
      </ide>
      <emit>
        <CNPJ>{issuer_cnpj}</CNPJ>
        <xNome>{_xml_text(issuer_name)}</xNome>
        {_address("enderEmit", city_code, city, state, cep)}
        <IE>ISENTO</IE>
        <CRT>3</CRT>
      </emit>
      <dest>
        <{recipient_tag}>{recipient}</{recipient_tag}>
        <xNome>{_xml_text(recipient_name)}</xNome>
        {_address("enderDest", city_code, city, state, cep)}
        <indIEDest>9</indIEDest>
      </dest>
      {rows}
      <total>
        <ICMSTot>
          <vBC>{_reais(total_products)}</vBC>
          <vICMS>{_reais(icms_total)}</vICMS>
          <vICMSDeson>0.00</vICMSDeson>
          <vFCP>0.00</vFCP>
          <vBCST>{_reais(total_products if icms_st_total else 0)}</vBCST>
          <vST>{_reais(icms_st_total)}</vST>
          <vFCPST>0.00</vFCPST>
          <vFCPSTRet>0.00</vFCPSTRet>
          <vProd>{_reais(total_products)}</vProd>
          <vFrete>{_reais(freight)}</vFrete>
          <vSeg>0.00</vSeg>
          <vDesc>{_reais(discount)}</vDesc>
          <vII>0.00</vII>
          <vIPI>{_reais(ipi_total)}</vIPI>
          <vIPIDevol>0.00</vIPIDevol>
          <vPIS>0.00</vPIS>
          <vCOFINS>0.00</vCOFINS>
          <vOutro>0.00</vOutro>
          <vNF>{_reais(invoice_total)}</vNF>
        </ICMSTot>
      </total>
      <transp><modFrete>{freight_mode}</modFrete></transp>
      <pag>
        <detPag>
          <tPag>01</tPag>
          <vPag>{_reais(invoice_total)}</vPag>
        </detPag>
      </pag>
    </infNFe>
  </NFe>
  <protNFe versao="4.00">
    <infProt>
      <tpAmb>2</tpAmb>
      <verAplic>SYN</verAplic>
      <chNFe>{access_key}</chNFe>
      <dhRecbto>{issued_at}</dhRecbto>
      <nProt>135240000{number:06d}</nProt>
      <digVal>AAAAAAAAAAAAAAAAAAAAAAAAAAA=</digVal>
      <cStat>100</cStat>
      <xMotivo>Autorizado o uso da NF-e</xMotivo>
    </infProt>
  </protNFe>
</nfeProc>
"""


def _item_xml(item: _ItemDraft) -> str:
    return f"""<det nItem="{item.printed_order}">
        <prod>
          <cProd>{item.product_code}</cProd>
          <cEAN>SEM GTIN</cEAN>
          <xProd>{_xml_text(item.description)}</xProd>
          <NCM>{item.ncm}</NCM>
          <CFOP>{item.cfop}</CFOP>
          <uCom>UN</uCom>
          <qCom>{item.quantity_xml}</qCom>
          <vUnCom>{item.unit_price_xml}</vUnCom>
          <vProd>{_reais(item.item_total)}</vProd>
          <cEANTrib>SEM GTIN</cEANTrib>
          <uTrib>UN</uTrib>
          <qTrib>{item.quantity_xml}</qTrib>
          <vUnTrib>{item.unit_price_xml}</vUnTrib>
          <indTot>1</indTot>
        </prod>
        <imposto>
          {_icms_xml(item)}
          {_ipi_xml(item)}
        </imposto>
      </det>"""


def _icms_xml(item: _ItemDraft) -> str:
    base = _reais(item.item_total)
    icms = _reais(item.icms)
    if item.icms_st:
        return (
            "<ICMS><ICMS10><orig>0</orig><CST>10</CST><modBC>3</modBC>"
            f"<vBC>{base}</vBC><pICMS>18.00</pICMS><vICMS>{icms}</vICMS>"
            "<modBCST>4</modBCST>"
            f"<vBCST>{base}</vBCST><pICMSST>12.00</pICMSST>"
            f"<vICMSST>{_reais(item.icms_st)}</vICMSST></ICMS10></ICMS>"
        )
    return (
        "<ICMS><ICMS00><orig>0</orig><CST>00</CST><modBC>3</modBC>"
        f"<vBC>{base}</vBC><pICMS>18.00</pICMS><vICMS>{icms}</vICMS>"
        "</ICMS00></ICMS>"
    )


def _ipi_xml(item: _ItemDraft) -> str:
    if item.ipi == 0:
        return "<IPI><cEnq>999</cEnq><IPINT><CST>53</CST></IPINT></IPI>"
    return (
        "<IPI><cEnq>999</cEnq><IPITrib><CST>50</CST>"
        f"<vBC>{_reais(item.item_total)}</vBC><pIPI>5.00</pIPI>"
        f"<vIPI>{_reais(item.ipi)}</vIPI></IPITrib></IPI>"
    )


def _address(tag: str, city_code: str, city: str, state: str, cep: str) -> str:
    return (
        f"<{tag}><xLgr>Rua Sintetica</xLgr><nro>100</nro><xBairro>Centro</xBairro>"
        f"<cMun>{city_code}</cMun><xMun>{_xml_text(city)}</xMun><UF>{state}</UF>"
        f"<CEP>{cep}</CEP><cPais>1058</cPais><xPais>BRASIL</xPais></{tag}>"
    )


def _render_pdf(xml: str) -> bytes:
    danfe = Danfe(xml=xml)
    danfe.set_creation_date(_PDF_EPOCH)
    return bytes(danfe.output())


def _access_key(
    *,
    state_code: str,
    issue_date: date,
    issuer_cnpj: str,
    series: int,
    number: int,
    invoice_code: str,
) -> str:
    body = (
        f"{state_code}"
        f"{issue_date.year % 100:02d}{issue_date.month:02d}"
        f"{issuer_cnpj}"
        "55"
        f"{series:03d}"
        f"{number:09d}"
        "1"
        f"{invoice_code}"
    )
    check_digit = access_key_check_digit(body + "0")
    if check_digit is None:
        raise RuntimeError("access key body must contain 43 digits")
    return body + check_digit


def _item_counts(multipage: frozenset[int], count: int) -> dict[int, int]:
    multi_ids = sorted(multipage)
    single_ids = sorted(set(range(count)) - multipage)
    counts = {multi_ids[0]: 40, single_ids[0]: 1}
    for offset, doc_id in enumerate(multi_ids[1:]):
        counts[doc_id] = _MULTIPAGE_MIN_ITEMS + (offset % (40 - _MULTIPAGE_MIN_ITEMS))
    for offset, doc_id in enumerate(single_ids[1:]):
        counts[doc_id] = 2 + (offset % (_MULTIPAGE_MIN_ITEMS - 2))
    return counts


def _pick(rng: random.Random, count: int, size: int) -> frozenset[int]:
    indexes = list(range(count))
    rng.shuffle(indexes)
    return frozenset(indexes[:size])


def _cnpj(rng: random.Random) -> str:
    while True:
        base = f"{rng.randrange(10**12):012d}"
        if len(set(base)) == 1:
            continue
        digits = base + cnpj_check_digits(base)
        if len(set(digits)) > 1:
            return digits


def _cpf(rng: random.Random) -> str:
    while True:
        base = f"{rng.randrange(10**9):09d}"
        if len(set(base)) == 1:
            continue
        digits = base + cpf_check_digits(base)
        if len(set(digits)) > 1:
            return digits


def _unit_price_xml(printed: Decimal) -> str:
    for shift in (Decimal("1e-8"), Decimal("1e-9"), Decimal("-1e-8")):
        candidate = f"{printed + shift:.10f}"
        if (
            _displayed(candidate, _PRICE_PRECISION) == printed
            and Decimal(candidate) != printed
        ):
            return candidate
    raise RuntimeError(f"no XML unit price prints as {printed}")


def _displayed(xml_text: str, precision: int) -> Decimal:
    """Value BrazilFiscalReport prints, which formats through float."""
    return Decimal(format(float(xml_text), f".{precision}f"))


def _cents(amount: Decimal) -> int:
    return int((amount * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _reais(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def _xml_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _write(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _metadata(document: SyntheticDocument) -> dict[str, object]:
    metadata: dict[str, object] = {
        "nfe_id": document.nfe_id,
        "has_icms_st": document.has_icms_st,
        "origin": document.origin,
        "split": document.split,
    }
    if document.planted_error is not None:
        metadata["planted_error"] = {
            "rule": document.planted_error.rule,
            "field": document.planted_error.field,
        }
    return metadata


def _planted_rules(seed: int) -> dict[int, str]:
    """Choose documents without touching the RNG that builds invoice content."""
    rng = random.Random(f"planted-errors:{seed}")
    indexes = list(range(SYNTHETIC_COUNT))
    rng.shuffle(indexes)
    return {
        index: _PLANTED_RULES[position % len(_PLANTED_RULES)]
        for position, index in enumerate(indexes[:_PLANTED_COUNT])
    }


def _plant_error(document: SyntheticDocument, rule: str) -> SyntheticDocument:
    planters = {
        CNPJ_CPF_CHECK_DIGIT: _plant_recipient_check_digit,
        ACCESS_KEY_CHECK_DIGIT: _plant_access_key_check_digit,
        ACCESS_KEY_CROSS_CONSISTENCY: _plant_series_mismatch,
        QUANTITY_TIMES_UNIT_PRICE: _plant_unit_price,
        TOTALS: _plant_invoice_total,
        NCM_CFOP_ISSUE_DATE: _plant_ncm,
    }
    return planters[rule](document)


def _plant_recipient_check_digit(document: SyntheticDocument) -> SyntheticDocument:
    current = document.ground_truth.header.recipient_cnpj_cpf
    planted = _flip_last_digit(current)
    tag = "CPF" if len(current) == 11 else "CNPJ"
    xml = _replace_once(
        document.xml,
        f"<{tag}>{current}</{tag}>",
        f"<{tag}>{planted}</{tag}>",
    )
    header = document.ground_truth.header.model_copy(
        update={"recipient_cnpj_cpf": planted}
    )
    return _with_planted_error(
        document,
        xml=xml,
        ground_truth=document.ground_truth.model_copy(update={"header": header}),
        rule=CNPJ_CPF_CHECK_DIGIT,
        field=RECIPIENT_CNPJ_CPF,
    )


def _plant_access_key_check_digit(document: SyntheticDocument) -> SyntheticDocument:
    current = document.ground_truth.header.access_key
    planted = _flip_last_digit(current)
    if document.xml.count(current) != 2:
        raise RuntimeError("access key must appear in the NF-e id and the protocol")
    xml = document.xml.replace(current, planted)
    xml = _replace_once(
        xml,
        f"<cDV>{current[-1]}</cDV>",
        f"<cDV>{planted[-1]}</cDV>",
    )
    header = document.ground_truth.header.model_copy(update={"access_key": planted})
    return _with_planted_error(
        document,
        xml=xml,
        ground_truth=document.ground_truth.model_copy(update={"header": header}),
        rule=ACCESS_KEY_CHECK_DIGIT,
        field=ACCESS_KEY,
    )


def _plant_series_mismatch(document: SyntheticDocument) -> SyntheticDocument:
    current = document.ground_truth.header.series
    xml = _replace_once(
        document.xml,
        f"<serie>{current}</serie>",
        f"<serie>{_PLANTED_SERIES}</serie>",
    )
    header = document.ground_truth.header.model_copy(update={"series": _PLANTED_SERIES})
    return _with_planted_error(
        document,
        xml=xml,
        ground_truth=document.ground_truth.model_copy(update={"header": header}),
        rule=ACCESS_KEY_CROSS_CONSISTENCY,
        field=SERIES,
    )


def _plant_unit_price(document: SyntheticDocument) -> SyntheticDocument:
    item = document.ground_truth.line_items[0]
    printed = item.unit_price + _UNIT_PRICE_SHIFT
    xml_price = _unit_price_xml(printed)
    actual = Decimal(item.item_total) / Decimal(100)
    if abs(item.quantity * printed - actual) <= Decimal("0.01"):
        raise RuntimeError(
            "planted unit price is still within R$ 0.01 of the item total"
        )

    def transform(block: str) -> str:
        return _replace_first_tag(
            _replace_first_tag(block, "vUnCom", xml_price),
            "vUnTrib",
            xml_price,
        )

    xml = _map_block(document.xml, '<det nItem="1">', "</det>", transform)
    items = list(document.ground_truth.line_items)
    items[0] = item.model_copy(update={"unit_price": printed})
    return _with_planted_error(
        document,
        xml=xml,
        ground_truth=document.ground_truth.model_copy(
            update={"line_items": tuple(items)}
        ),
        rule=QUANTITY_TIMES_UNIT_PRICE,
        field=_FIRST_UNIT_PRICE,
    )


def _plant_invoice_total(document: SyntheticDocument) -> SyntheticDocument:
    current = document.ground_truth.header.invoice_total
    planted = current + _PLANTED_CENTS
    xml = _replace_once(
        document.xml,
        f"<vNF>{_reais(current)}</vNF>",
        f"<vNF>{_reais(planted)}</vNF>",
    )
    xml = _replace_once(
        xml,
        f"<vPag>{_reais(current)}</vPag>",
        f"<vPag>{_reais(planted)}</vPag>",
    )
    header = document.ground_truth.header.model_copy(update={"invoice_total": planted})
    return _with_planted_error(
        document,
        xml=xml,
        ground_truth=document.ground_truth.model_copy(update={"header": header}),
        rule=TOTALS,
        field=_INVOICE_TOTAL,
    )


def _plant_ncm(document: SyntheticDocument) -> SyntheticDocument:
    xml = _map_block(
        document.xml,
        '<det nItem="1">',
        "</det>",
        lambda block: _replace_first_tag(block, "NCM", _PLANTED_NCM),
    )
    items = list(document.ground_truth.line_items)
    items[0] = items[0].model_copy(update={"ncm": _PLANTED_NCM})
    return _with_planted_error(
        document,
        xml=xml,
        ground_truth=document.ground_truth.model_copy(
            update={"line_items": tuple(items)}
        ),
        rule=NCM_CFOP_ISSUE_DATE,
        field=_FIRST_NCM,
    )


def _with_planted_error(
    document: SyntheticDocument,
    *,
    xml: str,
    ground_truth: Extraction,
    rule: str,
    field: str,
) -> SyntheticDocument:
    return replace(
        document,
        split=_PLANTED_SPLIT,
        planted_error=PlantedError(rule=rule, field=field),
        xml=xml,
        pdf=_render_pdf(xml),
        ground_truth=ground_truth,
    )


def _map_block(
    xml: str,
    start_marker: str,
    end_marker: str,
    transform: Callable[[str], str],
) -> str:
    start = xml.find(start_marker)
    end = xml.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError(f"missing {start_marker}")
    return xml[:start] + transform(xml[start:end]) + xml[end:]


def _replace_first_tag(block: str, tag: str, value: str) -> str:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = block.find(open_tag)
    end = block.find(close_tag, start)
    if start < 0 or end < 0:
        raise RuntimeError(f"missing <{tag}>")
    return block[: start + len(open_tag)] + value + block[end:]


def _replace_once(text: str, old: str, new: str) -> str:
    found = text.count(old)
    if found != 1:
        raise RuntimeError(f"expected one occurrence of {old!r}, found {found}")
    return text.replace(old, new, 1)


def _flip_last_digit(value: str) -> str:
    return value[:-1] + str((int(value[-1]) + 1) % 10)


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    destination = Path(args[0]) if args else Path("data/synthetic")
    write_synthetic_set(destination, generate_synthetic_set())
