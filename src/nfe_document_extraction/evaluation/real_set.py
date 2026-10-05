"""Read the Real set from a local path and derive printed ground truth."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from nfe_document_extraction.domain import (
    Extraction,
    Header,
    LineItem,
    UnsupportedComponents,
)
from nfe_document_extraction.domain.normalization import printed_decimal
from nfe_document_extraction.evaluation.dataset import (
    DIGITAL_PDF,
    PHONE_PHOTO,
    DatasetExample,
)

_PRICE_PRECISION = 4
_QUANTITY_PRECISION = 4
_MONEY_PRECISION = 2
_REAL_ORIGIN = "real"
_CLEAN_SPLIT = "clean"
_XML_NAME = "nfe.xml"
_PDF_NAME = "danfe.pdf"
_PHONE_PHOTO_DIRECTORY = "phone-photo"


def ground_truth_from_xml(xml: str) -> Extraction:
    """Ground truth as printed on the DANFE, using the synthetic derivation.

    Quantity and unit price keep the four decimals BrazilFiscalReport prints.
    Money is integer cents. Unsupported components follow the totals that the
    invoice-total rule does not model.
    """
    root = ET.fromstring(xml)
    invoice = _first(root, "infNFe")
    ide = _child(invoice, "ide")
    emit = _child(invoice, "emit")
    recipient = _child(invoice, "dest")
    totals = _child(_child(invoice, "total"), "ICMSTot")
    return Extraction(
        header=Header(
            access_key=_access_key(invoice),
            state=_text(_child(emit, "enderEmit"), "UF"),
            model=_text(ide, "mod"),
            series=_text(ide, "serie"),
            number=_text(ide, "nNF"),
            issue_date=_issue_date(ide),
            issuer_cnpj=_text(emit, "CNPJ"),
            recipient_cnpj_cpf=_recipient(recipient),
            total_products=_cents(_text(totals, "vProd")),
            freight=_cents(_text(totals, "vFrete")),
            insurance=_cents(_text(totals, "vSeg")),
            discount=_cents(_text(totals, "vDesc")),
            ipi_total=_cents(_text(totals, "vIPI")),
            other_expenses=_cents(_text(totals, "vOutro")),
            icms_total=_cents(_text(totals, "vICMS")),
            icms_st_total=_cents(_text(totals, "vST")),
            invoice_total=_cents(_text(totals, "vNF")),
            unsupported_components=UnsupportedComponents(
                import_tax=_present(totals, "vII"),
                returned_ipi=_present(totals, "vIPIDevol"),
                icms_relief=_present(totals, "vICMSDeson"),
                fcp_st=_present(totals, "vFCPST"),
            ),
        ),
        line_items=tuple(
            _line_item(item)
            for item in sorted(_children(invoice, "det"), key=_printed_order)
        ),
    )


def load_real_documents(root: Path) -> tuple[DatasetExample, ...]:
    """Read invoices kept outside version control.

    Each directory holds ``nfe.xml`` and at least one Rendition: ``danfe.pdf``
    and/or ``phone-photo/page-*.jpg``. Ground truth is derived from the XML.
    Nothing is written back to the directory.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Real set directory does not exist: {root}")
    documents: list[DatasetExample] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if folder.name.startswith("."):
            continue
        documents.extend(_invoice(folder))
    return tuple(documents)


def _invoice(folder: Path) -> tuple[DatasetExample, ...]:
    xml_path = folder / _XML_NAME
    if not xml_path.is_file():
        raise FileNotFoundError(f"{folder.name} is missing {_XML_NAME}")
    ground_truth = ground_truth_from_xml(xml_path.read_text(encoding="utf-8-sig"))
    pdf_path = folder / _PDF_NAME
    photo_pages = _phone_pages(folder)
    if not pdf_path.is_file() and not photo_pages:
        raise FileNotFoundError(f"{folder.name} has no DANFE rendition")
    examples: list[DatasetExample] = []
    if pdf_path.is_file():
        examples.append(
            _example(
                folder,
                ground_truth,
                pdf=pdf_path.read_bytes(),
                rendition=DIGITAL_PDF,
                pages=(),
            )
        )
    if photo_pages:
        examples.append(
            _example(
                folder,
                ground_truth,
                pdf=b"",
                rendition=PHONE_PHOTO,
                pages=photo_pages,
            )
        )
    return tuple(examples)


def _example(
    folder: Path,
    ground_truth: Extraction,
    *,
    pdf: bytes,
    rendition: str,
    pages: tuple[bytes, ...],
) -> DatasetExample:
    return DatasetExample(
        nfe_id=folder.name,
        origin=_REAL_ORIGIN,
        pdf=pdf,
        ground_truth=ground_truth,
        planted_rule=None,
        has_icms_st=ground_truth.header.icms_st_total > 0,
        split=_CLEAN_SPLIT,
        planted_field=None,
        rendition=rendition,
        pages=pages,
    )


def _phone_pages(folder: Path) -> tuple[bytes, ...]:
    photo_dir = folder / _PHONE_PHOTO_DIRECTORY
    if not photo_dir.exists():
        return ()
    if not photo_dir.is_dir():
        raise FileNotFoundError(f"{folder.name} phone photo path is not a directory")
    pages = tuple(path.read_bytes() for path in sorted(photo_dir.glob("page-*.jpg")))
    if not pages:
        raise FileNotFoundError(f"{folder.name} is missing phone photo pages")
    return pages


def _line_item(item: ET.Element) -> LineItem:
    product = _child(item, "prod")
    return LineItem(
        printed_order=_printed_order(item),
        product_code=_text(product, "cProd"),
        description=_text(product, "xProd"),
        ncm=_text(product, "NCM"),
        cfop=_text(product, "CFOP"),
        quantity=printed_decimal(_text(product, "qCom"), _QUANTITY_PRECISION),
        unit_price=printed_decimal(_text(product, "vUnCom"), _PRICE_PRECISION),
        item_total=_cents(_text(product, "vProd")),
    )


def _access_key(invoice: ET.Element) -> str:
    identifier = invoice.attrib.get("Id", "")
    if identifier.startswith("NFe") and identifier[3:].isdigit():
        return identifier[3:]
    raise ValueError("NF-e is missing its access key")


def _issue_date(ide: ET.Element) -> str:
    issued = _find(ide, "dhEmi")
    if issued is None:
        issued = _find(ide, "dEmi")
    if issued is None or not (issued.text and issued.text.strip()):
        raise ValueError("NF-e is missing dhEmi")
    return issued.text.strip()


def _recipient(recipient: ET.Element) -> str:
    for name in ("CNPJ", "CPF"):
        element = _find(recipient, name)
        if element is not None and element.text and element.text.strip():
            return element.text.strip()
    raise ValueError("NF-e recipient has no CNPJ or CPF")


def _printed_order(item: ET.Element) -> int:
    raw = item.attrib.get("nItem", "")
    if not raw.isdigit():
        raise ValueError("NF-e line item is missing nItem")
    return int(raw)


def _present(parent: ET.Element, name: str) -> bool:
    element = _find(parent, name)
    if element is None or element.text is None or element.text.strip() == "":
        return False
    return _cents(element.text.strip()) != 0


def _cents(xml_text: str) -> int:
    amount = printed_decimal(xml_text, _MONEY_PRECISION)
    return int((amount * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _first(root: ET.Element, name: str) -> ET.Element:
    if _local(root.tag) == name:
        return root
    for element in root.iter():
        if _local(element.tag) == name:
            return element
    raise ValueError(f"NF-e is missing {name}")


def _child(parent: ET.Element, name: str) -> ET.Element:
    element = _find(parent, name)
    if element is None:
        raise ValueError(f"NF-e is missing {name}")
    return element


def _find(parent: ET.Element, name: str) -> ET.Element | None:
    matches = _children(parent, name)
    if not matches:
        return None
    return matches[0]


def _children(parent: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(parent) if _local(child.tag) == name]


def _text(parent: ET.Element, name: str) -> str:
    element = _child(parent, name)
    if element.text is None or element.text.strip() == "":
        raise ValueError(f"NF-e is missing {name}")
    return element.text.strip()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
