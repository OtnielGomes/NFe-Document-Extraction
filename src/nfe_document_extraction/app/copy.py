"""Visitor copy. English sentences are the keys; Portuguese is presentation.

The validator, the service, and the API keep the English text. Rewording one of
those sentences breaks the coverage test until this table is updated.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

CHOOSE_FILE = "Choose a PDF, JPG, or PNG."
CHOOSE_GALLERY_ITEM = "Choose an invoice from the gallery."
EMPTY_SCREEN = "Select a gallery invoice or upload a DANFE."
PAGE_LIMIT = "At most 3 pages and 10 MB."
GALLERY_NOTE = (
    "The gallery shows cached results for synthetic invoices. "
    "Selecting one does not call a model and does not create a trace."
)
UPLOAD_RULE = "Upload a PDF, JPG, or PNG."
HEIC_RULE = (
    "HEIC photos are not accepted. Export the photo as JPG and upload that file."
)
COMMENT_HINT = "Optional. Do not include invoice data; long numbers are removed."
FEEDBACK_FAILED = "Feedback could not be saved."
READING_QUESTION = "Was this reading correct?"
REJECTION_CONTROL = "This is a DANFE"
CACHED_REVIEW_NOTE = (
    "This cached result is not an open review. "
    "Upload a document to correct flagged fields."
)
REVIEW_INTRO = "Edit only the flagged fields. The same rules check your corrections."
DECIMAL_HINT = "Use ponto como separador decimal."

_EXACT: dict[str, str] = {
    CHOOSE_FILE: "Escolha um PDF, JPG ou PNG.",
    CHOOSE_GALLERY_ITEM: "Escolha uma nota na galeria.",
    EMPTY_SCREEN: "Escolha uma nota na galeria ou envie um DANFE.",
    PAGE_LIMIT: "No máximo 3 páginas e 10 MB.",
    GALLERY_NOTE: (
        "A galeria mostra leituras guardadas de notas sintéticas. "
        "Escolher uma não chama o modelo e não cria um registro."
    ),
    UPLOAD_RULE: "Envie um PDF, JPG ou PNG.",
    HEIC_RULE: (
        "Fotos HEIC não são aceitas. Exporte a foto como JPG e envie esse arquivo."
    ),
    COMMENT_HINT: "Opcional. Não inclua dados da nota; números longos são removidos.",
    FEEDBACK_FAILED: "Não foi possível salvar a sua resposta.",
    READING_QUESTION: "Esta leitura está correta?",
    REJECTION_CONTROL: "Isto é um DANFE",
    CACHED_REVIEW_NOTE: (
        "Esta leitura guardada não é uma revisão aberta. "
        "Envie um documento para corrigir os campos sinalizados."
    ),
    REVIEW_INTRO: (
        "Edite apenas os campos sinalizados. As mesmas regras conferem as correções."
    ),
    "Synthetic invoices": "Notas sintéticas",
    "Extract": "Extrair",
    "Outcome": "Resultado",
    "Field": "Campo",
    "Status": "Situação",
    "Value": "Valor",
    "Messages": "Mensagens",
    "Fields": "Campos",
    "Download Excel": "Baixar Excel",
    "Submit corrections": "Enviar correções",
    "Comment": "Comentário",
    "Thumbs up": "Correta",
    "Thumbs down": "Incorreta",
    "Accepted": "Aceito",
    "Needs review": "Precisa de revisão",
    "Verified": "Verificado",
    "Unverifiable": "Não verificável",
    "Flagged": "Sinalizado",
    "Summary": "Resumo",
    "Header": "Cabeçalho",
    "Line items": "Itens",
    "Review messages": "Mensagens da revisão",
    "digital PDF": "PDF digital",
    "simulated photo": "Foto simulada",
    "phone photo": "Foto de celular",
    "planted error": "Erro plantado",
    "Access key": "Chave de acesso",
    "State": "UF",
    "Model": "Modelo",
    "Series": "Série",
    "Number": "Número",
    "Issue date": "Data de emissão",
    "Issuer CNPJ": "CNPJ do emitente",
    "Recipient CNPJ/CPF": "CNPJ/CPF do destinatário",
    "Total products": "Total dos produtos",
    "Freight": "Frete",
    "Insurance": "Seguro",
    "Discount": "Desconto",
    "IPI total": "Total do IPI",
    "Other expenses": "Outras despesas",
    "ICMS total": "Total do ICMS",
    "ICMS-ST total": "Total do ICMS-ST",
    "Invoice total": "Valor total da nota",
    "Import tax": "Imposto de importação",
    "Returned IPI": "IPI devolvido",
    "ICMS relief": "Desoneração do ICMS",
    "FCP-ST": "FCP-ST",
    "Printed order": "Ordem impressa",
    "Product code": "Código do produto",
    "Description": "Descrição",
    "NCM": "NCM",
    "CFOP": "CFOP",
    "Quantity": "Quantidade",
    "Unit price": "Valor unitário",
    "Item total": "Valor do item",
    "Issuer CNPJ must contain 14 digits.": "O CNPJ do emitente deve ter 14 dígitos.",
    "Issuer CNPJ cannot be a sequence of repeated digits.": (
        "O CNPJ do emitente não pode ser uma sequência de dígitos repetidos."
    ),
    "Issuer CNPJ check digit does not match.": (
        "O dígito verificador do CNPJ do emitente não confere."
    ),
    "Recipient CNPJ/CPF must contain 11 digits (CPF) or 14 digits (CNPJ).": (
        "O CNPJ/CPF do destinatário deve ter 11 dígitos (CPF) ou 14 dígitos (CNPJ)."
    ),
    "Access key must contain 44 digits.": "A chave de acesso deve ter 44 dígitos.",
    "Access key check digit does not match.": (
        "O dígito verificador da chave de acesso não confere."
    ),
    (
        "Access key must contain 44 digits to compare state, year/month, "
        "issuer CNPJ, model, series, and number."
    ): (
        "A chave de acesso deve ter 44 dígitos para comparar UF, ano/mês, "
        "CNPJ do emitente, modelo, série e número."
    ),
    "Access key state does not match state.": (
        "A UF da chave de acesso não confere com a UF."
    ),
    "Access key year/month does not match issue date.": (
        "O ano/mês da chave de acesso não confere com a data de emissão."
    ),
    "Access key issuer CNPJ does not match issuer CNPJ.": (
        "O CNPJ do emitente na chave de acesso não confere com o CNPJ do emitente."
    ),
    "Access key model does not match model.": (
        "O modelo da chave de acesso não confere com o modelo."
    ),
    "Access key series does not match series.": (
        "A série da chave de acesso não confere com a série."
    ),
    "Access key number does not match number.": (
        "O número da chave de acesso não confere com o número."
    ),
    "Sum of item totals does not equal total products.": (
        "A soma dos valores dos itens não é igual ao total dos produtos."
    ),
    (
        "Invoice total does not equal total products - discount + ICMS-ST total "
        "+ freight + insurance + other expenses + IPI total."
    ): (
        "O valor total da nota não é igual ao total dos produtos − desconto "
        "+ total do ICMS-ST + frete + seguro + outras despesas + total do IPI."
    ),
    "NCM must contain 8 digits.": "O NCM deve ter 8 dígitos.",
    "CFOP must contain 4 digits.": "O CFOP deve ter 4 dígitos.",
    "Issue date is in the future.": "A data de emissão está no futuro.",
    "This PDF could not be read.": "Não foi possível ler este PDF.",
    "This address has already uploaded 1 document today. Try again tomorrow.": (
        "Este endereço já enviou 1 documento hoje. Tente novamente amanhã."
    ),
    "The demo has reached its daily upload limit. Try again tomorrow.": (
        "A demonstração atingiu o limite diário de envios. Tente novamente amanhã."
    ),
    "The demo has reached its daily spend limit. Try again tomorrow.": (
        "A demonstração atingiu o limite diário de custo. Tente novamente amanhã."
    ),
    "The document could not be read.": "Não foi possível ler o documento.",
    "This document does not look like a DANFE.": "Este documento não parece um DANFE.",
    "No document is waiting for review on this thread.": (
        "Nenhum documento aguarda revisão neste envio."
    ),
    "An edit is not a valid value for that field.": (
        "O valor informado não é válido para esse campo."
    ),
}

_COMPONENT_PT = {
    "import tax": "Imposto de importação",
    "returned IPI": "IPI devolvido",
    "ICMS relief": "Desoneração do ICMS",
    "FCP-ST": "FCP-ST",
}
_HEADER_FIELDS = {
    "access_key": "Access key",
    "state": "State",
    "model": "Model",
    "series": "Series",
    "number": "Number",
    "issue_date": "Issue date",
    "issuer_cnpj": "Issuer CNPJ",
    "recipient_cnpj_cpf": "Recipient CNPJ/CPF",
    "total_products": "Total products",
    "freight": "Freight",
    "insurance": "Insurance",
    "discount": "Discount",
    "ipi_total": "IPI total",
    "other_expenses": "Other expenses",
    "icms_total": "ICMS total",
    "icms_st_total": "ICMS-ST total",
    "invoice_total": "Invoice total",
}
_COMPONENT_FIELDS = {
    "import_tax": "Import tax",
    "returned_ipi": "Returned IPI",
    "icms_relief": "ICMS relief",
    "fcp_st": "FCP-ST",
}
_LINE_FIELDS = {
    "printed_order": "Printed order",
    "product_code": "Product code",
    "description": "Description",
    "ncm": "NCM",
    "cfop": "CFOP",
    "quantity": "Quantity",
    "unit_price": "Unit price",
    "item_total": "Item total",
}
_LINE_COLUMNS = (
    "product_code",
    "description",
    "ncm",
    "cfop",
    "quantity",
    "unit_price",
    "item_total",
)
_STATUS = {
    "verified": "Verified",
    "unverifiable": "Unverifiable",
    "flagged": "Flagged",
}
_OUTCOME = {
    "accepted": "Accepted",
    "human_review": "Needs review",
}
_LINE_ITEM = re.compile(r"line_items\[(\d+)\]\.([A-Za-z0-9_]+)\Z")
_Pattern = Callable[[re.Match[str]], str]


def _recipient_repeated(match: re.Match[str]) -> str:
    kind = match.group(1)
    return f"O {kind} do destinatário não pode ser uma sequência de dígitos repetidos."


def _recipient_check(match: re.Match[str]) -> str:
    kind = match.group(1)
    return f"O dígito verificador do {kind} do destinatário não confere."


def _item_total(match: re.Match[str]) -> str:
    return (
        f"Item {match.group(1)}: quantidade × valor unitário não é igual ao "
        "valor do item, dentro de R$ 0.01."
    )


def _unsupported(match: re.Match[str]) -> str:
    name = _COMPONENT_PT[match.group(1)]
    return (
        f"{name} é um componente não suportado; a regra do valor total da nota "
        "fica de fora e o documento vai direto para revisão humana."
    )


def _pages(match: re.Match[str]) -> str:
    return (
        f"Este documento tem {match.group(1)} páginas, acima do limite de "
        f"{match.group(2)}."
    )


def _megabytes(match: re.Match[str]) -> str:
    return f"Este arquivo é maior que {match.group(1)} MB."


def _bytes(match: re.Match[str]) -> str:
    return f"Este arquivo é maior que {match.group(1)} bytes."


def _documents(match: re.Match[str]) -> str:
    return (
        f"Este endereço já enviou {match.group(1)} documentos hoje. "
        "Tente novamente amanhã."
    )


def _flagged_only(match: re.Match[str]) -> str:
    return f"Só campos sinalizados podem ser editados: {match.group(1)}."


_PATTERNS: tuple[tuple[re.Pattern[str], _Pattern], ...] = (
    (
        re.compile(r"Recipient (CPF|CNPJ) cannot be a sequence of repeated digits\."),
        _recipient_repeated,
    ),
    (
        re.compile(r"Recipient (CPF|CNPJ) check digit does not match\."),
        _recipient_check,
    ),
    (
        re.compile(
            r"Item (\d+): quantity × unit price does not equal the item total "
            r"within R\$ 0\.01\."
        ),
        _item_total,
    ),
    (
        re.compile(
            r"(import tax|returned IPI|ICMS relief|FCP-ST) is an unsupported "
            r"component; the invoice total rule is skipped and the document goes "
            r"straight to human review\."
        ),
        _unsupported,
    ),
    (
        re.compile(
            r"This document has (\d+) pages, which is over the limit of (\d+)\."
        ),
        _pages,
    ),
    (re.compile(r"This file is larger than (\d+) MB\."), _megabytes),
    (re.compile(r"This file is larger than (\d+) bytes\."), _bytes),
    (
        re.compile(
            r"This address has already uploaded (\d+) documents today\. "
            r"Try again tomorrow\."
        ),
        _documents,
    ),
    (
        re.compile(r"Only flagged fields can be edited: (.+)\."),
        _flagged_only,
    ),
)


def visitor_text(english: str, language: str) -> str:
    """The sentence a visitor reads. Portuguese must be in the table."""
    if language == "en":
        return english
    if language != "pt":
        raise ValueError(language)
    translated = _EXACT.get(english)
    if translated is not None:
        return translated
    for pattern, render in _PATTERNS:
        match = pattern.fullmatch(english)
        if match is not None:
            return render(match)
    raise KeyError(english)


def intro_body(language: str) -> str:
    """The page text under the title. The product name stays above it."""
    return (
        f"{visitor_text(UPLOAD_RULE, language)} {visitor_text(PAGE_LIMIT, language)} "
        f"{visitor_text(HEIC_RULE, language)}\n\n"
        f"{visitor_text(GALLERY_NOTE, language)}"
    )


def decimal_hint(language: str) -> str:
    """Shown on the Portuguese review form. English has no extra hint."""
    if language == "pt":
        return DECIMAL_HINT
    if language != "en":
        raise ValueError(language)
    return ""


def status_label(status: str, language: str) -> str:
    return visitor_text(_STATUS[status], language)


def outcome_label(outcome: str, language: str) -> str:
    english = _OUTCOME.get(outcome)
    if english is None:
        return outcome
    return visitor_text(english, language)


def line_headers(language: str) -> tuple[str, ...]:
    return tuple(visitor_text(_LINE_FIELDS[name], language) for name in _LINE_COLUMNS)


def field_label(
    path: str,
    language: str,
    extraction: Mapping[str, object] | None = None,
) -> str:
    """A readable name for a validator path. Line items use the printed order."""
    if path.startswith("line_items["):
        match = _LINE_ITEM.fullmatch(path)
        if match is None:
            raise KeyError(path)
        english = _LINE_FIELDS[match.group(2)]
        order = _printed_order(extraction, int(match.group(1)))
        return f"Item {order} · {visitor_text(english, language)}"
    return visitor_text(_english_header(path), language)


def gallery_caption(caption: str, language: str) -> str:
    """Translate a rendition name and a planted-error mark. The id stays."""
    parts: list[str] = []
    for part in caption.split(" · "):
        if " " in part:
            parts.append(visitor_text(part, language))
        else:
            parts.append(part)
    return " · ".join(parts)


def language_code(choice: object) -> str:
    """``en`` for the English option. Anything else stays Portuguese."""
    if isinstance(choice, (list, tuple)) and choice:
        choice = choice[0]
    if choice == "English" or choice == "en":
        return "en"
    return "pt"


def _english_header(path: str) -> str:
    if path.startswith("header.unsupported_components."):
        name = path.removeprefix("header.unsupported_components.")
        try:
            return _COMPONENT_FIELDS[name]
        except KeyError as exc:
            raise KeyError(path) from exc
    if path.startswith("header."):
        name = path.removeprefix("header.")
        try:
            return _HEADER_FIELDS[name]
        except KeyError as exc:
            raise KeyError(path) from exc
    raise KeyError(path)


def _printed_order(extraction: Mapping[str, object] | None, index: int) -> int:
    if extraction is None:
        return index + 1
    items = extraction.get("line_items")
    if not isinstance(items, list) or not 0 <= index < len(items):
        return index + 1
    item = items[index]
    if not isinstance(item, dict):
        return index + 1
    printed = item.get("printed_order")
    if isinstance(printed, int) and not isinstance(printed, bool):
        return printed
    return index + 1
