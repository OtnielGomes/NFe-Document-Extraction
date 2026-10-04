"""Section agents: strict schemas, repository prompts, pinned snapshots."""

from pathlib import Path

from openai.lib._parsing import type_to_response_format_param
from test_identifier_rules import ACCESS_KEY, ISSUER_CNPJ, RECIPIENT_CPF

from nfe_document_extraction.agents.config import STRONG_VISION, load_approach_config
from nfe_document_extraction.agents.schema import (
    HeaderPayload,
    LineItemPayload,
    LineItemsSectionPayload,
    UnsupportedPayload,
)
from nfe_document_extraction.agents.sections import (
    RepairFieldNote,
    Section,
    SectionRequest,
    VisionSectionReader,
    interpret_header_output,
    interpret_line_items_output,
)
from nfe_document_extraction.evaluation import ExtractionFailure

_PDF = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic"
    / "synthetic-001"
    / "danfe.pdf"
)
_PROMPTS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "nfe_document_extraction"
    / "agents"
    / "prompts"
)


class _UsageDetails:
    cached_tokens = 0


class _Usage:
    input_tokens = 11
    output_tokens = 7
    input_tokens_details = _UsageDetails()


class _InvalidResponse:
    output_text = "not json"
    output = ()
    usage = _Usage()


class _Responses:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return _InvalidResponse()


class _Client:
    def __init__(self) -> None:
        self.responses = _Responses()


def test_header_schema_is_strict_and_has_no_line_items() -> None:
    body = _schema_body(HeaderPayload)

    assert body["additionalProperties"] is False
    assert "line_items" not in body["properties"]
    assert "access_key" in body["required"]
    assert "unsupported_components" in body["required"]
    assert body["$defs"]["UnsupportedPayload"]["additionalProperties"] is False


def test_line_items_schema_is_strict_and_has_no_header() -> None:
    body = _schema_body(LineItemsSectionPayload)

    assert body["additionalProperties"] is False
    assert body["required"] == ["line_items"]
    assert "header" not in body["properties"]
    assert body["$defs"]["LineItemPayload"]["additionalProperties"] is False


def test_header_agent_sends_its_repository_prompt_and_the_pinned_snapshot() -> None:
    client = _Client()
    reader = VisionSectionReader(client, load_approach_config(STRONG_VISION))  # type: ignore[arg-type]

    reading = reader.read(
        SectionRequest(section=Section.HEADER, document=_PDF.read_bytes())
    )

    assert reading.failure is ExtractionFailure.PARSE_FAILURE
    assert reading.input_tokens == 11
    assert reading.output_tokens == 7
    assert client.responses.kwargs["model"] == "gpt-4.1-2025-04-14"
    assert client.responses.kwargs["instructions"] == (
        _PROMPTS / "header.md"
    ).read_text(encoding="utf-8")
    text_format = client.responses.kwargs["text"]
    assert isinstance(text_format, dict)
    assert text_format["format"]["name"] == "HeaderPayload"
    assert text_format["format"]["strict"] is True
    content = client.responses.kwargs["input"]
    assert isinstance(content, list)
    parts = content[0]["content"]
    assert [part["type"] for part in parts] == [
        "input_text",
        "input_image",
        "input_image",
    ]
    assert parts[1]["detail"] == "high"


def test_line_items_repair_sends_the_failing_fields_and_the_same_message() -> None:
    client = _Client()
    reader = VisionSectionReader(client, load_approach_config(STRONG_VISION))  # type: ignore[arg-type]
    message = "Sum of item totals does not equal total products."

    reader.read(
        SectionRequest(
            section=Section.LINE_ITEMS,
            document=_PDF.read_bytes(),
            notes=(
                RepairFieldNote(
                    message=message,
                    fields=("line_items[0].item_total",),
                ),
            ),
        )
    )

    instructions = client.responses.kwargs["instructions"]
    assert isinstance(instructions, str)
    prompt = (_PROMPTS / "line_items.md").read_text(encoding="utf-8")
    assert instructions.startswith(prompt.rstrip())
    assert "line_items[0].item_total" in instructions
    assert message in instructions
    assert "Do not calculate a replacement" in instructions
    text_format = client.responses.kwargs["text"]
    assert isinstance(text_format, dict)
    assert text_format["format"]["name"] == "LineItemsSectionPayload"
    assert text_format["format"]["strict"] is True


def test_printed_header_normalizes_into_a_header() -> None:
    header, failure = interpret_header_output(
        HeaderPayload(
            access_key=ACCESS_KEY,
            state="sp",
            model="55",
            series="1",
            number="123",
            issue_date="03/10/2024",
            issuer_cnpj="00.000.000/0001-91",
            recipient_cnpj_cpf="529.982.247-25",
            total_products="10,00",
            freight="0,00",
            insurance="0",
            discount="0",
            ipi_total="0",
            other_expenses="0",
            icms_total="0",
            icms_st_total="1,80",
            invoice_total="11,80",
            unsupported_components=UnsupportedPayload(
                import_tax=False,
                returned_ipi=False,
                icms_relief=False,
                fcp_st=False,
            ),
        ),
        None,
    )

    assert failure is None
    assert header is not None
    assert header.state == "SP"
    assert header.issuer_cnpj == ISSUER_CNPJ
    assert header.recipient_cnpj_cpf == RECIPIENT_CPF
    assert header.total_products == 1_000
    assert header.invoice_total == 1_180


def test_printed_line_items_keep_quantity_precision() -> None:
    items, failure = interpret_line_items_output(
        LineItemsSectionPayload(
            line_items=[
                LineItemPayload(
                    printed_order=1,
                    product_code=" sku-1 ",
                    description=" Café ",
                    ncm="84833090",
                    cfop="5102",
                    quantity="1,2500",
                    unit_price="8,00",
                    item_total="10,00",
                )
            ]
        ),
        None,
    )

    assert failure is None
    assert items is not None
    assert items[0].product_code == "sku-1"
    assert items[0].description == "Café"
    assert str(items[0].quantity) == "1.2500"
    assert items[0].item_total == 1_000


def test_a_refusal_is_an_extraction_error() -> None:
    header, failure = interpret_header_output(None, "I will not read this document.")

    assert header is None
    assert failure is ExtractionFailure.REFUSAL


def _schema_body(model: type[object]) -> dict[str, object]:
    schema = type_to_response_format_param(model)
    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["strict"] is True
    body = schema["json_schema"]["schema"]
    assert isinstance(body, dict)
    return body
