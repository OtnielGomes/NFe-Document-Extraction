"""Single-call baseline: strict schema, repository prompt, extraction errors."""

import base64
import io
from datetime import date
from decimal import Decimal
from pathlib import Path

from openai.lib._parsing import type_to_response_format_param
from pypdf import PdfReader
from test_identifier_rules import (
    ACCESS_KEY,
    ISSUER_CNPJ,
    RECIPIENT_CPF,
    _extraction,
)

from nfe_document_extraction.agents.baseline import (
    extract_baseline,
    interpret_baseline_output,
)
from nfe_document_extraction.agents.config import load_baseline_config
from nfe_document_extraction.agents.schema import (
    BaselinePayload,
    HeaderPayload,
    LineItemPayload,
    UnsupportedPayload,
)
from nfe_document_extraction.evaluation import ExtractionFailure
from nfe_document_extraction.evaluation.dataset import (
    DIGITAL_PDF,
    SIMULATED_PHOTO,
    load_synthetic_pdfs,
)
from nfe_document_extraction.evaluation.judge import judge_reading

_PDF = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic"
    / "synthetic-001"
    / "danfe.pdf"
)
_AS_OF = date(2026, 10, 3)


class _UsageDetails:
    cached_tokens = 4


class _Usage:
    input_tokens = 20
    output_tokens = 5
    input_tokens_details = _UsageDetails()


class _Refusal:
    type = "refusal"
    refusal = "I will not read this document."


class _Message:
    type = "message"
    content = (_Refusal(),)


class _Response:
    output_text = ""
    output = (_Message(),)
    usage = _Usage()


class _InvalidResponse:
    output_text = "not json"
    output = ()
    usage = _Usage()


class _Responses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.response


class _Client:
    def __init__(self, response: object) -> None:
        self.responses = _Responses(response)


def test_structured_output_schema_is_strict() -> None:
    schema = type_to_response_format_param(BaselinePayload)

    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["strict"] is True
    body = schema["json_schema"]["schema"]
    assert body["additionalProperties"] is False
    assert body["required"] == ["header", "line_items"]
    for name in ("HeaderPayload", "LineItemPayload", "UnsupportedPayload"):
        assert body["$defs"][name]["additionalProperties"] is False


def test_printed_payload_normalizes_into_an_extraction() -> None:
    payload = BaselinePayload(
        header=HeaderPayload(
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
        ],
    )

    extraction, failure = interpret_baseline_output(payload, None)

    assert failure is None
    assert extraction is not None
    assert extraction.header.state == "SP"
    assert extraction.header.issuer_cnpj == ISSUER_CNPJ
    assert extraction.header.recipient_cnpj_cpf == RECIPIENT_CPF
    assert extraction.header.issue_date == date(2024, 10, 3)
    assert extraction.header.total_products == 1_000
    assert extraction.header.icms_st_total == 180
    assert extraction.header.invoice_total == 1_180
    assert extraction.line_items[0].product_code == "sku-1"
    assert extraction.line_items[0].description == "Café"
    assert extraction.line_items[0].quantity == Decimal("1.2500")
    assert extraction.line_items[0].item_total == 1_000


def test_refusal_is_an_extraction_error() -> None:
    client = _Client(_Response())

    reading = extract_baseline(client, _PDF.read_bytes())  # type: ignore[arg-type]

    assert reading.extraction is None
    assert reading.failure is ExtractionFailure.REFUSAL
    assert reading.input_tokens == 20
    assert reading.cached_input_tokens == 4
    assert reading.output_tokens == 5
    instructions = client.responses.kwargs["instructions"]
    assert isinstance(instructions, str)
    assert "DANFE" in instructions
    assert client.responses.kwargs["model"] == load_baseline_config().model
    text_format = client.responses.kwargs["text"]
    assert isinstance(text_format, dict)
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


def test_baseline_sends_simulated_photo_pages() -> None:
    client = _Client(_InvalidResponse())
    pages = (b"\xff\xd8\xff\x00page-one",)

    reading = extract_baseline(client, b"not a pdf", pages=pages)  # type: ignore[arg-type]

    assert reading.failure is ExtractionFailure.PARSE_FAILURE
    content = client.responses.kwargs["input"]
    assert isinstance(content, list)
    image = content[0]["content"][1]
    encoded = base64.b64encode(pages[0]).decode("ascii")
    assert image["image_url"] == f"data:image/jpeg;base64,{encoded}"


def test_parse_failure_keeps_recorded_tokens() -> None:
    client = _Client(_InvalidResponse())

    reading = extract_baseline(client, _PDF.read_bytes())  # type: ignore[arg-type]

    assert reading.extraction is None
    assert reading.failure is ExtractionFailure.PARSE_FAILURE
    assert reading.input_tokens == 20
    assert reading.output_tokens == 5


def test_unparsed_output_is_a_parse_failure() -> None:
    extraction, failure = interpret_baseline_output(None, None)

    assert extraction is None
    assert failure is ExtractionFailure.PARSE_FAILURE


def test_planted_rule_failure_is_detection_not_a_clean_validator_catch() -> None:
    invoice = _extraction()
    broken = _extraction(invoice_total=1)

    planted = judge_reading(
        approach="single-call baseline",
        rendition="digital PDF",
        origin="synthetic",
        ground_truth=invoice,
        extraction=broken,
        extraction_failure=None,
        planted_rule="totals",
        cost_usd=Decimal("0.01"),
        latency_seconds=1.5,
        today=_AS_OF,
    )
    clean = judge_reading(
        approach="single-call baseline",
        rendition="digital PDF",
        origin="synthetic",
        ground_truth=invoice,
        extraction=broken,
        extraction_failure=None,
        planted_rule=None,
        cost_usd=Decimal("0.01"),
        latency_seconds=1.5,
        today=_AS_OF,
    )

    assert planted.planted_error
    assert planted.planted_error_detected
    assert planted.validator_caught_error is False
    assert clean.planted_error is False
    assert clean.validator_caught_error
    assert clean.score is not None
    assert clean.score.perfect is False


def test_unsupported_component_is_not_counted_as_a_validator_error() -> None:
    invoice = _extraction()
    reviewed = _extraction(
        unsupported_components={
            "import_tax": True,
            "returned_ipi": False,
            "icms_relief": False,
            "fcp_st": False,
        }
    )

    judged = judge_reading(
        approach="single-call baseline",
        rendition="digital PDF",
        origin="synthetic",
        ground_truth=invoice,
        extraction=reviewed,
        extraction_failure=None,
        planted_rule=None,
        cost_usd=Decimal("0.01"),
        latency_seconds=1,
        today=_AS_OF,
    )

    assert judged.validator_caught_error is False


def test_failed_extraction_of_a_planted_document_is_not_detected() -> None:
    judged = judge_reading(
        approach="single-call baseline",
        rendition="digital PDF",
        origin="synthetic",
        ground_truth=_extraction(),
        extraction=None,
        extraction_failure=ExtractionFailure.PARSE_FAILURE,
        planted_rule="totals",
        cost_usd=Decimal(0),
        latency_seconds=0.2,
        today=_AS_OF,
    )

    assert judged.score is None
    assert judged.planted_error_detected is False
    assert judged.extraction_failure is ExtractionFailure.PARSE_FAILURE


def test_synthetic_loader_reads_ground_truth_and_planted_rules() -> None:
    documents = load_synthetic_pdfs(
        Path(__file__).resolve().parents[1] / "data" / "synthetic"
    )
    pdfs = [document for document in documents if document.input_type == DIGITAL_PDF]
    photos = [
        document for document in documents if document.input_type == SIMULATED_PHOTO
    ]

    assert len(documents) == 200
    assert [document.nfe_id for document in pdfs] == [
        f"synthetic-{index:03d}" for index in range(1, 101)
    ]
    assert [document.nfe_id for document in photos] == [
        document.nfe_id for document in pdfs
    ]
    assert documents[0].origin == "synthetic"
    assert documents[0].ground_truth.header.model == "55"
    assert documents[0].pdf.startswith(b"%PDF")
    assert all(
        pdf.ground_truth == photo.ground_truth
        for pdf, photo in zip(pdfs, photos, strict=True)
    )
    assert all(
        pdf.planted_rule == photo.planted_rule
        for pdf, photo in zip(pdfs, photos, strict=True)
    )
    assert all(photo.pages and photo.pdf == b"" for photo in photos)
    assert sum(document.planted_rule is not None for document in pdfs) == 15
    multipage = next(
        document
        for document in pdfs
        if len(PdfReader(io.BytesIO(document.pdf)).pages) > 1
    )
    photo = next(item for item in photos if item.nfe_id == multipage.nfe_id)
    assert len(photo.pages) == len(PdfReader(io.BytesIO(multipage.pdf)).pages)
