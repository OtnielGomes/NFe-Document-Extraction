"""API seam: FastAPI test client over the graph with a scripted model."""

from __future__ import annotations

import io
import json
import os
import threading
from collections.abc import Callable, Mapping
from datetime import date
from decimal import Decimal
from typing import Any

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from test_identifier_rules import ISSUER_CNPJ, _extraction

from nfe_document_extraction.agents.sections import (
    Section,
    SectionReading,
    SectionRequest,
)
from nfe_document_extraction.app.api import create_app
from nfe_document_extraction.app.limits import CostLimits
from nfe_document_extraction.app.monitoring import (
    ExtractTrace,
    LangSmithMonitor,
    build_langsmith_client,
)
from nfe_document_extraction.domain import Extraction

_TODAY = date(2026, 10, 5)
_IP = "198.51.100.10"
# 1_250_000 output tokens * $1.60 / 1_000_000 = $2.00 in agents/models.json.
_OVER_CAP_OUTPUT_TOKENS = 1_250_000
_JPEG = b"\xff\xd8\xff\xd9"
_PNG = b"\x89PNG\r\n\x1a\n"
_HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00"


class ScriptedReader:
    """Returns one fixed reading and records every request."""

    def __init__(self, invoice: Extraction, *, output_tokens: int = 0) -> None:
        self._invoice = invoice
        self._output_tokens = output_tokens
        self.requests: list[SectionRequest] = []
        self._lock = threading.Lock()

    def read(self, request: SectionRequest) -> SectionReading:
        with self._lock:
            self.requests.append(request)
        if request.section is Section.HEADER:
            return SectionReading(
                section=Section.HEADER,
                header=self._invoice.header,
                output_tokens=self._output_tokens,
            )
        return SectionReading(
            section=Section.LINE_ITEMS,
            line_items=self._invoice.line_items,
        )


class RecordingMonitor:
    """Captures the traces the API decided to publish."""

    def __init__(self) -> None:
        self.extractions: list[ExtractTrace] = []
        self.corrections: list[tuple[str, dict[str, object]]] = []

    def record_extraction(self, trace: ExtractTrace) -> None:
        self.extractions.append(trace)

    def record_human_correction(self, run_id: str, edits: Mapping[str, object]) -> None:
        self.corrections.append((run_id, dict(edits)))


class RecordingLangSmith:
    """Stands in for the LangSmith client at the network boundary."""

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.feedback: list[dict[str, Any]] = []

    def create_run(
        self,
        name: str,
        inputs: dict[str, Any],
        run_type: str,
        **kwargs: Any,
    ) -> None:
        self.runs.append(
            {"name": name, "inputs": inputs, "run_type": run_type, **kwargs}
        )

    def create_feedback(
        self,
        run_id: str | None = None,
        key: str = "unnamed",
        **kwargs: Any,
    ) -> None:
        self.feedback.append({"run_id": run_id, "key": key, **kwargs})


def test_extract_returns_the_reading_field_statuses_and_accepted_outcome() -> None:
    invoice = _extraction()
    reader = ScriptedReader(invoice)

    with _client(reader) as client:
        response = _upload(client, _pdf(1), "invoice.pdf", "application/pdf")

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "accepted"
    assert body["thread_id"] is None
    assert body["review_messages"] == []
    assert body["extraction"]["header"]["issuer_cnpj"] == ISSUER_CNPJ
    assert body["extraction"]["header"]["issue_date"] == "2024-10-03"
    assert body["extraction"]["line_items"][0]["description"] == "Plain bearing"
    assert body["extraction"]["line_items"][0]["quantity"] == "1.2500"
    assert _field(body, "header.issuer_cnpj") == {
        "field": "header.issuer_cnpj",
        "status": "verified",
        "messages": [],
    }
    assert _field(body, "line_items[0].description")["status"] == "unverifiable"
    assert len(reader.requests) == 2
    assert all(request.document.startswith(b"%PDF") for request in reader.requests)
    assert all(request.pages == () for request in reader.requests)


def test_a_jpg_or_png_is_read_as_a_page_image() -> None:
    reader = ScriptedReader(_extraction())

    with _client(reader) as client:
        jpeg = _upload(client, _JPEG, "photo.jpg", "image/jpeg")
        png = _upload(client, _PNG, "photo.png", "image/png", ip="198.51.100.11")

    assert jpeg.status_code == 200
    assert jpeg.json()["outcome"] == "accepted"
    assert png.status_code == 200
    assert [request.pages for request in reader.requests[:2]] == [(_JPEG,), (_JPEG,)]
    assert [request.document for request in reader.requests[:2]] == [b"", b""]
    assert [request.pages for request in reader.requests[2:]] == [(_PNG,), (_PNG,)]


def test_an_oversized_upload_is_rejected_before_the_model() -> None:
    reader = ScriptedReader(_extraction())
    limits = CostLimits(uploads_per_ip_per_day=1)

    with _client(reader, limits=limits) as client:
        oversized = _upload(
            client,
            b"x" * (10 * 1024 * 1024 + 1),
            "big.pdf",
            "application/pdf",
        )
        accepted = _upload(client, _pdf(1), "invoice.pdf", "application/pdf")
        blocked = _upload(client, _pdf(1), "another.pdf", "application/pdf")

    assert oversized.status_code == 413
    assert oversized.json()["detail"] == "This file is larger than 10 MB."
    assert accepted.status_code == 200
    assert blocked.status_code == 429
    assert len(reader.requests) == 2


def test_a_pdf_over_three_pages_is_rejected_and_three_pages_are_accepted() -> None:
    reader = ScriptedReader(_extraction())

    with _client(reader) as client:
        rejected = _upload(client, _pdf(4), "long.pdf", "application/pdf")
        accepted = _upload(client, _pdf(3), "short.pdf", "application/pdf")

    assert rejected.status_code == 400
    assert rejected.json()["detail"] == (
        "This document has 4 pages, which is over the limit of 3."
    )
    assert accepted.status_code == 200
    assert len(reader.requests) == 2


def test_heic_is_rejected_with_guidance_to_export_jpg() -> None:
    reader = ScriptedReader(_extraction())

    with _client(reader) as client:
        response = _upload(client, _HEIC, "photo.heic", "image/heic")

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "HEIC photos are not accepted. Export the photo as JPG and upload that file."
    )
    assert reader.requests == []


def test_a_file_that_is_not_pdf_jpg_or_png_is_rejected() -> None:
    reader = ScriptedReader(_extraction())

    with _client(reader) as client:
        response = _upload(client, b"not an invoice", "notes.txt", "text/plain")

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload a PDF, JPG, or PNG."
    assert reader.requests == []


def test_an_unreadable_pdf_is_rejected_before_the_model() -> None:
    reader = ScriptedReader(_extraction())

    with _client(reader) as client:
        response = _upload(
            client, b"%PDF-1.4 not a pdf", "broken.pdf", "application/pdf"
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "This PDF could not be read."
    assert reader.requests == []


def test_review_revalidates_flagged_edits_without_another_model_call() -> None:
    reader = ScriptedReader(_extraction(issuer_cnpj="00000000000181"))
    limits = CostLimits(uploads_per_ip_per_day=1, daily_spend_cap_usd=Decimal(100))

    with _client(reader, limits=limits) as client:
        extracted = _upload(client, _pdf(1), "invoice.pdf", "application/pdf")
        assert extracted.status_code == 200
        body = extracted.json()
        assert body["outcome"] == "human_review"
        thread_id = body["thread_id"]
        assert isinstance(thread_id, str) and thread_id
        assert _field(body, "header.issuer_cnpj")["status"] == "flagged"
        assert (
            "Issuer CNPJ check digit does not match."
            in _field(body, "header.issuer_cnpj")["messages"]
        )
        calls = len(reader.requests)

        reviewed = client.post(
            f"/review/{thread_id}",
            json={"edits": {"header.issuer_cnpj": ISSUER_CNPJ}},
        )
        assert reviewed.status_code == 200
        done = reviewed.json()
        assert done["outcome"] == "accepted"
        assert done["thread_id"] is None
        assert done["extraction"]["header"]["issuer_cnpj"] == ISSUER_CNPJ
        assert _field(done, "header.issuer_cnpj")["status"] == "verified"
        assert _field(done, "header.issuer_cnpj")["messages"] == []
        assert len(reader.requests) == calls

        finished = client.post(f"/review/{thread_id}", json={"edits": {}})
        another = _upload(client, _pdf(1), "invoice.pdf", "application/pdf")

    assert finished.status_code == 404
    assert another.status_code == 429
    assert another.json()["detail"] == (
        "This address has already uploaded 1 document today. Try again tomorrow."
    )


def test_an_edit_to_a_field_that_is_not_flagged_is_rejected() -> None:
    reader = ScriptedReader(_extraction(issuer_cnpj="00000000000181"))

    with _client(reader) as client:
        extracted = _upload(client, _pdf(1), "invoice.pdf", "application/pdf")
        thread_id = extracted.json()["thread_id"]
        rejected = client.post(
            f"/review/{thread_id}",
            json={"edits": {"header.number": "999"}},
        )
        accepted = client.post(
            f"/review/{thread_id}",
            json={"edits": {"header.issuer_cnpj": ISSUER_CNPJ}},
        )

    assert rejected.status_code == 400
    assert rejected.json()["detail"] == (
        "Only flagged fields can be edited: header.number."
    )
    assert accepted.status_code == 200
    assert accepted.json()["extraction"]["header"]["number"] == "123"
    assert accepted.json()["outcome"] == "accepted"


def test_an_unusable_edit_keeps_the_review_open() -> None:
    reader = ScriptedReader(_extraction(issuer_cnpj="00000000000181"))

    with _client(reader) as client:
        thread_id = _upload(client, _pdf(1), "invoice.pdf", "application/pdf").json()[
            "thread_id"
        ]
        rejected = client.post(
            f"/review/{thread_id}",
            json={"edits": {"header.issuer_cnpj": {"nope": True}}},
        )
        accepted = client.post(
            f"/review/{thread_id}",
            json={"edits": {"header.issuer_cnpj": ISSUER_CNPJ}},
        )

    assert rejected.status_code == 422
    assert rejected.json()["detail"] == "An edit is not a valid value for that field."
    assert accepted.status_code == 200
    assert accepted.json()["outcome"] == "accepted"


def test_an_unknown_review_thread_is_not_found() -> None:
    reader = ScriptedReader(_extraction())

    with _client(reader) as client:
        response = client.post("/review/missing-thread", json={"edits": {}})

    assert response.status_code == 404
    assert (
        response.json()["detail"] == "No document is waiting for review on this thread."
    )
    assert reader.requests == []


def test_five_uploads_per_address_then_the_sixth_is_refused() -> None:
    reader = ScriptedReader(_extraction())

    with _client(reader) as client:
        for _ in range(5):
            assert (
                _upload(client, _pdf(1), "invoice.pdf", "application/pdf").status_code
                == 200
            )
        calls = len(reader.requests)
        blocked = _upload(client, _pdf(1), "invoice.pdf", "application/pdf")
        after_block = len(reader.requests)
        other = _upload(
            client, _pdf(1), "invoice.pdf", "application/pdf", ip="198.51.100.20"
        )

    assert blocked.status_code == 429
    assert blocked.json()["detail"] == (
        "This address has already uploaded 5 documents today. Try again tomorrow."
    )
    assert after_block == calls
    assert len(reader.requests) == calls + 2
    assert other.status_code == 200


def test_the_visitor_address_is_the_first_forwarded_for_entry() -> None:
    reader = ScriptedReader(_extraction())
    limits = CostLimits(uploads_per_ip_per_day=1)

    with _client(reader, limits=limits) as client:
        first = _upload(
            client,
            _pdf(1),
            "invoice.pdf",
            "application/pdf",
            ip="203.0.113.8, 10.0.0.4",
        )
        same_visitor = _upload(
            client,
            _pdf(1),
            "invoice.pdf",
            "application/pdf",
            ip="203.0.113.8, 10.9.9.9",
        )
        other_visitor = _upload(
            client,
            _pdf(1),
            "invoice.pdf",
            "application/pdf",
            ip="203.0.113.9, 10.0.0.4",
        )

    assert first.status_code == 200
    assert same_visitor.status_code == 429
    assert other_visitor.status_code == 200


def test_a_direct_call_is_limited_by_the_socket_peer() -> None:
    reader = ScriptedReader(_extraction())
    limits = CostLimits(uploads_per_ip_per_day=1)

    with _client(reader, limits=limits) as client:
        first = client.post(
            "/extract",
            files={"document": ("invoice.pdf", _pdf(1), "application/pdf")},
        )
        second = client.post(
            "/extract",
            files={"document": ("invoice.pdf", _pdf(1), "application/pdf")},
        )
        forwarded = _upload(
            client, _pdf(1), "invoice.pdf", "application/pdf", ip="203.0.113.50"
        )

    assert first.status_code == 200
    assert second.status_code == 429
    assert forwarded.status_code == 200


def test_the_global_daily_upload_cap_is_shared_across_addresses() -> None:
    reader = ScriptedReader(_extraction())
    limits = CostLimits(global_uploads_per_day=2)

    with _client(reader, limits=limits) as client:
        assert (
            _upload(
                client, _pdf(1), "a.pdf", "application/pdf", ip="198.51.100.1"
            ).status_code
            == 200
        )
        assert (
            _upload(
                client, _pdf(1), "b.pdf", "application/pdf", ip="198.51.100.2"
            ).status_code
            == 200
        )
        calls = len(reader.requests)
        blocked = _upload(
            client, _pdf(1), "c.pdf", "application/pdf", ip="198.51.100.3"
        )

    assert blocked.status_code == 429
    assert blocked.json()["detail"] == (
        "The demo has reached its daily upload limit. Try again tomorrow."
    )
    assert len(reader.requests) == calls


def test_estimated_spend_from_the_pricing_table_stops_the_next_upload() -> None:
    reader = ScriptedReader(_extraction(), output_tokens=_OVER_CAP_OUTPUT_TOKENS)
    monitor = RecordingMonitor()

    with _client(reader, monitor=monitor) as client:
        first = _upload(client, _pdf(1), "invoice.pdf", "application/pdf")
        calls = len(reader.requests)
        second = _upload(client, _pdf(1), "invoice.pdf", "application/pdf")

    assert first.status_code == 200
    assert Decimal(str(monitor.extractions[0].metadata["cost_usd"])) == Decimal(2)
    assert second.status_code == 429
    assert second.json()["detail"] == (
        "The demo has reached its daily spend limit. Try again tomorrow."
    )
    assert len(reader.requests) == calls


def test_daily_counters_reset_on_the_next_utc_day() -> None:
    reader = ScriptedReader(_extraction())
    clock = _Clock(_TODAY)
    limits = CostLimits(uploads_per_ip_per_day=1)

    with _client(reader, limits=limits, clock=clock) as client:
        assert (
            _upload(client, _pdf(1), "invoice.pdf", "application/pdf").status_code
            == 200
        )
        assert (
            _upload(client, _pdf(1), "invoice.pdf", "application/pdf").status_code
            == 429
        )
        clock.day = date(2026, 10, 6)
        reset = _upload(client, _pdf(1), "invoice.pdf", "application/pdf")

    assert reset.status_code == 200


def test_traces_hide_the_document_and_human_corrections_are_feedback() -> None:
    reader = ScriptedReader(_extraction(issuer_cnpj="00000000000181"))
    monitor = RecordingMonitor()

    with _client(reader, monitor=monitor) as client:
        extracted = _upload(client, _pdf(1), "invoice.pdf", "application/pdf")
        thread_id = extracted.json()["thread_id"]
        client.post(
            f"/review/{thread_id}",
            json={"edits": {"header.issuer_cnpj": ISSUER_CNPJ}},
        )

    assert len(monitor.extractions) == 1
    metadata = dict(monitor.extractions[0].metadata)
    assert set(metadata) == {
        "outcome",
        "cost_usd",
        "latency_seconds",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "failing_rules",
    }
    assert metadata["outcome"] == "human_review"
    assert set(metadata["failing_rules"]) == {
        "cnpj_cpf_check_digit",
        "access_key_cross_consistency",
    }
    published = json.dumps(metadata)
    assert "Plain bearing" not in published
    assert ISSUER_CNPJ not in published
    assert "%PDF" not in published
    assert monitor.corrections == [
        (monitor.extractions[0].run_id, {"header.issuer_cnpj": ISSUER_CNPJ})
    ]


def test_langsmith_monitor_sends_empty_inputs_and_outputs() -> None:
    client = RecordingLangSmith()
    monitor = LangSmithMonitor(client)
    trace = ExtractTrace(
        run_id="11111111-1111-1111-1111-111111111111",
        metadata={
            "outcome": "accepted",
            "cost_usd": "0.25",
            "latency_seconds": 1.5,
            "input_tokens": 10,
            "cached_input_tokens": 0,
            "output_tokens": 4,
            "failing_rules": [],
            "document": "secret invoice",
        },
    )

    monitor.record_extraction(trace)
    monitor.record_human_correction(trace.run_id, {"header.series": "1"})

    assert client.runs[0]["inputs"] == {}
    assert client.runs[0]["outputs"] == {}
    assert client.runs[0]["extra"]["metadata"] == {
        "outcome": "accepted",
        "cost_usd": "0.25",
        "latency_seconds": 1.5,
        "input_tokens": 10,
        "cached_input_tokens": 0,
        "output_tokens": 4,
        "failing_rules": [],
    }
    assert "document" not in client.runs[0]["extra"]["metadata"]
    assert client.feedback == [
        {
            "run_id": trace.run_id,
            "key": "human_correction",
            "value": {"header.series": "1"},
        }
    ]


def test_the_langsmith_client_hides_inputs_and_outputs(monkeypatch: Any) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    client = build_langsmith_client()

    assert client._hide_inputs is True
    assert client._hide_outputs is True
    assert client._omit_traced_runtime_info is True


def test_creating_the_api_turns_automatic_tracing_off(monkeypatch: Any) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    from langsmith.utils import tracing_is_enabled

    create_app(ScriptedReader(_extraction()))

    assert tracing_is_enabled() is False
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"
    assert os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"


def test_the_public_demo_caps() -> None:
    limits = CostLimits()

    assert limits.max_bytes == 10 * 1024 * 1024
    assert limits.max_pages == 3
    assert limits.uploads_per_ip_per_day == 5
    assert limits.global_uploads_per_day == 50
    assert limits.daily_spend_cap_usd == Decimal("0.50")


class _Clock:
    def __init__(self, day: date) -> None:
        self.day = day

    def __call__(self) -> date:
        return self.day


def _client(
    reader: ScriptedReader,
    *,
    limits: CostLimits | None = None,
    clock: Callable[[], date] | None = None,
    monitor: RecordingMonitor | None = None,
) -> TestClient:
    return TestClient(
        create_app(
            reader,
            limits=limits,
            clock=clock or (lambda: _TODAY),
            monitor=monitor,
        )
    )


def _upload(
    client: TestClient,
    data: bytes,
    name: str,
    content_type: str,
    *,
    ip: str = _IP,
) -> Any:
    return client.post(
        "/extract",
        files={"document": (name, data, content_type)},
        headers={"X-Forwarded-For": ip},
    )


def _pdf(pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _field(body: dict[str, Any], name: str) -> dict[str, Any]:
    for field in body["fields"]:
        if field["field"] == name:
            return field
    raise AssertionError(name)
