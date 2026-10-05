"""Public JSON API: ``POST /extract`` and ``POST /review/{thread_id}``."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from nfe_document_extraction.agents.config import CHEAP_VISION, load_approach_config
from nfe_document_extraction.agents.sections import SectionReader
from nfe_document_extraction.app.errors import RequestRejected
from nfe_document_extraction.app.limits import CostLimits
from nfe_document_extraction.app.monitoring import (
    DemoMonitor,
    default_monitor,
    silence_automatic_traces,
)
from nfe_document_extraction.app.service import ExtractionService
from nfe_document_extraction.evaluation.pricing import TokenPrice


class ReviewBody(BaseModel):
    """Edits to flagged fields, keyed by field path."""

    model_config = ConfigDict(extra="forbid")

    edits: dict[str, Any]


_DOCUMENT = File(...)


def create_app(
    reader: SectionReader,
    *,
    monitor: DemoMonitor | None = None,
    limits: CostLimits | None = None,
    clock: Callable[[], date] | None = None,
    price: TokenPrice | None = None,
) -> FastAPI:
    """FastAPI app over the document graph. Tracing of model calls stays off."""
    silence_automatic_traces()
    service = ExtractionService(
        reader,
        monitor=default_monitor() if monitor is None else monitor,
        limits=CostLimits() if limits is None else limits,
        clock=_utc_today if clock is None else clock,
        price=load_approach_config(CHEAP_VISION) if price is None else price,
    )
    app = FastAPI(title="NFe Document Extraction")

    @app.exception_handler(RequestRejected)
    def _rejected(_request: Request, exc: RequestRejected) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.message}
        )

    @app.post("/extract")
    def extract(
        request: Request,
        document: UploadFile = _DOCUMENT,
    ) -> JSONResponse:
        data = document.file.read(service.limits.max_bytes + 1)
        body = service.extract(
            data,
            filename=document.filename,
            content_type=document.content_type,
            client_ip=client_ip(request),
        )
        return JSONResponse(body)

    @app.post("/review/{thread_id}")
    def review(thread_id: str, body: ReviewBody) -> JSONResponse:
        return JSONResponse(service.submit_review(thread_id, body.edits))

    return app


def build_demo_app() -> FastAPI:
    """Cheap-vision section agents behind the public limits."""
    from pathlib import Path

    from openai import OpenAI

    from nfe_document_extraction.agents.sections import VisionSectionReader
    from nfe_document_extraction.env import load_env_file

    load_env_file(Path(".env"))
    reader = VisionSectionReader(OpenAI(), load_approach_config(CHEAP_VISION))
    return create_app(reader)


def client_ip(request: Request) -> str:
    """Visitor address for the per-address upload limit.

    Hugging Face Spaces proxies the request and sets ``X-Forwarded-For``.
    The left-most entry is the visitor when the header is a single address
    or a client-then-proxy chain. A direct call has no header, so the
    socket peer is used.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = request.client
    if client is not None and client.host:
        return client.host
    return "unknown"


def _utc_today() -> date:
    return datetime.now(UTC).date()
