"""Public JSON API: ``POST /extract`` and ``POST /review/{thread_id}``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from nfe_document_extraction.agents.config import CHEAP_VISION, load_approach_config
from nfe_document_extraction.agents.sections import SectionReader
from nfe_document_extraction.app.errors import RequestRejected
from nfe_document_extraction.app.gallery import GalleryItem
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
    gallery: Sequence[GalleryItem] | None = None,
) -> FastAPI:
    """FastAPI app over the document graph. Tracing of model calls stays off.

    When ``gallery`` is set, the Gradio page is mounted at ``/`` after the
    JSON routes, and both share this process's extraction service.
    """
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

    if gallery is not None:
        from nfe_document_extraction.app.ui import DemoActions, mount_demo

        actions = DemoActions(service, gallery)
        app.state.demo_actions = actions
        mount_demo(app, actions)

    return app


def build_demo_app() -> FastAPI:
    """Cheap-vision section agents behind the public limits."""
    from pathlib import Path

    from nfe_document_extraction.agents.sections import VisionSectionReader
    from nfe_document_extraction.app.gallery import load_gallery
    from nfe_document_extraction.env import load_env_file

    load_env_file(Path(".env"))
    reader = VisionSectionReader(_openai_client(), load_approach_config(CHEAP_VISION))
    today = _utc_today()
    return create_app(reader, gallery=load_gallery(Path("data/synthetic"), today=today))


def client_ip(request: Request) -> str:
    """Visitor address for the per-address upload limit.

    Hugging Face Spaces proxies the request and sets ``X-Forwarded-For``.
    The left-most entry is the visitor when the header is a single address
    or a client-then-proxy chain. A direct call has no header, so the
    socket peer is used.
    """
    client = request.client
    peer = client.host if client is not None and client.host else None
    return visitor_ip(request.headers.get("x-forwarded-for"), peer)


def visitor_ip(forwarded_for: str | None, peer: str | None) -> str:
    """Left-most forwarded address, otherwise the socket peer."""
    forwarded = forwarded_for or ""
    if forwarded.strip():
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    if peer:
        return peer
    return "unknown"


def _openai_client() -> object:
    """The demo client. A missing key still lets the cached gallery start."""
    import os

    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return OpenAI(api_key=key)
    return OpenAI(api_key="missing")


def _utc_today() -> date:
    return datetime.now(UTC).date()
