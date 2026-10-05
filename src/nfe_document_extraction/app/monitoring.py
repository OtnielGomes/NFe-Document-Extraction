"""Demo traces: metadata, cost, and review feedback. Never the document."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol
from uuid import uuid4

from nfe_document_extraction.graph import GraphRun

_TRACE_KEYS = (
    "outcome",
    "cost_usd",
    "latency_seconds",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "failing_rules",
)


@dataclass(frozen=True)
class ExtractTrace:
    """One extraction as LangSmith may store it."""

    run_id: str
    metadata: Mapping[str, object]


class DemoMonitor(Protocol):
    """Where a finished extraction and a human correction are recorded."""

    def record_extraction(self, trace: ExtractTrace) -> None: ...

    def record_human_correction(
        self, run_id: str, edits: Mapping[str, object]
    ) -> None: ...


class NullMonitor:
    """Drops traces when LangSmith is not configured."""

    def record_extraction(self, trace: ExtractTrace) -> None:
        return None

    def record_human_correction(self, run_id: str, edits: Mapping[str, object]) -> None:
        return None


class TraceClient(Protocol):
    """The LangSmith calls the demo monitor makes."""

    def create_run(
        self,
        name: str,
        inputs: dict[str, Any],
        run_type: str,
        **kwargs: Any,
    ) -> None: ...

    def create_feedback(
        self,
        run_id: str | None = None,
        key: str = "unnamed",
        **kwargs: Any,
    ) -> Any: ...


class LangSmithMonitor:
    """Posts a metadata-only run and attaches human corrections as feedback."""

    def __init__(self, client: TraceClient) -> None:
        self._client = client

    def record_extraction(self, trace: ExtractTrace) -> None:
        now = datetime.now(UTC)
        self._client.create_run(
            "danfe-extract",
            {},
            "chain",
            id=trace.run_id,
            outputs={},
            start_time=now,
            end_time=now,
            extra={
                "metadata": {
                    key: trace.metadata[key]
                    for key in _TRACE_KEYS
                    if key in trace.metadata
                }
            },
        )

    def record_human_correction(self, run_id: str, edits: Mapping[str, object]) -> None:
        self._client.create_feedback(
            run_id=run_id,
            key="human_correction",
            value=dict(edits),
        )


def extraction_trace(run: GraphRun, cost: Decimal) -> ExtractTrace:
    """Metadata the demo is allowed to send. No field values."""
    failing: list[str] = []
    if run.report is not None:
        failing = [rule.rule_id for rule in run.report.rules if rule.failures]
    outcome = None if run.outcome is None else run.outcome.value
    metadata: dict[str, object] = {
        "outcome": outcome,
        "cost_usd": format(cost, "f"),
        "latency_seconds": run.latency_seconds,
        "input_tokens": run.input_tokens,
        "cached_input_tokens": run.cached_input_tokens,
        "output_tokens": run.output_tokens,
        "failing_rules": failing,
    }
    return ExtractTrace(run_id=str(uuid4()), metadata=metadata)


def silence_automatic_traces() -> None:
    """Stop SDK auto-tracing from uploading a DANFE or its extraction.

    The demo still posts its own metadata run. Hiding inputs and outputs
    covers any client created later in this process.
    """
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_TRACING",
    ):
        os.environ[name] = "false"
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
    from langsmith.utils import get_env_var

    get_env_var.cache_clear()
    import langsmith

    langsmith.configure(enabled=False)


def build_langsmith_client() -> Any:
    """A client that stores neither inputs nor outputs."""
    from langsmith import Client

    return Client(
        hide_inputs=True,
        hide_outputs=True,
        omit_traced_runtime_info=True,
    )


def default_monitor() -> DemoMonitor:
    """LangSmith when a key is set, otherwise a monitor that keeps nothing."""
    if not os.environ.get("LANGSMITH_API_KEY"):
        return NullMonitor()
    return LangSmithMonitor(build_langsmith_client())
