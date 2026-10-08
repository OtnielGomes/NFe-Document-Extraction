"""Extract a DANFE and resume human review, with the cost brakes in front."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from nfe_document_extraction.agents.sections import SectionReader
from nfe_document_extraction.app.errors import RequestRejected
from nfe_document_extraction.app.limits import CostLimits, UsageLedger
from nfe_document_extraction.app.monitoring import (
    DemoMonitor,
    extraction_trace,
)
from nfe_document_extraction.app.uploads import prepare_upload
from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.evaluation.pricing import TokenPrice, token_cost_usd
from nfe_document_extraction.graph import (
    GraphRun,
    NonFlaggedFieldEdit,
    apply_extraction_edits,
    compile_document_graph,
    run_document,
    submit_human_review,
)
from nfe_document_extraction.validation.report import DocumentOutcome, ValidationReport

UNREADABLE_DOCUMENT = "The document could not be read."
NOT_A_DANFE = "This document does not look like a DANFE."
NO_REVIEW = "No document is waiting for review on this thread."
INVALID_EDIT = "An edit is not a valid value for that field."


class ExtractionService:
    """The service the HTTP API calls. One checkpointer per process."""

    def __init__(
        self,
        reader: SectionReader,
        *,
        monitor: DemoMonitor,
        limits: CostLimits,
        clock: Callable[[], date],
        price: TokenPrice,
    ) -> None:
        self.limits = limits
        self._reader = reader
        self._monitor = monitor
        self._clock = clock
        self._price = price
        self._checkpointer = InMemorySaver()
        self._ledger = UsageLedger(limits, clock)
        self._trace_ids: dict[str, str] = {}

    def extract(
        self,
        data: bytes,
        *,
        filename: str | None,
        content_type: str | None,
        client_ip: str,
    ) -> dict[str, object]:
        """Read one upload. File and quota failures happen before the model."""
        prepared = prepare_upload(
            data,
            filename=filename,
            content_type=content_type,
            limits=self.limits,
        )
        with self._ledger.admission(client_ip) as admission:
            run = run_document(
                self._reader,
                prepared.pdf,
                pages=prepared.pages,
                today=self._clock(),
                checkpointer=self._checkpointer,
            )
            cost = token_cost_usd(
                run.input_tokens,
                run.cached_input_tokens,
                run.output_tokens,
                self._price,
                cache_write_tokens=run.cache_write_tokens,
            )
            admission.add_spend(cost)
        trace = extraction_trace(run, cost)
        self._monitor.record_extraction(trace)
        if run.thread_id is not None and run.awaiting_human_review:
            self._trace_ids[run.thread_id] = trace.run_id
        if run.rejected:
            raise RequestRejected(NOT_A_DANFE, 422)
        if run.extraction is None or run.report is None or run.outcome is None:
            raise RequestRejected(UNREADABLE_DOCUMENT, 422)
        return _document_body(run)

    def submit_review(
        self, thread_id: str, edits: Mapping[str, object]
    ) -> dict[str, object]:
        """Revalidate edits to flagged fields. Does not call the model."""
        snapshot = self._snapshot(thread_id)
        if not snapshot.interrupts:
            raise RequestRejected(NO_REVIEW, 404)
        extraction = snapshot.values.get("extraction")
        if not isinstance(extraction, Extraction):
            raise RequestRejected(NO_REVIEW, 404)
        flagged = _flagged_paths(snapshot)
        applicable = {path: value for path, value in edits.items() if path in flagged}
        # A value the schema rejects must not reach the checkpoint: the graph
        # would keep that edit and refuse later resumes. The graph applies
        # the same edits again after this check.
        try:
            apply_extraction_edits(extraction, applicable)
        except (ValidationError, ValueError) as exc:
            raise RequestRejected(INVALID_EDIT, 422) from exc
        try:
            run = submit_human_review(
                self._reader,
                thread_id,
                edits,
                checkpointer=self._checkpointer,
                today=self._clock(),
            )
        except NonFlaggedFieldEdit as exc:
            raise RequestRejected(str(exc), 400) from exc
        trace_id = self._trace_ids.get(thread_id)
        if trace_id is not None:
            self._monitor.record_human_correction(trace_id, dict(edits))
        if run.extraction is None or run.report is None or run.outcome is None:
            raise RequestRejected(UNREADABLE_DOCUMENT, 422)
        return _document_body(run)

    def _snapshot(self, thread_id: str) -> Any:
        compiled = compile_document_graph(
            self._reader,
            today=self._clock(),
            checkpointer=self._checkpointer,
        )
        return compiled.get_state({"configurable": {"thread_id": thread_id}})


def _document_body(run: GraphRun) -> dict[str, object]:
    extraction = run.extraction
    report = run.report
    outcome = run.outcome
    if extraction is None or report is None or outcome is None:
        raise RequestRejected(UNREADABLE_DOCUMENT, 422)
    return document_payload(
        extraction,
        report,
        outcome=outcome,
        thread_id=run.thread_id if run.awaiting_human_review else None,
    )


def document_payload(
    extraction: Extraction,
    report: ValidationReport,
    *,
    outcome: DocumentOutcome,
    thread_id: str | None,
) -> dict[str, object]:
    """The document JSON the API and the demo UI both return."""
    return {
        "outcome": outcome.value,
        "thread_id": thread_id,
        "extraction": extraction.model_dump(mode="json"),
        "fields": [
            {
                "field": field.field,
                "status": field.status.value,
                "messages": [failure.message for failure in field.failures],
            }
            for field in report.fields
        ],
        "review_messages": list(report.review_messages),
    }


def _flagged_paths(snapshot: Any) -> set[str]:
    paths: set[str] = set()
    for interrupt in snapshot.interrupts:
        value = getattr(interrupt, "value", None)
        if not isinstance(value, dict):
            continue
        fields = value.get("flagged_fields") or []
        paths.update(str(field) for field in fields)
    return paths
