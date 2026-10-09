"""Gradio demo: cached gallery, upload, and review of flagged fields."""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from PIL import Image

from nfe_document_extraction.app.errors import RequestRejected
from nfe_document_extraction.app.excel import excel_download_path
from nfe_document_extraction.app.gallery import GalleryItem, load_gallery
from nfe_document_extraction.app.service import (
    NO_REVIEW,
    ExtractionService,
    FeedbackNotRecorded,
)
from nfe_document_extraction.app.uploads import HEIC_NOT_ACCEPTED, UNSUPPORTED_TYPE

__all__ = [
    "DemoActions",
    "Refusal",
    "document_panel",
    "editable_field_paths",
    "field_rows",
    "gradio_client_ip",
    "load_gallery",
    "present_gallery_selection",
    "present_review",
    "present_upload",
    "present_visitor_feedback",
    "render_result",
    "review_inputs",
    "visitor_feedback_prompt",
]


@dataclass(frozen=True)
class Refusal:
    """A limit or upload rule, in the words the visitor should read."""

    message: str


_STATUS_LABEL = {
    "verified": "Verified",
    "unverifiable": "Unverifiable",
    "flagged": "Flagged",
}
_LINE_ITEM = re.compile(r"line_items\[(\d+)\]\.([A-Za-z0-9_]+)\Z")
_HEADER_MONEY = frozenset(
    {
        "total_products",
        "freight",
        "insurance",
        "discount",
        "ipi_total",
        "other_expenses",
        "icms_total",
        "icms_st_total",
        "invoice_total",
    }
)
_CHOOSE_FILE = "Choose a PDF, JPG, or PNG."
_CHOOSE_GALLERY_ITEM = "Choose an invoice from the gallery."
_OUTCOME_LABEL = {
    "accepted": "Accepted",
    "human_review": "Needs review",
}
_INTRO = (
    "# NFe Document Extraction\n\n"
    f"{UNSUPPORTED_TYPE} At most 3 pages and 10 MB. {HEIC_NOT_ACCEPTED}\n\n"
    "The gallery shows cached results for synthetic invoices. "
    "Selecting one does not call a model and does not create a trace."
)
_GRADIO_MAX_UPLOAD_BYTES = 32 * 1024 * 1024
_COMMENT_HINT = "Optional. Do not include invoice data; long numbers are removed."
_FEEDBACK_FAILED = "Feedback could not be saved."
_READING_QUESTION = "Was this reading correct?"
_REJECTION_CONTROL = "This is a DANFE"


class DemoActions:
    """What the Gradio page calls. Gallery reads never touch the service."""

    def __init__(
        self, service: ExtractionService, gallery: Sequence[GalleryItem]
    ) -> None:
        self._service = service
        self._gallery = tuple(gallery)

    def gallery_cards(self) -> tuple[tuple[Image.Image, str], ...]:
        return tuple((item.image, item.caption) for item in self._gallery)

    def open_gallery_item(self, index: int) -> dict[str, object]:
        return copy.deepcopy(self._gallery[index].payload)

    def upload(
        self,
        data: bytes,
        *,
        filename: str | None,
        content_type: str | None,
        client_ip: str,
    ) -> dict[str, object] | Refusal:
        try:
            return self._service.extract(
                data,
                filename=filename,
                content_type=content_type,
                client_ip=client_ip,
            )
        except RequestRejected as exc:
            return Refusal(exc.message)

    def open_upload(
        self,
        data: bytes,
        *,
        filename: str | None,
        content_type: str | None,
        client_ip: str,
    ) -> dict[str, object]:
        """The screen for one file, including a feedback handle when a trace exists."""
        try:
            uploaded = self._service.read_upload(
                data,
                filename=filename,
                content_type=content_type,
                client_ip=client_ip,
            )
        except RequestRejected as exc:
            refused: dict[str, object] = {"notice": exc.message, "body": None}
            if exc.feedback_ref is not None:
                refused["feedback_ref"] = exc.feedback_ref
                refused["feedback_kind"] = "rejection"
            return refused
        shown: dict[str, object] = {"notice": None, "body": uploaded.body}
        if uploaded.feedback_ref is not None:
            shown["feedback_ref"] = uploaded.feedback_ref
            shown["feedback_kind"] = "reading"
        return shown

    def submit_feedback(
        self, feedback_ref: str, *, score: int, comment: str
    ) -> Refusal | None:
        try:
            self._service.submit_visitor_feedback(
                feedback_ref, score=score, comment=comment
            )
        except FeedbackNotRecorded:
            return Refusal(_FEEDBACK_FAILED)
        return None

    def submit_review(
        self, thread_id: str, edits: Mapping[str, object]
    ) -> dict[str, object] | Refusal:
        try:
            return self._service.submit_review(thread_id, edits)
        except RequestRejected as exc:
            return Refusal(exc.message)


def editable_field_paths(body: Mapping[str, object]) -> tuple[str, ...]:
    """Flagged fields of an open human review. A cached result has none."""
    if not body.get("thread_id"):
        return ()
    fields = body.get("fields")
    if not isinstance(fields, list):
        return ()
    return tuple(
        str(field["field"])
        for field in fields
        if isinstance(field, dict) and field.get("status") == "flagged"
    )


@dataclass(frozen=True)
class ReviewInput:
    """One flagged field the review form can edit."""

    path: str
    value: str


def review_inputs(body: Mapping[str, object]) -> tuple[ReviewInput, ...]:
    """Text boxes for an open review. Empty when the document is not paused."""
    extraction = body.get("extraction")
    if not isinstance(extraction, dict):
        return ()
    return tuple(
        ReviewInput(path=path, value=field_value(extraction, path))
        for path in editable_field_paths(body)
    )


def present_gallery_selection(actions: DemoActions, index: object) -> dict[str, object]:
    """The cached document for a gallery click, or a prompt to pick one."""
    resolved = _gallery_index(index)
    if resolved is None or not 0 <= resolved < len(actions.gallery_cards()):
        return {"notice": _CHOOSE_GALLERY_ITEM, "body": None}
    return {"notice": None, "body": actions.open_gallery_item(resolved)}


def present_upload(
    actions: DemoActions, path: str | None, *, client_ip: str
) -> dict[str, object]:
    """Read one chosen file through the extraction service."""
    if path is None or path == "":
        return {"notice": _CHOOSE_FILE, "body": None}
    file = Path(path)
    return actions.open_upload(
        file.read_bytes(),
        filename=file.name,
        content_type=None,
        client_ip=client_ip,
    )


def present_review(
    actions: DemoActions,
    screen: Mapping[str, object],
    edits: Mapping[str, object],
) -> dict[str, object]:
    """Revalidate flagged edits. A rejection keeps the open review on screen."""
    body = screen.get("body")
    if not isinstance(body, dict):
        return _continue_screen(screen, notice=NO_REVIEW, body=None)
    thread_id = body.get("thread_id")
    if not isinstance(thread_id, str) or thread_id == "":
        return _continue_screen(screen, notice=NO_REVIEW, body=body)
    result = actions.submit_review(thread_id, edits)
    if isinstance(result, Refusal):
        return _continue_screen(screen, notice=result.message, body=body)
    return _continue_screen(
        screen, notice=None, body=result, keep_feedback_notice=False
    )


def _continue_screen(
    screen: Mapping[str, object],
    *,
    notice: str | None,
    body: object,
    keep_feedback_notice: bool = True,
) -> dict[str, object]:
    """A new screen that keeps the same extraction's feedback handle."""
    shown: dict[str, object] = {"notice": notice, "body": body}
    ref = screen.get("feedback_ref")
    if isinstance(ref, str) and ref:
        shown["feedback_ref"] = ref
    kind = screen.get("feedback_kind")
    if kind in ("reading", "rejection"):
        shown["feedback_kind"] = kind
    if not keep_feedback_notice:
        return shown
    feedback_notice = screen.get("feedback_notice")
    if isinstance(feedback_notice, str) and feedback_notice:
        shown["feedback_notice"] = feedback_notice
    return shown


def visitor_feedback_prompt(screen: Mapping[str, object]) -> str | None:
    """The question for a traced upload. Gallery results and a silent monitor have none."""
    ref = screen.get("feedback_ref")
    if not isinstance(ref, str) or ref == "":
        return None
    if screen.get("feedback_kind") == "rejection":
        return _REJECTION_CONTROL
    if isinstance(screen.get("body"), dict):
        return _READING_QUESTION
    return None


def present_visitor_feedback(
    actions: DemoActions,
    screen: Mapping[str, object],
    *,
    score: int,
    comment: str,
) -> dict[str, object]:
    """Record thumbs and an optional comment. The document on screen stays put."""
    ref = screen.get("feedback_ref")
    if not isinstance(ref, str) or ref == "":
        return _with_feedback_notice(screen, _FEEDBACK_FAILED)
    failure = actions.submit_feedback(ref, score=score, comment=comment)
    if failure is not None:
        return _with_feedback_notice(screen, failure.message)
    return _with_feedback_notice(screen, None)


def _with_feedback_notice(
    screen: Mapping[str, object], feedback_notice: str | None
) -> dict[str, object]:
    shown = dict(screen)
    if feedback_notice:
        shown["feedback_notice"] = feedback_notice
    else:
        shown.pop("feedback_notice", None)
    return shown


def gradio_client_ip(request: object) -> str:
    """Visitor address from a Gradio request, using the API's rule."""
    from nfe_document_extraction.app.api import visitor_ip

    if request is None:
        return visitor_ip(None, None)
    headers = getattr(request, "headers", None)
    forwarded: str | None = None
    if headers is not None:
        try:
            pairs = dict(headers).items()
        except (TypeError, ValueError):
            pairs = ()
        for key, value in pairs:
            if str(key).lower() == "x-forwarded-for":
                forwarded = None if value is None else str(value)
                break
    client = getattr(request, "client", None)
    peer = getattr(client, "host", None) if client is not None else None
    if not isinstance(peer, str) or peer == "":
        peer = None
    return visitor_ip(forwarded, peer)


def field_rows(body: Mapping[str, object]) -> list[list[str]]:
    """Field, status label, printed value, and validator messages."""
    extraction = body["extraction"]
    fields = body["fields"]
    if not isinstance(extraction, dict) or not isinstance(fields, list):
        raise TypeError("A document needs an extraction and a field list.")
    rows: list[list[str]] = []
    for field in fields:
        if not isinstance(field, dict):
            raise TypeError("A field entry must be an object.")
        path = str(field["field"])
        messages = field["messages"]
        if not isinstance(messages, list):
            raise TypeError("Field messages must be a list.")
        rows.append(
            [
                path,
                _STATUS_LABEL[str(field["status"])],
                field_value(extraction, path),
                "; ".join(str(message) for message in messages),
            ]
        )
    return rows


def field_value(extraction: Mapping[str, object], path: str) -> str:
    """The value the document JSON stores at a validator field path.

    Money is stored as integer cents. The page shows reais, because a review
    text box submits a string and the schema reads money strings as reais.
    """
    if path.startswith("header.unsupported_components."):
        name = path.removeprefix("header.unsupported_components.")
        return _display(_nested(extraction, ("header", "unsupported_components", name)))
    if path.startswith("header."):
        name = path.removeprefix("header.")
        return _display(
            _nested(extraction, ("header", name)),
            money=name in _HEADER_MONEY,
        )
    match = _LINE_ITEM.fullmatch(path)
    if match is None:
        raise KeyError(path)
    return _display(
        _nested(extraction, ("line_items", match.group(1), match.group(2))),
        money=match.group(2) == "item_total",
    )


def _nested(value: object, parts: tuple[str, ...]) -> object:
    current = value
    for part in parts:
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(part)
    return current


def _display(value: object, *, money: bool = False) -> str:
    if money and isinstance(value, int) and not isinstance(value, bool):
        sign = "-" if value < 0 else ""
        amount = abs(value)
        return f"{sign}{amount // 100}.{amount % 100:02d}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _gallery_index(index: object) -> int | None:
    if isinstance(index, bool):
        return None
    if isinstance(index, int):
        return index
    if isinstance(index, (list, tuple)) and index and isinstance(index[0], int):
        first = index[0]
        if isinstance(first, bool):
            return None
        return first
    return None


def _uploaded_path(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, list):
        return _uploaded_path(value[0]) if value else None
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    return None


def document_panel(
    actions: DemoActions, body: Mapping[str, object], screen: gr.State
) -> None:
    """Outcome, fields, Excel download, and the review form for one document."""
    outcome = str(body.get("outcome"))
    label = _OUTCOME_LABEL.get(outcome, outcome)
    gr.Markdown(f"**Outcome:** {label}", elem_id="document-outcome")
    messages = body.get("review_messages")
    if isinstance(messages, list) and messages:
        gr.Markdown("\n".join(f"- {message}" for message in messages))
    gr.Dataframe(
        value=field_rows(body),
        headers=["Field", "Status", "Value", "Messages"],
        label="Fields",
        interactive=False,
        max_height=480,
    )
    gr.DownloadButton(
        label="Download Excel",
        value=str(excel_download_path(body)),
        elem_id="download-excel",
    )
    if outcome == "human_review" and not body.get("thread_id"):
        gr.Markdown(
            "This cached result is not an open review. "
            "Upload a document to correct flagged fields."
        )
    fields = review_inputs(body)
    if not fields:
        return
    gr.Markdown("Edit only the flagged fields. The same rules check your corrections.")
    boxes = [
        gr.Textbox(
            value=field.value,
            label=field.path,
            key=field.path,
            interactive=True,
        )
        for field in fields
    ]
    submit = gr.Button("Submit corrections")

    def _submit(current: object, *values: object) -> dict[str, object]:
        edits = {field.path: value for field, value in zip(fields, values, strict=True)}
        base = current if isinstance(current, dict) else {"notice": None, "body": body}
        return present_review(actions, base, edits)

    submit.click(_submit, [screen, *boxes], screen)


def build_blocks(actions: DemoActions) -> gr.Blocks:
    """The demo page. Gallery clicks never call the extraction service."""
    with gr.Blocks(title="NFe Document Extraction") as demo:
        gr.Markdown(_INTRO)
        screen = gr.State({"notice": None, "body": None})
        with gr.Row():
            with gr.Column():
                gallery = gr.Gallery(
                    value=list(actions.gallery_cards()),
                    label="Synthetic invoices",
                    columns=3,
                    allow_preview=False,
                    object_fit="contain",
                    buttons=[],
                    interactive=False,
                )
                # An empty sources list is treated as the upload default.
                gallery.sources = []
                upload = gr.File(
                    label="DANFE",
                    file_count="single",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png", ".heic", ".heif"],
                    type="filepath",
                )
                extract = gr.Button("Extract")
            with gr.Column():

                @gr.render(inputs=screen)
                def _draw(current: object) -> None:
                    render_result(actions, current, screen)

        def _on_select(evt: gr.SelectData) -> dict[str, object]:
            return present_gallery_selection(actions, evt.index)

        def _on_extract(file_value: object, request: gr.Request) -> dict[str, object]:
            return present_upload(
                actions,
                _uploaded_path(file_value),
                client_ip=gradio_client_ip(request),
            )

        gallery.select(_on_select, None, screen)
        extract.click(_on_extract, upload, screen)
    return demo


def mount_demo(app: FastAPI, actions: DemoActions) -> None:
    """Serve the page at ``/``. API routes must already be registered."""
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_SSR_MODE"] = "False"
    blocks = build_blocks(actions)
    app.state.demo_blocks = blocks
    gr.mount_gradio_app(
        app,
        blocks,
        path="/",
        ssr_mode=False,
        enable_monitoring=False,
        max_file_size=_GRADIO_MAX_UPLOAD_BYTES,
    )


def render_result(actions: DemoActions, current: object, screen: gr.State) -> None:
    """Outcome, review, and visitor feedback for the current screen."""
    notice, body = _screen_parts(current)
    if notice:
        gr.Markdown(notice)
    if isinstance(body, dict):
        document_panel(actions, body, screen)
    elif not notice:
        gr.Markdown("Select a gallery invoice or upload a DANFE.")
    _render_visitor_feedback(actions, current, screen)


def _render_visitor_feedback(
    actions: DemoActions, current: object, screen: gr.State
) -> None:
    if not isinstance(current, dict):
        return
    prompt = visitor_feedback_prompt(current)
    if prompt is None:
        return
    feedback_notice = current.get("feedback_notice")
    if isinstance(feedback_notice, str) and feedback_notice:
        gr.Markdown(feedback_notice)
    if prompt != _REJECTION_CONTROL:
        gr.Markdown(f"**{prompt}**")
    comment = gr.Textbox(label="Comment", info=_COMMENT_HINT)
    if prompt == _REJECTION_CONTROL:
        confirm = gr.Button(_REJECTION_CONTROL)
        confirm.click(_feedback_handler(actions, 0), [screen, comment], screen)
        return
    with gr.Row():
        up = gr.Button("Thumbs up")
        down = gr.Button("Thumbs down")
    up.click(_feedback_handler(actions, 1), [screen, comment], screen)
    down.click(_feedback_handler(actions, 0), [screen, comment], screen)


def _feedback_handler(
    actions: DemoActions, score: int
) -> Callable[[object, object], dict[str, object]]:
    def _handler(state: object, text: object) -> dict[str, object]:
        base = state if isinstance(state, dict) else {}
        return present_visitor_feedback(
            actions,
            base,
            score=score,
            comment=text if isinstance(text, str) else "",
        )

    return _handler


def _screen_parts(current: object) -> tuple[str | None, object]:
    if not isinstance(current, dict):
        return None, None
    notice = current.get("notice")
    return (notice if isinstance(notice, str) and notice else None), current.get("body")
