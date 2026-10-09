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

from nfe_document_extraction.app.copy import (
    CACHED_REVIEW_NOTE,
    CHOOSE_FILE,
    CHOOSE_GALLERY_ITEM,
    COMMENT_HINT,
    EMPTY_SCREEN,
    FEEDBACK_FAILED,
    READING_QUESTION,
    REJECTION_CONTROL,
    REVIEW_INTRO,
    decimal_hint,
    field_label,
    gallery_caption,
    intro_body,
    language_code,
    outcome_label,
    status_label,
    visitor_text,
)
from nfe_document_extraction.app.errors import RequestRejected
from nfe_document_extraction.app.excel import excel_download_path
from nfe_document_extraction.app.gallery import GalleryItem, load_gallery
from nfe_document_extraction.app.service import (
    NO_REVIEW,
    ExtractionService,
    FeedbackNotRecorded,
)

__all__ = [
    "DemoActions",
    "LanguageChrome",
    "Refusal",
    "ScreenCopy",
    "document_panel",
    "editable_field_paths",
    "field_rows",
    "gradio_client_ip",
    "language_chrome",
    "load_gallery",
    "present_gallery_selection",
    "present_review",
    "present_upload",
    "present_visitor_feedback",
    "render_result",
    "review_inputs",
    "screen_copy",
    "visitor_feedback_prompt",
]


@dataclass(frozen=True)
class Refusal:
    """A limit or upload rule, in the words the visitor should read."""

    message: str


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
_GRADIO_MAX_UPLOAD_BYTES = 32 * 1024 * 1024
_LANGUAGE_KEY = "nfe-demo-language"
_STORE_LANGUAGE_JS = f"""
(choice) => {{
  const code = choice === "English" ? "en" : "pt";
  localStorage.setItem("{_LANGUAGE_KEY}", code);
  return [choice];
}}
"""
_RESTORE_LANGUAGE_JS = f"""
(choice) => {{
  const stored = localStorage.getItem("{_LANGUAGE_KEY}");
  if (stored === "en") return ["English"];
  return [choice];
}}
"""
_LANGUAGE_CSS = """
#demo-title-row { align-items: center; flex-wrap: wrap; }
#demo-title-row > .block {
  min-width: min(16rem, 100%) !important;
  flex: 1 1 16rem !important;
  overflow: visible !important;
}
#demo-language { margin-left: auto; }
#demo-language .wrap { flex-direction: row; flex-wrap: nowrap; gap: 0.75rem 1rem; align-items: center; }
#demo-language label { white-space: nowrap; }
"""


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
            return Refusal(FEEDBACK_FAILED)
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
    label: str


def review_inputs(
    body: Mapping[str, object], *, language: str = "pt"
) -> tuple[ReviewInput, ...]:
    """Text boxes for an open review. Empty when the document is not paused."""
    extraction = body.get("extraction")
    if not isinstance(extraction, dict):
        return ()
    return tuple(
        ReviewInput(
            path=path,
            value=field_value(extraction, path),
            label=field_label(path, language, extraction),
        )
        for path in editable_field_paths(body)
    )


def present_gallery_selection(actions: DemoActions, index: object) -> dict[str, object]:
    """The cached document for a gallery click, or a prompt to pick one."""
    resolved = _gallery_index(index)
    if resolved is None or not 0 <= resolved < len(actions.gallery_cards()):
        return {"notice": CHOOSE_GALLERY_ITEM, "body": None}
    return {"notice": None, "body": actions.open_gallery_item(resolved)}


def present_upload(
    actions: DemoActions, path: str | None, *, client_ip: str
) -> dict[str, object]:
    """Read one chosen file through the extraction service."""
    if path is None or path == "":
        return {"notice": CHOOSE_FILE, "body": None}
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
        return REJECTION_CONTROL
    if isinstance(screen.get("body"), dict):
        return READING_QUESTION
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
        return _with_feedback_notice(screen, FEEDBACK_FAILED)
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


def field_rows(body: Mapping[str, object], *, language: str = "pt") -> list[list[str]]:
    """Field label, status, printed value, and validator messages."""
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
                field_label(path, language, extraction),
                status_label(str(field["status"]), language),
                field_value(extraction, path),
                "; ".join(visitor_text(str(message), language) for message in messages),
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


@dataclass(frozen=True)
class LanguageChrome:
    """Title-row text and gallery cards for one language. No service call."""

    language: str
    intro: str
    gallery_label: str
    extract_label: str
    cards: tuple[tuple[Image.Image, str], ...]

    @property
    def captions(self) -> tuple[str, ...]:
        return tuple(caption for _, caption in self.cards)


@dataclass(frozen=True)
class ScreenCopy:
    """What the visitor reads for a screen the page already holds."""

    language: str
    notice: str | None
    outcome_line: str | None
    review_messages: tuple[str, ...]
    table_label: str
    headers: tuple[str, str, str, str]
    rows: tuple[tuple[str, str, str, str], ...]
    download_label: str
    cached_note: str | None
    review_intro: str | None
    decimal_hint: str | None
    review_fields: tuple[ReviewInput, ...]
    submit_label: str
    empty_hint: str | None
    feedback_notice: str | None
    feedback_question: str | None
    rejection_label: str | None
    comment_label: str
    comment_hint: str
    positive_label: str
    negative_label: str
    show_feedback: bool


def language_chrome(actions: DemoActions, choice: object) -> LanguageChrome:
    """Redraw the page chrome. Gallery cards are the cache, with a new caption."""
    language = language_code(choice)
    return LanguageChrome(
        language=language,
        intro=intro_body(language),
        gallery_label=visitor_text("Synthetic invoices", language),
        extract_label=visitor_text("Extract", language),
        cards=tuple(
            (image, gallery_caption(caption, language))
            for image, caption in actions.gallery_cards()
        ),
    )


def screen_copy(screen: Mapping[str, object], language: str) -> ScreenCopy:
    """Translate the screen already held. Does not read a file or call a model."""
    body = screen.get("body")
    document = body if isinstance(body, dict) else None
    fields = review_inputs(document, language=language) if document else ()
    hint = decimal_hint(language) if fields else ""
    prompt = visitor_feedback_prompt(screen)
    return ScreenCopy(
        language=language,
        notice=_translate(_text(screen.get("notice")), language),
        outcome_line=_outcome_line(document, language),
        review_messages=_review_messages(document, language),
        table_label=visitor_text("Fields", language),
        headers=(
            visitor_text("Field", language),
            visitor_text("Status", language),
            visitor_text("Value", language),
            visitor_text("Messages", language),
        ),
        rows=_rows(document, language),
        download_label=visitor_text("Download Excel", language),
        cached_note=_cached_note(document, language),
        review_intro=visitor_text(REVIEW_INTRO, language) if fields else None,
        decimal_hint=hint or None,
        review_fields=fields,
        submit_label=visitor_text("Submit corrections", language),
        empty_hint=_empty_hint(document, screen, language),
        feedback_notice=_translate(_text(screen.get("feedback_notice")), language),
        feedback_question=(
            visitor_text(prompt, language)
            if prompt and prompt != REJECTION_CONTROL
            else None
        ),
        rejection_label=(
            visitor_text(REJECTION_CONTROL, language)
            if prompt == REJECTION_CONTROL
            else None
        ),
        comment_label=visitor_text("Comment", language),
        comment_hint=visitor_text(COMMENT_HINT, language),
        positive_label=visitor_text("Thumbs up", language),
        negative_label=visitor_text("Thumbs down", language),
        show_feedback=prompt is not None,
    )


def document_panel(
    actions: DemoActions,
    body: Mapping[str, object],
    screen: gr.State,
    language: str = "pt",
) -> None:
    """Outcome, fields, Excel download, and the review form for one document."""
    _draw_document(actions, body, screen, screen_copy({"body": body}, language))


def build_blocks(actions: DemoActions) -> gr.Blocks:
    """The demo page. Gallery clicks never call the extraction service."""
    with gr.Blocks(title="NFe Document Extraction") as demo:
        with gr.Row(elem_id="demo-title-row"):
            gr.Markdown("# NFe Document Extraction", min_width=240, scale=1)
            language_choice = gr.Radio(
                choices=["Português", "English"],
                value="Português",
                show_label=False,
                elem_id="demo-language",
                min_width=240,
                scale=0,
            )
        language = gr.State("pt")
        screen = gr.State({"notice": None, "body": None})
        opening = language_chrome(actions, "Português")
        intro = gr.Markdown(opening.intro)
        with gr.Row():
            with gr.Column():
                gallery = gr.Gallery(
                    value=list(opening.cards),
                    label=opening.gallery_label,
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
                extract = gr.Button(opening.extract_label)
            with gr.Column():

                @gr.render(inputs=[screen, language], show_progress="hidden")
                def _draw(current: object, selected: object) -> None:
                    render_result(
                        actions,
                        current,
                        screen,
                        selected if isinstance(selected, str) else "pt",
                    )

        def _apply(choice: object) -> tuple[object, ...]:
            chrome = language_chrome(actions, choice)
            return _chrome_update(chrome)

        def _restore(choice: object) -> tuple[object, ...]:
            selected = "English" if language_code(choice) == "en" else "Português"
            return (selected, *_chrome_update(language_chrome(actions, selected)))

        # Queued on purpose: Gradio only tells @gr.render that a State changed
        # when the event runs through the queue. queue=False updates the chrome
        # and leaves the open document in the previous language.
        language_choice.change(
            _apply,
            language_choice,
            [language, intro, gallery, extract],
            js=_STORE_LANGUAGE_JS,
            show_progress="hidden",
        )
        demo.load(
            _restore,
            language_choice,
            [language_choice, language, intro, gallery, extract],
            js=_RESTORE_LANGUAGE_JS,
            show_progress="hidden",
        )

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
        css=_LANGUAGE_CSS,
    )


def render_result(
    actions: DemoActions,
    current: object,
    screen: gr.State,
    language: str = "pt",
) -> None:
    """Outcome, review, and visitor feedback for the current screen."""
    data = current if isinstance(current, dict) else {}
    view = screen_copy(data, language)
    if view.notice:
        gr.Markdown(view.notice)
    body = data.get("body")
    if isinstance(body, dict):
        _draw_document(actions, body, screen, view)
    elif view.empty_hint:
        gr.Markdown(view.empty_hint)
    _draw_feedback(actions, screen, view)


def _draw_document(
    actions: DemoActions,
    body: Mapping[str, object],
    screen: gr.State,
    view: ScreenCopy,
) -> None:
    if view.outcome_line:
        gr.Markdown(view.outcome_line, elem_id="document-outcome")
    if view.review_messages:
        gr.Markdown("\n".join(f"- {message}" for message in view.review_messages))
    gr.Dataframe(
        value=[list(row) for row in view.rows],
        headers=list(view.headers),
        label=view.table_label,
        interactive=False,
        max_height=480,
    )
    gr.DownloadButton(
        label=view.download_label,
        value=str(excel_download_path(body, language=view.language)),
        elem_id="download-excel",
    )
    if view.cached_note:
        gr.Markdown(view.cached_note)
    if not view.review_fields:
        return
    if view.review_intro:
        gr.Markdown(view.review_intro)
    if view.decimal_hint:
        gr.Markdown(view.decimal_hint)
    boxes = [
        gr.Textbox(
            value=field.value,
            label=field.label,
            # The language is part of the key so a switch builds new boxes.
            # Gradio would otherwise keep text the visitor typed and has not sent.
            key=(view.language, field.path),
            interactive=True,
        )
        for field in view.review_fields
    ]
    submit = gr.Button(view.submit_label)

    def _submit(current: object, *values: object) -> dict[str, object]:
        edits = {
            field.path: value
            for field, value in zip(view.review_fields, values, strict=True)
        }
        base = current if isinstance(current, dict) else {"notice": None, "body": body}
        return present_review(actions, base, edits)

    submit.click(_submit, [screen, *boxes], screen)


def _draw_feedback(actions: DemoActions, screen: gr.State, view: ScreenCopy) -> None:
    if not view.show_feedback:
        return
    if view.feedback_notice:
        gr.Markdown(view.feedback_notice)
    if view.feedback_question:
        gr.Markdown(f"**{view.feedback_question}**")
    comment = gr.Textbox(label=view.comment_label, info=view.comment_hint)
    if view.rejection_label:
        confirm = gr.Button(view.rejection_label)
        confirm.click(_feedback_handler(actions, 0), [screen, comment], screen)
        return
    with gr.Row():
        up = gr.Button(view.positive_label)
        down = gr.Button(view.negative_label)
    up.click(_feedback_handler(actions, 1), [screen, comment], screen)
    down.click(_feedback_handler(actions, 0), [screen, comment], screen)


def _chrome_update(chrome: LanguageChrome) -> tuple[object, ...]:
    return (
        chrome.language,
        chrome.intro,
        gr.update(value=list(chrome.cards), label=chrome.gallery_label),
        gr.update(value=chrome.extract_label),
    )


def _rows(
    document: Mapping[str, object] | None, language: str
) -> tuple[tuple[str, str, str, str], ...]:
    if document is None:
        return ()
    return tuple(
        (row[0], row[1], row[2], row[3])
        for row in field_rows(document, language=language)
    )


def _outcome_line(document: Mapping[str, object] | None, language: str) -> str | None:
    if document is None:
        return None
    label = outcome_label(str(document.get("outcome")), language)
    return f"**{visitor_text('Outcome', language)}:** {label}"


def _review_messages(
    document: Mapping[str, object] | None, language: str
) -> tuple[str, ...]:
    if document is None:
        return ()
    messages = document.get("review_messages")
    if not isinstance(messages, list):
        return ()
    return tuple(visitor_text(str(message), language) for message in messages)


def _cached_note(document: Mapping[str, object] | None, language: str) -> str | None:
    if document is None:
        return None
    if document.get("outcome") == "human_review" and not document.get("thread_id"):
        return visitor_text(CACHED_REVIEW_NOTE, language)
    return None


def _translate(english: str | None, language: str) -> str | None:
    if english is None:
        return None
    return visitor_text(english, language)


def _empty_hint(
    document: Mapping[str, object] | None,
    screen: Mapping[str, object],
    language: str,
) -> str | None:
    if document is None and _text(screen.get("notice")) is None:
        return visitor_text(EMPTY_SCREEN, language)
    return None


def _text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


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
