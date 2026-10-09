"""Visitor feedback on a demo extraction, recorded against its LangSmith run."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import gradio as gr
from test_api import (
    RecordingLangSmith,
    RecordingMonitor,
    ScriptedReader,
    _client,
    _pdf,
    _upload,
)
from test_identifier_rules import ISSUER_CNPJ, _extraction
from test_ui import _IP

from nfe_document_extraction.agents.config import CHEAP_VISION, load_approach_config
from nfe_document_extraction.app.feedback import redact_feedback_comment
from nfe_document_extraction.app.gallery import load_gallery
from nfe_document_extraction.app.limits import CostLimits
from nfe_document_extraction.app.monitoring import (
    DemoMonitor,
    LangSmithMonitor,
    NullMonitor,
)
from nfe_document_extraction.app.service import ExtractionService
from nfe_document_extraction.app.ui import (
    DemoActions,
    present_gallery_selection,
    present_review,
    present_upload,
    present_visitor_feedback,
    render_result,
    visitor_feedback_prompt,
)

_COMMENT_HINT = "Optional. Do not include invoice data; long numbers are removed."


def test_long_digit_runs_with_separators_are_replaced_by_redacted() -> None:
    assert redact_feedback_comment("CNPJ 83.606.369/1229-50") == "CNPJ [redacted]"
    assert redact_feedback_comment("cpf 123.456.789-09") == "cpf [redacted]"
    assert (
        redact_feedback_comment("key 3524 0123 4567 8901 2345 6789 0123 4567 8901 2345")
        == "key [redacted]"
    )
    assert redact_feedback_comment("id 12345678901.") == "id [redacted]."
    assert redact_feedback_comment("12 and 12345678901") == "12 and [redacted]"


def test_shorter_numbers_stay_in_the_comment() -> None:
    assert redact_feedback_comment("quantity 1.2500") == "quantity 1.2500"
    assert redact_feedback_comment("code 1234567890") == "code 1234567890"
    assert redact_feedback_comment("pages 12-34") == "pages 12-34"


def test_the_comment_is_cut_to_500_characters_after_redaction() -> None:
    assert redact_feedback_comment("a" * 501) == "a" * 500
    assert (
        redact_feedback_comment("a" * 490 + "12345678901234567890")
        == "a" * 490 + "[redacted]"
    )
    assert redact_feedback_comment("a" * 500 + "12345678901234567890") == "a" * 500


def test_thumbs_and_a_comment_are_feedback_on_that_extraction_run(
    tmp_path: Path,
) -> None:
    monitor = RecordingMonitor()
    actions = _actions(ScriptedReader(_extraction()), monitor)

    liked = present_upload(actions, _pdf_path(tmp_path), client_ip=_IP)
    labels = _visible_text(actions, liked)
    assert visitor_feedback_prompt(liked) == "Was this reading correct?"
    assert "Thumbs up" in labels
    assert "Thumbs down" in labels
    assert _COMMENT_HINT in labels

    saved = present_visitor_feedback(
        actions,
        liked,
        score=1,
        comment="Looks right, CNPJ 83.606.369/1229-50",
    )

    assert isinstance(liked["body"], dict)
    assert liked["body"]["outcome"] == "accepted"
    assert saved["body"] == liked["body"]
    assert saved.get("feedback_notice") is None
    assert monitor.visitor_feedback == [
        _note(
            monitor,
            index=0,
            score=1,
            comment="Looks right, CNPJ [redacted]",
        )
    ]

    disliked = present_upload(
        actions, _pdf_path(tmp_path, "another.pdf"), client_ip="198.51.100.11"
    )
    present_visitor_feedback(actions, disliked, score=0, comment="")

    assert monitor.visitor_feedback[1] == _note(monitor, index=1, score=0, comment="")


def test_a_second_submission_updates_the_earlier_feedback(tmp_path: Path) -> None:
    monitor = RecordingMonitor()
    actions = _actions(ScriptedReader(_extraction()), monitor)
    screen = present_upload(actions, _pdf_path(tmp_path), client_ip=_IP)
    present_visitor_feedback(actions, screen, score=1, comment="first")
    first_id = monitor.visitor_feedback[0]["feedback_id"]

    present_visitor_feedback(
        actions, screen, score=0, comment="the description is wrong"
    )

    assert monitor.visitor_feedback == [
        {
            "feedback_id": first_id,
            "run_id": monitor.extractions[0].run_id,
            "score": 0,
            "comment": "the description is wrong",
        }
    ]


def test_langsmith_updates_visitor_feedback_instead_of_creating_another() -> None:
    client = RecordingLangSmith()
    monitor = LangSmithMonitor(client)

    monitor.record_visitor_feedback(
        "11111111-1111-1111-1111-111111111111",
        score=1,
        comment="ok",
        feedback_id="22222222-2222-2222-2222-222222222222",
    )
    monitor.record_visitor_feedback(
        "11111111-1111-1111-1111-111111111111",
        score=0,
        comment="nope",
        feedback_id="22222222-2222-2222-2222-222222222222",
    )

    assert client.feedback == [
        {
            "run_id": "11111111-1111-1111-1111-111111111111",
            "key": "visitor_feedback",
            "score": 1,
            "comment": "ok",
            "feedback_id": "22222222-2222-2222-2222-222222222222",
        }
    ]
    assert client.updates == [
        {
            "feedback_id": "22222222-2222-2222-2222-222222222222",
            "score": 0,
            "comment": "nope",
        }
    ]


def test_feedback_before_and_after_human_review(tmp_path: Path) -> None:
    monitor = RecordingMonitor()
    actions = _actions(
        ScriptedReader(_extraction(issuer_cnpj="00000000000181")), monitor
    )
    uploaded = present_upload(actions, _pdf_path(tmp_path), client_ip=_IP)
    assert isinstance(uploaded["body"], dict)
    assert uploaded["body"]["outcome"] == "human_review"
    assert visitor_feedback_prompt(uploaded) == "Was this reading correct?"

    present_visitor_feedback(actions, uploaded, score=0, comment="issuer looks wrong")
    first_id = monitor.visitor_feedback[0]["feedback_id"]
    assert monitor.visitor_feedback == [
        _note(monitor, index=0, score=0, comment="issuer looks wrong")
    ]

    reviewed = present_review(actions, uploaded, {"header.issuer_cnpj": ISSUER_CNPJ})
    assert isinstance(reviewed["body"], dict)
    assert reviewed["body"]["outcome"] == "accepted"
    assert visitor_feedback_prompt(reviewed) == "Was this reading correct?"

    present_visitor_feedback(actions, reviewed, score=1, comment="fixed")

    assert monitor.visitor_feedback == [
        {
            "feedback_id": first_id,
            "run_id": monitor.extractions[0].run_id,
            "score": 1,
            "comment": "fixed",
        }
    ]
    assert len(monitor.extractions) == 1


def test_feedback_on_a_rejected_document_says_this_is_a_danfe(tmp_path: Path) -> None:
    monitor = RecordingMonitor()
    actions = _actions(ScriptedReader(_extraction(access_key="")), monitor)

    rejected = present_upload(actions, _pdf_path(tmp_path), client_ip=_IP)

    assert rejected["notice"] == "This document does not look like a DANFE."
    assert rejected["body"] is None
    assert visitor_feedback_prompt(rejected) == "This is a DANFE"
    labels = _visible_text(actions, rejected)
    assert "This is a DANFE" in labels
    assert "Thumbs up" not in labels
    assert _COMMENT_HINT in labels

    saved = present_visitor_feedback(
        actions, rejected, score=0, comment="it is a printed invoice"
    )

    assert saved["notice"] == rejected["notice"]
    assert saved["body"] is None
    assert monitor.visitor_feedback == [
        _note(monitor, index=0, score=0, comment="it is a printed invoice")
    ]


def test_gallery_results_and_a_null_monitor_show_no_feedback_controls(
    tmp_path: Path,
) -> None:
    gallery_monitor = RecordingMonitor()
    gallery = DemoActions(
        _service(ScriptedReader(_extraction()), gallery_monitor),
        load_gallery(Path("data/synthetic"), today=date(2026, 10, 6)),
    )
    for index in range(len(gallery.gallery_cards())):
        selected = present_gallery_selection(gallery, index)
        assert visitor_feedback_prompt(selected) is None
    cached = present_gallery_selection(gallery, 0)
    assert _feedback_controls(gallery, cached) == set()
    assert gallery_monitor.extractions == []

    silent = _actions(ScriptedReader(_extraction()), NullMonitor())
    uploaded = present_upload(silent, _pdf_path(tmp_path), client_ip=_IP)
    assert isinstance(uploaded["body"], dict)
    assert uploaded["body"]["outcome"] == "accepted"
    assert visitor_feedback_prompt(uploaded) is None
    assert _feedback_controls(silent, uploaded) == set()

    rejected_actions = _actions(
        ScriptedReader(_extraction(access_key="")), NullMonitor()
    )
    rejected = present_upload(
        rejected_actions, _pdf_path(tmp_path, "rejected.pdf"), client_ip=_IP
    )
    assert rejected["notice"] == "This document does not look like a DANFE."
    assert visitor_feedback_prompt(rejected) is None
    assert _feedback_controls(rejected_actions, rejected) == set()


def test_a_failed_feedback_write_keeps_the_result_and_says_so(tmp_path: Path) -> None:
    actions = _actions(
        ScriptedReader(_extraction(issuer_cnpj="00000000000181")),
        _DownMonitor(),
    )
    screen = present_upload(actions, _pdf_path(tmp_path), client_ip=_IP)
    body = screen["body"]

    failed = present_visitor_feedback(
        actions, screen, score=0, comment="check the issuer"
    )

    assert failed["feedback_notice"] == "Feedback could not be saved."
    assert failed["body"] == body
    assert failed["notice"] is None
    assert "Feedback could not be saved." in _visible_text(actions, failed)
    assert isinstance(body, dict)
    assert _visible_text(actions, failed) & {"**Outcome:** Needs review"}

    rejected_actions = _actions(
        ScriptedReader(_extraction(access_key="")), _DownMonitor()
    )
    rejected = present_upload(
        rejected_actions, _pdf_path(tmp_path, "rejected.pdf"), client_ip=_IP
    )
    still = present_visitor_feedback(
        rejected_actions, rejected, score=0, comment="printed"
    )
    assert still["feedback_notice"] == "Feedback could not be saved."
    assert still["notice"] == "This document does not look like a DANFE."
    assert still["body"] is None


def test_the_document_payload_sent_to_the_browser_contains_no_trace_id(
    tmp_path: Path,
) -> None:
    monitor = RecordingMonitor()
    actions = _actions(
        ScriptedReader(_extraction(issuer_cnpj="00000000000181")), monitor
    )
    screen = present_upload(actions, _pdf_path(tmp_path), client_ip=_IP)
    run_id = monitor.extractions[0].run_id
    body = screen["body"]
    assert isinstance(body, dict)
    assert set(body) == {
        "outcome",
        "thread_id",
        "extraction",
        "fields",
        "review_messages",
    }
    assert screen["feedback_ref"] != run_id
    assert run_id not in json.dumps(screen)

    saved = present_visitor_feedback(actions, screen, score=1, comment="fine")
    feedback_id = str(monitor.visitor_feedback[0]["feedback_id"])
    assert feedback_id not in json.dumps(saved)
    assert run_id not in json.dumps(saved)

    reviewed = present_review(actions, screen, {"header.issuer_cnpj": ISSUER_CNPJ})
    assert run_id not in json.dumps(reviewed)
    assert feedback_id not in json.dumps(reviewed)

    api_monitor = RecordingMonitor()
    reader = ScriptedReader(_extraction())
    with _client(reader, monitor=api_monitor) as client:
        response = _upload(client, _pdf(1), "invoice.pdf", "application/pdf")
    assert response.status_code == 200
    api_run_id = api_monitor.extractions[0].run_id
    assert api_run_id not in response.text
    assert "run_id" not in response.json()
    assert "feedback_ref" not in response.json()


class _DownMonitor(RecordingMonitor):
    """Accepts feedback, then fails the write."""

    def record_visitor_feedback(
        self,
        run_id: str,
        *,
        score: int,
        comment: str,
        feedback_id: str,
    ) -> None:
        raise RuntimeError("langsmith down")


def _service(reader: ScriptedReader, monitor: DemoMonitor) -> ExtractionService:
    return ExtractionService(
        reader,
        monitor=monitor,
        limits=CostLimits(),
        clock=lambda: date(2026, 10, 6),
        price=load_approach_config(CHEAP_VISION),
    )


def _actions(reader: ScriptedReader, monitor: DemoMonitor) -> DemoActions:
    return DemoActions(_service(reader, monitor), ())


def _pdf_path(folder: Path, name: str = "invoice.pdf") -> str:
    path = folder / name
    path.write_bytes(_pdf(1))
    return str(path)


_FEEDBACK_LABELS = frozenset(
    {
        "Thumbs up",
        "Thumbs down",
        "This is a DANFE",
        "**Was this reading correct?**",
        _COMMENT_HINT,
    }
)


def _feedback_controls(actions: DemoActions, screen: dict[str, object]) -> set[str]:
    return _visible_text(actions, screen) & _FEEDBACK_LABELS


def _visible_text(actions: DemoActions, screen: dict[str, object]) -> set[str]:
    with gr.Blocks() as demo:
        state = gr.State(screen)
        render_result(actions, screen, state)
    visible: set[str] = set()
    for block in demo.blocks.values():
        if isinstance(block, (gr.Button, gr.Markdown)) and isinstance(block.value, str):
            visible.add(block.value)
        if isinstance(block, gr.Textbox) and isinstance(block.info, str):
            visible.add(block.info)
    return visible


def _note(
    monitor: RecordingMonitor, *, index: int, score: int, comment: str
) -> dict[str, object]:
    stored = monitor.visitor_feedback[index]
    return {
        "feedback_id": stored["feedback_id"],
        "run_id": monitor.extractions[index].run_id,
        "score": score,
        "comment": comment,
    }
