"""Tracing can be switched off. While it is off, nothing is sent to LangSmith."""

import os

import pytest

from nfe_document_extraction.evaluation.tracing import (
    disable_tracing,
    resolve_tracing,
    send_to_langsmith,
    tracing_is_disabled,
)


def test_disabling_tracing_turns_the_sdk_off_after_it_already_read_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    from langsmith import Client
    from langsmith.utils import tracing_is_enabled

    assert Client.__module__.startswith("langsmith")
    assert tracing_is_enabled() is True

    disable_tracing()

    assert tracing_is_enabled() is False
    assert tracing_is_disabled()


def test_disabling_tracing_turns_the_sdk_off_when_tracing_v2_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    from langsmith.utils import get_env_var, tracing_is_enabled

    get_env_var.cache_clear()
    assert tracing_is_enabled() is True

    disable_tracing()

    assert tracing_is_enabled() is False
    assert tracing_is_disabled()
    assert os.environ["LANGSMITH_TRACING_V2"] == "false"


def test_disabling_tracing_overrides_configure_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langsmith
    from langsmith.utils import tracing_is_enabled

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    langsmith.configure(enabled=True)
    assert tracing_is_enabled() is True

    disable_tracing()

    assert tracing_is_enabled() is False
    assert tracing_is_disabled()


def test_trace_flag_turns_the_sdk_on_when_tracing_v2_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")
    from langsmith.utils import get_env_var, tracing_is_enabled

    get_env_var.cache_clear()
    assert tracing_is_enabled() is False

    choice = resolve_tracing(trace=True, no_trace=False)

    assert choice.trace_models is True
    assert tracing_is_enabled() is True


def test_trace_flag_turns_the_sdk_on_after_it_already_read_tracing_as_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    from langsmith.utils import tracing_is_enabled

    assert tracing_is_enabled() is False

    choice = resolve_tracing(trace=True, no_trace=False)

    assert choice.trace_models is True
    assert tracing_is_enabled() is True


def test_disabling_tracing_sends_nothing_and_can_be_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    sent: list[str] = []

    disable_tracing()
    send_to_langsmith(lambda: sent.append("invoice"))

    assert sent == []
    assert tracing_is_disabled()
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_forced_off_stays_off_when_the_env_var_is_turned_back_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    disable_tracing()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    sent: list[str] = []

    send_to_langsmith(lambda: sent.append("invoice"))

    assert sent == []
    assert tracing_is_disabled() is False


def test_a_key_with_tracing_on_is_allowed_to_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    sent: list[str] = []

    send_to_langsmith(lambda: sent.append("scores"))

    assert sent == ["scores"]
    assert tracing_is_disabled() is False


def test_no_trace_wins_over_a_key_and_an_explicit_trace_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    choice = resolve_tracing(trace=True, no_trace=True)

    assert choice.upload is False
    assert choice.trace_models is False
    assert tracing_is_disabled()


def test_a_key_does_not_trace_when_only_tracing_v2_was_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    from langsmith.utils import get_env_var, tracing_is_enabled

    get_env_var.cache_clear()
    assert tracing_is_enabled() is True

    choice = resolve_tracing(trace=False, no_trace=False)

    assert choice.upload is True
    assert choice.trace_models is False
    assert tracing_is_enabled() is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_a_key_uploads_without_tracing_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    choice = resolve_tracing(trace=False, no_trace=False)

    assert choice.upload is True
    assert choice.trace_models is False
    assert "LANGSMITH_TRACING" not in os.environ

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    traced = resolve_tracing(trace=False, no_trace=False)
    assert traced.upload is True
    assert traced.trace_models is True

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    disabled = resolve_tracing(trace=False, no_trace=False)
    assert disabled.upload is False
    assert disabled.trace_models is False
    assert tracing_is_disabled()


def test_trace_flag_turns_model_tracing_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    choice = resolve_tracing(trace=True, no_trace=False)

    assert choice.upload is True
    assert choice.trace_models is True
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_without_a_key_tracing_stays_off_and_unforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choice = resolve_tracing(trace=True, no_trace=False)

    assert choice.upload is False
    assert choice.trace_models is False
    assert tracing_is_disabled() is False
    assert "LANGSMITH_TRACING" not in os.environ
