"""Per-run LangSmith tracing switch.

Once tracing is disabled, this process does not send data to LangSmith.
``tracing_is_disabled`` stays true only while that choice is still visible in
the environment and the LangSmith SDK reports tracing off, so a caller can
refuse to start when it cannot confirm it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass

_forced_off: ContextVar[bool] = ContextVar(
    "nfe_langsmith_tracing_forced_off",
    default=False,
)

_OFF = frozenset({"false", "0", "no", "off"})
_ON = frozenset({"true", "1", "yes", "on"})


def disable_tracing() -> None:
    """Force tracing off for this process and the environment."""
    _forced_off.set(True)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    _forget_cached_tracing_flag()


def tracing_is_disabled() -> bool:
    """True when tracing was disabled and LangSmith still agrees it is off."""
    env_off = (
        _forced_off.get()
        and os.environ.get("LANGSMITH_TRACING") == "false"
        and os.environ.get("LANGCHAIN_TRACING_V2") == "false"
    )
    return env_off and _sdk_tracing_is_enabled() is False


def send_to_langsmith(send: Callable[[], None]) -> None:
    """Run ``send`` only when this process may publish to LangSmith."""
    if _forced_off.get():
        return
    flag = os.environ.get("LANGSMITH_TRACING", "").strip().lower()
    if flag in _OFF:
        return
    if not os.environ.get("LANGSMITH_API_KEY"):
        return
    send()


@dataclass(frozen=True)
class TracingChoice:
    """Whether this run may upload an experiment, and whether model calls are traced.

    Model tracing is off unless ``--trace`` or ``LANGSMITH_TRACING`` asks for it,
    so a LangSmith key alone does not send the graph state. ``--no-trace`` sends
    nothing.
    """

    upload: bool
    trace_models: bool


def resolve_tracing(*, trace: bool, no_trace: bool) -> TracingChoice:
    """Choose tracing for one run. An explicit off switch wins."""
    if _forced_off.get() or no_trace:
        disable_tracing()
        return TracingChoice(upload=False, trace_models=False)
    env = os.environ.get("LANGSMITH_TRACING")
    if env is not None and env.strip().lower() in _OFF:
        disable_tracing()
        return TracingChoice(upload=False, trace_models=False)
    if not os.environ.get("LANGSMITH_API_KEY"):
        return TracingChoice(upload=False, trace_models=False)
    trace_models = trace or (env is not None and env.strip().lower() in _ON)
    if trace_models:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        _forget_cached_tracing_flag()
    return TracingChoice(upload=True, trace_models=trace_models)


def _forget_cached_tracing_flag() -> None:
    """Drop LangSmith's cached env read so a later check sees this run's choice."""
    from langsmith.utils import get_env_var

    get_env_var.cache_clear()


def _sdk_tracing_is_enabled() -> bool | str:
    from langsmith.utils import tracing_is_enabled

    return tracing_is_enabled()
