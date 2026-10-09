#!/usr/bin/env python3
"""Cursor SDK runtime adapter for skill-creator scripts.

Every dependency on `cursor_sdk` is isolated in this module so the eval and
description scripts stay free of SDK details.

Upstream skill-creator drove the Anthropic `claude` CLI through subprocesses and
scraped its JSON stream to find out which tools the agent used. On Cursor the
equivalent is the Cursor SDK, which hands back the agent loop, its tool calls and
its token usage as typed objects, so the scraping layer disappears entirely.

Run this module directly for an environment check:

    python -m scripts.cursor_runtime
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "composer-2.5"
SDK_PACKAGE = "cursor-sdk"
INSTALL_HINT = f"pip install {SDK_PACKAGE}"
API_KEY_ENV = "CURSOR_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 120.0

TERMINAL_OK_STATUS = "finished"


class RuntimeUnavailableError(RuntimeError):
    """Raised when the Cursor SDK or its credentials cannot be used."""


@dataclass
class ToolCall:
    """A single tool invocation observed during an agent run.

    Attributes:
        name: Tool name reported by the agent, e.g. ``Read`` or ``Shell``.
        args: Raw arguments the agent passed to the tool.
        status: Tool call status, one of ``running``, ``completed`` or ``error``.
    """

    name: str
    args: Any
    status: str

    def mentions(self, needle: str) -> bool:
        """Check whether a string appears in the tool name or its arguments.

        Args:
            needle: Substring to look for, typically a unique skill directory name.

        Returns:
            True when the needle occurs in the tool name or serialized arguments.
        """
        if needle in self.name:
            return True
        try:
            serialized = json.dumps(self.args, default=str)
        except (TypeError, ValueError):
            serialized = str(self.args)
        return needle in serialized


@dataclass
class RunObservation:
    """Outcome of an observed agent run.

    Attributes:
        status: Terminal run status reported by the SDK.
        text: Final assistant text, empty when the run was stopped early.
        duration_ms: Wall clock duration measured by the SDK, or locally on early stop.
        total_tokens: Token total reported by the SDK, zero when unavailable.
        tool_calls: Tool calls observed while streaming.
        stopped_early: True when the caller's predicate cancelled the run.
        error: Human-readable failure reason when the run did not finish cleanly.
    """

    status: str
    text: str = ""
    duration_ms: int = 0
    total_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    stopped_early: bool = False
    error: str | None = None


def _import_sdk() -> Any:
    """Import the Cursor SDK, translating an absent package into a clear error.

    Returns:
        The imported ``cursor_sdk`` module.

    Raises:
        RuntimeUnavailableError: When the package is not installed.
    """
    try:
        import cursor_sdk
    except ImportError as exc:
        raise RuntimeUnavailableError(
            f"The Cursor SDK is not installed. Install it with: {INSTALL_HINT}"
        ) from exc
    return cursor_sdk


def resolve_api_key(explicit: str | None = None) -> str:
    """Resolve the Cursor API key from an explicit value or the environment.

    Args:
        explicit: Key supplied by the caller, takes precedence when set.

    Returns:
        The resolved API key with surrounding whitespace stripped.

    Raises:
        RuntimeUnavailableError: When no key is available.
    """
    key = (explicit or os.environ.get(API_KEY_ENV, "")).strip()
    if not key:
        raise RuntimeUnavailableError(
            f"No Cursor API key found. Set {API_KEY_ENV} or pass --api-key. "
            "Create one at https://cursor.com/dashboard/integrations"
        )
    return key


def _field(message: Any, name: str, default: Any = None) -> Any:
    """Read a field from an SDK message that may be a dataclass or a mapping.

    The SDK message union includes ``Mapping[str, Any]`` for forward
    compatibility, so both shapes have to be supported.

    Args:
        message: SDK message object or mapping.
        name: Field name to read.
        default: Value returned when the field is absent.

    Returns:
        The field value, or the default.
    """
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def _tool_call_from(message: Any) -> ToolCall:
    """Build a ToolCall from an SDK ``tool_call`` message."""
    return ToolCall(
        name=str(_field(message, "name", "") or ""),
        args=_field(message, "args"),
        status=str(_field(message, "status", "") or ""),
    )


def _build_options(
    sdk: Any,
    *,
    cwd: Path | str,
    model: str,
    api_key: str,
    setting_sources: Sequence[str],
) -> Any:
    """Assemble AgentOptions for a local run.

    The runtime is always set explicitly: the SDK silently defaults to local when
    neither ``local`` nor ``cloud`` is given, and a silent default is a bad thing
    to depend on.
    """
    return sdk.AgentOptions(
        model=model,
        api_key=api_key,
        local=sdk.LocalAgentOptions(
            cwd=str(cwd),
            setting_sources=list(setting_sources),
        ),
    )


def observe_run(
    prompt: str,
    *,
    cwd: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    setting_sources: Sequence[str] = ("project",),
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    stop_when: Callable[[ToolCall], bool] | None = None,
) -> RunObservation:
    """Run a prompt locally and observe the agent's tool calls.

    The timeout is evaluated between streamed messages rather than preempting a
    blocking read, so a run that produces no messages at all can exceed it. In
    practice agents emit messages steadily, and `stop_when` usually ends the run
    long before the deadline.

    Args:
        prompt: The user prompt to send to the agent.
        cwd: Workspace directory the local agent runs against.
        model: Model identifier; required for local runs.
        api_key: Explicit API key, falling back to the environment.
        setting_sources: Cursor setting sources to load. Use ``("project",)`` when
            the run needs to discover skills from the workspace, or ``()`` for an
            isolated run that should see no ambient configuration.
        timeout_seconds: Deadline after which the run is cancelled.
        stop_when: Predicate over observed tool calls; returning True cancels the
            run immediately, which is how trigger detection avoids paying for a
            full completion.

    Returns:
        A RunObservation describing what happened.

    Raises:
        RuntimeUnavailableError: When the SDK or credentials are unusable.
    """
    sdk = _import_sdk()
    key = resolve_api_key(api_key)
    options = _build_options(
        sdk, cwd=cwd, model=model, api_key=key, setting_sources=setting_sources
    )

    tool_calls: list[ToolCall] = []
    stopped_early = False
    timed_out = False
    started = time.monotonic()

    try:
        with sdk.Agent.create(options) as agent:
            run = agent.send(prompt)
            logger.debug("run started: run_id=%s", getattr(run, "run_id", "?"))

            for message in run.messages():
                if _field(message, "type") == "tool_call":
                    call = _tool_call_from(message)
                    tool_calls.append(call)
                    if stop_when is not None and stop_when(call):
                        stopped_early = True
                        break

                if time.monotonic() - started > timeout_seconds:
                    timed_out = True
                    break

            if (stopped_early or timed_out) and run.supports("cancel"):
                run.cancel()

            result = run.wait()

    except sdk.CursorAgentError as exc:
        # The run never executed: auth, configuration or network.
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return RunObservation(
            status="error",
            duration_ms=elapsed_ms,
            tool_calls=tool_calls,
            stopped_early=stopped_early,
            error=f"agent did not start: {exc}",
        )

    usage = getattr(result, "usage", None)
    return RunObservation(
        status=str(getattr(result, "status", "unknown")),
        text=str(getattr(result, "result", "") or ""),
        duration_ms=int(getattr(result, "duration_ms", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        tool_calls=tool_calls,
        stopped_early=stopped_early,
        error="timed out" if timed_out else None,
    )


def complete_text(
    prompt: str,
    *,
    cwd: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> str:
    """Send a one-shot prompt and return the final assistant text.

    Ambient settings are deliberately not loaded: this is a pure text completion
    and project skills or rules would only add noise and cost.

    Args:
        prompt: The prompt to send.
        cwd: Workspace directory for the local runtime.
        model: Model identifier.
        api_key: Explicit API key, falling back to the environment.

    Returns:
        The final assistant message text.

    Raises:
        RuntimeUnavailableError: When the SDK is missing, credentials are absent,
            or the run failed.
    """
    sdk = _import_sdk()
    key = resolve_api_key(api_key)
    options = _build_options(sdk, cwd=cwd, model=model, api_key=key, setting_sources=())

    try:
        result = sdk.Agent.prompt(prompt, options)
    except sdk.CursorAgentError as exc:
        raise RuntimeUnavailableError(f"Cursor agent did not start: {exc}") from exc

    status = str(getattr(result, "status", "unknown"))
    if status != TERMINAL_OK_STATUS:
        raise RuntimeUnavailableError(f"Cursor run ended with status {status!r}")

    return str(getattr(result, "result", "") or "")


def doctor(api_key: str | None = None) -> dict[str, Any]:
    """Check that the Cursor SDK runtime is usable and report what is missing.

    Args:
        api_key: Explicit API key to test instead of the environment variable.

    Returns:
        A report with ``sdk_installed``, ``api_key_present``, ``models`` and a
        list of human-readable ``problems``.
    """
    report: dict[str, Any] = {
        "sdk_installed": False,
        "api_key_present": False,
        "models": [],
        "problems": [],
    }

    try:
        sdk = _import_sdk()
        report["sdk_installed"] = True
    except RuntimeUnavailableError as exc:
        report["problems"].append(str(exc))
        return report

    try:
        resolve_api_key(api_key)
        report["api_key_present"] = True
    except RuntimeUnavailableError as exc:
        report["problems"].append(str(exc))
        return report

    try:
        models = sdk.Cursor.models.list()
        items = getattr(models, "data", models)
        report["models"] = [str(getattr(m, "id", m)) for m in items]
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        report["problems"].append(f"could not list models: {exc}")

    return report


def main() -> None:
    """Print an environment report for the Cursor SDK runtime."""
    import argparse

    parser = argparse.ArgumentParser(description="Check the Cursor SDK runtime")
    parser.add_argument("--api-key", default=None, help=f"Override {API_KEY_ENV}")
    args = parser.parse_args()

    report = doctor(args.api_key)
    print(json.dumps(report, indent=2))

    if report["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
