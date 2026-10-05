"""Keep LangSmith off unless a test turns it on."""

from collections.abc import Iterator

import pytest

from nfe_document_extraction.evaluation import tracing


@pytest.fixture(autouse=True)
def _isolate_langsmith(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    token = tracing._forced_off.set(False)
    yield
    tracing._forced_off.reset(token)
