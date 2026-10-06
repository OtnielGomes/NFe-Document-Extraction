FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SSR_MODE=False \
    PORT=7860 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY data/synthetic ./data/synthetic

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

CMD ["python", "-m", "nfe_document_extraction.app"]
