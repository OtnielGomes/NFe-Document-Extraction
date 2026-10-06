"""Run the public demo: JSON API and Gradio UI in one process."""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    from nfe_document_extraction.app.api import build_demo_app

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(build_demo_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
