"""Upload the repository dataset to LangSmith.

The repository stays the source of truth. One example is created or updated
per Rendition. Requires LANGSMITH_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from langsmith import Client

from nfe_document_extraction.env import load_env_file
from nfe_document_extraction.evaluation.dataset import load_synthetic_pdfs
from nfe_document_extraction.evaluation.langsmith import (
    DATASET_NAME,
    LangSmithDatasetStore,
    sync_dataset,
)

_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_ROOT / "data" / "synthetic",
        help="Synthetic set directory (default: data/synthetic).",
    )
    parser.add_argument(
        "--name",
        default=DATASET_NAME,
        help=f"LangSmith dataset name (default: {DATASET_NAME}).",
    )
    args = parser.parse_args(argv)
    load_env_file(_ROOT / ".env")
    if not os.environ.get("LANGSMITH_API_KEY"):
        print(
            "LANGSMITH_API_KEY is missing. Copy .env.example to .env and set the key.",
            file=sys.stderr,
        )
        return 2

    documents = load_synthetic_pdfs(args.dataset)
    version = sync_dataset(
        LangSmithDatasetStore(Client()),
        documents,
        name=args.name,
    )
    print(
        f"Synced {len(documents)} renditions to {args.name}. Dataset version: {version}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    os.chdir(_ROOT)
    sys.exit(main())
