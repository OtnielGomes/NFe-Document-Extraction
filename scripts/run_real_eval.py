"""Run the Real set locally and write aggregated rows into docs/results.md.

Tracing is forced off for the whole process, whatever the environment says.
The run refuses to start when that cannot be confirmed. Documents stay on
this machine; the results file receives only aggregated metrics.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

from openai import OpenAI

from nfe_document_extraction.agents.config import (
    CHEAP_VISION,
    STRONG_VISION,
    TEXT_LAYER,
    load_approach_config,
    load_baseline_config,
)
from nfe_document_extraction.env import load_env_file
from nfe_document_extraction.evaluation.dataset import PHOTO_RENDITIONS, DatasetExample
from nfe_document_extraction.evaluation.metrics import (
    InvoiceEvaluation,
    ResultRow,
    not_applicable_row,
    summarize,
)
from nfe_document_extraction.evaluation.real_set import load_real_documents
from nfe_document_extraction.evaluation.results import merge_real_rows, render_results
from nfe_document_extraction.evaluation.tracing import (
    disable_tracing,
    tracing_is_disabled,
)

_ROOT = Path(__file__).resolve().parent.parent
_SECTION_APPROACHES = (CHEAP_VISION, STRONG_VISION, TEXT_LAYER)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_ROOT / "data" / "real",
        help="Real set directory outside version control (default: data/real).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "docs" / "results.md",
        help="Results file to update (default: docs/results.md).",
    )
    args = parser.parse_args(argv)

    load_env_file(_ROOT / ".env")
    disable_tracing()
    if not tracing_is_disabled():
        print(
            "Tracing is still enabled. Refusing to read the Real set.",
            file=sys.stderr,
        )
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and set the key.",
            file=sys.stderr,
        )
        return 2
    try:
        configs = (
            load_baseline_config(),
            *(load_approach_config(approach) for approach in _SECTION_APPROACHES),
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    # After tracing is off. Also lets `python scripts/run_real_eval.py` import this.
    from scripts.run_eval import collect_evaluations, snapshot_label

    try:
        documents = load_real_documents(args.dataset)
    except (FileNotFoundError, ValueError, ET.ParseError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if not documents:
        print(f"Real set is empty: {args.dataset}", file=sys.stderr)
        return 1

    client = OpenAI()
    scored = collect_evaluations(
        documents,
        client,
        configs,
        announce=_announce_without_identifiers,
    )
    rendered = render_results(
        _result_rows(tuple(evaluation for evaluation, _reading in scored), documents),
        model_snapshot=snapshot_label(configs),
        pricing_as_of=configs[0].pricing_as_of.isoformat(),
        generated_by="scripts/run_real_eval.py",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file():
        try:
            rendered = merge_real_rows(
                args.output.read_text(encoding="utf-8"),
                rendered,
            )
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return 0


def _announce_without_identifiers(
    approach: str, nfe_id: str, index: int, total: int
) -> None:
    del nfe_id
    print(f"{approach}: {index}/{total}", flush=True)


def _result_rows(
    evaluations: Sequence[InvoiceEvaluation],
    documents: Sequence[DatasetExample],
) -> tuple[ResultRow, ...]:
    rows = list(summarize(evaluations))
    origins = tuple(dict.fromkeys(document.origin for document in documents))
    photo_renditions = tuple(
        dict.fromkeys(
            document.rendition
            for document in documents
            if document.rendition in PHOTO_RENDITIONS
        )
    )
    for origin in origins:
        for rendition in photo_renditions:
            rows.append(not_applicable_row(TEXT_LAYER, rendition, origin))
    return tuple(rows)


if __name__ == "__main__":
    os.chdir(_ROOT)
    sys.exit(main())
