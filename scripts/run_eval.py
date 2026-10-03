"""Run the local eval and write docs/results.md. No LangSmith account is required."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

from nfe_document_extraction.agents.baseline import extract_baseline
from nfe_document_extraction.agents.config import load_baseline_config
from nfe_document_extraction.evaluation import render_results, summarize, token_cost_usd
from nfe_document_extraction.evaluation.dataset import load_synthetic_pdfs
from nfe_document_extraction.evaluation.judge import judge_reading

_ROOT = Path(__file__).resolve().parent.parent
_RENDITION = "digital PDF"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_ROOT / "data" / "synthetic",
        help="Synthetic set directory (default: data/synthetic).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "docs" / "results.md",
        help="Results file to write (default: docs/results.md).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N documents.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    config = load_baseline_config()
    documents = load_synthetic_pdfs(args.dataset)
    if args.limit is not None:
        documents = documents[: args.limit]
    client = OpenAI()
    evaluations = []
    total = len(documents)
    for index, document in enumerate(documents, start=1):
        print(f"{document.nfe_id} ({index}/{total})", flush=True)
        reading = extract_baseline(client, document.pdf, config=config)
        evaluations.append(
            judge_reading(
                approach=config.approach,
                rendition=_RENDITION,
                origin=document.origin,
                ground_truth=document.ground_truth,
                extraction=reading.extraction,
                extraction_failure=reading.failure,
                planted_rule=document.planted_rule,
                cost_usd=token_cost_usd(
                    reading.input_tokens,
                    reading.cached_input_tokens,
                    reading.output_tokens,
                    config,
                ),
                latency_seconds=reading.latency_seconds,
            )
        )
    text = render_results(
        summarize(evaluations),
        model_snapshot=config.model,
        pricing_as_of=config.pricing_as_of.isoformat(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    os.chdir(_ROOT)
    sys.exit(main())
