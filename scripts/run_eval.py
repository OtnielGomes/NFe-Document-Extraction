"""Run the local eval and write docs/results.md.

LangSmith is optional. When LANGSMITH_API_KEY is set, the same run is uploaded
as an experiment. ``--trace`` also traces model calls. ``--no-trace`` sends nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from langsmith import Client
from openai import OpenAI

from nfe_document_extraction.agents.baseline import extract_baseline
from nfe_document_extraction.agents.config import (
    CHEAP_VISION,
    STRONG_VISION,
    TEXT_LAYER,
    ApproachConfig,
    load_approach_config,
    load_baseline_config,
)
from nfe_document_extraction.agents.sections import (
    SectionReader,
    TextLayerSectionReader,
    VisionSectionReader,
)
from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.env import load_env_file
from nfe_document_extraction.evaluation import render_results, summarize, token_cost_usd
from nfe_document_extraction.evaluation.dataset import (
    PHONE_PHOTO,
    PHOTO_RENDITIONS,
    SIMULATED_PHOTO,
    DatasetExample,
    load_synthetic_pdfs,
)
from nfe_document_extraction.evaluation.judge import judge_reading
from nfe_document_extraction.evaluation.langsmith import (
    DatasetNotSynced,
    ScoredReading,
    prediction_outputs,
    repository_commit,
    upload_recorded_run,
)
from nfe_document_extraction.evaluation.metrics import (
    ExtractionFailure,
    InvoiceEvaluation,
    ResultRow,
    not_applicable_row,
)
from nfe_document_extraction.evaluation.tracing import resolve_tracing
from nfe_document_extraction.graph import compile_document_graph, run_document
from nfe_document_extraction.validation import DocumentOutcome

_ROOT = Path(__file__).resolve().parent.parent
_SECTION_APPROACHES = (CHEAP_VISION, STRONG_VISION, TEXT_LAYER)
_PHOTO_ROWS = (SIMULATED_PHOTO, PHONE_PHOTO)


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
        help=(
            "Evaluate only the first N dataset examples. "
            "Examples are ordered by NF-e, digital PDF then simulated photo."
        ),
    )
    tracing_flags = parser.add_mutually_exclusive_group()
    tracing_flags.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Trace model calls as well as uploading the experiment. "
            "Requires a LangSmith key."
        ),
    )
    tracing_flags.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable tracing for this run. No data is sent to LangSmith.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    load_env_file(_ROOT / ".env")
    tracing = resolve_tracing(trace=args.trace, no_trace=args.no_trace)
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

    documents = load_synthetic_pdfs(args.dataset)
    if args.limit is not None:
        documents = documents[: args.limit]
    client = _model_client(tracing.trace_models)
    scored = collect_evaluations(documents, client, configs)
    text = render_results(
        _result_rows(tuple(evaluation for evaluation, _ in scored), documents),
        model_snapshot=_snapshot_label(configs),
        pricing_as_of=configs[0].pricing_as_of.isoformat(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    if not tracing.upload:
        return 0
    try:
        upload_recorded_run(
            Client(),
            tuple(reading for _, reading in scored),
            documents,
            commit=repository_commit(_ROOT),
        )
    except DatasetNotSynced as exc:
        print(
            f"LangSmith dataset {exc.name!r} is missing. Run scripts/sync_dataset.py.",
            file=sys.stderr,
        )
        return 1
    return 0


def _model_client(tracing_on: bool) -> OpenAI:
    client = OpenAI()
    if not tracing_on:
        return client
    return wrap_openai(client)


def wrap_openai(client: OpenAI) -> OpenAI:
    """Trace the OpenAI client. Tests replace this; model calls are not traced here."""
    from langsmith.wrappers import wrap_openai as _wrap

    return _wrap(client)


def collect_evaluations(
    documents: Sequence[DatasetExample],
    client: OpenAI,
    configs: Sequence[ApproachConfig],
) -> tuple[tuple[InvoiceEvaluation, ScoredReading], ...]:
    """Score the baseline, both vision approaches, and the text-layer approach."""
    baseline, *section_configs = configs
    scored: list[tuple[InvoiceEvaluation, ScoredReading]] = []
    scored.extend(_baseline(documents, client, baseline))
    for config in section_configs:
        scored.extend(_section_agents(documents, client, config))
    return tuple(scored)


def _baseline(
    documents: Sequence[DatasetExample],
    client: OpenAI,
    config: ApproachConfig,
) -> list[tuple[InvoiceEvaluation, ScoredReading]]:
    scored: list[tuple[InvoiceEvaluation, ScoredReading]] = []
    total = len(documents)
    for index, document in enumerate(documents, start=1):
        print(f"{config.approach}: {document.nfe_id} ({index}/{total})", flush=True)
        reading = extract_baseline(
            client,
            document.pdf,
            config=config,
            pages=document.pages,
        )
        scored.append(
            _scored(
                document,
                config.approach,
                judge_reading(
                    approach=config.approach,
                    rendition=document.rendition,
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
                ),
                reading.extraction,
                reading.failure,
            )
        )
    return scored


def _result_rows(
    evaluations: Sequence[InvoiceEvaluation],
    documents: Sequence[DatasetExample],
) -> tuple[ResultRow, ...]:
    """Measured rows, plus photo rows the text-layer approach cannot run."""
    rows = list(summarize(evaluations))
    origins = tuple(dict.fromkeys(document.origin for document in documents))
    for origin in origins:
        for rendition in _PHOTO_ROWS:
            rows.append(not_applicable_row(TEXT_LAYER, rendition, origin))
    return tuple(rows)


def _section_reader(client: OpenAI, config: ApproachConfig) -> SectionReader:
    if config.approach == TEXT_LAYER:
        return TextLayerSectionReader(client, config)
    return VisionSectionReader(client, config)


def _documents_for_approach(
    config: ApproachConfig,
    documents: Sequence[DatasetExample],
) -> Sequence[DatasetExample]:
    if config.approach != TEXT_LAYER:
        return documents
    return tuple(
        document for document in documents if document.rendition not in PHOTO_RENDITIONS
    )


def _section_agents(
    documents: Sequence[DatasetExample],
    client: OpenAI,
    config: ApproachConfig,
) -> list[tuple[InvoiceEvaluation, ScoredReading]]:
    reader = _section_reader(client, config)
    selected = _documents_for_approach(config, documents)
    graph = compile_document_graph(reader)
    scored: list[tuple[InvoiceEvaluation, ScoredReading]] = []
    total = len(selected)
    for index, document in enumerate(selected, start=1):
        print(f"{config.approach}: {document.nfe_id} ({index}/{total})", flush=True)
        run = run_document(
            reader,
            document.pdf,
            graph=graph,
            pages=document.pages,
        )
        initial_rules = None if run.initial_report is None else run.initial_report.rules
        scored.append(
            _scored(
                document,
                config.approach,
                judge_reading(
                    approach=config.approach,
                    rendition=document.rendition,
                    origin=document.origin,
                    ground_truth=document.ground_truth,
                    extraction=run.extraction,
                    extraction_failure=run.extraction_failure,
                    planted_rule=document.planted_rule,
                    cost_usd=token_cost_usd(
                        run.input_tokens,
                        run.cached_input_tokens,
                        run.output_tokens,
                        config,
                    ),
                    latency_seconds=run.latency_seconds,
                    repaired=run.repair_attempted
                    and run.outcome is DocumentOutcome.ACCEPTED,
                    reached_human_review=run.outcome is DocumentOutcome.HUMAN_REVIEW,
                    initial_rules=initial_rules,
                ),
                run.extraction,
                run.extraction_failure,
            )
        )
    return scored


def _scored(
    document: DatasetExample,
    approach: str,
    evaluation: InvoiceEvaluation,
    extraction: Extraction | None,
    failure: ExtractionFailure | None,
) -> tuple[InvoiceEvaluation, ScoredReading]:
    return evaluation, ScoredReading(
        approach=approach,
        nfe_id=document.nfe_id,
        input_type=document.input_type,
        outputs=prediction_outputs(
            extraction,
            None if failure is None else failure.value,
        ),
    )


def _snapshot_label(configs: Sequence[ApproachConfig]) -> str:
    grouped: dict[str, list[str]] = {}
    for config in configs:
        grouped.setdefault(config.model, []).append(config.approach)
    return "; ".join(
        f"{model} ({', '.join(names)})" for model, names in grouped.items()
    )


if __name__ == "__main__":
    os.chdir(_ROOT)
    sys.exit(main())
