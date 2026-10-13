# NFe Document Extraction

**English** · [Português](README.pt-BR.md)

The model reads a Brazilian DANFE. The code decides whether that reading can be trusted. Header and line-item agents extract the invoice; a deterministic validator, one repair attempt, and human review accept or hold each field. Numbers below come only from the generated table in [`docs/results.md`](docs/results.md).

> [!NOTE]
> This repository is still in development. Evaluation, tracing, and monitoring already run through LangSmith. The public demo is not deployed yet. A later push to `main` is planned to publish a Hugging Face Space; see [`docs/deploy.md`](docs/deploy.md).

## Results

Sample of 10 synthetic invoices per row (`synthetic-001` through `synthetic-010`), digital PDF then simulated photo. Adopted cheap vision is `gpt-6-luna` (Responses API, reasoning effort none, temperature 0) with the structured Header and Line items prompts ([ADR 0008](docs/adr/0008-cheap-vision-adopts-gpt-6-luna.md)). Pricing table date: 2026-10-08. The full generated table, including strong vision and the text layer, is in [`docs/results.md`](docs/results.md).

| Extraction approach | Rendition | Field accuracy | Cost per invoice (USD) | Errors reaching Human review | Planted-error detection rate |
| --- | --- | --- | --- | --- | --- |
| single-call baseline | digital PDF | 0.9322 | 0.003345 | 0 | 1.0000 |
| single-call baseline | simulated photo | 0.5677 | 0.003186 | 0 | 1.0000 |
| cheap vision | digital PDF | 0.9870 | 0.002035 | 8 | 1.0000 |
| cheap vision | simulated photo | 0.6420 | 0.002367 | 9 | 1.0000 |
| text layer + text model | digital PDF | 0.9870 | 0.004457 | 5 | 1.0000 |

On this sample, cheap vision matches the text layer on digital field accuracy (0.9870) at a lower cost (US$ 0.002035 versus US$ 0.004457) and reads simulated photos, which the text layer cannot. The single-call baseline is cheaper to review (0 documents sent to human review on digital PDF) because it does not repair, and its digital field accuracy is lower (0.9322). Repair fixed 0 documents in every published row. Planted-error detection is 1.0000 and clean documents rejected is 0.

![Field accuracy and review counts before deploy](docs/images/5-results-final-configs-before-deploy.png)

## How evaluation chose the model

LangSmith is the platform for tracing, offline experiments, and monitoring. The eval itself stays local: `scripts/run_eval.py` scores with pure Python and writes `docs/results.md` with no LangSmith account. When `LANGSMITH_API_KEY` is set, the same run uploads as an experiment.

The cheap-vision decision was an eval → trace → change → eval cycle on the restarted sample in the table (10 documents per rendition). An earlier pass over the full synthetic set was not used. The adopted row is step 2: `gpt-6-luna` plus the structured prompts. On that row, digital cost stays under the comparison cap recorded in [ADR 0008](docs/adr/0008-cheap-vision-adopts-gpt-6-luna.md), planted-error detection stays 1.0000, and no clean document is rejected. Simulated-photo human review is 9. The write-up of each measured step is in the [case study](docs/case-study.md) and [ADR 0008](docs/adr/0008-cheap-vision-adopts-gpt-6-luna.md). Only figures that also appear in `docs/results.md` are repeated here.

## Run

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/check.py
```

`scripts/check.py` runs the formatter, the linter, the type checker, and the test suite. It does not call a model.

Copy [`.env.example`](.env.example) to `.env` and set `OPENAI_API_KEY` before an eval. LangSmith is optional.

```bash
uv run python scripts/run_eval.py --limit 2
```

`--limit 2` scores the first examples (one NF-e, digital PDF then simulated photo) and rewrites `docs/results.md`. Omit `--limit` for the full synthetic set. Add `--trace` to trace model calls. `scripts/sync_dataset.py` uploads the synthetic set when a LangSmith key is present.

The local UI (Portuguese by default, English on the page) and JSON API:

```bash
uv run python -m nfe_document_extraction.app
```

Open `http://127.0.0.1:7860`. Uploads call the model and need `OPENAI_API_KEY`.

## Case study

Architecture, the baseline comparison, validator and repair counts, limitations, and next steps: [`docs/case-study.md`](docs/case-study.md).
