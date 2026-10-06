# Deploy and the manual LLM eval

Pushes to `main` run `.github/workflows/checks.yml`. When that run succeeds, `.github/workflows/deploy-space.yml` publishes the container to a Hugging Face Docker Space on free CPU (`cpu-basic`). Deploys run one at a time. A run publishes only when its commit is still the tip of `main`, so an older green check cannot replace a newer one. Free CPU sleeps after 48 hours of inactivity and starts again when someone opens the Space. The gallery is cached in the image, so that wake does not call a model.

The Space README (title, `sdk: docker`, `app_port: 7860`) is written at publish time. The repository README is still written last, with the case study.

## Repository secrets

Create these Actions secrets. The workflows read them from GitHub. Nothing in the repository stores the values. The publish script copies the OpenAI and LangSmith keys onto the Space as secrets; the Hugging Face token only authenticates the publish.

| Secret | Use |
| --- | --- |
| `HF_TOKEN` | Hugging Face token with write access to the Space |
| `HF_SPACE_ID` | Space id, `username/nfe-document-extraction` |
| `OPENAI_API_KEY` | Demo uploads and the manual LLM eval |
| `LANGSMITH_API_KEY` | Demo metadata traces, dataset sync, and the experiment upload |

The script creates the Space when `HF_SPACE_ID` does not exist yet, as a public Docker Space on `cpu-basic`. Each later publish keeps that hardware, so the Space keeps sleeping when it is idle.

## Manual step: OpenAI budget

Set a monthly spend limit of **US$ 5** on the OpenAI project that owns `OPENAI_API_KEY`, and turn on **Enforce a hard limit**. This is a manual step in the OpenAI dashboard. This repository cannot set it.

A limit without enforcement only sends an alert; API calls continue. The hard limit is the last barrier: after the project reaches US$ 5 in the calendar month, OpenAI rejects further requests. Organization and project hard limits can both apply. Set the project that this key bills.

Steps are in the [OpenAI spend limits guide](https://developers.openai.com/api/docs/guides/spend-limits): open the project limits, edit the monthly spend limit, enter 5 USD, enable the hard limit, and save.

## Manual LLM eval

`.github/workflows/llm-eval.yml` runs only from the Actions tab (**Run workflow**). It does not run on push or pull request. The run syncs the Synthetic set, runs `scripts/run_eval.py` (no `--limit`), uploads the LangSmith experiment, and stores `docs/results.md` as the artifact `synthetic-eval-results`.

That run spends the OpenAI budget. The US$ 5 hard limit above is what stops it. The Real set stays local; this workflow does not run `scripts/run_real_eval.py`.
