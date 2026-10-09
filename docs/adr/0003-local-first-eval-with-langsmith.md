# Local-first evaluation, LangSmith as the observability platform

LangSmith is the single platform for tracing, offline evaluation, and monitoring. The eval still runs fully offline: evaluators are pure Python functions, a single script produces the results table without any LangSmith account, and when a LangSmith key is present the same run is uploaded as an experiment. This keeps the published eval reproducible by anyone who clones the repository.

## Consequences

- The repository is the source of truth for the dataset; a sync script uploads it to LangSmith, and every experiment records the code commit and dataset version.
- Prompts live in the repository, not in a remote prompt hub, so neither the demo nor the eval needs LangSmith to run.
- The Real set always runs with tracing disabled; only aggregated results leave the machine. OpenAI is the only third party that sees real invoices.
- The public demo traces with inputs and outputs hidden: only metadata, cost, latency, tokens, failing rules, and review outcome reach LangSmith. Human corrections are recorded as run feedback. The cached gallery produces no traces.
- CI runs the deterministic validator regression on every push; the LLM eval runs only on manual dispatch because of the US$ 5/month spend cap.
