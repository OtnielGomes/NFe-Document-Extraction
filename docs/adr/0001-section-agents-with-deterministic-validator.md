# Section agents in a LangGraph graph, validator as a deterministic node

Extraction is split between two Section agents (Header and Line items) orchestrated by a LangGraph graph. The validator, the repair routing, and the acceptance decision are plain code nodes, never agents: the thesis is that the model reads and the code decides whether the reading can be trusted. Human review uses LangGraph `interrupt()` with an in-memory or SQLite checkpointer, and human edits pass through the same Validation rules as model output.

## Consequences

- Orchestration is a deterministic fan-out: both Section agents receive the whole document in parallel, a merge node combines their output, and no supervisor LLM ever decides the flow.
- Repair routes each Flagged field back to the agent that owns it. A rule spanning both sections (sum of line items equals total products) sends the same message to both agents, which repair in parallel within the single Repair attempt. There is no reconciliation agent: an agent that sees the whole sum is the one that adjusts numbers until it closes.
- The eval includes a Single-call baseline (cheap model, Synthetic set) so the split is justified by data, not assumed.
