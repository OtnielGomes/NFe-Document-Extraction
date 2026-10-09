# Cursor adaptations

This skill was ported from Anthropic's [skill-creator](https://github.com/anthropics/skills/tree/main/skills/skill-creator), which was written for Claude Code. This file records what changed and why. Read it when running evals, packaging, or optimizing a description.

## Contents

- [Skill storage paths](#skill-storage-paths)
- [Tool mapping](#tool-mapping)
- [Eval workspace layout](#eval-workspace-layout)
- [Task prompt templates](#task-prompt-templates)
- [Shell command mapping](#shell-command-mapping)
- [Script compatibility](#script-compatibility)
- [Description optimization runtime](#description-optimization-runtime)

## Skill storage paths

| Scope | Path |
|-------|------|
| Project | `.cursor/skills/<skill-name>/` |
| Personal | `~/.cursor/skills/<skill-name>/` (Windows: `%USERPROFILE%/.cursor/skills/`) |
| Legacy CLI install | `.agents/skills/<skill-name>/` (still discovered) |

Never write to `~/.cursor/skills-cursor/`. That directory holds Cursor's built-in skills and is managed automatically; anything you put there can be overwritten without warning.

## Tool mapping

| Claude Code | Cursor |
|-------------|--------|
| Subagent spawn | `Task` tool with `subagent_type` (`generalPurpose`, `explore`, `shell`) |
| `Bash` | `Shell` |
| `Edit` | `StrReplace` |
| `Read` / `Write` | `Read` / `Write` (unchanged) |
| `present_files` | No equivalent — save the file and tell the user the path |
| `/skill-test` command | No equivalent — the eval loop in SKILL.md replaces it |
| `.claude/commands/` | Not used — skills live in `.cursor/skills/` |
| `claude -p` subprocess | Cursor SDK via `scripts/cursor_runtime.py` |

Two behavioral differences matter more than the names:

- **Parallelism comes from batching.** Multiple `Task` calls in a single message run concurrently. Spread across messages, they serialize, and a with-skill run that finishes before you launch its baseline is a contaminated comparison.
- **Subagents start blank.** They don't see the conversation. Every eval prompt must carry its own context: the full task, input paths, and the output directory.

## Eval workspace layout

Create the workspace as a sibling of the skill directory:

```
.cursor/skills/my-skill/
.cursor/skills/my-skill-workspace/
└── iteration-1/
    ├── benchmark.json
    ├── review.html
    ├── feedback.json
    └── extracts-tables-from-scanned-pdf/
        ├── eval_metadata.json
        ├── with_skill/
        │   ├── outputs/
        │   ├── transcript.md
        │   ├── timing.json
        │   └── grading.json
        └── without_skill/          # or old_skill/ when improving a skill
            ├── outputs/
            ├── transcript.md
            └── grading.json
```

Name eval directories after what they test, not `eval-0`. The name becomes a section header in the viewer, and "extracts-tables-from-scanned-pdf" tells the reviewer what they're looking at while "eval-2" makes them go digging.

## Task prompt templates

### With-skill run

```
Execute this eval task independently. Assume no prior conversation context.

- Skill path: .cursor/skills/<skill-name>/SKILL.md
- Read the skill first and follow its instructions.
- Task: <eval prompt>
- Input files: <paths, or "none">
- Save all deliverables to: <workspace>/iteration-<N>/<eval-name>/with_skill/outputs/
- Write transcript.md in <workspace>/iteration-<N>/<eval-name>/with_skill/ summarizing
  the steps you took, the tools you used, and anything that went wrong.
- Outputs to save: <what matters — e.g. "the generated .py file", "the final CSV">
```

### Baseline run

Identical, with two changes: remove the skill path and the instruction to read it, and save to `without_skill/outputs/`. When improving an existing skill, point at the snapshot instead and save to `old_skill/outputs/`.

The baseline prompt must be otherwise word-for-word identical. Any difference in phrasing turns the measurement into noise.

### Grader run

```
Read .cursor/skills/skill-creator/agents/grader.md and follow it.

- expectations: <list of assertions>
- transcript_path: <path to transcript.md>
- outputs_dir: <path to outputs/>

Write grading.json as a sibling of outputs_dir. The expectations array must use the
field names text, passed, evidence — the eval viewer reads those exact names.
```

## Shell command mapping

The upstream skill assumed a POSIX shell. On Windows:

| POSIX | PowerShell |
|-------|------------|
| `cp -r src dst` | `Copy-Item -Recurse src dst` |
| `/tmp/file` | `$env:TEMP/file` |
| `nohup cmd &` | `Start-Process -NoNewWindow cmd` |
| `kill $PID` | `Stop-Process -Id $PID` |
| `open file.html` | `Invoke-Item file.html` |
| `lsof -ti :3117` | `netstat -ano` (handled internally by `generate_review.py`) |

Line continuation is a backtick, not a backslash. Use forward slashes in paths regardless — PowerShell and Python both accept them, and backslashes get eaten as escapes when the path passes through Python or JSON.

## Script compatibility

| Script | Status | Notes |
|--------|--------|-------|
| `scripts/quick_validate.py` | Works | Also accepts Cursor's `disable-model-invocation` field |
| `scripts/package_skill.py` | Works | Validates before zipping |
| `scripts/aggregate_benchmark.py` | Works | Pure stdlib |
| `scripts/generate_report.py` | Works | Pure stdlib HTML generation |
| `eval-viewer/generate_review.py` | Works | Port-freeing ported to `netstat`/`taskkill`; prefer `--static` on Windows |
| `scripts/cursor_runtime.py` | New, optional | Cursor SDK adapter; run it directly for an environment check |
| `scripts/run_eval.py` | Ported, optional | Trigger detection rewritten on the SDK |
| `scripts/improve_description.py` | Ported, optional | Same meta-prompt, SDK transport |
| `scripts/run_loop.py` | Ported, optional | Orchestration unchanged |

Install the base dependency once:

```powershell
pip install -r .cursor/skills/skill-creator/requirements.txt
```

The four scripts marked optional additionally need `pip install "cursor-sdk>=1.0.28"` and a `CURSOR_API_KEY`. They are not part of the default workflow: description calibration is a manual loop unless the user asks for the automated one.

## Description optimization runtime

This whole path is opt-in. The default workflow in SKILL.md calibrates a description through manual rounds in fresh chats, which needs no SDK, no API key and no tokens beyond the conversation itself. What follows describes the automated alternative and why it had to be rebuilt rather than translated.

Upstream detected triggering by writing a temporary command file into `.claude/commands/`, running `claude -p --output-format stream-json`, and scraping the event stream for a `Skill` or `Read` tool call. Three things about that don't survive the move to Cursor: there is no `.claude/commands/`, there is no `claude` binary, and the stream reader used `select.select()` on pipes, which on Windows only works with sockets.

The port replaces all of it with the Cursor SDK:

1. `run_eval.py` writes the candidate description into a throwaway skill inside a temporary workspace.
2. It runs a local agent against that workspace with `setting_sources=["project"]`, so the throwaway skill is discovered the way a real one would be.
3. It watches the typed `tool_call` messages. A tool call that references the throwaway directory means the agent chose to consult the skill; detection cancels the run immediately, so a positive costs a few seconds instead of a full completion.

Two consequences worth knowing:

- **Isolation is deliberate.** The probe lives in a temp directory so a crashed run can't leave a junk skill registered in the user's Cursor. The tradeoff is that the candidate competes against no other skills, so reported trigger rates are an upper bound for skills that overlap with something already installed.
- **`RunResult` carries `duration_ms` and `usage.total_tokens`.** Unlike the `Task` tool, SDK runs report timing and token usage reliably, so anything driven through `cursor_runtime.py` can populate `timing.json` for real.

Requirements, when you do opt in: `cursor-sdk` installed and `CURSOR_API_KEY` set. Verify both, plus the detection plumbing, before trusting any numbers:

```powershell
python -m scripts.cursor_runtime
python -m scripts.run_eval --self-test
```

If the SDK or the key is missing, the scripts fail with an explicit message rather than a stack trace. Everything else in this skill — drafting, evals, grading, benchmarking, the viewer, packaging — runs without them.
