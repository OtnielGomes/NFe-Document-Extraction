#!/usr/bin/env python3
"""Run trigger evaluation for a skill description, on the Cursor runtime.

Tests whether a candidate description causes the Cursor agent to consult the
skill for a set of queries, and outputs the results as JSON.

How triggering is measured: the candidate description is written into a throwaway
skill inside a temporary workspace, an agent is pointed at that workspace with
project settings loaded, and its tool calls are watched. A tool call that
references the throwaway skill directory means the agent decided to consult it.

The throwaway skill lives in a temp directory rather than the real project so a
crashed or cancelled run can never leave a junk skill registered in the user's
Cursor. The cost is that the probe competes against no other skills; keep that in
mind when reading trigger rates for skills that overlap with an installed one.

Usage:
    python -m scripts.run_eval --eval-set queries.json --skill-path .cursor/skills/my-skill
    python -m scripts.run_eval --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.cursor_runtime import (
    DEFAULT_MODEL,
    RuntimeUnavailableError,
    ToolCall,
    observe_run,
)
from scripts.utils import parse_skill_md

PROBE_TIMEOUT_SECONDS = 60.0
DEFAULT_WORKERS = 5
DEFAULT_RUNS_PER_QUERY = 3
DEFAULT_TRIGGER_THRESHOLD = 0.5


def find_project_root() -> Path:
    """Find the workspace root by walking up from cwd looking for `.cursor/`.

    Returns:
        The nearest ancestor containing a `.cursor` directory, or the cwd.
    """
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".cursor").is_dir():
            return parent
    return current


def _render_probe_skill(probe_name: str, display_name: str, description: str) -> str:
    """Render the SKILL.md of the throwaway probe skill.

    The description goes into a YAML block scalar because real descriptions
    contain colons, quotes and commas that would break a plain scalar.

    Args:
        probe_name: Unique directory and frontmatter name for the probe.
        display_name: The real skill's name, used in the body for realism.
        description: The candidate description under test.

    Returns:
        The full SKILL.md contents.
    """
    indented = "\n  ".join(description.strip().splitlines())
    return (
        "---\n"
        f"name: {probe_name}\n"
        "description: |\n"
        f"  {indented}\n"
        "---\n\n"
        f"# {display_name}\n\n"
        f"This skill handles: {description.strip()}\n"
    )


def _write_probe_workspace(root: Path, skill_name: str, description: str) -> str:
    """Create a temporary workspace containing only the probe skill.

    Args:
        root: Directory that will act as the workspace root.
        skill_name: The real skill's name.
        description: The candidate description under test.

    Returns:
        The unique probe directory name, used as the detection sentinel.
    """
    probe_name = f"{skill_name}-probe-{uuid.uuid4().hex[:8]}"
    skill_dir = root / ".cursor" / "skills" / probe_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        _render_probe_skill(probe_name, skill_name, description), encoding="utf-8"
    )
    return probe_name


def run_single_query(
    query: str,
    skill_name: str,
    description: str,
    model: str,
    api_key: str | None,
    timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
) -> bool:
    """Run one query and report whether the agent consulted the probe skill.

    Args:
        query: The user prompt to send.
        skill_name: The real skill's name.
        description: The candidate description under test.
        model: Model identifier for the local run.
        api_key: Explicit API key, falling back to the environment.
        timeout_seconds: Deadline for the run.

    Returns:
        True when a tool call referenced the probe skill.
    """
    with tempfile.TemporaryDirectory(prefix="skill-trigger-") as tmp:
        root = Path(tmp)
        probe_name = _write_probe_workspace(root, skill_name, description)

        def consulted_probe(call: ToolCall) -> bool:
            return call.mentions(probe_name)

        observation = observe_run(
            query,
            cwd=root,
            model=model,
            api_key=api_key,
            setting_sources=("project",),
            timeout_seconds=timeout_seconds,
            stop_when=consulted_probe,
        )

        if observation.stopped_early:
            return True
        return any(call.mentions(probe_name) for call in observation.tool_calls)


def run_eval(
    eval_set: list[dict],
    skill_name: str,
    description: str,
    num_workers: int = DEFAULT_WORKERS,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    runs_per_query: int = DEFAULT_RUNS_PER_QUERY,
    trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> dict:
    """Run the full eval set and aggregate trigger rates per query.

    Args:
        eval_set: Items with a ``query`` string and a ``should_trigger`` boolean.
        skill_name: The real skill's name.
        description: The candidate description under test.
        num_workers: Concurrent runs. Each run is a separate agent, so this is
            bounded by rate limits rather than local CPU.
        timeout: Per-run deadline in seconds.
        runs_per_query: Repetitions per query, since triggering is stochastic.
        trigger_threshold: Trigger rate above which a query counts as triggered.
        model: Model identifier.
        api_key: Explicit API key, falling back to the environment.

    Returns:
        A dict with per-query results and a pass/fail summary.
    """
    query_triggers: dict[str, list[bool]] = {}
    query_items: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_item = {}
        for item in eval_set:
            for _ in range(runs_per_query):
                future = executor.submit(
                    run_single_query,
                    item["query"],
                    skill_name,
                    description,
                    model,
                    api_key,
                    timeout,
                )
                future_to_item[future] = item

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            query = item["query"]
            query_items[query] = item
            query_triggers.setdefault(query, [])
            try:
                query_triggers[query].append(future.result())
            except RuntimeUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad run must not kill the set
                print(f"Warning: query failed: {exc}", file=sys.stderr)
                query_triggers[query].append(False)

    results = []
    for query, triggers in query_triggers.items():
        item = query_items[query]
        trigger_rate = sum(triggers) / len(triggers)
        should_trigger = item["should_trigger"]
        did_pass = (
            trigger_rate >= trigger_threshold
            if should_trigger
            else trigger_rate < trigger_threshold
        )
        results.append(
            {
                "query": query,
                "should_trigger": should_trigger,
                "trigger_rate": trigger_rate,
                "triggers": sum(triggers),
                "runs": len(triggers),
                "pass": did_pass,
            }
        )

    passed = sum(1 for r in results if r["pass"])
    return {
        "skill_name": skill_name,
        "description": description,
        "results": results,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
        },
    }


def self_test(model: str, api_key: str | None) -> int:
    """Verify the trigger-detection plumbing end to end.

    Runs one query that must trigger and one that must not, against a probe skill
    with an unambiguous description. If the positive case fails, the problem is
    the plumbing rather than the description under test.

    Args:
        model: Model identifier.
        api_key: Explicit API key, falling back to the environment.

    Returns:
        Process exit code: 0 when both cases behave as expected.
    """
    description = (
        "Convert legacy COBOL copybook definitions into typed Python dataclasses. "
        "Use whenever the user mentions COBOL copybooks, EBCDIC record layouts, or "
        "migrating mainframe record definitions into Python."
    )
    cases = [
        (
            "i've got a folder of cobol copybooks from our mainframe team and i need "
            "them turned into python dataclasses with the right field widths. can you "
            "set up the conversion?",
            True,
        ),
        (
            "can you bump the version in package.json and write a changelog entry for "
            "the 2.1.0 release?",
            False,
        ),
    ]

    ok = True
    for query, should_trigger in cases:
        triggered = run_single_query(
            query, "cobol-copybook-converter", description, model, api_key
        )
        verdict = "OK" if triggered == should_trigger else "UNEXPECTED"
        if triggered != should_trigger:
            ok = False
        print(
            f"[{verdict}] triggered={triggered} expected={should_trigger}: {query[:60]}",
            file=sys.stderr,
        )

    if ok:
        print("Self-test passed: trigger detection is working.", file=sys.stderr)
        return 0

    print(
        "Self-test failed. Trigger rates from this environment are not trustworthy.",
        file=sys.stderr,
    )
    return 1


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run trigger evaluation for a skill description"
    )
    parser.add_argument("--eval-set", help="Path to eval set JSON file")
    parser.add_argument("--skill-path", help="Path to skill directory")
    parser.add_argument("--description", default=None, help="Override description to test")
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_WORKERS, help="Concurrent runs"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=PROBE_TIMEOUT_SECONDS,
        help="Per-run deadline in seconds",
    )
    parser.add_argument(
        "--runs-per-query", type=int, default=DEFAULT_RUNS_PER_QUERY, help="Runs per query"
    )
    parser.add_argument(
        "--trigger-threshold",
        type=float,
        default=DEFAULT_TRIGGER_THRESHOLD,
        help="Trigger rate threshold",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model for the eval runs")
    parser.add_argument("--api-key", default=None, help="Override CURSOR_API_KEY")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Verify trigger detection works before trusting any numbers",
    )
    args = parser.parse_args()

    try:
        if args.self_test:
            raise SystemExit(self_test(args.model, args.api_key))

        if not args.eval_set or not args.skill_path:
            parser.error("--eval-set and --skill-path are required unless --self-test")

        skill_path = Path(args.skill_path)
        if not (skill_path / "SKILL.md").exists():
            print(f"Error: No SKILL.md found at {skill_path}", file=sys.stderr)
            raise SystemExit(1)

        eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
        name, original_description, _ = parse_skill_md(skill_path)
        description = args.description or original_description

        if args.verbose:
            print(f"Evaluating: {description}", file=sys.stderr)

        output = run_eval(
            eval_set=eval_set,
            skill_name=name,
            description=description,
            num_workers=args.num_workers,
            timeout=args.timeout,
            runs_per_query=args.runs_per_query,
            trigger_threshold=args.trigger_threshold,
            model=args.model,
            api_key=args.api_key,
        )

        if args.verbose:
            summary = output["summary"]
            print(f"Results: {summary['passed']}/{summary['total']} passed", file=sys.stderr)
            for r in output["results"]:
                status = "PASS" if r["pass"] else "FAIL"
                print(
                    f"  [{status}] rate={r['triggers']}/{r['runs']} "
                    f"expected={r['should_trigger']}: {r['query'][:70]}",
                    file=sys.stderr,
                )

        print(json.dumps(output, indent=2))

    except RuntimeUnavailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
