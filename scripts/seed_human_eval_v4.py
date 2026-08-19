"""Seed the human-evaluation web app with answers from an agent eval run.

Reads an agent-eval results file (e.g. eval/cache/eval_v4_agent_all_*.json,
whose ``results[].run`` entries carry the raw agent output) and registers
each question as a completed run in the human-evaluation store, owned by the
``seed:eval-v4`` system user and published by default so anonymous visitors
can read — and signed-in users can review — the answers immediately.

Idempotent: each seed run uses ``client_request_id = "eval-v4:<question id>"``,
so rerunning the script skips questions that are already seeded.

Usage:
    uv run python scripts/seed_human_eval_v4.py \
        --agent-results eval/cache/eval_v4_agent_all_kimi_k27_v9.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from human_eval.agent_executor import _public_result
from human_eval.contracts import RunRequest
from human_eval.store import EvaluationStore

SEED_USER = "seed:eval-v4"


def seed_run(store: EvaluationStore, item: dict, *, publish: bool) -> str:
    """Create one seeded run; returns 'created', 'skipped', or 'failed'."""
    question_id = item["id"]
    run = item.get("run") or {}
    request = RunRequest(
        question=item["question"],
        as_of=item.get("as_of"),
        required_hops=max(1, min(5, int(item.get("required_hops") or 1))),
        client_request_id=f"eval-v4:{question_id}",
    )
    provenance = {
        "seeded_from": "eval_v4_agent_results",
        "source_eval_id": question_id,
        "category": item.get("category"),
        "corpus_version": "dof-full-v1",
        "chunker_version": "dof-chunker-v1",
        "model": run.get("model"),
        "configuration": {
            "retrieval_mode": "lexical",
            "max_model_turns": run.get("model_turns"),
            "max_tool_calls": run.get("tool_calls"),
        },
    }
    record, created = store.create_run(
        request, user_id=SEED_USER, provenance=provenance
    )
    if not created:
        return "skipped"
    store.append_event(record["run_id"], "started")
    if run.get("answer", {}).get("answer"):
        store.append_event(record["run_id"], "succeeded", _public_result(run))
        if publish:
            store.publish_run(record["run_id"], publisher_id=SEED_USER)
        return "created"
    store.append_event(
        record["run_id"],
        "failed",
        {
            "code": "seeded_incomplete_run",
            "message": "La ejecución original no produjo respuesta.",
        },
    )
    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-results", type=Path, required=True)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("var/human_evaluation.sqlite"),
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="create the runs but leave them unpublished",
    )
    args = parser.parse_args()

    data = json.loads(args.agent_results.read_text())
    results = data.get("results")
    if not isinstance(results, list):
        raise SystemExit(f"{args.agent_results}: no 'results' list found")

    store = EvaluationStore(args.db)
    store.initialize()
    counts = {"created": 0, "skipped": 0, "failed": 0}
    for item in results:
        outcome = seed_run(store, item, publish=not args.no_publish)
        counts[outcome] += 1
        print(f"{item['id']}: {outcome}")
    print(
        f"done: {counts['created']} created, {counts['skipped']} already "
        f"present, {counts['failed']} without answer"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
