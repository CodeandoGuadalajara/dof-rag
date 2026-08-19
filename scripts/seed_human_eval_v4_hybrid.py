"""Seed the human-evaluation app with LIVE agent runs for eval v4.

Unlike ``seed_human_eval_v4.py`` (which imports frozen eval JSON and can
therefore never include the "Proceso de investigación" timeline), this script
drives real ``AgentRunExecutor`` runs so every seeded answer carries its full
progress timeline, hybrid retrieval evidence, and live provenance.

Idempotent: each seed run uses
``client_request_id = "eval-v4-hybrid:<question id>"`` owned by the
``seed:eval-v4`` system user. ``--replace`` deletes existing ``seed:`` runs
first (the store keeps deletion restricted to seed users).

Usage:
    set -a; source .env; set +a
    export DOF_AGENT_PROVIDER=kimi-code DOF_AGENT_MODEL=kimi-for-coding \
        DOF_RETRIEVAL_MODE=hybrid
    uv run python scripts/seed_human_eval_v4_hybrid.py --replace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from human_eval.agent_executor import AgentExecutorConfig, AgentRunExecutor
from human_eval.contracts import RunRequest
from human_eval.service import PublicExecutionError
from human_eval.store import EvaluationStore

SEED_USER = "seed:eval-v4"


def load_queries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed_live_run(
    store: EvaluationStore,
    executor: AgentRunExecutor,
    item: dict[str, Any],
    *,
    publish: bool,
) -> str:
    """Run one question live; return 'created', 'skipped', 'failed', or 'abort'."""
    question_id = item["id"]
    request = RunRequest(
        question=item["question"],
        as_of=item.get("as_of"),
        required_hops=max(1, min(5, int(item.get("required_hops") or 1))),
        client_request_id=f"eval-v4-hybrid:{question_id}",
    )
    existing = store.find_idempotent_run(SEED_USER, request.client_request_id)
    if existing is not None:
        if existing["status"] == "succeeded":
            if publish and existing["published_at"] is None:
                store.publish_run(existing["run_id"], publisher_id=SEED_USER)
            return "skipped"
        if existing["status"] == "failed":
            store.delete_seed_run(existing["run_id"], user_prefix=SEED_USER)
        else:
            # Do not delete a queued/running run owned by another process.
            return "pending"
    record, created = store.create_run(
        request, user_id=SEED_USER, provenance=executor.provenance()
    )
    if not created:
        return "skipped"
    run_id = record["run_id"]
    store.append_event(run_id, "started")
    try:
        result = executor.execute(
            request,
            on_progress=lambda event_type, payload: store.append_progress(
                run_id, event_type, payload
            ),
        )
    except PublicExecutionError as exc:
        store.append_event(run_id, "failed", {"code": exc.code, "message": str(exc)})
        # provider_unavailable is session-wide (auth/quota); stop the batch.
        return "abort" if exc.code == "provider_unavailable" else "failed"
    except Exception as exc:
        store.append_event(
            run_id,
            "failed",
            {
                "code": "internal_error",
                "message": "La ejecución no pudo completarse.",
            },
        )
        print(f"  internal error: {type(exc).__name__}: {exc}", flush=True)
        return "failed"
    store.append_event(run_id, "succeeded", result)
    if publish:
        store.publish_run(run_id, publisher_id=SEED_USER)
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", type=Path, default=Path("eval/dof_queries_v4.jsonl")
    )
    parser.add_argument(
        "--db", type=Path, default=Path("var/human_evaluation.sqlite")
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--ids", help="Comma-separated v4 question IDs to seed")
    parser.add_argument(
        "--replace",
        action="store_true",
        help=f"delete existing {SEED_USER} runs before seeding",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="create the runs but leave them unpublished",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries)
    if args.ids:
        wanted = {value.strip() for value in args.ids.split(",")}
        queries = [item for item in queries if item["id"] in wanted]
        missing = wanted - {item["id"] for item in queries}
        if missing:
            parser.error(f"unknown query ids: {sorted(missing)}")

    store = EvaluationStore(args.db)
    store.initialize()
    config = AgentExecutorConfig.from_env(args.repo_root)
    executor = AgentRunExecutor(config)
    # Pre-warm the embedding server so provenance records vector_used from the
    # first run on (the embedder otherwise starts lazily mid-run).
    executor.query_embedder()
    try:
        if args.replace:
            deleted = store.delete_seed_runs(user_prefix=SEED_USER)
            print(f"--replace: deleted {deleted} existing {SEED_USER} runs", flush=True)
        counts = {"created": 0, "skipped": 0, "pending": 0, "failed": 0}
        for index, item in enumerate(queries, 1):
            outcome = seed_live_run(store, executor, item, publish=not args.no_publish)
            if outcome == "abort":
                counts["failed"] += 1
                print(
                    f"[{index}/{len(queries)}] {item['id']}: failed "
                    "(provider unavailable; aborting batch)",
                    flush=True,
                )
                break
            counts[outcome] += 1
            print(f"[{index}/{len(queries)}] {item['id']}: {outcome}", flush=True)
        print(
            f"done: {counts['created']} created, {counts['skipped']} already "
            f"present, {counts['pending']} pending, {counts['failed']} failed"
        )
    finally:
        executor.close()
    return 1 if counts["failed"] or counts["pending"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
