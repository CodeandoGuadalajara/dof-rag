"""Run the bounded DOF tool-calling agent against evidence eval v4.

By default this selects one frozen question per category (seven total):

    .venv/bin/python scripts/eval_v4_agent.py --model MODEL

With a Kimi Code membership key:

    .venv/bin/python scripts/eval_v4_agent.py \
        --provider kimi-code --model kimi-for-coding

Use ``--ids SP-001,NE-001`` for a smaller smoke test or ``--all`` for all 42.
The output contains the complete tool trace, token usage, latency, and citation
metrics. API credentials are read by the OpenAI SDK and are never written out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_tools.agent import (
    AgentRunner,
    DofToolbox,
    OpenAIChatCompletionsBackend,
    OpenAIResponsesBackend,
)
from agent_tools.retrieval import DofRetriever, LlamaQueryEmbedder


def load_queries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_queries(
    queries: list[dict[str, Any]], *, ids: set[str] | None, run_all: bool
) -> list[dict[str, Any]]:
    if ids:
        selected = [query for query in queries if query["id"] in ids]
        missing = ids - {query["id"] for query in selected}
        if missing:
            raise ValueError(f"unknown query ids: {sorted(missing)}")
        return selected
    if run_all:
        return queries
    by_category: dict[str, dict[str, Any]] = {}
    for query in queries:
        by_category.setdefault(query["category"], query)
    return [by_category[category] for category in sorted(by_category)]


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [item for item in results if "run" in item]
    completed = [item for item in runs if item["run"].get("stop_reason") == "completed"]
    if not completed:
        return {
            "n": len(results),
            "runs": len(runs),
            "completed": 0,
            "completion_rate": 0.0,
        }
    precisions: list[float] = []
    recalls: list[float] = []
    false_premise: list[bool] = []
    comparison_coverage: list[bool] = []
    tool_errors = 0
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for item in completed:
        gold = {
            evidence["chunk_id"]
            for document in item["gold_documents"]
            for evidence in document["evidence"]
        }
        cited = set(item["run"]["answer"]["citations"])
        precisions.append(len(gold & cited) / len(cited) if cited else 0.0)
        recalls.append(len(gold & cited) / len(gold))
        tool_errors += sum(
            not trace["output"].get("ok", False) for trace in item["run"]["traces"]
        )
        for key in totals:
            totals[key] += item["run"]["usage"].get(key, 0)
        coverage = item["run"].get("coverage", {})
        if coverage:
            comparison_coverage.append(all(coverage.values()))
        if item["category"] == "negative_false_premise":
            false_premise.append(item["run"]["answer"]["premise_status"] == "false")
    n = len(completed)
    return {
        "n": len(results),
        "runs": len(runs),
        "completed": n,
        "completion_rate": n / len(results),
        "citation_precision": sum(precisions) / n,
        "citation_recall": sum(recalls) / n,
        "false_premise_correction_accuracy": (
            sum(false_premise) / len(false_premise) if false_premise else None
        ),
        "coverage_completion_rate": (
            sum(comparison_coverage) / len(comparison_coverage)
            if comparison_coverage
            else None
        ),
        "tool_error_count": tool_errors,
        "average_tool_calls": sum(item["run"]["tool_calls"] for item in completed) / n,
        "average_model_turns": sum(item["run"]["model_turns"] for item in completed)
        / n,
        "average_latency_ms": sum(item["run"]["elapsed_ms"] for item in completed) / n,
        "usage": totals,
        "answer_correctness": "pending human or judge-model adjudication",
    }


def fatal_provider_error(exc: Exception) -> bool:
    """Return true for session-wide failures that retries cannot repair."""
    if type(exc).__name__ in {"AuthenticationError", "PermissionDeniedError"}:
        return True
    details = " ".join(
        str(value)
        for value in (
            getattr(exc, "code", ""),
            getattr(exc, "body", ""),
            str(exc),
        )
    ).casefold()
    return any(
        marker in details
        for marker in (
            "insufficient_quota",
            "credit_balance_exhausted",
            "invalid_api_key",
            "reached your usage limit",
            "reached kimi monthly usage limit",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="eval/dof_queries_v4.jsonl")
    parser.add_argument(
        "--provider",
        choices=["openai-responses", "kimi-code"],
        default="openai-responses",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", help="Comma-separated v4 question IDs")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--base-url")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-model-turns", type=int, default=7)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--corpus-db", default="dof_db/dof_corpus_l3.sqlite")
    parser.add_argument("--chunks-db", default="dof_db/dof_chunks.sqlite")
    parser.add_argument("--vec0-db", default="dof_db/dof_vec0_jina_binary.sqlite")
    parser.add_argument("--no-vector", action="store_true")
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--output", default="eval/cache/eval_v4_agent_smoke.json")
    args = parser.parse_args()
    if not args.model:
        parser.error("set OPENAI_MODEL or pass --model")
    ids = {value.strip() for value in args.ids.split(",")} if args.ids else None
    try:
        queries = select_queries(
            load_queries(Path(args.queries)), ids=ids, run_all=args.all
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.no_vector:
        args.vec0_db = None

    if args.provider == "kimi-code":
        api_key = os.environ.get("KIMI_API_KEY", "")
        if not api_key:
            parser.error("kimi-code provider requires KIMI_API_KEY")
        backend = OpenAIChatCompletionsBackend(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url or "https://api.kimi.com/coding/v1",
        )
    else:
        backend = OpenAIResponsesBackend(
            model=args.model,
            base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
            reasoning_effort=args.reasoning_effort or None,
        )
    results: list[dict[str, Any]] = []
    with DofRetriever(
        corpus_db=args.corpus_db,
        chunks_db=args.chunks_db,
        vec0_db=args.vec0_db,
    ) as retriever:
        embedder = LlamaQueryEmbedder(args.gguf, port=args.port) if args.gguf else None
        try:
            toolbox = DofToolbox(retriever, embedder=embedder)
            runner = AgentRunner(
                backend,
                toolbox,
                max_model_turns=args.max_model_turns,
                max_tool_calls=args.max_tool_calls,
            )
            for index, query in enumerate(queries, 1):
                item = dict(query)
                abort = False
                error_type = ""
                try:
                    run = runner.run(query["question"], as_of=query["as_of"])
                    item["run"] = run.to_dict()
                    answer = run.answer
                    print(
                        f"[{index}/{len(queries)}] {query['id']} "
                        f"stop={run.stop_reason} tools={run.tool_calls} "
                        f"citations={answer.citations}",
                        flush=True,
                    )
                except Exception as exc:
                    error_type = type(exc).__name__
                    item["error"] = {"type": error_type, "message": str(exc)}
                    abort = fatal_provider_error(exc)
                    print(
                        f"[{index}/{len(queries)}] {query['id']} ERROR {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                results.append(item)
                if abort:
                    for skipped in queries[index:]:
                        skipped_item = dict(skipped)
                        skipped_item["error"] = {
                            "type": "NotRunAfterFatalProviderError",
                            "message": (
                                f"not run after {query['id']} failed with {error_type}"
                            ),
                        }
                        results.append(skipped_item)
                    print(
                        f"aborted remaining questions after fatal provider error on {query['id']}",
                        flush=True,
                    )
                    break
        finally:
            if embedder:
                embedder.close()

    settings = {
        "queries": args.queries,
        "provider": args.provider,
        "selection": "ids" if ids else "all" if args.all else "one_per_category",
        "ids": sorted(ids) if ids else None,
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "max_model_turns": args.max_model_turns,
        "max_tool_calls": args.max_tool_calls,
        "corpus_db": args.corpus_db,
        "chunks_db": args.chunks_db,
        "vec0_db": args.vec0_db,
        "gguf": str(args.gguf) if args.gguf else None,
    }
    payload = {
        "settings": settings,
        "metrics": calculate_metrics(results),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0 if payload["metrics"]["completed"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
