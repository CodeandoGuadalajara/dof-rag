"""Run the agent retrieval tool against the frozen evidence eval v4.

Retrieval-only smoke test:
    .venv/bin/python scripts/eval_v4_tools.py --limit 3 --no-vector

With the local Jina query embedder and an optional LLM answer pass:
    .venv/bin/python scripts/eval_v4_tools.py --gguf /path/to/model.gguf \
        --with-llm --base-url https://api.openai.com/v1 --model MODEL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_tools.llm import ChatClient, answer_with_context
from agent_tools.retrieval import DofRetriever, LlamaQueryEmbedder

TOP_K = (1, 5, 10, 20)


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    def one(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"n": 0}
        doc_recall = defaultdict(float)
        evidence_recall = defaultdict(float)
        all_hop = defaultdict(float)
        document_mrr = 0.0
        evidence_mrr = 0.0
        for item in items:
            gold_docs = {doc["document_id"] for doc in item["gold_documents"]}
            gold_chunks = {
                evidence["chunk_id"]
                for doc in item["gold_documents"]
                for evidence in doc["evidence"]
            }
            docs = item["retrieved_document_ids"]
            chunks = item["retrieved_evidence_ids"]
            document_positions = [
                docs.index(doc_id) + 1 for doc_id in gold_docs if doc_id in docs
            ]
            if document_positions:
                document_mrr += 1.0 / min(document_positions)
            positions = [chunks.index(cid) + 1 for cid in gold_chunks if cid in chunks]
            if positions:
                evidence_mrr += 1.0 / min(positions)
            for k in TOP_K:
                doc_recall[k] += len(gold_docs & set(docs[:k])) / len(gold_docs)
                evidence_recall[k] += len(gold_chunks & set(chunks[:k])) / len(
                    gold_chunks
                )
                all_hop[k] += gold_docs.issubset(set(docs[:k]))
        n = len(items)
        return {
            "n": n,
            "mrr_first_document": document_mrr / n,
            "mrr_first_evidence": evidence_mrr / n,
            "document_recall_at_k": {str(k): doc_recall[k] / n for k in TOP_K},
            "evidence_chunk_recall_at_k": {
                str(k): evidence_recall[k] / n for k in TOP_K
            },
            "all_hop_at_k": {str(k): all_hop[k] / n for k in TOP_K},
        }

    result = {"overall": one(records), "per_category": {}}
    for category in sorted({item["category"] for item in records}):
        result["per_category"][category] = one(
            [item for item in records if item["category"] == category]
        )
    answered = [item for item in records if "answer" in item]
    if answered:
        precision, recall, premise = [], [], []
        invalid_citations = 0
        for item in answered:
            gold = {
                evidence["chunk_id"]
                for doc in item["gold_documents"]
                for evidence in doc["evidence"]
            }
            cited = set(item["answer"].get("citations", []))
            invalid_citations += len(item["answer"].get("invalid_citations", []))
            precision.append(len(cited & gold) / len(cited) if cited else 0.0)
            recall.append(len(cited & gold) / len(gold))
            if item["category"] == "negative_false_premise":
                premise.append(item["answer"].get("premise_status") == "false")
        result["generation"] = {
            "n": len(answered),
            "citation_precision": sum(precision) / len(precision),
            "citation_recall": sum(recall) / len(recall),
            "invalid_citation_count": invalid_citations,
            "false_premise_correction_accuracy": sum(premise) / len(premise)
            if premise
            else None,
            "answer_correctness": "pending human or judge-model adjudication",
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="eval/dof_queries_v4.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--corpus-db", default="dof_db/dof_corpus_l3.sqlite")
    parser.add_argument("--chunks-db", default="dof_db/dof_chunks.sqlite")
    parser.add_argument("--vec0-db", default="dof_db/dof_vec0_jina_binary.sqlite")
    parser.add_argument("--no-vector", action="store_true")
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--bm25-depth", type=int, default=50)
    parser.add_argument("--vector-k", type=int, default=200)
    parser.add_argument("--document-depth", type=int, default=20)
    parser.add_argument("--evidence-depth", type=int, default=20)
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--protocol", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    parser.add_argument("--output", default="eval/cache/eval_v4_tools.json")
    args = parser.parse_args()
    queries = load(Path(args.queries))
    if args.limit:
        queries = queries[: args.limit]
    if args.with_llm and (not args.base_url or not args.model):
        parser.error("--with-llm requires --base-url and --model")
    if args.document_depth < max(TOP_K):
        parser.error(f"--document-depth must be at least {max(TOP_K)}")
    if args.evidence_depth < max(TOP_K):
        parser.error(f"--evidence-depth must be at least {max(TOP_K)}")
    if args.no_vector:
        args.vec0_db = None
    client = (
        ChatClient(base_url=args.base_url, model=args.model, protocol=args.protocol)
        if args.with_llm
        else None
    )
    results: list[dict[str, Any]] = []
    with DofRetriever(
        corpus_db=args.corpus_db, chunks_db=args.chunks_db, vec0_db=args.vec0_db
    ) as retriever:
        embedder = LlamaQueryEmbedder(args.gguf, port=args.port) if args.gguf else None
        try:
            for index, query in enumerate(queries, 1):
                vector = embedder.embed_query(query["question"]) if embedder else None
                search = retriever.search(
                    query["question"],
                    query_vector=vector,
                    as_of=query["as_of"],
                    bm25_depth=args.bm25_depth,
                    vector_k=args.vector_k,
                    document_depth=args.document_depth,
                    evidence_depth=args.evidence_depth,
                )
                item = dict(query)
                item["retrieved_document_ids"] = search.document_ids
                item["retrieved_evidence_ids"] = search.evidence_ids
                item["search"] = search.to_dict()
                if client:
                    item["answer"] = answer_with_context(
                        client, query["question"], search
                    ).to_dict()
                results.append(item)
                print(
                    f"[{index}/{len(queries)}] {query['id']} docs={search.document_ids[:3]} evidence={search.evidence_ids[:3]}",
                    flush=True,
                )
        finally:
            if embedder:
                embedder.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    if isinstance(settings.get("gguf"), Path):
        settings["gguf"] = str(settings["gguf"])
    payload = {"settings": settings, "metrics": metrics(results), "results": results}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
