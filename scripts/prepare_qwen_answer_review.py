"""Prepare a passage-based review pack; never infer correctness automatically.

Cited passages come from that run's successful reads. Gold passages are read from
the frozen bundle for review only, never sent to a generation model.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def review_rows(data):
    rows, cited = [], {}
    for item in data["results"]:
        run = item.get("run", {})
        read = {}
        for trace in run.get("traces", []):
            if trace["name"] == "read_chunks" and trace["output"].get("ok"):
                for chunk in trace["output"]["data"]["chunks"]:
                    read[chunk["chunk_id"]] = chunk
        answer = run.get("answer", {})
        citations = answer.get("citations") or []
        missing = [chunk_id for chunk_id in citations if chunk_id not in read]
        for chunk_id in citations:
            if chunk_id in read:
                if str(chunk_id) in cited:
                    assert cited[str(chunk_id)]["text"] == read[chunk_id]["text"], (
                        "Cited text changed across runs"
                    )
                cited[str(chunk_id)] = read[chunk_id]
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "as_of": item.get("as_of"),
                "reference_answer": item["reference_answer"],
                "stop_reason": run.get("stop_reason"),
                "answer": answer,
                "cited_ids_not_read_in_this_run": missing,
                "gold_chunk_ids": [
                    hit["chunk_id"]
                    for doc in item["gold_documents"]
                    for hit in doc["evidence"]
                ],
                "last_model_text": run.get("turns", [{}])[-1].get("final_text")
                if run.get("turns")
                else None,
                "tool_errors": [
                    trace["output"]["error"]
                    for trace in run.get("traces", [])
                    if not trace["output"].get("ok")
                ],
                "elapsed_ms": run.get("elapsed_ms"),
                "provider_error": item.get("error"),
                "review_verdict": "pending",
            }
        )
    return rows, cited


def main():
    from agent_tools.retrieval import DofRetriever

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    rows, cited = review_rows(data)
    args.output.mkdir(parents=True, exist_ok=False)
    gold_ids = sorted({chunk_id for row in rows for chunk_id in row["gold_chunk_ids"]})
    with DofRetriever(
        corpus_db=args.inputs / "dof_corpus_l3.sqlite",
        chunks_db=args.inputs / "dof_chunks.sqlite",
        vec0_db=None,
    ) as retriever:
        from dataclasses import asdict

        gold = {
            str(hit.chunk_id): asdict(hit)
            for hit in retriever.read_chunks(gold_ids, neighbor_window=0)
        }
    assert set(map(int, gold)) == set(gold_ids), (
        "Missing gold passages from frozen corpus"
    )
    (args.output / "rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "cited-passages.json").write_text(
        json.dumps(cited, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "gold-passages.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2) + "\n"
    )
    manifest = {
        "results": str(args.results.resolve()),
        "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "inputs": str(args.inputs.resolve()),
        "rows": len(rows),
        "distinct_cited_passages": len(cited),
        "distinct_gold_passages": len(gold),
        "limitations": [
            "Review pack is not an adjudication.",
            "An automatic completed label is not semantic correctness.",
            "Gold-ID mismatch alone does not invalidate alternative supporting citations.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
