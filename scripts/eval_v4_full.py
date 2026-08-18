"""Evaluate v4 against full BM25 and the current binary-vector index.

V4 differs from the older synthetic evaluations in two important ways:
questions may require more than one gold document, and every gold document
contains one or more evidence chunk ids.  This runner therefore reports
fractional document recall, all-hop recall, and vector evidence recall in
addition to first-relevant MRR.

The vector index may be incomplete.  Results are emitted for both all v4
questions and the subset whose complete gold-document set is present in the
current vec0 store.  The latter is a fair retrieval comparison but may be too
small for a stable quality conclusion.

Usage:
    uv run python scripts/eval_v4_full.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from eval_bm25_full import DfPruner  # noqa: E402

from corpus_store.db import connect  # noqa: E402
from corpus_store.embed import (  # noqa: E402
    MODEL_ID,
    PREFIX_QUERY,
    embed_batch,
    pack_binary,
    start_server,
)

TOP_K = (1, 5, 10, 20)
ALPHAS = (0.25, 0.5, 0.75)


def load_queries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def gold_docs(query: dict[str, Any]) -> set[int]:
    return {doc["document_id"] for doc in query["gold_documents"]}


def gold_evidence(query: dict[str, Any]) -> set[int]:
    return {
        evidence["chunk_id"]
        for doc in query["gold_documents"]
        for evidence in doc["evidence"]
    }


def code_snapshot() -> dict[str, Any]:
    """Capture the Git revision and whether local changes affect reproducibility."""
    repository = Path(__file__).resolve().parent.parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"code_revision": "unknown", "code_dirty": True}
    return {"code_revision": revision, "code_dirty": dirty}


def ensure_query_embeddings(
    queries: list[dict[str, Any]], cache_dir: Path, gguf: Path, port: int
) -> np.ndarray:
    bin_path = cache_dir / "v4_jina_v5_small_queries_bin.npy"
    meta_path = cache_dir / "v4_jina_v5_small_queries_meta.json"
    expected_meta = {
        "model": MODEL_ID,
        "prefix": PREFIX_QUERY,
        "queries": [{"id": q["id"], "question": q["question"]} for q in queries],
    }
    if bin_path.exists() and meta_path.exists():
        cached_meta = json.loads(meta_path.read_text())
        if cached_meta == expected_meta:
            packed = np.load(bin_path)
            if packed.shape == (len(queries), 128):
                print(f"reusing {len(queries)} cached v4 query embeddings", flush=True)
                return packed

    print(f"embedding {len(queries)} v4 queries with {gguf.name}", flush=True)
    proc = start_server(gguf, ctx=8192, port=port)
    try:
        vectors = embed_batch(
            [PREFIX_QUERY + query["question"] for query in queries], port
        )
    finally:
        proc.terminate()
        proc.wait()
    if vectors.shape != (len(queries), 1024):
        raise RuntimeError(f"unexpected query embedding shape: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(norms > 0.99):
        raise RuntimeError(f"query embeddings are not normalized: min={norms.min()}")
    packed = np.stack(
        [np.frombuffer(pack_binary(vector), dtype=np.uint8) for vector in vectors]
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(bin_path, packed)
    meta_path.write_text(json.dumps(expected_meta, ensure_ascii=False, indent=2))
    return packed


def bm25_ranked_lists(
    corpus: sqlite3.Connection,
    queries: list[dict[str, Any]],
    depth: int,
) -> tuple[list[list[tuple[int, float]]], float, int]:
    import re

    n_docs = corpus.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    pruner = DfPruner(corpus, n_docs)
    ranked_lists: list[list[tuple[int, float]]] = []
    tokens_pruned = 0
    started = time.time()
    for i, query in enumerate(queries, 1):
        tokens = re.findall(r"\w+", query["question"], flags=re.UNICODE)
        kept = pruner.prune(tokens)
        tokens_pruned += len(tokens) - len(kept)
        match = " OR ".join(f'"{token}"' for token in kept)
        rows = corpus.execute(
            "SELECT rowid, bm25(documents_fts) FROM documents_fts"
            " WHERE documents_fts MATCH ?"
            " ORDER BY bm25(documents_fts) LIMIT ?",
            (match, depth),
        ).fetchall()
        ranked_lists.append([(row[0], -row[1]) for row in rows])
        if i % 10 == 0:
            print(f"  BM25 {i}/{len(queries)}", flush=True)
    return ranked_lists, time.time() - started, tokens_pruned


def chunk_documents(chunks: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, int]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    return dict(
        chunks.execute(
            f"SELECT chunk_id, document_id FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
    )


def vector_ranked_lists(
    vec0: sqlite3.Connection,
    chunks: sqlite3.Connection,
    packed_queries: np.ndarray,
    scan_k: int,
    doc_depth: int,
) -> tuple[list[list[tuple[int, float]]], list[list[tuple[int, float]]], float]:
    chunk_lists: list[list[tuple[int, float]]] = []
    document_lists: list[list[tuple[int, float]]] = []
    started = time.time()
    for i, query_vector in enumerate(packed_queries, 1):
        rows = vec0.execute(
            "SELECT rowid, distance FROM chunk_vec"
            " WHERE embedding MATCH vec_bit(?) AND k = ?",
            (query_vector.tobytes(), scan_k),
        ).fetchall()
        chunk_lists.append([(rowid, -float(distance)) for rowid, distance in rows])
        cid_to_doc = chunk_documents(chunks, [rowid for rowid, _ in rows])
        seen: set[int] = set()
        collapsed: list[tuple[int, float]] = []
        for rowid, distance in rows:
            document_id = cid_to_doc.get(rowid)
            if document_id is not None and document_id not in seen:
                seen.add(document_id)
                collapsed.append((document_id, -float(distance)))
                if len(collapsed) >= doc_depth:
                    break
        document_lists.append(collapsed)
        if i % 10 == 0:
            print(f"  vector {i}/{len(packed_queries)}", flush=True)
    return chunk_lists, document_lists, time.time() - started


def fuse_weighted(
    lexical: list[list[tuple[int, float]]],
    vector: list[list[tuple[int, float]]],
    alpha: float,
) -> list[list[int]]:
    def normalize(ranked: list[tuple[int, float]]) -> dict[int, float]:
        if not ranked:
            return {}
        scores = np.array([score for _, score in ranked])
        low, high = float(scores.min()), float(scores.max())
        span = high - low if high > low else 1.0
        return {item: (score - low) / span for item, score in ranked}

    fused: list[list[int]] = []
    for left, right in zip(lexical, vector):
        nl, nr = normalize(left), normalize(right)
        left_rank = {item: rank for rank, (item, _) in enumerate(left)}
        right_rank = {item: rank for rank, (item, _) in enumerate(right)}
        keys = set(nl) | set(nr)
        score = {
            key: alpha * nl.get(key, 0.0) + (1.0 - alpha) * nr.get(key, 0.0)
            for key in keys
        }
        fused.append(
            sorted(
                keys,
                key=lambda key: (
                    -score[key],
                    min(left_rank.get(key, 10**9), right_rank.get(key, 10**9)),
                    key,
                ),
            )
        )
    return fused


def fuse_rrf(
    lexical: list[list[tuple[int, float]]],
    vector: list[list[tuple[int, float]]],
    constant: int = 60,
) -> list[list[int]]:
    fused: list[list[int]] = []
    for left, right in zip(lexical, vector):
        scores: dict[int, float] = defaultdict(float)
        best_rank: dict[int, int] = {}
        for ranked in (left, right):
            for rank, (item, _) in enumerate(ranked, 1):
                scores[item] += 1.0 / (constant + rank)
                best_rank[item] = min(best_rank.get(item, 10**9), rank)
        fused.append(
            sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))
        )
    return fused


def ranking_summary(
    ranked_ids: list[list[int]], queries: list[dict[str, Any]]
) -> dict[str, Any]:
    def aggregate(indices: list[int]) -> dict[str, Any]:
        mrr = 0.0
        any_hit = defaultdict(float)
        doc_recall = defaultdict(float)
        all_hop = defaultdict(float)
        for index in indices:
            gold = gold_docs(queries[index])
            ranked = ranked_ids[index]
            positions = [ranked.index(doc) + 1 for doc in gold if doc in ranked]
            if positions:
                mrr += 1.0 / min(positions)
            for k in TOP_K:
                found = gold & set(ranked[:k])
                any_hit[k] += bool(found)
                doc_recall[k] += len(found) / len(gold)
                all_hop[k] += gold.issubset(set(ranked[:k]))
        n = len(indices)
        return {
            "n_queries": n,
            "mrr_first_gold": mrr / n,
            "any_gold_at_k": {str(k): any_hit[k] / n for k in TOP_K},
            "document_recall_at_k": {str(k): doc_recall[k] / n for k in TOP_K},
            "all_hop_recall_at_k": {str(k): all_hop[k] / n for k in TOP_K},
        }

    all_indices = list(range(len(queries)))
    result = aggregate(all_indices)
    result["per_category"] = {}
    categories = sorted({query["category"] for query in queries})
    for category in categories:
        indices = [
            index
            for index, query in enumerate(queries)
            if query["category"] == category
        ]
        result["per_category"][category] = aggregate(indices)
    return result


def evidence_summary(
    ranked_chunks: list[list[int]], queries: list[dict[str, Any]]
) -> dict[str, Any]:
    def aggregate(indices: list[int]) -> dict[str, Any]:
        recall = defaultdict(float)
        all_evidence = defaultdict(float)
        for index in indices:
            gold = gold_evidence(queries[index])
            ranked = ranked_chunks[index]
            for k in TOP_K:
                found = gold & set(ranked[:k])
                recall[k] += len(found) / len(gold)
                all_evidence[k] += gold.issubset(set(ranked[:k]))
        n = len(indices)
        return {
            "n_queries": n,
            "evidence_chunk_recall_at_k": {str(k): recall[k] / n for k in TOP_K},
            "all_evidence_recall_at_k": {str(k): all_evidence[k] / n for k in TOP_K},
        }

    result = aggregate(list(range(len(queries))))
    result["per_category"] = {}
    for category in sorted({query["category"] for query in queries}):
        indices = [
            index
            for index, query in enumerate(queries)
            if query["category"] == category
        ]
        result["per_category"][category] = aggregate(indices)
    return result


def select(items: list[Any], indices: list[int]) -> list[Any]:
    return [items[index] for index in indices]


def coverage_snapshot(
    chunks: sqlite3.Connection,
    queries: list[dict[str, Any]],
    max_vector_id: int,
) -> tuple[dict[str, Any], list[int]]:
    unique_gold = sorted({doc for query in queries for doc in gold_docs(query)})
    placeholders = ",".join("?" for _ in unique_gold)
    doc_max = dict(
        chunks.execute(
            f"SELECT document_id, MAX(chunk_id) FROM chunks"
            f" WHERE document_id IN ({placeholders}) GROUP BY document_id",
            unique_gold,
        )
    )
    covered_docs = {doc for doc, maximum in doc_max.items() if maximum <= max_vector_id}
    eligible = [
        index
        for index, query in enumerate(queries)
        if gold_docs(query).issubset(covered_docs)
    ]
    evidence_ids = [cid for query in queries for cid in gold_evidence(query)]
    return (
        {
            "unique_gold_documents": len(unique_gold),
            "covered_unique_gold_documents": len(covered_docs),
            "gold_document_hops": sum(len(gold_docs(query)) for query in queries),
            "covered_gold_document_hops": sum(
                len(gold_docs(query) & covered_docs) for query in queries
            ),
            "gold_evidence_chunks": len(evidence_ids),
            "covered_gold_evidence_chunks": sum(
                chunk_id <= max_vector_id for chunk_id in evidence_ids
            ),
            "fully_covered_questions": len(eligible),
            "eligible_question_ids": [queries[index]["id"] for index in eligible],
        },
        eligible,
    )


def write_ranked_lists(
    path: Path,
    queries: list[dict[str, Any]],
    bm25: list[list[tuple[int, float]]],
    vector_docs: list[list[tuple[int, float]]],
    vector_chunks: list[list[tuple[int, float]]],
) -> None:
    with path.open("w") as handle:
        for query, lexical, documents, chunks in zip(
            queries, bm25, vector_docs, vector_chunks
        ):
            handle.write(
                json.dumps(
                    {
                        "id": query["id"],
                        "bm25_documents": lexical,
                        "vector_documents": documents,
                        "vector_chunks": chunks,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def report_table(systems: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| System | MRR | Doc R@5 | Doc R@10 | All-hop@10 | All-hop@20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in sorted(
        systems.items(), key=lambda item: -item[1]["mrr_first_gold"]
    ):
        lines.append(
            f"| {name} | {metrics['mrr_first_gold']:.3f} | "
            f"{metrics['document_recall_at_k']['5']:.3f} | "
            f"{metrics['document_recall_at_k']['10']:.3f} | "
            f"{metrics['all_hop_recall_at_k']['10']:.3f} | "
            f"{metrics['all_hop_recall_at_k']['20']:.3f} |"
        )
    return lines


def write_report(path: Path, results: dict[str, Any]) -> None:
    coverage = results["vector_coverage"]
    all_systems = results["all_questions"]["systems"]
    eligible_systems = results["eligible_questions"]["systems"]
    best_hybrid = max(
        (name for name in all_systems if name.startswith(("W", "RRF"))),
        key=lambda name: all_systems[name]["mrr_first_gold"],
    )
    lines = [
        "# Eval v4: full BM25 vs partial vectors and hybrid",
        "",
        f"Run date: {results['run_date']}",
        "",
        "## Index snapshot",
        "",
        f"- BM25: {results['corpus_documents']:,} documents (complete).",
        f"- Binary vectors: {results['vectors_in_store']:,} / "
        f"{results['total_chunks']:,} chunks "
        f"({results['vectors_in_store'] / results['total_chunks']:.1%}).",
        f"- Vec0 contiguous through chunk id {results['max_vector_chunk_id']:,}: "
        f"{results['vector_store_contiguous']}.",
        f"- Fully vector-covered v4 questions: "
        f"{coverage['fully_covered_questions']} / {results['n_queries']} "
        f"({', '.join(coverage['eligible_question_ids']) or 'none'}).",
        f"- Covered unique gold documents: "
        f"{coverage['covered_unique_gold_documents']} / "
        f"{coverage['unique_gold_documents']}.",
        "",
        "## All 42 questions",
        "",
        "This is the operational result against today's indexes. Vector and hybrid "
        "scores are coverage-confounded because most gold documents have not yet "
        "been embedded.",
        "",
        *report_table(all_systems),
        "",
        "## Fully covered subset",
        "",
        f"Only {coverage['fully_covered_questions']} questions currently qualify. "
        "This cut is fair to the vector leg but too small for a stable model choice.",
        "",
        *report_table(eligible_systems),
        "",
        "## Vector evidence retrieval",
        "",
        "| Cut | Evidence R@5 | Evidence R@10 | Evidence R@20 | All evidence@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metrics in (
        ("All 42", results["all_questions"]["vector_evidence"]),
        ("Fully covered", results["eligible_questions"]["vector_evidence"]),
    ):
        lines.append(
            f"| {label} | {metrics['evidence_chunk_recall_at_k']['5']:.3f} | "
            f"{metrics['evidence_chunk_recall_at_k']['10']:.3f} | "
            f"{metrics['evidence_chunk_recall_at_k']['20']:.3f} | "
            f"{metrics['all_evidence_recall_at_k']['20']:.3f} |"
        )
    lines += [
        "",
        "## Per-category MRR on all questions",
        "",
        f"The hybrid column uses `{best_hybrid}`, the best hybrid by all-question MRR.",
        "",
        "| Category | BM25 | Vector | Hybrid |",
        "|---|---:|---:|---:|",
    ]
    for category in sorted(all_systems["BM25-doc"]["per_category"]):
        lines.append(
            f"| {category} | "
            f"{all_systems['BM25-doc']['per_category'][category]['mrr_first_gold']:.3f} | "
            f"{all_systems['jina-binary-partial']['per_category'][category]['mrr_first_gold']:.3f} | "
            f"{all_systems[best_hybrid]['per_category'][category]['mrr_first_gold']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- BM25 is the only complete-index baseline in this run.",
        "- The all-question vector score primarily measures current index coverage, "
        "not the final embedding model's retrieval quality.",
        "- The fully covered subset should be treated as a mechanical smoke test; "
        "rerun the identical command after the vector build and vec0 top-off complete.",
        "- Preserve these outputs as the partial-index checkpoint and compare the final "
        "run using the same frozen v4 questions and runner.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run python scripts/build_vec0_full.py \\",
        "  --vectors-db dof_db/dof_vectors_jina_binary.sqlite \\",
        "  --vec0-db dof_db/dof_vec0_jina_binary.sqlite",
        "uv run python scripts/eval_v4_full.py",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="eval/dof_queries_v4.jsonl")
    parser.add_argument("--corpus-db", default="dof_db/dof_corpus_l3.sqlite")
    parser.add_argument("--chunks-db", default="dof_db/dof_chunks.sqlite")
    parser.add_argument("--vec0-db", default="dof_db/dof_vec0_jina_binary.sqlite")
    parser.add_argument("--vectors-db", default="dof_db/dof_vectors_jina_binary.sqlite")
    parser.add_argument("--bm25-depth", type=int, default=50)
    parser.add_argument("--vector-k", type=int, default=200)
    parser.add_argument("--vector-doc-depth", type=int, default=50)
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--cache-dir", default="eval/cache")
    parser.add_argument("--report", default="reports/eval_v4_retrieval.md")
    parser.add_argument("--output", default="eval/cache/eval_v4_full_comparison.json")
    parser.add_argument(
        "--ranked-output", default="eval/cache/eval_v4_ranked_lists.jsonl"
    )
    args = parser.parse_args()

    query_path = Path(args.queries)
    queries = load_queries(query_path)
    cache_dir = Path(args.cache_dir)
    vectors = sqlite3.connect(args.vectors_db)
    vector_meta = dict(vectors.execute("SELECT key, value FROM vector_meta"))
    vectors.close()
    if vector_meta.get("model") != MODEL_ID:
        raise RuntimeError(
            f"vector model {vector_meta.get('model')!r} != runner model {MODEL_ID!r}"
        )
    gguf = Path(vector_meta["gguf"])
    packed = ensure_query_embeddings(queries, cache_dir, gguf, args.port)

    corpus = connect(args.corpus_db)
    n_corpus_docs = corpus.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"BM25 over {n_corpus_docs:,} documents", flush=True)
    lexical, bm25_seconds, tokens_pruned = bm25_ranked_lists(
        corpus, queries, args.bm25_depth
    )
    corpus.close()

    vec0 = sqlite3.connect(args.vec0_db)
    vec0.enable_load_extension(True)
    sqlite_vec.load(vec0)
    n_vectors, min_vector_id, max_vector_id = vec0.execute(
        "SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM chunk_vec"
    ).fetchone()
    contiguous = min_vector_id == 1 and n_vectors == max_vector_id
    chunks = sqlite3.connect(args.chunks_db)
    total_chunks = chunks.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    coverage, eligible_indices = coverage_snapshot(chunks, queries, max_vector_id)
    print(
        f"vector search over {n_vectors:,} chunks; "
        f"{len(eligible_indices)}/{len(queries)} questions fully covered",
        flush=True,
    )
    vector_chunks, vector_docs, vector_seconds = vector_ranked_lists(
        vec0, chunks, packed, args.vector_k, args.vector_doc_depth
    )
    vec0.close()
    chunks.close()

    lexical_ids = [[document for document, _ in ranked] for ranked in lexical]
    vector_doc_ids = [[document for document, _ in ranked] for ranked in vector_docs]
    vector_chunk_ids = [[chunk for chunk, _ in ranked] for ranked in vector_chunks]
    system_lists: dict[str, list[list[int]]] = {
        "BM25-doc": lexical_ids,
        "jina-binary-partial": vector_doc_ids,
        "RRF(BM25,jina-binary)": fuse_rrf(lexical, vector_docs),
    }
    for alpha in ALPHAS:
        system_lists[f"W{alpha}(BM25,jina-binary)"] = fuse_weighted(
            lexical, vector_docs, alpha
        )

    all_systems = {
        name: ranking_summary(ranked, queries) for name, ranked in system_lists.items()
    }
    eligible_queries = select(queries, eligible_indices)
    eligible_systems = {
        name: ranking_summary(select(ranked, eligible_indices), eligible_queries)
        for name, ranked in system_lists.items()
    }
    all_evidence = evidence_summary(vector_chunk_ids, queries)
    eligible_evidence = evidence_summary(
        select(vector_chunk_ids, eligible_indices), eligible_queries
    )

    results = {
        "run_date": date.today().isoformat(),
        "evaluation": "DOF-RAG Evidence Evaluation v4",
        "provenance": {
            **code_snapshot(),
            "query_file": str(query_path),
            "query_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
            "embedding_model": vector_meta.get("model"),
            "embedding_prefix": vector_meta.get("prefix"),
        },
        "n_queries": len(queries),
        "corpus_documents": n_corpus_docs,
        "total_chunks": total_chunks,
        "vectors_in_store": n_vectors,
        "min_vector_chunk_id": min_vector_id,
        "max_vector_chunk_id": max_vector_id,
        "vector_store_contiguous": contiguous,
        "vector_coverage": coverage,
        "settings": {
            "bm25_depth": args.bm25_depth,
            "vector_scan_k": args.vector_k,
            "vector_document_depth": args.vector_doc_depth,
            "fusion_alphas_bm25_weight": list(ALPHAS),
            "bm25_tokens_pruned": tokens_pruned,
        },
        "timings_seconds": {"bm25": bm25_seconds, "vector": vector_seconds},
        "all_questions": {
            "systems": all_systems,
            "vector_evidence": all_evidence,
        },
        "eligible_questions": {
            "ids": [queries[index]["id"] for index in eligible_indices],
            "systems": eligible_systems,
            "vector_evidence": eligible_evidence,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    write_ranked_lists(
        Path(args.ranked_output), queries, lexical, vector_docs, vector_chunks
    )
    write_report(Path(args.report), results)

    print("\nAll 42 questions")
    for line in report_table(all_systems):
        print(line)
    print(f"\nFully covered subset: {coverage['eligible_question_ids']}")
    for line in report_table(eligible_systems):
        print(line)
    print(f"\nwrote {output} and {args.report}")


if __name__ == "__main__":
    main()
