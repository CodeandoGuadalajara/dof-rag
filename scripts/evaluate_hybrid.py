"""Hybrid retrieval evaluation: BM25 + embedding fusion (RRF / weighted).

Evaluates hybrid retrieval on the round-3 query set (499 docs, 3,023 queries),
fusing SQLite FTS5 BM25 rankings with embedding rankings per model/variant.

Fusion methods:
- RRF (Reciprocal Rank Fusion), k=60
- Weighted sum of min-max-normalized scores, alpha sweep

Embeddings are cached under eval/cache/ so fusion experiments rerun fast.

Run from repo root:
    uv run python scripts/evaluate_hybrid.py [--corpus ./dof_md]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_retrieval import (  # noqa: E402
    TOP_K,
    _apply_variant,
    _bm25_match_query,
    _chunk_level_metrics,
    _create_queries,
    _cosine_similarity,
    _FTS5_TOKENIZER,
    _load_query_dataset,
    _metrics_from_ranked_lists,
)

QUERIES_PATH = Path("eval/dof_queries_v2.jsonl")
CACHE_DIR = Path("eval/cache")
REPORT_PATH = Path("reports/hybrid_retrieval.md")
FUSION_DEPTH = 50  # how deep each system's ranked list goes into fusion
RRF_K = 60
ALPHAS = [0.25, 0.4, 0.5, 0.6, 0.75]  # weight on BM25 in weighted fusion

# Production-candidate configurations: (label, HF model, post-hoc variant)
MODEL_CONFIGS = [
    ("F2LLM-0.6B fp32", "codefuse-ai/F2LLM-v2-0.6B", "full_fp32"),
    ("F2LLM-0.6B int8", "codefuse-ai/F2LLM-v2-0.6B", "int8"),
    ("jina-v5-small fp32", "jinaai/jina-embeddings-v5-text-small", "full_fp32"),
    ("jina-v5-small binary", "jinaai/jina-embeddings-v5-text-small", "binary"),
]


def bm25_ranked(docs: list[dict], queries: list[dict], depth: int) -> tuple[list[list[tuple[int, float]]], list[str]]:
    """Per-query BM25 ranked lists of (chunk_index, score), best first.

    FTS5 bm25() returns negative scores (more negative = better); we negate
    so that higher = better everywhere downstream.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute(
        f"CREATE VIRTUAL TABLE chunks USING fts5(text, doc_id UNINDEXED, "
        f"tokenize = '{_FTS5_TOKENIZER}')"
    )
    chunk_doc_ids: list[str] = []
    rows = []
    for doc in docs:
        for chunk_text in doc["chunks"]:
            rows.append((chunk_text, doc["doc_id"]))
            chunk_doc_ids.append(doc["doc_id"])
    conn.executemany("INSERT INTO chunks(text, doc_id) VALUES (?, ?)", rows)

    ranked: list[list[tuple[int, float]]] = []
    for q in queries:
        match = _bm25_match_query(q["query"])
        if not match:
            ranked.append([])
            continue
        cur = conn.execute(
            "SELECT rowid, bm25(chunks) FROM chunks WHERE chunks MATCH ? "
            "ORDER BY bm25(chunks) LIMIT ?",
            (match, depth),
        )
        ranked.append([(r[0] - 1, -r[1]) for r in cur.fetchall()])  # rowid → 0-based, negate
    conn.close()
    return ranked, chunk_doc_ids


def embed_with_cache(model_name: str, texts: list[str], kind: str, device: str) -> np.ndarray:
    """Embed texts with sentence-transformers, caching fp32 vectors to disk."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{slug}_{kind}.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(texts):
            print(f"  cached {kind} embeddings ({emb.shape})")
            return emb
    from sentence_transformers import SentenceTransformer

    model_kwargs = {}
    if "jina" in model_name.lower():
        model_kwargs["default_task"] = "retrieval"
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True,
                                model_kwargs=model_kwargs)
    t0 = time.perf_counter()
    if kind == "queries" and "jina" in model_name.lower():
        emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True,
                           task="retrieval")
    else:
        emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    print(f"  embedded {len(texts):,} {kind} in {time.perf_counter() - t0:.0f}s")
    np.save(cache, emb)
    del model
    import gc
    gc.collect()
    return emb


def vector_ranked(query_emb: np.ndarray, chunk_emb: np.ndarray, depth: int) -> list[list[tuple[int, float]]]:
    """Per-query cosine ranked lists of (chunk_index, score), best first."""
    sims = _cosine_similarity(query_emb, chunk_emb)
    ranked = []
    for i in range(sims.shape[0]):
        top = np.argsort(sims[i])[::-1][:depth]
        ranked.append([(int(idx), float(sims[i, idx])) for idx in top])
    return ranked


def fuse_rrf(list_a: list[list[tuple[int, float]]], list_b: list[list[tuple[int, float]]],
             k: int = RRF_K) -> list[list[int]]:
    """Reciprocal Rank Fusion of two chunk-ranked lists → ranked chunk indices."""
    fused = []
    for ra, rb in zip(list_a, list_b):
        scores: dict[int, float] = defaultdict(float)
        for rank, (idx, _) in enumerate(ra, 1):
            scores[idx] += 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(rb, 1):
            scores[idx] += 1.0 / (k + rank)
        fused.append([idx for idx, _ in sorted(scores.items(), key=lambda kv: -kv[1])])
    return fused


def fuse_weighted(list_a: list[list[tuple[int, float]]], list_b: list[list[tuple[int, float]]],
                  alpha: float) -> list[list[int]]:
    """Weighted fusion of min-max-normalized scores: alpha*A + (1-alpha)*B."""
    def norm(ranked: list[tuple[int, float]]) -> dict[int, float]:
        if not ranked:
            return {}
        scores = np.array([s for _, s in ranked])
        lo, hi = scores.min(), scores.max()
        span = hi - lo if hi > lo else 1.0
        return {idx: (s - lo) / span for idx, s in ranked}

    fused = []
    for ra, rb in zip(list_a, list_b):
        na, nb = norm(ra), norm(rb)
        keys = set(na) | set(nb)
        scores = {idx: alpha * na.get(idx, 0.0) + (1 - alpha) * nb.get(idx, 0.0)
                  for idx in keys}
        fused.append([idx for idx, _ in sorted(scores.items(), key=lambda kv: -kv[1])])
    return fused


def evaluate_ranking(ranked_chunks: list[list[int]], chunk_doc_ids: list[str],
                     queries: list[dict]) -> dict:
    ranked_doc_ids = [[chunk_doc_ids[idx] for idx in ranked] for ranked in ranked_chunks]
    metrics = _metrics_from_ranked_lists(ranked_doc_ids, queries)
    chunk_level = _chunk_level_metrics(ranked_chunks, queries)
    if chunk_level:
        metrics["chunk_level"] = chunk_level
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="./dof_md")
    parser.add_argument("--queries", default=str(QUERIES_PATH))
    args = parser.parse_args()

    print(f"Loading query dataset from {args.queries}...")
    docs, generated = _load_query_dataset(Path(args.corpus), Path(args.queries))
    queries = _create_queries(docs, generated)
    all_chunks = [c for doc in docs for c in doc["chunks"]]
    print(f"  {len(docs)} documents, {len(all_chunks):,} chunks, {len(queries):,} queries")

    print("BM25 (FTS5) ranked lists...")
    bm25_lists, chunk_doc_ids = bm25_ranked(docs, queries, FUSION_DEPTH)

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    query_texts = [q["query"] for q in queries]
    results: dict[str, dict] = {}

    # BM25-only baseline (truncate to metric depth)
    results["BM25"] = evaluate_ranking(
        [ [idx for idx, _ in r] for r in bm25_lists ], chunk_doc_ids, queries)

    vec_lists: dict[str, list[list[tuple[int, float]]]] = {}
    embedded: set[str] = set()
    for label, model_name, variant in MODEL_CONFIGS:
        print(f"{label}...")
        if model_name not in embedded:
            q_emb = embed_with_cache(model_name, query_texts, "queries", device)
            c_emb = embed_with_cache(model_name, all_chunks, "chunks", device)
            embedded.add(model_name)
        q_var = _apply_variant(q_emb, variant)
        c_var = _apply_variant(c_emb, variant)
        vec_lists[label] = vector_ranked(q_var, c_var, FUSION_DEPTH)
        results[label] = evaluate_ranking(
            [[idx for idx, _ in r] for r in vec_lists[label]], chunk_doc_ids, queries)

        # RRF fusion
        fused = fuse_rrf(bm25_lists, vec_lists[label])
        results[f"RRF(BM25, {label})"] = evaluate_ranking(fused, chunk_doc_ids, queries)

        # Weighted fusion sweep
        for alpha in ALPHAS:
            fused = fuse_weighted(bm25_lists, vec_lists[label], alpha)
            results[f"W{alpha:.2f}(BM25, {label})"] = evaluate_ranking(
                fused, chunk_doc_ids, queries)

    # ---- report ----
    lines = [
        "# Hybrid retrieval evaluation: BM25 + embeddings",
        "",
        f"Corpus: `{args.corpus}` | Query set: `{args.queries}`",
        f"{len(docs)} documentos, {len(all_chunks):,} chunks, {len(queries):,} queries",
        f"Fecha: {time.strftime('%Y-%m-%d')}",
        "",
        f"Fusión a nivel chunk, profundidad {FUSION_DEPTH} por sistema. "
        f"RRF k={RRF_K}; weighted = alpha·BM25 + (1−alpha)·vectores con scores "
        "min-max normalizados por query.",
        "",
        "## Overall (doc-level)",
        "",
        "| Sistema | MRR | R@1 | R@5 | R@10 | R@5 chunk | MRR chunk |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        cl = m.get("chunk_level", {})
        lines.append(
            f"| {name} | {m['mrr']:.3f} | {m['recall_at_k'][1]:.3f} | "
            f"{m['recall_at_k'][5]:.3f} | {m['recall_at_k'][10]:.3f} | "
            f"{cl.get('recall_at_k', {}).get(5, float('nan')):.3f} | "
            f"{cl.get('mrr', float('nan')):.3f} |"
        )

    qtypes = sorted({q["query_type"] for q in queries})
    lines += ["", "## Recall@1 por tipo de query", "",
              "| Sistema | " + " | ".join(qtypes) + " |",
              "|---|" + "---|" * len(qtypes)]
    for name, m in results.items():
        qtm = m["query_type_metrics"]
        lines.append(f"| {name} | " + " | ".join(f"{qtm[t][1]:.3f}" for t in qtypes) + " |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # machine-readable dump
    out = {name: {"mrr": m["mrr"], "recall_at_k": m["recall_at_k"],
                  "query_type_metrics": m["query_type_metrics"],
                  "chunk_level": m.get("chunk_level")}
           for name, m in results.items()}
    (CACHE_DIR / "hybrid_results.json").write_text(json.dumps(out, indent=2))

    print(f"\nReport written to {REPORT_PATH}")
    print("\nTop systems by MRR:")
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["mrr"])[:8]:
        print(f"  {name:42s} MRR={m['mrr']:.3f} R@1={m['recall_at_k'][1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
