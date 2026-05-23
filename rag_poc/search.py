"""Hybrid search: vector + FTS5 with reciprocal rank fusion."""
from __future__ import annotations

import logging
from typing import Any

from rag_poc.config import FINAL_TOP_K, FTS_TOP_K, RRF_K, VECTOR_TOP_K
from rag_poc.database import RAGDatabase
from rag_poc.embedder import embed_query

logger = logging.getLogger("rag_poc.search")


def hybrid_search(
    query: str,
    db: RAGDatabase | None = None,
    vector_k: int = VECTOR_TOP_K,
    fts_k: int = FTS_TOP_K,
    final_k: int = FINAL_TOP_K,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Hybrid search combining vector similarity and FTS5 full-text.

    Uses Reciprocal Rank Fusion (RRF) to merge and re-rank results.
    """
    if db is None:
        db = RAGDatabase()

    # 1. Vector search
    query_vec = embed_query(query)
    vec_results = db.vector_search(query_vec, top_k=vector_k)
    logger.debug("Vector hits: %s", len(vec_results))

    # 2. FTS search
    fts_results = db.fts_search(query, top_k=fts_k)
    logger.debug("FTS hits: %s", len(fts_results))

    # 3. RRF merge
    scores: dict[int, float] = {}
    metadata: dict[int, dict[str, Any]] = {}

    for rank, row in enumerate(vec_results, start=1):
        cid = row["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rank + rrf_k)
        metadata[cid] = row

    for rank, row in enumerate(fts_results, start=1):
        cid = row["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rank + rrf_k)
        if cid not in metadata:
            metadata[cid] = row

    # Sort by RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:final_k]

    results = []
    for cid, score in ranked:
        row = dict(metadata[cid])
        row["rrf_score"] = round(score, 6)
        row["source"] = _infer_source(cid, vec_results, fts_results)
        results.append(row)

    return results


def _infer_source(cid: int, vec_results: list, fts_results: list) -> str:
    in_vec = any(r["id"] == cid for r in vec_results)
    in_fts = any(r["id"] == cid for r in fts_results)
    if in_vec and in_fts:
        return "both"
    if in_vec:
        return "vector"
    return "fts"
