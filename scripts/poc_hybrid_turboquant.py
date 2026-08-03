"""PoC: end-to-end hybrid MRR with sqlite-vector TurboQuant as the vector ranker.

Same doc-level fusion harness as scripts/evaluate_hybrid_doclevel.py
(BM25 over whole documents, chunk-level vectors collapsed to docs,
weighted min-max fusion), but vector ranked lists come from the real
sqlite-vector extension (TurboQuant scans over the cached eval embeddings)
instead of numpy. References computed here: jina fp32 doc-collapsed.

Baselines from eval/cache/hybrid_doclevel_results.json:
  W0.5(BM25doc, F2LLM-int8)   MRR 0.662   (quality option)
  W0.5(BM25doc, jina-binary)  MRR 0.650   (sqlite-vec bit fallback)

Run from repo root:
    uv run python scripts/poc_hybrid_turboquant.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_retrieval import _apply_variant, _cosine_similarity  # noqa: E402
import evaluate_hybrid_doclevel as H  # noqa: E402  (module runs the doc-level harness)

DB = Path("poc/data/vectors_eval.sqlite")
EXT = Path("poc/extensions/vector.dylib")
OUT = Path("poc/data/hybrid_turboquant_results.json")
DEPTH = 50


def turbo_ranked(qbits: int, query_emb: np.ndarray, chunk_doc_ids: list[str]) -> list[list[tuple[str, float]]]:
    """Doc-collapsed ranked lists from a TurboQuant scan (scores = -distance)."""
    conn = sqlite3.connect(DB)
    conn.enable_load_extension(True)
    conn.load_extension(str(EXT))
    conn.execute(
        "SELECT vector_init('chunks_vec', 'embedding', "
        "'type=FLOAT32,dimension=1024,distance=COSINE')")
    conn.execute("SELECT vector_quantize_cleanup('chunks_vec', 'embedding')")
    conn.execute(
        f"SELECT vector_quantize('chunks_vec', 'embedding', 'qtype=TURBO,qbits={qbits}')")
    conn.execute("SELECT vector_quantize_preload('chunks_vec', 'embedding')")
    t0 = time.perf_counter()
    out = []
    for i in range(query_emb.shape[0]):
        cur = conn.execute(
            "SELECT rowid, distance FROM vector_quantize_scan("
            " 'chunks_vec', 'embedding', ?, ?)", (query_emb[i].tobytes(), DEPTH))
        best: dict[str, float] = {}
        for rowid, dist in cur.fetchall():
            d = chunk_doc_ids[rowid - 1]
            if d not in best:
                best[d] = -dist
        out.append(list(best.items()))
    dt = time.perf_counter() - t0
    conn.close()
    print(f"  turbo{qbits}: {query_emb.shape[0]:,} queries in {dt:.1f}s "
          f"({dt / query_emb.shape[0] * 1000:.1f} ms/query)")
    return out


def main() -> int:
    docs, queries = H.docs, H.queries
    chunk_doc_ids = H.chunk_doc_ids
    bm25_lists = H.bm25_lists
    print(f"{len(docs)} docs, {len(chunk_doc_ids):,} chunks, {len(queries):,} queries")

    slug = "jinaai_jina_embeddings_v5_text_small"
    q_fp32 = _apply_variant(np.load(H.CACHE / f"{slug}_queries.npy"), "full_fp32")
    c_fp32 = _apply_variant(np.load(H.CACHE / f"{slug}_chunks.npy"), "full_fp32")

    results: dict[str, dict] = {}

    def evaluate(label: str, vec_lists: list[list[tuple[str, float]]]) -> None:
        results[f"{label} (doc-collapsed)"] = H._metrics_from_ranked_lists(
            [[k for k, _ in r] for r in vec_lists], queries)
        results[f"RRF(BM25doc, {label})"] = H._metrics_from_ranked_lists(
            H.fuse_rrf(bm25_lists, vec_lists), queries)
        for a in H.ALPHAS:
            results[f"W{a}(BM25doc, {label})"] = H._metrics_from_ranked_lists(
                H.fuse_weighted(bm25_lists, vec_lists, a), queries)

    # fp32 reference (the ranking TurboQuant approximates)
    sims = _cosine_similarity(q_fp32, c_fp32)
    evaluate("jina-fp32", [H.collapse_vec(sims[i], chunk_doc_ids)
                           for i in range(len(queries))])

    for qbits in (4, 3):
        evaluate(f"jina-turbo{qbits}", turbo_ranked(qbits, q_fp32, chunk_doc_ids))

    print(f"\n{'system':44s} {'MRR':>6s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s}")
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["mrr"]):
        print(f"{name:44s} {m['mrr']:.3f}  {m['recall_at_k'][1]:.3f}  "
              f"{m['recall_at_k'][5]:.3f}  {m['recall_at_k'][10]:.3f}")

    out = {n: {"mrr": m["mrr"], "recall_at_k": m["recall_at_k"],
               "query_type_metrics": m["query_type_metrics"]}
           for n, m in results.items()}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
