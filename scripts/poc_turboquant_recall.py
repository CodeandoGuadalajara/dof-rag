"""PoC: validate sqlite-vector TurboQuant recall on the cached eval embeddings.

Ground truth: numpy fp32 cosine top-k over the same cached embeddings used
by scripts/evaluate_hybrid.py (jina-v5-text-small, 8,065 chunks, 3,023
queries, 1,024 dims). Compares against:

- vector_full_scan (exact, SIMD) — sanity: should match numpy ~exactly
- vector_quantize_scan with TurboQuant 2/3/4-bit — recall@10/@50 + latency

Saves per-config ranked lists (all queries) for TURBO3/TURBO4 so the hybrid
fusion harness (step 6) can compute end-to-end MRR with the real extension.

Run from repo root:
    uv run python scripts/poc_turboquant_recall.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_retrieval import (  # noqa: E402
    _cosine_similarity, _create_queries, _load_query_dataset)

CACHE = Path("eval/cache")
EXT = Path("poc/extensions/vector.dylib")
DB_PATH = Path("poc/data/vectors_eval.sqlite")
OUT_PATH = Path("poc/data/turboquant_recall.json")
RANKED_PATH = Path("poc/data/turboquant_ranked.npz")
MODEL_SLUG = "jinaai_jina_embeddings_v5_text_small"
DEPTH = 50
N_SUBSET = 500  # queries used for the exact-scan latency/recall comparison


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    conn.load_extension(str(EXT))
    return conn


def build_db(chunk_emb: np.ndarray) -> sqlite3.Connection:
    fresh = not DB_PATH.exists()
    conn = connect()
    if fresh:
        conn.execute("CREATE TABLE chunks_vec (embedding BLOB)")
        rows = [(chunk_emb[i].tobytes(),) for i in range(chunk_emb.shape[0])]
        conn.executemany("INSERT INTO chunks_vec (embedding) VALUES (?)", rows)
        conn.commit()
    conn.execute(
        "SELECT vector_init('chunks_vec', 'embedding', "
        "'type=FLOAT32,dimension=1024,distance=COSINE')")
    return conn


def scan(conn: sqlite3.Connection, module: str, query: bytes, k: int) -> list[int]:
    cur = conn.execute(
        f"SELECT rowid FROM vector_{module}('chunks_vec', 'embedding', ?, ?)",
        (query, k))
    return [r[0] for r in cur.fetchall()]


def recall(approx: list[list[int]], truth: list[list[int]], k: int) -> float:
    hits = [len(set(a[:k]) & set(t[:k])) / k for a, t in zip(approx, truth)]
    return float(np.mean(hits))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../dof_md")
    ap.add_argument("--queries", default="eval/dof_queries_v2.jsonl")
    args = ap.parse_args()

    docs, generated = _load_query_dataset(Path(args.corpus), Path(args.queries))
    queries = _create_queries(docs, generated)
    n_chunks = sum(len(d["chunks"]) for d in docs)
    print(f"{len(docs)} docs, {n_chunks:,} chunks, {len(queries):,} queries")

    chunk_emb = np.load(CACHE / f"{MODEL_SLUG}_chunks.npy").astype(np.float32)
    query_emb = np.load(CACHE / f"{MODEL_SLUG}_queries.npy").astype(np.float32)
    assert chunk_emb.shape[0] == n_chunks, "cache/chunk order mismatch"
    assert query_emb.shape[0] == len(queries)
    norms = np.linalg.norm(chunk_emb[:100], axis=1)
    print(f"embeddings {chunk_emb.shape}, sample norms {norms.min():.4f}–{norms.max():.4f}")

    # numpy fp32 ground truth (all queries)
    sims = _cosine_similarity(query_emb, chunk_emb)
    truth_all = [list(np.argsort(sims[i])[::-1][:DEPTH] + 1) for i in range(sims.shape[0])]  # rowid = idx+1

    rng = np.random.default_rng(42)
    subset = np.sort(rng.choice(len(queries), size=min(N_SUBSET, len(queries)), replace=False))
    print(f"SQL-scan comparison on {len(subset)}-query subset (seed 42)")

    conn = build_db(chunk_emb)
    results: dict[str, dict] = {}

    def run_config(name: str, module: str, qidx: np.ndarray) -> list[list[int]]:
        lat = []
        ranked = []
        for i in qidx:
            q = query_emb[i].tobytes()
            t0 = time.perf_counter()
            ranked.append(scan(conn, module, q, DEPTH))
            lat.append((time.perf_counter() - t0) * 1000)
        lat = np.array(lat)
        truth_sub = [truth_all[i] for i in qidx]
        results.setdefault(name, {}).update({
            "recall@10_vs_numpy": recall(ranked, truth_sub, 10),
            "recall@50_vs_numpy": recall(ranked, truth_sub, DEPTH),
            "latency_ms": {"p50": float(np.percentile(lat, 50)),
                           "p95": float(np.percentile(lat, 95)),
                           "mean": float(lat.mean())},
        })
        r = results[name]
        print(f"  {name:12s} R@10={r['recall@10_vs_numpy']:.4f} R@50={r['recall@50_vs_numpy']:.4f} "
              f"p50={r['latency_ms']['p50']:.1f}ms p95={r['latency_ms']['p95']:.1f}ms")
        return ranked

    # exact scan sanity (subset only — slow path)
    exact_ranked = run_config("full_scan", "full_scan", subset)
    results["full_scan"]["recall@10_vs_full_scan"] = 1.0
    results["full_scan"]["recall@50_vs_full_scan"] = 1.0

    ranked_full: dict[str, np.ndarray] = {}
    for qbits in (2, 3, 4):
        conn.execute("SELECT vector_quantize_cleanup('chunks_vec', 'embedding')")
        conn.execute(
            f"SELECT vector_quantize('chunks_vec', 'embedding', 'qtype=TURBO,qbits={qbits}')")
        mem = conn.execute(
            "SELECT vector_quantize_memory('chunks_vec', 'embedding')").fetchone()[0]
        conn.execute("SELECT vector_quantize_preload('chunks_vec', 'embedding')")
        name = f"turbo{qbits}"
        results[name] = {"quantized_bytes": mem}
        sub_ranked = run_config(name, "quantize_scan", subset)
        # agreement with the extension's own exact scan
        results[name]["recall@10_vs_full_scan"] = recall(sub_ranked, exact_ranked, 10)
        results[name]["recall@50_vs_full_scan"] = recall(sub_ranked, exact_ranked, DEPTH)
        # full-query run for the hybrid-fusion step (turbo3/4 only)
        if qbits in (3, 4):
            t0 = time.perf_counter()
            full = [scan(conn, "quantize_scan", query_emb[i].tobytes(), DEPTH)
                    for i in range(len(queries))]
            dt = time.perf_counter() - t0
            print(f"    full {len(queries):,}-query run: {dt:.1f}s "
                  f"({dt / len(queries) * 1000:.1f} ms/query)")
            results[name]["recall@10_all_queries"] = recall(full, truth_all, 10)
            results[name]["recall@50_all_queries"] = recall(full, truth_all, DEPTH)
            ranked_full[name] = np.array(full, dtype=np.int32)

    results["_meta"] = {
        "model": MODEL_SLUG, "n_chunks": int(chunk_emb.shape[0]),
        "n_queries": len(queries), "depth": DEPTH, "subset_size": len(subset),
        "db_bytes": DB_PATH.stat().st_size,
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))
    np.savez_compressed(RANKED_PATH, **ranked_full)
    print(f"\nWrote {OUT_PATH} and {RANKED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
