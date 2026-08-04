"""Pilot: validate the binary (sign) vector store at ~100k chunks.

Validates the production storage path before committing to the ~14-day
full-corpus embedding run:

1. Real on-disk size of the bit-vector store (both the plain checkpoint
   table written by corpus_store.embed and a sqlite-vec vec0 bit[1024]
   search table), extrapolated to ~6.67M chunks.
2. Hamming scan latency via sqlite-vec (vec0 MATCH, k=50) with a numpy
   XOR+popcount reference, extrapolated to the full corpus.
3. Bit-exactness spot check: re-embedded chunks sign-pack to (nearly) the
   stored blobs.
4. Spot MRR: eval queries whose gold document is inside the 10k-doc PoC
   corpus (13 of 500 eval docs), chunk-level hamming retrieval collapsed
   to documents. Reference anchor from the PoC (different corpus/mix, so
   not a strict target): jina-binary doc-collapsed MRR 0.5452 over the
   499-doc corpus.

Run from repo root (llama-server must be on PATH), after corpus_store.embed
has populated the pilot vectors db:
    uv run python scripts/pilot_binary_vectors.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import sqlite_vec

sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus_store.db import connect  # noqa: E402
from corpus_store.embed import (  # noqa: E402
    DIMS, PREFIX_DOCUMENT, PREFIX_QUERY, embed_batch, pack_binary,
    start_server)
from corpus_store.chunk_index import normalized_text, reconstruct  # noqa: E402
from corpus_store.db import fetch_document_text  # noqa: E402
from rag_poc.chunker import DocPattern  # noqa: E402

K = 50
FULL_CORPUS_CHUNKS = 6_670_000  # 101,351 chunks / 10k docs * 657,867 docs


def build_vec0(vectors: sqlite3.Connection, vec0_path: Path) -> sqlite3.Connection:
    """Materialize the sqlite-vec bit[1024] search table from chunk_vectors."""
    fresh = not vec0_path.exists()
    conn = sqlite3.connect(str(vec0_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    if fresh:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(
            "CREATE VIRTUAL TABLE chunk_vec USING vec0(embedding bit[1024])")
        rows = vectors.execute(
            "SELECT chunk_id, embedding FROM chunk_vectors ORDER BY chunk_id")
        batch = []
        t0 = time.time()
        n = 0
        while True:
            chunk = rows.fetchmany(10000)
            if not chunk:
                break
            batch.extend(chunk)
            conn.executemany(
                "INSERT INTO chunk_vec(rowid, embedding) VALUES (?, vec_bit(?))",
                batch)
            conn.commit()
            n += len(batch)
            batch.clear()
            print(f"  vec0 insert {n:,} ({time.time() - t0:.0f}s)", flush=True)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return conn


def knn_vec0(conn: sqlite3.Connection, qblob: bytes, k: int) -> list[tuple[int, float]]:
    return conn.execute(
        "SELECT rowid, distance FROM chunk_vec"
        " WHERE embedding MATCH vec_bit(?) AND k = ?", (qblob, k)).fetchall()


def doc_collapse(ranked: list[tuple[int, float]],
                 chunk_doc: dict[int, int]) -> list[int]:
    """Chunk ranked list -> doc ids ordered by best (min) hamming distance."""
    best: dict[int, float] = {}
    for chunk_id, dist in ranked:
        d = chunk_doc[chunk_id]
        if d not in best or dist < best[d]:
            best[d] = dist
    return [d for d, _ in sorted(best.items(), key=lambda kv: kv[1])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-db", default="poc/data/dof_corpus_l3.sqlite")
    ap.add_argument("--chunks-db", default="poc/data/dof_chunks.sqlite")
    ap.add_argument("--vectors-db", default="poc/data/pilot_vectors.sqlite")
    ap.add_argument("--vec0-db", default="poc/data/pilot_vec0.sqlite")
    ap.add_argument("--manifest", default="poc/data/manifest_10k.jsonl")
    ap.add_argument("--queries", default="eval/dof_queries_v2.jsonl")
    ap.add_argument("--gguf", type=Path,
                    default=Path.home() / "dof-gguf/jina-v5-small-retrieval-F16.gguf")
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--out", default="poc/data/pilot_binary_results.json")
    args = ap.parse_args()

    vectors = sqlite3.connect(args.vectors_db)
    n_vecs = vectors.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
    print(f"{n_vecs:,} embedded chunks")
    meta = dict(vectors.execute("SELECT key, value FROM vector_meta"))
    print(f"config: {meta}")

    results: dict = {"n_vectors": n_vecs, "meta": meta}

    # --- 1. on-disk size -------------------------------------------------
    plain_size = Path(args.vectors_db).stat().st_size
    print(f"\n[1] plain checkpoint db: {plain_size / 2**20:.1f} MiB "
          f"({plain_size / n_vecs:.0f} B/vec -> "
          f"{plain_size / n_vecs * FULL_CORPUS_CHUNKS / 2**30:.2f} GiB full corpus)")
    vec0 = build_vec0(vectors, Path(args.vec0_db))
    vec0_size = Path(args.vec0_db).stat().st_size
    print(f"    vec0 search db:      {vec0_size / 2**20:.1f} MiB "
          f"({vec0_size / n_vecs:.0f} B/vec -> "
          f"{vec0_size / n_vecs * FULL_CORPUS_CHUNKS / 2**30:.2f} GiB full corpus)")
    results["size"] = {
        "plain_bytes": plain_size, "vec0_bytes": vec0_size,
        "plain_b_per_vec": plain_size / n_vecs,
        "vec0_b_per_vec": vec0_size / n_vecs,
        "vec0_full_corpus_gib": vec0_size / n_vecs * FULL_CORPUS_CHUNKS / 2**30,
    }

    chunks = sqlite3.connect(args.chunks_db)
    chunk_doc = dict(chunks.execute("SELECT chunk_id, document_id FROM chunks"))

    # --- 2. hamming scan latency ------------------------------------------
    print(f"\n[2] hamming scan latency (k={K}) over {n_vecs:,} vectors")
    rng = np.random.default_rng(42)
    probe_ids = rng.choice(n_vecs, size=min(64, n_vecs), replace=False) + 1
    probes = [vectors.execute(
        "SELECT embedding FROM chunk_vectors WHERE chunk_id = ?",
        (int(i),)).fetchone()[0] for i in probe_ids]
    lat = []
    for q in probes:
        t0 = time.perf_counter()
        knn_vec0(vec0, q, K)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    print(f"    sqlite-vec: mean {lat.mean():.2f} ms, p95 "
          f"{np.percentile(lat, 95):.2f} ms "
          f"(x{FULL_CORPUS_CHUNKS / n_vecs:.0f} -> "
          f"~{lat.mean() * FULL_CORPUS_CHUNKS / n_vecs / 1000:.2f} s/query full corpus)")
    mat = np.unpackbits(np.frombuffer(
        b"".join(b for b, in vectors.execute(
            "SELECT embedding FROM chunk_vectors ORDER BY chunk_id")),
        dtype=np.uint8)).reshape(-1, DIMS)
    qm = np.unpackbits(np.frombuffer(b"".join(probes), dtype=np.uint8)).reshape(-1, DIMS)
    t0 = time.perf_counter()
    for q in qm[:16]:
        d = np.count_nonzero(mat != q, axis=1)
        np.argpartition(d, K)[:K]
    np_ms = (time.perf_counter() - t0) / 16 * 1000
    print(f"    numpy ref:  mean {np_ms:.2f} ms")
    results["latency"] = {
        "vec0_ms_mean": float(lat.mean()), "vec0_ms_p95": float(np.percentile(lat, 95)),
        "numpy_ms_mean": np_ms,
        "vec0_full_corpus_s_est": lat.mean() * FULL_CORPUS_CHUNKS / n_vecs / 1000,
    }

    # --- 3 + 4. server-dependent checks -----------------------------------
    proc = start_server(args.gguf, 8192, args.port)
    try:
        embed_batch(["warmup"], args.port)

        # --- 3. bit-exactness spot check ----------------------------------
        print("\n[3] bit-exactness: re-embed 64 random chunks, compare blobs")
        corpus = connect(args.corpus_db)
        spot_ids = (rng.choice(n_vecs, size=64, replace=False) + 1).tolist()
        rows = chunks.execute(
            "SELECT chunk_id, document_id, spans_json, pattern FROM chunks"
            f" WHERE chunk_id IN ({','.join('?' * len(spot_ids))})",
            [int(i) for i in spot_ids]).fetchall()
        texts = []
        for chunk_id, doc_id, spans_json, pattern in rows:
            raw = fetch_document_text(corpus, doc_id)
            c = normalized_text(raw, DocPattern(pattern))
            texts.append((chunk_id, PREFIX_DOCUMENT + reconstruct(
                json.loads(spans_json), c)))
        emb = embed_batch([t for _, t in texts], args.port)
        dists = []
        for (chunk_id, _), e in zip(texts, emb):
            stored = vectors.execute(
                "SELECT embedding FROM chunk_vectors WHERE chunk_id = ?",
                (chunk_id,)).fetchone()[0]
            a = np.unpackbits(np.frombuffer(pack_binary(e), dtype=np.uint8))
            b = np.unpackbits(np.frombuffer(stored, dtype=np.uint8))
            dists.append(int(np.count_nonzero(a != b)))
        dists = np.array(dists)
        print(f"    hamming(re-embedded, stored): max {dists.max()}, "
              f"mean {dists.mean():.2f} of {DIMS} bits")
        results["bit_exactness"] = {"max": int(dists.max()),
                                    "mean": float(dists.mean())}

        # --- 4. spot MRR over pilot corpus ---------------------------------
        print("\n[4] spot MRR (gold docs present in the 10k PoC corpus)")
        manifest_paths = {json.loads(l)["relpath"]
                          for l in Path(args.manifest).read_text().splitlines()}
        corpus = connect(args.corpus_db)
        path_doc = dict(corpus.execute("SELECT path, document_id FROM documents"))
        eval_queries = []
        for line in Path(args.queries).read_text().splitlines():
            rec = json.loads(line)
            if rec.get("error") or not rec.get("queries"):
                continue
            if rec["relpath"] in manifest_paths:
                for q in rec["queries"]:
                    eval_queries.append((q["query"], path_doc[rec["relpath"]],
                                         q["type"]))
        print(f"    {len(eval_queries)} queries over "
              f"{len({d for _, d, _ in eval_queries})} gold docs")
        q_emb = embed_batch([PREFIX_QUERY + q for q, _, _ in eval_queries],
                            args.port)
        rr, r1, r10 = [], 0, 0
        by_type: dict[str, list[float]] = {}
        for (qtext, gold_doc, qtype), e in zip(eval_queries, q_emb):
            ranked = knn_vec0(vec0, pack_binary(e), K)
            docs = doc_collapse(ranked, chunk_doc)
            rank = docs.index(gold_doc) + 1 if gold_doc in docs else None
            rr.append(1.0 / rank if rank else 0.0)
            by_type.setdefault(qtype, []).append(rr[-1])
            r1 += rank == 1
            r10 += rank is not None and rank <= 10
        mrr = float(np.mean(rr))
        print(f"    MRR {mrr:.4f}  R@1 {r1 / len(rr):.4f}  "
              f"R@10 {r10 / len(rr):.4f}  (n={len(rr)})")
        for t, v in sorted(by_type.items()):
            print(f"      {t:18s} MRR {np.mean(v):.4f} (n={len(v)})")
        results["spot_mrr"] = {
            "mrr": mrr, "r_at_1": r1 / len(rr), "r_at_10": r10 / len(rr),
            "n_queries": len(rr),
            "reference_poc_jina_binary_doc_collapsed_mrr": 0.5452,
            "note": "only 13/500 eval docs are inside the 10k pilot corpus; "
                    "harder distractor set than the PoC, small n",
        }
    finally:
        proc.terminate()
        proc.wait()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
