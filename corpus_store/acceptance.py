"""Acceptance benchmarks for the corpus-store PoC.

Checks the criteria from docs/corpus-storage-architecture.md against a
built corpus database:

1. compression ratio >= 8x (file level, after VACUUM + checkpoint)
2. exact round-trip equality with source Markdown (sha256 per document,
   including reassembly of segmented oversized documents)
3. p95 single-document read/decompression latency < 50 ms
4. FTS5 external-content creation, rebuild, BM25 queries, and index size
5. database shrinks after maintenance + VACUUM
6. versions recorded in corpus_meta

Usage:
    uv run python -m corpus_store.acceptance --db poc/data/dof_corpus_l3.sqlite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from corpus_store.db import connect

FTS_DDL = """CREATE VIRTUAL TABLE documents_fts USING fts5(
    markdown, content='documents', content_rowid='document_id'
)"""


def file_size(db: Path) -> int:
    size = db.stat().st_size
    for suffix in ("-wal", "-shm"):
        p = db.with_name(db.name + suffix)
        if p.exists():
            size += p.stat().st_size
    return size


def checkpoint_vacuum(conn) -> None:
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../dof_md")
    ap.add_argument("--manifest", default="poc/data/manifest_10k.jsonl")
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    db = Path(args.db)
    out = Path(args.out or f"poc/data/acceptance_{db.stem}.json")
    corpus = Path(args.corpus)
    manifest = [json.loads(l) for l in Path(args.manifest).read_text().splitlines()]
    results: dict = {"db": str(db), "checks": {}}

    conn = connect(db)

    # ---- 0. versions recorded ----
    meta = dict(conn.execute("SELECT key, value FROM corpus_meta").fetchall())
    results["corpus_meta"] = meta
    print("corpus_meta:", meta)

    # ---- 1. compression ratio ----
    total_text = conn.execute(
        "SELECT SUM(byte_length) FROM documents").fetchone()[0]
    checkpoint_vacuum(conn)
    size = file_size(db)
    results["checks"]["compression"] = {
        "source_bytes": total_text, "db_file_bytes": size,
        "ratio": total_text / size,
    }
    print(f"[1] compression: {total_text / 2**20:.1f} MiB -> {size / 2**20:.1f} MiB "
          f"= {total_text / size:.2f}x (target >= 8x)")

    # ---- 2. round-trip equality ----
    t0 = time.time()
    n_ok = n_seg = 0
    failures = []
    for rec in manifest:
        rel = rec["relpath"]
        raw = (corpus / rel).read_bytes()
        expect = hashlib.sha256(raw).hexdigest()
        row = conn.execute(
            "SELECT document_id, markdown, sha256 FROM documents WHERE path = ?",
            (rel,)).fetchone()
        if row is None:
            failures.append((rel, "missing"))
            continue
        doc_id, text, stored_hash = row
        if stored_hash.hex() != expect:
            failures.append((rel, "stored sha256 mismatch"))
            continue
        if text:
            got = text.encode("utf-8")
        else:  # segmented oversized document
            segs = conn.execute(
                "SELECT segment_text FROM document_segments WHERE document_id = ?"
                " ORDER BY segment_index", (doc_id,)).fetchall()
            got = "".join(s[0] for s in segs).encode("utf-8")
            n_seg += 1
        if hashlib.sha256(got).hexdigest() == expect:
            n_ok += 1
        else:
            failures.append((rel, "round-trip mismatch"))
    results["checks"]["roundtrip"] = {
        "docs_ok": n_ok, "segmented_docs": n_seg, "failures": failures,
        "seconds": time.time() - t0,
    }
    print(f"[2] round-trip: {n_ok:,} OK ({n_seg} segmented), "
          f"{len(failures)} failures ({time.time() - t0:.0f}s)")

    # ---- 3. point-read latency ----
    ids = [r[0] for r in conn.execute(
        "SELECT document_id FROM documents WHERE markdown != ''")]
    rng = random.Random(42)
    sample = rng.sample(ids, min(500, len(ids)))
    # warm-up
    for i in sample[:20]:
        conn.execute("SELECT markdown FROM documents WHERE document_id = ?",
                     (i,)).fetchone()
    lat = []
    for i in sample:
        t1 = time.perf_counter()
        conn.execute("SELECT markdown FROM documents WHERE document_id = ?",
                     (i,)).fetchone()
        lat.append((time.perf_counter() - t1) * 1000)
    lat = np.array(lat)
    results["checks"]["read_latency_ms"] = {
        "n": len(sample), "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)), "max": float(lat.max()),
    }
    print(f"[3] read latency: p50={np.percentile(lat, 50):.2f}ms "
          f"p95={np.percentile(lat, 95):.2f}ms max={lat.max():.1f}ms (target p95 < 50ms)")

    # ---- 4. FTS5 external content + size ----
    has_fts = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'documents_fts'").fetchone()
    if not has_fts:
        t2 = time.time()
        conn.execute(FTS_DDL)
        conn.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        conn.commit()
        fts_build_s = time.time() - t2
    else:
        fts_build_s = None
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    size_with_fts = file_size(db)
    fts_bytes = size_with_fts - size
    # BM25 smoke queries
    bm25_ok = True
    for q in ("declaracion", "licitacion mantenimiento", "presupuesto egresos"):
        n = conn.execute(
            "SELECT COUNT(*) FROM (SELECT rowid FROM documents_fts"
            " WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) LIMIT 10)",
            (q,)).fetchone()[0]
        if n == 0:
            bm25_ok = False
    snip = conn.execute(
        "SELECT snippet(documents_fts, 0, '[', ']', '...', 8) FROM documents_fts"
        " WHERE documents_fts MATCH 'declaracion' ORDER BY bm25(documents_fts)"
        " LIMIT 1").fetchone()
    results["checks"]["fts"] = {
        "build_seconds": fts_build_s, "fts_bytes": fts_bytes,
        "fts_ratio_of_text": fts_bytes / total_text,
        "bm25_queries_return_hits": bm25_ok,
        "snippet_sample": snip[0][:120] if snip else None,
    }
    print(f"[4] FTS5: built in {fts_build_s and round(fts_build_s, 1)}s, "
          f"index {fts_bytes / 2**20:.1f} MiB ({fts_bytes / total_text:.2f}x text), "
          f"BM25 hits: {bm25_ok}")

    # ---- 5. VACUUM after maintenance ----
    conn.execute("SELECT zstd_incremental_maintenance(600, 1.0)")
    conn.commit()
    before = file_size(db)
    checkpoint_vacuum(conn)
    after = file_size(db)
    results["checks"]["vacuum"] = {"before_bytes": before, "after_bytes": after}
    print(f"[5] maintenance+VACUUM: {before / 2**20:.1f} -> {after / 2**20:.1f} MiB")

    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
