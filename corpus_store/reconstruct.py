"""Benchmark query-time chunk reconstruction (the read path of the RAG).

Simulates: vector search returned chunk_ids -> fetch chunk recipes -> load
and decompress only those documents -> rebuild normalized text -> slice
spans -> verify hash. This is the latency that matters at query time.

Usage:
    uv run python -m corpus_store.reconstruct \
        --corpus-db poc/data/dof_corpus_l3.sqlite \
        --chunks-db poc/data/dof_chunks.sqlite [--n 500]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import time

import numpy as np

from rag_poc.chunker import DocPattern
from corpus_store.chunk_index import normalized_text, reconstruct
from corpus_store.db import connect, fetch_document_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-db", default="poc/data/dof_corpus_l3.sqlite")
    ap.add_argument("--chunks-db", required=True)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    corpus = connect(args.corpus_db)
    chunks = sqlite3.connect(args.chunks_db)

    rows = chunks.execute(
        "SELECT chunk_id, document_id, pattern, spans_json, chunk_hash"
        " FROM chunks").fetchall()
    rng = random.Random(42)
    sample = rng.sample(rows, min(args.n, len(rows)))

    # warm-up
    for r in sample[:20]:
        fetch_document_text(corpus, r[1])

    lat_doc, lat_norm, lat_rebuild = [], [], []
    n_ok = 0
    for chunk_id, doc_id, pattern, spans_json, chunk_hash in sample:
        t0 = time.perf_counter()
        raw = fetch_document_text(corpus, doc_id)
        t1 = time.perf_counter()
        c = normalized_text(raw, DocPattern(pattern))
        t2 = time.perf_counter()
        text = reconstruct(json.loads(spans_json), c)
        t3 = time.perf_counter()
        if hashlib.sha256(text.encode("utf-8")).digest() == chunk_hash:
            n_ok += 1
        lat_doc.append((t1 - t0) * 1000)
        lat_norm.append((t2 - t1) * 1000)
        lat_rebuild.append((t3 - t2) * 1000)

    total = np.array(lat_doc) + np.array(lat_norm) + np.array(lat_rebuild)
    print(f"reconstructed {n_ok}/{len(sample)} chunks with verified hash")
    for name, arr in [("doc fetch+decompress", lat_doc), ("normalize", lat_norm),
                      ("rebuild", lat_rebuild), ("total", total)]:
        a = np.array(arr)
        print(f"  {name:22s} p50={np.percentile(a, 50):6.2f}ms "
              f"p95={np.percentile(a, 95):6.2f}ms max={a.max():7.1f}ms")

    # neighbor expansion: fetch adjacent chunks of the same document
    t0 = time.perf_counter()
    for _, doc_id, *_ in sample[:100]:
        chunks.execute(
            "SELECT chunk_index, spans_json FROM chunks WHERE document_id = ?"
            " ORDER BY chunk_index", (doc_id,)).fetchall()
    print(f"  neighbor fetch (100 docs) {(time.perf_counter() - t0) * 10:.2f}ms avg")


if __name__ == "__main__":
    main()
