"""Resumable chunk embedding into a sign-quantized (binary) vector store.

Per the post-PoC decision (see docs/corpus-storage-architecture.md), the
full-corpus build stores jina binary embeddings: sign() runs in this
pipeline on the in-memory fp32 vector returned by llama.cpp and only the
packed 128-byte bit blob (1024 dims / 8) touches disk. No fp32 vectors are
ever written, sidestepping both the ~27 GB fp32 disk problem and the extra
quantization pass (sqlite-vector's vector_quantize requires stored fp32
blobs). Hamming scan (XOR+popcount) is the fastest exact scan option.

CRITICAL (jina via llama.cpp): chunks MUST be prefixed with "Document: "
and queries with "Query: " or embeddings silently degrade
(cosine 0.958 vs 0.9999 agreement with the reference implementation).

Chunk text comes from recipe reconstruction (fast, hash-verified at chunk
build time): decompress doc -> normalized text -> slice spans.

Resumability (the full run is ~14 days and will be interrupted):
- chunks are processed in chunk_id order and inserted contiguously, so
  resume = continue after MAX(chunk_id) already in chunk_vectors;
- every batch commits inside one transaction;
- vector_meta records model/prefix/packing config; a resume with a
  mismatched config aborts instead of silently mixing embeddings.

Usage:
    uv run python -m corpus_store.embed \
        --corpus-db poc/data/dof_corpus_l3.sqlite \
        --chunks-db poc/data/dof_chunks.sqlite \
        --vectors-db poc/data/dof_vectors_jina_binary.sqlite \
        --gguf ~/dof-gguf/jina-v5-small-retrieval-F16.gguf
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from rag_poc.chunker import DocPattern
from corpus_store.chunk_index import normalized_text, reconstruct
from corpus_store.db import connect, fetch_document_text

MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
PREFIX_DOCUMENT = "Document: "
PREFIX_QUERY = "Query: "
QUANTIZATION = "sign-packbits-big"  # bits = (fp32 >= 0), np.packbits bitorder='big'
DIMS = 1024

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS vector_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

INSERT_EVERY = 2048  # vectors per committed batch


def start_server(gguf: Path, ctx: int, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            "llama-server", "-m", str(gguf),
            "--embedding", "--port", str(port),
            "-c", str(ctx), "-b", "8192", "-ub", "4096",
            "--log-disable",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("llama-server did not become healthy")


def embed_batch(texts: list[str], port: int, retries: int = 6) -> np.ndarray:
    """fp32 embeddings from llama-server /v1/embeddings, with backoff.

    Retries ride out transient server hiccups during multi-day runs; on
    persistent failure the exception propagates and the run stops with all
    prior batches already committed (resume-safe).
    """
    body = json.dumps({"input": texts}).encode()
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/embeddings", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.load(resp)
            data["data"].sort(key=lambda d: d["index"])
            return np.array([d["embedding"] for d in data["data"]],
                            dtype=np.float32)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise AssertionError("unreachable")


def pack_binary(emb: np.ndarray) -> bytes:
    """sign() quantization: (emb >= 0) packed to DIMS/8 bytes."""
    return np.packbits(emb >= 0).tobytes()


def init_vectors_db(path: Path, meta: dict[str, str]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(SCHEMA)
    for k, v in meta.items():
        conn.execute("INSERT INTO vector_meta VALUES (?, ?)", (k, v))
    conn.commit()
    return conn


def check_meta(conn: sqlite3.Connection, expected: dict[str, str]) -> None:
    """Resume guard: refuse to mix embedding configs in one store."""
    stored = dict(conn.execute("SELECT key, value FROM vector_meta"))
    for k, v in expected.items():
        if k in stored and stored[k] != v:
            sys.exit(f"vector_meta mismatch on {k!r}: stored {stored[k]!r}, "
                     f"requested {v!r} — refusing to mix embedding configs")


def iter_pending(chunks: sqlite3.Connection, after_id: int):
    """(chunk_id, document_id, spans_json, pattern) in chunk_id order."""
    cur = chunks.execute(
        "SELECT chunk_id, document_id, spans_json, pattern FROM chunks"
        " WHERE chunk_id > ? ORDER BY chunk_id", (after_id,))
    while True:
        rows = cur.fetchmany(4096)
        if not rows:
            return
        yield from rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-db", required=True)
    ap.add_argument("--chunks-db", required=True)
    ap.add_argument("--vectors-db", required=True)
    ap.add_argument("--gguf", type=Path,
                    default=Path.home() / "dof-gguf/jina-v5-small-retrieval-F16.gguf")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="embed at most N chunks this run (pilot sizing)")
    args = ap.parse_args()

    meta = {
        "model": MODEL_ID,
        "gguf": str(args.gguf),
        "dims": str(DIMS),
        "quantization": QUANTIZATION,
        "prefix_document": PREFIX_DOCUMENT,
        "prefix_query": PREFIX_QUERY,
    }

    vectors_path = Path(args.vectors_db)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not vectors_path.exists()
    vectors = (init_vectors_db(vectors_path, meta) if fresh
               else sqlite3.connect(str(vectors_path)))
    if not fresh:
        check_meta(vectors, meta)
    last_id = vectors.execute(
        "SELECT COALESCE(MAX(chunk_id), 0) FROM chunk_vectors").fetchone()[0]

    corpus = connect(args.corpus_db)
    chunks = sqlite3.connect(args.chunks_db)
    total = chunks.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    pending_total = total - chunks.execute(
        "SELECT COUNT(*) FROM chunks WHERE chunk_id <= ?", (last_id,)).fetchone()[0]
    if args.max_chunks:
        pending_total = min(pending_total, args.max_chunks)
    print(f"{total:,} chunks in store, {last_id:,} already embedded, "
          f"embedding {pending_total:,}", flush=True)
    if pending_total == 0:
        return

    proc = start_server(args.gguf, args.ctx, args.port)
    try:
        # probe: dims sanity + prefix applied (the request is what it is;
        # the pilot's MRR spot check validates prefix correctness)
        probe = embed_batch([PREFIX_DOCUMENT + "probe"], args.port)
        if probe.shape[1] != DIMS:
            sys.exit(f"embedding dims {probe.shape[1]} != expected {DIMS}")

        buf: list[tuple[int, str]] = []  # (chunk_id, prefixed text)
        n_done = 0
        t0 = time.time()
        cur_doc = None
        c_text = ""
        for chunk_id, doc_id, spans_json, pattern in iter_pending(chunks, last_id):
            if doc_id != cur_doc:
                # first row seen per doc carries the doc-level pattern that
                # chunk_index used to build normalized text
                raw = fetch_document_text(corpus, doc_id)
                c_text = normalized_text(raw, DocPattern(pattern))
                cur_doc = doc_id
            text = reconstruct(json.loads(spans_json), c_text)
            buf.append((chunk_id, PREFIX_DOCUMENT + text))
            if len(buf) >= INSERT_EVERY:
                n_done += flush(vectors, buf, args.port, args.batch_size)
                log_progress(n_done, pending_total, t0)
            if n_done + len(buf) >= pending_total:
                break
        if buf and n_done < pending_total:
            del buf[pending_total - n_done:]
            n_done += flush(vectors, buf, args.port, args.batch_size)
            log_progress(n_done, pending_total, t0)
        vectors.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        size = vectors_path.stat().st_size
        dt = time.time() - t0
        print(f"\nDone: {n_done:,} vectors in {dt / 3600:.1f}h "
              f"({n_done / max(dt, 1):.2f} chunks/s), "
              f"db size {size / 2**20:.1f} MiB "
              f"({size / max(n_done + last_id, 1):.0f} B/vec incl. overhead)")
    finally:
        proc.terminate()
        proc.wait()
    vectors.close()


def flush(vectors: sqlite3.Connection, buf: list[tuple[int, str]],
          port: int, batch_size: int) -> int:
    """Embed the buffer in sub-batches and insert in one transaction.

    chunk_ids in the buffer are contiguous, so MAX(chunk_id) remains a
    valid resume checkpoint. Returns the number of vectors inserted.
    """
    rows: list[tuple[int, bytes]] = []
    for i in range(0, len(buf), batch_size):
        part = buf[i:i + batch_size]
        emb = embed_batch([t for _, t in part], port)
        rows.extend((cid, pack_binary(e)) for (cid, _), e in zip(part, emb))
    with vectors:
        vectors.executemany(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)", rows)
    n = len(buf)
    buf.clear()
    return n


def log_progress(done: int, total: int, t0: float) -> None:
    dt = time.time() - t0
    rate = done / max(dt, 1)
    eta_h = (total - done) / max(rate, 1e-9) / 3600
    print(f"  {done:,}/{total:,} vectors ({rate:.2f} chunks/s, "
          f"ETA {eta_h:.1f}h)", flush=True)


if __name__ == "__main__":
    main()
