"""Build the sqlite-vec bit[1024] search store from chunk_vectors.

Resumable: skips chunk_ids already present (MAX(rowid) of chunk_vec).
Can be run against a partial chunk_vectors table while embeddings are
still being written, then re-run after the embedding run completes to
top it off.

Usage:
    uv run python scripts/build_vec0_full.py \
        --vectors-db dof_db/dof_vectors_jina_binary.sqlite \
        --vec0-db dof_db/dof_vec0_jina_binary.sqlite
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import sqlite_vec

DDL = "CREATE VIRTUAL TABLE chunk_vec USING vec0(embedding bit[1024])"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors-db", required=True)
    ap.add_argument("--vec0-db", required=True)
    args = ap.parse_args()

    vec0_path = Path(args.vec0_db)
    conn = sqlite3.connect(str(vec0_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    fresh = not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='chunk_vec'").fetchone()
    if fresh:
        conn.execute(DDL)
        conn.commit()
        print("created chunk_vec vec0(bit[1024])", flush=True)

    lo = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM chunk_vec")
    lo = lo.fetchone()[0]
    if lo:
        print(f"resuming after rowid {lo:,}", flush=True)

    vectors = sqlite3.connect(args.vectors_db)
    total = vectors.execute(
        "SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id > ?",
        (lo,)).fetchone()[0]
    print(f"to insert: {total:,}", flush=True)

    rows = vectors.execute(
        "SELECT chunk_id, embedding FROM chunk_vectors"
        " WHERE chunk_id > ? ORDER BY chunk_id", (lo,))
    t0 = time.time()
    n = 0
    while True:
        batch = rows.fetchmany(10_000)
        if not batch:
            break
        conn.executemany(
            "INSERT INTO chunk_vec(rowid, embedding) VALUES (?, vec_bit(?))",
            batch)
        conn.commit()
        n += len(batch)
        rate = n / (time.time() - t0)
        eta = (total - n) / rate / 60 if rate else 0
        print(f"  vec0 insert {n:,}/{total:,} ({rate:.0f}/s, ETA {eta:.1f} min)",
              flush=True)

    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    have = conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0]
    print(f"done: {have:,} vectors in vec0 store "
          f"({time.time() - t0:.0f}s)", flush=True)
    conn.close()
    vectors.close()


if __name__ == "__main__":
    main()
