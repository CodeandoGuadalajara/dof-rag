"""One-off: build documents_fts (FTS5 external-content) on the full corpus db.

Batched by document_id ranges so we get progress logging and small
transactions (the embedding run reads this db concurrently in WAL mode).

NOTE: documents_fts is an EXTERNAL CONTENT table, so COUNT(*)/MAX(rowid)
against it scan the content table (documents), NOT the FTS index. Progress
is therefore tracked in a sidecar table `_fts_build_meta` instead.

Usage:
    uv run python scripts/build_fts_full.py \
        --corpus-db dof_db/dof_corpus_l3.sqlite [--batch 50000] [--reset]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus_store.db import connect, fetch_document_text  # noqa: E402

FTS_DDL = """CREATE VIRTUAL TABLE documents_fts USING fts5(
    markdown, content='documents', content_rowid='document_id',
    tokenize='unicode61 remove_diacritics 1'
)"""

META_DDL = ("CREATE TABLE IF NOT EXISTS _fts_build_meta"
            "(k TEXT PRIMARY KEY, v INTEGER)")


def get_meta(conn, key: str) -> int:
    row = conn.execute(
        "SELECT v FROM _fts_build_meta WHERE k = ?", (key,)).fetchone()
    return row[0] if row else 0


def set_meta(conn, key: str, value: int) -> None:
    conn.execute(
        "INSERT INTO _fts_build_meta(k, v) VALUES (?, ?) "
        "ON CONFLICT(k) DO UPDATE SET v = excluded.v", (key, value))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-db", required=True)
    ap.add_argument("--batch", type=int, default=50_000)
    ap.add_argument("--reset", action="store_true",
                    help="drop documents_fts and start over")
    args = ap.parse_args()

    t0 = time.time()
    conn = connect(args.corpus_db)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-2000000")  # ~2 GiB page cache
    conn.execute("PRAGMA temp_store=MEMORY")

    if args.reset:
        conn.execute("DROP TABLE IF EXISTS documents_fts")
        conn.execute("DROP TABLE IF EXISTS _fts_build_meta")
        conn.commit()
        print("dropped existing documents_fts", flush=True)

    has_fts = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='documents_fts'"
    ).fetchone()
    if not has_fts:
        conn.execute(FTS_DDL)
        conn.commit()
        print(f"created documents_fts ({time.time() - t0:.0f}s)", flush=True)
    conn.execute(META_DDL)
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    max_id = conn.execute("SELECT MAX(document_id) FROM documents").fetchone()[0]
    lo = get_meta(conn, "indexed_through")
    done = get_meta(conn, "indexed_count")
    print(f"documents: {total:,}  max_id: {max_id:,}  "
          f"already indexed: {done:,} (through id {lo:,})", flush=True)

    while lo < max_id:
        hi = lo + args.batch
        t = time.time()
        with conn:
            cur = conn.execute(
                "INSERT INTO documents_fts(rowid, markdown) "
                "SELECT document_id, markdown FROM documents "
                "WHERE document_id > ? AND document_id <= ? "
                "AND markdown != ''",
                (lo, hi),
            )
            done += cur.rowcount
            set_meta(conn, "indexed_through", hi)
            set_meta(conn, "indexed_count", done)
        el = time.time() - t0
        rate = done / el if el else 0
        eta = (total - done) / rate / 3600 if rate else 0
        print(f"  {done:,}/{total:,} docs ({rate:.0f} docs/s, ETA {eta:.1f}h, "
              f"batch {time.time() - t:.1f}s)", flush=True)
        lo = hi

    # 32 oversized docs are stored as segments (markdown = ''), reassemble
    if not get_meta(conn, "segmented_done"):
        seg_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT document_id FROM document_segments")]
        with conn:
            for doc_id in seg_ids:
                conn.execute(
                    "INSERT INTO documents_fts(rowid, markdown) VALUES (?, ?)",
                    (doc_id, fetch_document_text(conn, doc_id)),
                )
            done += len(seg_ids)
            set_meta(conn, "indexed_count", done)
            set_meta(conn, "segmented_done", 1)
        print(f"indexed {len(seg_ids)} segmented docs", flush=True)

    assert done == total, f"fts count {done} != documents {total}"
    # real index sanity: a MATCH query (COUNT(*) scans the content table!)
    n = conn.execute(
        "SELECT COUNT(*) FROM documents_fts"
        " WHERE documents_fts MATCH 'decreto'").fetchone()[0]
    assert n > 10_000, f"suspiciously few MATCH results: {n}"
    set_meta(conn, "complete", 1)
    conn.commit()
    print(f"FTS5 build complete: {done:,} docs in "
          f"{(time.time() - t0) / 3600:.2f}h ('decreto' matches: {n:,})",
          flush=True)
    conn.close()


if __name__ == "__main__":
    main()
