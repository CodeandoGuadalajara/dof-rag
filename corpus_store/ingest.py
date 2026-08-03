"""Ingest the sampled DOF corpus into a compressed SQLite corpus store.

Implements the layout from docs/corpus-storage-architecture.md:

- documents: one row per Markdown file, metadata indexes created BEFORE
  transparent compression is enabled;
- document_segments: oversized documents (> --segment-threshold, default
  32 MiB) split into ordered segments with byte offsets;
- corpus_meta: corpus version, source manifest, build settings;
- ingestion_log: one row per committed batch (resumability observability).

Ingestion is idempotent by documents.path, so an interrupted run resumes
safely: re-running the command skips already-ingested paths.

Usage:
    uv run python -m corpus_store.ingest \
        --manifest poc/data/manifest_10k.jsonl \
        --db poc/data/dof_corpus_l3.sqlite --level 3 [--max-docs 5000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

from corpus_store.db import connect, init_fresh_db

CORPUS_VERSION = "poc-10k-seed42"  # default; full builds pass --corpus-version
BATCH_SIZE = 256

SOURCE_RE = re.compile(r"^[a-z0-9_]+$")  # safe for dict_chooser group names

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL DEFAULT 'dof',
    year INTEGER NOT NULL,
    publication_date TEXT,
    section TEXT,
    markdown TEXT NOT NULL,
    byte_length INTEGER NOT NULL,
    sha256 BLOB NOT NULL,
    corpus_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_source_idx ON documents(source);
CREATE INDEX IF NOT EXISTS documents_year_idx ON documents(year);
CREATE INDEX IF NOT EXISTS documents_section_idx ON documents(section);

CREATE TABLE IF NOT EXISTS document_segments (
    document_id INTEGER NOT NULL REFERENCES documents(document_id),
    segment_index INTEGER NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    segment_text TEXT NOT NULL,
    PRIMARY KEY (document_id, segment_index)
);

CREATE TABLE IF NOT EXISTS corpus_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ingestion_log (
    batch_id INTEGER PRIMARY KEY,
    finished_at TEXT NOT NULL DEFAULT (datetime('now')),
    docs_inserted INTEGER NOT NULL,
    bytes_inserted INTEGER NOT NULL
);
"""

# dict_chooser is source-aware so future sources (constitucion, state laws)
# train their own dictionaries instead of diluting the DOF ones. Changing
# the chooser expression is safe: decompression uses the stored dict ids.
ENABLE_ZSTD = """SELECT zstd_enable_transparent('{{
    "table": "documents", "column": "markdown",
    "compression_level": {level},
    "dict_chooser": "printf(''%s_%d'', source, year)"
}}')"""
ENABLE_ZSTD_SEGMENTS = """SELECT zstd_enable_transparent('{{
    "table": "document_segments", "column": "segment_text",
    "compression_level": {level},
    "dict_chooser": "''[nodict]''"
}}')"""  # [nodict]: no dictionary — few huge rows break ZDICT training


def already_compressed(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = '_documents_zstd'").fetchone()
    return row is not None


def ingest_batch(conn: sqlite3.Connection, corpus: Path, batch: list[dict],
                 segment_threshold: int, source: str = "dof",
                 corpus_version: str = CORPUS_VERSION) -> tuple[int, int]:
    """Insert one batch inside a single transaction. Returns (docs, bytes).

    documents.path is namespaced per source: DOF keeps its historical
    relpaths (1999/01/...); future sources must prefix their paths with
    '<source>/' (e.g. 'constitucion/...') so path stays globally UNIQUE.
    """
    n_docs = n_bytes = 0
    with conn:
        for rec in batch:
            rel = rec["relpath"]
            exists = conn.execute(
                "SELECT 1 FROM documents WHERE path = ?", (rel,)).fetchone()
            if exists:
                continue  # resume: already ingested
            raw = (corpus / rel).read_bytes()
            digest = hashlib.sha256(raw).digest()
            size = len(raw)
            text = raw.decode("utf-8")
            if size > segment_threshold:
                # metadata row with empty text; content lives in segments
                cur = conn.execute(
                    "INSERT INTO documents (path, source, year, publication_date, section,"
                    " markdown, byte_length, sha256, corpus_version)"
                    " VALUES (?, ?, ?, ?, ?, '', ?, ?, ?)",
                    (rel, source, rec["year"], rec["publication_date"], rec["section"],
                     size, digest, corpus_version))
                doc_id = cur.lastrowid
                for i, start in enumerate(range(0, size, segment_threshold)):
                    seg = text[start:start + segment_threshold]
                    conn.execute(
                        "INSERT INTO document_segments (document_id, segment_index,"
                        " start_offset, end_offset, segment_text) VALUES (?, ?, ?, ?, ?)",
                        (doc_id, i, start, start + len(seg), seg))
            else:
                conn.execute(
                    "INSERT INTO documents (path, source, year, publication_date, section,"
                    " markdown, byte_length, sha256, corpus_version)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (rel, source, rec["year"], rec["publication_date"], rec["section"],
                     text, size, digest, corpus_version))
            n_docs += 1
            n_bytes += size
        conn.execute(
            "INSERT INTO ingestion_log (docs_inserted, bytes_inserted) VALUES (?, ?)",
            (n_docs, n_bytes))
    return n_docs, n_bytes


def enable_compression(conn: sqlite3.Connection, level: int) -> None:
    conn.commit()  # zstd_enable_transparent needs a clean transaction state
    conn.execute(ENABLE_ZSTD.format(level=level))
    conn.execute(ENABLE_ZSTD_SEGMENTS.format(level=level))
    conn.commit()
    # The maintenance todo query GROUPs BY the dict_chooser expression over
    # all uncompressed rows; without this index SQLite spills the whole
    # uncompressed corpus to a temp b-tree (26+ GiB at full scale — fills
    # the disk, SQLITE_FULL). The expression must match dict_chooser.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS _group_dict_idx ON _documents_zstd"
        "(_markdown_dict, printf('%s_%d', source, year))")
    conn.commit()


def pretrain_dicts(conn: sqlite3.Connection, source: str = "dof",
                   max_sample_bytes: int = int(1.8 * 2**30)) -> None:
    """Train per-source-year dicts via the year index, before maintenance.

    The extension's own training path runs one unindexed query per chooser
    group (full-table scan each), and reservoir-samples ALL rows of groups
    larger than ~2 GiB, which exceeds ZDICT's 2 GB sample limit (the 2011
    group is 2.35 GiB). Indexed per-year queries with a reservoir cap avoid
    both. Idempotent: skips chooser keys that already have a dict. Mirrors
    the extension defaults: dict target = 1% of group bytes (min 5000).
    """
    groups = conn.execute(
        "SELECT year, COUNT(*), SUM(byte_length) FROM documents"
        " WHERE source = ? GROUP BY year", (source,)).fetchall()
    for year, count, total in groups:
        key = f"{source}_{year}"
        if conn.execute("SELECT 1 FROM _zstd_dicts WHERE chooser_key = ?",
                        (key,)).fetchone():
            continue
        avg = total / count
        dict_target = max(int(0.01 * total), 5000)
        samples = min(count, int(max_sample_bytes / avg))
        t0 = time.time()
        conn.execute(
            "SELECT zstd_train_dict_and_save(markdown, ?, ?, ?)"
            " FROM documents WHERE year = ? AND source = ?",
            (dict_target, float(samples), key, year, source)).fetchone()
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        print(f"  trained {key}: target={dict_target} samples={samples}"
              f" ({time.time() - t0:.0f}s)", flush=True)


def run_maintenance(conn: sqlite3.Connection, pass_seconds: int = 1800,
                    max_passes: int = 24) -> None:
    """Dict training + recompression until done, checkpointing between passes.

    One long call risks a giant WAL (disk-full on the full corpus) and loses
    all progress on failure; bounded passes with wal_checkpoint(TRUNCATE)
    keep peak disk use low and every pass resumable. The function returns 0
    when no work remains.
    """
    for i in range(max_passes):
        t0 = time.time()
        remaining = conn.execute(
            f"SELECT zstd_incremental_maintenance({pass_seconds}, 1.0)"
        ).fetchone()[0]
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        print(f"  maintenance pass {i + 1}: {time.time() - t0:.0f}s, "
              f"remaining={remaining}", flush=True)
        if not remaining:
            return
    print(f"WARN: maintenance not finished after {max_passes} passes; "
          "re-run to continue", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../dof_md")
    ap.add_argument("--manifest", default="poc/data/manifest_10k.jsonl")
    ap.add_argument("--db", required=True)
    ap.add_argument("--level", type=int, default=3, choices=[3, 19])
    ap.add_argument("--segment-threshold", type=int, default=32 * 2**20)
    ap.add_argument("--source", default="dof",
                    help="corpus source id; recorded per row and used by the"
                         " zstd dict_chooser (groups per source_year). Future"
                         " sources must namespace their manifest relpaths as"
                         " '<source>/...' to keep documents.path UNIQUE.")
    ap.add_argument("--corpus-version", default=CORPUS_VERSION,
                    help="recorded on every row and in corpus_meta")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="ingest only the first N manifest docs (resume testing)")
    args = ap.parse_args()

    if not SOURCE_RE.match(args.source):
        sys.exit(f"--source must match {SOURCE_RE.pattern!r} "
                 f"(it is used in zstd dict group names), got {args.source!r}")

    corpus = Path(args.corpus)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(l) for l in Path(args.manifest).read_text().splitlines()]
    if args.max_docs:
        manifest = manifest[: args.max_docs]

    fresh = not db_path.exists()
    conn = connect(db_path)
    if fresh:
        init_fresh_db(conn)
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO corpus_meta VALUES ('corpus_version', ?)",
                     (args.corpus_version,))
        conn.execute("INSERT INTO corpus_meta VALUES ('source', ?)",
                     (args.source,))
        conn.execute("INSERT INTO corpus_meta VALUES ('source_manifest', ?)",
                     (str(args.manifest),))
        conn.execute("INSERT INTO corpus_meta VALUES ('compression_level', ?)",
                     (str(args.level),))
        conn.commit()

    t0 = time.time()
    total_docs = total_bytes = 0
    for i in range(0, len(manifest), BATCH_SIZE):
        d, b = ingest_batch(conn, corpus, manifest[i:i + BATCH_SIZE],
                            args.segment_threshold, args.source,
                            args.corpus_version)
        total_docs += d
        total_bytes += b
        done = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        print(f"  {done:,}/{len(manifest):,} docs "
              f"({time.time() - t0:.0f}s)", flush=True)

    n_segments = conn.execute("SELECT COUNT(*) FROM document_segments").fetchone()[0]
    print(f"Ingested {total_docs:,} new docs ({total_bytes / 2**20:.1f} MiB), "
          f"{n_segments} oversized segments")

    if not already_compressed(conn):
        print(f"Enabling transparent compression at level {args.level} ...")
        t1 = time.time()
        enable_compression(conn, args.level)
        print(f"  compression enabled in {time.time() - t1:.0f}s")
        pretrain_dicts(conn, args.source)
        print("Running incremental maintenance (recompression)...")
        t2 = time.time()
        run_maintenance(conn)
        print(f"  maintenance in {time.time() - t2:.0f}s")
    else:
        print("Database already compressed; running maintenance on new rows...")
        conn.commit()
        pretrain_dicts(conn, args.source)
        run_maintenance(conn)

    print(f"DB file size: {db_path.stat().st_size / 2**20:.1f} MiB")
    conn.close()


if __name__ == "__main__":
    main()
