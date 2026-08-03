"""Connection helpers for the corpus-store PoC.

Every connection to a compressed corpus database must load the sqlite-zstd
extension; sqlite-vector is loaded on demand for vector databases.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

EXT_DIR = Path(__file__).parent.parent / "poc" / "extensions"
ZSTD_EXT = EXT_DIR / "sqlitezstd.dylib"
VECTOR_EXT = EXT_DIR / "vector.dylib"


def connect(db_path: str | Path, *, vector: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    conn.load_extension(str(ZSTD_EXT))
    if vector:
        conn.load_extension(str(VECTOR_EXT))
    return conn


def fetch_document_text(conn: sqlite3.Connection, doc_id: int) -> str:
    """Document text, reassembling segmented oversized documents."""
    text = conn.execute(
        "SELECT markdown FROM documents WHERE document_id = ?", (doc_id,)
    ).fetchone()[0]
    if text:
        return text
    segs = conn.execute(
        "SELECT segment_text FROM document_segments WHERE document_id = ?"
        " ORDER BY segment_index", (doc_id,)).fetchall()
    return "".join(s[0] for s in segs)


def init_fresh_db(conn: sqlite3.Connection) -> None:
    """Pragmas that must be set before any table is created."""
    conn.execute("PRAGMA auto_vacuum = FULL")  # must precede WAL and any table
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
