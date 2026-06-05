"""SQLite + sqlite-vec + FTS5 hybrid storage."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import sqlite_vec
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install sqlite-vec") from exc

from rag_poc.config import DB_PATH, EMBED_DIM


class RAGDatabase:
    """Manages the SQLite schema with FTS5 + sqlite-vec."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()

        # Documents metadata
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                title TEXT,
                url TEXT,
                doc_size INTEGER,
                created_at TEXT
            )
            """
        )

        # Chunks content
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                header_context TEXT,
                chunk_number INTEGER,
                pattern TEXT,
                has_image INTEGER,
                created_at TEXT
            )
            """
        )

        # FTS5 external-content table pointing at chunks
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                header_context,
                content='chunks',
                content_rowid='id'
            )
            """
        )

        # Migration: add pattern / has_image to existing DBs
        self._add_column_if_missing("chunks", "pattern", "TEXT")
        self._add_column_if_missing("chunks", "has_image", "INTEGER")

        # Triggers to keep FTS5 in sync
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text, header_context)
                VALUES (new.id, new.text, new.header_context);
            END
            """
        )
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, header_context)
                VALUES ('delete', old.id, old.text, old.header_context);
            END
            """
        )
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, header_context)
                VALUES ('delete', old.id, old.text, old.header_context);
                INSERT INTO chunks_fts(rowid, text, header_context)
                VALUES (new.id, new.text, new.header_context);
            END
            """
        )

        # sqlite-vec virtual table for embeddings
        dim = EMBED_DIM
        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{dim}]
            )
            """
        )

        self.conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def upsert_document(
        self, file_path: str, title: str | None = None, url: str | None = None, size: int = 0
    ) -> int:
        """Insert or replace a document row. Returns the document id."""
        cur = self.conn.cursor()
        now = _now()
        cur.execute(
            """
            INSERT INTO documents(file_path, title, url, doc_size, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                title=excluded.title,
                url=excluded.url,
                doc_size=excluded.doc_size,
                created_at=excluded.created_at
            RETURNING id
            """,
            (file_path, title, url, size, now),
        )
        row = cur.fetchone()
        self.conn.commit()
        return int(row[0])

    def _add_column_if_missing(self, table: str, column: str, dtype: str) -> None:
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cur.fetchall()}
        if column not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")
            self.conn.commit()

    def insert_chunk(
        self,
        document_id: int,
        text: str,
        header_context: str,
        chunk_number: int,
        embedding: list[float],
        pattern: str | None = None,
        has_image: bool = False,
    ) -> int:
        """Insert a chunk and its vector. Returns chunk id."""
        cur = self.conn.cursor()
        now = _now()
        cur.execute(
            """
            INSERT INTO chunks(document_id, text, header_context, chunk_number, pattern, has_image, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, text, header_context, chunk_number, pattern, int(has_image), now),
        )
        chunk_id = cur.lastrowid

        # Pack the vector for sqlite-vec
        vec_blob = sqlite_vec.serialize_float32(embedding)
        cur.execute(
            "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, vec_blob),
        )
        self.conn.commit()
        return int(chunk_id)

    def clear_for_path(self, file_path: str) -> None:
        """Delete all chunks, vectors, and the document for a given file path."""
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM documents WHERE file_path = ?", (file_path,))
        row = cur.fetchone()
        if row:
            doc_id = row[0]
            # sqlite-vec virtual tables do not support ON DELETE CASCADE,
            # so we must delete orphan vectors explicitly.
            cur.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
            chunk_ids = [r[0] for r in cur.fetchall()]
            if chunk_ids:
                placeholders = ",".join("?" * len(chunk_ids))
                cur.execute(f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})", chunk_ids)
            cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self.conn.commit()

    # ------------------------------------------------------------------
    # Read (hybrid search)
    # ------------------------------------------------------------------
    def vector_search(
        self, query_embedding: list[float], top_k: int = 20
    ) -> list[dict[str, Any]]:
        """KNN search via sqlite-vec. Returns rows with chunk metadata."""
        vec_blob = sqlite_vec.serialize_float32(query_embedding)
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
                c.id,
                c.document_id,
                c.text,
                c.header_context,
                c.chunk_number,
                c.pattern,
                c.has_image,
                d.file_path,
                d.url,
                distance
            FROM chunks_vec
            JOIN chunks c ON c.id = chunks_vec.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (vec_blob, top_k),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def fts_search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Full-text search via FTS5, ranked by bm25.

        Implicit AND: each term is searched independently and results
        must contain all terms.  Phrase matching requires the user to
        wrap terms in double quotes.
        """
        cur = self.conn.cursor()
        # Tokenise and escape each term individually → implicit AND
        terms = query.strip().split()
        safe_terms = [t.replace('"', '""') for t in terms if t]
        # If the user already quoted a phrase, keep it quoted; otherwise
        # each term is a separate token.
        match_expr = " ".join(
            f'"{t}"' if t.startswith('"') and t.endswith('"') else t
            for t in safe_terms
        )
        cur.execute(
            """
            SELECT
                c.id,
                c.document_id,
                c.text,
                c.header_context,
                c.chunk_number,
                c.pattern,
                c.has_image,
                d.file_path,
                d.url,
                rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match_expr, top_k),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def get_stats(self) -> dict[str, Any]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        doc_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks_vec")
        vec_count = cur.fetchone()[0]
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "vectors": vec_count,
        }

    def close(self) -> None:
        self.conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
