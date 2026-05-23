"""Indexing pipeline: chunk → embed (contextual) → store."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from rag_poc.chunker import chunk_markdown, get_dof_url
from rag_poc.database import RAGDatabase
from rag_poc.embedder import embed_documents

logger = logging.getLogger("rag_poc.index")


def index_files(
    file_paths: Sequence[Path],
    db: RAGDatabase | None = None,
    embed_batch_size: int = 8,
) -> dict:
    """
    Index a list of markdown files into the RAG database.

    Chunks are embedded CONTEXTUALLY per document (all chunks of a doc
    sent together), then L2-normalised so Euclidean distance in sqlite-vec
    is equivalent to cosine distance.

    Returns stats dict with counts.
    """
    if db is None:
        db = RAGDatabase()

    total_docs = 0
    pending: list[tuple[int, list[dict]]] = []  # (doc_id, chunks)

    for md_path in tqdm(file_paths, desc="Indexing documents"):
        try:
            doc_id, chunks = _prepare_document(md_path, db)
            if doc_id is None:
                continue
            pending.append((doc_id, chunks))
            total_docs += 1

            # Flush when batch is full
            if len(pending) >= embed_batch_size:
                _flush_batch(pending, db)
                pending = []
        except Exception as exc:
            logger.error("Failed to index %s: %s", md_path, exc)
            continue

    # Flush remaining
    if pending:
        _flush_batch(pending, db)

    stats = db.get_stats()
    logger.info("Index complete: %s docs, %s chunks", stats["documents"], stats["chunks"])
    return stats


def _prepare_document(
    md_path: Path, db: RAGDatabase
) -> tuple[int | None, list[dict]]:
    """Read, chunk, and upsert metadata for one file. Returns (doc_id, chunks)."""
    file_path_str = str(md_path)
    title = md_path.stem
    url = get_dof_url(md_path)
    size = md_path.stat().st_size

    db.clear_for_path(file_path_str)
    doc_id = db.upsert_document(file_path_str, title=title, url=url, size=size)

    chunks = list(chunk_markdown(md_path))
    if not chunks:
        logger.warning("No chunks for %s", md_path)
        return None, []

    return doc_id, chunks


def _flush_batch(
    pending: list[tuple[int, list[dict]]], db: RAGDatabase
) -> None:
    """Embed a batch of documents contextually and store their chunks."""
    # Build the contextual input: list[list[str]]
    docs_input: list[list[str]] = []
    for _doc_id, chunks in pending:
        doc_chunks: list[str] = []
        for ch in chunks:
            header = ch["header_context"].strip()
            body = ch["text"].strip()
            doc_chunks.append(f"{header}\n\n{body}" if header else body)
        docs_input.append(doc_chunks)

    # Embed — each inner list is one document's chunks
    all_embeddings = embed_documents(docs_input, batch_size=len(pending))

    # Store
    for (doc_id, chunks), doc_embeddings in zip(pending, all_embeddings):
        if len(chunks) != len(doc_embeddings):
            raise RuntimeError(
                f"Chunk/embedding count mismatch for doc {doc_id}: "
                f"{len(chunks)} chunks vs {len(doc_embeddings)} embeddings"
            )
        for ch, emb in zip(chunks, doc_embeddings):
            db.insert_chunk(
                document_id=doc_id,
                text=ch["text"],
                header_context=ch["header_context"],
                chunk_number=ch["chunk_number"],
                embedding=emb,
                pattern=ch.get("pattern"),
                has_image=ch.get("has_image", False),
            )
