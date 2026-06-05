"""Indexing pipeline: chunk → embed (contextual) → store."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from rag_poc.chunker import chunk_markdown, get_dof_url
from rag_poc.database import RAGDatabase
from rag_poc.embedder import embed_documents

logger = logging.getLogger("rag_poc.index")

# Conservative ceiling for the contextual model (32K token limit).
# We leave headroom for SEP tokens and doc_context prepended to each chunk.
_MAX_DOC_CHARS = 85_000  # ≈ 28K tokens @ 3 chars/token


def index_files(
    file_paths: Sequence[Path],
    db: RAGDatabase | None = None,
    embed_batch_size: int = 8,
) -> dict:
    """
    Index a list of markdown files into the RAG database.

    Each document is embedded contextually: all its chunks (plus a
    document-level front matter) are concatenated with SEP tokens so the
    model sees them in shared context.  If a document is too large to fit
    in the model's 32K token window, it is split into sub-groups that
    each still carry the front matter.

    Returns stats dict with counts.
    """
    if db is None:
        db = RAGDatabase()

    total_docs = 0
    # pending items: (doc_id, chunks, doc_context)
    pending: list[tuple[int, list[dict], str]] = []

    for md_path in tqdm(file_paths, desc="Indexing documents"):
        try:
            doc_id, chunks, doc_context = _prepare_document(md_path, db)
            if doc_id is None:
                continue
            pending.append((doc_id, chunks, doc_context))
            total_docs += 1

            if len(pending) >= embed_batch_size:
                _flush_batch(pending, db)
                pending = []
        except Exception as exc:
            logger.error("Failed to index %s: %s", md_path, exc)
            continue

    if pending:
        _flush_batch(pending, db)

    stats = db.get_stats()
    logger.info("Index complete: %s docs, %s chunks", stats["documents"], stats["chunks"])
    return stats


def _prepare_document(
    md_path: Path, db: RAGDatabase
) -> tuple[int | None, list[dict], str]:
    """Read, chunk, upsert metadata, and extract doc-level front matter.

    Returns (doc_id, chunks, doc_context).  doc_id is None when the
    document has no chunks.
    """
    file_path_str = str(md_path)
    title = md_path.stem
    url = get_dof_url(md_path)
    size = md_path.stat().st_size

    db.clear_for_path(file_path_str)
    doc_id = db.upsert_document(file_path_str, title=title, url=url, size=size)

    chunks = list(chunk_markdown(md_path))
    if not chunks:
        logger.warning("No chunks for %s", md_path)
        return None, [], ""

    # Extract document-level context (front matter) from raw text
    raw = md_path.read_text(encoding="utf-8", errors="replace")
    doc_context = _extract_doc_context(raw, title)

    return doc_id, chunks, doc_context


def _extract_doc_context(raw_text: str, fallback_title: str) -> str:
    """Build a front-matter string with document title and top headings."""
    parts: list[str] = []

    # First H1 heading → document title
    h1_match = re.search(r"^#\s+(.+)$", raw_text, re.MULTILINE)
    if h1_match:
        parts.append(f"# {h1_match.group(1).strip()}")

    # First meaningful H2 (skip boilerplate)
    for line in raw_text.splitlines():
        m = re.match(r"^##\s+(.+)$", line)
        if m:
            h2 = m.group(1).strip()
            if not re.match(
                r"Al margen|Escudo Nacional|Sufragio|Lo que comunico|"
                r"Dado en la Ciudad|En fe de lo cual|Atentamente|Rúbrica",
                h2,
                re.IGNORECASE,
            ):
                parts.append(f"## {h2}")
                break

    if not parts:
        parts.append(f"# {fallback_title}")

    return "\n".join(parts)


def _flush_batch(
    pending: list[tuple[int, list[dict], str]], db: RAGDatabase
) -> None:
    """Embed a batch of documents and store their chunks.

    Each chunk is prepended with doc_context before concatenation so the
    contextual model sees document-level metadata for every chunk.  Large
    documents that would exceed the model's token limit are split into
    sub-groups; each sub-group still carries doc_context.
    """
    # Build embedding inputs and track metadata for storage
    docs_input: list[list[str]] = []          # groups sent to the model
    meta: list[tuple[int, list[dict]]] = []   # (doc_id, original chunks)

    for doc_id, chunks, doc_context in pending:
        # 1. Build full chunk texts with doc_context prepended
        chunk_texts: list[str] = []
        for ch in chunks:
            header = ch["header_context"].strip()
            body = ch["text"].strip()
            text = f"{header}\n\n{body}" if header else body
            if doc_context:
                text = f"{doc_context}\n\n{text}"
            chunk_texts.append(text)

        # 2. Check total size and split into groups if needed
        total_chars = sum(len(t) for t in chunk_texts)
        if total_chars > _MAX_DOC_CHARS and len(chunk_texts) > 1:
            # Determine number of groups; each must fit under the limit.
            # Add a small penalty for doc_context duplication.
            overhead = len(doc_context) if doc_context else 0
            n_groups = max(1, (total_chars + overhead * len(chunk_texts)) // _MAX_DOC_CHARS)
            group_size = max(1, len(chunk_texts) // n_groups)
            for i in range(0, len(chunk_texts), group_size):
                group = chunk_texts[i : i + group_size]
                docs_input.append(group)
                meta.append((doc_id, chunks[i : i + group_size]))
                logger.debug(
                    "Doc %s: split group %s-%s (chars=%s)",
                    doc_id, i, i + len(group) - 1, sum(len(t) for t in group),
                )
        else:
            docs_input.append(chunk_texts)
            meta.append((doc_id, chunks))

    # 3. Embed all groups
    all_embeddings = embed_documents(docs_input, batch_size=len(docs_input))

    # 4. Store
    for (doc_id, chunks), doc_embeddings in zip(meta, all_embeddings):
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
