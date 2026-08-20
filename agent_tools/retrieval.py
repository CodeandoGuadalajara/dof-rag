"""Evidence retrieval over the existing DOF-RAG SQLite stores.

This module deliberately keeps retrieval deterministic and independent from an
LLM. Public methods map directly to the small tools exposed to an agent, while
``search`` remains as the deterministic retrieve-then-answer baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import sqlite_vec

from corpus_store.chunk_index import normalized_text, reconstruct
from corpus_store.db import connect, fetch_document_text
from rag_poc.chunker import DocPattern

from .headers import DocumentHeader, extract_document_header
from .models import (
    DocumentHit,
    DocumentOutline,
    DocumentSearchResult,
    EvidenceHit,
    EvidenceSearchResult,
    IndexVersions,
    OutlineChunk,
    PublicationHit,
    RetrievalStrategy,
    SearchFilters,
    SearchResult,
)

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
NORM_IDENTIFIER_RE = re.compile(r"\b(?:NOM|NMX)-\d{3}(?:-[A-Z]+)*(?:-\d{4})?\b", re.I)
DATED_DOCUMENT_RE = re.compile(
    r"\b(Plan Nacional de Desarrollo\s+(?:19|20)\d{2}-(?:19|20)\d{2})\b",
    re.I,
)
DOCUMENT_NAME_START_RE = re.compile(r"\b(?:ley|c[oó]digo|reglamento)\b", re.I)
DOCUMENT_NAME_STOP_WORDS = {
    "al",
    "articulo",
    "conforme",
    "declararse",
    "es",
    "fue",
    "publicada",
    "publicado",
    "que",
    "reglamentaria",
    "reglamentario",
    "segun",
    "se",
    "son",
}
QUERY_ALIASES = {
    "inegi": ["instituto", "nacional", "estadistica", "geografia"],
    "inpc": ["indice", "nacional", "precios", "consumidor"],
    "uma": ["unidad", "medida", "actualizacion"],
}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(_fold(text))


def _match_query(query: str) -> list[str]:
    """Return FTS terms, dropping high-document-frequency terms."""
    terms = _tokens(query)
    expanded = list(terms)
    for term in terms:
        expanded.extend(QUERY_ALIASES.get(term, []))
    return list(dict.fromkeys(expanded))


def _normative_title_boost(query: str, title: str | None) -> float:
    """Prefer the issuing norm over documents that merely cite its identifier."""
    if not title:
        return 0.0
    query_ids = {match.group(0).upper() for match in NORM_IDENTIFIER_RE.finditer(query)}
    if not query_ids:
        return 0.0
    folded_title = _fold(title)
    boost = 0.0
    for identifier in query_ids:
        if _fold(identifier) in folded_title:
            boost = max(boost, 100.0 if identifier.count("-") >= 3 else 40.0)
    if boost and folded_title.startswith("norma oficial mexicana"):
        boost += 40.0
    if boost:
        overlap = set(_tokens(query)) & set(_tokens(title))
        boost += min(len(overlap), 4) * 12.0
    return boost


def _recency_boosts(
    candidates: list[tuple[int, str | None]], weight: float
) -> dict[int, float]:
    """Rank-based recency bonus: the newest dated candidate gets ``weight``.

    Boosts decay linearly by recency rank so relevance still dominates, and
    undated documents receive no bonus. Ordering by ``(-date, document_id)``
    keeps the result deterministic for a static corpus.
    """
    dated = sorted(
        ((date, document_id) for document_id, date in candidates if date),
        key=lambda item: (item[0], -item[1]),
        reverse=True,
    )
    n = len(dated)
    return {
        document_id: weight * (n - rank) / n
        for rank, (_, document_id) in enumerate(dated)
    }


def _apply_recency_to_ranked(
    ranked_ids: list[int],
    dates: dict[int, str | None],
    weight: float,
) -> list[int]:
    """Blend rank-based relevance with document recency.

    The goal is visibility, not dominance: the best chunk from a recent
    document should reach the top-k so the agent can judge it, without
    pushing irrelevant recent chunks above highly relevant older ones.
    """
    n = len(ranked_ids)
    if not n or weight <= 0.0:
        return ranked_ids
    relevance = {chunk_id: (n - rank) / n for rank, chunk_id in enumerate(ranked_ids)}
    recency = _recency_boosts(
        [(chunk_id, dates.get(chunk_id)) for chunk_id in ranked_ids], 1.0
    )
    return sorted(
        ranked_ids,
        key=lambda chunk_id: (
            -(
                (1.0 - weight) * relevance[chunk_id]
                + weight * recency.get(chunk_id, 0.0)
            ),
            -relevance[chunk_id],
            chunk_id,
        ),
    )


def _document_name_phrases(query: str) -> list[str]:
    """Extract explicit legal-instrument names for an exact-phrase lookup."""
    phrases: list[str] = []
    for match in DOCUMENT_NAME_START_RE.finditer(query):
        tail = query[match.start() :]
        tokens = TOKEN_RE.findall(tail)
        selected: list[str] = []
        for token in tokens[:12]:
            if len(selected) >= 3 and _fold(token) in DOCUMENT_NAME_STOP_WORDS:
                break
            selected.append(token)
        if len(selected) >= 3:
            phrases.append(" ".join(selected))
    return list(dict.fromkeys(phrases))


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> bytes:
        """Return the packed 1024-bit query embedding."""


class LlamaQueryEmbedder:
    """Keep one local llama-server alive for an interactive session."""

    def __init__(self, gguf: Path, *, port: int = 8086, ctx: int = 8192):
        from corpus_store.embed import start_server

        self.port = port
        self.process = start_server(gguf, ctx=ctx, port=port)

    def embed_query(self, query: str) -> bytes:
        from corpus_store.embed import PREFIX_QUERY, embed_batch, pack_binary

        vector = embed_batch([PREFIX_QUERY + query], self.port)[0]
        return pack_binary(vector)

    def close(self) -> None:
        self.process.terminate()
        self.process.wait()

    def __enter__(self) -> "LlamaQueryEmbedder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class _ChunkRecord:
    chunk_id: int
    document_id: int
    path: str
    chunk_index: int
    pattern: str
    spans_json: str
    chunk_hash: bytes
    heading_path: list[str]
    token_count: int
    publication_date: str | None
    section: str | None


@dataclass(frozen=True)
class _ScoredChunk:
    record: _ChunkRecord
    text: str
    score: float


def _normalize_scores(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    span = high - low if high > low else 1.0
    return {key: (value - low) / span for key, value in values.items()}


def _fuse_documents(
    bm25: list[tuple[int, float]],
    vector: list[tuple[int, float]],
    weight: float,
) -> list[tuple[int, float, float | None, float | None]]:
    """Fuse document scores using the same weighted formula as eval v4."""
    lexical = dict(bm25)
    semantic = dict(vector)
    nl = _normalize_scores(lexical)
    nv = _normalize_scores(semantic)
    left_rank = {item: rank for rank, (item, _) in enumerate(bm25)}
    right_rank = {item: rank for rank, (item, _) in enumerate(vector)}
    scores = {
        item: weight * nl.get(item, 0.0) + (1.0 - weight) * nv.get(item, 0.0)
        for item in set(nl) | set(nv)
    }
    return sorted(
        (
            (item, scores[item], lexical.get(item), semantic.get(item))
            for item in scores
        ),
        key=lambda row: (
            -row[1],
            min(left_rank.get(row[0], 10**9), right_rank.get(row[0], 10**9)),
            row[0],
        ),
    )


def _rrf(ids: list[list[int]], constant: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranked in ids:
        for rank, item in enumerate(ranked, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (constant + rank)
            best_rank[item] = min(best_rank.get(item, 10**9), rank)
    return sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))


def _filter_clauses(
    filters: SearchFilters, *, alias: str = "d"
) -> tuple[list[str], list[Any]]:
    """Translate supported metadata filters to parameterized SQL clauses."""
    clauses: list[str] = []
    params: list[Any] = []
    if filters.as_of:
        clauses.append(f"{alias}.publication_date <= ?")
        params.append(filters.as_of)
    if filters.date_from:
        clauses.append(f"{alias}.publication_date >= ?")
        params.append(filters.date_from)
    if filters.date_to:
        clauses.append(f"{alias}.publication_date <= ?")
        params.append(filters.date_to)
    if filters.section:
        clauses.append(f"{alias}.section = ?")
        params.append(filters.section)
    return clauses, params


def _parse_heading_path(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _bm25_chunk_scores(query: str, texts: list[str]) -> list[float]:
    """Rank a bounded set of candidate chunks with standard BM25.

    This is the deterministic fallback until a complete chunk-level FTS5
    index exists. IDF is computed over the candidate chunks, and document
    length normalization prevents the previous bias toward longer chunks.
    """
    terms = list(dict.fromkeys(_tokens(query)))
    if not terms or not texts:
        return [0.0] * len(texts)
    tokenized = [_tokens(text) for text in texts]
    lengths = [max(len(words), 1) for words in tokenized]
    average_length = sum(lengths) / len(lengths)
    frequencies = [Counter(words) for words in tokenized]
    document_frequency = {
        term: sum(1 for counts in frequencies if counts.get(term, 0) > 0)
        for term in terms
    }
    n = len(texts)
    k1 = 1.2
    b = 0.75
    phrase = " ".join(terms)
    provisions = re.findall(r"\b\d+\.\d+\b", query)
    definition_query = "defin" in _fold(query)
    scores: list[float] = []
    for text, counts, length in zip(texts, frequencies, lengths, strict=True):
        score = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1.0 - b + b * length / average_length)
            score += inverse_document_frequency * tf * (k1 + 1.0) / denominator
        if phrase and phrase in _fold(text):
            score += 0.25
        for provision in provisions:
            if re.search(
                rf"(?im)^\s*(?:>\s*)?(?:\*\*)?{re.escape(provision)}"
                r"(?:\*\*)?(?:\s|$)",
                text,
            ):
                score += 20.0
        if definition_query:
            for heading in re.findall(
                r"(?im)^\s*\*\*\d+(?:\.\d+)+\s+([^*:\n]+):?\*\*", text
            ):
                overlap = set(terms) & set(_tokens(heading))
                if len(overlap) >= 2:
                    score += 20.0
                    break
        scores.append(score)
    return scores


class DfPruner:
    def __init__(self, corpus: sqlite3.Connection, n_docs: int):
        self.corpus = corpus
        self.max_df = int(n_docs * 0.5)
        self.cache: dict[str, int] = {}

    def prune(self, terms: list[str]) -> list[str]:
        kept: list[str] = []
        for term in terms:
            if term not in self.cache:
                row = self.corpus.execute(
                    "SELECT doc FROM documents_fts_vocab WHERE term = ?", (term,)
                ).fetchone()
                self.cache[term] = row[0] if row else 0
            if self.cache[term] <= self.max_df:
                kept.append(term)
        return kept or terms


class DofRetriever:
    """Provider-neutral tools over the BM25, chunk, and vector stores."""

    def __init__(
        self,
        *,
        corpus_db: str | Path = "dof_db/dof_corpus_l3.sqlite",
        chunks_db: str | Path = "dof_db/dof_chunks.sqlite",
        vec0_db: str | Path | None = "dof_db/dof_vec0_jina_binary.sqlite",
    ):
        self.corpus = connect(corpus_db)
        self.chunks = sqlite3.connect(str(chunks_db))
        self.corpus.execute("PRAGMA query_only = ON")
        self.chunks.execute("PRAGMA query_only = ON")
        self.vec0: sqlite3.Connection | None = None
        if vec0_db and Path(vec0_db).exists():
            self.vec0 = sqlite3.connect(str(vec0_db))
            self.vec0.enable_load_extension(True)
            sqlite_vec.load(self.vec0)
            self.vec0.execute("PRAGMA query_only = ON")
        n_docs = self.corpus.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        self.pruner = DfPruner(self.corpus, n_docs)
        self._header_cache: dict[int, DocumentHeader] = {}
        self._fast_header_cache: dict[int, DocumentHeader] = {}
        corpus_meta = dict(self.corpus.execute("SELECT key, value FROM corpus_meta"))
        chunk_meta = self.chunks.execute(
            "SELECT chunker_version, corpus_version FROM chunks LIMIT 1"
        ).fetchone()
        self.versions = IndexVersions(
            corpus_version=(
                chunk_meta[1] if chunk_meta else corpus_meta.get("corpus_version")
            ),
            chunker_version=(chunk_meta[0] if chunk_meta else None),
            vector_available=self.vec0 is not None,
        )

    def close(self) -> None:
        self.corpus.close()
        self.chunks.close()
        if self.vec0 is not None:
            self.vec0.close()

    def __enter__(self) -> "DofRetriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def list_publications(
        self, filters: SearchFilters, *, limit: int = 50
    ) -> list[PublicationHit]:
        """List publications by enforceable metadata, newest first."""
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        clauses, params = _filter_clauses(filters)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.corpus.execute(
            "SELECT d.document_id, d.path, d.publication_date, d.section "
            "FROM _documents_zstd d "
            f"{where} ORDER BY d.publication_date DESC, d.document_id DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
        hits = []
        for row in rows:
            document_id = int(row[0])
            header = self._document_header(document_id, full=True)
            hits.append(
                PublicationHit(
                    document_id=document_id,
                    path=row[1],
                    publication_date=row[2],
                    section=row[3],
                    title=header.title,
                    institution=header.institution,
                )
            )
        return hits

    def _document_header(
        self, document_id: int, *, full: bool = False
    ) -> DocumentHeader:
        if full:
            if document_id not in self._header_cache:
                text = fetch_document_text(self.corpus, document_id)
                self._header_cache[document_id] = extract_document_header(text)
            return self._header_cache[document_id]
        if document_id not in self._fast_header_cache:
            self._document_headers([document_id])
        return self._fast_header_cache[document_id]

    def _document_headers(self, document_ids: list[int]) -> dict[int, DocumentHeader]:
        missing = [
            document_id
            for document_id in document_ids
            if document_id not in self._fast_header_cache
        ]
        if missing:
            placeholders = ",".join("?" for _ in missing)
            rows = self.chunks.execute(
                "SELECT document_id, heading_path FROM chunks "
                f"WHERE document_id IN ({placeholders}) "
                "AND heading_path NOT IN ('', '[]') ORDER BY document_id, chunk_index",
                missing,
            ).fetchall()
            for document_id, raw_headings in rows:
                document_id = int(document_id)
                if document_id in self._fast_header_cache:
                    continue
                headings = _parse_heading_path(raw_headings)
                if headings:
                    self._fast_header_cache[document_id] = DocumentHeader(
                        institution=None, title=headings[0]
                    )
            for document_id in missing:
                self._fast_header_cache.setdefault(
                    document_id, DocumentHeader(institution=None, title=None)
                )
        return {
            document_id: self._fast_header_cache[document_id]
            for document_id in document_ids
        }

    def _identifier_documents(
        self, query: str, depth: int, filters: SearchFilters
    ) -> list[tuple[int, float, str, str | None, str | None]]:
        identifiers = list(dict.fromkeys(NORM_IDENTIFIER_RE.findall(query)))
        if not identifiers:
            return []
        filter_clauses, filter_params = _filter_clauses(filters)
        results: dict[int, tuple[int, float, str, str | None, str | None]] = {}
        for identifier in identifiers:
            phrase = " ".join(_tokens(identifier))
            clauses = ["documents_fts MATCH ?", *filter_clauses]
            rows = self.corpus.execute(
                "SELECT d.document_id, bm25(documents_fts), d.path, "
                "d.publication_date, d.section "
                "FROM documents_fts JOIN _documents_zstd d "
                "ON d.document_id = documents_fts.rowid "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY bm25(documents_fts) LIMIT ?",
                [f'"{phrase}"', *filter_params, depth],
            ).fetchall()
            for row in rows:
                result = (int(row[0]), -float(row[1]), row[2], row[3], row[4])
                results.setdefault(result[0], result)
        return list(results.values())

    def _heading_phrase_documents(
        self, query: str, filters: SearchFilters
    ) -> tuple[list[tuple[int, float, str, str | None, str | None]], dict[int, str]]:
        phrases = list(dict.fromkeys(DATED_DOCUMENT_RE.findall(query)))
        if not phrases:
            return [], {}
        matched_headings: dict[int, str] = {}
        for phrase in phrases:
            rows = self.chunks.execute(
                "SELECT document_id, heading_path FROM chunks "
                "WHERE heading_path LIKE ? LIMIT 200",
                (f"%{phrase}%",),
            ).fetchall()
            for document_id, raw_headings in rows:
                headings = _parse_heading_path(raw_headings)
                exact = next(
                    (
                        heading
                        for heading in headings
                        if _fold(phrase) in _fold(heading)
                    ),
                    None,
                )
                if exact:
                    matched_headings.setdefault(int(document_id), exact)
        if not matched_headings:
            return [], {}
        ids = list(matched_headings)
        placeholders = ",".join("?" for _ in ids)
        clauses, params = _filter_clauses(filters)
        where = [f"d.document_id IN ({placeholders})", *clauses]
        rows = self.corpus.execute(
            "SELECT d.document_id, d.path, d.publication_date, d.section "
            "FROM _documents_zstd d "
            f"WHERE {' AND '.join(where)}",
            [*ids, *params],
        ).fetchall()
        hits = [(int(row[0]), 0.0, row[1], row[2], row[3]) for row in rows]
        kept = {row[0] for row in hits}
        return hits, {
            document_id: heading
            for document_id, heading in matched_headings.items()
            if document_id in kept
        }

    def _exact_title_documents(
        self, query: str, filters: SearchFilters
    ) -> tuple[list[tuple[int, float, str, str | None, str | None]], dict[int, str]]:
        """Find issuing documents whose extracted title contains a named instrument.

        Ordinary BM25 often ranks documents that repeatedly cite a law above the
        decree that issued it. Exact-phrase candidates let the title reranker see
        the source document without assuming a publication date.
        """
        phrases = _document_name_phrases(query)
        if not phrases:
            return [], {}
        filter_clauses, filter_params = _filter_clauses(filters)
        candidates: dict[int, tuple[int, float, str, str | None, str | None]] = {}
        for phrase in phrases:
            match = '"' + " ".join(_tokens(phrase)).replace('"', '""') + '"'
            clauses = ["documents_fts MATCH ?", *filter_clauses]
            rows = self.corpus.execute(
                "SELECT d.document_id, bm25(documents_fts), d.path, "
                "d.publication_date, d.section "
                "FROM documents_fts JOIN _documents_zstd d "
                "ON d.document_id = documents_fts.rowid "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY bm25(documents_fts) LIMIT 100",
                [match, *filter_params],
            ).fetchall()
            folded_phrase = _fold(phrase)
            for row in rows:
                document_id = int(row[0])
                header = self._document_header(document_id, full=True)
                if header.title and folded_phrase in _fold(header.title):
                    candidates.setdefault(
                        document_id,
                        (document_id, -float(row[1]), row[2], row[3], row[4]),
                    )
        titles = {
            document_id: self._document_header(document_id, full=True).title or ""
            for document_id in candidates
        }
        return list(candidates.values()), titles

    def _bm25_documents(
        self, query: str, depth: int, filters: SearchFilters
    ) -> list[tuple[int, float, str, str | None, str | None]]:
        terms = self.pruner.prune(_match_query(query))
        if not terms:
            return []
        match = " OR ".join(
            f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms
        )
        filter_clauses, filter_params = _filter_clauses(filters)
        clauses = ["documents_fts MATCH ?", *filter_clauses]
        params: list[Any] = [match, *filter_params]
        params.append(depth)
        rows = self.corpus.execute(
            "SELECT d.document_id, bm25(documents_fts), d.path, "
            "d.publication_date, d.section "
            "FROM documents_fts JOIN _documents_zstd d "
            "ON d.document_id = documents_fts.rowid "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY bm25(documents_fts) LIMIT ?",
            params,
        ).fetchall()
        return [(int(row[0]), -float(row[1]), row[2], row[3], row[4]) for row in rows]

    def _vector_chunks(
        self,
        query_vector: bytes | None,
        scan_k: int,
        filters: SearchFilters,
    ) -> tuple[list[tuple[int, float]], int]:
        if self.vec0 is None or query_vector is None:
            return [], 0
        rows = self.vec0.execute(
            "SELECT rowid, distance FROM chunk_vec "
            "WHERE embedding MATCH vec_bit(?) AND k = ?",
            (query_vector, scan_k),
        ).fetchall()
        chunk_ids = [int(row[0]) for row in rows]
        metadata = self._chunk_metadata(chunk_ids)
        kept: list[tuple[int, float]] = []
        for rowid, distance in rows:
            row = metadata.get(int(rowid))
            if row is None:
                continue
            if filters.as_of and (
                row.publication_date is None or row.publication_date > filters.as_of
            ):
                continue
            if filters.date_from and (
                row.publication_date is None or row.publication_date < filters.date_from
            ):
                continue
            if filters.date_to and (
                row.publication_date is None or row.publication_date > filters.date_to
            ):
                continue
            if filters.section and row.section != filters.section:
                continue
            kept.append((int(rowid), -float(distance)))
        return kept, len(rows)

    def _chunk_metadata(self, chunk_ids: list[int]) -> dict[int, _ChunkRecord]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.chunks.execute(
            "SELECT c.chunk_id, c.document_id, c.path, c.chunk_index, c.pattern, "
            "c.spans_json, c.chunk_hash, c.heading_path, c.token_count FROM chunks c "
            f"WHERE c.chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        if not rows:
            return {}
        document_ids = sorted({int(row[1]) for row in rows})
        doc_placeholders = ",".join("?" for _ in document_ids)
        doc_meta = {
            int(row[0]): (row[1], row[2])
            for row in self.corpus.execute(
                "SELECT document_id, publication_date, section "
                f"FROM _documents_zstd WHERE document_id IN ({doc_placeholders})",
                document_ids,
            )
        }
        result: dict[int, _ChunkRecord] = {}
        for row in rows:
            publication_date, section = doc_meta.get(int(row[1]), (None, None))
            result[int(row[0])] = _ChunkRecord(
                chunk_id=int(row[0]),
                document_id=int(row[1]),
                path=row[2],
                chunk_index=int(row[3]),
                pattern=row[4],
                spans_json=row[5],
                chunk_hash=bytes(row[6]),
                heading_path=_parse_heading_path(row[7]),
                token_count=int(row[8]),
                publication_date=publication_date,
                section=section,
            )
        return result

    def _records_for_documents(self, document_ids: list[int]) -> list[_ChunkRecord]:
        if not document_ids:
            return []
        placeholders = ",".join("?" for _ in document_ids)
        chunk_ids = [
            int(row[0])
            for row in self.chunks.execute(
                "SELECT chunk_id FROM chunks "
                f"WHERE document_id IN ({placeholders}) "
                "ORDER BY document_id, chunk_index",
                document_ids,
            )
        ]
        metadata = self._chunk_metadata(chunk_ids)
        return [metadata[chunk_id] for chunk_id in chunk_ids if chunk_id in metadata]

    def _reconstruct_records(
        self, records: list[_ChunkRecord]
    ) -> list[tuple[_ChunkRecord, str]]:
        normalized_cache: dict[tuple[int, str], str] = {}
        reconstructed: list[tuple[_ChunkRecord, str]] = []
        for record in records:
            cache_key = (record.document_id, record.pattern)
            if cache_key not in normalized_cache:
                normalized_cache[cache_key] = normalized_text(
                    fetch_document_text(self.corpus, record.document_id),
                    DocPattern(record.pattern),
                )
            text = reconstruct(
                json.loads(record.spans_json), normalized_cache[cache_key]
            )
            if hashlib.sha256(text.encode("utf-8")).digest() != record.chunk_hash:
                raise ValueError(f"chunk {record.chunk_id} failed hash verification")
            reconstructed.append((record, text))
        return reconstructed

    def _lexical_chunks(
        self, document_ids: list[int], query: str
    ) -> list[_ScoredChunk]:
        reconstructed = self._reconstruct_records(
            self._records_for_documents(document_ids)
        )
        scores = _bm25_chunk_scores(query, [text for _, text in reconstructed])
        return [
            _ScoredChunk(record=record, text=text, score=score)
            for (record, text), score in zip(reconstructed, scores, strict=True)
            if score > 0.0
        ]

    def get_document_outline(self, document_id: int) -> DocumentOutline:
        """Return a compact heading/chunk map without loading chunk text."""
        row = self.corpus.execute(
            "SELECT document_id, path, publication_date, section "
            "FROM _documents_zstd WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown document_id {document_id}")
        chunks = self.chunks.execute(
            "SELECT chunk_id, chunk_index, heading_path, token_count FROM chunks "
            "WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        header = self._document_header(document_id)
        return DocumentOutline(
            document_id=int(row[0]),
            path=row[1],
            publication_date=row[2],
            section=row[3],
            chunks=[
                OutlineChunk(
                    chunk_id=int(chunk[0]),
                    chunk_index=int(chunk[1]),
                    heading_path=_parse_heading_path(chunk[2]),
                    token_count=int(chunk[3]),
                )
                for chunk in chunks
            ],
            title=header.title,
            institution=header.institution,
        )

    def read_chunks(
        self, chunk_ids: list[int], *, neighbor_window: int = 0
    ) -> list[EvidenceHit]:
        """Read requested chunks and optional neighbors, with hash verification."""
        if not chunk_ids:
            return []
        if neighbor_window < 0 or neighbor_window > 3:
            raise ValueError("neighbor_window must be between 0 and 3")
        base = self._chunk_metadata(list(dict.fromkeys(chunk_ids)))
        missing = set(chunk_ids) - set(base)
        if missing:
            raise KeyError(f"unknown chunk ids: {sorted(missing)}")
        selected_ids = set(base)
        if neighbor_window:
            by_document: dict[int, set[int]] = {}
            for record in base.values():
                indices = by_document.setdefault(record.document_id, set())
                indices.update(
                    range(
                        max(0, record.chunk_index - neighbor_window),
                        record.chunk_index + neighbor_window + 1,
                    )
                )
            for document_id, indices in by_document.items():
                placeholders = ",".join("?" for _ in indices)
                rows = self.chunks.execute(
                    "SELECT chunk_id FROM chunks WHERE document_id = ? "
                    f"AND chunk_index IN ({placeholders})",
                    [document_id, *sorted(indices)],
                ).fetchall()
                selected_ids.update(int(item[0]) for item in rows)
        records = self._chunk_metadata(sorted(selected_ids))
        ordered = sorted(
            records.values(), key=lambda item: (item.document_id, item.chunk_index)
        )
        return [
            EvidenceHit(
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                path=record.path,
                publication_date=record.publication_date,
                section=record.section,
                chunk_index=record.chunk_index,
                heading_path=record.heading_path,
                text=text,
                score=0.0,
                source="read",
                rank=rank,
            )
            for rank, (record, text) in enumerate(self._reconstruct_records(ordered), 1)
        ]

    def search_documents(
        self,
        query: str,
        *,
        strategy: RetrievalStrategy | str = RetrievalStrategy.HYBRID,
        filters: SearchFilters | None = None,
        query_vector: bytes | None = None,
        bm25_depth: int = 50,
        vector_k: int = 200,
        top_k: int = 20,
        bm25_weight: float = 0.75,
        prefer_recent: bool = False,
        recency_weight: float = 0.15,
    ) -> DocumentSearchResult:
        """Find candidate documents using lexical, vector, or hybrid ranking."""
        started = perf_counter()
        strategy = RetrievalStrategy(strategy)
        filters = filters or SearchFilters()
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        if strategy is RetrievalStrategy.VECTOR and (
            self.vec0 is None or query_vector is None
        ):
            raise ValueError("vector strategy requires a vector index and query_vector")

        lexical = []
        heading_titles: dict[int, str] = {}
        exact_titles: dict[int, str] = {}
        if strategy is not RetrievalStrategy.VECTOR:
            lexical = self._bm25_documents(query, bm25_depth, filters)
            known = {row[0] for row in lexical}
            lexical.extend(
                row
                for row in self._identifier_documents(query, bm25_depth, filters)
                if row[0] not in known
            )
            known = {row[0] for row in lexical}
            title_hits, exact_titles = self._exact_title_documents(query, filters)
            lexical.extend(row for row in title_hits if row[0] not in known)
            known = {row[0] for row in lexical}
            heading_hits, heading_titles = self._heading_phrase_documents(
                query, filters
            )
            lexical.extend(row for row in heading_hits if row[0] not in known)
        vector_chunks: list[tuple[int, float]] = []
        vector_count = 0
        if strategy is not RetrievalStrategy.LEXICAL and query_vector is not None:
            vector_chunks, vector_count = self._vector_chunks(
                query_vector, vector_k, filters
            )
        vector_docs: dict[int, tuple[float, int]] = {}
        chunk_meta = self._chunk_metadata([chunk_id for chunk_id, _ in vector_chunks])
        for rank, (chunk_id, score) in enumerate(vector_chunks):
            record = chunk_meta.get(chunk_id)
            if record is None:
                continue
            doc_id = record.document_id
            if doc_id not in vector_docs:
                vector_docs[doc_id] = (score, rank)
        if strategy is RetrievalStrategy.LEXICAL or not vector_docs:
            fused = [(row[0], row[1], row[1], None) for row in lexical]
        elif strategy is RetrievalStrategy.VECTOR:
            fused = [
                (doc_id, score, None, score)
                for doc_id, (score, _) in vector_docs.items()
            ]
        else:
            fused = _fuse_documents(
                [(row[0], row[1]) for row in lexical],
                [(doc_id, score) for doc_id, (score, _) in vector_docs.items()],
                bm25_weight,
            )
        doc_info = {row[0]: (row[2], row[3], row[4]) for row in lexical}
        if vector_docs:
            ids = list(vector_docs)
            placeholders = ",".join("?" for _ in ids)
            for row in self.corpus.execute(
                f"SELECT document_id, path, publication_date, section FROM _documents_zstd "
                f"WHERE document_id IN ({placeholders})",
                ids,
            ):
                doc_info[int(row[0])] = (row[1], row[2], row[3])
        candidate_limit = len(fused)
        reranked = []
        headers = self._document_headers([row[0] for row in fused[:candidate_limit]])
        for document_id, heading in heading_titles.items():
            if document_id in headers:
                headers[document_id] = DocumentHeader(institution=None, title=heading)
        for document_id, title in exact_titles.items():
            if document_id in headers:
                headers[document_id] = DocumentHeader(institution=None, title=title)
        for fused_rank, row in enumerate(fused[:candidate_limit]):
            document_id = row[0]
            header = headers[document_id]
            reranked.append(
                (
                    *row,
                    _normative_title_boost(query, header.title)
                    + (
                        200.0
                        if document_id in heading_titles or document_id in exact_titles
                        else 0.0
                    ),
                    fused_rank,
                )
            )
        recency = (
            _recency_boosts(
                [
                    (row[0], doc_info.get(row[0], (None, None, None))[1])
                    for row in reranked
                ],
                recency_weight * max((row[1] for row in reranked), default=0.0),
            )
            if prefer_recent and reranked
            else {}
        )
        reranked.sort(
            key=lambda row: (-(row[1] + row[4] + recency.get(row[0], 0.0)), row[5], row[0])
        )
        documents = []
        for rank, (
            doc_id,
            score,
            bm25_score,
            vector_score,
            title_boost,
            _,
        ) in enumerate(reranked[:top_k], 1):
            if doc_id not in doc_info:
                continue
            header = headers[doc_id]
            if header.title is None:
                header = self._document_header(doc_id, full=True)
            documents.append(
                DocumentHit(
                    document_id=doc_id,
                    path=doc_info[doc_id][0],
                    publication_date=doc_info[doc_id][1],
                    section=doc_info[doc_id][2],
                    score=score + title_boost + recency.get(doc_id, 0.0),
                    bm25_score=bm25_score,
                    vector_score=vector_score,
                    rank=rank,
                    title=header.title or headers[doc_id].title,
                    institution=header.institution,
                    title_boost=title_boost,
                    recency_boost=recency.get(doc_id, 0.0),
                )
            )
        return DocumentSearchResult(
            query=query,
            strategy=strategy,
            filters=filters,
            documents=documents,
            vector_candidates_scanned=vector_count,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            versions=self.versions,
            settings={
                "bm25_depth": bm25_depth,
                "vector_k": vector_k,
                "top_k": top_k,
                "bm25_weight": bm25_weight,
                "vector_filtering": "post_filter",
                "normative_title_rerank": True,
                "dated_heading_candidates": bool(heading_titles),
                "exact_title_candidates": bool(exact_titles),
                "prefer_recent": prefer_recent,
                "recency_weight": recency_weight if prefer_recent else 0.0,
            },
        )

    def search_evidence(
        self,
        query: str,
        document_ids: list[int],
        *,
        strategy: RetrievalStrategy | str = RetrievalStrategy.HYBRID,
        query_vector: bytes | None = None,
        top_k: int = 20,
        candidate_depth: int = 200,
        vector_k: int = 200,
        prefer_recent: bool = False,
        recency_weight: float = 0.25,
    ) -> EvidenceSearchResult:
        """Search deeply for citable chunks inside candidate documents."""
        started = perf_counter()
        strategy = RetrievalStrategy(strategy)
        document_ids = list(dict.fromkeys(document_ids))
        if not query.strip():
            raise ValueError("query must not be empty")
        if not document_ids:
            return EvidenceSearchResult(
                query=query,
                strategy=strategy,
                document_ids=[],
                elapsed_ms=(perf_counter() - started) * 1000.0,
                versions=self.versions,
            )
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        if strategy is RetrievalStrategy.VECTOR and (
            self.vec0 is None or query_vector is None
        ):
            raise ValueError("vector strategy requires a vector index and query_vector")

        reconstructed = self._reconstruct_records(
            self._records_for_documents(document_ids)
        )
        by_id = {record.chunk_id: (record, text) for record, text in reconstructed}
        lexical_scores = _bm25_chunk_scores(query, [text for _, text in reconstructed])
        score_by_id = {
            record.chunk_id: score
            for (record, _), score in zip(reconstructed, lexical_scores, strict=True)
        }
        lexical_ids = [
            record.chunk_id
            for (record, _), score in sorted(
                zip(reconstructed, lexical_scores, strict=True),
                key=lambda item: (-item[1], item[0][0].chunk_id),
            )
            if score > 0.0
        ][:candidate_depth]

        vector_chunks: list[tuple[int, float]] = []
        vector_count = 0
        if strategy is not RetrievalStrategy.LEXICAL and query_vector is not None:
            vector_chunks, vector_count = self._vector_chunks(
                query_vector, vector_k, SearchFilters()
            )
        vector_score = dict(vector_chunks)
        vector_ids = [chunk_id for chunk_id, _ in vector_chunks if chunk_id in by_id]
        if strategy is RetrievalStrategy.LEXICAL or not vector_ids:
            ranked_ids = lexical_ids
        elif strategy is RetrievalStrategy.VECTOR:
            ranked_ids = vector_ids
        else:
            ranked_ids = _rrf([lexical_ids, vector_ids])
        if prefer_recent:
            ranked_ids = _apply_recency_to_ranked(
                ranked_ids,
                {
                    chunk_id: by_id[chunk_id][0].publication_date
                    for chunk_id in ranked_ids
                    if chunk_id in by_id
                },
                recency_weight,
            )
        ranked_ids = ranked_ids[:top_k]

        evidence: list[EvidenceHit] = []
        for rank, chunk_id in enumerate(ranked_ids, 1):
            item = by_id.get(chunk_id)
            if item is None:
                continue
            record, text = item
            in_lexical = chunk_id in lexical_ids
            in_vector = chunk_id in vector_score
            source = "hybrid" if in_lexical and in_vector else "bm25_chunk"
            if in_vector and not in_lexical:
                source = "vector"
            evidence.append(
                EvidenceHit(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    path=record.path,
                    publication_date=record.publication_date,
                    section=record.section,
                    chunk_index=record.chunk_index,
                    heading_path=record.heading_path,
                    text=text,
                    score=score_by_id.get(chunk_id, vector_score.get(chunk_id, 0.0)),
                    source=source,
                    rank=rank,
                )
            )
        return EvidenceSearchResult(
            query=query,
            strategy=strategy,
            document_ids=document_ids,
            evidence=evidence,
            vector_candidates_scanned=vector_count,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            versions=self.versions,
            settings={
                "top_k": top_k,
                "candidate_depth": candidate_depth,
                "vector_k": vector_k,
                "vector_filtering": "post_filter",
                "lexical_ranker": "bounded_bm25",
                "prefer_recent": prefer_recent,
                "recency_weight": recency_weight if prefer_recent else 0.0,
            },
        )

    def search(
        self,
        query: str,
        *,
        query_vector: bytes | None = None,
        as_of: str | None = None,
        section: str | None = None,
        bm25_depth: int = 50,
        vector_k: int = 200,
        document_depth: int = 20,
        evidence_depth: int = 20,
        evidence_candidate_depth: int = 200,
        bm25_weight: float = 0.75,
    ) -> SearchResult:
        """Compatibility baseline: discover documents, then retrieve evidence."""
        filters = SearchFilters(as_of=as_of, section=section)
        strategy = (
            RetrievalStrategy.HYBRID
            if self.vec0 is not None and query_vector is not None
            else RetrievalStrategy.LEXICAL
        )
        documents = self.search_documents(
            query,
            strategy=strategy,
            filters=filters,
            query_vector=query_vector,
            bm25_depth=bm25_depth,
            vector_k=vector_k,
            top_k=document_depth,
            bm25_weight=bm25_weight,
        )
        evidence = self.search_evidence(
            query,
            documents.document_ids,
            strategy=strategy,
            query_vector=query_vector,
            top_k=evidence_depth,
            candidate_depth=evidence_candidate_depth,
            vector_k=vector_k,
        )
        return SearchResult(
            query=query,
            as_of=as_of,
            documents=documents.documents,
            evidence=evidence.evidence,
            vector_available=self.vec0 is not None and query_vector is not None,
            vector_count=max(
                documents.vector_candidates_scanned,
                evidence.vector_candidates_scanned,
            ),
            settings={
                "bm25_depth": bm25_depth,
                "vector_k": vector_k,
                "document_depth": document_depth,
                "evidence_depth": evidence_depth,
                "bm25_weight": bm25_weight,
                "strategy": strategy.value,
                "document_elapsed_ms": documents.elapsed_ms,
                "evidence_elapsed_ms": evidence.elapsed_ms,
            },
            versions=self.versions,
        )
