"""Typed, provider-neutral contracts exposed to a DOF research agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any


class RetrievalStrategy(StrEnum):
    """Available retrieval components for a tool call."""

    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"


def _validate_iso_date(value: str | None, field_name: str) -> None:
    if value is None:
        return
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date (YYYY-MM-DD)") from exc


@dataclass(frozen=True)
class SearchFilters:
    """Metadata filters currently supported by the corpus database.

    Institution and document-type filters are intentionally absent until the
    corpus has versioned, validated columns for them. A tool must never pretend
    to apply a filter that the store cannot enforce.
    """

    as_of: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    section: str | None = None

    def __post_init__(self) -> None:
        _validate_iso_date(self.as_of, "as_of")
        _validate_iso_date(self.date_from, "date_from")
        _validate_iso_date(self.date_to, "date_to")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        if self.section is not None:
            normalized = self.section.strip().upper()
            if not normalized:
                raise ValueError("section must not be empty")
            object.__setattr__(self, "section", normalized)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IndexVersions:
    corpus_version: str | None
    chunker_version: str | None
    vector_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicationHit:
    document_id: int
    path: str
    publication_date: str | None
    section: str | None
    title: str | None = None
    institution: str | None = None


@dataclass(frozen=True)
class DocumentHit:
    document_id: int
    path: str
    publication_date: str | None
    section: str | None
    score: float
    bm25_score: float | None = None
    vector_score: float | None = None
    rank: int = 0
    title: str | None = None
    institution: str | None = None
    title_boost: float = 0.0
    recency_boost: float = 0.0


@dataclass(frozen=True)
class EvidenceHit:
    chunk_id: int
    document_id: int
    path: str
    publication_date: str | None
    section: str | None
    chunk_index: int
    heading_path: list[str]
    text: str
    score: float
    source: str
    rank: int = 0
    title: str | None = None


@dataclass(frozen=True)
class OutlineChunk:
    chunk_id: int
    chunk_index: int
    heading_path: list[str]
    token_count: int


@dataclass(frozen=True)
class DocumentOutline:
    document_id: int
    path: str
    publication_date: str | None
    section: str | None
    chunks: list[OutlineChunk]
    title: str | None = None
    institution: str | None = None


@dataclass
class DocumentSearchResult:
    query: str
    strategy: RetrievalStrategy
    filters: SearchFilters
    documents: list[DocumentHit] = field(default_factory=list)
    vector_candidates_scanned: int = 0
    elapsed_ms: float = 0.0
    versions: IndexVersions | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def document_ids(self) -> list[int]:
        return [hit.document_id for hit in self.documents]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["strategy"] = self.strategy.value
        data["document_ids"] = self.document_ids
        return data


@dataclass
class EvidenceSearchResult:
    query: str
    strategy: RetrievalStrategy
    document_ids: list[int]
    evidence: list[EvidenceHit] = field(default_factory=list)
    vector_candidates_scanned: int = 0
    elapsed_ms: float = 0.0
    versions: IndexVersions | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_ids(self) -> list[int]:
        return [hit.chunk_id for hit in self.evidence]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["strategy"] = self.strategy.value
        data["evidence_ids"] = self.evidence_ids
        return data


@dataclass
class SearchResult:
    """Compatibility result for the original retrieve-then-answer baseline."""

    query: str
    as_of: str | None
    documents: list[DocumentHit] = field(default_factory=list)
    evidence: list[EvidenceHit] = field(default_factory=list)
    vector_available: bool = False
    vector_count: int = 0
    settings: dict[str, Any] = field(default_factory=dict)
    versions: IndexVersions | None = None

    @property
    def document_ids(self) -> list[int]:
        return [hit.document_id for hit in self.documents]

    @property
    def evidence_ids(self) -> list[int]:
        return [hit.chunk_id for hit in self.evidence]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["document_ids"] = self.document_ids
        data["evidence_ids"] = self.evidence_ids
        return data
