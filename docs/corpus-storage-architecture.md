# DOF Corpus Storage Architecture

Status: proposed
Date: 2026-07-28

## Problem

The active text corpus contains approximately:

- 657,867 Markdown files
- 31.47 GiB of Markdown
- ~6.6–6.7 million expected chunks at the current 800-token configuration
  (the PoC measured 10.1 chunks/doc on 10k documents, more than the ~5.1M
  originally estimated)

A 1,024-dimensional float32 vector index for all chunks would require approximately 27 GiB before database and index overhead — this does not fit the disk budget, so the production build uses binary (sign) quantization instead (see "Vector store"). Storing another copy of every chunk's text would add roughly 33–40 GiB, making the combined source/chunk/vector layout difficult to maintain on a 460 GiB MacBook disk.

The architecture should store document text once, avoid duplicated chunk text, and keep vector indexes reproducible from versioned corpus metadata.

## Corpus-store choice: sqlite-zstd

Use [sqlite-zstd](https://github.com/phiresky/sqlite-zstd) for the first corpus-store implementation. It provides transparent, dictionary-based, row-level Zstandard compression for SQLite while retaining random access.

Compared with storing the corpus as individual Markdown files, this should provide:

- one compact corpus database;
- efficient point lookups by `document_id`;
- much less filesystem metadata overhead;
- dictionary compression across similar DOF documents;
- straightforward joins to document and chunk metadata.

A 3,000-file archive sample compressed by 11.18× at `zstd -3` and 17.35× at `zstd -19`. Row-level compression will not necessarily match an archive, but DOF dictionaries should capture much of the repeated boilerplate. The proof of concept should target at least an 8× reduction for ordinary Markdown rows.

Parquet/Zstandard remains the fallback for bulk analytical storage if sqlite-zstd does not meet correctness, support, or performance requirements.

## sqlite-zstd caveats

sqlite-zstd is promising but should be treated as experimental:

- The upstream README explicitly says not to trust it without backups.
- The SQLite extension must be loaded on every connection.
- Compressed databases cannot be attached with `ATTACH`.
- DDL support is only partial.
- Incremental compression maintenance and `VACUUM` are required to reclaim space.
- Published release binaries do not currently include macOS ARM, so this project may need to build the extension locally.
- The extension is LGPL-3.0 licensed; review distribution implications before shipping it inside another application.
- Very large rows are decompressed into memory. The corpus currently contains at least one Markdown file larger than 600 MiB, so oversized documents need special handling.

The raw Markdown tree remains the source of truth and backup. The SQLite corpus is a derived artifact that can always be rebuilt.

## sqlite-zstd index and DDL strategy

Transparent compression renames the original table to a backing table and replaces the original name with an editable view. The extension also rejects compressing a column that already participates in a normal SQLite index. Therefore:

- never create a normal index directly on `documents.markdown`;
- create required metadata indexes, such as year, section, and path, before enabling compression;
- treat the compressed table schema as stable and migrate by rebuilding instead of relying on later `ALTER TABLE` statements;
- create lexical and vector acceleration as separate derived objects after compression;
- do not place vector `BLOB` columns in the compressed documents table.

This limitation does not prevent BM25 or vector search, but it makes schema evolution less convenient. If that operational risk is unacceptable, use application-managed Zstandard blobs instead: store `markdown_zstd BLOB` in an ordinary SQLite table, decompress in application code, and retain unrestricted SQLite DDL at the cost of custom dictionary and compression management.

## Logical layout

Keep three derived stores separate because they have different rebuild lifecycles:

```text
dof_corpus.sqlite       # compressed documents + corpus metadata
dof_chunks.sqlite       # chunk references and chunker metadata
dof_vectors_<model>/    # sqlite-vector, LanceDB, or FAISS index
```

The stores can live under `dof_db/`, which should remain excluded from Git.

## Document store

Illustrative schema:

```sql
CREATE TABLE documents (
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

CREATE INDEX documents_source_idx ON documents(source);
CREATE INDEX documents_year_idx ON documents(year);
CREATE INDEX documents_section_idx ON documents(section);
```

`source` identifies the corpus a document belongs to (`'dof'` for the
current corpus; future sources such as the federal constitution or state
laws get their own ids). Because the compressed schema is
migrate-by-rebuild, this column had to land before the full-corpus build.
`documents.path` is namespaced per source: DOF keeps its historical
relpaths (`1999/01/...`), and future sources must prefix their paths with
`'<source>/'` (e.g. `constitucion/...`) so `path` stays globally UNIQUE
and idempotent-by-path resume keeps working.

Enable transparent compression on `documents.markdown`. Start with level 3 and benchmark level 19 afterward; level 3 may provide most of the savings with much lower ingestion cost.

The zstd `dict_chooser` is source-aware — `printf('%s_%d', source, year)` —
so each source trains its own per-year dictionaries instead of diluting the
DOF ones. Changing the chooser expression between builds is safe because
decompression resolves dictionaries by the stored dict id, not by the
expression.

Documents above a configurable threshold, initially 32 MiB uncompressed, should be segmented rather than stored as one very large compressed row. Segment metadata should preserve ordering and byte offsets.

## Multi-source readiness

Adding non-DOF sources (federal constitution, state laws, regulations) is
incremental on top of this architecture, with two deliberate constraints:

- **Per-source chunkers use the existing `chunker_version` mechanism.** A
  constitution chunker (article-aware) or a state-law chunker registers a
  new `chunker_version` and coexists with `dof-chunker-v1` chunks in the
  same chunk store (`UNIQUE(document_id, chunk_index, chunker_version)`).
  Span recipes already reference normalized text per chunker, so no schema
  change is needed.
- **Per-source evaluation is required before mixing.** BM25 IDF statistics
  and vector-candidate distributions change when sources with different
  vocabulary and document sizes share one index, so each source needs its
  own eval slice plus a mixed-corpus eval before fusion weights (α) tuned
  on DOF-only data are trusted.

The `documents.source` column and source-aware zstd dictionaries (above)
are the only schema-level changes; everything else is additive.

## Chunk store

Do not store full chunk text. Store references into the source document:

```sql
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(document_id),
    chunk_index INTEGER NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    heading_path TEXT,
    chunk_hash BLOB NOT NULL,
    chunker_version TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    UNIQUE(document_id, chunk_index, chunker_version)
);

CREATE INDEX chunks_document_idx ON chunks(document_id);
CREATE INDEX chunks_chunker_idx ON chunks(chunker_version);
```

The offsets refer to the exact normalized Markdown representation used by the chunker. Record `corpus_version`, document hashes, and `chunker_version` so stale chunk and vector indexes can be detected.

At query time:

1. Search vectors and receive chunk IDs.
2. Fetch chunk offsets and document IDs.
3. Read only the required top-k documents from `dof_corpus.sqlite`.
4. Decompress those documents and slice the requested ranges.
5. Add neighboring chunks or parent sections when more context is needed.

## Lexical search and BM25

SQLite FTS5 can index the editable `documents` view as an external-content source. The FTS index stores its own token postings while document text remains compressed in the backing table. Create FTS after enabling compression, for example:

```sql
CREATE VIRTUAL TABLE documents_fts USING fts5(
    markdown,
    content='documents',
    content_rowid='document_id'
);

INSERT INTO documents_fts(documents_fts) VALUES('rebuild');
```

Use `bm25(documents_fts)` for document-level lexical candidates, then fuse those candidates with chunk-level vector results. Auxiliary functions such as `snippet()` or `highlight()` will decompress only the documents needed for returned rows.

The proof of concept must validate this interaction explicitly. In particular, test FTS creation after compression, full rebuild, top-k reads, and document metadata indexes. The initial corpus is immutable, so explicit periodic rebuilds are preferable to maintaining additional triggers around the compressed view. Chunk-level BM25 can be added later through a separate contentless FTS table if document-level lexical retrieval is insufficient.

## Vector store

**Decision (post-PoC): the full-corpus build stores jina binary (sign)
embeddings.** `sign()` runs inside the embedding pipeline on the in-memory
fp32 vector returned by llama.cpp and only the packed 128-byte blob
(1,024 dims / 8) touches disk — no fp32 vectors are stored and no separate
quantization pass is needed. Rationale:

- Hybrid MRR is 0.649–0.650 for binary vs 0.656 for TurboQuant4/fp32
  (~1 point; binary was already the validated production fallback).
- fp32 storage does not fit: ~6.67M chunks × 4 KiB ≈ 27 GiB.
- sqlite-vector's `vector_quantize` (TurboQuant) requires stored fp32
  blobs as input, so TurboQuant cannot avoid the fp32 disk problem either.
- Hamming scan (XOR + popcount) is the fastest exact scan option and works
  with sqlite-vec `bit[1024]` tables (`vec_bit()`, already a dependency)
  or sqlite-vector 1Bit mode.

Expected store size: ~6.67M × 128 B ≈ **0.85 GiB** plus row overhead.

TurboQuant4 remains the documented quality upgrade path if disk headroom
improves (e.g. offloading fp32 to external storage during the build, or a
two-stage binary-scan + int8/fp32-rerank design). The recall and hybrid
harnesses (`scripts/poc_turboquant_recall.py`,
`scripts/poc_hybrid_turboquant.py`) make re-validation cheap.

For larger or latency-sensitive deployments, evaluate LanceDB IVF-PQ, FAISS IVF-PQ, or PostgreSQL pgvector (which also supports binary quantization with reranking). Build only one full production model index; continue evaluating alternative models on sampled subsets.

The extension licenses still need review before production deployment:
sqlite-vector uses a modified Elastic License 2.0 (this repository is
MIT-licensed, which appears to satisfy the open-source grant) and
sqlite-zstd is LGPL-3.0. sqlite-vec is MIT. Neither blocks the local build.

## Estimated storage

Approximate full-corpus targets (updated with the PoC's 10.1 chunks/doc
measurement, i.e. ~6.67M chunks):

| Component | Target |
|---|---:|
| Compressed Markdown corpus | 2.3–2.9 GiB |
| Doc-level FTS5 | ~2.8 GiB |
| Chunk metadata and span recipes | ~2.8 GiB |
| Binary (sign) vector store, 128 B/vec | ~0.85 GiB |
| TurboQuant4 upgrade path, 524 B/vec | ~3.5 GiB |
| fp32 vectors (rejected: does not fit) | ~27 GiB |
| Model caches | ~10 GiB |

The raw Google Drive corpus may consume another ~60 GiB if fully pinned. Derived databases must remain on local APFS storage, not inside Google Drive.

## Storage-engine alternatives

### PostgreSQL + TOAST LZ4 + pgvector

This is the strongest limited-production alternative. PostgreSQL provides many concurrent readers and writers, transactions, point-in-time recovery, replication, mature full-text search, partitioning, and pgvector HNSW/IVFFlat indexes. `halfvec` halves vector storage and pgvector also supports binary quantization with reranking.

TOAST LZ4 is useful for large document rows, but it compresses values independently and does not train a cross-document dictionary. Expect a worse compression ratio than sqlite-zstd or a Zstandard archive, especially given the DOF's repeated boilerplate. The likely tradeoff is more disk usage in exchange for a much more mature production database.

A production-oriented stack could be:

```text
PostgreSQL documents table, TOAST LZ4
PostgreSQL chunks table, offsets only
pgvector halfvec or binary-quantized HNSW
PostgreSQL FTS, pg_search, or pgturbohybrid for lexical/hybrid retrieval
```

### DuckDB

DuckDB is excellent for corpus ingestion, compression experiments, analytics, and Parquet interoperability. It is less suitable as the production serving store: a database file allows many readers, but cross-process read-write access is effectively limited to one writer/process at a time. DuckDB's vector-search ecosystem is also less mature than pgvector, Qdrant, or LanceDB.

Use DuckDB as the ETL and benchmarking engine, not as the first multi-client query service.

### Dedicated search engines

Qdrant is a good dedicated-vector alternative if pgvector or sqlite-vector cannot meet latency or filtering requirements. It supports HNSW, scalar quantization, snapshots, and payload references while document text remains in SQLite or PostgreSQL.

OpenSearch or Elasticsearch can combine source text, BM25, dense vectors, and hybrid ranking in one production system, but they require substantially more memory and operational effort than SQLite or PostgreSQL.

### pgturbohybrid

[pgturbohybrid](https://github.com/mayflower/pgturbohybrid) is promising for dense-vector plus BM25 hybrid retrieval and reports strong 4-bit throughput/recall benchmarks. It is explicitly alpha software with changing APIs and index formats. Evaluate it after establishing a stable pgvector baseline; do not make the initial corpus architecture depend on it.

### Recommended path

1. Use sqlite-zstd plus sqlite-vector for the local proof of concept.
2. Keep document and chunk schemas independent from the vector extension.
3. Benchmark TurboQuant4 recall and latency against exact float32 results.
4. Prototype PostgreSQL with TOAST LZ4 and pgvector in parallel if limited-production deployment is likely.
5. Move to Qdrant, pgvectorscale, or another dedicated engine only if measured recall/latency requires it.

## Proof of concept

Build the first implementation over 10,000 randomly selected Markdown documents.

Acceptance criteria:

- compression ratio of at least 8× for ordinary documents;
- exact round-trip equality with source Markdown;
- correct chunk reconstruction from offsets;
- p95 single-document read/decompression latency under 50 ms for ordinary documents;
- predictable memory use for oversized documents;
- FTS5 external-content creation, rebuild, and BM25 queries work against the compressed view;
- sqlite-zstd and sqlite-vector coexist in one Python process;
- database size shrinks after maintenance and `VACUUM`;
- ingestion can resume safely after interruption;
- corpus, chunker, and model versions are recorded.

Measure both sqlite-zstd level 3 and level 19. If level 19 adds little compression, prefer level 3 for faster ingestion.

## Operational rules

- Treat the raw Markdown tree as immutable input.
- Never store `.env` values or credentials in corpus databases.
- Keep `.bak` files and media assets out of the initial text corpus.
- Do not store full chunk text unless a later benchmark proves that query-time reconstruction is a bottleneck.
- Batch inserts, checkpoint the WAL regularly, and run `VACUUM` only after major ingestion phases.
- Record corpus version, document hashes, chunker version, model ID, dimensions, quantization settings, and embedding prefixes (`Document: `/`Query: ` — required by jina via llama.cpp).
- Keep only one full production vector index until disk headroom improves.
- The embedding run must be resumable from the start: checkpoint by
  contiguous `chunk_id` ranges, batch inserts in single transactions, and
  refuse to resume with a mismatched model/prefix/packing config.
