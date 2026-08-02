# DOF Corpus Storage Architecture

Status: proposed
Date: 2026-07-28

## Problem

The active text corpus contains approximately:

- 657,867 Markdown files
- 31.47 GiB of Markdown
- ~5.1 million expected chunks at the current 800-token configuration

A 1,024-dimensional float32 vector index for all chunks would require approximately 20.8 GiB before database and index overhead. Storing another copy of every chunk's text would add roughly 33–40 GiB, making the combined source/chunk/vector layout difficult to maintain on a 460 GiB MacBook disk.

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
    year INTEGER NOT NULL,
    publication_date TEXT,
    section TEXT,
    markdown TEXT NOT NULL,
    byte_length INTEGER NOT NULL,
    sha256 BLOB NOT NULL,
    corpus_version TEXT NOT NULL
);

CREATE INDEX documents_year_idx ON documents(year);
CREATE INDEX documents_section_idx ON documents(section);
```

Enable transparent compression on `documents.markdown`. Start with level 3 and benchmark level 19 afterward; level 3 may provide most of the savings with much lower ingestion cost.

Use dictionary groups that reflect corpus structure, such as year and optionally document-size class. Validate the exact `dict_chooser` expression during the proof of concept.

Documents above a configurable threshold, initially 32 MiB uncompressed, should be segmented rather than stored as one very large compressed row. Segment metadata should preserve ordering and byte offsets.

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

Use [sqliteai/sqlite-vector](https://github.com/sqliteai/sqlite-vector) for the next SQLite proof of concept, but keep it in a separate database from the compressed corpus. It is actively maintained, provides macOS ARM binaries, stores vectors as ordinary `BLOB` columns, and supports SIMD exact scans plus 2-, 3-, and 4-bit TurboQuant scans. Test loading sqlite-zstd and sqlite-vector in the same Python process even if the databases remain separate.

TurboQuant is a scan rather than an HNSW/IVF preindex. For approximately 5.1 million 1,024-dimensional vectors, 4-bit TurboQuant should require roughly 2.7 GiB before database overhead. Based on the upstream 1-million-vector macOS ARM benchmark, a full-corpus scan may take on the order of one second when preloaded. That is acceptable for initial limited-user workloads if recall is validated; otherwise use an ANN engine.

The extension uses a modified Elastic License 2.0. This repository is MIT-licensed, which appears to satisfy the open-source grant, but production or managed-service use should be reviewed before deployment.

For larger or latency-sensitive indexes, evaluate LanceDB IVF-PQ, FAISS IVF-PQ, or PostgreSQL pgvector. A quantized 1,024-dimensional index should be much smaller than the ~20.8 GiB raw float32 matrix. Build only one full production model index; continue evaluating alternative models on sampled subsets.

## Estimated storage

Approximate full-corpus targets:

| Component | Target |
|---|---:|
| Compressed Markdown corpus | 2–8 GiB |
| Chunk metadata and offsets | 1–2 GiB |
| sqlite-vec float32 index | 25–30 GiB |
| sqlite-vector TurboQuant4 scan | 3–6 GiB |
| Quantized ANN index | 1–8 GiB |
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
- Record corpus version, document hashes, chunker version, model ID, dimensions, and quantization settings.
- Keep only one full production vector index until disk headroom improves.
