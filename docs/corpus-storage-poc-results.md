# Corpus Storage PoC Results

Status: complete
Date: 2026-08-03
Branch: `feat/corpus-store-poc`
Scope: 10,000 randomly selected documents (seed 42, manifest at
`poc/data/manifest_10k.jsonl`, 479.8 MiB of Markdown from 1999–2026),
per the acceptance criteria in `docs/corpus-storage-architecture.md`.

## Verdict

All acceptance criteria pass. sqlite-zstd and sqlite-vector are both
viable for the production build; TurboQuant4 is quality-equivalent to
fp32 after hybrid fusion.

## Acceptance criteria

| Criterion | Target | Result |
|---|---|---|
| Compression ratio | >= 8x | **10.92x** at zstd level 3 (479.8 -> 43.9 MiB); **13.51x** at level 19 (35.5 MiB) |
| Exact round-trip | sha256 equality | 10,000/10,000 docs, incl. reassembled 71.5 MiB segmented doc |
| Chunk reconstruction from offsets | correct | 101,351 chunks, span recipes hash-verified at build; 500/500 random reconstructions hash-verified through the full query path |
| p95 read/decompress | < 50 ms | **0.15 ms** for docs; **6.4 ms** p95 for full chunk reconstruction (fetch + normalize + slice) |
| Oversized docs | predictable memory | >32 MiB docs segmented (sample contains one 71.5 MiB doc, 3 segments); max reconstruct latency 72.7 ms |
| FTS5 external content over compressed view | works | Builds in 4.0 s; BM25 + `snippet()` verified; requires explicit `content_rowid` (views have no `rowid`) |
| sqlite-zstd + sqlite-vector coexist | one process | Verified |
| Shrink after maintenance + VACUUM | works | Verified (57.1 -> 43.9 MiB after first VACUUM) |
| Resumable ingestion | safe after interruption | Idempotent by path; SIGKILL during compression maintenance recovers cleanly; post-compression resume inserts verified by hash |
| Version recording | corpus/chunker/model | `corpus_meta` table + per-row `corpus_version`; per-chunk `chunker_version` |

## Key measurements

### Compression: level 3 vs 19

| Level | Corpus db | Ratio | Maintenance |
|---|---:|---:|---:|
| 3 | 43.9 MiB | 10.92x | 11 s |
| 19 | 35.5 MiB | 13.51x | 47 s |

Level 19 compresses 24% better; read latency is identical (0.15 ms p95).
For the full corpus the difference is ~0.5 GiB — either is fine; level 3
keeps ingestion cheap, level 19 is a one-time 4-5x maintenance cost.

### Doc-level FTS5 index size (the main unmeasured risk)

External-content FTS5 over 479.8 MiB of text: **43.2 MiB (0.09x text)**,
built in 4 s. The earlier 1.36x figure was for a regular-content
chunk-level FTS (which stores full text); external content stores only
postings, and DOF boilerplate shares vocabulary heavily.

### Chunk store

101,351 chunks from 10k docs (patterns: 55,479 giant_table, 17,615
bold_headers, 16,530 h2_compound, 7,461 small, 4,266 plain_text).
Chunks store **no text**: each records a span recipe into the chunker's
normalized text `C(doc)` (deterministically recomputable: image inlining +
boilerplate removal, keeping H2 heading lines for compound docs).

- Recipes average **110 B/chunk** (vs ~2.5 KB of chunk text).
- 1.31% of chunks fall back to embedded literal text (near-duplicate
  SENTENCIA compound docs defeat span alignment); reconstruction is
  exact either way — fallbacks are a size optimization, not correctness.
- `dof_chunks.sqlite`: 43 MiB for 10k docs (~2.8 GiB extrapolated to
  ~5.1M chunks — slightly above the 1–2 GiB target; `token_count` and
  `heading_path` could be trimmed if needed).

### Vector store: sqlite-vector TurboQuant

Extension: sqlite-vector 1.0.0 prebuilt macOS ARM64 binary (NEON).
Dataset: cached eval embeddings (8,065 chunks / 3,023 queries / 1,024
dims, jina-v5-text-small fp32), numpy fp32 cosine as ground truth,
depth 50.

| Mode | R@10 | R@50 | ms/query | B/vec |
|---|---:|---:|---:|---:|
| full scan (exact) | 0.992 | 0.997 | 8.7 | 4096 |
| TurboQuant4 | 0.953 | 0.966 | 2.1 | 524 |
| TurboQuant3 | 0.924 | 0.939 | 3.8 | 396 |
| TurboQuant2 | 0.866 | 0.890 | 0.8 | 268 |

(TurboQuant3 is slower than 4 — matches upstream's real-data benchmark.)

**End-to-end hybrid MRR** (doc-level BM25 + doc-collapsed vectors,
weighted fusion, `evaluate_hybrid_doclevel.py` harness with ranked lists
from the real extension):

| system, alpha=0.5 | MRR | R@1 |
|---|---:|---:|
| W(BM25doc, jina-turbo4) | **0.656** | 0.581 |
| W(BM25doc, jina-fp32) | 0.656 | 0.584 |
| W(BM25doc, jina-turbo3) | 0.654 | 0.579 |
| W(BM25doc, jina-binary) — sqlite-vec fallback | 0.649 | 0.574 |
| W(BM25doc, F2LLM-int8) — quality option | 0.662 | 0.594 |

TurboQuant4 is quality-equivalent to fp32 after fusion and beats the
validated sqlite-vec binary fallback, at 7.8x smaller storage.

## Full-corpus storage extrapolation (31.47 GiB, ~5.1M chunks)

| Component | Estimate | Basis |
|---|---:|---|
| Compressed corpus (L3 / L19) | 2.9 / 2.3 GiB | 10.92x / 13.51x |
| Doc-level FTS5 | ~2.8 GiB | 0.09x text |
| Chunk metadata + recipes | ~2.8 GiB | 43 MiB per 10k docs |
| TurboQuant4 vectors | ~2.5 GiB | 524 B/vec |
| **Total** | **~11 GiB** | fits comfortably in 19 GiB free |

## Engineering notes (gotchas worth keeping)

- **sqlite-zstd local build** (no published macOS ARM binary): built
  0.3.5 via `mise use rust` in 23 s. Required a vendored libsqlite3-sys
  patch: the loadable-extension version guard compares the *bundled*
  SQLite header (3.49) against the host runtime (uv Python 3.12 links
  3.47.1); floored the guard to 3.34. The extension file must be named
  `sqlitezstd.dylib` to match its init symbol.
- `PRAGMA auto_vacuum=FULL` must be set **before** `journal_mode=WAL`,
  or it silently does nothing.
- Transparent compression renames dict indexes per *column name* — two
  compressed `markdown` columns collide; the segments table uses
  `segment_text`.
- ZDICT training fails on few huge rows ("Src size is incorrect") —
  oversized segments use `"dict_chooser": "'[nodict]'"`.
- FTS5 external content on the compressed view requires explicit
  `content_rowid='document_id'` (views have no `rowid`); FTS queries
  then use `rowid` as the document id.
- `vector_init` context is per-connection; call it on every connection
  before `vector_quantize*` / scans.
- Python's implicit transactions must be committed before
  `zstd_enable_transparent`.

## Licenses (must review before production)

- **sqlite-vector**: modified Elastic License 2.0. This repo is MIT,
  which appears to satisfy the open-source grant, but production or
  managed-service use needs review.
- **sqlite-zstd**: LGPL-3.0. As a dynamically loaded extension the
  obligations are likely limited, but distribution inside another
  application needs review. Upstream also warns not to trust it without
  backups — the raw Markdown tree remains the source of truth.

## Next steps

1. Full-corpus build (657,867 docs): ~6 min/10k docs ingestion +
  chunking pipeline; embeddings via llama.cpp GGUF f16 (~14 days at
  5.42 chunks/s) with explicit "Document: "/"Query: " prefixes.
2. PostgreSQL + pgvector prototype only if limited-production
  concurrency is confirmed (see architecture doc).
3. Blog post on the PoC results (dof-rag-website).
