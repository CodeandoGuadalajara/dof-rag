# Full-Corpus Build Log

Status: in progress
Date: 2026-08-03
Branch: `feat/full-corpus-build`

Production build over all 657,867 DOF documents (31.47 GiB Markdown,
manifest at `dof_db/manifest_full.jsonl`, 0 dataless files), following
`docs/corpus-storage-architecture.md` and the post-PoC decisions in
`docs/corpus-storage-poc-results.md`.

## Corpus store: done

`dof_db/dof_corpus_l3.sqlite` — **3.52 GiB** (8.9x), zstd level 3,
`corpus_version = dof-full-v1`.

- Ingestion: 657,867 docs in 428 s (~1,540 docs/s, matches PoC
  throughput); 105 oversized documents segmented (>32 MiB, 2.63 GiB raw).
- Multi-source schema landed first: `documents.source DEFAULT 'dof'`,
  source-aware `dict_chooser` (`dof_1999` … `dof_2026`), path namespacing
  convention documented.
- Per-year dictionaries trained via indexed queries (dict = 1% of group
  bytes, ~4.4–24.7 MB per year); every row compressed with its year dict,
  0 uncompressed rows.
- 300/300 randomly sampled docs hash-verified against the raw tree.
- Compression ratio 8.9x vs the PoC's 10.92x: the delta is the 2.63 GiB
  of `[nodict]` oversized segments (~3x without dictionaries); ordinary
  documents match the PoC ratio.

### Full-scale maintenance gotcha (fixed in ingest.py)

The extension's own maintenance path fails on the full corpus:

- Its todo query `GROUP BY dict_chooser` over uncompressed rows has no
  usable index and spills the whole uncompressed corpus to a temp b-tree
  (measured 26+ GiB etilqs file) → `SQLITE_FULL` with a "full" disk.
- Dict training runs one unindexed full-table scan per chooser group.
- Reservoir sampling keeps ALL rows of groups >2 GiB, exceeding ZDICT's
  2 GB sample limit (the 2011 group is 2.35 GiB).

Fixes now in `corpus_store/ingest.py`: an expression index on the backing
table matching the `dict_chooser`, indexed per-year dict pre-training with
a 1.8 GiB reservoir cap, and bounded maintenance passes (1800 s) with WAL
checkpoints between them. With these, full maintenance completes in ~6 min
and the db shrinks 32 GiB → 3.52 GiB incrementally (auto_vacuum=FULL), so
peak disk use stays modest.

## Binary vector-store pilot: passed

101,351 chunks (the 10k-doc PoC corpus), embedded via llama.cpp GGUF f16
with `Document: ` prefix, sign-packed in-pipeline, only 128-byte blobs on
disk (`corpus_store/embed.py`, resumable by contiguous chunk_id ranges,
config recorded in `vector_meta`, mismatched-config resume refused).

| Check | Result | Full-corpus extrapolation (x66) |
|---|---|---|
| Embedding throughput | 5.36 chunks/s (5.2 h for 101k) | ~14 days for 6.67M |
| Plain checkpoint db | 142 B/vec | 0.88 GiB |
| sqlite-vec `bit[1024]` vec0 db | 151 B/vec | 0.94 GiB |
| Hamming scan k=50 | 5.0 ms mean, 5.3 ms p95 | ~0.33 s/query |
| Bit-exactness (re-embed 64) | max 1 / 1024 bits | — |
| Spot MRR (doc-collapsed) | 0.359 (n=51, 13 gold docs) | see note |

Spot MRR note: only 13 of the 500 eval docs fall inside the 10k pilot
corpus, so the check uses 51 queries against a ~20x harder distractor set
than the PoC (reference anchor 0.545 over the 499-doc corpus). The value
is sane — the storage path itself is proven by the bit-exactness check.
The real quality gate is the full-corpus eval below.

sqlite-vec's hamming scan is ~5x faster than numpy XOR+popcount at this
scale, and `bit[1024]` vec0 needs no new dependencies. Decision confirmed:
sqlite-vec for the binary store; sqlite-vector/TurboQuant4 stays the
documented upgrade path.

## Chunk store: in progress

`dof_db/dof_chunks.sqlite` building at ~52 docs/s (span recipes,
hash-verified; ETA ~3.5 h for 657,867 docs, expected ~6.6–6.7M chunks).

## Remaining

1. Full embedding run over ~6.67M chunks (~14 days continuous at
   5.36 chunks/s; resumable, so interruptions are expected and safe).
2. Doc-level FTS5 build on the full corpus (est. ~2.8 GiB).
3. Full-corpus eval: 499-doc / 3,023-query set over the real stores —
   BM25 vs vectors vs hybrid α=0.5. MRR will drop vs the 499-doc subset;
   that's signal, not regression.
4. License review before any production deployment (sqlite-vector
   modified Elastic 2.0, sqlite-zstd LGPL-3.0; sqlite-vec is MIT).
