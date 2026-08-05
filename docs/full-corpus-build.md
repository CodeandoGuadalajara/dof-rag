# Full-Corpus Build Log

Status: embedding run in progress (ETA ~13.5 days from 2026-08-04)
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

## Chunk store: done

`dof_db/dof_chunks.sqlite` — **2.68 GiB**, 657,867 docs -> **6,730,304
chunks** (10.2 chunks/doc, matching the PoC's 10.1), built in 4.2 h
(~44 docs/s with the batch tokenizer path).

- 0.75% literal fallbacks (PoC: 1.31%); recipes average 91 B/chunk.
- 500/500 random reconstructions hash-verified through the full query
  path (p50 1.05 ms, p95 8.85 ms per chunk).
- The 671.7 MiB monster doc chunks fine (583 s, 10,658 chunks).

### Chunker fixes (exact-output-preserving, parity 799/799)

Two pre-existing chunker bugs surfaced at full scale (doc 129,449, a
6.2 MiB ASCII-grid table, stalled the build for hours):

- `_flush_table`: a "header" larger than MAX_TOKENS made
  `max_row_tokens <= 0`, so `_force_split` emitted 1-char pieces each
  prepended with the giant header — 519k chunks / 21 GiB of chunk text
  from one 6.2 MiB doc. Guard: oversized headers are treated as no
  header. Only affects docs that previously could not finish at all.
- Per-row/per-paragraph tokenizer calls now batch through the inner Rust
  tokenizer's `encode_batch` (counts verified equal), recounts are
  memoized, and oversized-row force-splits run in parallel (the Rust
  tokenizer releases the GIL). The pathological doc now chunks in 4 s.

Parity was verified against the HEAD implementation on all 499 eval docs
plus 300 random docs (799/799 bit-identical), so `chunker_version` stays
`dof-chunker-v1` and cached eval embeddings remain valid. A seeded
binary-search variant of `_force_split` was tried and **reverted**: token
counts are not monotone in prefix length (rare BPE merges), and seeding
changed cut points in 7/799 docs.

## Full embedding run: in progress

`dof_db/dof_vectors_jina_binary.sqlite`, all 6,730,304 chunks,
`corpus_store/embed` (resumable by contiguous chunk_id ranges, config in
`vector_meta`). Measured 5.36–5.77 chunks/s -> ETA ~13.5 days;
interruptions are expected and safe — reruns resume after
`MAX(chunk_id)`. Monitor `logs/full_embed.log`.

## Doc-level FTS5: done

Built 2026-08-04 in ~5 min (~2,400 docs/s) via `scripts/build_fts_full.py`
(batched by `document_id` ranges, sidecar `_fts_build_meta` progress table
for resumability). Final db size 6.2 GiB => FTS index ~2.7 GiB, matching
the ~2.8 GiB estimate. Sanity: 657,867 rows in `documents_fts_docsize`,
`'de'` matches 99.97% of docs, `bm25` + `snippet` queries return sane
results.

Two gotchas hit and handled:

- The 32 segmented oversized docs have `markdown = ''` in `documents`;
  they are reassembled from `document_segments` and indexed separately
  (the `'rebuild'` path would have indexed them as empty).
- On an external-content FTS5 table, `COUNT(*)` / `MAX(rowid)` scan the
  CONTENT table, not the index — a naive resume check saw 657,867
  "already indexed" rows in an empty index. Use a real `MATCH` query or
  the `docsize` shadow table to check index state.

Rebuilt 2026-08-05 with `tokenize='unicode61 remove_diacritics 1'` to
match the eval-harness baseline (so full-corpus vs subset deltas are
corpus-size effects, not tokenizer effects); the first build used the
default tokenizer (no diacritics folding). Also added
`documents_fts_vocab` (fts5vocab, 2.35M terms) for df lookups.

Full-corpus BM25 querying gotcha: the harness's OR-of-quoted-tokens MATCH
is pathological at this scale — stopword-class tokens ('de' has df
657,642/657,867) force 300k+ row doclist scans, 17-45 s/query (~21 h for
the eval set). Tokens with df > N/2 have zero/negative IDF in FTS5 bm25
and are pruned from MATCH (verified: identical top-50 doc sets, unchanged
gold ranks): 0.3-0.8 s/query. See `scripts/eval_bm25_full.py`.

## Eval prep (while embeddings run)

- **Queries pre-embedded** (2026-08-05): all 3,023 eval queries via the
  same GGUF/llama-server with `Query: ` prefix
  (`scripts/embed_eval_queries.py`, 582 s). Cached in `eval/cache/` as
  `gguf_jina_v5_small_queries_{float.npy,bin.npy,meta.jsonl}`
  (gitignored, deterministic from model + queries).
- **Partial spot check** (`scripts/spot_check_partial.py`, rerunnable as
  the run progresses): 245 eligible queries (expected doc fully embedded)
  against 73,719 docs / 583,680 chunks, hamming k=50 doc-collapsed:
  **MRR 0.218** (first_words 0.403, paraphrase 0.336, factual 0.214,
  thematic 0.110, verbatim_title 0.054) at 31 ms/query. Distractor set is
  ~148x the 499-doc subset, so a large drop from the 0.545 anchor is
  expected; watch verbatim_title in the full eval (near-identical titles
  among same-era/same-section docs + binary quantization). vec0 store
  topped off to 583,680 vectors (resumable, ~2 s per top-off).

## Full-corpus eval: BM25 leg done (2026-08-05)

`scripts/eval_bm25_full.py` over all 3,023 queries against
documents_fts (657,867 docs, depth 50, df-pruned MATCH):

| metric | 499-doc subset | full corpus |
|---|---|---|
| MRR | 0.589 | **0.170** |
| R@1 | 0.530 | 0.119 |
| R@5 | 0.668 | 0.224 |
| R@10 | 0.713 | 0.269 |

Per type (full): factual 0.282, first_words 0.227, paraphrase 0.118,
article_specific 0.118, verbatim_title 0.082, thematic 0.025. The ~3.5x
MRR drop vs the subset is the expected distractor effect (~1,319x more
docs). Ranked lists saved to `eval/cache/full_corpus_bm25_lists.jsonl`
for the hybrid fusion; 34 min wall time.

## Remaining

1. ~~Full embedding run~~ (in progress, see above).
2. ~~Doc-level FTS5 build~~ (done, see above).
3. ~~BM25 leg of full-corpus eval~~ (done, see above).
3. Build the sqlite-vec `bit[1024]` vec0 search store from
   `chunk_vectors` once embeddings complete — script ready:
   `scripts/build_vec0_full.py` (resumable after MAX(rowid); re-run the
   same command to top off). Dry-run on the first 311k partial vectors:
   146k inserts/s (~50 s for 6.73M), 151 B/vec (~0.97 GiB full), k=50
   hamming 16.5 ms -> extrapolates to ~0.36 s/query at 6.73M (expected
   ~0.3 s).
4. Full-corpus eval: 499-doc / 3,023-query set over the real stores —
   BM25 vs vectors vs hybrid α=0.5. Queries already embedded (see Eval
   prep). BM25 leg can run any time against documents_fts. MRR will drop
   vs the 499-doc subset; that's signal, not regression. Do not commit
   re-run noise to `eval/cache/hybrid_doclevel_results.json` (harness is
   slightly nondeterministic at the 4th decimal).
5. License review before any production deployment (sqlite-vector
   modified Elastic 2.0, sqlite-zstd LGPL-3.0; sqlite-vec is MIT).
