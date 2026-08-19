# Eval v4: full BM25 vs vectors and hybrid

Run date: 2026-08-19

## Index snapshot

- BM25: 657,867 documents (complete).
- Binary vectors: 6,730,304 / 6,730,304 chunks (100.0%).
- Vec0 contiguous through chunk id 6,730,304: True.
- Fully vector-covered v4 questions: 42 / 42 (SP-001, SP-002, SP-003, SP-004, SP-005, SP-006, LI-001, LI-002, LI-003, LI-004, LI-005, LI-006, TE-001, TE-002, TE-003, TE-004, TE-005, TE-006, CR-001, CR-002, CR-003, CR-004, CR-005, CR-006, MD-001, MD-002, MD-003, MD-004, MD-005, MD-006, MO-001, MO-002, MO-003, MO-004, MO-005, MO-006, NE-001, NE-002, NE-003, NE-004, NE-005, NE-006).
- Covered unique gold documents: 14 / 14.

## All 42 questions

This is the operational result against today's indexes. All questions are fully vector-covered, so vector and hybrid scores are unbiased.

| System | MRR | Doc R@5 | Doc R@10 | All-hop@10 | All-hop@20 |
|---|---:|---:|---:|---:|---:|
| W0.5(BM25,jina-binary) | 0.339 | 0.369 | 0.452 | 0.429 | 0.595 |
| RRF(BM25,jina-binary) | 0.331 | 0.369 | 0.464 | 0.429 | 0.571 |
| W0.75(BM25,jina-binary) | 0.326 | 0.405 | 0.452 | 0.429 | 0.476 |
| W0.25(BM25,jina-binary) | 0.312 | 0.405 | 0.476 | 0.452 | 0.548 |
| jina-binary-partial | 0.284 | 0.381 | 0.476 | 0.452 | 0.476 |
| BM25-doc | 0.221 | 0.381 | 0.429 | 0.405 | 0.429 |

## Fully covered subset

All questions qualify, so this cut is identical to the full set.

| System | MRR | Doc R@5 | Doc R@10 | All-hop@10 | All-hop@20 |
|---|---:|---:|---:|---:|---:|
| W0.5(BM25,jina-binary) | 0.339 | 0.369 | 0.452 | 0.429 | 0.595 |
| RRF(BM25,jina-binary) | 0.331 | 0.369 | 0.464 | 0.429 | 0.571 |
| W0.75(BM25,jina-binary) | 0.326 | 0.405 | 0.452 | 0.429 | 0.476 |
| W0.25(BM25,jina-binary) | 0.312 | 0.405 | 0.476 | 0.452 | 0.548 |
| jina-binary-partial | 0.284 | 0.381 | 0.476 | 0.452 | 0.476 |
| BM25-doc | 0.221 | 0.381 | 0.429 | 0.405 | 0.429 |

## Vector evidence retrieval

| Cut | Evidence R@5 | Evidence R@10 | Evidence R@20 | All evidence@20 |
|---|---:|---:|---:|---:|
| All 42 | 0.282 | 0.341 | 0.365 | 0.333 |
| Fully covered | 0.282 | 0.341 | 0.365 | 0.333 |

## Per-category MRR on all questions

The hybrid column uses `W0.5(BM25,jina-binary)`, the best hybrid by all-question MRR.

| Category | BM25 | Vector | Hybrid |
|---|---:|---:|---:|
| cross_reference | 0.056 | 0.042 | 0.048 |
| list_enumeration | 0.151 | 0.204 | 0.359 |
| monitoring | 0.097 | 0.107 | 0.100 |
| multi_document | 0.282 | 0.429 | 0.538 |
| negative_false_premise | 0.208 | 0.219 | 0.261 |
| single_passage | 0.500 | 0.548 | 0.667 |
| temporal_transitorio | 0.255 | 0.435 | 0.403 |

## Interpretation

- Every v4 question is fully vector-covered; this is the final full-corpus comparison, not a partial-index checkpoint.
- The eligible-question cut is identical to the all-question table and is kept only for continuity with earlier partial runs.

## Reproduction

```bash
uv run python scripts/build_vec0_full.py \
  --vectors-db dof_db/dof_vectors_jina_binary.sqlite \
  --vec0-db dof_db/dof_vec0_jina_binary.sqlite
uv run python scripts/eval_v4_full.py
```
