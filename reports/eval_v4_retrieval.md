# Eval v4: full BM25 vs partial vectors and hybrid

Run date: 2026-08-09

Reproducibility note: this partial-index checkpoint predates versioning of its
runner, so its exact Git revision is unavailable. The result is intentionally
preserved; subsequent runs from `scripts/eval_v4_full.py` record the Git
revision, dirty state, query hash, embedding model, and live chunk count.

## Index snapshot

- BM25: 657,867 documents (complete).
- Binary vectors: 2,574,336 / 6,730,304 chunks (38.2%).
- Vec0 contiguous through chunk id 2,574,336: True.
- Fully vector-covered v4 questions: 3 / 42 (SP-002, MD-003, MO-002).
- Covered unique gold documents: 2 / 14.

## All 42 questions

This is the operational result against today's indexes. Vector and hybrid scores are coverage-confounded because most gold documents have not yet been embedded.

| System | MRR | Doc R@5 | Doc R@10 | All-hop@10 | All-hop@20 |
|---|---:|---:|---:|---:|---:|
| W0.75(BM25,jina-binary) | 0.237 | 0.381 | 0.429 | 0.405 | 0.429 |
| BM25-doc | 0.221 | 0.381 | 0.429 | 0.405 | 0.429 |
| W0.5(BM25,jina-binary) | 0.118 | 0.167 | 0.381 | 0.357 | 0.405 |
| RRF(BM25,jina-binary) | 0.114 | 0.131 | 0.333 | 0.310 | 0.429 |
| W0.25(BM25,jina-binary) | 0.046 | 0.036 | 0.107 | 0.095 | 0.214 |
| jina-binary-partial | 0.014 | 0.036 | 0.036 | 0.024 | 0.024 |

## Fully covered subset

Only 3 questions currently qualify. This cut is fair to the vector leg but too small for a stable model choice.

| System | MRR | Doc R@5 | Doc R@10 | All-hop@10 | All-hop@20 |
|---|---:|---:|---:|---:|---:|
| RRF(BM25,jina-binary) | 0.389 | 0.167 | 0.500 | 0.333 | 0.333 |
| W0.5(BM25,jina-binary) | 0.364 | 0.167 | 0.167 | 0.000 | 0.333 |
| W0.75(BM25,jina-binary) | 0.343 | 0.167 | 0.167 | 0.000 | 0.000 |
| W0.25(BM25,jina-binary) | 0.278 | 0.500 | 0.500 | 0.333 | 0.333 |
| jina-binary-partial | 0.194 | 0.500 | 0.500 | 0.333 | 0.333 |
| BM25-doc | 0.111 | 0.167 | 0.167 | 0.000 | 0.000 |

## Vector evidence retrieval

| Cut | Evidence R@5 | Evidence R@10 | Evidence R@20 | All evidence@20 |
|---|---:|---:|---:|---:|
| All 42 | 0.008 | 0.008 | 0.008 | 0.000 |
| Fully covered | 0.111 | 0.111 | 0.111 | 0.000 |

## Per-category MRR on all questions

The hybrid column uses `W0.75(BM25,jina-binary)`, the best hybrid by all-question MRR.

| Category | BM25 | Vector | Hybrid |
|---|---:|---:|---:|
| cross_reference | 0.056 | 0.000 | 0.053 |
| list_enumeration | 0.151 | 0.000 | 0.150 |
| monitoring | 0.097 | 0.056 | 0.102 |
| multi_document | 0.282 | 0.042 | 0.391 |
| negative_false_premise | 0.208 | 0.000 | 0.208 |
| single_passage | 0.500 | 0.000 | 0.500 |
| temporal_transitorio | 0.255 | 0.000 | 0.252 |

## Interpretation

- BM25 is the only complete-index baseline in this run.
- The all-question vector score primarily measures current index coverage, not the final embedding model's retrieval quality.
- The fully covered subset should be treated as a mechanical smoke test; rerun the identical command after the vector build and vec0 top-off complete.
- Preserve these outputs as the partial-index checkpoint and compare the final run using the same frozen v4 questions and runner.

## Reproduction

```bash
uv run python scripts/build_vec0_full.py \
  --vectors-db dof_db/dof_vectors_jina_binary.sqlite \
  --vec0-db dof_db/dof_vec0_jina_binary.sqlite
uv run python scripts/eval_v4_full.py
```
