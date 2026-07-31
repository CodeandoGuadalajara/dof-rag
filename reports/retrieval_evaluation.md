# Evaluación de calidad de recuperación: modelos de embedding

Corpus: `../dof_md-local`
Muestra: **200** documentos markdown (seed 42)
Fecha: 2026-07-30

## Tabla maestra: modelo × variante

Variantes post-hoc sobre los mismos embeddings fp32: `full_fp32` (baseline), `mrl_768` (truncado Matryoshka a 768 dims), `int8` (cuantización escalar), `binary` (signo, 1 bit/dim).

| Modelo | Variante | Dims ef. | Bytes/vec | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG |
|---|---|---|---|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | full_fp32 | 1024 | 4096 | 0.380 | 0.545 | 0.625 | 0.451 | 0.492 |
| perplexity-ai/pplx-embed-context-v1-0.6b | mrl_768 | 768 | 3072 | 0.378 | 0.550 | 0.615 | 0.446 | 0.486 |
| perplexity-ai/pplx-embed-context-v1-0.6b | int8 | 1024 | 1024 | 0.383 | 0.545 | 0.623 | 0.452 | 0.492 |
| perplexity-ai/pplx-embed-context-v1-0.6b | binary | 1024 | 128 | 0.362 | 0.537 | 0.603 | 0.435 | 0.475 |
| codefuse-ai/F2LLM-v2-1.7B | full_fp32 | 2048 | 8192 | 0.480 | 0.650 | 0.725 | 0.554 | 0.595 |
| codefuse-ai/F2LLM-v2-1.7B | mrl_768 | 768 | 3072 | 0.463 | 0.625 | 0.700 | 0.539 | 0.577 |
| codefuse-ai/F2LLM-v2-1.7B | int8 | 2048 | 2048 | 0.480 | 0.650 | 0.725 | 0.554 | 0.595 |
| codefuse-ai/F2LLM-v2-1.7B | binary | 2048 | 256 | 0.482 | 0.637 | 0.693 | 0.547 | 0.582 |
| codefuse-ai/F2LLM-v2-0.6B | full_fp32 | 1024 | 4096 | 0.453 | 0.598 | 0.677 | 0.513 | 0.552 |
| codefuse-ai/F2LLM-v2-0.6B | mrl_768 | 768 | 3072 | 0.450 | 0.600 | 0.657 | 0.512 | 0.547 |
| codefuse-ai/F2LLM-v2-0.6B | int8 | 1024 | 1024 | 0.453 | 0.598 | 0.675 | 0.513 | 0.551 |
| codefuse-ai/F2LLM-v2-0.6B | binary | 1024 | 128 | 0.420 | 0.575 | 0.642 | 0.491 | 0.528 |
| jinaai/jina-embeddings-v5-text-small | full_fp32 | 1024 | 4096 | 0.350 | 0.500 | 0.547 | 0.415 | 0.447 |
| jinaai/jina-embeddings-v5-text-small | mrl_768 | 768 | 3072 | 0.335 | 0.490 | 0.537 | 0.402 | 0.435 |
| jinaai/jina-embeddings-v5-text-small | int8 | 1024 | 1024 | 0.350 | 0.497 | 0.547 | 0.415 | 0.447 |
| jinaai/jina-embeddings-v5-text-small | binary | 1024 | 128 | 0.345 | 0.492 | 0.542 | 0.408 | 0.441 |
| BM25 (SQLite FTS5) | bm25 | - | - | 0.557 | 0.637 | 0.647 | 0.592 | 0.606 |

## Ranking por MRR (full fp32; BM25 como baseline)

1. **BM25 (SQLite FTS5)**: MRR=0.592, Recall@1=0.557, Recall@5=0.637
2. **codefuse-ai/F2LLM-v2-1.7B**: MRR=0.554, Recall@1=0.480, Recall@5=0.650
3. **codefuse-ai/F2LLM-v2-0.6B**: MRR=0.513, Recall@1=0.453, Recall@5=0.598
4. **perplexity-ai/pplx-embed-context-v1-0.6b**: MRR=0.451, Recall@1=0.380, Recall@5=0.545
5. **jinaai/jina-embeddings-v5-text-small**: MRR=0.415, Recall@1=0.350, Recall@5=0.500

## Impacto de la cuantización (Δ MRR vs full fp32)

| Modelo | int8 Δ | binary Δ | mrl_768 Δ |
|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | +0.1 pts | -1.6 pts | -0.5 pts |
| codefuse-ai/F2LLM-v2-1.7B | +0.1 pts | -0.7 pts | -1.5 pts |
| codefuse-ai/F2LLM-v2-0.6B | +0.0 pts | -2.2 pts | -0.1 pts |
| jinaai/jina-embeddings-v5-text-small | +0.0 pts | -0.7 pts | -1.3 pts |

## Notas

- Las queries sintéticas se generan a partir de títulos de documentos y primeras palabras de chunks.
- Recall@k mide si el documento correcto aparece en los top-k chunks recuperados.
- La cuantización int8 es escalar por-vector (absmax); la binaria es sign().
- `mrl_768` solo aplica a modelos con más de 768 dims nativas.
- BM25 usa SQLite FTS5 (`unicode61 remove_diacritics 1`), MATCH con OR de términos, sin stemming; el ranking usa `bm25(chunks)` de FTS5 sobre los mismos chunks y queries.
- Muestra determinística (seed 42, archivos ordenados): reproducible en cualquier máquina.
