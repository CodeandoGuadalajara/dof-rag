# Evaluación de calidad de recuperación: modelos de embedding

Corpus: `../dof_md-local`
Muestra: **499** documentos markdown (seed 42)
Query set: `eval/dof_queries_v2.jsonl` (verbatim + tipos generados por LLM)
Fecha: 2026-08-01

## Tabla maestra: modelo × variante

Variantes post-hoc sobre los mismos embeddings fp32: `full_fp32` (baseline), `mrl_768` (truncado Matryoshka a 768 dims), `int8` (cuantización escalar), `binary` (signo, 1 bit/dim).

| Modelo | Variante | Dims ef. | Bytes/vec | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG |
|---|---|---|---|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | full_fp32 | 1024 | 4096 | 0.493 | 0.647 | 0.716 | 0.559 | 0.596 |
| perplexity-ai/pplx-embed-context-v1-0.6b | mrl_768 | 768 | 3072 | 0.486 | 0.640 | 0.707 | 0.554 | 0.590 |
| perplexity-ai/pplx-embed-context-v1-0.6b | int8 | 1024 | 1024 | 0.492 | 0.647 | 0.715 | 0.559 | 0.596 |
| perplexity-ai/pplx-embed-context-v1-0.6b | binary | 1024 | 128 | 0.443 | 0.611 | 0.668 | 0.514 | 0.551 |
| codefuse-ai/F2LLM-v2-1.7B | full_fp32 | 2048 | 8192 | 0.534 | 0.677 | 0.725 | 0.595 | 0.627 |
| codefuse-ai/F2LLM-v2-1.7B | mrl_768 | 768 | 3072 | 0.527 | 0.669 | 0.721 | 0.588 | 0.620 |
| codefuse-ai/F2LLM-v2-1.7B | int8 | 2048 | 2048 | 0.533 | 0.677 | 0.726 | 0.595 | 0.627 |
| codefuse-ai/F2LLM-v2-1.7B | binary | 2048 | 256 | 0.518 | 0.662 | 0.714 | 0.579 | 0.611 |
| codefuse-ai/F2LLM-v2-0.6B | full_fp32 | 1024 | 4096 | 0.495 | 0.647 | 0.707 | 0.561 | 0.596 |
| codefuse-ai/F2LLM-v2-0.6B | mrl_768 | 768 | 3072 | 0.485 | 0.644 | 0.704 | 0.553 | 0.589 |
| codefuse-ai/F2LLM-v2-0.6B | int8 | 1024 | 1024 | 0.496 | 0.646 | 0.707 | 0.561 | 0.596 |
| codefuse-ai/F2LLM-v2-0.6B | binary | 1024 | 128 | 0.450 | 0.612 | 0.668 | 0.516 | 0.553 |
| jinaai/jina-embeddings-v5-text-small | full_fp32 | 1024 | 4096 | 0.493 | 0.645 | 0.697 | 0.558 | 0.591 |
| jinaai/jina-embeddings-v5-text-small | mrl_768 | 768 | 3072 | 0.487 | 0.640 | 0.692 | 0.550 | 0.584 |
| jinaai/jina-embeddings-v5-text-small | int8 | 1024 | 1024 | 0.494 | 0.645 | 0.697 | 0.558 | 0.592 |
| jinaai/jina-embeddings-v5-text-small | binary | 1024 | 128 | 0.470 | 0.631 | 0.686 | 0.538 | 0.573 |
| BM25 (SQLite FTS5) | bm25 | - | - | 0.561 | 0.687 | 0.728 | 0.616 | 0.643 |

## Ranking por MRR (full fp32; BM25 como baseline)

1. **BM25 (SQLite FTS5)**: MRR=0.616, Recall@1=0.561, Recall@5=0.687
2. **codefuse-ai/F2LLM-v2-1.7B**: MRR=0.595, Recall@1=0.534, Recall@5=0.677
3. **codefuse-ai/F2LLM-v2-0.6B**: MRR=0.561, Recall@1=0.495, Recall@5=0.647
4. **perplexity-ai/pplx-embed-context-v1-0.6b**: MRR=0.559, Recall@1=0.493, Recall@5=0.647
5. **jinaai/jina-embeddings-v5-text-small**: MRR=0.558, Recall@1=0.493, Recall@5=0.645

## Desglose por tipo de query (full fp32 / bm25)

| Modelo | Tipo | n | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | article_specific | 110 | 0.291 | 0.409 | 0.491 |
| perplexity-ai/pplx-embed-context-v1-0.6b | factual | 1009 | 0.504 | 0.662 | 0.722 |
| perplexity-ai/pplx-embed-context-v1-0.6b | first_words | 499 | 0.613 | 0.808 | 0.908 |
| perplexity-ai/pplx-embed-context-v1-0.6b | paraphrase | 428 | 0.762 | 0.932 | 0.974 |
| perplexity-ai/pplx-embed-context-v1-0.6b | thematic | 478 | 0.456 | 0.667 | 0.770 |
| perplexity-ai/pplx-embed-context-v1-0.6b | verbatim_title | 499 | 0.196 | 0.244 | 0.285 |
| codefuse-ai/F2LLM-v2-1.7B | article_specific | 110 | 0.291 | 0.427 | 0.473 |
| codefuse-ai/F2LLM-v2-1.7B | factual | 1009 | 0.484 | 0.625 | 0.676 |
| codefuse-ai/F2LLM-v2-1.7B | first_words | 499 | 0.770 | 0.926 | 0.960 |
| codefuse-ai/F2LLM-v2-1.7B | paraphrase | 428 | 0.832 | 0.960 | 0.986 |
| codefuse-ai/F2LLM-v2-1.7B | thematic | 478 | 0.521 | 0.736 | 0.816 |
| codefuse-ai/F2LLM-v2-1.7B | verbatim_title | 499 | 0.212 | 0.289 | 0.335 |
| codefuse-ai/F2LLM-v2-0.6B | article_specific | 110 | 0.282 | 0.391 | 0.455 |
| codefuse-ai/F2LLM-v2-0.6B | factual | 1009 | 0.469 | 0.603 | 0.660 |
| codefuse-ai/F2LLM-v2-0.6B | first_words | 499 | 0.683 | 0.880 | 0.942 |
| codefuse-ai/F2LLM-v2-0.6B | paraphrase | 428 | 0.771 | 0.923 | 0.965 |
| codefuse-ai/F2LLM-v2-0.6B | thematic | 478 | 0.441 | 0.692 | 0.793 |
| codefuse-ai/F2LLM-v2-0.6B | verbatim_title | 499 | 0.220 | 0.279 | 0.321 |
| jinaai/jina-embeddings-v5-text-small | article_specific | 110 | 0.300 | 0.464 | 0.582 |
| jinaai/jina-embeddings-v5-text-small | factual | 1009 | 0.494 | 0.621 | 0.677 |
| jinaai/jina-embeddings-v5-text-small | first_words | 499 | 0.597 | 0.852 | 0.916 |
| jinaai/jina-embeddings-v5-text-small | paraphrase | 428 | 0.783 | 0.937 | 0.979 |
| jinaai/jina-embeddings-v5-text-small | thematic | 478 | 0.531 | 0.734 | 0.791 |
| jinaai/jina-embeddings-v5-text-small | verbatim_title | 499 | 0.146 | 0.188 | 0.214 |
| BM25 (SQLite FTS5) | article_specific | 110 | 0.482 | 0.691 | 0.773 |
| BM25 (SQLite FTS5) | factual | 1009 | 0.703 | 0.803 | 0.842 |
| BM25 (SQLite FTS5) | first_words | 499 | 0.876 | 0.990 | 0.996 |
| BM25 (SQLite FTS5) | paraphrase | 428 | 0.565 | 0.766 | 0.794 |
| BM25 (SQLite FTS5) | thematic | 478 | 0.301 | 0.485 | 0.552 |
| BM25 (SQLite FTS5) | verbatim_title | 499 | 0.222 | 0.275 | 0.331 |

## Chunk-level (queries con chunk esperado anotado)

Recall si el chunk exacto que responde la query aparece en top-k.

| Modelo | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | 1618 | 0.434 | 0.635 | 0.709 | 0.521 |
| codefuse-ai/F2LLM-v2-1.7B | 1618 | 0.459 | 0.643 | 0.690 | 0.536 |
| codefuse-ai/F2LLM-v2-0.6B | 1618 | 0.423 | 0.600 | 0.660 | 0.500 |
| jinaai/jina-embeddings-v5-text-small | 1618 | 0.428 | 0.625 | 0.688 | 0.511 |
| BM25 (SQLite FTS5) | 1618 | 0.618 | 0.798 | 0.842 | 0.696 |

## Impacto de la cuantización (Δ MRR vs full fp32)

| Modelo | int8 Δ | binary Δ | mrl_768 Δ |
|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | +0.0 pts | -4.5 pts | -0.6 pts |
| codefuse-ai/F2LLM-v2-1.7B | -0.0 pts | -1.7 pts | -0.7 pts |
| codefuse-ai/F2LLM-v2-0.6B | +0.0 pts | -4.4 pts | -0.8 pts |
| jinaai/jina-embeddings-v5-text-small | +0.1 pts | -2.0 pts | -0.7 pts |

## Notas

- Las queries sintéticas se generan a partir de títulos de documentos y primeras palabras de chunks.
- Recall@k mide si el documento correcto aparece en los top-k chunks recuperados.
- La cuantización int8 es escalar por-vector (absmax); la binaria es sign().
- `mrl_768` solo aplica a modelos con más de 768 dims nativas.
- BM25 usa SQLite FTS5 (`unicode61 remove_diacritics 1`), MATCH con OR de términos, sin stemming; el ranking usa `bm25(chunks)` de FTS5 sobre los mismos chunks y queries.
- Muestra determinística (seed 42, archivos ordenados): reproducible en cualquier máquina.
