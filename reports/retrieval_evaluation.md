# Evaluación de calidad de recuperación: modelos de embedding

Muestra: **50** documentos markdown de `./dof_md`
Fecha: 2026-07-27

## Resumen general

| Modelo | Dim | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG |
|---|---|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | - | 0.520 | 0.700 | 0.770 | 0.590 | 0.633 |
| perplexity-ai/pplx-embed-v1-0.6b | - | 0.520 | 0.710 | 0.760 | 0.598 | 0.637 |
| nvidia/Nemotron-3-Embed-1B-BF16 | - | 0.350 | 0.560 | 0.630 | 0.434 | 0.481 |
| jinaai/jina-embeddings-v5-text-small | - | 0.450 | 0.660 | 0.710 | 0.537 | 0.579 |
| jinaai/jina-embeddings-v5-text-nano | - | 0.430 | 0.650 | 0.710 | 0.519 | 0.565 |
| Octen/Octen-Embedding-0.6B | - | 0.470 | 0.630 | 0.710 | 0.537 | 0.578 |

## Métricas por tipo de query

| Modelo | Query type | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | title | 0.260 | 0.400 | 0.540 |
| perplexity-ai/pplx-embed-context-v1-0.6b | first_words | 0.780 | 1.000 | 1.000 |
| perplexity-ai/pplx-embed-v1-0.6b | title | 0.260 | 0.440 | 0.520 |
| perplexity-ai/pplx-embed-v1-0.6b | first_words | 0.780 | 0.980 | 1.000 |
| nvidia/Nemotron-3-Embed-1B-BF16 | title | 0.080 | 0.240 | 0.320 |
| nvidia/Nemotron-3-Embed-1B-BF16 | first_words | 0.620 | 0.880 | 0.940 |
| jinaai/jina-embeddings-v5-text-small | title | 0.160 | 0.360 | 0.420 |
| jinaai/jina-embeddings-v5-text-small | first_words | 0.740 | 0.960 | 1.000 |
| jinaai/jina-embeddings-v5-text-nano | title | 0.180 | 0.340 | 0.420 |
| jinaai/jina-embeddings-v5-text-nano | first_words | 0.680 | 0.960 | 1.000 |
| Octen/Octen-Embedding-0.6B | title | 0.220 | 0.360 | 0.460 |
| Octen/Octen-Embedding-0.6B | first_words | 0.720 | 0.900 | 0.960 |

## Notas

- Las queries sintéticas se generan a partir de títulos de documentos y primeras palabras de chunks.
- Recall@k mide si el documento correcto aparece en los top-k chunks recuperados.
- MRR (Mean Reciprocal Rank) y NDCG son métricas estándar de ranking.
- Todos los modelos usan los mismos documentos y queries.

## Conclusión provisional

**Ranking por MRR:**
1. **perplexity-ai/pplx-embed-v1-0.6b**: MRR=0.598, Recall@1=0.520, Recall@5=0.710
2. **perplexity-ai/pplx-embed-context-v1-0.6b**: MRR=0.590, Recall@1=0.520, Recall@5=0.700
3. **jinaai/jina-embeddings-v5-text-small**: MRR=0.537, Recall@1=0.450, Recall@5=0.660
4. **Octen/Octen-Embedding-0.6B**: MRR=0.537, Recall@1=0.470, Recall@5=0.630
5. **jinaai/jina-embeddings-v5-text-nano**: MRR=0.519, Recall@1=0.430, Recall@5=0.650
6. **nvidia/Nemotron-3-Embed-1B-BF16**: MRR=0.434, Recall@1=0.350, Recall@5=0.560

## Siguientes pasos

- Evaluar con queries reales de usuarios.
- Probar late chunking con pplx-embed-context-v1.
- Medir latencia de búsqueda vectorial con sqlite-vec.
- Optimizar hiperparámetros (top-k, umbral de similitud).
