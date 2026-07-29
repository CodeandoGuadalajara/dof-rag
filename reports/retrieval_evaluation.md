# Evaluación de calidad de recuperación: modelos de embedding

Muestra: **50** documentos markdown de `./dof_md`
Fecha: 2026-07-29

## Resumen general

| Modelo | Dim | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG |
|---|---|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | - | 0.420 | 0.650 | 0.700 | 0.511 | 0.557 |
| perplexity-ai/pplx-embed-v1-0.6b | - | 0.450 | 0.610 | 0.640 | 0.512 | 0.543 |
| nvidia/Nemotron-3-Embed-1B-BF16 | - | 0.300 | 0.440 | 0.470 | 0.359 | 0.386 |
| jinaai/jina-embeddings-v5-text-small | - | 0.410 | 0.560 | 0.580 | 0.464 | 0.492 |
| jinaai/jina-embeddings-v5-text-nano | - | 0.380 | 0.530 | 0.570 | 0.443 | 0.474 |
| Octen/Octen-Embedding-0.6B | - | 0.410 | 0.530 | 0.570 | 0.455 | 0.482 |

## Métricas por tipo de query

| Modelo | Query type | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | first_words | 0.640 | 0.980 | 1.000 |
| perplexity-ai/pplx-embed-context-v1-0.6b | title | 0.200 | 0.320 | 0.400 |
| perplexity-ai/pplx-embed-v1-0.6b | first_words | 0.660 | 0.960 | 1.000 |
| perplexity-ai/pplx-embed-v1-0.6b | title | 0.240 | 0.260 | 0.280 |
| nvidia/Nemotron-3-Embed-1B-BF16 | first_words | 0.540 | 0.820 | 0.880 |
| nvidia/Nemotron-3-Embed-1B-BF16 | title | 0.060 | 0.060 | 0.060 |
| jinaai/jina-embeddings-v5-text-small | first_words | 0.700 | 0.960 | 0.980 |
| jinaai/jina-embeddings-v5-text-small | title | 0.120 | 0.160 | 0.180 |
| jinaai/jina-embeddings-v5-text-nano | first_words | 0.620 | 0.900 | 0.980 |
| jinaai/jina-embeddings-v5-text-nano | title | 0.140 | 0.160 | 0.160 |
| Octen/Octen-Embedding-0.6B | first_words | 0.700 | 0.900 | 0.980 |
| Octen/Octen-Embedding-0.6B | title | 0.120 | 0.160 | 0.160 |

## Notas

- Las queries sintéticas se generan a partir de títulos de documentos y primeras palabras de chunks.
- Recall@k mide si el documento correcto aparece en los top-k chunks recuperados.
- MRR (Mean Reciprocal Rank) y NDCG son métricas estándar de ranking.
- Todos los modelos usan los mismos documentos y queries.

## Conclusión provisional

**Ranking por MRR:**
1. **perplexity-ai/pplx-embed-v1-0.6b**: MRR=0.512, Recall@1=0.450, Recall@5=0.610
2. **perplexity-ai/pplx-embed-context-v1-0.6b**: MRR=0.511, Recall@1=0.420, Recall@5=0.650
3. **jinaai/jina-embeddings-v5-text-small**: MRR=0.464, Recall@1=0.410, Recall@5=0.560
4. **Octen/Octen-Embedding-0.6B**: MRR=0.455, Recall@1=0.410, Recall@5=0.530
5. **jinaai/jina-embeddings-v5-text-nano**: MRR=0.443, Recall@1=0.380, Recall@5=0.530
6. **nvidia/Nemotron-3-Embed-1B-BF16**: MRR=0.359, Recall@1=0.300, Recall@5=0.440

## Siguientes pasos

- Evaluar con queries reales de usuarios.
- Probar late chunking con pplx-embed-context-v1.
- Medir latencia de búsqueda vectorial con sqlite-vec.
- Optimizar hiperparámetros (top-k, umbral de similitud).
