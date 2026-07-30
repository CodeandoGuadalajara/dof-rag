# Evaluación de calidad de recuperación: modelos de embedding

Muestra: **50** documentos markdown de `./dof_md`
Fecha: 2026-07-30

## Tabla maestra: modelo × variante

Variantes post-hoc sobre los mismos embeddings fp32: `full_fp32` (baseline), `mrl_768` (truncado Matryoshka a 768 dims), `int8` (cuantización escalar), `binary` (signo, 1 bit/dim).

| Modelo | Variante | Dims ef. | Bytes/vec | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG |
|---|---|---|---|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | full_fp32 | 1024 | 4096 | 0.420 | 0.650 | 0.700 | 0.511 | 0.557 |
| perplexity-ai/pplx-embed-context-v1-0.6b | mrl_768 | 768 | 3072 | 0.440 | 0.620 | 0.670 | 0.514 | 0.552 |
| perplexity-ai/pplx-embed-context-v1-0.6b | int8 | 1024 | 1024 | 0.420 | 0.650 | 0.700 | 0.511 | 0.557 |
| perplexity-ai/pplx-embed-context-v1-0.6b | binary | 1024 | 128 | 0.390 | 0.600 | 0.660 | 0.483 | 0.526 |
| perplexity-ai/pplx-embed-v1-0.6b | full_fp32 | 1024 | 4096 | 0.450 | 0.610 | 0.640 | 0.512 | 0.543 |
| perplexity-ai/pplx-embed-v1-0.6b | mrl_768 | 768 | 3072 | 0.420 | 0.570 | 0.600 | 0.482 | 0.511 |
| perplexity-ai/pplx-embed-v1-0.6b | int8 | 1024 | 1024 | 0.450 | 0.610 | 0.640 | 0.512 | 0.543 |
| perplexity-ai/pplx-embed-v1-0.6b | binary | 1024 | 128 | 0.420 | 0.570 | 0.640 | 0.485 | 0.522 |
| nvidia/Nemotron-3-Embed-1B-BF16 | full_fp32 | 2048 | 8192 | 0.300 | 0.440 | 0.470 | 0.359 | 0.386 |
| nvidia/Nemotron-3-Embed-1B-BF16 | mrl_768 | 768 | 3072 | 0.290 | 0.450 | 0.460 | 0.351 | 0.378 |
| nvidia/Nemotron-3-Embed-1B-BF16 | int8 | 2048 | 2048 | 0.310 | 0.440 | 0.470 | 0.363 | 0.389 |
| nvidia/Nemotron-3-Embed-1B-BF16 | binary | 2048 | 256 | 0.280 | 0.440 | 0.470 | 0.340 | 0.371 |
| jinaai/jina-embeddings-v5-text-small | full_fp32 | 1024 | 4096 | 0.410 | 0.560 | 0.580 | 0.464 | 0.492 |
| jinaai/jina-embeddings-v5-text-small | mrl_768 | 768 | 3072 | 0.400 | 0.550 | 0.570 | 0.454 | 0.482 |
| jinaai/jina-embeddings-v5-text-small | int8 | 1024 | 1024 | 0.410 | 0.560 | 0.580 | 0.464 | 0.492 |
| jinaai/jina-embeddings-v5-text-small | binary | 1024 | 128 | 0.420 | 0.540 | 0.580 | 0.469 | 0.496 |
| jinaai/jina-embeddings-v5-text-nano | full_fp32 | 768 | 3072 | 0.380 | 0.530 | 0.570 | 0.443 | 0.474 |
| jinaai/jina-embeddings-v5-text-nano | int8 | 768 | 768 | 0.380 | 0.530 | 0.570 | 0.443 | 0.474 |
| jinaai/jina-embeddings-v5-text-nano | binary | 768 | 96 | 0.350 | 0.490 | 0.550 | 0.419 | 0.450 |
| Octen/Octen-Embedding-0.6B | full_fp32 | 1024 | 4096 | 0.410 | 0.530 | 0.570 | 0.455 | 0.482 |
| Octen/Octen-Embedding-0.6B | mrl_768 | 768 | 3072 | 0.380 | 0.520 | 0.550 | 0.433 | 0.462 |
| Octen/Octen-Embedding-0.6B | int8 | 1024 | 1024 | 0.410 | 0.530 | 0.570 | 0.455 | 0.482 |
| Octen/Octen-Embedding-0.6B | binary | 1024 | 128 | 0.390 | 0.520 | 0.530 | 0.437 | 0.460 |
| codefuse-ai/F2LLM-v2-1.7B | full_fp32 | 2048 | 8192 | 0.500 | 0.620 | 0.640 | 0.542 | 0.566 |
| codefuse-ai/F2LLM-v2-1.7B | mrl_768 | 768 | 3072 | 0.500 | 0.600 | 0.600 | 0.537 | 0.552 |
| codefuse-ai/F2LLM-v2-1.7B | int8 | 2048 | 2048 | 0.500 | 0.620 | 0.640 | 0.542 | 0.566 |
| codefuse-ai/F2LLM-v2-1.7B | binary | 2048 | 256 | 0.470 | 0.600 | 0.620 | 0.519 | 0.543 |
| microsoft/harrier-oss-v1-0.6b | full_fp32 | 1024 | 4096 | 0.360 | 0.590 | 0.610 | 0.464 | 0.501 |
| microsoft/harrier-oss-v1-0.6b | mrl_768 | 768 | 3072 | 0.320 | 0.600 | 0.620 | 0.442 | 0.487 |
| microsoft/harrier-oss-v1-0.6b | int8 | 1024 | 1024 | 0.360 | 0.590 | 0.610 | 0.464 | 0.501 |
| microsoft/harrier-oss-v1-0.6b | binary | 1024 | 128 | 0.340 | 0.610 | 0.630 | 0.452 | 0.496 |
| Qwen/Qwen3-Embedding-0.6B | full_fp32 | 1024 | 4096 | 0.410 | 0.510 | 0.530 | 0.449 | 0.469 |
| Qwen/Qwen3-Embedding-0.6B | mrl_768 | 768 | 3072 | 0.400 | 0.510 | 0.530 | 0.440 | 0.462 |
| Qwen/Qwen3-Embedding-0.6B | int8 | 1024 | 1024 | 0.410 | 0.510 | 0.530 | 0.449 | 0.469 |
| Qwen/Qwen3-Embedding-0.6B | binary | 1024 | 128 | 0.360 | 0.500 | 0.530 | 0.420 | 0.446 |
| codefuse-ai/F2LLM-v2-0.6B | full_fp32 | 1024 | 4096 | 0.440 | 0.590 | 0.610 | 0.500 | 0.527 |
| codefuse-ai/F2LLM-v2-0.6B | mrl_768 | 768 | 3072 | 0.450 | 0.590 | 0.610 | 0.502 | 0.528 |
| codefuse-ai/F2LLM-v2-0.6B | int8 | 1024 | 1024 | 0.440 | 0.590 | 0.610 | 0.500 | 0.527 |
| codefuse-ai/F2LLM-v2-0.6B | binary | 1024 | 128 | 0.400 | 0.540 | 0.580 | 0.458 | 0.488 |

## Ranking por MRR (full fp32)

1. **codefuse-ai/F2LLM-v2-1.7B**: MRR=0.542, Recall@1=0.500, Recall@5=0.620
2. **perplexity-ai/pplx-embed-v1-0.6b**: MRR=0.512, Recall@1=0.450, Recall@5=0.610
3. **perplexity-ai/pplx-embed-context-v1-0.6b**: MRR=0.511, Recall@1=0.420, Recall@5=0.650
4. **codefuse-ai/F2LLM-v2-0.6B**: MRR=0.500, Recall@1=0.440, Recall@5=0.590
5. **jinaai/jina-embeddings-v5-text-small**: MRR=0.464, Recall@1=0.410, Recall@5=0.560
6. **microsoft/harrier-oss-v1-0.6b**: MRR=0.464, Recall@1=0.360, Recall@5=0.590
7. **Octen/Octen-Embedding-0.6B**: MRR=0.455, Recall@1=0.410, Recall@5=0.530
8. **Qwen/Qwen3-Embedding-0.6B**: MRR=0.449, Recall@1=0.410, Recall@5=0.510
9. **jinaai/jina-embeddings-v5-text-nano**: MRR=0.443, Recall@1=0.380, Recall@5=0.530
10. **nvidia/Nemotron-3-Embed-1B-BF16**: MRR=0.359, Recall@1=0.300, Recall@5=0.440

## Impacto de la cuantización (Δ MRR vs full fp32)

| Modelo | int8 Δ | binary Δ | mrl_768 Δ |
|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | +0.0 pts | -2.8 pts | +0.3 pts |
| perplexity-ai/pplx-embed-v1-0.6b | +0.0 pts | -2.7 pts | -3.0 pts |
| nvidia/Nemotron-3-Embed-1B-BF16 | +0.3 pts | -2.0 pts | -0.8 pts |
| jinaai/jina-embeddings-v5-text-small | -0.0 pts | +0.5 pts | -1.1 pts |
| jinaai/jina-embeddings-v5-text-nano | +0.0 pts | -2.5 pts | - |
| Octen/Octen-Embedding-0.6B | +0.0 pts | -1.8 pts | -2.2 pts |
| codefuse-ai/F2LLM-v2-1.7B | +0.0 pts | -2.3 pts | -0.5 pts |
| microsoft/harrier-oss-v1-0.6b | +0.0 pts | -1.2 pts | -2.2 pts |
| Qwen/Qwen3-Embedding-0.6B | +0.0 pts | -2.9 pts | -0.9 pts |
| codefuse-ai/F2LLM-v2-0.6B | +0.1 pts | -4.2 pts | +0.2 pts |

## Notas

- Las queries sintéticas se generan a partir de títulos de documentos y primeras palabras de chunks.
- Recall@k mide si el documento correcto aparece en los top-k chunks recuperados.
- La cuantización int8 es escalar por-vector (absmax); la binaria es sign().
- `mrl_768` solo aplica a modelos con más de 768 dims nativas; jina-v5-text-nano (768 nativas) no tiene variante mrl.
- Muestra determinística (seed 42, archivos ordenados): reproducible en cualquier máquina.
