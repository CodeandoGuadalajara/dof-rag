# Comparación unificada de embedding models: performance + calidad

Fecha: 2026-07-29
Plataforma: MacBook Pro M3 (36 GB), MPS (Metal Performance Shaders)
Corpus de prueba: 100 archivos DOF (1,378 chunks) para velocidad; 50 documentos (100 queries sintéticas) para calidad.

## Tabla maestra

Ordenada por MRR (calidad de recuperación).

| Modelo | Params | Dims | Chunks/s | Mem (MB) | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG |
|---|---|---|---|---|---|---|---|---|---|
| **codefuse-ai/F2LLM-v2-1.7B** | 1.7B | 2,048 | 1.7 | 21 | **0.500** | 0.620 | 0.640 | **0.542** | **0.566** |
| **perplexity-ai/pplx-embed-v1-0.6b** | 0.6B | 1,024 | 3.3 | 57 | 0.450 | 0.610 | 0.640 | 0.512 | 0.543 |
| **perplexity-ai/pplx-embed-context-v1-0.6b** | 0.6B | 1,024 | 3.2 | 393 | 0.420 | **0.650** | **0.700** | 0.511 | 0.557 |
| jinaai/jina-embeddings-v5-text-small | 1.6B | 1,024 | 2.8 | 276 | 0.410 | 0.560 | 0.580 | 0.464 | 0.492 |
| microsoft/harrier-oss-v1-0.6b | 0.6B | 1,024 | **3.7** | 90 | 0.360 | 0.590 | 0.610 | 0.464 | 0.501 |
| Octen/Octen-Embedding-0.6B | 0.6B | 1,024 | 3.6 | 0 | 0.410 | 0.530 | 0.570 | 0.455 | 0.482 |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 1,024 | **3.7** | 16 | 0.410 | 0.510 | 0.530 | 0.449 | 0.469 |
| jinaai/jina-embeddings-v5-text-nano | 0.6B | 768 | **11.2** | 237 | 0.380 | 0.530 | 0.570 | 0.443 | 0.474 |
| nvidia/Nemotron-3-Embed-1B-BF16 | 1B | 2,048 | 2.7 | 97 | 0.300 | 0.440 | 0.470 | 0.359 | 0.386 |

## Frontera de Pareto (velocidad vs calidad)

```
MRR
0.55 ┤  F2LLM-1.7B ●
0.50 ┤        pplx-v1 ●  pplx-ctx ●
0.45 ┤                     jina-sm harrier Octen Qwen3 jina-nano
0.40 ┤
0.35 ┤                                          Nemotron-1B
     └──────────────────────────────────────────────────────
      1.7        3.0        3.7              11.2   chunks/s
```

Tres ganadores según prioridad:

- **Máxima calidad**: F2LLM-v2-1.7B (MRR 0.542, Recall@1 0.500) — pero el más lento (1.7 chunks/s) y el doble de almacenamiento (2,048 dims).
- **Balance calidad/velocidad**: pplx-embed-v1 (MRR 0.512, 3.3 chunks/s). pplx-context-v1 es casi idéntico pero con mejor Recall@5/@10 (0.650/0.700) y habilita late chunking contextual.
- **Máxima velocidad**: jina-v5-text-nano (11.2 chunks/s, MRR 0.443) — 3x más rápido que cualquier otro, con solo 13% menos MRR que pplx.

## Estimados para el corpus completo (~1M chunks)

| Modelo | Tiempo de embedding | Almacenamiento (fp32) | Almacenamiento (int8) |
|---|---|---|---|
| jina-v5-text-nano | ~25 h | 3 GB | 0.8 GB |
| harrier-oss-v1-0.6b | ~75 h | 4 GB | 1 GB |
| Qwen3-Embedding-0.6B | ~75 h | 4 GB | 1 GB |
| Octen-Embedding-0.6B | ~77 h | 4 GB | 1 GB |
| pplx-embed-v1 / context-v1 | ~85 h | 4 GB | 1 GB (nativo int8) |
| jina-v5-text-small | ~99 h | 4 GB | 1 GB |
| Nemotron-3-Embed-1B | ~103 h | 8 GB | 2 GB |
| F2LLM-v2-1.7B | ~163 h | 8 GB | 2 GB |

## Análisis de los nuevos modelos

**codefuse-ai/F2LLM-v2-1.7B** — El ganador de calidad. Recall@1 de 0.500: la mitad de las queries encuentran el documento correcto en primer lugar. Federated/Fineweb-trained (F2LLM v2), 2,048 dims. Si el tiempo de indexación no es crítico (se corre una sola vez), es el candidato a producción.

**microsoft/harrier-oss-v1-0.6b** — Debut sólido de Microsoft en embeddings open. Empata con jina-v5-small en MRR (0.464) siendo 3x más chico y 32% más rápido (3.7 vs 2.8). Buen Recall@5 (0.590).

**Qwen/Qwen3-Embedding-0.6B** — Sorpresa negativa. En el leaderboard MTEB/RTEB está por encima de Octen-0.6B y jina-v5-small, pero en nuestro corpus de español legal queda debajo de ambos (MRR 0.449 vs 0.455/0.464). Confirma que los benchmarks públicos (dominados por inglés) no transfieren directamente al dominio DOF; el eval local es indispensable.

**pplx-embed-context-v1** — Segundo en Recall@5/@10 (0.650/0.700, los más altos). Su ventaja debería crecer con late chunking real (embeddings derivados del contexto completo del documento), que aún no está activado en este eval — aquí se evaluó como embedder estándar chunk-por-chunk.

**Nemotron-3-Embed-1B** — Queda eliminado: peor calidad (MRR 0.359), lento (2.7), y el doble de dims (2,048) sin beneficio.

## Calidad por tipo de query (los 3 nuevos)

| Modelo | Query tipo | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| F2LLM-v2-1.7B | first_words | 0.76 | 0.94 | 0.96 |
| F2LLM-v2-1.7B | title | 0.24 | 0.30 | 0.32 |
| harrier-oss-v1-0.6b | first_words | 0.58 | 0.92 | 0.96 |
| harrier-oss-v1-0.6b | title | 0.14 | 0.26 | 0.26 |
| Qwen3-Embedding-0.6B | first_words | 0.68 | 0.86 | 0.90 |
| Qwen3-Embedding-0.6B | title | 0.14 | 0.16 | 0.16 |

Todos los modelos resuelven bien las queries de contenido (first_words: 0.58-0.78 Recall@1) y batallan con queries de solo-título (0.06-0.24), lo que sugiere que el título solo no es buena proxy de query; en producción las queries de usuarios serán de lenguaje natural tipo first_words.

## Notas metodológicas

- Velocidad: 1,378 chunks, batch 32, MPS, medido con `sentence-transformers`.
- Calidad: 100 queries sintéticas (50 títulos + 50 first_words) sobre 50 documentos; métricas estándar (Recall@k, MRR, NDCG) con similitud coseno.
- La memoria RSS sub-reporta en MPS (buffers Metal unificados no aparecen completos); valores son cota inferior.
- Muestra determinística (seed 42, archivos ordenados): reproducible en cualquier máquina.

## Siguientes pasos

1. **Activar late chunking real** para pplx-embed-context-v1 (su ventaja natural) y re-evaluar.
2. Probar Tier 1 pendiente: Qwen3-Embedding-4B, Octen-Embedding-4B (best-in-class RTEB ≤4B), dinghy-law-0.6b.
3. Evaluar F2LLM-v2-1.7B con int8 para reducir almacenamiento a la mitad.
4. Medir latencia de búsqueda sqlite-vec por modelo (2,048 dims vs 1,024 vs 768 afecta KNN).
