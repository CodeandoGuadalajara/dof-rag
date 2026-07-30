# Comparación: modelos de embedding para DOF RAG

Muestra: **100** archivos markdown de `./dof_md`
Fecha: 2026-07-29

## Resumen general

| Modelo | Device | Dim | Chunks | Tiempo (s) | Chunks/s | Memoria pico (MB) |
|---|---|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | mps | 1,024 | 1,378 | 435.03 | 3.2 | 393 |
| perplexity-ai/pplx-embed-v1-0.6b | mps | 1,024 | 1,378 | 423.01 | 3.3 | 57 |
| nvidia/Nemotron-3-Embed-1B-BF16 | mps | 2,048 | 1,378 | 519.65 | 2.7 | 97 |
| jinaai/jina-embeddings-v5-text-small | mps | 1,024 | 1,378 | 494.85 | 2.8 | 276 |
| jinaai/jina-embeddings-v5-text-nano | mps | 768 | 1,378 | 122.53 | 11.2 | 237 |
| Octen/Octen-Embedding-0.6B | mps | 1,024 | 1,378 | 386.88 | 3.6 | 0 |
| codefuse-ai/F2LLM-v2-1.7B | mps | 2,048 | 1,378 | 834.78 | 1.7 | 21 |
| microsoft/harrier-oss-v1-0.6b | mps | 1,024 | 1,378 | 377.49 | 3.7 | 90 |
| Qwen/Qwen3-Embedding-0.6B | mps | 1,024 | 1,378 | 374.51 | 3.7 | 16 |

## Notas

- Todos los modelos usan el mismo tokenizer y los mismos chunks de entrada.
- La memoria pico es aproximada (RSS del proceso).
- En MacBook Pro M3, `mps` usa Metal Performance Shaders; `cpu` es fallback.
- `pplx-embed-context-v1` es contextual: los embeddings de chunks del mismo documento se benefician de verse juntos.

## Conclusión provisional

- **perplexity-ai/pplx-embed-context-v1-0.6b**: 1,024 dims, 3.2 chunks/s, 393 MB pico.
- **perplexity-ai/pplx-embed-v1-0.6b**: 1,024 dims, 3.3 chunks/s, 57 MB pico.
- **nvidia/Nemotron-3-Embed-1B-BF16**: 2,048 dims, 2.7 chunks/s, 97 MB pico.
- **jinaai/jina-embeddings-v5-text-small**: 1,024 dims, 2.8 chunks/s, 276 MB pico.
- **jinaai/jina-embeddings-v5-text-nano**: 768 dims, 11.2 chunks/s, 237 MB pico.
- **Octen/Octen-Embedding-0.6B**: 1,024 dims, 3.6 chunks/s, 0 MB pico.
- **codefuse-ai/F2LLM-v2-1.7B**: 2,048 dims, 1.7 chunks/s, 21 MB pico.
- **microsoft/harrier-oss-v1-0.6b**: 1,024 dims, 3.7 chunks/s, 90 MB pico.
- **Qwen/Qwen3-Embedding-0.6B**: 1,024 dims, 3.7 chunks/s, 16 MB pico.

## Siguientes pasos

- Evaluar calidad de recuperación con queries sintéticas del DOF.
- Probar quantización (int8, int4) para reducir memoria en modelos grandes.
- Medir latencia de búsqueda vectorial con sqlite-vec.
