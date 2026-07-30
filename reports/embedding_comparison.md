# Comparación: modelos de embedding para DOF RAG

Muestra: **100** archivos markdown de `./dof_md`
Fecha: 2026-07-30

## Resumen general

| Modelo | Device | Dim | Chunks | Tiempo (s) | Chunks/s | Memoria pico (MB) |
|---|---|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | mps | 1,024 | 1,378 | 424.82 | 3.2 | 397 |
| perplexity-ai/pplx-embed-v1-0.6b | mps | 1,024 | 1,378 | 421.22 | 3.3 | 54 |
| nvidia/Nemotron-3-Embed-1B-BF16 | mps | 2,048 | 1,378 | 517.85 | 2.7 | 102 |
| jinaai/jina-embeddings-v5-text-small | mps | 1,024 | 1,378 | 486.60 | 2.8 | 232 |
| jinaai/jina-embeddings-v5-text-nano | mps | 768 | 1,378 | 122.03 | 11.3 | 270 |
| Octen/Octen-Embedding-0.6B | mps | 1,024 | 1,378 | 382.96 | 3.6 | 0 |
| codefuse-ai/F2LLM-v2-1.7B | mps | 2,048 | 1,378 | 792.90 | 1.7 | 104 |
| microsoft/harrier-oss-v1-0.6b | mps | 1,024 | 1,378 | 377.93 | 3.6 | 102 |
| Qwen/Qwen3-Embedding-0.6B | mps | 1,024 | 1,378 | 372.82 | 3.7 | 15 |
| codefuse-ai/F2LLM-v2-0.6B | mps | 1,024 | 1,378 | 373.87 | 3.7 | 103 |

## Notas

- Todos los modelos usan el mismo tokenizer y los mismos chunks de entrada.
- La memoria pico es aproximada (RSS del proceso).
- En MacBook Pro M3, `mps` usa Metal Performance Shaders; `cpu` es fallback.
- `pplx-embed-context-v1` es contextual: los embeddings de chunks del mismo documento se benefician de verse juntos.

## Conclusión provisional

- **perplexity-ai/pplx-embed-context-v1-0.6b**: 1,024 dims, 3.2 chunks/s, 397 MB pico.
- **perplexity-ai/pplx-embed-v1-0.6b**: 1,024 dims, 3.3 chunks/s, 54 MB pico.
- **nvidia/Nemotron-3-Embed-1B-BF16**: 2,048 dims, 2.7 chunks/s, 102 MB pico.
- **jinaai/jina-embeddings-v5-text-small**: 1,024 dims, 2.8 chunks/s, 232 MB pico.
- **jinaai/jina-embeddings-v5-text-nano**: 768 dims, 11.3 chunks/s, 270 MB pico.
- **Octen/Octen-Embedding-0.6B**: 1,024 dims, 3.6 chunks/s, 0 MB pico.
- **codefuse-ai/F2LLM-v2-1.7B**: 2,048 dims, 1.7 chunks/s, 104 MB pico.
- **microsoft/harrier-oss-v1-0.6b**: 1,024 dims, 3.6 chunks/s, 102 MB pico.
- **Qwen/Qwen3-Embedding-0.6B**: 1,024 dims, 3.7 chunks/s, 15 MB pico.
- **codefuse-ai/F2LLM-v2-0.6B**: 1,024 dims, 3.7 chunks/s, 103 MB pico.

## Siguientes pasos

- Evaluar calidad de recuperación con queries sintéticas del DOF.
- Probar quantización (int8, int4) para reducir memoria en modelos grandes.
- Medir latencia de búsqueda vectorial con sqlite-vec.
