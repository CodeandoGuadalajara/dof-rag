# Comparación: modelos de embedding para DOF RAG

Muestra: **100** archivos markdown de `./dof_md`
Fecha: 2026-07-29

## Resumen general

| Modelo | Device | Dim | Chunks | Tiempo (s) | Chunks/s | Memoria pico (MB) |
|---|---|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | mps | 1,024 | 1,378 | 477.18 | 2.9 | 1776 |
| perplexity-ai/pplx-embed-v1-0.6b | mps | 1,024 | 1,378 | 423.10 | 3.3 | 55 |
| nvidia/Nemotron-3-Embed-1B-BF16 | mps | 2,048 | 1,378 | 522.25 | 2.6 | 0 |
| jinaai/jina-embeddings-v5-text-small | mps | 1,024 | 1,378 | 495.28 | 2.8 | 244 |
| jinaai/jina-embeddings-v5-text-nano | mps | 768 | 1,378 | 122.79 | 11.2 | 452 |
| Octen/Octen-Embedding-0.6B | mps | 1,024 | 1,378 | 391.11 | 3.5 | 0 |

## Notas

- Todos los modelos usan el mismo tokenizer y los mismos chunks de entrada.
- La memoria pico es aproximada (RSS del proceso).
- En MacBook Pro M3, `mps` usa Metal Performance Shaders; `cpu` es fallback.
- `pplx-embed-context-v1` es contextual: los embeddings de chunks del mismo documento se benefician de verse juntos.

## Conclusión provisional

- **perplexity-ai/pplx-embed-context-v1-0.6b**: 1,024 dims, 2.9 chunks/s, 1776 MB pico.
- **perplexity-ai/pplx-embed-v1-0.6b**: 1,024 dims, 3.3 chunks/s, 55 MB pico.
- **nvidia/Nemotron-3-Embed-1B-BF16**: 2,048 dims, 2.6 chunks/s, 0 MB pico.
- **jinaai/jina-embeddings-v5-text-small**: 1,024 dims, 2.8 chunks/s, 244 MB pico.
- **jinaai/jina-embeddings-v5-text-nano**: 768 dims, 11.2 chunks/s, 452 MB pico.
- **Octen/Octen-Embedding-0.6B**: 1,024 dims, 3.5 chunks/s, 0 MB pico.

## Siguientes pasos

- Evaluar calidad de recuperación con queries sintéticas del DOF.
- Probar quantización (int8, int4) para reducir memoria en modelos grandes.
- Medir latencia de búsqueda vectorial con sqlite-vec.
