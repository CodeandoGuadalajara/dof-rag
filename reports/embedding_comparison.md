# Comparación: modelos de embedding para DOF RAG

Muestra: **100** archivos markdown de `./dof_md`
Fecha: 2026-07-27

## Resumen general

| Modelo | Device | Dim | Chunks | Tiempo (s) | Chunks/s | Memoria pico (MB) |
|---|---|---|---|---|---|---|
| perplexity-ai/pplx-embed-context-v1-0.6b | cpu | 1,024 | 773 | 1394.49 | 0.6 | 2188 |
| perplexity-ai/pplx-embed-v1-0.6b | cpu | 1,024 | 773 | 1399.11 | 0.6 | 1862 |
| nvidia/Nemotron-3-Embed-1B-BF16 | cpu | 2,048 | 773 | 1069.47 | 0.7 | 2078 |
| jinaai/jina-embeddings-v5-text-small | cpu | 1,024 | 773 | 1193.92 | 0.6 | 1176 |
| jinaai/jina-embeddings-v5-text-nano | cpu | 768 | 773 | 411.52 | 1.9 | 810 |
| Octen/Octen-Embedding-0.6B | cpu | 1,024 | 773 | 737.80 | 1.0 | 1021 |

## Notas

- Todos los modelos usan el mismo tokenizer y los mismos chunks de entrada.
- La memoria pico es aproximada (RSS del proceso).
- En MacBook Pro M3, `mps` usa Metal Performance Shaders; `cpu` es fallback.
- `pplx-embed-context-v1` es contextual: los embeddings de chunks del mismo documento se benefician de verse juntos.

## Conclusión provisional

- **perplexity-ai/pplx-embed-context-v1-0.6b**: 1,024 dims, 0.6 chunks/s, 2188 MB pico.
- **perplexity-ai/pplx-embed-v1-0.6b**: 1,024 dims, 0.6 chunks/s, 1862 MB pico.
- **nvidia/Nemotron-3-Embed-1B-BF16**: 2,048 dims, 0.7 chunks/s, 2078 MB pico.
- **jinaai/jina-embeddings-v5-text-small**: 1,024 dims, 0.6 chunks/s, 1176 MB pico.
- **jinaai/jina-embeddings-v5-text-nano**: 768 dims, 1.9 chunks/s, 810 MB pico.
- **Octen/Octen-Embedding-0.6B**: 1,024 dims, 1.0 chunks/s, 1021 MB pico.

## Siguientes pasos

- Evaluar calidad de recuperación con queries sintéticas del DOF.
- Probar quantización (int8, int4) para reducir memoria en modelos grandes.
- Medir latencia de búsqueda vectorial con sqlite-vec.
