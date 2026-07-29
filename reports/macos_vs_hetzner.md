# Hetzner vs MacBook Pro M3: comparación de rendimiento en benchmarks

Fecha: 2026-07-29

Comparación de los benchmarks de chunking y embeddings entre:

- **Hetzner**: servidor dedicado, AMD Ryzen 7 PRO 8700GE, 64 GB RAM, Linux x86_64, CPU-only.
- **Mac**: MacBook Pro M3, 36 GB RAM unificada, macOS arm64, MPS (Metal Performance Shaders) para embeddings.

Corpus: 131,830 archivos markdown del DOF (2020–2026), idéntico en ambas máquinas.

## Chunking (CPU puro, 1,000 archivos)

| Chunker | Hetzner (chunks/s) | Mac (chunks/s) | Ratio Mac/Hetzner |
|---|---|---|---|
| Custom | 298 | 303 | 1.02x |
| Chonkie Recursive | 352 | 387 | 1.10x |
| Chonkie H2 | 278 | 279 | 1.00x |
| Chonkie Table | 760 | 1,384 | 1.82x* |
| Chonkie Token | 595 | 664 | 1.12x |
| Chonkie Sentence | 345 | 374 | 1.08x |
| Chonkie Pipeline | 3,695 | 7,109 | 1.92x* |
| Chonkie Pipeline Rev | 409 | 467 | 1.14x |

\* Las muestras originales de Hetzner se eligieron con orden de filesystem distinto; los ratios de Table/Pipeline están sesgados por la composición de la muestra (más tablas en la muestra de Mac). Para el resto de chunkers la diferencia de muestra es pequeña.

**Conclusión chunking**: el M3 rinde a la par o ligeramente mejor que el Ryzen 7 PRO 8700GE para chunking CPU-bound (tokenización + regex). El cuello de botella no es la CPU sino la tokenización HF (Rust, nativa en ambas plataformas).

## Embeddings: Hetzner CPU vs Mac MPS (1,378 chunks)

| Modelo | Hetzner CPU (chunks/s) | Mac MPS (chunks/s) | Speedup |
|---|---|---|---|
| pplx-embed-context-v1-0.6b | 0.6 | 2.9 | **4.8x** |
| pplx-embed-v1-0.6b | 0.6 | 3.3 | **5.5x** |
| Nemotron-3-Embed-1B-BF16 | 0.7 | 2.6 | **3.7x** |
| jina-embeddings-v5-text-small | 0.6 | 2.8 | **4.7x** |
| jina-embeddings-v5-text-nano | 1.9 | 11.2 | **5.9x** |
| Octen-Embedding-0.6B | 1.0 | 3.5 | **3.5x** |

**Conclusión embeddings**: MPS en el M3 acelera la generación de embeddings ~4-6x sobre el CPU de Hetzner. `jina-v5-text-nano` es el más rápido en ambas plataformas (11.2 chunks/s en Mac). Con este rendimiento, embedder el corpus completo estimado (~1M chunks) tomaría:

| Modelo | Tiempo estimado en Mac (MPS) | Tiempo estimado en Hetzner (CPU) |
|---|---|---|
| jina-v5-text-nano | ~25 horas | ~6 días |
| Octen-0.6B | ~3.3 días | ~11.6 días |
| pplx-embed-v1 | ~3.5 días | ~19 días |

## Calidad de recuperación (ranking consistente en ambas máquinas)

| Modelo | MRR (Hetzner) | MRR (Mac) |
|---|---|---|
| pplx-embed-v1 | 0.598 | 0.512 |
| pplx-embed-context-v1 | 0.590 | 0.511 |
| jina-v5-text-small | 0.537 | 0.464 |
| Octen-0.6B | 0.537 | 0.455 |
| jina-v5-text-nano | 0.519 | 0.443 |
| Nemotron-3-Embed-1B | 0.434 | 0.359 |

Los valores absolutos difieren porque la muestra de 50 documentos se eligió con orden distinto en cada máquina; **el ranking de modelos es idéntico**, lo que valida las conclusiones de calidad.

## Fixes necesarios para correr en macOS

1. **Dependencias actualizadas** (Hetzner las tenía ad-hoc; el lockfile estaba viejo):
   - `sentence-transformers>=5.6` — requerido por el código custom de pplx-embed (`Module` no existe en ST 3.4.1).
   - `transformers>=5.9` — requerido para la arquitectura `ministral3` de Nemotron-3-Embed.
2. **Sampling determinístico**: los scripts ahora ordenan la lista de archivos antes de `random.sample`, así la misma semilla elige los mismos archivos en cualquier filesystem.
3. **File discovery con `os.walk(followlinks=True)`**: `pathlib.Path.rglob` no sigue symlinks de directorio en Python 3.12; necesario porque `dof_md` es un symlink.
4. **Memoria en MPS**: `psutil` RSS sub-reporta la memoria real en Metal (buffers unificados no aparecen en RSS); los MB reportados para embeddings en Mac son una cota inferior.

## Recomendación

- **Generación masiva de embeddings**: correr en la Mac con MPS; es ~5x más rápida que Hetzner CPU y no tiene costo de transferencia. `jina-v5-text-nano` es el candidato de throughput; `pplx-embed-v1`/`context-v1` los de calidad.
- **Chunking**: indistinto; ambas máquinas rinden igual.
- **Servidor de búsqueda**: Hetzner sigue siendo el lugar correcto para servir (sqlite-vec + FTS5 son CPU/livianos), pero la generación inicial de embeddings conviene hacerla en la Mac y subir la base de datos.
