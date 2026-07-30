# Comparación unificada de embedding models: performance + calidad + cuantización

Fecha: 2026-07-30
Plataforma: MacBook Pro M3 (36 GB), MPS (Metal Performance Shaders)
Corpus de prueba: 100 archivos DOF (1,378 chunks) para velocidad; 50 documentos (100 queries sintéticas) para calidad.

## Tabla maestra

Ordenada por MRR (calidad de recuperación, full fp32).

| Modelo | Params | Dims | Chunks/s | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG |
|---|---|---|---|---|---|---|---|---|
| codefuse-ai/F2LLM-v2-1.7B | 1.72B | 2,048 | 1.7 | **0.500** | 0.620 | 0.640 | **0.542** | **0.566** |
| perplexity-ai/pplx-embed-v1-0.6b | 0.6B | 1,024 | 3.3 | 0.450 | 0.610 | 0.640 | 0.512 | 0.543 |
| perplexity-ai/pplx-embed-context-v1-0.6b | 0.6B | 1,024 | 3.2 | 0.420 | **0.650** | **0.700** | 0.511 | 0.557 |
| **codefuse-ai/F2LLM-v2-0.6B** | 0.6B | 1,024 | **3.7** | 0.440 | 0.590 | 0.610 | 0.500 | 0.527 |
| jinaai/jina-embeddings-v5-text-small | 0.6B | 1,024 | 2.8 | 0.410 | 0.560 | 0.580 | 0.464 | 0.492 |
| microsoft/harrier-oss-v1-0.6b | 0.6B | 1,024 | 3.6 | 0.360 | 0.590 | 0.610 | 0.464 | 0.501 |
| Octen/Octen-Embedding-0.6B | 0.6B | 1,024 | 3.6 | 0.410 | 0.530 | 0.570 | 0.455 | 0.482 |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 1,024 | 3.7 | 0.410 | 0.510 | 0.530 | 0.449 | 0.469 |
| jinaai/jina-embeddings-v5-text-nano | 0.2B | 768 | **11.3** | 0.380 | 0.530 | 0.570 | 0.443 | 0.474 |
| nvidia/Nemotron-3-Embed-1B-BF16 | 1.14B | 2,048 | 2.7 | 0.300 | 0.440 | 0.470 | 0.359 | 0.386 |

## Cuantización y dimensiones (Δ MRR vs full fp32)

Variantes post-hoc sobre los mismos embeddings: `int8` (cuantización escalar absmax por vector), `binary` (signo, 1 bit/dim), `mrl_768` (truncado Matryoshka a 768 dims).

| Modelo | int8 Δ | binary Δ | mrl_768 Δ |
|---|---|---|---|
| pplx-embed-context-v1 | +0.0 | -2.8 | **+0.3** |
| pplx-embed-v1 | +0.0 | -2.7 | -3.0 |
| Nemotron-3-Embed-1B | +0.3 | -2.0 | -0.8 |
| jina-v5-text-small | +0.0 | **+0.5** | -1.1 |
| jina-v5-text-nano | +0.0 | -2.5 | (768 nativo) |
| Octen-0.6B | +0.0 | -1.8 | -2.2 |
| F2LLM-v2-1.7B | +0.0 | -2.3 | -0.5 |
| harrier-oss-v1-0.6b | +0.0 | **-1.2** | -2.2 |
| Qwen3-Embedding-0.6B | +0.0 | -2.9 | -0.9 |
| F2LLM-v2-0.6B | +0.1 | -4.2 | +0.2 |

### Hallazgos clave

1. **int8 es gratis.** Δ MRR de +0.0 a +0.3 pts en los 10 modelos: reducción de almacenamiento 4x sin pérdida medible de calidad. Valida la ruta de sqlite-vec con int8 para producción; no hay razón para guardar fp32.

2. **Binary: solo jina-v5 está entrenado para ello.** jina-v5-text-small *mejora* +0.5 pts con binarización (32x menos bytes: 128 B/vec) — es el único modelo con soporte oficial de binary quantization y se nota. harrier (-1.2) y Octen (-1.8) degradan poco; el resto pierde 2-4 pts. F2LLM-v2-0.6B es el peor (-4.2): no binarizar los F2LLM.

3. **MRL a 768: pérdidas chicas pero innecesarias.** La mayoría pierde 0.5-3.0 pts al truncar. Curiosamente pplx-context (+0.3) y F2LLM-0.6B (+0.2) no pierden nada (ruido estadístico). Qwen3-Embedding —que sí está entrenado con MRL— pierde solo -0.9, el mejor de los >768 nativos junto con F2LLM-1.7B (-0.5). Pero dado que int8 ya da 4x gratis, truncar a 768 (25% menos) no compensa la pérdida: **mejor int8 a dims nativas**.

## Estimados para el corpus completo (~1M chunks), producción con int8

| Modelo | Tiempo embedding | Almacenamiento (int8, dims nativas) |
|---|---|---|
| jina-v5-text-nano (768d) | ~25 h | 0.75 GB |
| harrier / Qwen3 / Octen / F2LLM-0.6B (1024d) | ~75 h | 1 GB |
| pplx-v1 / context (1024d) | ~85 h | 1 GB |
| jina-v5-small (1024d) | ~99 h | 1 GB |
| Nemotron-1B (2048d) | ~103 h | 2 GB |
| F2LLM-v2-1.7B (2048d) | ~163 h | 2 GB |

## Análisis por modelo

**codefuse-ai/F2LLM-v2-0.6B (nuevo)** — La revelación del benchmark: MRR 0.500 a 0.6B/1,024 dims/3.7 chunks/s, apenas 4 pts abajo de su hermano 1.7B (0.542) con 3x menos parámetros y 2.2x más velocidad. Mejor calidad-por-tamaño de todo el benchmark. Supera a pplx-context en Recall@1 (0.440 vs 0.420) aunque no en Recall@5.

**codefuse-ai/F2LLM-v2-1.7B** — Sigue siendo el mejor en calidad absoluta (MRR 0.542, Recall@1 0.500). Si el tiempo de indexación no es crítico (se corre una vez), es el candidato a producción. Tolera bien int8 y mrl_768; mal binary.

**pplx-embed-v1 / context-v1** — Mejor balance entre los 0.6B junto con F2LLM-0.6B. context-v1 mantiene el mejor Recall@5/@10 (0.650/0.700) y su ventaja debería crecer con late chunking real (aún no activado en este eval). Bonus: tolere mrl_768 sin pérdida (+0.3).

**jina-v5-text-small** — Único modelo donde binary quantization mejora la calidad (+0.5 pts). Combinación imbatible para escala extrema: 128 B/vec → ~128 MB para todo el corpus.

**jina-v5-text-nano** — 11.3 chunks/s con solo 0.2B params. MRR 0.443 es 13% menos que pplx pero 3x más rápido. Nota: sus dims nativas ya son 768, así que "768 para todos" lo deja sin ventaja comparativa de almacenamiento.

**microsoft/harrier-oss-v1-0.6b** — Empata con jina-small en MRR siendo más rápido (3.6 vs 2.8). Segundo mejor en binary (-1.2).

**Qwen/Qwen3-Embedding-0.6B** — Decepción relativa: en MTEB/RTEB está arriba de Octen y jina-small, pero en español legal DOF queda debajo de ambos (0.449). Los benchmarks públicos dominados por inglés no transfieren directo al dominio.

**Octen-0.6B** — Media tabla en calidad (0.455) pese a ser top-15 RTEB global. Confirma lo mismo: eval local > leaderboard para este dominio.

**Nemotron-3-Embed-1B** — Eliminado: peor calidad (0.359), lento, 2,048 dims sin beneficio.

## Corrección de datos (params reales vía safetensors)

| Modelo | Params reportados antes | Params reales |
|---|---|---|
| jina-v5-text-small | ~~1.6B~~ | **0.596B** |
| jina-v5-text-nano | ~~0.6B~~ | **0.212B** |
| F2LLM-v2-0.6B | — | 0.596B |
| Nemotron-3-Embed-1B | 1B | 1.141B |

## Frontera de Pareto (velocidad vs calidad)

```
MRR
0.55 ┤ F2LLM-1.7B●
0.50 ┤      pplx-v1● pplx-ctx● F2LLM-0.6B●
0.45 ┤                  jina-sm harrier Octen Qwen3 jina-nano●
0.40 ┤
0.35 ┤                                     Nemotron-1B●
     └─────────────────────────────────────────────────
      1.7      3.0       3.7            11.3   chunks/s
```

Ganadores por prioridad:

- **Máxima calidad**: F2LLM-v2-1.7B (MRR 0.542)
- **Calidad por tamaño (0.6B)**: F2LLM-v2-0.6B (MRR 0.500, 3.7 chunks/s) — nueva recomendación
- **Balance + late chunking**: pplx-embed-context-v1 (mejor Recall@5/@10)
- **Escala extrema**: jina-v5-small binary (128 B/vec, MRR 0.469) o jina-v5-nano (11.3 chunks/s)

## Notas metodológicas

- Velocidad: 1,378 chunks, batch 32, MPS, `sentence-transformers`.
- Calidad: 100 queries sintéticas (50 títulos + 50 first_words) sobre 50 documentos; coseno.
- int8: cuantización escalar absmax por vector (la escala se cancela bajo coseno; mide solo error de redondeo). binary: sign().
- mrl_768: truncado post-hoc + renormalización; solo modelos con >768 dims nativas.
- Muestra determinística (seed 42, archivos ordenados): reproducible.

## Siguientes pasos

1. **Activar late chunking real** para pplx-embed-context-v1 y re-evaluar.
2. Decisión de producción: F2LLM-v2-1.7B (calidad) vs F2LLM-v2-0.6B (calidad/tamaño) vs pplx-context (late chunking) vs jina-small-binary (escala).
3. Medir latencia KNN de sqlite-vec con int8 a dims nativas (1,024 vs 2,048).
4. Probar Tier 1 pendiente del análisis MTEB: Qwen3-Embedding-4B, Octen-4B, dinghy-law-0.6b.
