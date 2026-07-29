# Candidatos de embedding models del MTEB leaderboard para DOF RAG

Fecha: 2026-07-29

Análisis del [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) usando el dataset `mteb/results` (8.5M filas de scores) para encontrar modelos que valga la pena agregar a nuestro benchmark, priorizando **RTEB(beta) multilingual** y **MTEB(Law, v1)** / **RTEB(Law, beta)**.

Restricciones: correr local en MacBook Pro M3 (36 GB), español legal/gubernamental (DOF), sentence-transformers compatible.

## Posición de los modelos que ya probamos

| Modelo | RTEB(beta) | RTEB(Law, beta) | Comentario |
|---|---|---|---|
| Octen-Embedding-0.6B | **74.94** (14 tasks) | **71.10** | Ya es top-15 global en RTEB |
| Nemotron-3-Embed-1B | 73.66 (14) | 61.46 | Sólido en general, flojo en ley |
| jina-v5-text-small | 67.80 (14) | — | Media tabla |
| jina-v5-text-nano | 64.77 (14) | — | El precio de la velocidad |
| pplx-embed-v1 / context-v1 | **no está en el leaderboard** | — | Nuestro eval custom es el único punto de comparación |

Nota: RTEB(beta) tiene 30 tasks; ningún modelo tiene cobertura completa (máx ~17). Los scores mostrados usan cobertura de 14 tasks salvo indicación.

## Top del leaderboard (lo que nos falta probar)

### RTEB(beta) multilingual

| Score | Modelo | Params | ¿Local en M3? |
|---|---|---|---|
| 81.89 | Octen/Octen-Embedding-8B | 8B | Ajustado (~16 GB bf16) |
| 79.87 | nvidia/Nemotron-3-Embed-8B-BF16 | 7.95B | Ajustado |
| **79.27** | **Octen/Octen-Embedding-4B** | 4.02B | ✅ Cómodo |
| 78.65 | bflhc/MoD-Embedding | 4.02B | ✅ pero solo 217 descargas (nuevo, riesgo) |
| 74.15 | Qwen/Qwen3-Embedding-8B | 7.57B | Ajustado |

Qwen/Qwen3-Embedding-4B y -0.6B solo tienen 2 tasks de cobertura en RTEB (85.19 y 77.18, no comparable directo), pero la familia Qwen3-Embedding es la más descargada del leaderboard (10.5M el 0.6B) y su variante 8B marca 74.15 con cobertura completa.

### MTEB(Law, v1)

| Score | Modelo | Params | Nota |
|---|---|---|---|
| 70.37 | Mira190/Euler-Legal-Embedding-V1 | 8.19B | Especializado en ley (inglés/chino/alemán) |
| 69.33 | minetta/nemotron-3-embed-8b-legal | 7.95B | Fine-tune legal de Nemotron |
| **65.83** | **Hanno-Labs/dinghy-law-0.6b-v1** | **0.6B** | ✅ Especializado en ley y diminuto |
| 65.39 | voyage-law-2 | API | No corre local |
| 63.68 | infly/inf-retriever-v1 | 7B | — |
| 57.86 | codefuse-ai/F2LLM-v2-0.6B | 0.6B | — |

### RTEB(Law, beta)

| Score | Modelo | Params |
|---|---|---|
| 77.58 | Octen-Embedding-8B | 8B |
| 74.31 | bflhc/MoD-Embedding | 4B |
| **73.91** | **Octen-Embedding-4B** | **4B** |
| 71.10 | Octen-Embedding-0.6B | 0.6B (ya probado) |
| 67.25 | Qwen3-Embedding-8B | 8B |
| 66.65 | Euler-Legal-Embedding-V1 | 8.19B |
| 59.79 | dinghy-law-0.6b-v1 | 0.6B |

## Recomendación: 4 modelos nuevos para el benchmark

### Tier 1 — agregar ya

1. **`Qwen/Qwen3-Embedding-4B`** (4.02B) — La familia más usada del leaderboard. El 8B marca 74.15 RTEB / 67.25 RTEB(Law); el 4B debería quedar cerca. Excelente multilingüe (100+ idiomas, incluido español), MRL (dimensiones truncables 2560→1024), bien soportado en sentence-transformers.
2. **`Qwen/Qwen3-Embedding-0.6B`** (0.6B) — Competidor directo de Octen-0.6B en nuestra clase de tamaño favorita. 10.5M descargas.
3. **`Octen/Octen-Embedding-4B`** (4.02B) — Mejor modelo local ≤4B en RTEB (79.27) y RTEB-Law (73.91). Ya validamos que Octen-0.6B corre bien en la Mac.
4. **`Hanno-Labs/dinghy-law-0.6b-v1`** (0.6B) — Único especializado en ley que cabe en cualquier lado. 65.83 MTEB(Law): mejor que todos los generalistas chicos. Barato de probar.

### Tier 2 — techo de calidad (si la memoria da)

5. **`nvidia/Nemotron-3-Embed-8B-BF16`** (7.95B, ~16 GB bf16) — 79.87 RTEB. Entra en 36 GB pero deja poco aire; mejor con batch chico.
6. **`bflhc/MoD-Embedding`** (4.02B) — 78.65 RTEB / 74.31 RTEB(Law), pero 217 descargas y muy nuevo; riesgo de mantenimiento.
7. **`minetta/nemotron-3-embed-8b-legal`** (7.95B) — 69.33 MTEB(Law), fine-tune legal de un modelo que ya conocemos.

### Descartados

- **voyage-4-large / voyage-law-2 / gemini-embedding-001 / Cohere-embed-v4.0**: API-only, no corren local.
- **Euler-Legal-Embedding-V1**: top en MTEB(Law) pero 8.19B y especializado en ley inglesa/china/alemana; el fine-tune legal de Nemotron (Tier 2) es mejor apuesta del mismo tamaño.
- **Snowflake arctic / bge-m3 / e5-mistral**: quedaron muy abajo en RTEB (49-53).

## Advertencia sobre los modelos especializados en ley

Los tasks de MTEB(Law, v1) (AILA, LegalBench, GerDaLIR, LeCaRD) son en **inglés, alemán y chino**. Un modelo "legal" que gana ahí no necesariamente transfiere a español jurídico mexicano. Nuestro eval de recuperación sobre DOF es exactamente el experimento que responde esa pregunta.

## Método de análisis

- Dataset: [`mteb/results`](https://huggingface.co/datasets/mteb/results) (4 parquet, 8.5M filas).
- Scores: promedio del main score por task en split `test`, modelos públicos.
- RTEB(beta): 30 tasks, cobertura máxima observada ~17; ranking usa cobertura ≥14.
- RTEB(Law, beta): 7 tasks, cobertura ≥2 (benchmark beta, poca cobertura).
- MTEB(Law, v1): 8 tasks, cobertura completa requerida (158 modelos).

## Siguiente paso sugerido

Agregar los 4 modelos Tier 1 a `scripts/compare_embeddings.py` y `scripts/evaluate_retrieval.py` y re-correr en la Mac con MPS. Estimado: ~1-2 horas de cómputo.
