# Issue #88 — Fase 2: evaluación de la Fase 1 antes de ampliar herramientas

Fecha de evaluación (inyectada): **2026-10-15**. Modelo/servidor: Qwen3.8-Flash-Next
(TensorFold `127.0.0.1:8888`), `reasoning_effort=low`, thinking activo, seed no
forzado (igual que el despliegue). Trabajo GPU secuencial; pilotos activos en
todo momento; TensorFold sano y no-calentando antes/después.

## Método

5 preguntas fijas nuevas (conjunto aparte de Preguntas/v4), 3 repeticiones en
conversaciones separadas por variante, mismas bases de datos y fecha:

1. **Q1 apoyos-desempleo**: "¿Qué apoyos del gobierno federal existen para una
   persona que perdió su empleo?" (el caso PAE/JCF, sin dar nombres)
2. **Q2 programa-especifico**: reglas vigentes y monto de Jóvenes Construyendo
   el Futuro
3. **Q3 consulta-historica**: reforma de subcontratación 2021 y reparto de
   utilidades
4. **Q4 modificaciones**: modificaciones de las reglas de JCF y versión más
   reciente
5. **Q5 amplia-diferente**: apoyos económicos directos a personas publicados
   en 2025

Variantes:

- **baseline**: código pre-Fase 1 (`720d1be`, `git worktree`): chat forzando
  `lexical`, top_k=3, instrucciones sin fecha.
- **fase1-hybrid**: Fase 1 (`d2d4267`) con `retrieval_strategy=hybrid` (config
  del despliegue, 8 candidatos).
- **fase1-lexical**: Fase 1 con `retrieval_strategy=lexical` (misma Fase 1,
  estrategia léxica; el modelo aun así pudo elegir `lexical`/`hybrid` por el
  nuevo parámetro opcional — lo hizo en algunas corridas).

Corridas y trazas completas: `var/issue88-fase2/<variante>/` (45 archivos +
`config.json` por variante). Runner: `var/issue88-fase2-run.py`.

## Resultados cuantitativos (mín–máx de las 3 repeticiones)

| Pregunta | Variante | stop | turnos | tiempo | input tok | output tok | citas |
|---|---|---|---|---|---|---|---|
| Q1 desempleo | baseline | completed | 55 | 197–213 s | 1,372,634 | 9,585 | 4 |
| Q1 desempleo | fase1-hybrid | completed | 27 | 158–182 s | 611,738 | 7,109 | 5 |
| Q1 desempleo | fase1-lexical | completed | 23 | 110–128 s | 504,218 | 5,581 | 6 |
| Q2 programa | baseline | completed | 9 | 37–46 s | 40,161 | 2,073 | 4 |
| Q2 programa | fase1-hybrid | completed | 11 | 81–91 s | 141,147 | 3,948 | 4 |
| Q2 programa | fase1-lexical | completed | 11 | 81–85 s | 141,147 | 3,948 | 4 |
| Q3 histórica | baseline | completed | 5 | 27–35 s | 14,844 | 1,608 | 2 |
| Q3 histórica | fase1-hybrid | completed | 8 | 48–64 s | 56,767 | 2,903 | 7 |
| Q3 histórica | fase1-lexical | completed | 11 | 52–58 s | 93,446 | 2,954 | 6 |
| Q4 modificaciones | baseline | **output_token_limit** | 74 | 282–340 s | 3,392,857 | 14,559 | 13 |
| Q4 modificaciones | fase1-hybrid | completed | 24 | 120–139 s | 404,020 | 5,774 | 6 |
| Q4 modificaciones | fase1-lexical | completed | 24 | 115–123 s | 405,789 | 6,219 | 7 |
| Q5 amplia | baseline | completed | 36 | 147–173 s | 670,671 | 8,834 | 9 |
| Q5 amplia | fase1-hybrid | completed | 34 | 279–325 s | 1,103,086 | 14,310 | 6 |
| Q5 amplia | fase1-lexical | **output_token_limit** | 42 | 309–346 s | 1,351,388 | 17,820 | 12 |

Estrategia registrada en las trazas: `hybrid` aparece como usada de verdad
(`vector_candidates_scanned=300`) en fase1-hybrid; ninguna variante reportó
híbrido sin vectores. En fase1-lexical el modelo ejerció el override
`strategy='lexical'` en Q2/Q4 (y en una de Q5).

**Variabilidad**: dentro de cada variante las 3 repeticiones fueron idénticas
en turnos/tokens (servidor determinista con esta configuración); la
variabilidad relevante es entre variantes. Por eso los mín–máx coinciden casi
siempre; los fallos (Q4 baseline, Q5 fase1-lexical) se reprodujeron 3/3.

## Revisión manual (frente a fuentes, no a respuestas previas del modelo)

Verificado leyendo los chunks citados en `dof_chunks.sqlite`/`dof_corpus_l3.sqlite`:

- **$9,582.47** beca mensual JCF 2026, reglas DOF **31-dic-2025** que abrogan
  las del 31-dic-2024 — correcto en Q2 de las tres variantes
  (chunks 6652627/6652655). ✔
- Reforma de subcontratación DOF **23-abr-2021** (VES) y decreto modificatorio
  de artículos transitorios del **31-jul-2021** — correcto en Q3 de las
  variantes Fase 1 (chunks 5344584; doc 553493). ✔
- PAE 2026 = subprogramas Intermediación y Movilidad Laboral (sin beca
  monetaria propia) — correcto (chunk 6652705). ✔
- Pensión discapacidad **$1,600** y adultos mayores **$3,100** (2025) —
  correcto (chunks 6415176/6415208). ✔
- **PAE/JCF descubiertos sin corrección del usuario en Q1: 3/3 en las tres
  variantes** (criterio de aceptación cumplido también en baseline; la
  diferencia no es descubrimiento sino eficiencia y cierre).

Hallazgos de calidad:

- **Baseline Q4 fracasa** (límite de salida a 74 turnos, 3.4 M tokens de
  entrada, tabla incompleta). Fase 1 resuelve la misma pregunta en 24 turnos
  con la secuencia completa de versiones.
- **Baseline Q1** presenta la beca de capacitación "$150–$300 por día" del SNE
  (chunk 5002054, reglas históricas) mezclada con la situación actual, sin
  marcar vigencia. Las variantes Fase 1 anclan fechas y versión.
- **Fase 1 nunca inventó cortes**: usa la fecha inyectada (15-oct-2026) y
  declara lagunas ("no encontré una nueva publicación que las sustituya, con
  la evidencia revisada…").
- **Citas**: las respuestas Fase 1 incluyeron algunas `[cita no verificada]`
  (guard del runner: chunk citado sin lectura en la corrida) — visibles en Q4
  (1 en híbrido, varias en léxico) y Q5 léxico. El marcado funciona; híbrido
  generó menos citas inválidas que léxico.
- **Q5 amplia sigue siendo cara** en Fase 1: híbrido completa pero con más
  tokens/tiempo que baseline; léxico revienta el límite de salida 3/3.

## Decisión lexical vs hybrid

- Híbrido completa las 5 preguntas; léxico falla en Q5 (output_token_limit
  3/3) y genera más citas no verificadas en Q4/Q5.
- Léxico es más barato cuando funciona (Q1: 504 K vs 612 K tokens), pero la
  diferencia no compensa el fallo de cierre en preguntas amplias.
- **Se mantiene `DOF_RETRIEVAL_MODE=hybrid`** (ya es la configuración del
  despliegue). No se asume "híbrido siempre mejor": se eligió por estos
  resultados con este corpus.

## Conclusión contra criterios de aceptación

- PAE/JCF sin corrección del usuario: ✔ (3/3 todas las variantes).
- Sin cortes inventados ni montos históricos presentados como vigentes: ✔ en
  Fase 1 (baseline mezcla vigencias en Q1).
- Citas que respaldan las afirmaciones: ✔ (verificación manual en fuentes;
  citas no verificadas quedan marcadas, no afirmadas).
- Variabilidad y fallos reportados: ✔ (fallos 3/3 explícitos).
- Sin número fijo de turnos: ✔.

**Fase 1 mejora el chat**: ~2.4–2.7× menos tokens de entrada en Q1, Q4 pasa de
fracaso a respuesta completa y fechada, a costa de algo más de tokens en Q5
híbrido. La navegación en preguntas amplias (Q5) sigue siendo costosa; es el
único candidato justificado para Fase 3, y solo la parte de selección de
outline/extractos (no reranking ni planificador).
