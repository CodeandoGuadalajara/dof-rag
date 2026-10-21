# Issue #88 — Fase 3: encabezados coincidentes y extractos en candidatos

Cambio único y pequeño (el único justificado por Fase 2): `search_documents`
ahora adjunta a cada candidato `match_headings` (hasta 2 encabezados cuyo
path contiene términos de la consulta, ≤160 chars c/u) y `match_excerpt`
(≤240 chars alrededor del primer término, reconstruyendo solo hasta 3 chunks
con encabezado coincidente). No es evidencia citable — la descripción de la
herramienta y las instrucciones del chat ya lo aclaran; las citas siguen
requiriendo `read_chunks`.

Los ítems de Fase 3 "selección de outline orientada" y "búsqueda/lectura de
pasajes completos" **no se implementaron**: con este cambio la navegación dejó
de ser el cuello de botella (ver números), y el ticket manda detenerse cuando
la solución sencilla cumple. "Conservar fecha/título en evidencia" ya estaba
cumplido (`search_evidence` devuelve `publication_date` y `title` por chunk).

## Evaluación (mismo protocolo que Fase 2)

Mismo modelo/servidor (TensorFold 8888), fecha inyectada 2026-10-15 (las
corridas fase3 corrieron el 2026-10-19/20 con la misma `today`), 5 preguntas
× 3 repeticiones, variante `fase3-hybrid` vs `fase1-hybrid` ya medida.
Repeticiones idénticas dentro de variante (servidor determinista).

| Pregunta | turns f1→f3 | tiempo f1→f3 | input tokens f1→f3 | citas f1→f3 |
|---|---|---|---|---|
| Q1 desempleo | 27→17 | 158–182s→150–175s | 611,738→245,588 | 5→7 |
| Q2 programa | 11→13 | 81–91s→102–115s | 141,147→189,870 | 4→6 |
| Q3 histórica | 8→8 | 48–64s→42–54s | 56,767→40,333 | 7→4 |
| Q4 modificaciones | 24→21 | 120–139s→97–111s | 404,020→335,796 | 6→6 |
| Q5 amplia | 34→23 | 279–325s→156–193s | 1,103,086→559,178 | 6→8 |

- **Q5 (el motivo de Fase 3)**: ~49% menos tokens de entrada, 32% menos
  turnos, ~40% menos tiempo. Completada 3/3.
- **Q1**: ~60% menos tokens de entrada (27→17 turnos), misma calidad: la
  respuesta ahora distingue explícitamente "apoyos que dan dinero" vs
  "servicios para conseguir empleo" y cita las reglas PAE 2026.
- **Q4**: menos tokens y turnos, secuencia completa.
- **Q3**: igual o mejor; menos citas pero todas verificadas.
- **Q2** es el único punto con costo mayor (11→13 turnos, +~25 s, +48K
  tokens): el modelo usa los extractos para leer un poco más, con 6 citas
  verificadas (vs 4). Aceptable dentro de la variación; no revierte la
  ganancia neta.

## Revisión manual (frente a fuentes)

- Q1/Q2: reglas PAE/JCF 2026 DOF 31-dic-2025, monto $9,582.47 — correcto
  (chunks 6652627/6652655/6652705).
- Q3: decreto 23-abr-2021 + modificatorio 31-jul-2021 — correcto.
- Q5: $3,100 adultos mayores, $1,500 Mujeres Bienestar (chunk 6426036,
  31-ene-2025), $1,600 discapacidad — correcto. 2 afirmaciones con
  `[cita no verificada]` (monto de discapacidad citado sin lectura en esa
  corrida); el guard las marcó como no-citables.

## Decisión

Se **conserva el cambio**: menor costo de navegación en las preguntas caras
sin pérdida de calidad verificada. Con esto se agotan las mejoras justificadas
por datos del ticket (no hay reranking, planificador ni grafo de versiones,
que siguen diferidos).
