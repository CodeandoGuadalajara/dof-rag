# Comparación: chunker custom vs Chonkie chunkers

Muestra: **1,000** archivos markdown de `./dof_md`
Límite de tokens por chunk: **800**

## Resumen general

| Chunker | Archivos | Errores | Chunks total | Chunks/archivo (med) | Tiempo (s) | Chunks/s |
|---|---|---|---|---|---|---|
| Custom | 1,000 | 0 | 9,553 | 1.0 | 31.50 | 303.2 |
| Chonkie Recursive | 1,000 | 0 | 9,171 | 1.0 | 23.71 | 386.9 |
| Chonkie H2 | 1,000 | 0 | 8,401 | 1.0 | 30.11 | 279.0 |
| Chonkie Table | 1,000 | 0 | 28,687 | 1.0 | 20.73 | 1383.7 |
| Chonkie Token | 1,000 | 0 | 7,808 | 1.0 | 11.76 | 664.2 |
| Chonkie Sentence | 1,000 | 0 | 7,981 | 1.0 | 21.35 | 373.9 |
| Chonkie Pipeline | 1,000 | 0 | 217,321 | 1.0 | 30.56 | 7111.0 |
| Chonkie Pipeline Rev | 1,000 | 0 | 7,754 | 1.0 | 16.62 | 466.4 |

## Tokens por chunk

| Chunker | Media | Mediana | P95 | Máx | Archivos con chunks >10% over max |
|---|---|---|---|---|---|
| Custom | 603.0 | 730.0 | 799.0 | 849 | 0 (0.0%) |
| Chonkie Recursive | 609.5 | 717.0 | 797.0 | 800 | 0 (0.0%) |
| Chonkie H2 | 667.6 | 708.0 | 799.0 | 111,625 | 54 (5.4%) |
| Chonkie Table | 5242.0 | 3283.0 | 17069.0 | 64,462 | 114 (11.4%) |
| Chonkie Token | 759.4 | 800.0 | 800.0 | 800 | 0 (0.0%) |
| Chonkie Sentence | 719.0 | 764.0 | 795.0 | 3,976 | 9 (0.9%) |
| Chonkie Pipeline | 685.0 | 744.0 | 797.0 | 800 | 0 (0.0%) |
| Chonkie Pipeline Rev | 705.0 | 757.5 | 798.0 | 800 | 0 (0.0%) |

## Distribución de patrones (chunker custom)

| Patrón | Archivos | Chunks/archivo (media) |
|---|---|---|
| bold_headers | 121 | 11.1 |
| giant_table | 98 | 54.2 |
| h2_compound | 40 | 45.6 |
| plain_text | 24 | 7.4 |
| small | 717 | 1.3 |

## Observaciones

- El chunker custom clasifica el documento antes de dividir; Chonkie RecursiveChunker aplica una regla markdown genérica.
- Chonkie H2 usa H2 como delimitador principal, lo que lo hace más comparable con el custom en documentos compuestos.
- Chonkie TableChunker detecta tablas en el documento y repite automáticamente el encabezado del documento en cada chunk.
- Chonkie TokenChunker y SentenceChunker tienen `chunk_overlap` integrado y garantizan respetar el límite de tokens.
- Chonkie Pipeline encadena TableChunker y RecursiveChunker para manejar documentos mixtos.
- Chonkie Pipeline Rev hace lo inverso: RecursiveChunker primero y TableChunker solo sobre los fragmentos que contienen tablas válidas.
- El contador de tokens es el mismo para los ocho (tokenizer de `pplx-embed-context-v1-0.6b`) para hacer la comparación justa.

## Conclusión provisional

- **Custom**: chunks más pequeños y granulares (mediana 730 tokens), 0 archivos con chunks que exceden el límite. Máximo observado: 849 tokens.
- **Chonkie Recursive**: chunks más grandes (mediana 717 tokens), máx 800, 0 errores.
- **Chonkie H2**: 8,401 chunks (mediana 708 tokens), pero 54 archivos (5.4%) producen chunks que exceden el límite; máx 111,625 tokens.
- **Chonkie Table**: 28,687 chunks (mediana 3283 tokens), pero 114 archivos (11.4%) producen chunks enormes; máx 64,462 tokens. TableChunker no respeta el límite de tokens en documentos con tablas grandes y genera chunks inmanejables para recuperación.
- **Chonkie Token**: 7,808 chunks (mediana 800 tokens), 0 archivos oversized; máx 800 tokens.
- **Chonkie Sentence**: 7,981 chunks (mediana 764 tokens), 9 archivos oversized; máx 3,976 tokens.
- **Chonkie Pipeline**: 217,321 chunks (mediana 744 tokens), 0 archivos oversized; máx 800 tokens. El pipeline explota el número de chunks en documentos con muchas tablas porque TableChunker divide en cada límite de tabla y RecursiveChunker vuelve a partir cada fragmento.
- **Chonkie Pipeline Rev**: 7,754 chunks (mediana 758 tokens), 0 archivos oversized; máx 800 tokens.
- El custom es más adecuado para el DOF porque respeta la estructura documental (H2s, tablas, negritas) y genera chunks recuperables; entre las opciones de Chonkie, RecursiveChunker es la más estable para markdown general, mientras que TokenChunker/SentenceChunker son las más seguras para límites estrictos. El Pipeline table+recursive no es recomendable para este corpus; el Pipeline recursivo+table es mejor pero aún puede partir tablas.
