# Comparación: chunker custom vs Chonkie RecursiveChunker

Muestra: **1,000** archivos markdown de `./dof_md`
Límite de tokens por chunk: **800**

## Resumen general

| Chunker | Archivos | Errores | Chunks total | Chunks/archivo (med) | Tiempo (s) | Chunks/s |
|---|---|---|---|---|---|---|
| Custom | 1,000 | 0 | 25,779 | 1.0 | 28.50 | 904.6 |
| Chonkie Recursive | 992 | 8 | 6,540 | 2.0 | 19.66 | 332.7 |

## Tokens por chunk

| Chunker | Media | Mediana | P95 | Máx | Archivos con chunks >10% over max |
|---|---|---|---|---|---|
| Custom | 172.8 | 35.0 | 794.0 | 2,986 | 291 (29.1%) |
| Chonkie Recursive | 599.0 | 703.5 | 796.0 | 800 | 0 (0.0%) |

## Distribución de patrones (chunker custom)

| Patrón | Archivos | Chunks/archivo (media) |
|---|---|---|
| bold_headers | 113 | 10.5 |
| giant_table | 110 | 205.6 |
| h2_compound | 33 | 34.0 |
| plain_text | 19 | 6.8 |
| small | 725 | 1.0 |

## Observaciones

- El chunker custom clasifica el documento antes de dividir; Chonkie RecursiveChunker aplica una regla markdown genérica.
- El contador de tokens es el mismo para ambos (tokenizer de `pplx-embed-context-v1-0.6b`) para hacer la comparación justa.
- Chonkie también ofrece `TableChunker` para documentos dominados por tablas; no se incluyó en esta comparación.

## Conclusión provisional

- **Custom**: chunks más pequeños (mediana 35 tokens) pero 291 archivos (29.1%) con chunks que exceden el límite. Máximo observado: 2,986 tokens.
- **Chonkie Recursive**: respeta el límite en todos los casos (máx 800), chunks más grandes (mediana 704 tokens), y 8 errores.
- El custom es más adecuado para el DOF por su granularidad y preservación de estructura, pero se debe trabajar en acotar los chunks oversized (especialmente tablas gigantes y documentos justo por debajo del umbral `small`).
