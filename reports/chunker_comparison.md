# Comparación: chunker custom vs Chonkie RecursiveChunker

Muestra: **1,000** archivos markdown de `./dof_md`
Límite de tokens por chunk: **800**

## Resumen general

| Chunker | Archivos | Errores | Chunks total | Chunks/archivo (med) | Tiempo (s) | Chunks/s |
|---|---|---|---|---|---|---|
| Custom | 1,000 | 0 | 26,636 | 2.0 | 27.39 | 972.5 |
| Chonkie Recursive | 1,000 | 0 | 7,157 | 2.0 | 20.31 | 352.4 |
| Chonkie H2 | 1,000 | 0 | 7,028 | 2.0 | 25.37 | 277.0 |

## Tokens por chunk

| Chunker | Media | Mediana | P95 | Máx | Archivos con chunks >10% over max |
|---|---|---|---|---|---|
| Custom | 165.0 | 36.0 | 775.0 | 853 | 0 (0.0%) |
| Chonkie Recursive | 601.9 | 707.0 | 796.0 | 800 | 0 (0.0%) |
| Chonkie H2 | 615.0 | 702.0 | 798.0 | 7,895 | 47 (4.7%) |

## Distribución de patrones (chunker custom)

| Patrón | Archivos | Chunks/archivo (media) |
|---|---|---|
| bold_headers | 248 | 6.4 |
| giant_table | 123 | 186.1 |
| h2_compound | 39 | 31.7 |
| plain_text | 32 | 5.6 |
| small | 558 | 1.3 |

## Observaciones

- El chunker custom clasifica el documento antes de dividir; Chonkie RecursiveChunker aplica una regla markdown genérica.
- Chonkie H2 usa H2 como delimitador principal, lo que lo hace más comparable con el custom en documentos compuestos.
- El contador de tokens es el mismo para los tres (tokenizer de `pplx-embed-context-v1-0.6b`) para hacer la comparación justa.
- Chonkie también ofrece `TableChunker` para documentos dominados por tablas; no se incluyó en esta comparación.

## Conclusión provisional

- **Custom**: chunks más pequeños y granulares (mediana 36 tokens), 0 archivos con chunks que exceden el límite. Máximo observado: 853 tokens.
- **Chonkie Recursive**: chunks más grandes (mediana 707 tokens), máx 800, 0 errores.
- **Chonkie H2**: con H2 como delimitador principal genera 7,028 chunks (mediana 702 tokens), pero 47 archivos (4.7%) producen chunks que exceden el límite; máximo observado: 7,895 tokens. Esto pasa cuando una sección H2 es un párrafo o tabla gigante sin sub-delimitadores.
- El custom es más adecuado para el DOF porque respeta la estructura documental (H2s, tablas, negritas), genera chunks más recuperables y no deja secciones completas sin partir.
