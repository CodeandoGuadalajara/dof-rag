# Comparación: chunker custom vs Chonkie chunkers

Muestra: **1,000** archivos markdown de `./dof_md`
Límite de tokens por chunk: **800**

## Resumen general

| Chunker | Archivos | Errores | Chunks total | Chunks/archivo (med) | Tiempo (s) | Chunks/s |
|---|---|---|---|---|---|---|
| Custom | 1,000 | 0 | 7,735 | 2.0 | 25.96 | 297.9 |
| Chonkie Recursive | 1,000 | 0 | 7,157 | 2.0 | 20.63 | 347.0 |
| Chonkie H2 | 1,000 | 0 | 7,028 | 2.0 | 25.84 | 272.0 |
| Chonkie Table | 1,000 | 0 | 13,856 | 1.0 | 18.14 | 763.9 |

## Tokens por chunk

| Chunker | Media | Mediana | P95 | Máx | Archivos con chunks >10% over max |
|---|---|---|---|---|---|
| Custom | 568.2 | 706.0 | 798.0 | 879 | 0 (0.0%) |
| Chonkie Recursive | 601.9 | 707.0 | 796.0 | 800 | 0 (0.0%) |
| Chonkie H2 | 615.0 | 702.0 | 798.0 | 7,895 | 47 (4.7%) |
| Chonkie Table | 4958.2 | 3633.5 | 12097.0 | 102,786 | 268 (26.8%) |

## Distribución de patrones (chunker custom)

| Patrón | Archivos | Chunks/archivo (media) |
|---|---|---|
| bold_headers | 248 | 6.4 |
| giant_table | 123 | 32.5 |
| h2_compound | 39 | 31.7 |
| plain_text | 32 | 5.6 |
| small | 558 | 1.3 |

## Observaciones

- El chunker custom clasifica el documento antes de dividir; Chonkie RecursiveChunker aplica una regla markdown genérica.
- Chonkie H2 usa H2 como delimitador principal, lo que lo hace más comparable con el custom en documentos compuestos.
- Chonkie TableChunker detecta tablas en el documento y repite automáticamente el encabezado del documento en cada chunk.
- El contador de tokens es el mismo para los cuatro (tokenizer de `pplx-embed-context-v1-0.6b`) para hacer la comparación justa.

## Conclusión provisional

- **Custom**: chunks más pequeños y granulares (mediana 706 tokens), 0 archivos con chunks que exceden el límite. Máximo observado: 879 tokens.
- **Chonkie Recursive**: chunks más grandes (mediana 707 tokens), máx 800, 0 errores.
- **Chonkie H2**: 7,028 chunks (mediana 702 tokens), pero 47 archivos (4.7%) producen chunks que exceden el límite; máx 7,895 tokens.
- **Chonkie Table**: 13,856 chunks (mediana 3634 tokens), pero 268 archivos (26.8%) producen chunks enormes; máx 102,786 tokens. TableChunker no respeta el límite de tokens en documentos con tablas grandes y genera chunks inmanejables para recuperación.
- El custom es más adecuado para el DOF porque respeta la estructura documental (H2s, tablas, negritas) y genera chunks recuperables; entre las opciones de Chonkie, RecursiveChunker es la más estable.
