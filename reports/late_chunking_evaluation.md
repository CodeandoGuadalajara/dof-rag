# Late chunking: pplx-embed-context-v1-0.6b

Corpus: `../dof_md-local`
Muestra: **200** documentos (200 usados), 400 queries (seed 42)
Fecha: 2026-07-30

Comparación pareada sobre **los mismos chunks y las mismas queries**:

- `standard`: cada chunk embeddado de forma aislada (como en la ronda 1).
- `late_chunking`: forward pass del documento completo (hasta 32,768 tokens) + mean-pooling del span de tokens de cada chunk.

| Encoding | Chunks | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG | Tiempo (s) |
|---|---|---|---|---|---|---|---|
| standard | 1,451 | 0.335 | 0.510 | 0.568 | 0.406 | 0.445 | 407.4 |
| late_chunking | 1,067 | 0.343 | 0.480 | 0.527 | 0.400 | 0.431 | 1245.4 |

## Delta late chunking vs standard

- Δ MRR: **-0.6 pts**
- Δ Recall@1: +0.8 pts
- Δ Recall@5: -3.0 pts

## Stats

- Documentos procesados: 200
- Documentos truncados a 32,768 tokens: 6
- Chunks late chunking: 1,067 (descartados por truncado: 384)
- Chunks standard: 1,451

## Notas

- Los chunks los genera un splitter con tracking de offsets (sin overlap ni prefijos de encabezado: el contexto lo da la codificación del documento completo).
- Las queries se codifican de forma estándar en ambos brazos.
- Solo fp32: la cuantización int8 se evalúa después si el delta lo justifica.
- Muestra determinística (seed 42, archivos ordenados).
