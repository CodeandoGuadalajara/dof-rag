# Hybrid retrieval evaluation: BM25 + embeddings

Corpus: `../dof_md` | Query set: `eval/dof_queries_v2.jsonl`
499 documentos, 8,065 chunks, 3,023 queries
Fecha: 2026-08-02

Fusión a nivel chunk, profundidad 50 por sistema. RRF k=60; weighted = alpha·BM25 + (1−alpha)·vectores con scores min-max normalizados por query.

## Overall (doc-level)

| Sistema | MRR | R@1 | R@5 | R@10 | R@5 chunk | MRR chunk |
|---|---|---|---|---|---|---|
| BM25 | 0.616 | 0.561 | 0.687 | 0.728 | 0.798 | 0.696 |
| F2LLM-0.6B fp32 | 0.561 | 0.495 | 0.647 | 0.707 | 0.600 | 0.500 |
| RRF(BM25, F2LLM-0.6B fp32) | 0.632 | 0.571 | 0.713 | 0.772 | 0.735 | 0.630 |
| W0.25(BM25, F2LLM-0.6B fp32) | 0.604 | 0.540 | 0.689 | 0.743 | 0.672 | 0.561 |
| W0.40(BM25, F2LLM-0.6B fp32) | 0.639 | 0.572 | 0.729 | 0.780 | 0.740 | 0.623 |
| W0.50(BM25, F2LLM-0.6B fp32) | 0.661 | 0.595 | 0.749 | 0.788 | 0.779 | 0.672 |
| W0.60(BM25, F2LLM-0.6B fp32) | 0.659 | 0.599 | 0.742 | 0.787 | 0.795 | 0.696 |
| W0.75(BM25, F2LLM-0.6B fp32) | 0.642 | 0.586 | 0.712 | 0.765 | 0.804 | 0.705 |
| F2LLM-0.6B int8 | 0.561 | 0.496 | 0.646 | 0.707 | 0.600 | 0.500 |
| RRF(BM25, F2LLM-0.6B int8) | 0.633 | 0.572 | 0.712 | 0.773 | 0.734 | 0.631 |
| W0.25(BM25, F2LLM-0.6B int8) | 0.604 | 0.540 | 0.689 | 0.744 | 0.671 | 0.562 |
| W0.40(BM25, F2LLM-0.6B int8) | 0.639 | 0.573 | 0.729 | 0.779 | 0.740 | 0.623 |
| W0.50(BM25, F2LLM-0.6B int8) | 0.661 | 0.596 | 0.749 | 0.789 | 0.779 | 0.672 |
| W0.60(BM25, F2LLM-0.6B int8) | 0.659 | 0.599 | 0.742 | 0.787 | 0.795 | 0.697 |
| W0.75(BM25, F2LLM-0.6B int8) | 0.642 | 0.586 | 0.711 | 0.766 | 0.804 | 0.705 |
| jina-v5-small fp32 | 0.558 | 0.493 | 0.645 | 0.697 | 0.625 | 0.511 |
| RRF(BM25, jina-v5-small fp32) | 0.616 | 0.550 | 0.707 | 0.766 | 0.758 | 0.630 |
| W0.25(BM25, jina-v5-small fp32) | 0.600 | 0.539 | 0.680 | 0.737 | 0.692 | 0.578 |
| W0.40(BM25, jina-v5-small fp32) | 0.631 | 0.565 | 0.721 | 0.776 | 0.753 | 0.630 |
| W0.50(BM25, jina-v5-small fp32) | 0.651 | 0.581 | 0.748 | 0.788 | 0.802 | 0.675 |
| W0.60(BM25, jina-v5-small fp32) | 0.651 | 0.587 | 0.738 | 0.787 | 0.807 | 0.701 |
| W0.75(BM25, jina-v5-small fp32) | 0.637 | 0.580 | 0.710 | 0.758 | 0.807 | 0.707 |
| jina-v5-small binary | 0.538 | 0.470 | 0.631 | 0.686 | 0.609 | 0.486 |
| RRF(BM25, jina-v5-small binary) | 0.612 | 0.548 | 0.704 | 0.761 | 0.748 | 0.617 |
| W0.25(BM25, jina-v5-small binary) | 0.585 | 0.519 | 0.672 | 0.727 | 0.674 | 0.555 |
| W0.40(BM25, jina-v5-small binary) | 0.623 | 0.554 | 0.722 | 0.775 | 0.759 | 0.614 |
| W0.50(BM25, jina-v5-small binary) | 0.646 | 0.573 | 0.744 | 0.784 | 0.800 | 0.665 |
| W0.60(BM25, jina-v5-small binary) | 0.649 | 0.586 | 0.736 | 0.788 | 0.809 | 0.696 |
| W0.75(BM25, jina-v5-small binary) | 0.635 | 0.580 | 0.707 | 0.757 | 0.805 | 0.704 |

## Recall@1 por tipo de query

| Sistema | article_specific | factual | first_words | paraphrase | thematic | verbatim_title |
|---|---|---|---|---|---|---|
| BM25 | 0.482 | 0.703 | 0.876 | 0.565 | 0.301 | 0.222 |
| F2LLM-0.6B fp32 | 0.282 | 0.469 | 0.683 | 0.771 | 0.441 | 0.220 |
| RRF(BM25, F2LLM-0.6B fp32) | 0.400 | 0.622 | 0.834 | 0.731 | 0.437 | 0.234 |
| W0.25(BM25, F2LLM-0.6B fp32) | 0.309 | 0.527 | 0.741 | 0.813 | 0.490 | 0.226 |
| W0.40(BM25, F2LLM-0.6B fp32) | 0.345 | 0.586 | 0.812 | 0.806 | 0.490 | 0.232 |
| W0.50(BM25, F2LLM-0.6B fp32) | 0.409 | 0.658 | 0.848 | 0.785 | 0.446 | 0.238 |
| W0.60(BM25, F2LLM-0.6B fp32) | 0.455 | 0.703 | 0.870 | 0.706 | 0.402 | 0.248 |
| W0.75(BM25, F2LLM-0.6B fp32) | 0.464 | 0.708 | 0.876 | 0.645 | 0.356 | 0.246 |
| F2LLM-0.6B int8 | 0.282 | 0.469 | 0.683 | 0.773 | 0.446 | 0.220 |
| RRF(BM25, F2LLM-0.6B int8) | 0.400 | 0.622 | 0.836 | 0.734 | 0.439 | 0.234 |
| W0.25(BM25, F2LLM-0.6B int8) | 0.309 | 0.527 | 0.741 | 0.813 | 0.492 | 0.226 |
| W0.40(BM25, F2LLM-0.6B int8) | 0.345 | 0.586 | 0.814 | 0.806 | 0.492 | 0.232 |
| W0.50(BM25, F2LLM-0.6B int8) | 0.409 | 0.659 | 0.848 | 0.783 | 0.450 | 0.238 |
| W0.60(BM25, F2LLM-0.6B int8) | 0.455 | 0.704 | 0.870 | 0.706 | 0.400 | 0.248 |
| W0.75(BM25, F2LLM-0.6B int8) | 0.464 | 0.708 | 0.876 | 0.645 | 0.354 | 0.246 |
| jina-v5-small fp32 | 0.300 | 0.494 | 0.597 | 0.783 | 0.531 | 0.146 |
| RRF(BM25, jina-v5-small fp32) | 0.418 | 0.608 | 0.768 | 0.703 | 0.460 | 0.200 |
| W0.25(BM25, jina-v5-small fp32) | 0.373 | 0.552 | 0.683 | 0.820 | 0.548 | 0.152 |
| W0.40(BM25, jina-v5-small fp32) | 0.400 | 0.594 | 0.760 | 0.797 | 0.546 | 0.170 |
| W0.50(BM25, jina-v5-small fp32) | 0.427 | 0.645 | 0.800 | 0.750 | 0.492 | 0.206 |
| W0.60(BM25, jina-v5-small fp32) | 0.473 | 0.706 | 0.828 | 0.680 | 0.408 | 0.226 |
| W0.75(BM25, jina-v5-small fp32) | 0.455 | 0.712 | 0.852 | 0.633 | 0.368 | 0.228 |
| jina-v5-small binary | 0.309 | 0.458 | 0.589 | 0.736 | 0.508 | 0.144 |
| RRF(BM25, jina-v5-small binary) | 0.409 | 0.597 | 0.766 | 0.722 | 0.462 | 0.198 |
| W0.25(BM25, jina-v5-small binary) | 0.355 | 0.523 | 0.673 | 0.783 | 0.525 | 0.160 |
| W0.40(BM25, jina-v5-small binary) | 0.391 | 0.573 | 0.743 | 0.801 | 0.531 | 0.174 |
| W0.50(BM25, jina-v5-small binary) | 0.445 | 0.634 | 0.792 | 0.748 | 0.481 | 0.198 |
| W0.60(BM25, jina-v5-small binary) | 0.464 | 0.703 | 0.826 | 0.685 | 0.404 | 0.226 |
| W0.75(BM25, jina-v5-small binary) | 0.455 | 0.712 | 0.862 | 0.624 | 0.364 | 0.228 |

## Conclusiones (análisis manual, 2026-08-02)

1. **La fusión weighted (α=0.5, min-max) es el mejor sistema**: MRR 0.661 con
   F2LLM-0.6B — +4.5 pts sobre BM25 solo (0.616) y +10 pts sobre el mejor
   embedding solo (0.561). La curva de α es plana entre 0.5 y 0.6; RRF (0.633)
   también supera a ambos padres pero queda ~3 pts debajo de weighted.
2. **La cuantización se mantiene en híbrido**: int8 ≡ fp32 en todas las métricas
   (4ª confirmación). jina-binary pierde solo ~0.5–1.5 pts en fusión (vs 2 pts
   solo) porque BM25 compensa donde binary degrada.
3. **Complementariedad confirmada, con sinergia**: en `paraphrase`, W0.25
   alcanza R@1 0.813 — mejor que *ambos* padres (F2LLM 0.771, BM25 0.565).
   En `factual`/`first_words` el híbrido α=0.5 queda ligeramente debajo de
   BM25 solo en R@1 pero lo supera en R@5/R@10.
4. **El α óptimo depende del tipo de query**: tipos lexicales
   (factual/first_words/article_specific) prefieren α=0.75 (BM25-pesado);
   tipos semánticos (paraphrase/thematic) prefieren α=0.25 (vector-pesado).
   Un weighting adaptativo por query — o el routing agéntico propuesto en el
   blog — debería capturar la mayor parte de esa brecha (potencialmente
   +2–4 pts de MRR sobre el α fijo).
5. **Chunk-level**: W0.75 logra el mejor R@5-chunk (0.804) y MRR-chunk
   (0.705), superando a BM25 solo (0.798/0.696).
6. **Decisión de producción (input)**: W0.5 con F2LLM-0.6B-int8 = 0.661 vs
   jina-v5-small-binary = 0.646. Diferencia de 1.5 pts de MRR por ~8× de
   almacenamiento de vectores (6.7 GB vs 0.83 GB para ~6.5M chunks).
