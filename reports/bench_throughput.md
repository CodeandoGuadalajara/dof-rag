# Throughput benchmark: batch size × dtype sweep on M3 MPS

**Date**: 2026-08-01
**Hardware**: MacBook Pro M3 (36 GB), MPS, torch 2.13.0, sentence-transformers 5.6.1
**Script**: `scripts/bench_throughput.py` (corpus `./dof_md`, 100 files, 1,378 chunks, 10 batches per measurement)

## Motivation

Round-1/2 speeds (F2LLM-v2-0.6B: 3.7, jina-v5-small: 2.8 chunks/s) came from the
sentence-transformers default (batch 32, fp32). Before the one-shot full-corpus
indexing job (~6.5M chunks), we tested whether larger batches or fp16 leave
performance on the table.

## Results (chunks/s)

| Model | fp32 bs=32 | fp32 bs=64 | fp32 bs=128 | fp32 bs=256 | fp16 bs=32 | fp16 bs=64 | fp16 bs=128 | fp16 bs=256 |
|---|---|---|---|---|---|---|---|---|
| F2LLM-v2-0.6B | **3.74** | 3.65 | 3.50 | 2.85 | 3.72 | 3.64 | 3.62 | 3.20 |
| jina-v5-text-small | **2.93** | 2.23 | 1.34 | OOM | 2.85 | 2.39 | — | — |

- jina fp32 bs=256 crashed with MPS OOM (tried to allocate 2.39 GiB over the
  high-watermark); fp16 rows for jina bs≥128 were not re-run after the crash —
  the fp32 trend (2.93 → 2.23 → 1.34) already shows larger batches only hurt.

## Numerical safety of fp16

- F2LLM-v2-0.6B: mean cosine(fp32, fp16) = 0.999581 (min 0.999277) on 8 test chunks
- jina-v5-text-small: mean cosine = 1.000000 (min 1.000000) on 4 test chunks

fp16 is numerically safe for both, but buys no speed.

## Conclusions

1. **The default config is already optimal within PyTorch MPS.** Both models are
   compute-bound at batch 32; larger batches degrade throughput (memory pressure,
   attention over long DOF chunks) and fp16 is neutral. Round-1 speeds stand:
   ~3.7 chunks/s F2LLM-0.6B, ~2.9 chunks/s jina-v5-small.
2. **Full-corpus indexing on this laptop remains impractical via PyTorch MPS:**
   ~20 days (F2LLM-0.6B) / ~26 days (jina) of continuous compute for ~6.5M chunks.
3. **Remaining levers** (per the round-2 blog report):
   - Native Apple Silicon ports: MLX or GGUF/llama.cpp (Metal backend, quantized
     weights). Qwen3-based models (F2LLM) have a conversion path; jina-v5 support
     needs verification.
   - Rented GPU (A100/H100) for the one-shot job: hours instead of weeks.

Logs: `logs/bench_throughput_f2llm06.log`, `logs/bench_throughput_jina.log`.

## Addendum: GGUF/llama.cpp (Metal) — 2026-08-01

Script: `scripts/bench_gguf.py` (llama-server `--embedding`, same 1,378 chunks).
GGUFs: `mradermacher/F2LLM-v2-0.6B-GGUF`, official `jinaai/jina-embeddings-v5-text-small-retrieval-GGUF`.

| Model | PyTorch MPS (best) | GGUF f16 bs=32 | GGUF Q8_0 bs=32 | GGUF f16 bs=128 | Speedup (f16) |
|---|---|---|---|---|---|
| F2LLM-v2-0.6B | 3.74 | **5.34** | 5.15 | 5.06 | 1.43× |
| jina-v5-text-small | 2.93 | **5.42** | 5.19 | 5.38 | 1.85× |

Agreement with sentence-transformers fp32 embeddings (16 chunks):

- F2LLM-v2-0.6B f16: mean cosine 0.999316 (min 0.999073) — drop-in replacement.
- jina-v5 f16: 0.958195 (min 0.746109) **raw**, but 0.999939 (min 0.999925) once
  chunks are prefixed with `"Document: "`. jina-v5's ST config prepends
  `Document: ` to passages / `Query: ` to queries
  (`config_sentence_transformers.json`); llama.cpp does not. Index-time chunks
  and query-time queries must carry their prefixes explicitly when using GGUF.

Conclusions:

1. GGUF/Metal is the fastest local path: ~5.3–5.4 chunks/s for both 0.6B models.
   Q8_0 weight quantization is *not* faster than f16 (compute-bound, not
   bandwidth-bound), and larger client batches don't help either.
2. Full-corpus estimate at GGUF f16 speeds: ~14 days continuous per model
   (down from 20–27). Feasible but still painful on a laptop; a rented GPU
   remains the fastest option for the one-shot job.
3. For jina, GGUF indexing is viable only with explicit `"Document: "` /
   `"Query: "` prefixes to stay consistent with the evaluated ST behavior.
