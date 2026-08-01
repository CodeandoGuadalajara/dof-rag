"""Throughput benchmark: batch size × dtype sweep on M3 MPS.

Tests whether the round-1 speeds (batch=32, fp32) leave performance on
the table for the final embedding job over ~6.5M chunks.

Run from repo root:
    uv run python scripts/bench_throughput.py [--model NAME] [--n-files 100]
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from compare_embeddings import _get_sample_chunks  # noqa: E402


def bench(model, chunks, batch_size, n_batches=None):
    """Time encoding of chunks at a given batch size."""
    n = len(chunks) if n_batches is None else min(len(chunks), batch_size * n_batches)
    t0 = time.perf_counter()
    for i in range(0, n, batch_size):
        model.encode(chunks[i : i + batch_size], convert_to_numpy=True,
                     show_progress_bar=False, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    return n / elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="codefuse-ai/F2LLM-v2-0.6B")
    parser.add_argument("--n-files", type=int, default=100)
    parser.add_argument("--batch-sizes", default="32,64,128,256")
    args = parser.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    chunks = _get_sample_chunks(Path("../dof_md-local"), args.n_files)
    print(f"{len(chunks):,} chunks from {args.n_files} files")

    from sentence_transformers import SentenceTransformer

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    for dtype_name, dtype in [("fp32", None), ("fp16", torch.float16)]:
        kwargs = {"trust_remote_code": True, "device": device}
        if "jina" in args.model.lower():
            kwargs["model_kwargs"] = {"default_task": "retrieval"}
        if dtype is not None:
            kwargs["model_kwargs"] = {**kwargs.get("model_kwargs", {}), "torch_dtype": dtype}
        model = SentenceTransformer(args.model, **kwargs)
        # warmup
        model.encode(["warmup"], convert_to_numpy=True, show_progress_bar=False)
        for bs in batch_sizes:
            tps = bench(model, chunks, bs, n_batches=20)
            print(f"  {dtype_name} batch={bs:4d}: {tps:6.2f} chunks/s")
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # sanity: fp16 vs fp32 embedding agreement
    print("\ncosine(fp32, fp16) on a test chunk:")
    emb32 = None
    for dtype_name, dtype in [("fp32", None), ("fp16", torch.float16)]:
        kwargs = {"trust_remote_code": True, "device": device}
        if dtype is not None:
            kwargs["model_kwargs"] = {"torch_dtype": dtype}
        model = SentenceTransformer(args.model, **kwargs)
        emb = model.encode(chunks[:8], convert_to_numpy=True, show_progress_bar=False)
        if emb32 is None:
            emb32 = emb
        else:
            a = emb32 / np.linalg.norm(emb32, axis=1, keepdims=True)
            b = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            cos = np.diag(a @ b.T)
            print(f"  mean cosine similarity: {cos.mean():.6f} (min {cos.min():.6f})")
        del model
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
