"""GGUF/llama.cpp throughput benchmark on M3 Metal.

Starts llama-server (embedding mode) for a GGUF model, times batched
/v1/embeddings calls over the same sample chunks used by
scripts/bench_throughput.py, and optionally checks cosine agreement
against sentence-transformers fp32 embeddings.

Run from repo root (llama-server must be on PATH):
    uv run python scripts/bench_gguf.py --gguf ~/dof-gguf/F2LLM-v2-0.6B.f16.gguf \
        [--hf-model codefuse-ai/F2LLM-v2-0.6B] [--batch-size 32]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from compare_embeddings import _get_sample_chunks  # noqa: E402

PORT = 8085


def start_server(gguf: Path, ctx: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            "llama-server", "-m", str(gguf),
            "--embedding", "--port", str(PORT),
            "-c", str(ctx), "-b", "8192", "-ub", "4096",
            "--log-disable",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import urllib.request
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("llama-server did not become healthy")


def embed_batch(chunks: list[str]) -> np.ndarray:
    import json
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/embeddings",
        data=json.dumps({"input": chunks}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    data["data"].sort(key=lambda d: d["index"])
    return np.array([d["embedding"] for d in data["data"]], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--hf-model", default=None,
                        help="HF model for cosine agreement check (skipped if omitted)")
    parser.add_argument("--corpus", default="./dof_md")
    parser.add_argument("--n-files", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ctx", type=int, default=8192)
    args = parser.parse_args()

    chunks = _get_sample_chunks(Path(args.corpus), args.n_files)
    print(f"{len(chunks):,} chunks from {args.n_files} files")

    proc = start_server(args.gguf, args.ctx)
    try:
        # warmup
        embed_batch(["warmup"])
        bs = args.batch_size
        t0 = time.perf_counter()
        embs = []
        for i in range(0, len(chunks), bs):
            embs.append(embed_batch(chunks[i : i + bs]))
        elapsed = time.perf_counter() - t0
        emb = np.vstack(embs)
        print(f"gguf {args.gguf.name} batch={bs}: {len(chunks) / elapsed:6.2f} chunks/s "
              f"({elapsed:.0f}s total, dims={emb.shape[1]})")
    finally:
        proc.terminate()
        proc.wait()

    if args.hf_model:
        from sentence_transformers import SentenceTransformer
        kwargs = {"trust_remote_code": True, "device": "mps"}
        if "jina" in args.hf_model.lower():
            kwargs["model_kwargs"] = {"default_task": "retrieval"}
        model = SentenceTransformer(args.hf_model, **kwargs)
        ref = model.encode(chunks[:16], convert_to_numpy=True, show_progress_bar=False)
        a = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        b = emb[:16] / np.linalg.norm(emb[:16], axis=1, keepdims=True)
        cos = np.diag(a @ b.T)
        print(f"cosine(ST fp32, GGUF) on 16 chunks: mean {cos.mean():.6f} (min {cos.min():.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
