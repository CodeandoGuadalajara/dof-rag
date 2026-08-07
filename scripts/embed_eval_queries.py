"""Pre-embed the eval queries with the local llama.cpp GGUF (jina v5 small).

The full-corpus vector store is embedded via the local GGUF, so the query
side must come from the SAME model/server (sign-packed hamming comparisons
require one embedding space). Uses the running llama-server (port 8085)
shared with the corpus embedding run — 3k short queries cost a few minutes.

Outputs (eval/cache/):
  gguf_jina_v5_small_queries_float.npy  float32 (n, 1024)
  gguf_jina_v5_small_queries_bin.npy    uint8   (n, 128)  sign-packed
  gguf_jina_v5_small_queries_meta.jsonl one line per query, in array order

Usage:
    uv run python scripts/embed_eval_queries.py [--port 8085] [--batch 64]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from corpus_store.embed import PREFIX_QUERY, pack_binary, embed_batch  # noqa: E402
from evaluate_retrieval import _create_queries, _load_query_dataset  # noqa: E402

CACHE = Path("eval/cache")
SLUG = "gguf_jina_v5_small"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--queries", default="eval/dof_queries_v2.jsonl")
    ap.add_argument("--slug", default=SLUG,
                    help="output name prefix inside eval/cache/")
    args = ap.parse_args()
    slug = args.slug

    docs, generated = _load_query_dataset(
        Path("../dof_md"), Path(args.queries))
    queries = _create_queries(docs, generated)
    print(f"{len(docs)} docs, {len(queries)} queries", flush=True)

    texts = [PREFIX_QUERY + q["query"] for q in queries]
    t0 = time.time()
    embs = []
    for i in range(0, len(texts), args.batch):
        embs.append(embed_batch(texts[i:i + args.batch], args.port))
        if (i // args.batch) % 10 == 0:
            print(f"  {i + len(embs[-1])}/{len(texts)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    emb = np.vstack(embs).astype(np.float32)
    assert emb.shape == (len(queries), 1024), emb.shape

    norms = np.linalg.norm(emb, axis=1)
    assert np.all(norms > 0.99), f"non-normalized rows: {norms.min()}"
    packed = np.stack([np.frombuffer(pack_binary(row), dtype=np.uint8)
                       for row in emb])

    CACHE.mkdir(exist_ok=True)
    np.save(CACHE / f"{slug}_queries_float.npy", emb)
    np.save(CACHE / f"{slug}_queries_bin.npy", packed)
    with open(CACHE / f"{slug}_queries_meta.jsonl", "w") as f:
        for i, q in enumerate(queries):
            f.write(json.dumps({
                "idx": i, "query": q["query"], "query_type": q["query_type"],
                "expected_doc_id": q["expected_doc_id"],
                "expected_chunk_index": q["expected_chunk_index"],
            }, ensure_ascii=False) + "\n")
    print(f"saved {emb.shape[0]} queries "
          f"(float {emb.nbytes / 2**20:.1f} MiB, bin {packed.nbytes / 2**20:.1f} MiB) "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
