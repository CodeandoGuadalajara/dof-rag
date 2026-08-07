"""Full-corpus hybrid eval: BM25 (cached lists) x binary vectors (vec0 hamming).

The vector + fusion leg of the full-corpus eval (milestone step 3).
Vector ranking: hamming k=50 over the sqlite-vec bit[1024] store,
doc-collapsed by min distance. Fusion: min-max-normalized weighted
(alpha) and RRF, same as scripts/evaluate_hybrid_doclevel.py.

Can run against a PARTIAL vec0 store while embeddings are still being
written: use --eligible-only to restrict metrics to queries whose gold
doc is fully embedded (fair intermediate signal). Without it, all 3,023
queries run (vector leg simply can't return unembedded docs yet).

Inputs (eval/cache/):
  gguf_jina_v5_small_queries_{bin.npy,meta.jsonl}   from embed_eval_queries.py
  full_corpus_bm25_lists.jsonl                      from eval_bm25_full.py

Outputs (eval/cache/):
  full_corpus_vector_lists.jsonl     doc-collapsed vector ranked lists
  full_corpus_hybrid_results.json    metrics for all systems

Usage:
    uv run python scripts/eval_hybrid_full.py [--k 50] [--eligible-only]
"""

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sqlite_vec

sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus_store.db import connect  # noqa: E402

CACHE = Path("eval/cache")
TOP_K = (1, 5, 10)
ALPHAS = (0.25, 0.5, 0.75)


def doc_collapse(rows, chunk_doc, depth):
    """Chunk ranked list -> [(doc_id, -min_hamming)], best-first.

    rows arrive ordered by ascending hamming distance, so the first
    occurrence of a doc is already its best chunk.
    """
    best: dict[int, float] = {}
    for rid, dist in rows:
        d = chunk_doc.get(rid)
        if d is not None and d not in best:
            best[d] = dist
            if len(best) >= depth:
                break
    return [(d, -dist) for d, dist in best.items()]


def fuse_weighted(a, b, alpha):
    def norm(ranked):
        if not ranked:
            return {}
        s = np.array([x[1] for x in ranked])
        lo, hi = s.min(), s.max()
        sp = hi - lo if hi > lo else 1.0
        return {k: (v - lo) / sp for k, v in ranked}
    out = []
    for ra, rb in zip(a, b):
        na, nb = norm(ra), norm(rb)
        sc = {k: alpha * na.get(k, 0) + (1 - alpha) * nb.get(k, 0)
              for k in set(na) | set(nb)}
        out.append(sorted(sc, key=lambda k: -sc[k]))
    return out


def fuse_rrf(a, b, k=60):
    out = []
    for ra, rb in zip(a, b):
        sc = defaultdict(float)
        for r, (d, _) in enumerate(ra, 1):
            sc[d] += 1 / (k + r)
        for r, (d, _) in enumerate(rb, 1):
            sc[d] += 1 / (k + r)
        out.append(sorted(sc, key=lambda d: -sc[d]))
    return out


def metrics(ranked_ids, queries):
    recall = defaultdict(float)
    rr, per_type = [], defaultdict(lambda: [0, 0.0])
    for ids, q in zip(ranked_ids, queries):
        expected = q["expected_document_id"]
        rank = ids.index(expected) + 1 if expected in ids else None
        val = 1.0 / rank if rank else 0.0
        rr.append(val)
        t = per_type[q["query_type"]]
        t[0] += 1
        t[1] += val
        for k in TOP_K:
            if rank is not None and rank <= k:
                recall[k] += 1
    n = len(queries)
    return {
        "n_queries": n, "mrr": sum(rr) / n,
        "recall_at_k": {str(k): recall[k] / n for k in TOP_K},
        "query_type_mrr": {t: v[1] / v[0] for t, v in sorted(per_type.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--eligible-only", action="store_true")
    ap.add_argument("--meta", default=str(CACHE / "gguf_jina_v5_small_queries_meta.jsonl"))
    ap.add_argument("--bin", dest="bin_npy",
                    default=str(CACHE / "gguf_jina_v5_small_queries_bin.npy"))
    ap.add_argument("--bm25-lists", default=str(CACHE / "full_corpus_bm25_lists.jsonl"))
    ap.add_argument("--queries", default="eval/dof_queries_v2.jsonl")
    ap.add_argument("--out-prefix", default="full_corpus_hybrid")
    args = ap.parse_args()

    meta = [json.loads(l) for l in open(args.meta)]
    qbin = np.load(args.bin_npy)
    bm25 = {}
    for l in open(args.bm25_lists):
        r = json.loads(l)
        bm25[r["idx"]] = r["ranked"]

    corpus = connect("dof_db/dof_corpus_l3.sqlite")
    rel2doc = {p: d for d, p in corpus.execute(
        "SELECT document_id, path FROM documents")}
    slug2rel = {}
    for line in open(args.queries):
        r = json.loads(line)
        if not r.get("error"):
            slug2rel[r["doc_id"]] = r["relpath"]
    for q in meta:
        q["expected_document_id"] = rel2doc.get(
            slug2rel.get(q["expected_doc_id"], ""))

    vec0 = sqlite3.connect("dof_db/dof_vec0_jina_binary.sqlite")
    vec0.enable_load_extension(True)
    sqlite_vec.load(vec0)
    n_vecs = vec0.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0]
    max_rowid = vec0.execute("SELECT MAX(rowid) FROM chunk_vec").fetchone()[0]

    chunks = sqlite3.connect("dof_db/dof_chunks.sqlite")
    t = time.time()
    chunk_doc = dict(chunks.execute("SELECT chunk_id, document_id FROM chunks"))
    print(f"{n_vecs:,} vectors in store; chunk->doc map "
          f"({len(chunk_doc):,}) in {time.time() - t:.0f}s", flush=True)

    # vector leg: hamming k=K per query, doc-collapsed
    vec_lists = []
    t0 = time.time()
    for i, q in enumerate(meta):
        rows = vec0.execute(
            "SELECT rowid, distance FROM chunk_vec"
            " WHERE embedding MATCH vec_bit(?) AND k = ?",
            (qbin[q["idx"]].tobytes(), args.k)).fetchall()
        vec_lists.append(doc_collapse(rows, chunk_doc, args.k))
        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            print(f"  {i + 1}/{len(meta)} ({dt / (i + 1) * 1000:.0f} ms/query)",
                  flush=True)
    with open(CACHE / f"{args.out_prefix}_vector_lists.jsonl", "w") as f:
        for q, ranked in zip(meta, vec_lists):
            f.write(json.dumps({"idx": q["idx"], "ranked": ranked}) + "\n")

    queries = meta
    if args.eligible_only:
        doc_chunks: dict[int, list[int]] = {}
        for cid, did in chunk_doc.items():
            doc_chunks.setdefault(did, []).append(cid)
        def eligible(q):
            cids = doc_chunks.get(q["expected_document_id"] or -1, [])
            return bool(cids) and all(c <= max_rowid for c in cids)
        sel = [i for i, q in enumerate(meta) if eligible(q)]
        queries = [meta[i] for i in sel]
        vec_lists = [vec_lists[i] for i in sel]
        bm25_lists = [bm25[q["idx"]] for q in queries]
        print(f"eligible-only: {len(queries)} queries "
              f"(gold doc fully embedded)", flush=True)
    else:
        bm25_lists = [bm25[q["idx"]] for q in queries]

    results = {
        "corpus_docs": 657_867, "depth": args.k,
        "vectors_in_store": n_vecs, "eligible_only": args.eligible_only,
        "systems": {},
    }
    vec_ids = [[d for d, _ in r] for r in vec_lists]
    bm25_ids = [[d for d, _ in r] for r in bm25_lists]
    results["systems"]["BM25-doc"] = metrics(bm25_ids, queries)
    results["systems"]["jina-binary (doc-collapsed)"] = metrics(vec_ids, queries)
    results["systems"]["RRF(BM25doc, jina-binary)"] = metrics(
        fuse_rrf(bm25_lists, vec_lists), queries)
    for a in ALPHAS:
        results["systems"][f"W{a}(BM25doc, jina-binary)"] = metrics(
            fuse_weighted(bm25_lists, vec_lists, a), queries)

    (CACHE / f"{args.out_prefix}_results.json").write_text(
        json.dumps(results, indent=2))

    print(f"\n{'system':34s} {'MRR':>6s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s}")
    for name, m in sorted(results["systems"].items(),
                          key=lambda kv: -kv[1]["mrr"]):
        print(f"{name:34s} {m['mrr']:.3f}  {m['recall_at_k']['1']:.3f}  "
              f"{m['recall_at_k']['5']:.3f}  {m['recall_at_k']['10']:.3f}")


if __name__ == "__main__":
    main()
