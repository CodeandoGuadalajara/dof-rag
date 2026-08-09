"""Spot-check: doc-collapsed hamming MRR on the partially-embedded corpus.

Runs the GGUF-embedded eval queries whose expected document is FULLY
embedded in the (partial) vec0 store, k=50 hamming per query, collapsed
to doc level. The distractor set grows with the embedding run, so this is
a lower bound on final quality — rerun any time to watch it converge.

Usage:
    uv run python scripts/spot_check_partial.py [--k 50]
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import sqlite_vec

sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus_store.db import connect  # noqa: E402

CACHE = Path("eval/cache")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()

    meta = [json.loads(l) for l in
            open(CACHE / "gguf_jina_v5_small_queries_meta.jsonl")]
    qbin = np.load(CACHE / "gguf_jina_v5_small_queries_bin.npy")

    # eval doc slug -> relpath
    slug2rel = {}
    for line in open("eval/dof_queries_v2.jsonl"):
        r = json.loads(line)
        if not r.get("error"):
            slug2rel[r["doc_id"]] = r["relpath"]

    # relpath -> document_id
    corpus = connect("dof_db/dof_corpus_l3.sqlite")
    rel2doc = {p: d for d, p in corpus.execute(
        "SELECT document_id, path FROM documents")}

    # chunk -> doc map (only for chunks present in vec0)
    vec0 = sqlite3.connect("dof_db/dof_vec0_jina_binary.sqlite")
    vec0.enable_load_extension(True)
    sqlite_vec.load(vec0)
    max_rowid = vec0.execute("SELECT MAX(rowid) FROM chunk_vec").fetchone()[0]
    chunks = sqlite3.connect("dof_db/dof_chunks.sqlite")
    chunk_doc = dict(chunks.execute(
        "SELECT chunk_id, document_id FROM chunks WHERE chunk_id <= ?",
        (max_rowid,)))
    # doc -> its chunk ids (to test full coverage)
    doc_chunks: dict[int, list[int]] = {}
    for cid, did in chunks.execute("SELECT chunk_id, document_id FROM chunks"):
        doc_chunks.setdefault(did, []).append(cid)

    # keep queries whose expected doc is fully embedded
    eligible = []
    for q in meta:
        did = rel2doc.get(slug2rel.get(q["expected_doc_id"], ""), None)
        if did is None:
            continue
        cids = doc_chunks.get(did, [])
        if cids and all(c in chunk_doc for c in cids):
            eligible.append((q, did))
    n_chunks = len(chunk_doc)
    n_docs = len(set(chunk_doc.values()))
    print(f"vec0 store: {max_rowid:,} vectors; eligible queries: "
          f"{len(eligible)} over {n_docs:,} embedded docs / {n_chunks:,} chunks",
          flush=True)
    if not eligible:
        print("nothing to check yet")
        return

    # hamming k=50 per query, collapse to doc level, find expected doc rank
    rr = []
    t0 = time.time()
    per_type: dict[str, list[float]] = {}
    for q, did in eligible:
        rows = vec0.execute(
            "SELECT rowid, distance FROM chunk_vec"
            " WHERE embedding MATCH vec_bit(?) AND k = ?",
            (qbin[q["idx"]].tobytes(), args.k)).fetchall()
        seen, ranked = set(), []
        for rid, dist in rows:
            d = chunk_doc[rid]
            if d not in seen:
                seen.add(d)
                ranked.append(d)
        rank = ranked.index(did) + 1 if did in ranked else None
        rr.append(1.0 / rank if rank else 0.0)
        per_type.setdefault(q["query_type"], []).append(rr[-1])
    dt = time.time() - t0

    print(f"\nspot MRR (doc-collapsed, k={args.k}): "
          f"{sum(rr) / len(rr):.3f}  (n={len(rr)}, {dt / len(rr) * 1000:.0f} ms/query)")
    for t, v in sorted(per_type.items()):
        print(f"  {t:18s} {sum(v) / len(v):.3f}  (n={len(v)})")


if __name__ == "__main__":
    main()
