"""Full-corpus BM25 eval: 3,023 eval queries against documents_fts (657,867 docs).

This is the BM25 leg of the full-corpus eval (milestone step 3). Runs the
same MATCH construction as scripts/evaluate_retrieval.py (OR of quoted
word tokens) against the real external-content FTS5 table, depth 50.

Optimization: tokens with document frequency > N/2 have zero/negative IDF
in FTS5's bm25 and contribute nothing meaningful to the score, but force
doclist scans of 300k+ rows per query (17-45 s/query). They are pruned
from the MATCH expression. Verified on samples: top-50 doc SETS identical,
gold-doc ranks unchanged, only tail order permutes. 0.3-0.8 s/query.
df lookups come from documents_fts_vocab (fts5vocab), so query tokens are
diacritics-folded the same way as the index (unicode61 remove_diacritics 1).

Outputs (eval/cache/):
  full_corpus_bm25_results.json  metrics overall + per query type
  full_corpus_bm25_lists.jsonl   raw ranked lists [[document_id, score], ...]
                                 (score = -bm25, higher is better) for the
                                 later hybrid fusion

Usage:
    uv run python scripts/eval_bm25_full.py [--depth 50] [--limit N]
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus_store.db import connect  # noqa: E402

CACHE = Path("eval/cache")
TOP_K = (1, 5, 10)


def fold_diacritics(t: str) -> str:
    """Approximate unicode61 remove_diacritics 1 for vocab lookups."""
    return "".join(c for c in unicodedata.normalize("NFKD", t.lower())
                   if not unicodedata.combining(c))


class DfPruner:
    """Drop tokens whose df exceeds N * threshold (zero/negative IDF)."""

    def __init__(self, conn, n_docs: int, threshold: float = 0.5):
        self.conn = conn
        self.max_df = int(n_docs * threshold)
        self.cache: dict[str, int] = {}

    def df(self, token: str) -> int:
        f = fold_diacritics(token)
        if f not in self.cache:
            r = self.conn.execute(
                "SELECT doc FROM documents_fts_vocab WHERE term = ?",
                (f,)).fetchone()
            self.cache[f] = r[0] if r else 0
        return self.cache[f]

    def prune(self, tokens: list[str]) -> list[str]:
        keep = [t for t in tokens if self.df(t) <= self.max_df]
        return keep or tokens  # never empty out a query


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N queries")
    args = ap.parse_args()

    meta = [json.loads(l) for l in
            open(CACHE / "gguf_jina_v5_small_queries_meta.jsonl")]
    if args.limit:
        meta = meta[: args.limit]

    slug2rel = {}
    for line in open("eval/dof_queries_v2.jsonl"):
        r = json.loads(line)
        if not r.get("error"):
            slug2rel[r["doc_id"]] = r["relpath"]

    corpus = connect("dof_db/dof_corpus_l3.sqlite")
    n_docs = corpus.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    pruner = DfPruner(corpus, n_docs)
    rel2doc = {p: d for d, p in corpus.execute(
        "SELECT document_id, path FROM documents")}

    for q in meta:
        q["expected_document_id"] = rel2doc.get(
            slug2rel.get(q["expected_doc_id"], ""))
    missing = sum(1 for q in meta if q["expected_document_id"] is None)
    print(f"{len(meta)} queries ({missing} without corpus doc mapping)",
          flush=True)

    lists = []
    n_pruned = 0
    t0 = time.time()
    for i, q in enumerate(meta):
        tokens = re.findall(r"\w+", q["query"], flags=re.UNICODE)
        kept = pruner.prune(tokens)
        n_pruned += len(tokens) - len(kept)
        match = " OR ".join(f'"{t}"' for t in kept)
        if not match:
            lists.append([])
            continue
        cur = corpus.execute(
            "SELECT rowid, bm25(documents_fts) FROM documents_fts"
            " WHERE documents_fts MATCH ?"
            " ORDER BY bm25(documents_fts) LIMIT ?",
            (match, args.depth))
        lists.append([(r[0], -r[1]) for r in cur.fetchall()])
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(meta) - i - 1) / 60
            print(f"  {i + 1}/{len(meta)} ({dt / (i + 1) * 1000:.0f} ms/query,"
                  f" ETA {eta:.0f} min)", flush=True)

    with open(CACHE / "full_corpus_bm25_lists.jsonl", "w") as f:
        for q, ranked in zip(meta, lists):
            f.write(json.dumps({"idx": q["idx"], "ranked": ranked}) + "\n")

    # metrics
    recall = defaultdict(float)
    rr = []
    per_type = defaultdict(lambda: {"n": 0, "rr": 0.0,
                                    **{f"r{k}": 0.0 for k in TOP_K}})
    for q, ranked in zip(meta, lists):
        expected = q["expected_document_id"]
        qt = q["query_type"]
        per_type[qt]["n"] += 1
        ids = [d for d, _ in ranked]
        rank = ids.index(expected) + 1 if expected in ids else None
        rr.append(1.0 / rank if rank else 0.0)
        per_type[qt]["rr"] += rr[-1]
        for k in TOP_K:
            if rank is not None and rank <= k:
                recall[k] += 1
                per_type[qt][f"r{k}"] += 1

    n = len(meta)
    results = {
        "system": "BM25-doc-full-corpus",
        "corpus_docs": 657_867, "depth": args.depth, "n_queries": n,
        "df_prune_threshold": pruner.max_df, "tokens_pruned": n_pruned,
        "mrr": sum(rr) / n,
        "recall_at_k": {str(k): recall[k] / n for k in TOP_K},
        "query_type_metrics": {
            qt: {"n": m["n"], "mrr": m["rr"] / m["n"],
                 **{f"recall@{k}": m[f"r{k}"] / m["n"] for k in TOP_K}}
            for qt, m in sorted(per_type.items())},
        "seconds": time.time() - t0,
    }
    (CACHE / "full_corpus_bm25_results.json").write_text(
        json.dumps(results, indent=2))

    print(f"\nBM25 over 657,867 docs (depth {args.depth}, n={n}, "
          f"pruned {n_pruned} stopword-ish tokens)")
    print(f"  MRR {results['mrr']:.3f}  "
          + "  ".join(f"R@{k} {results['recall_at_k'][str(k)]:.3f}"
                      for k in TOP_K))
    for qt, m in results["query_type_metrics"].items():
        print(f"  {qt:18s} MRR {m['mrr']:.3f} (n={m['n']})")


if __name__ == "__main__":
    main()
