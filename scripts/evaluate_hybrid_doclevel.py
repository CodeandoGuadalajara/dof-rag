"""Doc-level BM25 x chunk-level vector fusion (per corpus-storage-architecture.md).

Reuses cached embeddings from evaluate_hybrid.py. BM25 runs over whole
documents (external-content FTS in the real architecture); vector ranking is
chunk-level, collapsed to doc-level by best chunk score. Fusion at doc level.
"""
import json, re, sqlite3, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")
from evaluate_retrieval import (
    _apply_variant, _bm25_match_query, _create_queries, _cosine_similarity,
    _FTS5_TOKENIZER, _load_query_dataset, _metrics_from_ranked_lists)

CACHE = Path("eval/cache")
DEPTH = 50
ALPHAS = [0.25, 0.5, 0.75]

docs, generated = _load_query_dataset(Path("../dof_md"), Path("eval/dof_queries_v2.jsonl"))
queries = _create_queries(docs, generated)
print(f"{len(docs)} docs, {len(queries)} queries")

# doc-level BM25
conn = sqlite3.connect(":memory:")
conn.execute(f"CREATE VIRTUAL TABLE d USING fts5(text, doc_id UNINDEXED, tokenize='{_FTS5_TOKENIZER}')")
conn.executemany("INSERT INTO d(text, doc_id) VALUES (?, ?)",
                 [("\n\n".join(doc["chunks"]), doc["doc_id"]) for doc in docs])
bm25_lists = []
for q in queries:
    match = _bm25_match_query(q["query"])
    if not match:
        bm25_lists.append([])
        continue
    cur = conn.execute("SELECT doc_id, bm25(d) FROM d WHERE d MATCH ? ORDER BY bm25(d) LIMIT ?",
                       (match, DEPTH))
    bm25_lists.append([(r[0], -r[1]) for r in cur.fetchall()])
conn.close()

def collapse_vec(sims_row, chunk_doc_ids):
    best = {}
    order = np.argsort(sims_row)[::-1]
    for idx in order:
        d = chunk_doc_ids[idx]
        if d not in best:
            best[d] = float(sims_row[idx])
        if len(best) >= DEPTH:
            break
    return list(best.items())

def fuse_weighted(a, b, alpha):
    def norm(ranked):
        if not ranked: return {}
        s = np.array([x[1] for x in ranked]); lo, hi = s.min(), s.max()
        sp = hi - lo if hi > lo else 1.0
        return {k: (v - lo) / sp for k, v in ranked}
    out = []
    for ra, rb in zip(a, b):
        na, nb = norm(ra), norm(rb)
        sc = {k: alpha * na.get(k, 0) + (1 - alpha) * nb.get(k, 0) for k in set(na) | set(nb)}
        out.append([k for k, _ in sorted(sc.items(), key=lambda kv: -kv[1])])
    return out

def fuse_rrf(a, b, k=60):
    out = []
    for ra, rb in zip(a, b):
        sc = defaultdict(float)
        for r, (idx, _) in enumerate(ra, 1): sc[idx] += 1 / (k + r)
        for r, (idx, _) in enumerate(rb, 1): sc[idx] += 1 / (k + r)
        out.append([k2 for k2, _ in sorted(sc.items(), key=lambda kv: -kv[1])])
    return out

chunk_doc_ids = [doc["doc_id"] for doc in docs for _ in doc["chunks"]]
query_texts = [q["query"] for q in queries]

results = {}
results["BM25-doc"] = _metrics_from_ranked_lists([[k for k, _ in r] for r in bm25_lists], queries)

for label, slug, variant in [
    ("F2LLM-int8", "codefuse_ai_F2LLM_v2_0_6B", "int8"),
    ("jina-binary", "jinaai_jina_embeddings_v5_text_small", "binary"),
]:
    q_emb = _apply_variant(np.load(CACHE / f"{slug}_queries.npy"), variant)
    c_emb = _apply_variant(np.load(CACHE / f"{slug}_chunks.npy"), variant)
    sims = _cosine_similarity(q_emb, c_emb)
    vec_lists = [collapse_vec(sims[i], chunk_doc_ids) for i in range(len(queries))]
    results[f"{label} (doc-collapsed)"] = _metrics_from_ranked_lists(
        [[k for k, _ in r] for r in vec_lists], queries)
    results[f"RRF(BM25doc, {label})"] = _metrics_from_ranked_lists(
        fuse_rrf(bm25_lists, vec_lists), queries)
    for a in ALPHAS:
        results[f"W{a}(BM25doc, {label})"] = _metrics_from_ranked_lists(
            fuse_weighted(bm25_lists, vec_lists, a), queries)

print(f"\n{'system':38s} {'MRR':>6s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s}")
for name, m in sorted(results.items(), key=lambda kv: -kv[1]["mrr"]):
    print(f"{name:38s} {m['mrr']:.3f}  {m['recall_at_k'][1]:.3f}  {m['recall_at_k'][5]:.3f}  {m['recall_at_k'][10]:.3f}")

out = {n: {"mrr": m["mrr"], "recall_at_k": m["recall_at_k"],
           "query_type_metrics": m["query_type_metrics"]} for n, m in results.items()}
Path("eval/cache/hybrid_doclevel_results.json").write_text(json.dumps(out, indent=2))
