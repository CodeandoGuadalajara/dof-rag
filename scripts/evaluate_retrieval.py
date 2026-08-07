"""Evaluate embedding models on retrieval quality using synthetic DOF queries.

Run from repo root:
    python scripts/evaluate_retrieval.py [--corpus PATH] [--sample-size N]

Round 2 changes (post PR #57):
- Model list narrowed to the round-1 winners: pplx-embed-context-v1-0.6b,
  F2LLM-v2-1.7B, F2LLM-v2-0.6B, jina-embeddings-v5-text-small.
- BM25 full-text search baseline (SQLite FTS5) evaluated on the same
  chunks/queries/metrics as the embedding models.
- Corpus path and sample size are CLI args; defaults reproduce round 1.
- `--queries PATH` loads a versioned query set (JSONL from
  scripts/generate_queries.py): the eval runs on exactly those documents,
  adds the generated query types (paraphrase/thematic/factual/
  article_specific) to the verbatim ones, and reports a per-type breakdown
  plus chunk-level recall for queries with chunk-level ground truth.

For each embedding model:
1. Embed a sample of document chunks once (full fp32).
2. Evaluate retrieval quality on post-hoc variants:
   - full fp32 (baseline)
   - Matryoshka truncation to 768 dims (mrl_768)
   - int8 scalar quantization
   - binary (sign) quantization
3. Generate synthetic queries from document titles/headings.
4. Measure Recall@k, MRR, NDCG per variant.

Outputs a Markdown report to `reports/retrieval_evaluation.md`.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sqlite3
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from random import sample, seed

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_poc.chunker import split_file  # noqa: E402

REPORT_DIR = Path("reports")
DEFAULT_CORPUS = "./dof_md"
SAMPLE_SIZE = 50  # default files for retrieval evaluation (round-1 value)
SEED = 42
TOP_K = [1, 5, 10]
MRL_DIM = 768

# Variants evaluated per model (post-hoc transforms of the same fp32 embeddings)
VARIANTS = ["full_fp32", "mrl_768", "int8", "binary"]

MODELS = [
    "perplexity-ai/pplx-embed-context-v1-0.6b",
    "codefuse-ai/F2LLM-v2-1.7B",
    "codefuse-ai/F2LLM-v2-0.6B",
    "jinaai/jina-embeddings-v5-text-small",
]


def _iter_md_files(root: Path) -> list[Path]:
    """Recursively list .md files, following directory symlinks."""
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if name.endswith(".md"):
                files.append(Path(dirpath) / name)
    return files


def _get_documents(corpus: Path, n_files: int = SAMPLE_SIZE) -> list[dict]:
    """Get documents with their chunks and metadata."""
    seed(SEED)
    files = sorted(sample(sorted(_iter_md_files(corpus)), n_files))
    docs: list[dict] = []
    for f in files:
        try:
            chunks = list(split_file(f))
            if not chunks:
                continue
            doc_id = f.stem
            title = chunks[0].heading_path[0] if chunks[0].heading_path else doc_id
            docs.append({
                "doc_id": doc_id,
                "title": title,
                "chunks": [ch.text for ch in chunks],
                "chunk_indices": list(range(len(chunks))),
            })
        except Exception:
            continue
    return docs


def _create_queries(docs: list[dict], generated: dict[str, list[dict]] | None = None) -> list[dict]:
    """Create synthetic queries from document titles, plus generated ones.

    expected_chunk_index is stored as a GLOBAL chunk index (position in the
    flat all-chunks list, docs in order), or None for doc-level queries.
    """
    queries: list[dict] = []
    chunk_offset = 0
    for doc in docs:
        queries.append({
            "query": doc["title"],
            "expected_doc_id": doc["doc_id"],
            "query_type": "verbatim_title",
            "expected_chunk_index": None,
        })
        first_words = " ".join(doc["chunks"][0].split()[:20])
        if len(first_words) > 10:
            queries.append({
                "query": first_words,
                "expected_doc_id": doc["doc_id"],
                "query_type": "first_words",
                "expected_chunk_index": chunk_offset,
            })
        for gq in (generated or {}).get(doc["doc_id"], []):
            idx = gq.get("expected_chunk_index")
            queries.append({
                "query": gq["query"],
                "expected_doc_id": doc["doc_id"],
                "query_type": gq["type"],
                "expected_chunk_index": (chunk_offset + idx) if idx is not None else None,
            })
        chunk_offset += len(doc["chunks"])
    return queries


def _load_query_dataset(corpus: Path, queries_path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load a JSONL query dataset; return (docs, generated queries by doc_id).

    When a relpath appears more than once (e.g. after retrying errored docs),
    the last valid (non-error) record wins.
    """
    docs_by_rel: dict[str, dict] = {}
    generated_by_rel: dict[str, list[dict]] = {}
    skipped = 0
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("error") or not record.get("queries"):
            skipped += 1
            continue
        path = corpus / record["relpath"]
        try:
            chunks = list(split_file(path))
            if not chunks:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        doc_id = path.stem
        # The dataset record's title wins when present (v3 fixes slug
        # titles there); fall back to the chunker's heading path.
        title = (record.get("title")
                 or (chunks[0].heading_path[0] if chunks[0].heading_path
                     else doc_id))
        docs_by_rel[record["relpath"]] = {
            "doc_id": doc_id,
            "title": title,
            "chunks": [ch.text for ch in chunks],
            "chunk_indices": list(range(len(chunks))),
        }
        # Drop chunk references that no longer match the chunker output.
        gqs = []
        for q in record["queries"]:
            idx = q.get("expected_chunk_index")
            if idx is not None and not (0 <= idx < len(chunks)):
                q = {**q, "expected_chunk_index": None}
            gqs.append(q)
        generated_by_rel[record["relpath"]] = gqs
    if skipped:
        print(f"  ({skipped} dataset records skipped)")
    # Preserve dataset file order for reproducible chunk indexing.
    records = [json.loads(ln) for ln in queries_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rel_order = [r["relpath"] for r in records if not r.get("error") and r.get("queries")]
    seen = set()
    ordered_rels = [r for r in rel_order if not (r in seen or seen.add(r))]
    docs = [docs_by_rel[r] for r in ordered_rels if r in docs_by_rel]
    generated = {docs_by_rel[r]["doc_id"]: generated_by_rel[r] for r in ordered_rels if r in docs_by_rel}
    return docs, generated


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vectors."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)


def _apply_variant(emb: np.ndarray, variant: str) -> np.ndarray:
    """Apply a post-hoc transform to embeddings."""
    if variant == "full_fp32":
        return emb.astype(np.float32)
    if variant == "mrl_768":
        if emb.shape[1] <= MRL_DIM:
            return emb.astype(np.float32)
        truncated = emb[:, :MRL_DIM].astype(np.float32)
        norms = np.linalg.norm(truncated, axis=1, keepdims=True) + 1e-12
        return truncated / norms
    if variant == "int8":
        # Per-vector absmax scalar quantization; cosine on the int grid
        # (per-vector scale cancels under cosine normalization, leaving
        # only the rounding error, which is what we want to measure).
        scale = np.abs(emb).max(axis=1, keepdims=True) / 127.0 + 1e-12
        q = np.round(emb / scale).clip(-127, 127)
        return q.astype(np.float32)
    if variant == "binary":
        return np.sign(emb).astype(np.float32)
    raise ValueError(f"unknown variant {variant}")


def _bytes_per_vec(dims: int, variant: str) -> int:
    if variant == "binary":
        return dims // 8
    if variant == "int8":
        return dims
    if variant == "mrl_768":
        return min(dims, MRL_DIM) * 4
    return dims * 4


def _metrics_from_ranked_lists(
    ranked_doc_ids: list[list[str]],
    queries: list[dict],
) -> dict:
    """Compute Recall@k, MRR, NDCG from per-query ranked doc-id lists."""
    recall_at_k: dict = defaultdict(float)
    query_type_metrics: dict = defaultdict(lambda: defaultdict(float))
    reciprocal_ranks: list[float] = []
    dcg_scores: list[float] = []

    for ranked, q in zip(ranked_doc_ids, queries):
        expected_doc_id = q["expected_doc_id"]
        query_type = q["query_type"]

        rank = None
        for r, doc_id in enumerate(ranked[: max(TOP_K)], 1):
            if doc_id == expected_doc_id:
                rank = r
                break

        for k in TOP_K:
            if rank is not None and rank <= k:
                recall_at_k[k] += 1
                query_type_metrics[query_type][k] += 1

        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            dcg_scores.append(1.0 / np.log2(rank + 1))
        else:
            reciprocal_ranks.append(0.0)
            dcg_scores.append(0.0)

    for k in TOP_K:
        recall_at_k[k] /= len(queries)
    for qtype in query_type_metrics:
        type_count = sum(1 for q in queries if q["query_type"] == qtype)
        for k in TOP_K:
            query_type_metrics[qtype][k] /= type_count

    return {
        "recall_at_k": dict(recall_at_k),
        "mrr": float(np.mean(reciprocal_ranks)),
        "ndcg": float(np.mean(dcg_scores)),
        "query_type_metrics": {k: dict(v) for k, v in query_type_metrics.items()},
    }


def _chunk_level_metrics(
    ranked_chunk_indices: list[list[int]],
    queries: list[dict],
) -> dict | None:
    """Recall@k on the exact answering chunk (queries with chunk-level GT)."""
    targeted = [
        (ranked, q) for ranked, q in zip(ranked_chunk_indices, queries)
        if q.get("expected_chunk_index") is not None
    ]
    if not targeted:
        return None
    recall_at_k: dict = defaultdict(float)
    reciprocal_ranks: list[float] = []
    for ranked, q in targeted:
        target = q["expected_chunk_index"]
        rank = None
        for r, idx in enumerate(ranked[: max(TOP_K)], 1):
            if idx == target:
                rank = r
                break
        for k in TOP_K:
            if rank is not None and rank <= k:
                recall_at_k[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    n = len(targeted)
    return {
        "n": n,
        "recall_at_k": {k: v / n for k, v in recall_at_k.items()},
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def _compute_metrics(
    query_emb: np.ndarray,
    chunk_emb: np.ndarray,
    chunk_doc_ids: list[str],
    queries: list[dict],
) -> dict:
    """Compute Recall@k, MRR, NDCG for a given embedding pair."""
    similarities = _cosine_similarity(query_emb, chunk_emb)
    ranked_chunk_indices = [
        list(np.argsort(similarities[i])[::-1][: max(TOP_K)])
        for i in range(len(queries))
    ]
    ranked_doc_ids = [
        [chunk_doc_ids[idx] for idx in ranked] for ranked in ranked_chunk_indices
    ]
    metrics = _metrics_from_ranked_lists(ranked_doc_ids, queries)
    chunk_level = _chunk_level_metrics(ranked_chunk_indices, queries)
    if chunk_level:
        metrics["chunk_level"] = chunk_level
    return metrics


def _evaluate_model(
    model_name: str,
    docs: list[dict],
    queries: list[dict],
    device: str,
) -> dict:
    """Evaluate a single embedding model on retrieval quality across variants."""
    results: dict = {
        "model": model_name,
        "device": device,
        "documents": len(docs),
        "queries": len(queries),
        "errors": 0,
        "elapsed_seconds": 0.0,
        "vector_dim": 0,
        "variants": {},
    }

    try:
        from sentence_transformers import SentenceTransformer

        model_kwargs = {}
        if "jina" in model_name.lower():
            model_kwargs["default_task"] = "retrieval"
        model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )

        all_chunks: list[str] = []
        chunk_doc_ids: list[str] = []
        for doc in docs:
            for chunk_text in doc["chunks"]:
                all_chunks.append(chunk_text)
                chunk_doc_ids.append(doc["doc_id"])

        start_embed = time.perf_counter()
        chunk_embeddings = model.encode(
            all_chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        results["elapsed_seconds"] = time.perf_counter() - start_embed
        results["vector_dim"] = int(chunk_embeddings.shape[1])

        query_texts = [q["query"] for q in queries]
        if "jina" in model_name.lower():
            query_embeddings = model.encode(
                query_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                task="retrieval",
            )
        else:
            query_embeddings = model.encode(
                query_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        for variant in VARIANTS:
            # mrl_768 is a no-op for models already at or below MRL_DIM
            if variant == "mrl_768" and results["vector_dim"] <= MRL_DIM:
                continue
            q_var = _apply_variant(query_embeddings, variant)
            c_var = _apply_variant(chunk_embeddings, variant)
            metrics = _compute_metrics(q_var, c_var, chunk_doc_ids, queries)
            eff_dims = min(results["vector_dim"], MRL_DIM) if variant == "mrl_768" else results["vector_dim"]
            metrics["bytes_per_vec"] = _bytes_per_vec(results["vector_dim"], variant)
            metrics["eff_dims"] = eff_dims
            results["variants"][variant] = metrics

        del model
        gc.collect()

    except Exception as exc:
        results["errors"] = 1
        results["error_message"] = str(exc)

    return results


_FTS5_TOKENIZER = "unicode61 remove_diacritics 1"


def _bm25_match_query(query_text: str) -> str:
    """Build a safe FTS5 MATCH expression: OR of quoted word tokens."""
    tokens = re.findall(r"\w+", query_text, flags=re.UNICODE)
    return " OR ".join(f'"{t}"' for t in tokens)


def _evaluate_bm25(docs: list[dict], queries: list[dict]) -> dict:
    """Evaluate a plain BM25 full-text search baseline (SQLite FTS5)."""
    results: dict = {
        "model": "BM25 (SQLite FTS5)",
        "device": "-",
        "documents": len(docs),
        "queries": len(queries),
        "errors": 0,
        "elapsed_seconds": 0.0,
        "vector_dim": 0,
        "variants": {},
    }
    try:
        start = time.perf_counter()
        conn = sqlite3.connect(":memory:")
        conn.execute(
            f"CREATE VIRTUAL TABLE chunks USING fts5(text, doc_id UNINDEXED, "
            f"tokenize = '{_FTS5_TOKENIZER}')"
        )
        rows = [
            (chunk_text, doc["doc_id"])
            for doc in docs
            for chunk_text in doc["chunks"]
        ]
        conn.executemany("INSERT INTO chunks(text, doc_id) VALUES (?, ?)", rows)
        n_chunks = len(rows)

        ranked_doc_ids: list[list[str]] = []
        ranked_chunk_indices: list[list[int]] = []
        for q in queries:
            match = _bm25_match_query(q["query"])
            if not match:
                ranked_doc_ids.append([])
                ranked_chunk_indices.append([])
                continue
            cur = conn.execute(
                "SELECT rowid, doc_id FROM chunks WHERE chunks MATCH ? "
                "ORDER BY bm25(chunks) LIMIT ?",
                (match, max(TOP_K)),
            )
            fetched = cur.fetchall()
            ranked_doc_ids.append([r[1] for r in fetched])
            ranked_chunk_indices.append([r[0] - 1 for r in fetched])  # rowid → 0-based
        conn.close()
        results["elapsed_seconds"] = time.perf_counter() - start

        metrics = _metrics_from_ranked_lists(ranked_doc_ids, queries)
        chunk_level = _chunk_level_metrics(ranked_chunk_indices, queries)
        if chunk_level:
            metrics["chunk_level"] = chunk_level
        metrics["bytes_per_vec"] = "-"
        metrics["eff_dims"] = "-"
        metrics["n_chunks"] = n_chunks
        results["variants"]["bm25"] = metrics
    except Exception as exc:
        results["errors"] = 1
        results["error_message"] = str(exc)
    return results


def _format_report(results: list[dict], corpus: str, sample_size: int,
                   queries_path: str | None = None, queries: list[dict] | None = None) -> str:
    lines = [
        "# Evaluación de calidad de recuperación: modelos de embedding",
        "",
        f"Corpus: `{corpus}`",
        f"Muestra: **{sample_size}** documentos markdown (seed {SEED})",
    ]
    if queries_path:
        lines.append(f"Query set: `{queries_path}` (verbatim + tipos generados por LLM)")
    lines += [
        f"Fecha: {time.strftime('%Y-%m-%d')}",
        "",
        "## Tabla maestra: modelo × variante",
        "",
        "Variantes post-hoc sobre los mismos embeddings fp32: `full_fp32` (baseline), "
        "`mrl_768` (truncado Matryoshka a 768 dims), `int8` (cuantización escalar), "
        "`binary` (signo, 1 bit/dim).",
        "",
        "| Modelo | Variante | Dims ef. | Bytes/vec | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["errors"]:
            lines.append(f"| {r['model']} | - | - | - | ERROR | - | - | - | - |")
            continue
        for variant in [*VARIANTS, "bm25"]:
            v = r["variants"].get(variant)
            if v is None:
                continue
            lines.append(
                f"| {r['model']} | {variant} | {v['eff_dims']} | {v['bytes_per_vec']} | "
                f"{v['recall_at_k'][1]:.3f} | {v['recall_at_k'][5]:.3f} | "
                f"{v['recall_at_k'][10]:.3f} | {v['mrr']:.3f} | {v['ndcg']:.3f} |"
            )

    lines.extend([
        "",
        "## Ranking por MRR (full fp32; BM25 como baseline)",
        "",
    ])
    valid = []
    for r in results:
        if r["errors"]:
            continue
        v = r["variants"].get("full_fp32") or r["variants"].get("bm25")
        if v is not None:
            valid.append((r["model"], v))
    valid.sort(key=lambda x: x[1]["mrr"], reverse=True)
    for i, (name, v) in enumerate(valid, 1):
        lines.append(
            f"{i}. **{name}**: MRR={v['mrr']:.3f}, "
            f"Recall@1={v['recall_at_k'][1]:.3f}, Recall@5={v['recall_at_k'][5]:.3f}"
        )

    lines.extend([
        "",
        "## Desglose por tipo de query (full fp32 / bm25)",
        "",
        "| Modelo | Tipo | n | Recall@1 | Recall@5 | Recall@10 |",
        "|---|---|---|---|---|---|",
    ])
    type_counts: dict[str, int] = {}
    for q in (queries or []):
        type_counts[q["query_type"]] = type_counts.get(q["query_type"], 0) + 1
    for r in results:
        if r["errors"]:
            continue
        v = r["variants"].get("full_fp32") or r["variants"].get("bm25")
        if v is None:
            continue
        for qtype, ks in sorted(v["query_type_metrics"].items()):
            n = type_counts.get(qtype, 0) or "-"
            lines.append(
                f"| {r['model']} | {qtype} | {n} | {ks[1]:.3f} | {ks[5]:.3f} | {ks[10]:.3f} |"
            )

    chunk_rows = []
    for r in results:
        if r["errors"]:
            continue
        v = r["variants"].get("full_fp32") or r["variants"].get("bm25")
        cl = (v or {}).get("chunk_level")
        if cl:
            chunk_rows.append(
                f"| {r['model']} | {cl['n']} | {cl['recall_at_k'][1]:.3f} | "
                f"{cl['recall_at_k'][5]:.3f} | {cl['recall_at_k'][10]:.3f} | {cl['mrr']:.3f} |"
            )
    if chunk_rows:
        lines.extend([
            "",
            "## Chunk-level (queries con chunk esperado anotado)",
            "",
            "Recall si el chunk exacto que responde la query aparece en top-k.",
            "",
            "| Modelo | n | Recall@1 | Recall@5 | Recall@10 | MRR |",
            "|---|---|---|---|---|---|",
            *chunk_rows,
        ])

    lines.extend([
        "",
        "## Impacto de la cuantización (Δ MRR vs full fp32)",
        "",
        "| Modelo | int8 Δ | binary Δ | mrl_768 Δ |",
        "|---|---|---|---|",
    ])
    for r in results:
        if r["errors"] or "full_fp32" not in r["variants"]:
            continue
        base = r["variants"]["full_fp32"]["mrr"]
        def delta(variant: str) -> str:
            v = r["variants"].get(variant)
            if v is None:
                return "-"
            d = (v["mrr"] - base) * 100
            return f"{d:+.1f} pts"
        lines.append(
            f"| {r['model']} | {delta('int8')} | {delta('binary')} | {delta('mrl_768')} |"
        )

    lines.extend([
        "",
        "## Notas",
        "",
        "- Las queries sintéticas se generan a partir de títulos de documentos y primeras palabras de chunks.",
        "- Recall@k mide si el documento correcto aparece en los top-k chunks recuperados.",
        "- La cuantización int8 es escalar por-vector (absmax); la binaria es sign().",
        "- `mrl_768` solo aplica a modelos con más de 768 dims nativas.",
        "- BM25 usa SQLite FTS5 (`unicode61 remove_diacritics 1`), MATCH con OR de términos, "
        "sin stemming; el ranking usa `bm25(chunks)` de FTS5 sobre los mismos chunks y queries.",
        "- Muestra determinística (seed 42, archivos ordenados): reproducible en cualquier máquina.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help=f"Raíz del corpus markdown (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help=f"Número de documentos a muestrear (default: {SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--queries",
        default=None,
        help="JSONL generado por scripts/generate_queries.py; restringe el eval "
        "a esos documentos y agrega sus tipos de query",
    )
    args = parser.parse_args()

    root = Path(args.corpus)
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 1

    generated = None
    if args.queries:
        print(f"Loading query dataset from {args.queries}...")
        docs, generated = _load_query_dataset(root, Path(args.queries))
        print(f"  {len(docs)} documents from dataset")
    else:
        print("Getting documents...")
        docs = _get_documents(root, args.sample_size)
        print(f"  {len(docs)} documents")

    print("Creating queries...")
    queries = _create_queries(docs, generated)
    print(f"  {len(queries)} queries")

    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nEvaluating BM25 (SQLite FTS5) baseline...")
    bm25_result = _evaluate_bm25(docs, queries)
    if bm25_result["errors"]:
        print(f"  ERROR: {bm25_result.get('error_message', 'unknown')}")
    else:
        m = bm25_result["variants"]["bm25"]
        print(f"  bm25: MRR={m['mrr']:.3f} R@1={m['recall_at_k'][1]:.3f} "
              f"({m['n_chunks']:,} chunks)")

    results: list[dict] = []
    for model_name in MODELS:
        print(f"\nEvaluating {model_name}...")
        r = _evaluate_model(model_name, docs, queries, device)
        results.append(r)
        if r["errors"]:
            print(f"  ERROR: {r.get('error_message', 'unknown')}")
        else:
            base = r["variants"]["full_fp32"]
            print(
                f"  full: MRR={base['mrr']:.3f} R@1={base['recall_at_k'][1]:.3f} | "
                + " | ".join(
                    f"{v}: MRR={m['mrr']:.3f}"
                    for v, m in r["variants"].items()
                    if v != "full_fp32"
                )
            )

    results.append(bm25_result)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "retrieval_evaluation.md"
    report_path.write_text(
        _format_report(results, args.corpus, len(docs), args.queries, queries),
        encoding="utf-8",
    )
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
