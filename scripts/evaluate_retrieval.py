"""Evaluate embedding models on retrieval quality using synthetic DOF queries.

Run from repo root:
    python scripts/evaluate_retrieval.py

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

import gc
import os
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
SAMPLE_SIZE = 50  # files to use for retrieval evaluation
SEED = 42
TOP_K = [1, 5, 10]
MRL_DIM = 768

# Variants evaluated per model (post-hoc transforms of the same fp32 embeddings)
VARIANTS = ["full_fp32", "mrl_768", "int8", "binary"]


def _iter_md_files(root: Path) -> list[Path]:
    """Recursively list .md files, following directory symlinks."""
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if name.endswith(".md"):
                files.append(Path(dirpath) / name)
    return files


def _get_documents(n_files: int = SAMPLE_SIZE) -> list[dict]:
    """Get documents with their chunks and metadata."""
    seed(SEED)
    files = sorted(sample(sorted(_iter_md_files(Path("./dof_md"))), n_files))
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


def _create_queries(docs: list[dict]) -> list[dict]:
    """Create synthetic queries from document titles."""
    queries: list[dict] = []
    for doc in docs:
        queries.append({
            "query": doc["title"],
            "expected_doc_id": doc["doc_id"],
            "query_type": "title",
        })
        first_words = " ".join(doc["chunks"][0].split()[:20])
        if len(first_words) > 10:
            queries.append({
                "query": first_words,
                "expected_doc_id": doc["doc_id"],
                "query_type": "first_words",
            })
    return queries


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


def _compute_metrics(
    query_emb: np.ndarray,
    chunk_emb: np.ndarray,
    chunk_doc_ids: list[str],
    queries: list[dict],
) -> dict:
    """Compute Recall@k, MRR, NDCG for a given embedding pair."""
    similarities = _cosine_similarity(query_emb, chunk_emb)
    recall_at_k: dict = defaultdict(float)
    query_type_metrics: dict = defaultdict(lambda: defaultdict(float))
    reciprocal_ranks: list[float] = []
    dcg_scores: list[float] = []

    for i, q in enumerate(queries):
        expected_doc_id = q["expected_doc_id"]
        query_type = q["query_type"]
        sim_scores = similarities[i]
        top_indices = np.argsort(sim_scores)[::-1][: max(TOP_K)]

        rank = None
        for r, idx in enumerate(top_indices, 1):
            if chunk_doc_ids[idx] == expected_doc_id:
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


def _format_report(results: list[dict]) -> str:
    lines = [
        "# Evaluación de calidad de recuperación: modelos de embedding",
        "",
        f"Muestra: **{SAMPLE_SIZE}** documentos markdown de `./dof_md`",
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
        for variant in VARIANTS:
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
        "## Ranking por MRR (full fp32)",
        "",
    ])
    valid = [(r["model"], r["variants"]["full_fp32"]) for r in results if not r["errors"] and "full_fp32" in r["variants"]]
    valid.sort(key=lambda x: x[1]["mrr"], reverse=True)
    for i, (name, v) in enumerate(valid, 1):
        lines.append(
            f"{i}. **{name}**: MRR={v['mrr']:.3f}, "
            f"Recall@1={v['recall_at_k'][1]:.3f}, Recall@5={v['recall_at_k'][5]:.3f}"
        )

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
        "- `mrl_768` solo aplica a modelos con más de 768 dims nativas; jina-v5-text-nano (768 nativas) no tiene variante mrl.",
        "- Muestra determinística (seed 42, archivos ordenados): reproducible en cualquier máquina.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    root = Path("./dof_md")
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 1

    print("Getting documents...")
    docs = _get_documents(SAMPLE_SIZE)
    print(f"  {len(docs)} documents")

    print("Creating queries...")
    queries = _create_queries(docs)
    print(f"  {len(queries)} queries")

    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    models = [
        "perplexity-ai/pplx-embed-context-v1-0.6b",
        "perplexity-ai/pplx-embed-v1-0.6b",
        "nvidia/Nemotron-3-Embed-1B-BF16",
        "jinaai/jina-embeddings-v5-text-small",
        "jinaai/jina-embeddings-v5-text-nano",
        "Octen/Octen-Embedding-0.6B",
        "codefuse-ai/F2LLM-v2-1.7B",
        "microsoft/harrier-oss-v1-0.6b",
        "Qwen/Qwen3-Embedding-0.6B",
        "codefuse-ai/F2LLM-v2-0.6B",
    ]

    results: list[dict] = []
    for model_name in models:
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

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "retrieval_evaluation.md"
    report_path.write_text(_format_report(results), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
