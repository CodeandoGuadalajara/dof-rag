"""Evaluate embedding models on retrieval quality using synthetic DOF queries.

Run from repo root:
    python scripts/evaluate_retrieval.py

For each embedding model:
1. Embed a sample of document chunks
2. Generate synthetic queries from document titles/headings
3. Measure Recall@k, MRR, NDCG

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
            # Use filename as document ID
            doc_id = f.stem
            # Extract title from first chunk or filename
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
        # Query 1: use the document title
        queries.append({
            "query": doc["title"],
            "expected_doc_id": doc["doc_id"],
            "query_type": "title",
        })
        # Query 2: use first 20 words of first chunk (simulating a user query)
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
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def _evaluate_model(
    model_name: str,
    docs: list[dict],
    queries: list[dict],
    device: str,
) -> dict:
    """Evaluate a single embedding model on retrieval quality."""
    results: dict = {
        "model": model_name,
        "device": device,
        "documents": len(docs),
        "queries": len(queries),
        "errors": 0,
        "elapsed_seconds": 0.0,
        "recall_at_k": defaultdict(float),
        "mrr": 0.0,
        "ndcg": 0.0,
        "query_type_metrics": defaultdict(lambda: defaultdict(float)),
    }

    try:
        from sentence_transformers import SentenceTransformer

        # Load model
        model_kwargs = {}
        if "jina" in model_name.lower():
            model_kwargs["default_task"] = "retrieval"
        model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )

        # Embed all chunks from all documents
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

        # Embed queries
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

        # Compute similarities
        similarities = _cosine_similarity(query_embeddings, chunk_embeddings)

        # Evaluate each query
        reciprocal_ranks: list[float] = []
        dcg_scores: list[float] = []

        for i, q in enumerate(queries):
            expected_doc_id = q["expected_doc_id"]
            query_type = q["query_type"]

            # Get top-k chunk indices by similarity
            sim_scores = similarities[i]
            top_indices = np.argsort(sim_scores)[::-1][: max(TOP_K)]

            # Find the rank of the first chunk from the expected document
            rank = None
            for r, idx in enumerate(top_indices, 1):
                if chunk_doc_ids[idx] == expected_doc_id:
                    rank = r
                    break

            # Compute metrics
            for k in TOP_K:
                if rank is not None and rank <= k:
                    results["recall_at_k"][k] += 1
                    results["query_type_metrics"][query_type][k] += 1

            if rank is not None:
                reciprocal_ranks.append(1.0 / rank)
                # NDCG: binary relevance (1 if correct doc, 0 otherwise)
                dcg = 1.0 / np.log2(rank + 1)
                dcg_scores.append(dcg)
            else:
                reciprocal_ranks.append(0.0)
                dcg_scores.append(0.0)

        # Average metrics
        results["mrr"] = float(np.mean(reciprocal_ranks))
        results["ndcg"] = float(np.mean(dcg_scores))
        for k in TOP_K:
            results["recall_at_k"][k] /= len(queries)

        # Per-query-type metrics
        for qtype in results["query_type_metrics"]:
            type_queries = [q for q in queries if q["query_type"] == qtype]
            for k in TOP_K:
                results["query_type_metrics"][qtype][k] /= len(type_queries)

        # Cleanup
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
        "## Resumen general",
        "",
        "| Modelo | Dim | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["errors"]:
            lines.append(
                f"| {r['model']} | - | ERROR | - | - | - | - |"
            )
        else:
            lines.append(
                f"| {r['model']} | {r.get('vector_dim', '-')} | "
                f"{r['recall_at_k'][1]:.3f} | {r['recall_at_k'][5]:.3f} | "
                f"{r['recall_at_k'][10]:.3f} | {r['mrr']:.3f} | {r['ndcg']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Métricas por tipo de query",
            "",
            "| Modelo | Query type | Recall@1 | Recall@5 | Recall@10 |",
            "|---|---|---|---|---|",
        ]
    )
    for r in results:
        if not r["errors"]:
            for qtype, metrics in r["query_type_metrics"].items():
                lines.append(
                    f"| {r['model']} | {qtype} | "
                    f"{metrics[1]:.3f} | {metrics[5]:.3f} | {metrics[10]:.3f} |"
                )

    lines.extend(
        [
            "",
            "## Notas",
            "",
            "- Las queries sintéticas se generan a partir de títulos de documentos y primeras palabras de chunks.",
            "- Recall@k mide si el documento correcto aparece en los top-k chunks recuperados.",
            "- MRR (Mean Reciprocal Rank) y NDCG son métricas estándar de ranking.",
            "- Todos los modelos usan los mismos documentos y queries.",
            "",
            "## Conclusión provisional",
            "",
        ]
    )

    # Rank models by MRR
    valid_results = [r for r in results if not r["errors"]]
    if valid_results:
        sorted_by_mrr = sorted(valid_results, key=lambda x: x["mrr"], reverse=True)
        lines.append("**Ranking por MRR:**")
        for i, r in enumerate(sorted_by_mrr, 1):
            lines.append(
                f"{i}. **{r['model']}**: MRR={r['mrr']:.3f}, "
                f"Recall@1={r['recall_at_k'][1]:.3f}, "
                f"Recall@5={r['recall_at_k'][5]:.3f}"
            )
        lines.append("")

    lines.extend(
        [
            "## Siguientes pasos",
            "",
            "- Evaluar con queries reales de usuarios.",
            "- Probar late chunking con pplx-embed-context-v1.",
            "- Medir latencia de búsqueda vectorial con sqlite-vec.",
            "- Optimizar hiperparámetros (top-k, umbral de similitud).",
            "",
        ]
    )
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

    # Detect device
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
    ]

    results: list[dict] = []
    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        r = _evaluate_model(model_name, docs, queries, device)
        results.append(r)
        if r["errors"]:
            print(f"  ERROR: {r.get('error_message', 'unknown')}")
        else:
            print(
                f"  Recall@1={r['recall_at_k'][1]:.3f}, "
                f"Recall@5={r['recall_at_k'][5]:.3f}, "
                f"MRR={r['mrr']:.3f}"
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "retrieval_evaluation.md"
    report_path.write_text(_format_report(results), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
