"""Compare embedding models for DOF RAG on speed, memory, and dimensions.

Run from repo root:
    python scripts/compare_embeddings.py

Outputs a Markdown report to `reports/embedding_comparison.md`.

Models tested:
- perplexity-ai/pplx-embed-context-v1-0.6b (contextual, int8)
- perplexity-ai/pplx-embed-v1-0.6b (non-contextual)
- nvidia/Nemotron-3-Embed-1B-BF16
- jinaai/jina-embeddings-v5-text-small
- jinaai/jina-embeddings-v5-text-nano
- Octen/Octen-Embedding-0.6B

For MacBook Pro M3 (36GB RAM):
- Uses MPS (Metal Performance Shaders) if available
- Falls back to CPU otherwise
- Reports peak memory usage per model
"""
from __future__ import annotations

import gc
import os
import sys
import time
import warnings
from pathlib import Path
from random import sample, seed

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_poc.chunker import split_file  # noqa: E402

REPORT_DIR = Path("reports")
SAMPLE_SIZE = 100  # files to chunk for embedding test
SEED = 42


def _iter_md_files(root: Path) -> list[Path]:
    """Recursively list .md files, following directory symlinks."""
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if name.endswith(".md"):
                files.append(Path(dirpath) / name)
    return files


def _get_sample_chunks(n_files: int = SAMPLE_SIZE) -> list[str]:
    """Get a sample of chunks from the corpus."""
    seed(SEED)
    files = sorted(sample(sorted(_iter_md_files(Path("./dof_md"))), n_files))
    chunks: list[str] = []
    for f in files:
        try:
            for ch in split_file(f):
                chunks.append(ch.text)
        except Exception:
            continue
    return chunks


def _measure_memory() -> float:
    """Return current process memory in MB (approximate)."""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def _run_model(model_name: str, chunks: list[str], device: str) -> dict:
    """Run a single embedding model over the sample chunks."""
    results: dict = {
        "model": model_name,
        "device": device,
        "chunks": len(chunks),
        "errors": 0,
        "peak_memory_mb": 0.0,
        "elapsed_seconds": 0.0,
        "chunks_per_second": 0.0,
        "vector_dim": 0,
        "model_size_mb": 0.0,
    }

    try:
        from sentence_transformers import SentenceTransformer

        # Measure baseline memory
        gc.collect()
        baseline_memory = _measure_memory()

        # Load model
        start_load = time.perf_counter()
        # Jina models require a default_task
        model_kwargs = {}
        if "jina" in model_name.lower():
            model_kwargs["default_task"] = "retrieval"
        model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        load_time = time.perf_counter() - start_load
        results["load_time_seconds"] = load_time

        # Measure memory after load
        gc.collect()
        loaded_memory = _measure_memory()
        results["model_size_mb"] = loaded_memory - baseline_memory

        # Get vector dimension
        test_emb = model.encode(["test"], convert_to_numpy=True)
        results["vector_dim"] = test_emb.shape[1] if len(test_emb.shape) > 1 else test_emb.shape[0]

        # Embed all chunks
        start_embed = time.perf_counter()
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            # Jina models require a task parameter
            if "jina" in model_name.lower():
                _ = model.encode(batch, convert_to_numpy=True, task="retrieval")
            else:
                _ = model.encode(batch, convert_to_numpy=True)
            if i % (batch_size * 10) == 0:
                gc.collect()
                current_memory = _measure_memory()
                results["peak_memory_mb"] = max(
                    results["peak_memory_mb"], current_memory - baseline_memory
                )

        results["elapsed_seconds"] = time.perf_counter() - start_embed
        results["chunks_per_second"] = len(chunks) / results["elapsed_seconds"] if results["elapsed_seconds"] > 0 else 0

        # Cleanup
        del model
        gc.collect()

    except Exception as exc:
        results["errors"] = 1
        results["error_message"] = str(exc)

    return results


def _format_report(results: list[dict]) -> str:
    lines = [
        "# Comparación: modelos de embedding para DOF RAG",
        "",
        f"Muestra: **{SAMPLE_SIZE}** archivos markdown de `./dof_md`",
        f"Fecha: {time.strftime('%Y-%m-%d')}",
        "",
        "## Resumen general",
        "",
        "| Modelo | Device | Dim | Chunks | Tiempo (s) | Chunks/s | Memoria pico (MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["errors"]:
            lines.append(
                f"| {r['model']} | {r['device']} | - | {r['chunks']:,} | ERROR | - | - |"
            )
        else:
            lines.append(
                f"| {r['model']} | {r['device']} | {r['vector_dim']:,} | "
                f"{r['chunks']:,} | {r['elapsed_seconds']:.2f} | "
                f"{r['chunks_per_second']:.1f} | {r['peak_memory_mb']:.0f} |"
            )

    lines.extend(
        [
            "",
            "## Notas",
            "",
            "- Todos los modelos usan el mismo tokenizer y los mismos chunks de entrada.",
            "- La memoria pico es aproximada (RSS del proceso).",
            "- En MacBook Pro M3, `mps` usa Metal Performance Shaders; `cpu` es fallback.",
            "- `pplx-embed-context-v1` es contextual: los embeddings de chunks del mismo documento se benefician de verse juntos.",
            "",
            "## Conclusión provisional",
            "",
        ]
    )

    # Add per-model observations
    for r in results:
        if not r["errors"]:
            lines.append(
                f"- **{r['model']}**: {r['vector_dim']:,} dims, "
                f"{r['chunks_per_second']:.1f} chunks/s, "
                f"{r['peak_memory_mb']:.0f} MB pico."
            )
        else:
            lines.append(f"- **{r['model']}**: ERROR - {r.get('error_message', 'unknown')}")

    lines.extend(
        [
            "",
            "## Siguientes pasos",
            "",
            "- Evaluar calidad de recuperación con queries sintéticas del DOF.",
            "- Probar quantización (int8, int4) para reducir memoria en modelos grandes.",
            "- Medir latencia de búsqueda vectorial con sqlite-vec.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    root = Path("./dof_md")
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 1

    print("Getting sample chunks...")
    chunks = _get_sample_chunks(SAMPLE_SIZE)
    print(f"  {len(chunks):,} chunks from {SAMPLE_SIZE} files")

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
        "codefuse-ai/F2LLM-v2-1.7B",
        "microsoft/harrier-oss-v1-0.6b",
        "Qwen/Qwen3-Embedding-0.6B",
        "codefuse-ai/F2LLM-v2-0.6B",
    ]

    results: list[dict] = []
    for model_name in models:
        print(f"\nRunning {model_name}...")
        r = _run_model(model_name, chunks, device)
        results.append(r)
        if r["errors"]:
            print(f"  ERROR: {r.get('error_message', 'unknown')}")
        else:
            print(
                f"  {r['vector_dim']:,} dims, "
                f"{r['chunks_per_second']:.1f} chunks/s, "
                f"{r['peak_memory_mb']:.0f} MB"
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "embedding_comparison.md"
    report_path.write_text(_format_report(results), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
