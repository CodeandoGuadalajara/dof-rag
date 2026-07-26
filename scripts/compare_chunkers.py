"""Compare custom DOF chunker against Chonkie chunkers.

Run from repo root:
    python scripts/compare_chunkers.py

Outputs a Markdown report to `reports/chunker_comparison.md`.
"""
from __future__ import annotations

import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from random import sample, seed

from chonkie import RecursiveChunker, TableChunker
from chonkie.tokenizer import Tokenizer
from chonkie.types import RecursiveLevel, RecursiveRules


class PPLXTokenizer(Tokenizer):
    """Chonkie-compatible wrapper around the pplx-embed-context-v1 tokenizer."""

    def __init__(self, model_name: str = "perplexity-ai/pplx-embed-context-v1-0.6b"):
        super().__init__()
        from transformers import AutoTokenizer

        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    def __repr__(self) -> str:
        return f"PPLXTokenizer(model={self._model_name!r})"

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def tokenize(self, text: str) -> list[str]:
        return self._tokenizer.tokenize(text)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_poc.chunker import (  # noqa: E402
    _count_tokens,
    classify,
    split_file,
)
from rag_poc.config import MAX_TOKENS  # noqa: E402

REPORT_DIR = Path("reports")
SAMPLE_SIZE = 1_000
SEED = 42


def _sample_files(root: Path, n: int) -> list[Path]:
    seed(SEED)
    all_files = list(root.rglob("*.md"))
    if len(all_files) <= n:
        return all_files
    return sorted(sample(all_files, n))


def _run_custom(files: list[Path]) -> dict:
    results: dict = {
        "chunks": [],
        "tokens": [],
        "errors": [],
        "files_with_oversized": [],
        "pattern_counts": defaultdict(int),
        "pattern_chunks": defaultdict(list),
    }
    start = time.perf_counter()

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        size = f.stat().st_size
        pattern = classify(text, size)
        try:
            chunks = split_file(f)
        except Exception as exc:
            results["errors"].append((str(f), str(exc)))
            continue

        tokens = [_count_tokens(c.text) for c in chunks]
        results["chunks"].append(len(chunks))
        results["tokens"].extend(tokens)
        results["pattern_counts"][pattern.value] += 1
        results["pattern_chunks"][pattern.value].append(len(chunks))
        results["files_with_oversized"].append(
            1 if any(t > MAX_TOKENS * 1.10 for t in tokens) else 0
        )

    elapsed = time.perf_counter() - start
    return _summarize(results, elapsed, "custom")


def _run_chonkie_recursive(files: list[Path]) -> dict:
    results: dict = {
        "chunks": [],
        "tokens": [],
        "errors": [],
        "files_with_oversized": [],
    }
    chunker = RecursiveChunker(
        tokenizer=PPLXTokenizer(),
        chunk_size=MAX_TOKENS,
        rules=RecursiveRules.from_recipe("markdown"),
    )
    start = time.perf_counter()

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        try:
            chunks = chunker(text)
        except Exception as exc:
            results["errors"].append((str(f), str(exc)))
            continue

        tokens = [c.token_count for c in chunks]
        results["chunks"].append(len(chunks))
        results["tokens"].extend(tokens)
        results["files_with_oversized"].append(
            1 if any(t > MAX_TOKENS * 1.10 for t in tokens) else 0
        )

    elapsed = time.perf_counter() - start
    return _summarize(results, elapsed, "chonkie_recursive")


def _run_chonkie_h2(files: list[Path]) -> dict:
    """Chonkie RecursiveChunker with H2 as the primary delimiter."""
    results: dict = {
        "chunks": [],
        "tokens": [],
        "errors": [],
        "files_with_oversized": [],
    }
    rules = RecursiveRules(levels=[
        RecursiveLevel(delimiters=["\n## "]),
        RecursiveLevel(delimiters=["\n### "]),
        RecursiveLevel(delimiters=["\n\n"]),
        RecursiveLevel(delimiters=[". ", "! ", "? "]),
    ])
    chunker = RecursiveChunker(
        tokenizer=PPLXTokenizer(),
        chunk_size=MAX_TOKENS,
        rules=rules,
    )
    start = time.perf_counter()

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        try:
            chunks = chunker(text)
        except Exception as exc:
            results["errors"].append((str(f), str(exc)))
            continue

        tokens = [c.token_count for c in chunks]
        results["chunks"].append(len(chunks))
        results["tokens"].extend(tokens)
        results["files_with_oversized"].append(
            1 if any(t > MAX_TOKENS * 1.10 for t in tokens) else 0
        )

    elapsed = time.perf_counter() - start
    return _summarize(results, elapsed, "chonkie_h2")


def _run_chonkie_table(files: list[Path]) -> dict:
    """Chonkie TableChunker on the whole document."""
    results: dict = {
        "chunks": [],
        "tokens": [],
        "errors": [],
        "files_with_oversized": [],
    }
    chunker = TableChunker(
        tokenizer=PPLXTokenizer(),
        chunk_size=MAX_TOKENS,
    )
    start = time.perf_counter()

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        try:
            chunks = chunker(text)
        except Exception as exc:
            results["errors"].append((str(f), str(exc)))
            continue

        tokens = [c.token_count for c in chunks]
        results["chunks"].append(len(chunks))
        results["tokens"].extend(tokens)
        results["files_with_oversized"].append(
            1 if any(t > MAX_TOKENS * 1.10 for t in tokens) else 0
        )

    elapsed = time.perf_counter() - start
    return _summarize(results, elapsed, "chonkie_table")


def _summarize(results: dict, elapsed: float, label: str) -> dict:
    chunks_per_file = results.get("chunks", [])
    tokens = results.get("tokens", [])
    oversized = results.get("files_with_oversized", [])

    return {
        "label": label,
        "files": len(chunks_per_file),
        "errors": len(results.get("errors", [])),
        "total_chunks": sum(chunks_per_file),
        "chunks_per_file": {
            "mean": _safe_mean(chunks_per_file),
            "median": _safe_median(chunks_per_file),
            "max": max(chunks_per_file) if chunks_per_file else 0,
        },
        "tokens_per_chunk": {
            "mean": _safe_mean(tokens),
            "median": _safe_median(tokens),
            "max": max(tokens) if tokens else 0,
            "p95": _safe_percentile(tokens, 95) if tokens else 0,
        },
        "oversized_files": sum(oversized),
        "oversized_pct": sum(oversized) / len(oversized) * 100 if oversized else 0,
        "elapsed_seconds": elapsed,
        "chunks_per_second": sum(chunks_per_file) / elapsed if elapsed else 0,
        "pattern_counts": dict(results.get("pattern_counts", {})),
        "pattern_chunks": {k: _safe_mean(v) for k, v in results.get("pattern_chunks", {}).items()},
    }


def _safe_mean(values: list) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_median(values: list) -> float:
    return statistics.median(values) if values else 0.0


def _safe_percentile(values: list, percentile: int) -> float:
    s = sorted(values)
    idx = int(len(s) * percentile / 100)
    return s[min(idx, len(s) - 1)]


def _format_report(custom: dict, recursive: dict, h2: dict, table: dict) -> str:
    lines = [
        "# Comparación: chunker custom vs Chonkie chunkers",
        "",
        f"Muestra: **{SAMPLE_SIZE:,}** archivos markdown de `./dof_md`",
        f"Límite de tokens por chunk: **{MAX_TOKENS}**",
        "",
        "## Resumen general",
        "",
        "| Chunker | Archivos | Errores | Chunks total | Chunks/archivo (med) | Tiempo (s) | Chunks/s |",
        "|---|---|---|---|---|---|---|",
        f"| Custom | {custom['files']:,} | {custom['errors']} | {custom['total_chunks']:,} | {custom['chunks_per_file']['median']:.1f} | {custom['elapsed_seconds']:.2f} | {custom['chunks_per_second']:.1f} |",
        f"| Chonkie Recursive | {recursive['files']:,} | {recursive['errors']} | {recursive['total_chunks']:,} | {recursive['chunks_per_file']['median']:.1f} | {recursive['elapsed_seconds']:.2f} | {recursive['chunks_per_second']:.1f} |",
        f"| Chonkie H2 | {h2['files']:,} | {h2['errors']} | {h2['total_chunks']:,} | {h2['chunks_per_file']['median']:.1f} | {h2['elapsed_seconds']:.2f} | {h2['chunks_per_second']:.1f} |",
        f"| Chonkie Table | {table['files']:,} | {table['errors']} | {table['total_chunks']:,} | {table['chunks_per_file']['median']:.1f} | {table['elapsed_seconds']:.2f} | {table['chunks_per_second']:.1f} |",
        "",
        "## Tokens por chunk",
        "",
        "| Chunker | Media | Mediana | P95 | Máx | Archivos con chunks >10% over max |",
        "|---|---|---|---|---|---|",
        f"| Custom | {custom['tokens_per_chunk']['mean']:.1f} | {custom['tokens_per_chunk']['median']:.1f} | {custom['tokens_per_chunk']['p95']:.1f} | {custom['tokens_per_chunk']['max']:,} | {custom['oversized_files']} ({custom['oversized_pct']:.1f}%) |",
        f"| Chonkie Recursive | {recursive['tokens_per_chunk']['mean']:.1f} | {recursive['tokens_per_chunk']['median']:.1f} | {recursive['tokens_per_chunk']['p95']:.1f} | {recursive['tokens_per_chunk']['max']:,} | {recursive['oversized_files']} ({recursive['oversized_pct']:.1f}%) |",
        f"| Chonkie H2 | {h2['tokens_per_chunk']['mean']:.1f} | {h2['tokens_per_chunk']['median']:.1f} | {h2['tokens_per_chunk']['p95']:.1f} | {h2['tokens_per_chunk']['max']:,} | {h2['oversized_files']} ({h2['oversized_pct']:.1f}%) |",
        f"| Chonkie Table | {table['tokens_per_chunk']['mean']:.1f} | {table['tokens_per_chunk']['median']:.1f} | {table['tokens_per_chunk']['p95']:.1f} | {table['tokens_per_chunk']['max']:,} | {table['oversized_files']} ({table['oversized_pct']:.1f}%) |",
        "",
        "## Distribución de patrones (chunker custom)",
        "",
        "| Patrón | Archivos | Chunks/archivo (media) |",
        "|---|---|---|",
    ]
    for pattern, count in sorted(custom["pattern_counts"].items()):
        mean_chunks = custom["pattern_chunks"].get(pattern, 0)
        lines.append(f"| {pattern} | {count:,} | {mean_chunks:.1f} |")

    lines.extend([
        "",
        "## Observaciones",
        "",
        "- El chunker custom clasifica el documento antes de dividir; Chonkie RecursiveChunker aplica una regla markdown genérica.",
        "- Chonkie H2 usa H2 como delimitador principal, lo que lo hace más comparable con el custom en documentos compuestos.",
        "- Chonkie TableChunker detecta tablas en el documento y repite automáticamente el encabezado del documento en cada chunk.",
        "- El contador de tokens es el mismo para los cuatro (tokenizer de `pplx-embed-context-v1-0.6b`) para hacer la comparación justa.",
        "",
        "## Conclusión provisional",
        "",
        f"- **Custom**: chunks más pequeños y granulares (mediana {custom['tokens_per_chunk']['median']:.0f} tokens), {custom['oversized_files']} archivos con chunks que exceden el límite. Máximo observado: {custom['tokens_per_chunk']['max']:,} tokens.",
        f"- **Chonkie Recursive**: chunks más grandes (mediana {recursive['tokens_per_chunk']['median']:.0f} tokens), máx {recursive['tokens_per_chunk']['max']:,}, {recursive['errors']} errores.",
        f"- **Chonkie H2**: {h2['total_chunks']:,} chunks (mediana {h2['tokens_per_chunk']['median']:.0f} tokens), pero {h2['oversized_files']} archivos ({h2['oversized_pct']:.1f}%) producen chunks que exceden el límite; máx {h2['tokens_per_chunk']['max']:,} tokens.",
        f"- **Chonkie Table**: {table['total_chunks']:,} chunks (mediana {table['tokens_per_chunk']['median']:.0f} tokens), pero {table['oversized_files']} archivos ({table['oversized_pct']:.1f}%) producen chunks enormes; máx {table['tokens_per_chunk']['max']:,} tokens. TableChunker no respeta el límite de tokens en documentos con tablas grandes y genera chunks inmanejables para recuperación.",
        "- El custom es más adecuado para el DOF porque respeta la estructura documental (H2s, tablas, negritas) y genera chunks recuperables; entre las opciones de Chonkie, RecursiveChunker es la más estable.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    root = Path("./dof_md")
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 1

    files = _sample_files(root, SAMPLE_SIZE)
    print(f"Sampled {len(files):,} files from {root}")

    print("Running custom chunker...")
    custom = _run_custom(files)
    print(f"  {custom['total_chunks']:,} chunks in {custom['elapsed_seconds']:.2f}s")

    print("Running Chonkie RecursiveChunker...")
    recursive = _run_chonkie_recursive(files)
    print(f"  {recursive['total_chunks']:,} chunks in {recursive['elapsed_seconds']:.2f}s")

    print("Running Chonkie RecursiveChunker (H2 primary)...")
    h2 = _run_chonkie_h2(files)
    print(f"  {h2['total_chunks']:,} chunks in {h2['elapsed_seconds']:.2f}s")

    print("Running Chonkie TableChunker...")
    table = _run_chonkie_table(files)
    print(f"  {table['total_chunks']:,} chunks in {table['elapsed_seconds']:.2f}s")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "chunker_comparison.md"
    report_path.write_text(_format_report(custom, recursive, h2, table), encoding="utf-8")
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
