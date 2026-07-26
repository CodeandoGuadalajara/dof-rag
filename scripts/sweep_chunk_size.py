"""Sweep MAX_TOKENS and measure chunker behavior."""
from __future__ import annotations

import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from random import sample, seed

sys.path.insert(0, str(Path(__file__).parent.parent))

seed(42)
files = sorted(sample(list(Path("./dof_md").rglob("*.md")), 1000))


def run(max_tokens: int, h2_max: int) -> dict:
    import importlib

    import rag_poc.config as cfg

    cfg.MAX_TOKENS = max_tokens
    cfg.H2_MAX_TOKENS = h2_max
    cfg.OVERLAP_TOKENS = max(10, max_tokens // 20)  # 5% overlap

    import rag_poc.chunker as chunker_mod  # noqa: E402

    importlib.reload(chunker_mod)
    from rag_poc.chunker import split_file, _count_tokens, classify  # noqa: E402, I001

    results = {
        "chunks": [],
        "tokens": [],
        "files": 0,
        "errors": 0,
        "oversized": 0,
        "pattern_counts": defaultdict(int),
    }
    start = time.perf_counter()
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        try:
            chunks = split_file(f)
        except Exception:
            results["errors"] += 1
            continue
        tokens = [_count_tokens(c.text) for c in chunks]
        results["chunks"].append(len(chunks))
        results["tokens"].extend(tokens)
        results["files"] += 1
        results["oversized"] += sum(1 for t in tokens if t > int(max_tokens * 1.10))
        results["pattern_counts"][classify(text, f.stat().st_size).value] += 1
    results["elapsed"] = time.perf_counter() - start
    return results


def summarize(label: int, r: dict) -> dict:
    tokens = r["tokens"]
    chunks = r["chunks"]
    s = sorted(tokens)
    n = len(tokens)

    def p(x):
        return s[min(int(n * x / 100), n - 1)]

    return {
        "max_tokens": label,
        "files": r["files"],
        "errors": r["errors"],
        "total_chunks": sum(chunks),
        "chunks_per_file": statistics.median(chunks) if chunks else 0,
        "min": min(tokens) if tokens else 0,
        "median": statistics.median(tokens) if tokens else 0,
        "mean": statistics.mean(tokens) if tokens else 0,
        "p90": p(90),
        "p95": p(95),
        "p99": p(99),
        "max": max(tokens) if tokens else 0,
        "oversized_count": r["oversized"],
        "oversized_pct": r["oversized"] / len(tokens) * 100 if tokens else 0,
        "elapsed": r["elapsed"],
    }


def main() -> int:
    print(
        "| MAX_TOKENS | H2_MAX_TOKENS | Chunks | Chunks/file | Median | Mean | P95 | Max | Oversized | Time(s) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for max_tokens in [400, 800, 1200, 1600, 2000, 2500]:
        h2_max = int(max_tokens * 1.10)
        r = run(max_tokens, h2_max)
        s = summarize(max_tokens, r)
        print(
            f"| {s['max_tokens']} | {h2_max} | "
            f"{s['total_chunks']:,} | {s['chunks_per_file']:.1f} | "
            f"{s['median']:.0f} | {s['mean']:.0f} | {s['p95']:.0f} | {s['max']:.0f} | "
            f"{s['oversized_count']} ({s['oversized_pct']:.1f}%) | {s['elapsed']:.1f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
