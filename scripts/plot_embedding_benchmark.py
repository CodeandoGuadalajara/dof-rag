"""Generate publication-quality charts from the embedding benchmark reports.

Parses `reports/embedding_comparison.md` (speed) and
`reports/retrieval_evaluation.md` (quality + quantization deltas) as the
single source of truth, and renders:

1. `pareto.svg` — speed vs retrieval quality scatter (Pareto frontier).
2. `quantization.svg` — Δ MRR per variant (int8 / binary / mrl_768) per model.

Run from repo root:

    uv run python scripts/plot_embedding_benchmark.py [--out reports/figures]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path("reports")

# Params (B) from HF safetensors, for marker sizes
PARAMS_B = {
    "perplexity-ai/pplx-embed-context-v1-0.6b": 0.60,
    "perplexity-ai/pplx-embed-v1-0.6b": 0.60,
    "nvidia/Nemotron-3-Embed-1B-BF16": 1.14,
    "jinaai/jina-embeddings-v5-text-small": 0.60,
    "jinaai/jina-embeddings-v5-text-nano": 0.21,
    "Octen/Octen-Embedding-0.6B": 0.60,
    "codefuse-ai/F2LLM-v2-1.7B": 1.72,
    "microsoft/harrier-oss-v1-0.6b": 0.60,
    "Qwen/Qwen3-Embedding-0.6B": 0.60,
    "codefuse-ai/F2LLM-v2-0.6B": 0.60,
}

SHORT_NAMES = {
    "perplexity-ai/pplx-embed-context-v1-0.6b": "pplx-context",
    "perplexity-ai/pplx-embed-v1-0.6b": "pplx-v1",
    "nvidia/Nemotron-3-Embed-1B-BF16": "Nemotron-1B",
    "jinaai/jina-embeddings-v5-text-small": "jina-small",
    "jinaai/jina-embeddings-v5-text-nano": "jina-nano",
    "Octen/Octen-Embedding-0.6B": "Octen-0.6B",
    "codefuse-ai/F2LLM-v2-1.7B": "F2LLM-1.7B",
    "microsoft/harrier-oss-v1-0.6b": "harrier",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "codefuse-ai/F2LLM-v2-0.6B": "F2LLM-0.6B",
}

# Label offsets (dx, dy in points) to avoid overlaps, tuned by hand
LABEL_OFFSETS = {
    "F2LLM-1.7B": (10, 6),
    "pplx-v1": (10, 8),
    "pplx-context": (-12, 10),
    "F2LLM-0.6B": (10, 8),
    "jina-small": (-10, -16),
    "harrier": (-10, 8),
    "Octen-0.6B": (10, 2),
    "Qwen3-0.6B": (8, -18),
    "jina-nano": (-12, 8),
    "Nemotron-1B": (10, -16),
}


def parse_speed_report(path: Path) -> dict[str, dict]:
    """Parse `reports/embedding_comparison.md` -> {model: {dims, chunks_s}}."""
    out: dict[str, dict] = {}
    row_re = re.compile(
        r"^\| ([\w./-]+/[\w.-]+) \| \w+ \| ([\d,]+) \| [\d,]+ \| [\d.]+ \| ([\d.]+) \|"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        m = row_re.match(line)
        if m:
            out[m.group(1)] = {
                "dims": int(m.group(2).replace(",", "")),
                "chunks_s": float(m.group(3)),
            }
    return out


def parse_retrieval_report(path: Path) -> dict[str, dict]:
    """Parse `reports/retrieval_evaluation.md` -> {model: {mrr, r1, r5, deltas}}."""
    out: dict[str, dict] = {}
    fp32_re = re.compile(
        r"^\| ([\w./-]+/[\w.-]+) \| full_fp32 \| \d+ \| \d+ \| "
        r"([\d.]+) \| ([\d.]+) \| [\d.]+ \| ([\d.]+) \|"
    )
    delta_re = re.compile(
        r"^\| ([\w./-]+/[\w.-]+) \| ([+-][\d.]+) pts \| ([+-][\d.]+) pts \| ([+-][\d.]+|-)(?: pts)?\s*\|"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        m = fp32_re.match(line)
        if m:
            out[m.group(1)] = {
                "r1": float(m.group(2)),
                "r5": float(m.group(3)),
                "mrr": float(m.group(4)),
                "deltas": {},
            }
            continue
        m = delta_re.match(line)
        if m and m.group(1) in out:
            out[m.group(1)]["deltas"] = {
                "int8": float(m.group(2)) + 0.0,
                "binary": float(m.group(3)) + 0.0,
                "mrl_768": None if m.group(4) == "-" else float(m.group(4)) + 0.0,
            }
    return out


def plot_pareto(speed: dict, quality: dict, out_path: Path) -> None:
    models = [m for m in speed if m in quality]
    fig, ax = plt.subplots(figsize=(9, 6))

    xs = np.array([speed[m]["chunks_s"] for m in models])
    ys = np.array([quality[m]["mrr"] for m in models])
    sizes = np.array([PARAMS_B.get(m, 0.6) for m in models]) * 250 + 60
    dims = np.array([speed[m]["dims"] for m in models])

    sc = ax.scatter(
        xs, ys, s=sizes, c=dims, cmap="viridis", alpha=0.75,
        edgecolors="#333333", linewidths=0.8, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Dimensiones del vector", fontsize=10)

    for m, x, y in zip(models, xs, ys):
        name = SHORT_NAMES.get(m, m)
        dx, dy = LABEL_OFFSETS.get(name, (10, 6))
        ha = "right" if dx < 0 else "left"
        ax.annotate(
            name, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=9, ha=ha, color="#444444", zorder=4,
        )

    # Pareto frontier (maximize y while moving left = slower but better):
    # scan from fastest to slowest, keep running max of quality
    order = np.argsort(-xs)
    frontier_x, frontier_y = [], []
    best = -np.inf
    for i in order:
        if ys[i] > best:
            frontier_x.append(xs[i])
            frontier_y.append(ys[i])
            best = ys[i]
    ax.plot(frontier_x, frontier_y, "--", color="#999999", lw=1.2, zorder=2,
            label="Frontera de Pareto")

    ax.set_xscale("log")
    ax.set_xticks([1.7, 2, 3, 4, 5, 7, 11.3])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Velocidad de embedding (chunks/s, escala log)", fontsize=11)
    ax.set_ylabel("MRR (calidad de recuperación)", fontsize=11)
    ax.set_title("Benchmark de embeddings DOF: velocidad vs calidad\n"
                 "(tamaño del punto = parámetros; MacBook Pro M3, MPS)", fontsize=12)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(xs.min() * 0.75, xs.max() * 1.45)
    ax.set_ylim(ys.min() - 0.02, ys.max() + 0.02)

    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200)
    plt.close(fig)


def plot_quantization(quality: dict, out_path: Path) -> None:
    models = [m for m in quality if quality[m]["deltas"]]
    models.sort(key=lambda m: quality[m]["mrr"], reverse=True)
    variants = ["int8", "binary", "mrl_768"]
    colors = {"int8": "#2ca02c", "binary": "#d62728", "mrl_768": "#1f77b4"}

    y_pos = np.arange(len(models))
    height = 0.25
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, variant in enumerate(variants):
        vals = []
        for m in models:
            d = quality[m]["deltas"].get(variant)
            vals.append(0.0 if d is None else d)
        bars = ax.barh(
            y_pos + (i - 1) * height, vals, height,
            label=variant, color=colors[variant], alpha=0.85, zorder=3,
        )
        for bar, m, v in zip(bars, models, vals):
            if quality[m]["deltas"].get(variant) is None:
                continue
            v = 0.0 if v == 0 else v  # normalize -0.0 for display
            ax.text(
                bar.get_width() + (0.08 if v >= 0 else -0.08),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color="#444444",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT_NAMES.get(m, m) for m in models], fontsize=10)
    ax.invert_yaxis()  # best MRR on top
    ax.axvline(0, color="#333333", lw=0.8, zorder=2)
    ax.set_xlabel("Δ MRR vs full fp32 (puntos ×100)", fontsize=11)
    ax.set_title("Impacto de cuantización y truncado en calidad de recuperación\n"
                 "(int8 = cuantización escalar, binary = 1 bit/dim, mrl_768 = Matryoshka a 768)",
                 fontsize=12)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(-5.5, 1.5)

    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "figures",
                        help="Output directory (default: reports/figures)")
    args = parser.parse_args()

    speed = parse_speed_report(REPORT_DIR / "embedding_comparison.md")
    quality = parse_retrieval_report(REPORT_DIR / "retrieval_evaluation.md")
    print(f"Parsed {len(speed)} speed rows, {len(quality)} quality rows")

    args.out.mkdir(parents=True, exist_ok=True)
    plot_pareto(speed, quality, args.out / "pareto.svg")
    plot_quantization(quality, args.out / "quantization.svg")
    print(f"Charts written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
