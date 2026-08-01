"""Generate the round 2/3 Pareto chart for the blog post.

4 embedding models (round 3 MRR) + BM25 reference line.
Speed data from round 1 (same hardware, same models).
"""
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

models = [
    ("F2LLM-v2-1.7B",  1.7, 0.595, 1.72, 2048),
    ("F2LLM-v2-0.6B",  3.7, 0.561, 0.60, 1024),
    ("pplx-context",   3.2, 0.559, 0.60, 1024),
    ("jina-v5-small",  2.8, 0.558, 0.60, 1024),
]
BM25_MRR = 0.616

names  = [m[0] for m in models]
xs     = np.array([m[1] for m in models])
ys     = np.array([m[2] for m in models])
sizes  = np.array([m[3] for m in models]) * 250 + 60
dims   = np.array([m[4] for m in models])

fig, ax = plt.subplots(figsize=(8, 5.5))

# BM25 reference line
ax.axhline(BM25_MRR, color="#d62728", lw=1.5, ls=":", alpha=0.7, zorder=2)
ax.text(3.85, BM25_MRR + 0.003, "BM25 (MRR 0.616)", fontsize=9,
        color="#d62728", ha="right", va="bottom")

# Scatter
sc = ax.scatter(xs, ys, s=sizes, c=dims, cmap="viridis", alpha=0.8,
                edgecolors="#333333", linewidths=0.8, zorder=3)
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Dimensiones", fontsize=10)

# Labels
offsets = {
    "F2LLM-1.7B":  (10, 8),
    "F2LLM-0.6B":  (10, 8),
    "pplx-context": (10, -14),
    "jina-v5-small": (-10, -16),
}
for name, x, y in zip(names, xs, ys):
    dx, dy = offsets.get(name, (10, 6))
    ha = "right" if dx < 0 else "left"
    ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=10, ha=ha, color="#333333", zorder=4)

# Pareto frontier: F2LLM-1.7B (best quality, slowest) → F2LLM-0.6B (best speed)
pareto_x = [3.7, 1.7]
pareto_y = [0.561, 0.595]
ax.plot(pareto_x, pareto_y, "--", color="#999999", lw=1.2, zorder=2,
        label="Frontera de Pareto")

ax.set_xlabel("Velocidad de embedding (chunks/s)", fontsize=11)
ax.set_ylabel("MRR (calidad de recuperación, ronda 3)", fontsize=11)
ax.set_title("Velocidad vs calidad: 4 modelos + BM25\n"
             "(tamaño = parámetros; MacBook Pro M3, MPS; 499 docs, 3023 queries)",
             fontsize=11)
ax.set_xlim(1.3, 4.2)
ax.set_ylim(0.54, 0.63)
ax.grid(True, alpha=0.25, zorder=0)
ax.legend(loc="lower left", fontsize=9)

fig.tight_layout()
out = Path("reports/figures/pareto_round2.svg")
fig.savefig(out, format="svg")
fig.savefig(out.with_suffix(".png"), format="png", dpi=200)
plt.close(fig)
print(f"Saved {out} and {out.with_suffix('.png')}")
