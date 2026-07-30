"""Late chunking evaluation for pplx-embed-context-v1-0.6b.

Run from repo root:
    python scripts/evaluate_late_chunking.py [--corpus PATH] [--sample-size N]

Late chunking (Günther et al., 2023): instead of embedding each chunk in
isolation, embed the FULL document in one forward pass and mean-pool the
token-level embeddings over each chunk's token span. Every chunk embedding
has therefore attended to the whole document — the contextual advantage
pplx-embed-context-v1-0.6b was trained for.

Paired comparison on the SAME chunks and queries:
- `standard`: chunks embedded one by one (context-free), as in round 1.
- `late_chunking`: span-pooled token embeddings from full-document encoding.

Chunk spans are produced by an offset-tracking variant of the chunker's
paragraph splitter (no overlap, no heading prefixes, since context now comes
from the document encoding itself). Queries and metrics are identical to
scripts/evaluate_retrieval.py.

Outputs a Markdown report to `reports/late_chunking_evaluation.md`.
"""
from __future__ import annotations

import argparse
import gc
import re
import sys
import time
import warnings
from pathlib import Path
from random import sample, seed

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_retrieval import (  # noqa: E402
    DEFAULT_CORPUS,
    SAMPLE_SIZE,
    SEED,
    _compute_metrics,
    _create_queries,
    _iter_md_files,
)

from rag_poc.chunker import (  # noqa: E402
    BOILERPLATE_H,
    _count_tokens,
    _force_split,
    _inline_image_descriptions,
)
from rag_poc.config import MAX_TOKENS  # noqa: E402

REPORT_DIR = Path("reports")
MODEL_NAME = "perplexity-ai/pplx-embed-context-v1-0.6b"

H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)
H2_RE = re.compile(r"^## (.+)$", re.MULTILINE)


# ── Offset-tracking splitter ────────────────────────────────────────────
def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Char spans of paragraphs (blank-line separated)."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"\n{2,}", text):
        if text[start : m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def split_with_offsets(text: str, max_tokens: int = MAX_TOKENS) -> list[tuple[int, int]]:
    """Greedily merge paragraphs into chunk spans, respecting max_tokens.

    Mirrors rag_poc.chunker._split_by_tokens but without overlap and
    tracking char offsets into `text`. Oversized paragraphs are force-split
    into contiguous char pieces.
    """
    chunks: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_tokens
        if cur_start is not None:
            chunks.append((cur_start, cur_end))
        cur_start, cur_end, cur_tokens = None, None, 0

    for p_start, p_end in _paragraph_spans(text):
        para = text[p_start:p_end]
        para_tokens = _count_tokens(para)
        if para_tokens > max_tokens:
            flush()
            # Force-split into contiguous pieces and track their offsets.
            cursor = p_start
            for piece in _force_split(para, max_tokens):
                idx = text.find(piece, cursor)
                if idx == -1:
                    idx = cursor  # fallback: approximate
                chunks.append((idx, idx + len(piece)))
                cursor = idx + len(piece)
            continue
        if cur_start is not None and cur_tokens + para_tokens > max_tokens:
            flush()
        if cur_start is None:
            cur_start = p_start
        cur_end = p_end
        cur_tokens += para_tokens
    flush()
    return chunks


# ── Documents ────────────────────────────────────────────────────────────
def _get_lc_documents(corpus: Path, n_files: int) -> list[dict]:
    """Sample documents and produce offset-tracked chunk spans."""
    seed(SEED)
    files = sorted(sample(sorted(_iter_md_files(corpus)), n_files))
    docs: list[dict] = []
    for f in files:
        try:
            raw = f.read_text(encoding="utf-8", errors="replace")
            text = _inline_image_descriptions(raw)
            clean = BOILERPLATE_H.sub("", text).strip()
            if not clean:
                continue
            spans = split_with_offsets(clean)
            m = H1_RE.search(raw) or H2_RE.search(raw)
            title = m.group(1) if m else f.stem
            docs.append({
                "doc_id": f.stem,
                "title": title,
                "text": clean,
                "spans": spans,
                "chunks": [clean[s:e] for s, e in spans],
            })
        except Exception:
            continue
    return docs


# ── Encoders ─────────────────────────────────────────────────────────────
def _encode_standard(model, chunks: list[str]) -> np.ndarray:
    """Context-free baseline: each chunk embedded in isolation."""
    return model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def _encode_late_chunking(
    model, docs: list[dict], max_seq_length: int
) -> tuple[np.ndarray, list[str], dict]:
    """Full-document encoding + mean-pooling over chunk token spans."""
    chunk_embeddings: list[np.ndarray] = []
    chunk_doc_ids: list[str] = []
    stats = {"docs": 0, "docs_truncated": 0, "chunks": 0, "chunks_dropped": 0}

    for doc in docs:
        text = doc["text"]
        spans = doc["spans"]
        stats["docs"] += 1

        enc = model.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_length,
        )
        offsets = enc["offset_mapping"]
        n_tokens = len(offsets)
        total_tokens = len(model.tokenizer.encode(text, add_special_tokens=True))
        if total_tokens > n_tokens:
            stats["docs_truncated"] += 1

        token_emb = model.encode(
            [text],
            output_value="token_embeddings",
            convert_to_tensor=True,
            show_progress_bar=False,
        )[0]  # (n_tokens, dim)
        token_emb_np = token_emb.detach().float().cpu().numpy()

        for span_start, span_end in spans:
            # Tokens whose char span overlaps the chunk span.
            idx = [
                i
                for i, (ts, te) in enumerate(offsets)
                if i < token_emb_np.shape[0] and ts < span_end and te > span_start
            ]
            if not idx:
                stats["chunks_dropped"] += 1
                continue
            chunk_embeddings.append(token_emb_np[idx].mean(axis=0))
            chunk_doc_ids.append(doc["doc_id"])
            stats["chunks"] += 1

        del token_emb, token_emb_np
        gc.collect()

    if not chunk_embeddings:
        return np.zeros((0, 0), dtype=np.float32), [], stats
    return np.stack(chunk_embeddings).astype(np.float32), chunk_doc_ids, stats


# ── Report ────────────────────────────────────────────────────────────────
def _format_report(
    model_name: str,
    rows: dict[str, dict],
    stats: dict,
    elapsed: dict[str, float],
    n_docs: int,
    n_queries: int,
    corpus: str,
    sample_size: int,
) -> str:
    lines = [
        "# Late chunking: pplx-embed-context-v1-0.6b",
        "",
        f"Corpus: `{corpus}`",
        f"Muestra: **{sample_size}** documentos ({n_docs} usados), "
        f"{n_queries} queries (seed {SEED})",
        f"Fecha: {time.strftime('%Y-%m-%d')}",
        "",
        "Comparación pareada sobre **los mismos chunks y las mismas queries**:",
        "",
        "- `standard`: cada chunk embeddado de forma aislada (como en la ronda 1).",
        "- `late_chunking`: forward pass del documento completo (hasta "
        f"{stats['max_seq_length']:,} tokens) + mean-pooling del span de tokens de cada chunk.",
        "",
        "| Encoding | Chunks | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG | Tiempo (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, label in [("standard", "standard"), ("late_chunking", "late_chunking")]:
        m = rows[name]
        lines.append(
            f"| {label} | {m['n_chunks']:,} | {m['recall_at_k'][1]:.3f} | "
            f"{m['recall_at_k'][5]:.3f} | {m['recall_at_k'][10]:.3f} | "
            f"{m['mrr']:.3f} | {m['ndcg']:.3f} | {elapsed[name]:.1f} |"
        )

    base = rows["standard"]
    lc = rows["late_chunking"]
    d_mrr = (lc["mrr"] - base["mrr"]) * 100
    d_r1 = (lc["recall_at_k"][1] - base["recall_at_k"][1]) * 100
    d_r5 = (lc["recall_at_k"][5] - base["recall_at_k"][5]) * 100
    lines.extend([
        "",
        "## Delta late chunking vs standard",
        "",
        f"- Δ MRR: **{d_mrr:+.1f} pts**",
        f"- Δ Recall@1: {d_r1:+.1f} pts",
        f"- Δ Recall@5: {d_r5:+.1f} pts",
        "",
        "## Stats",
        "",
        f"- Documentos procesados: {stats['docs']}",
        f"- Documentos truncados a {stats['max_seq_length']:,} tokens: {stats['docs_truncated']}",
        f"- Chunks late chunking: {stats['chunks']:,} (descartados por truncado: {stats['chunks_dropped']})",
        f"- Chunks standard: {base['n_chunks']:,}",
        "",
        "## Notas",
        "",
        "- Los chunks los genera un splitter con tracking de offsets (sin overlap ni prefijos "
        "de encabezado: el contexto lo da la codificación del documento completo).",
        "- Las queries se codifican de forma estándar en ambos brazos.",
        "- Solo fp32: la cuantización int8 se evalúa después si el delta lo justifica.",
        "- Muestra determinística (seed 42, archivos ordenados).",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    root = Path(args.corpus)
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 1

    print("Getting documents (offset-tracked spans)...")
    docs = _get_lc_documents(root, args.sample_size)
    print(f"  {len(docs)} documents, {sum(len(d['chunks']) for d in docs):,} chunks")

    queries = _create_queries(docs)
    print(f"  {len(queries)} queries")

    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device=device, trust_remote_code=True)
    max_seq_length = model.max_seq_length
    print(f"max_seq_length: {max_seq_length:,}")

    all_chunks = [c for d in docs for c in d["chunks"]]
    chunk_doc_ids = [d["doc_id"] for d in docs for _ in d["chunks"]]

    print("\nEncoding queries...")
    query_emb = model.encode(
        [q["query"] for q in queries],
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    rows: dict[str, dict] = {}
    elapsed: dict[str, float] = {}

    print("\n[1/2] standard (context-free) encoding...")
    t0 = time.perf_counter()
    std_emb = _encode_standard(model, all_chunks)
    elapsed["standard"] = time.perf_counter() - t0
    rows["standard"] = _compute_metrics(query_emb, std_emb, chunk_doc_ids, queries)
    rows["standard"]["n_chunks"] = len(all_chunks)
    print(f"  MRR={rows['standard']['mrr']:.3f} R@1={rows['standard']['recall_at_k'][1]:.3f} "
          f"({elapsed['standard']:.1f}s)")

    print("\n[2/2] late chunking (full-document) encoding...")
    t0 = time.perf_counter()
    lc_emb, lc_doc_ids, stats = _encode_late_chunking(model, docs, max_seq_length)
    elapsed["late_chunking"] = time.perf_counter() - t0
    stats["max_seq_length"] = max_seq_length
    rows["late_chunking"] = _compute_metrics(query_emb, lc_emb, lc_doc_ids, queries)
    rows["late_chunking"]["n_chunks"] = len(lc_doc_ids)
    print(f"  MRR={rows['late_chunking']['mrr']:.3f} R@1={rows['late_chunking']['recall_at_k'][1]:.3f} "
          f"({elapsed['late_chunking']:.1f}s, {stats['docs_truncated']} docs truncated, "
          f"{stats['chunks_dropped']} chunks dropped)")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "late_chunking_evaluation.md"
    report_path.write_text(
        _format_report(args.model, rows, stats, elapsed, len(docs), len(queries),
                       args.corpus, args.sample_size),
        encoding="utf-8",
    )
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
