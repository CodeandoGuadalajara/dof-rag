#!/usr/bin/env python3
"""DOF RAG PoC CLI.

Usage:
    # Index a single year (e.g., 2020 sample)
    python -m rag_poc.cli index ./dof_md/2020/01/15012020/MAT

    # Index an entire year
    python -m rag_poc.cli index ./dof_md/2020

    # Search
    python -m rag_poc.cli search "subsidio federal articulo 47 vivienda"

    # Stats
    python -m rag_poc.cli stats

First run downloads the ONNX model (~1.2 GB) automatically.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag_poc.config import DB_PATH
from rag_poc.database import RAGDatabase
from rag_poc.embedder import get_provider_info
from rag_poc.index import index_files
from rag_poc.search import hybrid_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag_poc.cli")


def _collect_md_files(root: Path, limit: int | None = None) -> list[Path]:
    files = sorted(p for p in root.rglob("*.md") if not p.name.endswith(".bak"))
    if limit:
        files = files[:limit]
    return files


def cmd_index(args: argparse.Namespace) -> None:
    root = Path(args.path)
    if not root.exists():
        sys.exit(f"Path does not exist: {root}")

    files = _collect_md_files(root, limit=args.limit)
    if not files:
        sys.exit(f"No markdown files found under {root}")

    print(f"Found {len(files):,} markdown files to index")
    print(f"Embedding provider: {get_provider_info()}")
    print(f"Database: {DB_PATH}")

    db = RAGDatabase()
    stats = index_files(files, db=db, embed_batch_size=args.batch_size)
    print(f"\n✅ Index complete: {stats['documents']:,} docs, {stats['chunks']:,} chunks")


def cmd_search(args: argparse.Namespace) -> None:
    db = RAGDatabase()
    results = hybrid_search(
        args.query,
        db=db,
        vector_k=args.vector_k,
        fts_k=args.fts_k,
        final_k=args.top_k,
    )
    if not results:
        print("No results found.")
        return

    print(f"\n🔍 Query: {args.query}")
    print(f"{'─' * 60}\n")
    for i, r in enumerate(results, 1):
        header = r.get("header_context", "") or "(no heading)"
        text_preview = r["text"].replace("\n", " ")[:300]
        source = r.get("source", "?")
        score = r.get("rrf_score", 0)
        pattern = r.get("pattern", "?")
        img = "🖼" if r.get("has_image") else " "
        print(f"  {i}. [{source}] score={score:.4f}  pattern={pattern} {img}")
        print(f"     Header: {header[:120]}")
        print(f"     Text:   {text_preview}...")
        print(f"     File:   {r['file_path']}  (chunk {r['chunk_number']})")
        print()


def cmd_stats(args: argparse.Namespace) -> None:
    db = RAGDatabase()
    stats = db.get_stats()
    print(f"Documents: {stats['documents']:,}")
    print(f"Chunks:    {stats['chunks']:,}")
    print(f"Vectors:   {stats['vectors']:,}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DOF RAG PoC: index and search Diario Oficial documents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index markdown files into the RAG DB")
    p_index.add_argument("path", help="Directory of markdown files to index")
    p_index.add_argument("--limit", type=int, default=None, help="Only index N files (for testing)")
    p_index.add_argument("--batch-size", type=int, default=8, help="Documents per ONNX batch")
    p_index.set_defaults(func=cmd_index)

    # search
    p_search = sub.add_parser("search", help="Hybrid search the indexed corpus")
    p_search.add_argument("query", help="Search query in Spanish")
    p_search.add_argument("--top-k", type=int, default=10, help="Final number of results")
    p_search.add_argument("--vector-k", type=int, default=20, help="Vector candidates")
    p_search.add_argument("--fts-k", type=int, default=20, help="FTS candidates")
    p_search.set_defaults(func=cmd_search)

    # stats
    p_stats = sub.add_parser("stats", help="Show database stats")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
