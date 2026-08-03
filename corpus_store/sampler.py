"""Draw a reproducible random sample of DOF Markdown documents.

Enumerates the corpus tree reading only directory metadata, verifies that
each sampled file is locally materialized (Google Drive offline pin), and
writes a JSONL manifest that the ingestion step consumes. The manifest makes
the PoC reproducible and ingestion resumable.

Usage:
    uv run python -m corpus_store.sampler \
        --corpus ../dof_md --n 10000 --seed 42 --out poc/data/manifest_10k.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import stat
import sys
import time
from pathlib import Path

# macOS: file is a dataless placeholder (not downloaded from Drive).
SF_DATALESS = 0x40000000

SKIP_SUFFIXES = {".bak"}


def enumerate_markdown(corpus: Path) -> list[tuple[str, int, int]]:
    """Return [(relpath, size_bytes, st_flags)] for every .md file.

    Uses only directory metadata; never opens a file, so dataless
    placeholders do not trigger Drive downloads.
    """
    entries: list[tuple[str, int, int]] = []
    t0 = time.time()
    for dirpath, _dirnames, filenames in os.walk(corpus):
        for name in filenames:
            if not name.endswith(".md"):
                continue
            if any(name.endswith(s) for s in SKIP_SUFFIXES):
                continue
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except OSError as e:
                print(f"WARN: stat failed for {p}: {e}", file=sys.stderr)
                continue
            entries.append((str(p.relative_to(corpus)), st.st_size, st.st_flags))
        n = len(entries)
        if n and n % 100_000 < 5000:
            print(f"  enumerated {n:,} files ({time.time() - t0:.0f}s)", file=sys.stderr)
    return entries


def parse_metadata(relpath: str) -> dict:
    """Extract year/section/publication_date from corpus path layout.

    Layout: <year>/<MM>/<DDMMYYYY>/<SECTION>/<file>.md
    """
    parts = Path(relpath).parts
    meta: dict = {"year": None, "section": None, "publication_date": None}
    if len(parts) >= 4:
        try:
            meta["year"] = int(parts[0])
        except ValueError:
            pass
        meta["section"] = parts[-2]
        day_dir = parts[-3]  # DDMMYYYY
        if len(day_dir) == 8 and day_dir.isdigit():
            dd, mm, yyyy = day_dir[:2], day_dir[2:4], day_dir[4:]
            meta["publication_date"] = f"{yyyy}-{mm}-{dd}"
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../dof_md")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--all", action="store_true",
                    help="manifest the entire corpus (full build), not a sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="poc/data/manifest_10k.jsonl")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Enumerating {corpus} ...")
    entries = enumerate_markdown(corpus)
    total = len(entries)
    total_bytes = sum(sz for _, sz, _ in entries)
    print(f"Found {total:,} markdown files, {total_bytes / 2**30:.2f} GiB")

    n_dataless = sum(1 for _, _, fl in entries if fl & SF_DATALESS)
    if n_dataless:
        print(f"WARN: {n_dataless:,} dataless (non-materialized) files in corpus", file=sys.stderr)

    rng = random.Random(args.seed)
    sampled = entries if args.all else rng.sample(entries, min(args.n, total))
    sampled = sorted(sampled)

    n_skipped = 0
    with out.open("w") as f:
        for relpath, size, flags in sampled:
            if flags & SF_DATALESS:
                n_skipped += 1
                continue
            rec = {"relpath": relpath, "size_bytes": size, **parse_metadata(relpath)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    kept = len(sampled) - n_skipped
    print(f"Wrote {kept:,} sampled docs to {out} (skipped {n_skipped} dataless)")
    if n_skipped:
        print("WARN: sample is short; re-run with a fresh seed after pinning completes",
              file=sys.stderr)


if __name__ == "__main__":
    main()
