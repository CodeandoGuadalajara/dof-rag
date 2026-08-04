"""Build the chunk store (dof_chunks.sqlite) with exact span recipes.

Per docs/corpus-storage-architecture.md, chunk text is NOT stored. Instead,
each chunk records a recipe of spans into the chunker's normalized text
C(doc), plus short literals for synthetic text (heading prefixes, join
separators). C(doc) is deterministically recomputable from the stored
document at query time:

- GIANT_TABLE: C = inline_image_descriptions(raw)
- all other patterns: C = boilerplate_removal(inline_image_descriptions(raw))

Every chunk's recipe is hash-verified against the chunker's output at build
time, so any drift between solver and chunker fails loudly. Chunks that
cannot be solved (none expected) fall back to a single literal holding the
full chunk text; the fallback rate is reported.

Reconstruction (query time):
    raw = decompress(doc)               # from the corpus store
    C = normalized_text(raw, pattern)   # one regex pass, deterministic
    text = "".join(C[s:e] or lit for spans/literals in recipe)

Usage:
    uv run python -m corpus_store.chunk_index \
        --corpus-db poc/data/dof_corpus_l3.sqlite \
        --chunks-db poc/data/dof_chunks.sqlite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

from rag_poc.chunker import (
    BOILERPLATE_H, DocPattern, H2_RE, _count_tokens, _inline_image_descriptions,
    split_text)
from corpus_store.db import connect

CHUNKER_VERSION = "dof-chunker-v1"  # rag_poc.chunker, MAX_TOKENS=800, OVERLAP=50

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    pattern TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    spans_json TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    heading_path TEXT,
    chunk_hash BLOB NOT NULL,
    chunker_version TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    UNIQUE(document_id, chunk_index, chunker_version)
);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_chunker_idx ON chunks(chunker_version);
"""

ANCHOR = 24
MIN_EXT = 48  # global candidates must extend this far to be plausible
LOCAL_MIN_EXT = 8   # positionally grounded candidates need little extension
# Tiered search windows: (max distance from cursor, min extension).
# Positional grounding decays with distance, so nearer matches need less
# length. Short table rows qualify near the cursor; distant matches must
# be long enough to be trustworthy.
TIERS = [(2_048, 8), (8_192, 24), (None, MIN_EXT)]
CERTAIN_EXT = 1024  # a candidate extending this far is certainly the true source
MAX_CANDIDATES = 500


def normalized_text(raw: str, pattern: DocPattern) -> str:
    """The exact representation the chunker assembles chunks from.

    Matches the chunker's intermediate text per strategy:
    - GIANT_TABLE: image descriptions inlined; no boilerplate removal.
    - H2_COMPOUND: H2 heading lines kept verbatim (they are extracted as
      section headings and re-used as chunk prefixes, even when they are
      boilerplate); boilerplate removed from section contents; preamble
      before the first H2 discarded (the chunker never chunks it).
    - all others: boilerplate removal over the whole inlined text.
    """
    t = _inline_image_descriptions(raw)
    if pattern is DocPattern.GIANT_TABLE:
        return t
    if pattern is DocPattern.H2_COMPOUND:
        matches = list(H2_RE.finditer(t))
        if not matches:
            return BOILERPLATE_H.sub("", t)
        parts = []
        for k, m in enumerate(matches):
            line_end = t.find("\n", m.start())
            line_end = len(t) if line_end < 0 else line_end + 1
            sec_end = matches[k + 1].start() if k + 1 < len(matches) else len(t)
            parts.append(t[m.start():line_end])
            parts.append(BOILERPLATE_H.sub("", t[line_end:sec_end]))
        return "".join(parts)
    return BOILERPLATE_H.sub("", t)


def h2_sections(c: str) -> list[int]:
    """Start positions of H2 sections within normalized h2 text."""
    return [m.start() for m in H2_RE.finditer(c)]


def _extend(t: str, i: int, c: str, pos: int) -> int:
    """Longest L with t[i:i+L] == c[pos:pos+L] (exponential + binary search)."""
    max_l = min(len(t) - i, len(c) - pos)
    lo, hi = 0, 1
    while hi <= max_l and t[i:i + hi] == c[pos:pos + hi]:
        hi *= 2
    lo = hi // 2
    hi = min(hi, max_l + 1)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if t[i:i + mid] == c[pos:pos + mid]:
            lo = mid
        else:
            hi = mid
    return lo


MIN_ANCHOR = 6


def _find_best(t: str, i: int, c: str, start: int, end: int,
               min_ext: int, anchor_len: int = ANCHOR) -> tuple[int, int]:
    """Best-extension candidate for t[i:i+anchor_len] within c[start:end).

    Returns (pos, length) or (-1, 0) if no candidate reaches min_ext.
    """
    best_pos, best_len = -1, 0
    anchor = t[i:i + anchor_len]
    pos = c.find(anchor, start, end)
    candidates = 0
    while pos >= 0 and candidates < MAX_CANDIDATES:
        candidates += 1
        length = _extend(t, i, c, pos)
        if length > best_len:
            best_pos, best_len = pos, length
            if length >= CERTAIN_EXT or i + length >= len(t):
                break
        pos = c.find(anchor, pos + 1, end)
    if best_len < min_ext and i + best_len < len(t):
        return -1, 0
    return best_pos, best_len


def solve_spans(chunk_text: str, c: str, lo: int,
                hi: int | None = None, first_win: int = 0) -> tuple[list, int]:
    """Align chunk_text to c starting at lo. Returns (recipe, first_span_start).

    Recipe items: [start, end] (slice of c) or {"l": literal}.

    Three grounding rules, in order:
    - chunk-start rule: while the chunk has no span yet, candidates within
      first_win bytes of lo (the previous chunk's first span) are
      positionally grounded, so LOCAL_MIN_EXT suffices — short table rows
      at chunk boundaries qualify;
    - local rule: a match within LOCAL_WIN bytes of the cursor is the
      positional continuation across separators/indentation;
    - global rule: otherwise the max-extension candidate must reach MIN_EXT.
    Where nothing qualifies, literal chars keep reconstruction exact.
    """
    recipe: list = []
    lit: list[str] = []
    i = 0
    first = -1
    t = chunk_text
    hi = len(c) if hi is None else hi
    while i < len(t):
        # Anchor must not span a newline: chunk row/paragraph separators are
        # synthetic, so a cross-line anchor often has no match in c. Short
        # table rows therefore still anchor within their own line.
        nl = t.find("\n", i, i + ANCHOR)
        anchor_len = ANCHOR if nl < 0 else nl - i
        if anchor_len < MIN_ANCHOR:
            lit.append(t[i])
            i += 1
            if len(lit) > 500:
                raise ValueError(f"unsolvable literal >500 chars: {''.join(lit)[:80]!r}")
            continue
        if first < 0 and first_win > 0:
            tiers = [(first_win, LOCAL_MIN_EXT), (None, MIN_EXT)]
        else:
            tiers = TIERS
        for dist, min_ext in tiers:
            end = hi if dist is None else min(lo + dist, hi)
            best_pos, best_len = _find_best(t, i, c, lo, end, min_ext, anchor_len)
            if best_pos >= 0:
                break
        if best_pos < 0:
            lit.append(t[i])
            i += 1
            if len(lit) > 500:
                raise ValueError(f"unsolvable literal >500 chars: {''.join(lit)[:80]!r}")
            continue
        if lit:
            recipe.append({"l": "".join(lit)})
            lit = []
        recipe.append([best_pos, best_pos + best_len])
        if first < 0:
            first = best_pos
        i += best_len
        lo = best_pos + best_len
    if lit:
        recipe.append({"l": "".join(lit)})
    if first < 0:
        raise ValueError("no span aligned at all")
    return recipe, first


def reconstruct(recipe: list, c: str) -> str:
    parts = []
    for item in recipe:
        if isinstance(item, list):
            parts.append(c[item[0]:item[1]])
        else:
            parts.append(item["l"])
    return "".join(parts)


def iter_documents(conn: sqlite3.Connection):
    from corpus_store.db import fetch_document_text
    cur = conn.execute(
        "SELECT document_id, path, markdown FROM documents ORDER BY document_id")
    for doc_id, path, text in cur:
        if text:
            yield doc_id, path, text
        else:
            yield doc_id, path, fetch_document_text(conn, doc_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-db", default="poc/data/dof_corpus_l3.sqlite")
    ap.add_argument("--chunks-db", required=True)
    args = ap.parse_args()

    corpus = connect(args.corpus_db)
    corpus_version = corpus.execute(
        "SELECT value FROM corpus_meta WHERE key = 'corpus_version'").fetchone()[0]

    chunks_path = Path(args.chunks_db)
    fresh = not chunks_path.exists()
    chunks = sqlite3.connect(str(chunks_path))
    if fresh:
        chunks.execute("PRAGMA journal_mode = WAL")
        chunks.execute("PRAGMA synchronous = NORMAL")
        chunks.executescript(SCHEMA)
        chunks.commit()

    t0 = time.time()
    n_docs = n_chunks = n_fallback = n_recipe_bytes = 0
    stats_by_pattern: dict[str, int] = {}
    batch: list[tuple] = []
    for doc_id, path, raw in iter_documents(corpus):
        doc_t0 = time.time()
        done = chunks.execute(
            "SELECT 1 FROM chunks WHERE document_id = ? AND chunker_version = ?"
            " LIMIT 1", (doc_id, CHUNKER_VERSION)).fetchone()
        if done:
            continue  # resume
        doc_chunks = split_text(raw, len(raw.encode("utf-8")), Path(path).stem)
        c = normalized_text(raw, doc_chunks[0].pattern) if doc_chunks else ""
        is_h2 = doc_chunks and doc_chunks[0].pattern is DocPattern.H2_COMPOUND
        sections = h2_sections(c) if is_h2 else []
        cursor = 0
        lo = 0
        prev_len = 0
        for ch in doc_chunks:
            pat = ch.pattern.value
            recipe = None
            first = 0
            if is_h2:
                j = cursor
                while j < len(sections):
                    sec_end = sections[j + 1] if j + 1 < len(sections) else len(c)
                    try:
                        recipe, first = solve_spans(ch.text, c, sections[j], sec_end)
                        cursor = j
                        break
                    except ValueError:
                        j += 1
            else:
                try:
                    recipe, first = solve_spans(
                        ch.text, c, lo, first_win=2 * prev_len + 1024)
                    lo = first
                    prev_len = len(ch.text)
                except ValueError:
                    recipe = None
            if recipe is None:
                recipe = [{"l": ch.text}]
                first = 0
                n_fallback += 1
            elif reconstruct(recipe, c) != ch.text:
                recipe = [{"l": ch.text}]
                first = 0
                n_fallback += 1
            n_recipe_bytes += len(json.dumps(recipe))
            batch.append((
                doc_id, path, ch.chunk_index, pat,
                recipe[0][0] if isinstance(recipe[0], list) else 0,
                recipe[-1][1] if isinstance(recipe[-1], list) else 0,
                json.dumps(recipe, ensure_ascii=False),
                _count_tokens(ch.text),
                json.dumps(ch.heading_path, ensure_ascii=False),
                hashlib.sha256(ch.text.encode("utf-8")).digest(),
                CHUNKER_VERSION, corpus_version,
            ))
            n_chunks += 1
            stats_by_pattern[pat] = stats_by_pattern.get(pat, 0) + 1
        n_docs += 1
        doc_dt = time.time() - doc_t0
        if doc_dt > 60:
            print(f"  SLOW doc {doc_id} ({doc_dt:.0f}s, {len(raw) / 2**20:.1f} MiB,"
                  f" {len(doc_chunks)} chunks): {path}", flush=True)
        if len(batch) >= 5000:
            with chunks:
                chunks.executemany(
                    "INSERT INTO chunks (document_id, path, chunk_index, pattern,"
                    " start_offset, end_offset, spans_json, token_count,"
                    " heading_path, chunk_hash, chunker_version, corpus_version)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
            batch.clear()
        if n_docs % 1000 == 0:
            print(f"  {n_docs:,} docs, {n_chunks:,} chunks "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if batch:
        with chunks:
            chunks.executemany(
                "INSERT INTO chunks (document_id, path, chunk_index, pattern,"
                " start_offset, end_offset, spans_json, token_count,"
                " heading_path, chunk_hash, chunker_version, corpus_version)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)

    chunks.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    size = chunks_path.stat().st_size
    print(f"\nDone: {n_docs:,} docs, {n_chunks:,} chunks, "
          f"fallbacks {n_fallback} ({n_fallback / max(n_chunks, 1):.4%})")
    print(f"patterns: {stats_by_pattern}")
    print(f"recipe bytes total: {n_recipe_bytes / 2**20:.1f} MiB "
          f"({n_recipe_bytes / max(n_chunks, 1):.0f} B/chunk)")
    print(f"chunks db size: {size / 2**20:.1f} MiB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
