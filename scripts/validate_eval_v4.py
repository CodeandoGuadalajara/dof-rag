"""Validate the manually curated DOF evidence benchmark v4.

The validator is intentionally deterministic.  It checks the frozen corpus
snapshot, category balance, document/chunk identifiers, and that every quoted
gold span occurs in the regenerated chunk text.

Usage:
    uv run python scripts/validate_eval_v4.py
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_poc.chunker import split_file  # noqa: E402

EXPECTED_CATEGORIES = {
    "single_passage",
    "list_enumeration",
    "temporal_transitorio",
    "cross_reference",
    "multi_document",
    "monitoring",
    "negative_false_premise",
}
REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "persona",
    "difficulty",
    "as_of",
    "answerability",
    "reference_answer",
    "required_hops",
    "gold_documents",
    "unanswerable_reason",
}


def normalized(text: str) -> str:
    """Comparison form tolerant of Markdown punctuation and whitespace."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/dof_queries_v4.jsonl")
    ap.add_argument("--meta", default="eval/dof_queries_v4.meta.json")
    ap.add_argument("--corpus", default="../dof_md")
    ap.add_argument("--corpus-db", default="dof_db/dof_corpus_l3.sqlite")
    ap.add_argument("--chunks-db", default="dof_db/dof_chunks.sqlite")
    args = ap.parse_args()

    records = load_jsonl(Path(args.queries))
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    errors: list[str] = []

    if len(records) != meta["total_questions"]:
        errors.append(
            f"record count {len(records)} != metadata {meta['total_questions']}"
        )

    ids = [r.get("id") for r in records]
    duplicate_ids = [k for k, n in Counter(ids).items() if n > 1]
    if duplicate_ids:
        errors.append(f"duplicate ids: {duplicate_ids}")
    questions = [normalized(r.get("question", "")) for r in records]
    duplicate_questions = [k for k, n in Counter(questions).items() if k and n > 1]
    if duplicate_questions:
        errors.append(f"duplicate questions: {duplicate_questions}")

    counts = Counter(r.get("category") for r in records)
    if set(counts) != EXPECTED_CATEGORIES:
        errors.append(
            f"categories {sorted(counts)} != expected {sorted(EXPECTED_CATEGORIES)}"
        )
    for category in EXPECTED_CATEGORIES:
        if counts[category] != 6:
            errors.append(f"{category}: expected 6 questions, got {counts[category]}")
    if set(meta["categories"]) != EXPECTED_CATEGORIES:
        errors.append("metadata categories do not match the evaluation contract")
    if meta["questions_per_category"] != 6:
        errors.append("metadata questions_per_category must be 6")

    corpus = sqlite3.connect(args.corpus_db)
    chunks_db = sqlite3.connect(args.chunks_db)
    db_version = corpus.execute(
        "SELECT value FROM corpus_meta WHERE key = 'corpus_version'"
    ).fetchone()[0]
    if db_version != meta["corpus_version"]:
        errors.append(
            f"corpus version {db_version!r} != metadata {meta['corpus_version']!r}"
        )
    db_count, db_min, db_max = corpus.execute(
        "SELECT COUNT(*), MIN(publication_date), MAX(publication_date)"
        " FROM _documents_zstd"
    ).fetchone()
    if (db_count, db_min, db_max) != (
        meta["corpus_documents"],
        meta["corpus_date_min"],
        meta["corpus_date_max"],
    ):
        errors.append(
            "corpus snapshot mismatch: "
            f"db={(db_count, db_min, db_max)}, "
            f"meta={(meta['corpus_documents'], meta['corpus_date_min'], meta['corpus_date_max'])}"
        )

    doc_cache: dict[int, tuple[str, str, str]] = {}
    chunk_rows: dict[int, tuple[int, str, int, str, str]] = {}
    split_cache: dict[str, dict[int, str]] = {}
    corpus_root = Path(args.corpus)

    for record in records:
        rid = record.get("id", "<missing-id>")
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            errors.append(f"{rid}: missing fields {sorted(missing)}")
            continue
        if not record["question"].endswith("?"):
            errors.append(f"{rid}: question must end with ?")
        if record["difficulty"] not in {"easy", "medium", "hard"}:
            errors.append(f"{rid}: invalid difficulty {record['difficulty']!r}")
        if record["as_of"] > meta["corpus_date_max"]:
            errors.append(f"{rid}: as_of is after the frozen corpus")
        if not record["reference_answer"].strip():
            errors.append(f"{rid}: empty reference_answer")
        if not record["gold_documents"]:
            errors.append(f"{rid}: gold_documents must not be empty")
        if record["category"] == "negative_false_premise":
            if record["answerability"] != "false_premise":
                errors.append(f"{rid}: negative item must be false_premise")
            if not record["unanswerable_reason"]:
                errors.append(f"{rid}: negative item needs unanswerable_reason")
        else:
            if record["answerability"] != "answerable":
                errors.append(f"{rid}: non-negative item must be answerable")
            if record["unanswerable_reason"] is not None:
                errors.append(f"{rid}: answerable item has unanswerable_reason")
        if record["category"] == "multi_document":
            if len(record["gold_documents"]) < 2 or record["required_hops"] < 2:
                errors.append(f"{rid}: multi_document needs at least two documents/hops")
        elif record["required_hops"] < 1:
            errors.append(f"{rid}: required_hops must be positive")

        for gold in record["gold_documents"]:
            did = gold["document_id"]
            if did not in doc_cache:
                row = corpus.execute(
                    "SELECT path, publication_date, section FROM _documents_zstd"
                    " WHERE document_id = ?",
                    (did,),
                ).fetchone()
                if row:
                    doc_cache[did] = row
            row = doc_cache.get(did)
            expected = (
                gold["relpath"],
                gold["publication_date"],
                gold["section"],
            )
            if row != expected:
                errors.append(f"{rid}: document {did} metadata {row} != {expected}")
                continue

            relpath = gold["relpath"]
            if relpath not in split_cache:
                path = corpus_root / relpath
                if not path.exists():
                    errors.append(f"{rid}: missing corpus file {path}")
                    continue
                split_cache[relpath] = {
                    ch.chunk_index: ch.text for ch in split_file(path)
                }

            for evidence in gold["evidence"]:
                cid = evidence["chunk_id"]
                if cid not in chunk_rows:
                    row_chunk = chunks_db.execute(
                        "SELECT document_id, path, chunk_index, chunker_version,"
                        " corpus_version FROM chunks"
                        " WHERE chunk_id = ?",
                        (cid,),
                    ).fetchone()
                    if row_chunk:
                        chunk_rows[cid] = row_chunk
                expected_chunk = (
                    did,
                    relpath,
                    evidence["chunk_index"],
                    meta["chunker_version"],
                    meta["corpus_version"],
                )
                if chunk_rows.get(cid) != expected_chunk:
                    errors.append(
                        f"{rid}: chunk {cid} metadata {chunk_rows.get(cid)}"
                        f" != {expected_chunk}"
                    )
                    continue
                quote_norm = normalized(evidence["quote"])
                chunk_text = split_cache[relpath].get(evidence["chunk_index"])
                if chunk_text is None:
                    errors.append(
                        f"{rid}: regenerated chunk {evidence['chunk_index']} missing"
                    )
                elif len(quote_norm.split()) < 8:
                    errors.append(f"{rid}: evidence quote is too short: {evidence['quote']!r}")
                elif quote_norm not in normalized(chunk_text):
                    errors.append(
                        f"{rid}: quote not found in chunk {cid}: {evidence['quote']!r}"
                    )

    if errors:
        print(f"FAIL: {len(errors)} validation error(s)", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(records)} questions, {len(counts)} categories, "
        f"{len(doc_cache)} documents, {len(chunk_rows)} evidence chunks"
    )
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
