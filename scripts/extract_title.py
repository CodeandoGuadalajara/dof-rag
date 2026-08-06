"""Extract document titles from DOF markdown bold header blocks.

Most DOF documents don't use markdown headings (# ...); the institution
and title are rendered as bold spans (**SECRETARIA DE...** **ACUERDO...**)
at the start of the document. This extractor joins the first group of
bold spans into a title. Used for the eval-set v3 title fix and, later,
for the corpus-wide title column / title-search tool.

Usage (smoke test on the eval docs):
    uv run python scripts/extract_title.py
"""

import json
import re
import sys
from pathlib import Path

SLUG_RE = re.compile(r"^\d{3}_(AVISO|DOC|DOF|ACUERDO)_\d{8}_")
HEAD_CHARS = 3000
MAX_SPANS = 4


def _clean(parts: list[str]) -> str | None:
    title = re.sub(r"\s+", " ", " ".join(parts)).strip(" ,;.")
    if len(title) > 350:
        title = title[:330].rsplit(" ", 1)[0]  # truncate at word boundary
    return title if len(title) > 10 else None


def _from_headings(text: str) -> str | None:
    """Markdown heading lines at the document start (blank lines allowed)."""
    parts = []
    for line in text[:HEAD_CHARS].splitlines():
        line = line.strip()
        m = re.match(r"^#{1,3}\s+(.+)", line)
        if m:
            parts.append(m.group(1).strip())
        elif line and parts:
            break  # first non-heading content ends the header block
        if len(parts) >= 3:
            break
    return _clean(parts) if parts else None


def _from_bold(text: str) -> str | None:
    spans = []
    for m in re.finditer(r"\*\*([^*]{4,200})\*\*", text[:HEAD_CHARS]):
        spans.append(m.group(1).strip())
        if len(spans) >= MAX_SPANS:
            break
    return _clean(spans) if spans else None


def _from_first_lines(text: str) -> str | None:
    """Plain-text header block (court notices): first non-empty lines."""
    parts = []
    for line in text[:HEAD_CHARS].splitlines():
        line = line.strip()
        if line and not line.startswith("!["):
            parts.append(line.lstrip("#* "))
        elif line and parts:
            break
        if len(parts) >= 3:
            break
    return _clean(parts) if parts else None


def extract_title(text: str) -> str | None:
    """Title from the document header: headings, bold spans, or first lines."""
    return (_from_headings(text) or _from_bold(text)
            or _from_first_lines(text))


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    recs = [json.loads(l) for l in open("eval/dof_queries_v2.jsonl")]
    slug = [r for r in recs if not r.get("error") and SLUG_RE.match(r["title"])]
    ok, short, fail = 0, 0, []
    for r in slug:
        text = Path("../dof_md", r["relpath"]).read_text(
            encoding="utf-8", errors="replace")
        t = extract_title(text)
        if t:
            ok += 1
        else:
            fail.append((r["doc_id"], text[:120].replace("\n", " ")))
    print(f"{ok}/{len(slug)} slug-title docs got a bold-block title")
    for doc_id, head in fail[:10]:
        print(f"  FAIL {doc_id}: {head!r}")


if __name__ == "__main__":
    main()
