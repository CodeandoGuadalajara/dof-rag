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

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_tools.headers import extract_document_header  # noqa: E402

SLUG_RE = re.compile(r"^\d{3}_(AVISO|DOC|DOF|ACUERDO)_\d{8}_")


def extract_title(text: str) -> str | None:
    """Title from the document header: headings, bold spans, or first lines."""
    return extract_document_header(text).title


def main() -> None:
    with open("eval/dof_queries_v2.jsonl", encoding="utf-8") as source:
        recs = [json.loads(line) for line in source]
    slug = [r for r in recs if not r.get("error") and SLUG_RE.match(r["title"])]
    ok, fail = 0, []
    for r in slug:
        text = Path("../dof_md", r["relpath"]).read_text(
            encoding="utf-8", errors="replace"
        )
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
