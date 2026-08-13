"""Extract compact, best-effort metadata from a DOF Markdown header."""

from __future__ import annotations

import re
from dataclasses import dataclass

HEAD_CHARS = 3000
MAX_TITLE_CHARS = 350


@dataclass(frozen=True)
class DocumentHeader:
    institution: str | None
    title: str | None


def _clean(value: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", value).strip(" ,;.")
    if len(cleaned) > MAX_TITLE_CHARS:
        cleaned = cleaned[:330].rsplit(" ", 1)[0]
    return cleaned if len(cleaned) > 3 else None


def _heading_header(text: str) -> DocumentHeader | None:
    institution = None
    title = None
    for line in text[:HEAD_CHARS].splitlines():
        match = re.match(r"^\s*(#{1,3})\s+(.+?)\s*$", line)
        if not match:
            if line.strip() and (institution or title):
                break
            continue
        level = len(match.group(1))
        value = _clean(match.group(2))
        if level == 1 and institution is None:
            institution = value
        elif level == 2 and title is None:
            title = value
        if institution and title:
            break
    if institution or title:
        return DocumentHeader(institution=institution, title=title or institution)
    return None


def _plain_header(text: str) -> DocumentHeader:
    head = text[:HEAD_CHARS]
    bold = [
        value
        for match in re.finditer(r"\*\*([^*]{4,200})\*\*", head)
        if (value := _clean(match.group(1)))
    ]
    institution = _clean(" ".join(bold[:2])) if bold else None

    title = None
    for line in head.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("![") or stripped.startswith("**"):
            continue
        title = _clean(stripped.lstrip("#* "))
        if title:
            break
    if title is None and bold:
        title = _clean(" ".join(bold[:4]))
    return DocumentHeader(institution=institution, title=title)


def extract_document_header(text: str) -> DocumentHeader:
    """Return institution and title without claiming they are indexed metadata."""
    return _heading_header(text) or _plain_header(text)
