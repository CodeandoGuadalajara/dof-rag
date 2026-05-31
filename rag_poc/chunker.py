"""Custom DOF chunking: classify by pattern, split by strategy."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from rag_poc.config import MAX_TOKENS, OVERLAP_TOKENS


# ── Patterns ────────────────────────────────────────────────────────────
class DocPattern(Enum):
    SMALL = "small"              # < 10KB — chunk = doc completo
    H2_COMPOUND = "h2_compound"  # archivos compuestos, cada H2 es un decreto
    BOLD_HEADERS = "bold_headers"  # medianos con negritas como pseudo-headings
    PLAIN_TEXT = "plain_text"    # sin estructura
    GIANT_TABLE = "giant_table"  # >1MB dominado por tablas


@dataclass
class Chunk:
    text: str
    heading_path: list[str]
    chunk_index: int
    pattern: DocPattern
    has_image: bool


# ── Regexes ─────────────────────────────────────────────────────────────
BOILERPLATE_H = re.compile(
    r"^#{1,6}\s+(Al margen|Escudo Nacional|Sufragio Efectivo"
    r"|Lo que comunico|Dado en la Ciudad|En fe de lo cual"
    r"|Atentamente|Rúbrica)",
    re.MULTILINE | re.IGNORECASE,
)

H2_RE = re.compile(r"^## (.+)$", re.MULTILINE)
H3_RE = re.compile(r"^### (.+)$", re.MULTILINE)
BOLD_RE = re.compile(r"^\*\*([A-ZÁÉÍÓÚÑ][^*]{5,80})\*\*\s*$", re.MULTILINE)
TABLE_RE = re.compile(r"^\|", re.MULTILINE)
IMAGE_RE = re.compile(r"<!-- IMAGE_DESCRIPTION:", re.IGNORECASE)

# Inline IMAGE_DESCRIPTION into plain text
_IMAGE_DESC_RE = re.compile(
    r"<!--\s*IMAGE_DESCRIPTION:\s*(?P<ref>[^\n]+)\n"
    r"(?P<body>.*?)\n?-->\n?",
    re.DOTALL | re.IGNORECASE,
)


# ── Token approximation (Spanish legal: ~3.5 chars / token) ──────────────
# TODO: Replace with the model's real tokenizer for accurate limits.
# This heuristic under-counts markdown tables heavily: a 5 KB table
# with many |, spaces and - may report ~480 tokens while the real
# count is ~1,500.  That causes _flush_table to skip splitting tables
# that exceed MAX_TOKENS, producing oversized chunks.
# For the PoC this is acceptable; production should use
# tokenizer.encode(text, add_special_tokens=False) or similar.
def _count_tokens(text: str) -> int:
    return max(1, len(text) // 3)


# ── Classifier ───────────────────────────────────────────────────────────
def classify(text: str, size_bytes: int) -> DocPattern:
    if size_bytes < 10_000:
        return DocPattern.SMALL

    # Table dominance check — if most non-empty lines are table rows,
    # classify as GIANT_TABLE regardless of file size.
    lines = text.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    if non_empty:
        table_lines = sum(1 for ln in non_empty if ln.strip().startswith("|"))
        if table_lines / len(non_empty) > 0.40:
            return DocPattern.GIANT_TABLE

    if size_bytes > 1_000_000:
        return DocPattern.GIANT_TABLE

    h2_count = len(H2_RE.findall(text))
    if h2_count >= 2:
        return DocPattern.H2_COMPOUND
    bold_count = len(BOLD_RE.findall(text))
    if bold_count >= 2:
        return DocPattern.BOLD_HEADERS
    return DocPattern.PLAIN_TEXT


# ── Split entry point ───────────────────────────────────────────────────
def split_file(md_path: Path) -> list[Chunk]:
    """Classify a markdown file and split it into chunks."""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    text = _inline_image_descriptions(text)
    size = md_path.stat().st_size
    doc_id = md_path.stem
    pattern = classify(text, size)

    match pattern:
        case DocPattern.SMALL:
            return _split_small(text, doc_id, pattern)
        case DocPattern.H2_COMPOUND:
            return _split_h2_compound(text, doc_id, pattern)
        case DocPattern.BOLD_HEADERS:
            return _split_bold(text, doc_id, pattern)
        case DocPattern.PLAIN_TEXT:
            return _split_plain(text, doc_id, pattern)
        case DocPattern.GIANT_TABLE:
            return _split_giant_table(text, doc_id, pattern)
    return []  # pragma: no cover


# ── Strategy: SMALL ──────────────────────────────────────────────────────
def _split_small(text: str, doc_id: str, pattern: DocPattern) -> list[Chunk]:
    clean = BOILERPLATE_H.sub("", text).strip()
    return [
        Chunk(
            text=clean,
            heading_path=_extract_h1(text),
            chunk_index=0,
            pattern=pattern,
            has_image=bool(IMAGE_RE.search(text)),
        )
    ]


# ── Strategy: H2_COMPOUND ──────────────────────────────────────────────
def _split_h2_compound(text: str, doc_id: str, pattern: DocPattern) -> list[Chunk]:
    sections = _split_by_heading(text, H2_RE)
    chunks: list[Chunk] = []
    for heading, content in sections:
        if not content.strip():
            continue
        content = BOILERPLATE_H.sub("", content)
        token_count = _count_tokens(content)
        if token_count <= MAX_TOKENS:
            chunks.append(
                Chunk(
                    text=f"## {heading}\n\n{content}",
                    heading_path=[heading],
                    chunk_index=len(chunks),
                    pattern=pattern,
                    has_image=bool(IMAGE_RE.search(content)),
                )
            )
        else:
            # Partir por H3 dentro del H2
            sub_sections = _split_by_heading(content, H3_RE)
            for sub_heading, sub_content in sub_sections:
                sub_content = BOILERPLATE_H.sub("", sub_content)
                if not sub_content.strip():
                    continue
                parts = _split_by_tokens(sub_content, MAX_TOKENS, OVERLAP_TOKENS)
                for part in parts:
                    chunks.append(
                        Chunk(
                            text=f"## {heading}\n### {sub_heading}\n\n{part}",
                            heading_path=[heading, sub_heading],
                            chunk_index=len(chunks),
                            pattern=pattern,
                            has_image=bool(IMAGE_RE.search(part)),
                        )
                    )
    return chunks


# ── Strategy: BOLD_HEADERS ─────────────────────────────────────────────
def _split_bold(text: str, doc_id: str, pattern: DocPattern) -> list[Chunk]:
    clean = BOILERPLATE_H.sub("", text)
    if _count_tokens(clean) <= MAX_TOKENS:
        return [
            Chunk(
                text=clean,
                heading_path=_extract_bold_header(text),
                chunk_index=0,
                pattern=pattern,
                has_image=bool(IMAGE_RE.search(text)),
            )
        ]
    parts = re.split(r"\n{2,}", clean)
    return _merge_and_chunk(
        parts,
        doc_id,
        pattern,
        heading_path=_extract_bold_header(text),
    )


# ── Strategy: PLAIN_TEXT ───────────────────────────────────────────────
def _split_plain(text: str, doc_id: str, pattern: DocPattern) -> list[Chunk]:
    clean = BOILERPLATE_H.sub("", text).strip()
    if _count_tokens(clean) <= MAX_TOKENS:
        return [
            Chunk(
                text=clean,
                heading_path=[],
                chunk_index=0,
                pattern=pattern,
                has_image=bool(IMAGE_RE.search(text)),
            )
        ]
    parts = re.split(r"\n{2,}", clean)
    return _merge_and_chunk(parts, doc_id, pattern, heading_path=[])


# ── Strategy: GIANT_TABLE ──────────────────────────────────────────────
def _split_giant_table(text: str, doc_id: str, pattern: DocPattern) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_heading: list[str] = []
    current_text: list[str] = []
    in_table = False

    for line in text.splitlines(keepends=True):
        is_table_line = line.startswith("|")
        is_heading = re.match(r"^#{1,6} ", line)

        if is_heading and not BOILERPLATE_H.match(line):
            if in_table and current_text:
                chunks.extend(
                    _flush_table(
                        "".join(current_text), doc_id, current_heading, pattern
                    )
                )
                current_text = []
            current_heading = [line.strip().lstrip("#").strip()]
            in_table = False
        elif is_table_line:
            in_table = True
            current_text.append(line)
        else:
            if in_table and current_text:
                chunks.extend(
                    _flush_table(
                        "".join(current_text), doc_id, current_heading, pattern
                    )
                )
                current_text = []
                in_table = False
    if current_text:
        chunks.extend(
            _flush_table(
                "".join(current_text), doc_id, current_heading, pattern
            )
        )

    # Deduplicate chunk_index after all flushes
    for i, ch in enumerate(chunks):
        ch.chunk_index = i
    return chunks


# ── Helpers ──────────────────────────────────────────────────────────────
def _inline_image_descriptions(md_text: str) -> str:
    """Replace IMAGE_DESCRIPTION HTML comments with plain text paragraphs."""

    def _repl(m: re.Match) -> str:
        ref = m.group("ref").strip()
        body = m.group("body").strip()
        return f"[Imagen: {ref}] {body}\n\n"

    return _IMAGE_DESC_RE.sub(_repl, md_text)


def _split_by_heading(text: str, heading_re: re.Pattern) -> list[tuple[str, str]]:
    """Divide texto por un patrón de heading → (heading, contenido)."""
    positions = [(m.start(), m.group(1)) for m in heading_re.finditer(text)]
    if not positions:
        return [("", text)]
    result = []
    for i, (pos, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        nl_pos = text.index("\n", pos) + 1
        result.append((heading, text[nl_pos:end]))
    return result


def _split_by_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split por párrafos respetando límite de tokens, con overlap."""
    paragraphs = re.split(r"\n{2,}", text)
    # If a single paragraph is huge, split by single newlines first
    expanded: list[str] = []
    for para in paragraphs:
        if _count_tokens(para) > max_tokens:
            lines = para.splitlines()
            expanded.extend(lines)
        else:
            expanded.append(para)
    paragraphs = [p for p in expanded if p.strip()]

    chunks, current, current_tokens = [], [], 0
    for para in paragraphs:
        para_tokens = _count_tokens(para)
        # If even a single line is too big, force-split by chars
        if para_tokens > max_tokens:
            forced = _force_split(para, max_tokens)
            for f in forced:
                ft = _count_tokens(f)
                if current_tokens + ft > max_tokens and current:
                    chunks.append("\n".join(current))
                    overlap_paras, overlap_count = [], 0
                    for p in reversed(current):
                        t = _count_tokens(p)
                        if overlap_count + t > overlap:
                            break
                        overlap_paras.insert(0, p)
                        overlap_count += t
                    current = overlap_paras
                    current_tokens = overlap_count
                current.append(f)
                current_tokens += ft
            continue

        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            overlap_paras, overlap_count = [], 0
            for p in reversed(current):
                t = _count_tokens(p)
                if overlap_count + t > overlap:
                    break
                overlap_paras.insert(0, p)
                overlap_count += t
            current = overlap_paras
            current_tokens = overlap_count
        current.append(para)
        current_tokens += para_tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _force_split(text: str, max_tokens: int) -> list[str]:
    """Brute-force split by character count when no structural breaks exist."""
    max_chars = max_tokens * 3
    parts = []
    for i in range(0, len(text), max_chars):
        parts.append(text[i : i + max_chars])
    return parts


def _merge_and_chunk(
    parts: list[str],
    doc_id: str,
    pattern: DocPattern,
    heading_path: list[str],
) -> list[Chunk]:
    """Merge short parts into chunks respecting MAX_TOKENS."""
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # If a single part is huge, split it first
        if _count_tokens(part) > MAX_TOKENS:
            sub_parts = _split_by_tokens(part, MAX_TOKENS, OVERLAP_TOKENS)
            for sub in sub_parts:
                text = "\n\n".join(current + [sub]) if current else sub
                chunks.append(
                    Chunk(
                        text=text,
                        heading_path=list(heading_path),
                        chunk_index=len(chunks),
                        pattern=pattern,
                        has_image=bool(IMAGE_RE.search(text)),
                    )
                )
            current, current_tokens = [], 0
            continue
        t = _count_tokens(part)
        if current_tokens + t > MAX_TOKENS and current:
            text = "\n\n".join(current)
            chunks.append(
                Chunk(
                    text=text,
                    heading_path=list(heading_path),
                    chunk_index=len(chunks),
                    pattern=pattern,
                    has_image=bool(IMAGE_RE.search(text)),
                )
            )
            current, current_tokens = [], 0
        current.append(part)
        current_tokens += t
    if current:
        text = "\n\n".join(current)
        chunks.append(
            Chunk(
                text=text,
                heading_path=list(heading_path),
                chunk_index=len(chunks),
                pattern=pattern,
                has_image=bool(IMAGE_RE.search(text)),
            )
        )
    return chunks


def _flush_table(
    table_text: str,
    doc_id: str,
    heading: list[str],
    pattern: DocPattern,
) -> list[Chunk]:
    """Convierte una tabla en uno o más chunks, repitiendo el header."""
    lines = table_text.strip().splitlines()
    if not lines:
        return []
    header_lines = lines[:2]
    data_lines = lines[2:]
    header_text = "\n".join(header_lines) + "\n"
    header_tokens = _count_tokens("\n".join(header_lines))

    if _count_tokens(table_text) <= MAX_TOKENS:
        return [
            Chunk(
                text=table_text,
                heading_path=list(heading),
                chunk_index=0,
                pattern=pattern,
                has_image=False,
            )
        ]

    chunks: list[Chunk] = []
    batch: list[str] = []
    batch_tokens = header_tokens

    for row in data_lines:
        row_tokens = _count_tokens(row)
        # If a single row is too big, force-split it
        if row_tokens > MAX_TOKENS:
            # Flush current batch first
            if batch:
                chunks.append(
                    Chunk(
                        text=header_text + "\n".join(batch),
                        heading_path=list(heading),
                        chunk_index=len(chunks),
                        pattern=pattern,
                        has_image=False,
                    )
                )
                batch, batch_tokens = [], header_tokens
            forced = _force_split(row, MAX_TOKENS - header_tokens)
            for piece in forced:
                chunks.append(
                    Chunk(
                        text=header_text + piece,
                        heading_path=list(heading),
                        chunk_index=len(chunks),
                        pattern=pattern,
                        has_image=False,
                    )
                )
            continue

        if batch_tokens + row_tokens > MAX_TOKENS and batch:
            chunks.append(
                Chunk(
                    text=header_text + "\n".join(batch),
                    heading_path=list(heading),
                    chunk_index=len(chunks),
                    pattern=pattern,
                    has_image=False,
                )
            )
            batch, batch_tokens = [], header_tokens
        batch.append(row)
        batch_tokens += row_tokens

    if batch:
        chunks.append(
            Chunk(
                text=header_text + "\n".join(batch),
                heading_path=list(heading),
                chunk_index=len(chunks),
                pattern=pattern,
                has_image=False,
            )
        )
    return chunks


def _extract_h1(text: str) -> list[str]:
    m = re.search(r"^# (.+)$", text, re.MULTILINE)
    return [m.group(1)] if m else []


def _extract_bold_header(text: str) -> list[str]:
    """Extrae las primeras 2 líneas en negritas como identificador."""
    matches = BOLD_RE.findall(text)
    return matches[:2] if matches else []


# ── Legacy entry point for compatibility ────────────────────────────────
def chunk_markdown(file_path: Path) -> Iterator[dict]:
    """Yield chunk dicts for a single markdown file (legacy format)."""
    for ch in split_file(file_path):
        header_ctx = "\n".join(f"# {h}" if i == 0 else f"## {h}" for i, h in enumerate(ch.heading_path))
        yield {
            "text": ch.text,
            "header_context": header_ctx,
            "chunk_number": ch.chunk_index,
            "pattern": ch.pattern.value,
            "has_image": ch.has_image,
        }


def get_dof_url(file_path: Path) -> str:
    """Reconstruct the DOF PDF URL from the markdown file name."""
    stem = file_path.stem
    pdf_name = stem.replace("_", "-") + ".pdf"
    year = ""
    for p in stem.split("_"):
        if len(p) == 8 and p.isdigit():
            year = p[4:8]
            break
    if year:
        return f"https://diariooficial.gob.mx/abrirPDF.php?archivo={pdf_name}&anio={year}&repo=repositorio/"
    return ""
