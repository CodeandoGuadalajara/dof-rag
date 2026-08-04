"""Custom DOF chunking: classify by pattern, split by strategy."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from rag_poc.config import H2_MAX_TOKENS, MAX_TOKENS, OVERLAP_TOKENS


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


# ── Token counter (lazy-loaded real tokenizer) ────────────────────────────
_tokenizer = None


def _count_tokens(text: str) -> int:
    """Return token count using the model's real tokenizer if available."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(
                "perplexity-ai/pplx-embed-context-v1-0.6b",
                trust_remote_code=True,
            )
        except Exception:
            # If tokenizer is not available (e.g. transformers not installed),
            # fall back to a conservative heuristic.
            return max(1, len(text) // 3)
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _count_tokens_batch(texts: list[str]) -> list[int]:
    """Token counts for many strings in one Rust call.

    Same values as _count_tokens per item (verified), without the per-call
    Python overhead — hot loops (table rows, paragraphs) call this once
    instead of tokenizing per iteration. Uses the inner tokenizers-lib
    tokenizer directly: the model's custom wrapper does not expose
    encode_batch at the Python level.
    """
    if _tokenizer is None:
        return [_count_tokens(t) for t in texts]
    inner = getattr(_tokenizer, "_tokenizer", None)
    if inner is None or not hasattr(inner, "encode_batch"):
        return [_count_tokens(t) for t in texts]
    try:
        return [len(e) for e in inner.encode_batch(
            texts, add_special_tokens=False)]
    except Exception:
        return [_count_tokens(t) for t in texts]


# ── Classifier ───────────────────────────────────────────────────────────
def classify(text: str, size_bytes: int) -> DocPattern:
    # Size threshold tuned for the real tokenizer: 6 KB of markdown legal
    # text is roughly in the 800-1500 token range. Docs below this are very
    # likely to fit in a single chunk, so we avoid paying for structural
    # analysis on them.
    if size_bytes < 6_000:
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
    size = md_path.stat().st_size
    return split_text(text, size, md_path.stem)


def split_text(raw_text: str, size_bytes: int, doc_id: str) -> list[Chunk]:
    """Split in-memory markdown text (same behavior as split_file)."""
    text = _inline_image_descriptions(raw_text)
    pattern = classify(text, size_bytes)

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
    heading_path = _extract_h1(text)
    has_image = bool(IMAGE_RE.search(text))
    if _count_tokens(clean) <= MAX_TOKENS:
        return [
            Chunk(
                text=clean,
                heading_path=heading_path,
                chunk_index=0,
                pattern=pattern,
                has_image=has_image,
            )
        ]
    parts = _split_by_tokens(clean, MAX_TOKENS, OVERLAP_TOKENS)
    return [
        Chunk(
            text=part,
            heading_path=heading_path,
            chunk_index=i,
            pattern=pattern,
            has_image=has_image,
        )
        for i, part in enumerate(parts)
    ]


# ── Strategy: H2_COMPOUND ──────────────────────────────────────────────
def _split_h2_compound(text: str, doc_id: str, pattern: DocPattern) -> list[Chunk]:
    sections = _split_by_heading(text, H2_RE)
    chunks: list[Chunk] = []
    for heading, content in sections:
        if not content.strip():
            continue
        content = BOILERPLATE_H.sub("", content)
        chunk_text = f"## {heading}\n\n{content}"
        token_count = _count_tokens(chunk_text)
        if token_count <= H2_MAX_TOKENS:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    heading_path=[heading],
                    chunk_index=len(chunks),
                    pattern=pattern,
                    has_image=bool(IMAGE_RE.search(content)),
                )
            )
        else:
            # Partir por H3 dentro del H2
            sub_sections = _split_by_heading(content, H3_RE)
            # If there are no real H3 headings, split the whole section directly
            if len(sub_sections) == 1 and sub_sections[0][0] == "":
                sub_content = BOILERPLATE_H.sub("", sub_sections[0][1])
                if sub_content.strip():
                    prefix = f"## {heading}\n\n"
                    prefix_tokens = _count_tokens(prefix)
                    budget = max(1, MAX_TOKENS - prefix_tokens)
                    overlap = min(OVERLAP_TOKENS, budget - 1)
                    parts = _split_by_tokens(sub_content, budget, overlap)
                    for part in parts:
                        chunks.append(
                            Chunk(
                                text=prefix + part,
                                heading_path=[heading],
                                chunk_index=len(chunks),
                                pattern=pattern,
                                has_image=bool(IMAGE_RE.search(part)),
                            )
                        )
                continue

            for sub_heading, sub_content in sub_sections:
                sub_content = BOILERPLATE_H.sub("", sub_content)
                if not sub_content.strip():
                    continue
                prefix = f"## {heading}\n### {sub_heading}\n\n"
                prefix_tokens = _count_tokens(prefix)
                budget = max(1, MAX_TOKENS - prefix_tokens)
                overlap = min(OVERLAP_TOKENS, budget - 1)
                parts = _split_by_tokens(sub_content, budget, overlap)
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
    """Split a table-heavy document preserving both tables and non-table text."""
    chunks: list[Chunk] = []
    current_heading: list[str] = []

    table_buffer: list[str] = []
    text_buffer: list[str] = []

    def _flush_table_buffer() -> None:
        if table_buffer:
            chunks.extend(
                _flush_table(
                    "".join(table_buffer), doc_id, current_heading, pattern
                )
            )
            table_buffer.clear()

    def _flush_text_buffer() -> None:
        if text_buffer:
            txt = "".join(text_buffer).strip()
            if txt:
                for part in _split_by_tokens(txt, MAX_TOKENS, OVERLAP_TOKENS):
                    chunks.append(
                        Chunk(
                            text=part,
                            heading_path=list(current_heading),
                            chunk_index=len(chunks),
                            pattern=pattern,
                            has_image=bool(IMAGE_RE.search(part)),
                        )
                    )
            text_buffer.clear()

    for line in text.splitlines(keepends=True):
        is_table_line = _is_table_line(line)
        is_heading = re.match(r"^#{1,6} ", line)

        if is_heading and not BOILERPLATE_H.match(line):
            _flush_table_buffer()
            _flush_text_buffer()
            current_heading = [line.strip().lstrip("#").strip()]
        elif is_table_line:
            _flush_text_buffer()
            table_buffer.append(line)
        elif line.strip():
            _flush_table_buffer()
            text_buffer.append(line)
        else:
            # Empty line: keep text buffer alive but do not flush table.
            text_buffer.append(line)

    _flush_table_buffer()
    _flush_text_buffer()

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
        # Safe newline search: handle heading at EOF without trailing \n
        nl_pos = text.find("\n", pos)
        if nl_pos == -1:
            nl_pos = len(text)
        else:
            nl_pos += 1
        result.append((heading, text[nl_pos:end]))
    return result


def _split_by_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split por párrafos respetando límite de tokens, con overlap."""
    paragraphs = re.split(r"\n{2,}", text)
    # If a single paragraph is huge, split by single newlines first.
    # Batch the counts (same values, one Rust call) and memoize them: the
    # merge loop and the overlap recomputation recount the same strings.
    para_counts = _count_tokens_batch(paragraphs)
    expanded: list[str] = []
    for para, n in zip(paragraphs, para_counts):
        if n > max_tokens:
            lines = para.splitlines()
            expanded.extend(lines)
        else:
            expanded.append(para)
    paragraphs = [p for p in expanded if p.strip()]
    known: dict[str, int] = dict(zip(
        paragraphs, _count_tokens_batch(paragraphs)))

    chunks, current, current_tokens = [], [], 0
    for para in paragraphs:
        para_tokens = known[para]
        # If even a single line is too big, force-split by chars
        if para_tokens > max_tokens:
            forced = _force_split(para, max_tokens)
            for f in forced:
                ft = _count_tokens(f)
                if current_tokens + ft > max_tokens and current:
                    chunks.append("\n".join(current))
                    overlap_paras, overlap_count = [], 0
                    for p in reversed(current):
                        t = known.get(p) or _count_tokens(p)
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
                t = known.get(p) or _count_tokens(p)
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
    """Split text into pieces each respecting the token limit.

    Uses binary search over character prefixes to find the longest
    substring that fits within max_tokens. This is slower than a pure
    character split but guarantees token compliance.
    """
    parts: list[str] = []
    remaining = text
    while remaining:
        if _count_tokens(remaining) <= max_tokens:
            parts.append(remaining)
            break
        lo, hi = 1, len(remaining)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _count_tokens(remaining[:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            # Should not happen with a real tokenizer, but guard anyway
            lo = len(remaining)
        parts.append(remaining[:lo])
        remaining = remaining[lo:].lstrip()
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
    stripped = [p.strip() for p in parts]
    part_counts = dict(zip(stripped, _count_tokens_batch(stripped)))
    for part in stripped:
        if not part:
            continue
        # If a single part is huge, flush current first, then split it
        if part_counts[part] > MAX_TOKENS:
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
                current, current_tokens = [], 0
            sub_parts = _split_by_tokens(part, MAX_TOKENS, OVERLAP_TOKENS)
            for sub in sub_parts:
                chunks.append(
                    Chunk(
                        text=sub,
                        heading_path=list(heading_path),
                        chunk_index=len(chunks),
                        pattern=pattern,
                        has_image=bool(IMAGE_RE.search(sub)),
                    )
                )
            continue
        t = part_counts[part]
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
    """Split a table buffer into chunks, repeating a real header if present."""
    lines = table_text.strip("\n").splitlines()
    if not lines:
        return []

    # Detect markdown table header: [header row, separator row, ...]
    header_lines: list[str] = []
    data_lines: list[str] = list(lines)
    if len(lines) >= 2 and _is_table_separator(lines[1]):
        header_lines = lines[:2]
        data_lines = lines[2:]

    header_text = "\n".join(header_lines) + "\n" if header_lines else ""
    header_tokens = _count_tokens(header_text) if header_lines else 0
    if header_tokens >= MAX_TOKENS:
        # Pathological "header" (e.g. giant ASCII-grid separator lines):
        # it would consume the whole token budget, making max_row_tokens <= 0
        # and force-splitting every row into 1-char pieces (chunk text
        # amplification of thousands of x). Treat as no header instead.
        header_lines = []
        data_lines = list(lines)
        header_text = ""
        header_tokens = 0
    max_row_tokens = MAX_TOKENS - header_tokens

    # One batched count for all rows instead of one tokenizer call per row.
    row_counts = _count_tokens_batch(data_lines)

    # Oversized rows are force-split independently of each other; run those
    # in parallel (the tokenizer releases the GIL) and reassemble in row
    # order. Outputs are identical to sequential per-row force-splits.
    pieces_by_row: dict[int, list[str]] = {}
    oversized = [i for i, t in enumerate(row_counts) if t > max_row_tokens]
    if oversized:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(8, len(oversized))) as ex:
            for i, pieces in zip(oversized, ex.map(
                    lambda i: _force_split(data_lines[i], max_row_tokens),
                    oversized)):
                pieces_by_row[i] = pieces

    chunks: list[Chunk] = []
    batch: list[str] = []
    batch_tokens = header_tokens

    def _flush_batch() -> None:
        if not batch:
            return
        chunks.append(
            Chunk(
                text=header_text + "\n".join(batch),
                heading_path=list(heading),
                chunk_index=len(chunks),
                pattern=pattern,
                has_image=False,
            )
        )
        batch.clear()

    for i, row in enumerate(data_lines):
        row_tokens = row_counts[i]
        if row_tokens > max_row_tokens:
            _flush_batch()
            # Row alone exceeds the budget; force-split it.
            for piece in pieces_by_row[i]:
                chunks.append(
                    Chunk(
                        text=header_text + piece,
                        heading_path=list(heading),
                        chunk_index=len(chunks),
                        pattern=pattern,
                        has_image=False,
                    )
                )
            batch_tokens = header_tokens
            continue

        if batch_tokens + row_tokens > MAX_TOKENS and batch:
            _flush_batch()
            batch_tokens = header_tokens

        batch.append(row)
        batch_tokens += row_tokens

    _flush_batch()
    return chunks


def _is_table_line(line: str) -> bool:
    """Return True if the line belongs to a markdown table.

    Markdown tables may use `|` for data rows and `+` for separator rows
    (e.g. from marker-pdf conversions).  Bulleted lists use `+ `, so we
    exclude those.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("|"):
        return True
    if stripped.startswith("+") and not stripped.startswith("+ "):
        return True
    return False


def _is_table_separator(line: str) -> bool:
    """Return True if the line is a markdown table separator (e.g. |---|---| or +---+---+) ."""
    stripped = line.strip()
    if not stripped:
        return False
    if not (stripped.startswith(("|", "+")) and stripped.endswith(("|", "+"))):
        return False
    inner = stripped[1:-1]
    # Must contain at least one column separator dash/colon and only
    # allowed separator characters otherwise.
    return ("-" in inner or ":" in inner) and all(c in "-:+| \t" for c in inner)


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
