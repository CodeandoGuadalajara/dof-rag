"""Safe Markdown rendering for retrieved evidence chunks.

Chunk text comes from the parsed DOF corpus and frequently contains Markdown
(headings, lists, emphasis). This module renders that Markdown into sanitized
HTML for the human-evaluation UI:

- ``markdown-it-py`` (``default`` preset: CommonMark plus tables and
  strikethrough) parses the source. Raw HTML stays disabled, so any markup
  embedded in the chunk is escaped instead of rendered, and ``breaks=True``
  keeps single newlines visible because legal texts are hard-wrapped.
- ``nh3`` (Rust ammonia bindings) sanitizes the rendered HTML against an
  explicit allowlist as defense in depth.
- If rendering fails for any reason, the caller receives the escaped plain
  text wrapped in a ``<p class="chunk-text">`` element, preserving the
  previous plain-text behaviour.
"""

from __future__ import annotations

import html
from typing import Any

import nh3
from markdown_it import MarkdownIt

_ALLOWED_TAGS = {
    "a",
    "blockquote",
    "br",
    "caption",
    "code",
    "del",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "strong",
    "sub",
    "sup",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}

_ALLOWED_ATTRIBUTES = {
    "a": {"href", "title"},
    "code": {"class"},
    "td": {"align"},
    "th": {"align"},
}

_RENDERER: MarkdownIt | None = None


def _markdown_renderer() -> MarkdownIt:
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = MarkdownIt("default", {"breaks": True})
    return _RENDERER


def render_markdown_html(text: Any) -> str:
    """Return sanitized HTML for chunk *text*.

    Falls back to escaped plain text (newlines preserved via the
    ``chunk-text`` CSS rule) if Markdown rendering or sanitization fails.
    """
    if text is None:
        return ""
    source = str(text)
    if not source.strip():
        return ""
    try:
        rendered = _markdown_renderer().render(source)
        return nh3.clean(
            rendered,
            tags=_ALLOWED_TAGS,
            attributes=_ALLOWED_ATTRIBUTES,
            link_rel="noopener noreferrer nofollow",
        )
    except Exception:
        return f'<p class="chunk-text">{html.escape(source)}</p>'
