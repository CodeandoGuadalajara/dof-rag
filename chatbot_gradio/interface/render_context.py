# -*- coding: utf-8 -*-
"""Context rendering utilities for RAG chat system UI."""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Import temporal utilities
try:
    from config.config import calculate_document_age, extract_date_from_title
except ImportError:
    logger.warning("Could not import temporal utilities from config")
    def calculate_document_age(doc_date, ref_date=None):
        """Fallback function for calculating document age"""
        return {"age_description": "N/A", "emoji": "❓", "label": "Desconocida", "category": "unknown"}
    def extract_date_from_title(title: str) -> Any:
        """Fallback function for extracting date from title."""
        return None

# Constants for formatting
FRAGMENT_CONTAINER_HEIGHT = 150
DOCUMENT_EMOJI = "📄"
FRAGMENT_BULLET = "▪"

# Consolidated style dictionary for inline CSS
STYLE_CONFIG = {
    "details_base": "margin-bottom: 6px; margin-left: 0; border: 1px solid var(--border-color-accent); border-radius: 6px; padding: 0; background: var(--context-background);",
    "details_original": "margin-bottom: 0.3em; margin-left: 0 !important; padding: 0 !important; border: 1px solid var(--border-color-accent); border-radius: 6px; background: var(--background-fill-primary);",
    "summary_base": "cursor: pointer; font-weight: 600; font-size: 0.9em; color: var(--body-text-color); padding: 6px; margin: 0; background: var(--background-fill-secondary); border-radius: 6px 6px 0 0; border: none;",
    "summary_original": "transition: background-color 0.2s ease; padding: 6px !important; margin: 0 !important; background: var(--background-fill-secondary); border-radius: 6px 6px 0 0; color: var(--body-text-color); font-weight: 500; font-size: 0.85em; border: none !important;",
    "content_base": "margin-top: 0; padding: 6px; margin-left: 0; border-left: 2px solid var(--color-accent-soft); background: var(--background-fill-primary); border-radius: 0 0 6px 6px;",
    "content_original": "background: var(--background-fill-primary); color: var(--body-text-color); padding: 4px; margin-left: 0 !important; padding-left: 4px !important; border-radius: 0 0 4px 4px; font-size: 0.8em; line-height: 1.3;",
    "chunk_summary_base": "cursor: pointer; padding: 4px; margin-left: 0; font-weight: 500; font-size: 0.8em; color: var(--body-text-color); background: var(--context-background); border-radius: 3px 3px 0 0;",
    "chunk_summary_original": "cursor: pointer; padding: 4px; margin-left: 0; font-weight: 500; font-size: 0.8em; color: var(--body-text-color); background: var(--background-fill-secondary); border-radius: 3px 3px 0 0;",
    "error_container": "color: var(--color-error); padding: 8px; margin: 4px 0; border: 2px solid var(--color-error); border-radius: 4px; font-size: 0.9em; background: var(--background-fill-secondary);"
}

# Default age info for fallback cases
DEFAULT_AGE_INFO = {
    "formatted_date": "Fecha no disponible",
    "age_description": "N/A",
    "emoji": "❓",
    "label": "Desconocida",
    "recency_factor": 1.0,
    "category": "unknown"
}


def _get_age_info_from_title(title: str) -> Dict[str, Any]:
    """Get age info from document title with error handling."""
    try:
        if title:
            doc_date = extract_date_from_title(title)
            return calculate_document_age(doc_date)
    except Exception:
        pass
    return DEFAULT_AGE_INFO


def _enrich_sources_with_age_info(sources: List[Dict[str, Any]]) -> None:
    """Enrich each source with age_info calculated from its title."""
    for source in sources:
        title = source.get("title")
        source["age_info"] = _get_age_info_from_title(title)


def _get_age_text(source: Dict[str, Any]) -> str:
    """Get formatted age text for a source document."""
    age_info = source.get("age_info")
    if age_info:
        return f"{age_info['formatted_date']} ({age_info['age_description']}) {age_info['emoji']}"
    
    # Fallback: try to get age info from title
    title = source.get("title")
    if title:
        age_info = _get_age_info_from_title(title)
        return f"{age_info['formatted_date']} ({age_info['age_description']}) {age_info['emoji']}"
    
    return "Fecha no disponible ❓"


def _get_similarity_indicator(similarity: float) -> tuple:
    """Get similarity color and icon based on score."""
    if similarity >= 0.8:
        return "var(--color-success)", "🟢"
    elif similarity >= 0.6:
        return "var(--color-warning)", "🟡"
    else:
        return "var(--color-error)", "🔴"


def _process_markdown_to_html(text: str) -> str:
    """Convert basic Markdown formatting to HTML for better display."""
    if not text:
        return text
    
    processed_text = text
    
    # Escape HTML characters to prevent XSS
    processed_text = processed_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Process headers (order matters: longer patterns first)
    processed_text = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', processed_text, flags=re.MULTILINE)
    
    # Process bold text
    processed_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_text)
    processed_text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', processed_text)
    
    # Process italic text (avoid conflicts with bold)
    processed_text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<em>\1</em>', processed_text)
    processed_text = re.sub(r'(?<!_)_([^_]+?)_(?!_)', r'<em>\1</em>', processed_text)
    
    # Convert line breaks to HTML breaks
    processed_text = processed_text.replace('\n', '<br>')
    
    return processed_text


def _format_chunk_collapsible(chunk: Dict[str, Any], index: int, use_original_style: bool = False) -> str:
    """Format a single chunk as a collapsible section."""
    try:
        text = chunk.get("text", "")
        header = chunk.get("header", "")
        similarity = chunk.get("similarity", 0)
        
        formatted_text = _process_markdown_to_html(text)
        similarity_percentage = f"{similarity:.1%}" if similarity > 0 else "N/A"
        similarity_color, similarity_icon = _get_similarity_indicator(similarity)
        
        chunk_title = f"Fragmento {index}"
        if header:
            chunk_title += f" - {header}"
        
        # Choose styling based on use case
        summary_style = STYLE_CONFIG["chunk_summary_original" if use_original_style else "chunk_summary_base"]
        
        chunk_html = f"""
<details style="margin: 2px 0; margin-left: 0; border: 1px solid var(--border-color-accent); border-radius: 4px; background: var(--background-fill-primary);">
<summary style="{summary_style}">
    {FRAGMENT_BULLET} {chunk_title}
    <span style="float: right; font-size: 0.75em; color: {similarity_color}; font-weight: 600;">
        {similarity_icon} {similarity_percentage}
    </span>
</summary>

<div style="padding: 4px; margin-left: 0; background: var(--background-fill-primary); border-top: 1px solid var(--border-color-primary); max-height: {FRAGMENT_CONTAINER_HEIGHT}px; overflow-y: auto; line-height: 1.4; font-size: 0.75em; border-radius: 0 0 3px 3px;">
    <div style="background: var(--background-fill-secondary); padding: 4px; margin-left: 0; border-radius: 2px; color: var(--body-text-color); font-weight: 400; border: 1px solid var(--border-color-primary); font-family: 'Segoe UI', system-ui, sans-serif;">
        {formatted_text}
    </div>
</div>
</details>
"""
        
        return chunk_html
        
    except Exception as e:
        logger.error(f"Failed to format chunk: {e}")
        return f"""
<div style="{STYLE_CONFIG['error_container']}">
    {FRAGMENT_BULLET} Error al formatear fragmento: {str(e)}
</div>
"""


def _format_document_section(source: Dict[str, Any], index: int, use_original_style: bool = False) -> str:
    """Format a single document section with collapsible UX design."""
    try:
        title = source.get("title", "Documento sin título")
        chunks = source.get("chunks", [])
        url = source.get("url", "")
        file_path = source.get("file_path", "")
        
        # Calculate document statistics
        num_chunks = len(chunks)
        avg_similarity = sum(chunk.get("similarity", 0) for chunk in chunks) / num_chunks if chunks else 0
        best_similarity = max(chunk.get("similarity", 0) for chunk in chunks) if chunks else 0
        
        age_text = _get_age_text(source)
        
        # Choose styling based on use case
        style_key = "original" if use_original_style else "base"
        details_style = STYLE_CONFIG[f"details_{style_key}"]
        summary_style = STYLE_CONFIG[f"summary_{style_key}"]
        content_style = STYLE_CONFIG[f"content_{style_key}"]
        
        document_html = f"""
<details open style="{details_style}">
<summary style="{summary_style}">
    {DOCUMENT_EMOJI} {title}
    <span style="font-size: 0.75em; color: var(--body-text-color-subdued); font-weight: 400;">
        ({num_chunks} fragmentos • Mejor: {best_similarity:.1%} • Prom: {avg_similarity:.1%})
    </span>
</summary>

<div style="{content_style}">
"""
        
        # Add date information
        document_html += f"""
<div style="margin-bottom: 4px; margin-left: 0; font-size: 0.8em; color: var(--body-text-color); background: var(--background-fill-secondary); padding: 3px; border-radius: 3px; font-weight: 500;">
    📅 {age_text}
</div>"""
        
        # Add metadata if available
        if url:
            document_html += f"""
<div style="margin-bottom: 4px; margin-left: 0; font-size: 0.8em; color: var(--body-text-color); background: var(--background-fill-secondary); padding: 3px; border-radius: 3px;">
    🔗 <a href="{url}" target="_blank" style="color: var(--color-accent); text-decoration: none; font-weight: 500;">{url}</a>
</div>"""
        elif file_path:
            document_html += f"""
<div style="margin-bottom: 4px; margin-left: 0; font-size: 0.8em; color: var(--body-text-color); font-weight: 500; background: var(--background-fill-secondary); padding: 3px; border-radius: 3px;">
    📁 {file_path}
</div>"""
        
        # Add chunks
        for i, chunk in enumerate(chunks, 1):
            chunk_html = _format_chunk_collapsible(chunk, i, use_original_style)
            document_html += chunk_html
        
        document_html += """
</div>
</details>
"""
        
        return document_html
        
    except Exception as e:
        logger.error(f"Failed to format document section: {e}")
        return f"""
<div style="{STYLE_CONFIG['error_container']}">
    {DOCUMENT_EMOJI} Error al formatear documento: {str(e)}
</div>
"""


def render_sources(sources: List[Dict[str, Any]]) -> str:
    """Render sources into formatted HTML for UI display."""
    try:
        if not sources:
            return """
<div style="text-align: center; padding: 2em; color: var(--body-text-color-subdued); font-style: italic; background: var(--background-fill-secondary); border-radius: 8px; border: 2px solid var(--border-color-primary);">
    No se encontraron fuentes relevantes para esta consulta.
</div>
"""
        _enrich_sources_with_age_info(sources)
        
        html_content = """
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif; line-height: 1.5;">
"""
        
        # Add header with count
        html_content += f"""
<div style="margin-bottom: 6px; margin-left: 0; padding: 6px; background: var(--color-accent); border-radius: 6px; border-left: 3px solid var(--color-accent-soft);">
    <h4 style="margin: 0; margin-left: 0; color: white; font-size: 0.9em; font-weight: 600;">
        📚 Fuentes Consultadas ({len(sources)} documentos)
    </h4>
</div>
"""
        
        # Add each document section
        for i, source in enumerate(sources, 1):
            document_section = _format_document_section(source, i, use_original_style=False)
            html_content += document_section
        
        html_content += """
</div>
"""
        
        return html_content
        
    except Exception as e:
        logger.error(f"Failed to render sources: {e}")
        return f"""
<div style="{STYLE_CONFIG['error_container']}">
    <h4 style="margin: 0 0 8px 0; color: var(--color-error);">Error al renderizar las fuentes</h4>
    <p style="margin: 0; font-size: 0.9em; color: var(--body-text-color);">{str(e)}</p>
</div>
"""


def render_summary(sources: List[Dict[str, Any]]) -> str:
    """Render a comprehensive summary of the sources found."""
    try:
        if not sources:
            return "❌ **No se encontraron fuentes relevantes.**"
        
        num_documents = len(sources)
        total_chunks = sum(len(source.get("chunks", [])) for source in sources)
        
        # Calculate similarity statistics
        all_similarities = []
        for source in sources:
            for chunk in source.get("chunks", []):
                similarity = chunk.get("similarity", 0)
                if similarity > 0:
                    all_similarities.append(similarity)
        
        if all_similarities:
            avg_similarity = sum(all_similarities) / len(all_similarities)
            max_similarity = max(all_similarities)
        else:
            avg_similarity = max_similarity = 0
        
        # Calculate temporal statistics
        recent_docs = 0
        old_docs = 0
        for source in sources:
            age_info = source.get("age_info")
            if age_info:
                if age_info["category"] in ["very_recent", "recent"]:
                    recent_docs += 1
                elif age_info["category"] == "old":
                    old_docs += 1
                        
        # Quality assessment
        if avg_similarity >= 0.8:
            quality_icon = "🟢"
            quality_text = "Excelente"
        elif avg_similarity >= 0.6:
            quality_icon = "🟡"
            quality_text = "Buena"
        elif avg_similarity >= 0.4:
            quality_icon = "🟠"
            quality_text = "Regular"
        else:
            quality_icon = "🔴"
            quality_text = "Baja"
        
        # Count high-quality chunks
        high_quality_chunks = sum(1 for sim in all_similarities if sim >= 0.7)
        
        summary = f"""<div style="margin-top: 8px; padding: 8px; background: var(--color-accent-soft); border-radius: 6px; border: 1px solid var(--border-color-accent); color: var(--body-text-color); font-size: 0.75em;">
<strong>📊 Resumen de Consulta</strong><br/>
<strong>{num_documents}</strong> docs • <strong>{total_chunks}</strong> frags • Calidad: <strong>{quality_icon} {quality_text}</strong><br/>
Alta: <strong>{high_quality_chunks}</strong> • Máx: <strong style="color: var(--color-success);">{max_similarity:.1%}</strong> • Prom: <strong style="color: var(--color-info);">{avg_similarity:.1%}</strong><br/>
Recientes: <strong style="color: var(--color-success);">🟢 {recent_docs}</strong> • Antiguos: <strong style="color: var(--color-error);">🔴 {old_docs}</strong>
</div>"""
        return summary
        
    except Exception as e:
        logger.error(f"Failed to render summary: {e}")
        return "❌ **Error al generar resumen de fuentes.**"


def render_embedded_sources(sources: List[Dict[str, Any]], query_id: str) -> str:
    """Render sources as embedded collapsible section using original styling."""
    try:
        if not sources:
            return ""
        
        num_docs = len(sources)
        
        # Create embedded container with original-style content
        embedded_html = f"""<div class="embedded-sources-container" data-query-id="{query_id}" style="scroll-behavior: smooth;">
    <details class="embedded-sources" open>
        <summary class="embedded-sources-summary">
            📚 <strong>Fuentes consultadas ({num_docs} documentos)</strong>
        </summary>
        <div class="embedded-sources-content" style="scroll-behavior: smooth;">
            {_render_sources_original_format(sources)}
        </div>
    </details>
</div>"""
        
        return embedded_html
        
    except Exception as e:
        logger.error(f"Failed to render embedded sources: {e}")
        return f'<div style="color: var(--color-error); padding: 8px; margin: 4px 0; border: 2px solid var(--color-error); border-radius: 4px; font-size: 0.9em; background: var(--background-fill-secondary);">Error al mostrar fuentes para consulta {query_id}</div>'


def _render_sources_original_format(sources: List[Dict[str, Any]]) -> str:
    """Render sources using the exact original format from render_sources."""
    try:
        html_content = ""
        
        # Process each document using original format
        for i, source in enumerate(sources, 1):
            document_section = _format_document_section(source, i, use_original_style=True)
            html_content += document_section
        
        # Add the summary at the end
        summary_html = render_summary(sources)
        html_content += summary_html
        
        return html_content
        
    except Exception as e:
        logger.error(f"Failed to render sources in original format: {e}")
        return '<div style="color: var(--color-error); padding: 8px; margin: 4px 0; border: 2px solid var(--color-error); border-radius: 4px; font-size: 0.9em; background: var(--background-fill-secondary);">Error al renderizar fuentes</div>'