"""Read-only corpus passage rendering; never reads conversation records."""
from datetime import date
from html import escape

from agent_tools.models import EvidenceHit


def passage_body(chunk_id: int, passages: list[EvidenceHit]) -> str:
    selected = next(hit for hit in passages if hit.chunk_id == chunk_id)
    title = escape(selected.title or "Publicación del DOF")
    publication = escape(selected.publication_date or "Fecha no disponible")
    official = ""
    if selected.publication_date:
        try:
            day = date.fromisoformat(selected.publication_date)
        except ValueError:
            pass
        else:
            url = f"https://www.dof.gob.mx/index.php?year={day.year}&month={day.month}&day={day.day}"
            official = f'<p><a href="{escape(url, quote=True)}">Consultar las ediciones de esta fecha en el DOF oficial</a></p>'
    sections = []
    for hit in passages:
        cited = hit.chunk_id == chunk_id
        label = "Pasaje seleccionado" if cited else "Contexto cercano"
        heading = escape(" › ".join(hit.heading_path))
        link = f'/pasajes/{hit.chunk_id}#pasaje-{hit.chunk_id}'
        highlight = ' style="border:2px solid var(--accent-dark)"' if cited else ""
        sections.append(
            f'<section class="card" id="pasaje-{hit.chunk_id}"{highlight}>'
            f'<h2>{label} · <a href="{link}">chunk {hit.chunk_id}</a></h2>'
            f'<p class="meta">{heading}</p><p class="chunk-text">{escape(hit.text)}</p></section>'
        )
    return (
        f'<h1>{title}</h1><p>Fecha de publicación: {publication} · '
        f'Sección: {escape(selected.section or "No disponible")}</p>{official}'
        '<p role="note">Texto extraído de nuestra base; puede contener errores u omisiones, '
        'especialmente en publicaciones históricas. Se muestra un pasaje y su contexto cercano, '
        'no el documento completo. La fecha de publicación no determina la vigencia legal. '
        'Abrir un pasaje no verifica las afirmaciones de una respuesta.</p>'
        + "".join(sections)
    )
