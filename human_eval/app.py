"""Single-process Air application for controlled human evaluation of the DOF agent.

Authentication is provider-agnostic: ``create_app`` receives an
``auth.AuthBackend`` and never imports a provider. Production wires Clerk via
``build_default_app``; tests inject ``auth.FakeAuthBackend``.

Visibility model:
- anonymous visitors can read published answers (terminal, published runs);
- signed-in users can ask questions (rolling 24h quota; each question,
  including the first, requires reviewing a published answer first) and can
  evaluate any published answer as well as their own;
- admins (Clerk ``public_metadata.role == "admin"``) publish/unpublish
  answers and are exempt from the quota and the feedback gate.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import html
import json
import os
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import air
import uvicorn
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from .agent_executor import AgentExecutorConfig, AgentRunExecutor
from .auth import AuthBackend, User
from .contracts import ContractError, FeedbackRequest, RunRequest
from .markdown_render import render_markdown_html
from .service import (
    ActiveRunError,
    EvaluationService,
    IdempotencyConflictError,
    PublicExecutionError,
    QueueFullError,
    QuotaExceededError,
    ReviewRequiredError,
)
from .store import SCHEMA_VERSION, EvaluationStore

MAX_BODY_BYTES = 16 * 1024
ACTIVE_STATES = frozenset({"queued", "running"})
STATUS_LABELS = {
    "queued": "En cola",
    "running": "Consultando el DOF",
    "succeeded": "Respuesta terminada",
    "failed": "Ejecución fallida",
}
PROBLEM_LABELS = {
    "incorrect_answer": "Respuesta incorrecta",
    "missing_evidence": "Falta evidencia",
    "bad_citation": "Cita incorrecta",
    "incomplete_coverage": "Cobertura incompleta",
    "cutoff_error": "Error en fecha de corte",
    "hard_to_understand": "Difícil de entender",
    "other": "Otro",
}
PROGRESS_LABELS = {
    "agent_started": "Inicio",
    "model_turn_started": "Análisis",
    "tool_started": "Siguiente acción",
    "tool_completed": "Hallazgo",
    "answer_revision_requested": "Revisión",
    "verification_completed": "Verificación",
}


@dataclass(frozen=True)
class WebSettings:
    host: str
    port: int
    db_path: Path
    session_secret: str
    secure_cookie: bool = False
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver")
    session_max_age: int = 12 * 60 * 60
    daily_question_limit: int = 1
    queue_capacity: int = 20

    @classmethod
    def from_env(cls, repo_root: Path) -> "WebSettings":
        session_secret = os.environ.get("DOF_SESSION_SECRET", "")
        if len(session_secret) < 32:
            raise ValueError("set DOF_SESSION_SECRET to at least 32 characters")
        allowed_hosts = tuple(
            value.strip()
            for value in os.environ.get(
                "DOF_ALLOWED_HOSTS", "127.0.0.1,localhost"
            ).split(",")
            if value.strip()
        )
        return cls(
            host=os.environ.get("DOF_WEB_HOST", "127.0.0.1"),
            port=int(os.environ.get("DOF_WEB_PORT", "8765")),
            db_path=Path(
                os.environ.get(
                    "DOF_HUMAN_EVAL_DB", repo_root / "var/human_evaluation.sqlite"
                )
            ),
            session_secret=session_secret,
            secure_cookie=os.environ.get("DOF_SECURE_COOKIE", "false").lower()
            in {"1", "true", "yes"},
            allowed_hosts=allowed_hosts,
            session_max_age=int(
                os.environ.get("DOF_SESSION_MAX_AGE", str(12 * 60 * 60))
            ),
            daily_question_limit=int(os.environ.get("DOF_DAILY_QUESTION_LIMIT", "1")),
            queue_capacity=int(os.environ.get("DOF_QUEUE_CAPACITY", "20")),
        )


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _csrf(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not isinstance(token, str) or len(token) < 32:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token


def _csrf_valid(request: Request, submitted: Any) -> bool:
    expected = request.session.get("csrf_token")
    return (
        isinstance(expected, str)
        and isinstance(submitted, str)
        and hmac.compare_digest(expected, submitted)
    )


def _safe_next(raw: Any, default: str) -> str:
    """Same-origin absolute paths only; blocks open redirects."""
    if not isinstance(raw, str):
        return default
    raw = raw.strip()
    if not raw.startswith("/") or raw.startswith("//"):
        return default
    return raw


async def _form(request: Request) -> Any:
    raw_length = request.headers.get("content-length", "0")
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ContractError("invalid Content-Length header") from exc
    if length > MAX_BODY_BYTES:
        raise ContractError("request body is too large")
    return await request.form()


STYLE = """
:root { --paper:#f6f2e8; --ink:#17201b; --muted:#5f675f; --line:#cfc8b8;
  --accent:#176b4a; --accent-dark:#0f4d35; --warn:#8a3f18; --panel:#fffdf7; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:var(--paper); font-family:Inter,ui-sans-serif,
  system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.55; }
a { color:var(--accent-dark); }
.shell { width:min(980px,calc(100% - 2rem)); margin:0 auto; padding:2rem 0 5rem; }
header { display:flex; gap:1rem; align-items:flex-start; justify-content:space-between;
  border-bottom:1px solid var(--line); margin-bottom:2.5rem; padding-bottom:1.25rem; }
header .session { align-items:center; display:flex; gap:.75rem; }
.eyebrow { color:var(--accent); font-size:.76rem; font-weight:800; letter-spacing:.12em;
  margin:0 0 .35rem; text-transform:uppercase; }
h1,h2,h3 { font-family:Georgia,"Times New Roman",serif; line-height:1.14; margin-top:0; }
h1 { font-size:clamp(2rem,5vw,3.8rem); font-weight:500; letter-spacing:-.035em; margin-bottom:.6rem; }
h2 { font-size:1.65rem; font-weight:500; }
h3 { font-size:1.18rem; margin-bottom:.45rem; }
.lede { color:var(--muted); max-width:68ch; margin:0; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:4px;
  box-shadow:0 12px 32px rgba(30,35,28,.06); padding:clamp(1.1rem,3vw,2rem); margin:1.25rem 0; }
.login-panel { max-width:30rem; margin:2.5rem auto; text-align:center; }
.login-panel .lede { margin:0 auto 1.5rem; }
.sign-in { display:flex; justify-content:center; }
.grid { display:grid; gap:1rem; grid-template-columns:repeat(2,minmax(0,1fr)); }
.field { display:flex; flex-direction:column; gap:.4rem; margin-bottom:1rem; }
.field.full { grid-column:1/-1; }
label,legend { font-size:.9rem; font-weight:750; }
input,textarea,select,button { font:inherit; }
input,textarea,select { width:100%; color:var(--ink); background:#fff; border:1px solid #999486;
  border-radius:3px; padding:.72rem .78rem; }
textarea { min-height:9rem; resize:vertical; }
input:focus,textarea:focus,select:focus,button:focus { outline:3px solid rgba(23,107,74,.24);
  outline-offset:2px; border-color:var(--accent); }
button,.button { display:inline-block; border:0; border-radius:3px; color:white; background:var(--accent);
  cursor:pointer; font-weight:750; padding:.72rem 1rem; text-decoration:none; }
button:hover,.button:hover { background:var(--accent-dark); }
.secondary { background:transparent; border:1px solid var(--line); color:var(--ink); }
.danger { background:var(--warn); }
.danger:hover { background:#6d3013; }
.status { border-left:5px solid var(--accent); }
.status[data-state="failed"],.warning { border-left-color:var(--warn); }
.meta { color:var(--muted); font-size:.86rem; }
.tag { background:#e5eee7; border-radius:99px; color:var(--accent-dark); display:inline-block;
  font-size:.78rem; font-weight:750; padding:.18rem .55rem; }
.warning { background:#fff3e8; border-left:5px solid var(--warn); padding:.85rem 1rem; }
.answer { font-family:Georgia,"Times New Roman",serif; font-size:1.14rem; white-space:pre-wrap; }
details { border-top:1px solid var(--line); padding:.8rem 0; }
summary { cursor:pointer; font-weight:700; }
pre { background:#18201c; color:#e9eee9; border-radius:3px; max-height:28rem; overflow:auto;
  padding:1rem; white-space:pre-wrap; word-break:break-word; }
.checks { display:grid; gap:.5rem; grid-template-columns:repeat(2,minmax(0,1fr)); }
.check { align-items:flex-start; display:flex; gap:.5rem; font-weight:500; }
.check input { margin-top:.3rem; width:auto; }
.run-list { list-style:none; margin:0; padding:0; }
.run-list li { border-top:1px solid var(--line); padding:.8rem 0; }
.run-list a { display:block; font-weight:700; }
.notice { color:var(--accent-dark); font-weight:700; }
.stream-state { align-items:center; display:flex; gap:.55rem; }
.stream-state::before { animation:pulse 1.2s ease-in-out infinite; background:var(--accent);
  border-radius:50%; content:""; height:.55rem; width:.55rem; }
.activity { list-style:none; margin:1.25rem 0 0; padding:0; }
.activity li { border-top:1px solid var(--line); padding:.85rem 0 .85rem 1.4rem; position:relative; }
.activity li::before { background:var(--accent); border:3px solid var(--panel); border-radius:50%;
  box-shadow:0 0 0 1px var(--accent); content:""; height:.62rem; left:.08rem; position:absolute;
  top:1.16rem; width:.62rem; }
.activity strong { display:block; }
.activity-kind { color:var(--accent); display:block; font-size:.72rem; font-weight:800;
  letter-spacing:.08em; text-transform:uppercase; }
.decision-why { color:var(--muted); margin:.3rem 0 .55rem; }
.document-chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.65rem 0 .15rem; }
.document-chip { background:#edf2ec; border:1px solid #cbd8cc; border-radius:3px; color:var(--ink);
  font-size:.8rem; padding:.28rem .48rem; }
.chunk-links { display:grid; gap:.45rem; margin:.65rem 0 .15rem; }
.activity .chunk-link { background:#f5f8f3; border:1px solid #cbd8cc; border-radius:3px; padding:0; }
.activity .chunk-link summary { color:var(--accent-dark); padding:.55rem .65rem; text-decoration:underline;
  text-decoration-thickness:1px; text-underline-offset:3px; }
.chunk-content { border-top:1px solid #d9e2da; padding:.15rem .65rem .65rem; }
.chunk-content > p { margin:.45rem 0 0; white-space:pre-wrap; }
.chunk-text { white-space:pre-wrap; }
.chunk-truncated { display:block; margin-top:.35rem; }
.markdown-body { max-width:100%; overflow-wrap:anywhere; }
.markdown-body > :first-child { margin-top:.45rem; }
.markdown-body p, .markdown-body ul, .markdown-body ol, .markdown-body blockquote,
.markdown-body pre, .markdown-body table { margin:.55rem 0 .2rem; }
.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4,
.markdown-body h5, .markdown-body h6 { font-family:Georgia,"Times New Roman",serif;
  font-weight:600; line-height:1.3; margin:.9rem 0 .3rem; }
.markdown-body h1 { font-size:1.12rem; } .markdown-body h2 { font-size:1.06rem; }
.markdown-body h3, .markdown-body h4, .markdown-body h5, .markdown-body h6 { font-size:1rem; }
.markdown-body blockquote { border-left:3px solid var(--line); color:var(--muted);
  padding-left:.7rem; }
.markdown-body pre { background:#eef2ec; border:1px solid var(--line); border-radius:3px;
  color:var(--ink); max-height:24rem; overflow:auto; padding:.55rem .7rem; }
.markdown-body code { font-size:.86em; }
.markdown-body p code, .markdown-body li code { background:#eef2ec; border-radius:3px;
  padding:.08rem .3rem; }
.markdown-body table { border-collapse:collapse; display:block; overflow-x:auto; }
.markdown-body th, .markdown-body td { border:1px solid var(--line); padding:.25rem .5rem;
  text-align:left; }
.markdown-body a { color:var(--accent-dark); }
.activity .citation-tags { margin:.55rem 0 0; }
.process-archive { border:1px solid var(--line); border-radius:3px; padding:.75rem 1rem; }
.process-archive > summary { color:var(--accent-dark); font-family:Georgia,"Times New Roman",serif;
  font-size:1.1rem; }
.process-archive[open] > summary { margin-bottom:.5rem; }
.process-note { color:var(--muted); font-size:.86rem; margin:.55rem 0 0; }
@keyframes pulse { 50% { opacity:.35; transform:scale(.8); } }
footer { border-top:1px solid var(--line); color:var(--muted); font-size:.82rem; margin-top:3rem;
  padding-top:1.25rem; }
@media (max-width:680px) { .grid,.checks { grid-template-columns:1fr; } header { display:block; }
  header .session { margin-top:1rem; } }
"""


STREAM_SCRIPT = """
(() => {
  const labels = {
    agent_started: 'Inicio', model_turn_started: 'Análisis', tool_started: 'Siguiente acción',
    tool_completed: 'Hallazgo', answer_revision_requested: 'Revisión',
    verification_completed: 'Verificación'
  };

  const replaceWithStatus = async (node) => {
    const url = node.dataset.statusUrl;
    if (!url) return false;
    try {
      const response = await fetch(url, {credentials: 'same-origin', cache: 'no-store'});
      if (response.status === 401) { location.href = '/login'; return; }
      if (!response.ok) throw new Error('status failed');
      const holder = document.createElement('div');
      holder.innerHTML = await response.text();
      const replacement = holder.firstElementChild;
      if (!replacement || replacement.dataset.state === 'queued' || replacement.dataset.state === 'running') {
        return false;
      }
      node.replaceWith(replacement);
      return true;
    } catch (_) { return false; }
  };

  const appendProgress = (list, event) => {
    if (list.querySelector(`[data-sequence="${event.sequence}"]`)) return;
    list.querySelector('[data-empty]')?.remove();
    const item = document.createElement('li');
    item.dataset.sequence = event.sequence;
    const kind = document.createElement('span');
    kind.className = 'activity-kind';
    kind.textContent = labels[event.event_type] || 'Actividad';
    item.appendChild(kind);
    const title = document.createElement('strong');
    title.textContent = event.payload?.message || 'El agente registró actividad.';
    item.appendChild(title);
    if (event.payload?.why) {
      const why = document.createElement('p');
      why.className = 'decision-why';
      why.textContent = event.payload.why;
      item.appendChild(why);
    }
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = event.created_at;
    item.appendChild(meta);
    if (event.payload?.documents?.length) {
      const documents = document.createElement('div');
      documents.className = 'document-chips';
      event.payload.documents.forEach(documentItem => {
        const chip = document.createElement('span');
        chip.className = 'document-chip';
        const name = documentItem.title || documentItem.path || `Documento ${documentItem.document_id}`;
        chip.textContent = documentItem.publication_date ? `${name} · ${documentItem.publication_date}` : name;
        documents.appendChild(chip);
      });
      item.appendChild(documents);
    }
    if (event.payload?.chunks?.length) {
      const chunks = document.createElement('div');
      chunks.className = 'chunk-links';
      event.payload.chunks.forEach(chunk => {
        const details = document.createElement('details');
        details.className = 'chunk-link';
        details.dataset.chunkId = chunk.chunk_id;
        const summary = document.createElement('summary');
        const heading = Array.isArray(chunk.heading_path) && chunk.heading_path.length
          ? ` · ${chunk.heading_path.join(' › ')}` : '';
        summary.textContent = `Chunk ${chunk.chunk_id} · documento ${chunk.document_id}${heading}`;
        const content = document.createElement('div');
        content.className = 'chunk-content';
        const location = document.createElement('span');
        location.className = 'meta';
        location.textContent = chunk.path || 'Ruta no disponible';
        content.appendChild(location);
        const excerpt = chunk.excerpt || chunk.snippet;
        const truncated = chunk.excerpt_truncated || chunk.snippet_truncated;
        if (chunk.excerpt_html) {
          // Server-rendered and sanitized Markdown (see human_eval/markdown_render.py).
          const body = document.createElement('div');
          body.className = 'markdown-body';
          body.innerHTML = chunk.excerpt_html;
          content.appendChild(body);
          if (truncated) {
            const more = document.createElement('span');
            more.className = 'meta chunk-truncated';
            more.textContent = '…';
            content.appendChild(more);
          }
        } else if (excerpt) {
          const text = document.createElement('p');
          text.textContent = `${excerpt}${truncated ? '…' : ''}`;
          content.appendChild(text);
        }
        details.append(summary, content);
        chunks.appendChild(details);
      });
      item.appendChild(chunks);
    }
    if (event.payload?.citation_ids?.length) {
      const citations = document.createElement('p');
      citations.className = 'citation-tags';
      event.payload.citation_ids.forEach(chunkId => {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.textContent = `cita: chunk ${chunkId}`;
        citations.appendChild(tag);
        citations.appendChild(document.createTextNode(' '));
      });
      item.appendChild(citations);
    }
    list.appendChild(item);
  };

  const start = (node) => {
    const streamUrl = node.dataset.streamUrl;
    if (!streamUrl) return;
    const list = node.querySelector('[data-progress-list]');
    const state = node.querySelector('[data-stream-state]');
    let last = Number(node.dataset.lastEventId || 0);
    if (!window.EventSource) {
      state.textContent = 'Actualizando el estado…';
      const timer = setInterval(async () => {
        if (await replaceWithStatus(node)) clearInterval(timer);
      }, 2000);
      return;
    }
    const source = new EventSource(`${streamUrl}?after=${last}`);
    source.addEventListener('open', () => { state.textContent = 'Conectado al trabajo del agente'; });
    source.addEventListener('progress', (message) => {
      const event = JSON.parse(message.data);
      last = Math.max(last, Number(event.sequence || 0));
      node.dataset.lastEventId = last;
      appendProgress(list, event);
      state.textContent = 'Recibiendo actividad en vivo';
    });
    source.addEventListener('terminal', async () => {
      source.close();
      state.textContent = 'Preparando el resultado…';
      await replaceWithStatus(node);
    });
    source.onerror = async () => {
      state.textContent = 'Reconectando al stream…';
      await replaceWithStatus(node);
    };
  };
  document.querySelectorAll('[data-stream-url]').forEach(start);
})();
"""

# Enhance "Entrar" links with Clerk's sign-in modal when Clerk JS is loaded
# (production). Without Clerk (tests, offline dev) the links keep their
# normal /login navigation as a fallback.
LOGIN_MODAL_SCRIPT = """
document.addEventListener('click', async (event) => {
  const link = event.target.closest('a[href^="/login"]');
  if (!link || !window.Clerk) return;
  event.preventDefault();
  await window.Clerk.load();
  const url = new URL(link.href, window.location.origin);
  const candidate = url.searchParams.get('next')
    || (window.location.pathname + window.location.search);
  const next = candidate.startsWith('/')
    && !candidate.startsWith('//')
    && !candidate.includes(String.fromCharCode(92))
    && !/[\\u0000-\\u001f\\u007f]/.test(candidate)
    ? candidate
    : '/';
  window.Clerk.openSignIn({ redirectUrl: next });
});
"""


def _page(
    title: str,
    body: str,
    *,
    user: User | None = None,
    csrf_token: str = "",
    page_scripts: Callable[[User | None], str] | None = None,
    trailing_scripts: str = "",
) -> str:
    if user is not None:
        session_area = (
            f'<span class="meta">{_escape(user.email or user.id)}'
            f"{' · <a href="/admin">admin</a>' if user.is_admin else ''}</span>"
            f'<form method="post" action="/logout"><input type="hidden" name="csrf_token" '
            f'value="{_escape(csrf_token)}"><button class="secondary" type="submit">Salir</button></form>'
        )
    else:
        session_area = '<a class="button secondary" href="/login">Entrar</a>'
    scripts = page_scripts(user) if page_scripts is not None else ""
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(title)} · Agente del DOF</title><style>{STYLE}</style></head>
<body><main class="shell"><header><div><p class="eyebrow">Piloto de investigación</p>
<a href="/" style="text-decoration:none;color:inherit"><strong>Agente del Diario Oficial</strong></a></div>
<div class="session">{session_area}</div></header>{body}
<footer>Las preguntas, respuestas, evidencias y evaluaciones se guardan para análisis y mejora del sistema.
Las respuestas publicadas son públicas. Las cuentas se gestionan con Clerk; no registramos direcciones IP.</footer>
</main><script>{STREAM_SCRIPT}</script><script>{LOGIN_MODAL_SCRIPT}</script>{scripts}{trailing_scripts}</body></html>"""


def _run_list_items(
    runs: list[dict[str, Any]], *, base: str, show_status: bool = True
) -> str:
    items = []
    for run in runs:
        status = (
            f"{_escape(STATUS_LABELS.get(run['status'], run['status']))} · "
            if show_status
            else ""
        )
        date = run.get("published_at") or run.get("created_at") or ""
        items.append(
            f'<li><a href="{base}/{_escape(run["run_id"])}">{_escape(run["question"])}</a>'
            f'<span class="meta">{status}{_escape(date)}</span></li>'
        )
    return "".join(items)


def _home_page(
    request: Request,
    *,
    user: User | None,
    published: list[dict[str, Any]],
    my_runs: list[dict[str, Any]] | None = None,
    needs_review: bool = False,
    review_target: dict[str, Any] | None = None,
    quota_reached: bool = False,
    error: str | None = None,
    values: dict[str, Any] | None = None,
    page_scripts: Callable[[User | None], str] | None = None,
) -> str:
    values = values or {}
    csrf_token = _csrf(request)
    error_html = (
        f'<p class="warning" role="alert">{_escape(error)}</p>' if error else ""
    )

    if user is None:
        ask_section = f"""<section class="panel"><h2>Participa</h2>{error_html}
<p class="lede">Crea una cuenta, evalúa una respuesta publicada y desbloquea tu primera
pregunta. Cada pregunta nueva pide una evaluación más: tu participación es lo que
mejora este piloto.</p>
<p><a class="button" href="/login">Entrar o crear cuenta</a></p></section>"""
    elif needs_review:
        first = not my_runs
        if review_target is not None:
            base = "/answers" if review_target["published"] else "/runs"
            review_html = (
                f'<p class="warning">Antes de hacer {"tu primera" if first else "otra"} '
                "pregunta, evalúa una respuesta, por ejemplo: "
                f'<a href="{base}/{_escape(review_target["run_id"])}">'
                f"{_escape(review_target['question'])}</a></p>"
            )
        else:
            review_html = (
                '<p class="warning">Aún no hay respuestas publicadas disponibles '
                "para evaluar. Vuelve a intentarlo pronto.</p>"
            )
        ask_section = f'<section class="panel"><h2>Nueva pregunta</h2>{error_html}{review_html}</section>'
    elif quota_reached:
        ask_section = f"""<section class="panel"><h2>Nueva pregunta</h2>{error_html}
<p class="notice">Ya enviaste tu pregunta de este periodo de 24 horas. Vuelve a intentar más tarde.</p></section>"""
    else:
        hops = str(values.get("required_hops", "1"))
        options = "".join(
            f'<option value="{number}"{" selected" if hops == str(number) else ""}>{number}</option>'
            for number in range(1, 6)
        )
        ask_section = f"""<section class="panel"><h2>Nueva pregunta</h2>{error_html}<form method="post" action="/runs">
<input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
<input type="hidden" name="client_request_id" value="{_escape(values.get("client_request_id") or uuid.uuid4())}">
<div class="grid"><div class="field full"><label for="question">Pregunta</label>
<textarea id="question" name="question" minlength="3" maxlength="2000" required placeholder="¿Qué establece el decreto y qué publicaciones deben compararse?">{_escape(values.get("question", ""))}</textarea></div>
<div class="field"><label for="as_of">Fecha de corte <span class="meta">(opcional)</span></label>
<input id="as_of" name="as_of" type="date" value="{_escape(values.get("as_of", ""))}"></div>
<div class="field"><label for="required_hops">Documentos mínimos</label>
<select id="required_hops" name="required_hops">{options}</select>
<span class="meta">Usa 2 o más para comparaciones que requieran fuentes distintas.</span></div></div>
<button type="submit">Iniciar consulta</button></form></section>"""

    my_runs_html = ""
    if user is not None and my_runs:
        items = _run_list_items(my_runs, base="/runs")
        my_runs_html = f'<section class="panel"><h2>Mis preguntas</h2><ul class="run-list">{items}</ul></section>'

    if published:
        published_html = f"""<section class="panel"><h2>Respuestas publicadas</h2>
<ul class="run-list">{_run_list_items(published, base="/answers", show_status=False)}</ul></section>"""
    else:
        published_html = """<section class="panel"><h2>Respuestas publicadas</h2>
<p class="meta">Aún no hay respuestas publicadas. Las primeras aparecerán aquí en cuanto
el equipo editorial las revise.</p></section>"""

    body = f"""<section><p class="eyebrow">Evaluación humana</p><h1>Pregunta, inspecciona, evalúa.</h1>
<p class="lede">El agente usa hoy recuperación léxica completa. Las respuestas revisadas por el equipo
editorial son públicas; cualquier persona con cuenta puede evaluarlas.</p></section>
{ask_section}{my_runs_html}{published_html}"""
    return _page(
        "Preguntas al DOF",
        body,
        user=user,
        csrf_token=csrf_token,
        page_scripts=page_scripts,
    )


def _status_fragment(
    run: dict[str, Any],
    *,
    csrf_token: str = "",
    feedback_recorded: bool = False,
    feedback_next: str = "",
) -> str:
    state = run["status"]
    meta = f'<p class="meta">Creada: {_escape(run["created_at"])}</p>'
    if state in ACTIVE_STATES:
        progress = run.get("progress", [])
        last_event_id = progress[-1]["sequence"] if progress else 0
        return f"""<section id="run-status" class="panel status" data-state="{state}"
data-stream-url="/runs/{_escape(run["run_id"])}/events"
data-status-url="/runs/{_escape(run["run_id"])}/status"
data-last-event-id="{_escape(last_event_id)}" aria-live="polite">
<p class="eyebrow">{_escape(STATUS_LABELS[state])}</p><h2>La ejecución sigue en progreso</h2>
<p>Registro público de decisiones: qué intenta localizar, por qué consulta cada fuente y qué evidencia encuentra.</p>
<p class="stream-state meta" data-stream-state>Conectando al trabajo del agente…</p>
{_progress_timeline(progress)}{meta}</section>"""
    if state == "failed":
        error = run.get("error", {})
        return f"""<section id="run-status" class="panel status" data-state="failed" aria-live="polite">
<p class="eyebrow">Ejecución fallida</p><h2>{_escape(error.get("message", "No se pudo completar la consulta."))}</h2>
<p class="meta">Código: {_escape(error.get("code", "internal_error"))}</p>{meta}
{_completed_process(run.get("progress", []), open_by_default=True)}</section>"""

    result = run["result"]
    answer = result.get("answer", {})
    coverage = result.get("coverage", {})
    warnings = list(result.get("warnings", []))
    warning_html = ""
    if not coverage.get("complete", False):
        missing = ", ".join(str(item) for item in coverage.get("missing", []))
        warning_html += (
            '<p class="warning"><strong>Cobertura incompleta.</strong> '
            + (
                _escape(f"Falta: {missing}.")
                if missing
                else "La ejecución no verificó toda la cobertura requerida."
            )
            + "</p>"
        )
    if warnings:
        warning_html += (
            f'<p class="meta">Advertencias técnicas: {_escape(", ".join(warnings))}</p>'
        )
    citation_ids = answer.get("citation_ids", [])
    citation_links = (
        " ".join(
            f'<a class="tag" href="#chunk-{_escape(chunk_id)}">chunk {_escape(chunk_id)}</a>'
            for chunk_id in citation_ids
        )
        or '<span class="meta">Sin citas resueltas</span>'
    )
    documents = (
        "".join(
            f"""<details><summary>{"Citado · " if item.get("cited") else ""}{_escape(item.get("title") or item.get("path") or "Documento")}</summary>
<p class="meta">Documento {_escape(item.get("document_id"))} · {_escape(item.get("publication_date") or "fecha no disponible")} · {_escape(item.get("institution") or "")}</p>
<p>{_escape(item.get("path"))}</p></details>"""
            for item in result.get("documents", [])
        )
        or '<p class="meta">No se registraron documentos.</p>'
    )
    evidence = (
        "".join(
            f"""<details id="chunk-{_escape(item.get("chunk_id"))}"><summary>{"Citado · " if item.get("cited") else ""}chunk {_escape(item.get("chunk_id"))}</summary>
<p class="meta">Documento {_escape(item.get("document_id"))} · {_escape(item.get("path"))}</p>
<div class="markdown-body">{render_markdown_html(item.get("text"))}</div></details>"""
            for item in result.get("evidence", [])
        )
        or '<p class="meta">No se registraron pasajes leídos.</p>'
    )
    trace = _escape(json.dumps(result.get("trace", []), ensure_ascii=False, indent=2))
    provenance = _escape(
        json.dumps(run.get("provenance", {}), ensure_ascii=False, indent=2)
    )
    saved = (
        '<p class="notice" role="status">Evaluación guardada. Gracias.</p>'
        if feedback_recorded
        else ""
    )
    feedback = (
        _feedback_form(run["run_id"], csrf_token, next_url=feedback_next)
        if csrf_token
        else ""
    )
    return f"""<section id="run-status" data-state="succeeded" aria-live="polite">
<section class="panel status"><p class="eyebrow">Respuesta terminada</p><h2>Respuesta</h2>{warning_html}
<div class="answer">{_escape(answer.get("text", ""))}</div><p><strong>Citas:</strong> {citation_links}</p>
<p class="meta">Premisa: {_escape(answer.get("premise_status", "unknown"))} · {_escape(result.get("elapsed_ms"))} ms</p></section>
<section class="panel"><h2>Proceso de investigación</h2>
<p class="lede">El registro de decisiones y evidencia permanece disponible después de generar la respuesta.</p>
{_completed_process(run.get("progress", []))}</section>
<section class="panel"><h2>Evidencia verificable</h2><h3>Documentos consultados</h3>{documents}
<h3 style="margin-top:1.5rem">Pasajes leídos</h3>{evidence}</section>
<section class="panel"><h2>Transparencia de la ejecución</h2>
<details><summary>Búsquedas, lecturas y verificaciones</summary><pre>{trace}</pre></details>
<details><summary>Versión de código, índice, modelo y configuración</summary><pre>{provenance}</pre></details></section>
{saved}{feedback}</section>"""


def _progress_timeline(progress: list[dict[str, Any]]) -> str:
    items = "".join(_progress_event_html(event) for event in progress)
    if not items:
        items = '<li data-empty><span class="meta">Esperando la primera actividad…</span></li>'
    return f'<ol class="activity" data-progress-list>{items}</ol>'


def _completed_process(
    progress: list[dict[str, Any]], *, open_by_default: bool = False
) -> str:
    if not progress:
        return '<p class="meta">Esta ejecución no registró un proceso público.</p>'
    count = len(progress)
    label = "paso" if count == 1 else "pasos"
    open_attribute = " open" if open_by_default else ""
    return f"""<details class="process-archive"{open_attribute}>
<summary>Ver {count} {label} del proceso</summary>
<p class="process-note">Resumen público de decisiones observables; no contiene tokens privados de razonamiento.</p>
{_progress_timeline(progress)}</details>"""


def _progress_event_html(event: dict[str, Any]) -> str:
    payload = event.get("payload", {})
    why = (
        f'<p class="decision-why">{_escape(payload["why"])}</p>'
        if payload.get("why")
        else ""
    )
    documents = "".join(
        _progress_document_html(document) for document in payload.get("documents", [])
    )
    document_html = (
        f'<div class="document-chips">{documents}</div>' if documents else ""
    )
    chunks = "".join(_progress_chunk_html(chunk) for chunk in payload.get("chunks", []))
    chunk_html = f'<div class="chunk-links">{chunks}</div>' if chunks else ""
    citations = " ".join(
        f'<span class="tag">cita: chunk {_escape(chunk_id)}</span>'
        for chunk_id in payload.get("citation_ids", [])
    )
    citation_html = f'<p class="citation-tags">{citations}</p>' if citations else ""
    return f"""<li data-sequence="{_escape(event["sequence"])}">
<span class="activity-kind">{_escape(PROGRESS_LABELS.get(event["event_type"], "Actividad"))}</span>
<strong>{_escape(payload.get("message", "Actividad del agente"))}</strong>{why}
<span class="meta">{_escape(event["created_at"])}</span>{document_html}{chunk_html}{citation_html}</li>"""


def _progress_document_html(document: dict[str, Any]) -> str:
    label = document.get("title") or document.get("path")
    if not label:
        label = f"Documento {document.get('document_id')}"
    date = (
        f" · {_escape(document['publication_date'])}"
        if document.get("publication_date")
        else ""
    )
    return f'<span class="document-chip">{_escape(label)}{date}</span>'


def _progress_chunk_html(chunk: dict[str, Any]) -> str:
    heading = " › ".join(str(item) for item in chunk.get("heading_path") or [])
    heading_suffix = f" · {_escape(heading)}" if heading else ""
    excerpt = chunk.get("excerpt") or chunk.get("snippet") or ""
    truncated = chunk.get("excerpt_truncated") or chunk.get("snippet_truncated")
    excerpt_html = ""
    if excerpt:
        truncation = '<span class="meta chunk-truncated">…</span>' if truncated else ""
        excerpt_html = (
            f'<div class="markdown-body">{render_markdown_html(excerpt)}</div>'
            f"{truncation}"
        )
    return f"""<details class="chunk-link" data-chunk-id="{_escape(chunk.get("chunk_id"))}">
<summary>Chunk {_escape(chunk.get("chunk_id"))} · documento {_escape(chunk.get("document_id"))}{heading_suffix}</summary>
<div class="chunk-content"><span class="meta">{_escape(chunk.get("path") or "Ruta no disponible")}</span>{excerpt_html}</div></details>"""


def _attach_chunk_html(event: dict[str, Any]) -> None:
    """Add server-rendered, sanitized Markdown HTML to streamed chunk payloads.

    The live-stream client injects ``excerpt_html`` via ``innerHTML``, so the
    HTML must already be sanitized here (see ``markdown_render``).
    """
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return
    for chunk in payload.get("chunks") or []:
        if not isinstance(chunk, dict) or "excerpt_html" in chunk:
            continue
        excerpt = chunk.get("excerpt") or chunk.get("snippet") or ""
        if excerpt:
            chunk["excerpt_html"] = render_markdown_html(excerpt)


def _feedback_form(run_id: str, csrf_token: str, *, next_url: str) -> str:
    checks = "".join(
        f'<label class="check"><input type="checkbox" name="problem_types" value="{key}"> {_escape(label)}</label>'
        for key, label in PROBLEM_LABELS.items()
    )
    return f"""<section class="panel"><h2>Evalúa la respuesta</h2>
<p class="lede">Tu evaluación se guarda como un registro nuevo y queda asociada a tu cuenta.
No cambia esta respuesta ni el conjunto v4.</p>
<form method="post" action="/runs/{_escape(run_id)}/feedback">
<input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
<input type="hidden" name="next" value="{_escape(next_url)}">
<div class="field"><label for="rating">Evaluación general</label><select id="rating" name="rating" required>
<option value="helpful">Útil</option><option value="partially_helpful">Parcialmente útil</option>
<option value="not_helpful">No útil</option></select></div>
<fieldset class="field"><legend>¿Qué problema encontraste?</legend><div class="checks">{checks}</div></fieldset>
<div class="field"><label for="comment">Explicación breve <span class="meta">(opcional)</span></label>
<textarea id="comment" name="comment" maxlength="2000" style="min-height:6rem"></textarea></div>
<button type="submit">Guardar evaluación</button></form></section>"""


def _delete_run_form(run_id: str, csrf_token: str, *, next_url: str) -> str:
    return f"""<form method="post" action="/admin/runs/{_escape(run_id)}/delete" style="margin-top:.4rem"
onsubmit="return confirm('¿Eliminar esta pregunta y todos sus datos? Esta acción no se puede deshacer.');">
<input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
<input type="hidden" name="next" value="{_escape(next_url)}">
<button class="danger" type="submit">Eliminar pregunta y datos</button></form>"""


def _admin_panel(run: dict[str, Any], csrf_token: str) -> str:
    run_id = _escape(run["run_id"])
    published_at = run.get("published_at")
    next_url = _escape(f"/runs/{run['run_id']}")
    delete_form = _delete_run_form(run["run_id"], csrf_token, next_url="/admin")
    if published_at:
        return f"""<section class="panel"><h2>Moderación</h2>
<p class="meta">Publicada {_escape(published_at)} por {_escape(run.get("published_by") or "admin")}.</p>
<form method="post" action="/admin/runs/{run_id}/unpublish">
<input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
<input type="hidden" name="next" value="{next_url}">
<button class="secondary" type="submit">Retirar de la vista pública</button></form>
{delete_form}</section>"""
    if run["status"] not in ("succeeded", "failed"):
        return ""
    if run["status"] == "failed":
        return f"""<section class="panel"><h2>Moderación</h2>
<p class="meta">La ejecución falló; solo puede eliminarse.</p>
{delete_form}</section>"""
    return f"""<section class="panel"><h2>Moderación</h2>
<p class="lede">Publicar hace visible la pregunta y la respuesta para cualquier visitante.
Revisa que la pregunta no contenga datos personales antes de publicar.</p>
<form method="post" action="/admin/runs/{run_id}/publish">
<input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
<input type="hidden" name="next" value="{next_url}">
<button type="submit">Publicar respuesta</button></form>
{delete_form}</section>"""


def _admin_dashboard_page(
    runs: list[dict[str, Any]],
    *,
    user: User,
    csrf_token: str,
    page_scripts: Callable[[User | None], str] | None = None,
) -> str:
    rows = []
    for run in runs:
        state = _escape(STATUS_LABELS.get(run["status"], run["status"]))
        if run["published_at"]:
            state += f" · Publicada {_escape(run['published_at'])}"
        delete = ""
        if run["status"] not in ("queued", "running"):
            delete = _delete_run_form(run["run_id"], csrf_token, next_url="/admin")
        rows.append(
            f"""<li><a href="/runs/{_escape(run["run_id"])}">{_escape(run["question"])}</a>
<span class="meta">{state} · {_escape(run["created_at"])} · {_escape(run["user_id"])}</span>{delete}</li>"""
        )
    items = (
        "".join(rows)
        or '<li><span class="meta">No hay preguntas registradas.</span></li>'
    )
    body = f"""<p><a href="/">← Portada</a></p><section><p class="eyebrow">Administración</p>
<h1>Panel de administración</h1><p class="lede">Acciones editoriales del piloto.
Las cuentas y los roles se gestionan en Clerk.</p></section>
<section class="panel"><h2>Moderación</h2>
<p class="meta">Publica o retira respuestas de la vista pública.</p>
<p><a href="/admin/queue">Cola de publicación →</a></p></section>
<section class="panel"><h2>Preguntas</h2>
<p class="meta">Eliminar borra la pregunta, la respuesta, la evidencia registrada y las
evaluaciones asociadas. Las ejecuciones en curso no pueden eliminarse. Esta acción no se puede deshacer.</p>
<ul class="run-list">{items}</ul></section>"""
    return _page(
        "Administración",
        body,
        user=user,
        csrf_token=csrf_token,
        page_scripts=page_scripts,
    )


def _moderation_page(
    runs: list[dict[str, Any]],
    *,
    user: User,
    csrf_token: str,
    page_scripts: Callable[[User | None], str] | None = None,
) -> str:
    rows = []
    for run in runs:
        if run["published_at"]:
            action = "unpublish"
            label = "Retirar"
            state = f"Publicada {_escape(run['published_at'])}"
        else:
            action = "publish"
            label = "Publicar"
            state = "Sin publicar"
        rows.append(
            f"""<li><a href="/runs/{_escape(run["run_id"])}">{_escape(run["question"])}</a>
<span class="meta">{state} · {_escape(run["created_at"])}</span>
<form method="post" action="/admin/runs/{_escape(run["run_id"])}/{action}" style="margin-top:.4rem">
<input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
<input type="hidden" name="next" value="/admin/queue">
<button class="secondary" type="submit">{label}</button></form></li>"""
        )
    items = (
        "".join(rows)
        or '<li><span class="meta">No hay respuestas por moderar.</span></li>'
    )
    body = f"""<p><a href="/admin">← Panel de administración</a></p><section><p class="eyebrow">Moderación</p>
<h1>Cola de publicación</h1><p class="lede">Las respuestas publicadas son visibles para cualquier
visitante. Revisa cada pregunta antes de publicarla: se vuelve contenido público.</p></section>
<section class="panel"><ul class="run-list">{items}</ul></section>"""
    return _page(
        "Moderación",
        body,
        user=user,
        csrf_token=csrf_token,
        page_scripts=page_scripts,
    )


def _sanitize_login_next(raw: str | None) -> str:
    """Allow only same-origin absolute paths (open-redirect guard)."""
    if not raw:
        return "/"
    raw = raw.strip()
    if not raw:
        return "/"
    # Browsers normalize network-path references and backslashes differently
    # from urllib.parse, so reject those forms before parsing the URL.
    if raw.startswith("//") or "\\" in raw:
        return "/"
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in raw):
        return "/"
    parsed = urlparse(raw)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
        return "/"
    return raw


def create_app(
    service: EvaluationService,
    settings: WebSettings,
    provenance_factory: Callable[[], dict[str, Any]],
    *,
    auth_backend: AuthBackend,
    page_scripts: Callable[[User | None], str] | None = None,
    login_scripts: Callable[[str], str] | None = None,
) -> Any:
    """Build an Air app around injected service/auth so UI behavior is testable."""

    @asynccontextmanager
    async def lifespan(_: Any):
        service.start()
        try:
            yield
        finally:
            service.close()

    app = air.Air(lifespan=lifespan)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        session_cookie="dof_eval_session",
        max_age=settings.session_max_age,
        same_site="lax",
        https_only=settings.secure_cookie,
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts)
    )

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[..., Any]
    ) -> Response:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
            "form-action 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net "
            "https://challenges.cloudflare.com https://*.clerk.accounts.dev; "
            "connect-src 'self' https://*.clerk.accounts.dev https://clerk.com "
            "https://*.clerk.com; img-src 'self' https://img.clerk.com data:; "
            "frame-src https://challenges.cloudflare.com "
            "https://*.clerk.accounts.dev; worker-src 'self' blob:"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    unset = object()

    async def current_user(request: Request) -> User | None:
        value = getattr(request.state, "eval_user", unset)
        if value is unset:
            value = await auth_backend.get_user(request)
            request.state.eval_user = value
        return value

    def login_redirect(request: Request) -> RedirectResponse:
        path = request.url.path
        if request.url.query:
            path += f"?{request.url.query}"
        return RedirectResponse(f"/login?next={quote(path, safe='')}", status_code=303)

    def render_home(
        request: Request,
        user: User | None,
        *,
        error: str | None = None,
        values: dict[str, Any] | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        published = service.store.published_runs()
        my_runs = None
        needs_review = False
        review_target = None
        quota_reached = False
        if user is not None:
            my_runs = service.store.runs_for_user(user.id)
            if not user.is_admin:
                needs_review = not service.store.has_review_since_last_submission(
                    user.id
                )
                if needs_review:
                    review_target = service.store.next_answer_to_review(user.id)
                elif settings.daily_question_limit >= 1:
                    cutoff = (
                        (datetime.now(timezone.utc) - timedelta(hours=24))
                        .isoformat()
                        .replace("+00:00", "Z")
                    )
                    quota_reached = (
                        service.store.count_submissions_since(user.id, cutoff)
                        >= settings.daily_question_limit
                    )
        return HTMLResponse(
            _home_page(
                request,
                user=user,
                published=published,
                my_runs=my_runs,
                needs_review=needs_review,
                review_target=review_target,
                quota_reached=quota_reached,
                error=error,
                values=values,
                page_scripts=page_scripts,
            ),
            status_code=status_code,
        )

    @app.get("/login", response_class=HTMLResponse)
    async def login(request: Request, next: str = "/") -> Response:
        # Shadows airclerk's bare-fragment /login with the app layout. The
        # auth-provider mount snippet arrives via login_scripts so create_app
        # stays provider-agnostic; without it (tests, offline dev) the page
        # explains the header-based login instead.
        target = _sanitize_login_next(next)
        user = await current_user(request)
        if user is not None:
            return RedirectResponse(target, status_code=303)
        if login_scripts is None:
            body = (
                '<section class="panel login-panel"><h2>Entrar</h2>'
                '<p class="lede">El inicio de sesión interactivo usa Clerk. En '
                "desarrollo sin Clerk, identifícate con el encabezado "
                "<code>X-Eval-User</code>.</p></section>"
            )
            trailing = ""
        else:
            body = (
                '<section class="panel login-panel"><h2>Entrar o crear cuenta</h2>'
                '<p class="lede">Usa tu correo o una cuenta de Google o GitHub. '
                "Al entrar aceptas que tus preguntas y evaluaciones se guarden "
                'para mejorar el piloto.</p><div id="sign-in" class="sign-in">'
                "</div></section>"
            )
            trailing = login_scripts(target)
        return HTMLResponse(
            _page(
                "Entrar",
                body,
                user=None,
                csrf_token=_csrf(request),
                # login_scripts already bundles Clerk JS; adding page_scripts
                # too would load (and execute) clerk.browser.js twice.
                page_scripts=page_scripts if login_scripts is None else None,
                trailing_scripts=trailing,
            )
        )

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> Response:
        user = await current_user(request)
        return render_home(request, user)

    @app.post("/runs", response_class=HTMLResponse)
    async def create_run(request: Request) -> Response:
        user = await current_user(request)
        if user is None:
            return login_redirect(request)
        try:
            form = await _form(request)
        except ContractError as exc:
            return render_home(request, user, error=str(exc), status_code=400)
        values = {
            "question": form.get("question", ""),
            "as_of": form.get("as_of", ""),
            "required_hops": form.get("required_hops", "1"),
            "client_request_id": form.get("client_request_id", ""),
        }
        if not _csrf_valid(request, form.get("csrf_token")):
            return render_home(
                request,
                user,
                error="La sesión del formulario venció.",
                values=values,
                status_code=403,
            )
        try:
            run_request = RunRequest.from_dict(
                {
                    "question": values["question"],
                    "as_of": values["as_of"] or None,
                    "required_hops": int(str(values["required_hops"])),
                    "client_request_id": values["client_request_id"],
                }
            )
            run = service.submit(
                run_request,
                user_id=user.id,
                admin=user.is_admin,
                daily_question_limit=settings.daily_question_limit,
            )
        except (ContractError, ValueError) as exc:
            return render_home(
                request, user, error=str(exc), values=values, status_code=422
            )
        except ReviewRequiredError:
            return render_home(
                request,
                user,
                error=(
                    "Evalúa una respuesta publicada para desbloquear tu "
                    "siguiente pregunta."
                ),
                values=values,
                status_code=422,
            )
        except QuotaExceededError:
            return render_home(
                request,
                user,
                error="Ya enviaste tu pregunta de este periodo de 24 horas.",
                values=values,
                status_code=422,
            )
        except ActiveRunError:
            return render_home(
                request,
                user,
                error="Ya existe una ejecución activa para tu cuenta.",
                values=values,
                status_code=409,
            )
        except IdempotencyConflictError:
            return render_home(
                request,
                user,
                error="El identificador del formulario ya se usó para otra pregunta.",
                values=values,
                status_code=409,
            )
        except QueueFullError:
            return render_home(
                request,
                user,
                error="La cola local está llena; intenta más tarde.",
                values=values,
                status_code=503,
            )
        except PublicExecutionError as exc:
            return render_home(
                request,
                user,
                error=str(exc),
                values=values,
                status_code=503,
            )
        return RedirectResponse(f"/runs/{run['run_id']}", status_code=303)

    @app.get("/answers/{run_id}", response_class=HTMLResponse)
    async def answer_page(request: Request, run_id: str) -> Response:
        user = await current_user(request)
        try:
            run = service.public_run(
                run_id,
                user_id=user.id if user else None,
                admin=bool(user and user.is_admin),
            )
        except KeyError:
            return HTMLResponse("Respuesta no encontrada.", status_code=404)
        if run["status"] != "succeeded" or run.get("published_at") is None:
            # Owners/admins looking at their private run belong on /runs/.
            return RedirectResponse(f"/runs/{run_id}", status_code=303)
        csrf_token = _csrf(request)
        feedback_recorded = request.query_params.get("feedback") == "recorded"
        fragment = _status_fragment(
            run,
            csrf_token=csrf_token if user is not None else "",
            feedback_recorded=feedback_recorded,
            feedback_next=f"/answers/{run_id}",
        )
        body = f"""<p><a href="/">← Respuestas publicadas</a></p><section><p class="eyebrow">Respuesta publicada</p>
<h1>{_escape(run["question"])}</h1><p class="lede">Fecha de corte: {_escape(run.get("as_of") or "sin fecha")} ·
Publicada: {_escape(run.get("published_at"))}</p></section>
{fragment}"""
        return HTMLResponse(
            _page(
                "Respuesta",
                body,
                user=user,
                csrf_token=csrf_token,
                page_scripts=page_scripts,
            )
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_page(request: Request, run_id: str) -> Response:
        user = await current_user(request)
        if user is None:
            return RedirectResponse(f"/answers/{run_id}", status_code=303)
        is_owner = service.store.run_belongs_to(run_id, user.id)
        if not is_owner and not user.is_admin:
            return RedirectResponse(f"/answers/{run_id}", status_code=303)
        try:
            run = service.public_run(run_id, user_id=user.id, admin=True)
        except KeyError:
            return HTMLResponse("Ejecución no encontrada.", status_code=404)
        csrf_token = _csrf(request)
        feedback_recorded = request.query_params.get("feedback") == "recorded"
        fragment = _status_fragment(
            run,
            csrf_token=csrf_token,
            feedback_recorded=feedback_recorded,
            feedback_next=f"/runs/{run_id}",
        )
        admin_html = _admin_panel(run, csrf_token) if user.is_admin else ""
        body = f"""<p><a href="/">← Portada</a></p><section><p class="eyebrow">Ejecución</p>
<h1>{_escape(run["question"])}</h1><p class="lede">Fecha de corte: {_escape(run.get("as_of") or "sin fecha")} · Documentos mínimos: {_escape(run["required_hops"])}</p></section>
{fragment}{admin_html}"""
        return HTMLResponse(
            _page(
                "Ejecución",
                body,
                user=user,
                csrf_token=csrf_token,
                page_scripts=page_scripts,
            )
        )

    @app.get("/runs/{run_id}/status", response_class=HTMLResponse)
    async def run_status(request: Request, run_id: str) -> Response:
        user = await current_user(request)
        if user is None:
            return HTMLResponse("Sesión requerida.", status_code=401)
        if not user.is_admin and not service.store.run_belongs_to(run_id, user.id):
            return HTMLResponse("Ejecución no encontrada.", status_code=404)
        try:
            run = service.public_run(run_id, user_id=user.id, admin=True)
        except KeyError:
            return HTMLResponse("Ejecución no encontrada.", status_code=404)
        return HTMLResponse(
            _status_fragment(
                run,
                csrf_token=_csrf(request),
                feedback_next=f"/runs/{run_id}",
            )
        )

    @app.get("/runs/{run_id}/events")
    async def run_events(request: Request, run_id: str) -> Response:
        user = await current_user(request)
        if user is None:
            return Response("Sesión requerida.", status_code=401)
        if not user.is_admin and not service.store.run_belongs_to(run_id, user.id):
            return Response("Ejecución no encontrada.", status_code=404)
        try:
            after_values = [
                int(value)
                for value in (
                    request.query_params.get("after", "0"),
                    request.headers.get("last-event-id", "0"),
                )
            ]
            if any(value < 0 for value in after_values):
                raise ValueError
            after = max(after_values)
        except ValueError:
            return Response("Secuencia inválida.", status_code=400)

        async def event_stream() -> AsyncIterator[str]:
            cursor = after
            heartbeat_at = time.monotonic()
            while True:
                events = await asyncio.to_thread(
                    service.store.progress_for_run, run_id, after=cursor
                )
                for event in events:
                    cursor = event["sequence"]
                    _attach_chunk_html(event)
                    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    yield f"id: {cursor}\nevent: progress\ndata: {data}\n\n"
                    heartbeat_at = time.monotonic()
                run = await asyncio.to_thread(
                    service.public_run, run_id, user_id=user.id, admin=True
                )
                if run["status"] not in ACTIVE_STATES:
                    data = json.dumps({"status": run["status"]}, separators=(",", ":"))
                    yield f"event: terminal\ndata: {data}\n\n"
                    return
                if await request.is_disconnected():
                    return
                if time.monotonic() - heartbeat_at >= 15:
                    yield ": keep-alive\n\n"
                    heartbeat_at = time.monotonic()
                await asyncio.sleep(0.5)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.post("/runs/{run_id}/feedback", response_class=HTMLResponse)
    async def submit_feedback(request: Request, run_id: str) -> Response:
        user = await current_user(request)
        if user is None:
            return login_redirect(request)
        form = await _form(request)
        if not _csrf_valid(request, form.get("csrf_token")):
            return HTMLResponse("La sesión del formulario venció.", status_code=403)
        next_url = _safe_next(form.get("next"), f"/answers/{run_id}")
        try:
            feedback = FeedbackRequest.from_dict(
                {
                    "rating": form.get("rating"),
                    "problem_types": form.getlist("problem_types"),
                    "comment": form.get("comment", ""),
                }
            )
            service.submit_feedback(
                run_id,
                feedback,
                user_id=user.id,
                admin=user.is_admin,
            )
        except ContractError as exc:
            return HTMLResponse(_escape(str(exc)), status_code=422)
        except KeyError:
            return HTMLResponse("Ejecución no encontrada.", status_code=404)
        separator = "&" if "?" in next_url else "?"
        return RedirectResponse(
            f"{next_url}{separator}feedback=recorded", status_code=303
        )

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_dashboard(request: Request) -> Response:
        user = await current_user(request)
        if user is None:
            return login_redirect(request)
        if not user.is_admin:
            return HTMLResponse("Solo administradores.", status_code=403)
        return HTMLResponse(
            _admin_dashboard_page(
                service.store.admin_runs(),
                user=user,
                csrf_token=_csrf(request),
                page_scripts=page_scripts,
            )
        )

    @app.get("/admin/queue", response_class=HTMLResponse)
    async def moderation_queue(request: Request) -> Response:
        user = await current_user(request)
        if user is None:
            return login_redirect(request)
        if not user.is_admin:
            return HTMLResponse("Solo administradores.", status_code=403)
        return HTMLResponse(
            _moderation_page(
                service.store.runs_for_moderation(),
                user=user,
                csrf_token=_csrf(request),
                page_scripts=page_scripts,
            )
        )

    @app.post("/admin/runs/{run_id}/publish")
    async def publish_run(request: Request, run_id: str) -> Response:
        return await _moderate(request, run_id, publish=True)

    @app.post("/admin/runs/{run_id}/unpublish")
    async def unpublish_run(request: Request, run_id: str) -> Response:
        return await _moderate(request, run_id, publish=False)

    async def _moderate(request: Request, run_id: str, *, publish: bool) -> Response:
        user = await current_user(request)
        if user is None:
            return login_redirect(request)
        if not user.is_admin:
            return HTMLResponse("Solo administradores.", status_code=403)
        form = await _form(request)
        if not _csrf_valid(request, form.get("csrf_token")):
            return HTMLResponse("La sesión del formulario venció.", status_code=403)
        try:
            if publish:
                service.publish(run_id, admin_id=user.id)
            else:
                service.unpublish(run_id)
        except KeyError:
            return HTMLResponse("Ejecución no encontrada.", status_code=404)
        except ValueError as exc:
            return HTMLResponse(_escape(str(exc)), status_code=422)
        next_url = _safe_next(form.get("next"), "/admin/queue")
        return RedirectResponse(next_url, status_code=303)

    @app.post("/admin/runs/{run_id}/delete")
    async def delete_run(request: Request, run_id: str) -> Response:
        user = await current_user(request)
        if user is None:
            return login_redirect(request)
        if not user.is_admin:
            return HTMLResponse("Solo administradores.", status_code=403)
        form = await _form(request)
        if not _csrf_valid(request, form.get("csrf_token")):
            return HTMLResponse("La sesión del formulario venció.", status_code=403)
        try:
            service.delete_run(run_id)
        except KeyError:
            return HTMLResponse("Ejecución no encontrada.", status_code=404)
        except ValueError as exc:
            return HTMLResponse(_escape(str(exc)), status_code=422)
        next_url = _safe_next(form.get("next"), "/admin")
        if next_url in {f"/runs/{run_id}", f"/answers/{run_id}"}:
            next_url = "/admin"
        return RedirectResponse(next_url, status_code=303)

    @app.get("/api/v1/health")
    async def health() -> JSONResponse:
        healthy = service.store.check_health()
        return JSONResponse(
            {"status": "ok" if healthy else "unavailable"},
            status_code=200 if healthy else 503,
        )

    @app.get("/api/v1/capabilities")
    async def capabilities() -> JSONResponse:
        provenance = provenance_factory()
        return JSONResponse(
            {
                "contract_version": "v1",
                "schema_version": SCHEMA_VERSION,
                "retrieval_mode": provenance["configuration"]["retrieval_mode"],
                "vector_available": provenance["vector_available"],
                "model": provenance["model"],
                "limits": {
                    "question_characters": 2000,
                    "required_hops": 5,
                    "questions_per_day": settings.daily_question_limit,
                    "active_runs_per_user": 1,
                },
            }
        )

    return app


def build_default_app(repo_root: Path | None = None) -> tuple[Any, WebSettings]:
    root = (repo_root or Path(__file__).resolve().parent.parent).resolve()
    settings = WebSettings.from_env(root)
    # Imported lazily: airclerk validates Clerk environment variables at
    # import time, and tests/offline development use FakeAuthBackend instead.
    import airclerk

    from .clerk_auth import ClerkAuthBackend, clerk_login_scripts, clerk_page_scripts

    auth_backend = ClerkAuthBackend(secret_key=airclerk.settings.CLERK_SECRET_KEY)
    executor = AgentRunExecutor(AgentExecutorConfig.from_env(root))
    service = EvaluationService(
        EvaluationStore(settings.db_path),
        executor,
        executor.provenance,
        queue_capacity=settings.queue_capacity,
    )
    app = create_app(
        service,
        settings,
        executor.provenance,
        auth_backend=auth_backend,
        page_scripts=clerk_page_scripts,
        login_scripts=clerk_login_scripts,
    )
    app.include_router(airclerk.router)
    return app, settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the DOF human-evaluation site")
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    args = parser.parse_args()
    app, settings = build_default_app(args.repo_root)
    # Questions are stored deliberately, but client IP addresses are not part
    # of the evaluation dataset. Keep Uvicorn's per-request access log off.
    uvicorn.run(app, host=settings.host, port=settings.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
