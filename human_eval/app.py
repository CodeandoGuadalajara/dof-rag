"""Single-process Air application for controlled human evaluation of the DOF agent."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import html
import json
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import air
import uvicorn
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from .agent_executor import AgentExecutorConfig, AgentRunExecutor
from .contracts import ContractError, FeedbackRequest, RunRequest
from .service import (
    ActiveRunError,
    EvaluationService,
    IdempotencyConflictError,
    QueueFullError,
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


@dataclass(frozen=True)
class WebSettings:
    host: str
    port: int
    db_path: Path
    evaluator_tokens: tuple[str, ...]
    session_secret: str
    secure_cookie: bool = False
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver")
    session_max_age: int = 12 * 60 * 60
    rate_limit_per_hour: int = 10
    queue_capacity: int = 20

    @classmethod
    def from_env(cls, repo_root: Path) -> "WebSettings":
        tokens = tuple(
            token.strip()
            for token in os.environ.get("DOF_EVALUATOR_TOKENS", "").split(",")
            if token.strip()
        )
        if not tokens:
            raise ValueError(
                "set DOF_EVALUATOR_TOKENS to one or more invitation tokens"
            )
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
            evaluator_tokens=tokens,
            session_secret=session_secret,
            secure_cookie=os.environ.get("DOF_SECURE_COOKIE", "false").lower()
            in {"1", "true", "yes"},
            allowed_hosts=allowed_hosts,
            session_max_age=int(
                os.environ.get("DOF_SESSION_MAX_AGE", str(12 * 60 * 60))
            ),
            rate_limit_per_hour=int(os.environ.get("DOF_RATE_LIMIT_PER_HOUR", "10")),
            queue_capacity=int(os.environ.get("DOF_QUEUE_CAPACITY", "20")),
        )


class TokenAuthenticator:
    """Exchange an invitation token for its stable hash; never persist the token."""

    def __init__(self, tokens: tuple[str, ...]):
        self.hashes = tuple(self.hash_token(token) for token in tokens)

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def authenticate(self, token: str) -> str | None:
        candidate = self.hash_token(token.strip())
        return next(
            (known for known in self.hashes if hmac.compare_digest(candidate, known)),
            None,
        )


class HourlyRateLimiter:
    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("rate limit must be positive")
        self.limit = limit
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def consume(self, evaluator_hash: str) -> bool:
        now = time.monotonic()
        cutoff = now - 3600
        with self.lock:
            events = self.events[evaluator_hash]
            while events and events[0] < cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True


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


def _evaluator(request: Request) -> str | None:
    value = request.session.get("evaluator_hash")
    return value if isinstance(value, str) and len(value) == 64 else None


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
.eyebrow { color:var(--accent); font-size:.76rem; font-weight:800; letter-spacing:.12em;
  margin:0 0 .35rem; text-transform:uppercase; }
h1,h2,h3 { font-family:Georgia,"Times New Roman",serif; line-height:1.14; margin-top:0; }
h1 { font-size:clamp(2rem,5vw,3.8rem); font-weight:500; letter-spacing:-.035em; margin-bottom:.6rem; }
h2 { font-size:1.65rem; font-weight:500; }
h3 { font-size:1.18rem; margin-bottom:.45rem; }
.lede { color:var(--muted); max-width:68ch; margin:0; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:4px;
  box-shadow:0 12px 32px rgba(30,35,28,.06); padding:clamp(1.1rem,3vw,2rem); margin:1.25rem 0; }
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
footer { border-top:1px solid var(--line); color:var(--muted); font-size:.82rem; margin-top:3rem;
  padding-top:1.25rem; }
@media (max-width:680px) { .grid,.checks { grid-template-columns:1fr; } header { display:block; }
  header form { margin-top:1rem; } }
"""


POLL_SCRIPT = """
(() => {
  const poll = async (node) => {
    const url = node.dataset.pollUrl;
    if (!url) return;
    await new Promise(resolve => setTimeout(resolve, 2000));
    try {
      const response = await fetch(url, {credentials: 'same-origin', cache: 'no-store'});
      if (response.status === 401) { location.href = '/login'; return; }
      if (!response.ok) throw new Error('poll failed');
      const holder = document.createElement('div');
      holder.innerHTML = await response.text();
      const replacement = holder.firstElementChild;
      node.replaceWith(replacement);
      if (replacement.dataset.pollUrl) poll(replacement);
    } catch (_) { setTimeout(() => poll(node), 3000); }
  };
  document.querySelectorAll('[data-poll-url]').forEach(poll);
})();
"""


def _page(
    title: str,
    body: str,
    *,
    authenticated: bool = False,
    csrf_token: str = "",
) -> str:
    session_action = (
        f'<form method="post" action="/logout"><input type="hidden" name="csrf_token" '
        f'value="{_escape(csrf_token)}"><button class="secondary" type="submit">Salir</button></form>'
        if authenticated
        else ""
    )
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(title)} · Evaluación DOF</title><style>{STYLE}</style></head>
<body><main class="shell"><header><div><p class="eyebrow">Piloto de investigación</p>
<a href="/" style="text-decoration:none;color:inherit"><strong>Agente del Diario Oficial</strong></a></div>
{session_action}</header>{body}
<footer>Las preguntas, respuestas, evidencias y evaluaciones se guardan para análisis. El feedback no modifica automáticamente v4.</footer>
</main><script>{POLL_SCRIPT}</script></body></html>"""


def _login_page(csrf_token: str, error: str | None = None) -> str:
    error_html = (
        f'<p class="warning" role="alert">{_escape(error)}</p>' if error else ""
    )
    body = f"""<section><p class="eyebrow">Acceso controlado</p><h1>Prueba el agente del DOF</h1>
<p class="lede">Introduce el token de invitación. Se convertirá en una sesión firmada y no se guardará en la base de datos.</p></section>
<section class="panel" style="max-width:34rem">{error_html}<form method="post" action="/login">
<input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
<div class="field"><label for="token">Token de invitación</label>
<input id="token" name="token" type="password" required autocomplete="current-password"></div>
<button type="submit">Entrar al piloto</button></form></section>"""
    return _page("Acceso", body)


def _home_page(
    request: Request,
    runs: list[dict[str, Any]],
    *,
    error: str | None = None,
    values: dict[str, Any] | None = None,
) -> str:
    values = values or {}
    csrf_token = _csrf(request)
    error_html = (
        f'<p class="warning" role="alert">{_escape(error)}</p>' if error else ""
    )
    run_items = "".join(
        f'<li><a href="/runs/{_escape(run["run_id"])}">{_escape(run["question"])}</a>'
        f'<span class="meta">{_escape(STATUS_LABELS.get(run["status"], run["status"]))} · '
        f"{_escape(run['created_at'])}</span></li>"
        for run in runs
    )
    history = (
        f'<section class="panel"><h2>Ejecuciones recientes</h2><ul class="run-list">{run_items}</ul></section>'
        if run_items
        else ""
    )
    hops = str(values.get("required_hops", "1"))
    options = "".join(
        f'<option value="{number}"{" selected" if hops == str(number) else ""}>{number}</option>'
        for number in range(1, 6)
    )
    body = f"""<section><p class="eyebrow">Evaluación humana</p><h1>Pregunta, inspecciona, evalúa.</h1>
<p class="lede">El agente usa hoy recuperación léxica completa. Una respuesta puede tardar decenas de segundos; la página consultará su estado sin mantener una petición abierta.</p></section>
<section class="panel"><h2>Nueva pregunta</h2>{error_html}<form method="post" action="/runs">
<input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
<input type="hidden" name="client_request_id" value="{_escape(values.get("client_request_id") or uuid.uuid4())}">
<div class="grid"><div class="field full"><label for="question">Pregunta</label>
<textarea id="question" name="question" minlength="3" maxlength="2000" required placeholder="¿Qué establece el decreto y qué publicaciones deben compararse?">{_escape(values.get("question", ""))}</textarea></div>
<div class="field"><label for="as_of">Fecha de corte <span class="meta">(opcional)</span></label>
<input id="as_of" name="as_of" type="date" value="{_escape(values.get("as_of", ""))}"></div>
<div class="field"><label for="required_hops">Documentos mínimos</label>
<select id="required_hops" name="required_hops">{options}</select>
<span class="meta">Usa 2 o más para comparaciones que requieran fuentes distintas.</span></div></div>
<button type="submit">Iniciar consulta</button></form></section>{history}"""
    return _page("Nueva pregunta", body, authenticated=True, csrf_token=csrf_token)


def _status_fragment(
    run: dict[str, Any],
    *,
    csrf_token: str = "",
    feedback_recorded: bool = False,
) -> str:
    state = run["status"]
    poll = (
        f' data-poll-url="/runs/{_escape(run["run_id"])}/status"'
        if state in ACTIVE_STATES
        else ""
    )
    meta = f'<p class="meta">Creada: {_escape(run["created_at"])}</p>'
    if state in ACTIVE_STATES:
        return f"""<section id="run-status" class="panel status" data-state="{state}"{poll} aria-live="polite">
<p class="eyebrow">{_escape(STATUS_LABELS[state])}</p><h2>La ejecución sigue en progreso</h2>
<p>Puede tardar decenas de segundos. Esta sección se actualizará automáticamente.</p>{meta}</section>"""
    if state == "failed":
        error = run.get("error", {})
        return f"""<section id="run-status" class="panel status" data-state="failed" aria-live="polite">
<p class="eyebrow">Ejecución fallida</p><h2>{_escape(error.get("message", "No se pudo completar la consulta."))}</h2>
<p class="meta">Código: {_escape(error.get("code", "internal_error"))}</p>{meta}</section>"""

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
<p>{_escape(item.get("text"))}</p></details>"""
            for item in result.get("evidence", [])
        )
        or '<p class="meta">No se registraron pasajes leídos.</p>'
    )
    trace = _escape(json.dumps(result.get("trace", []), ensure_ascii=False, indent=2))
    provenance = _escape(
        json.dumps(run.get("provenance", {}), ensure_ascii=False, indent=2)
    )
    saved = (
        '<p class="notice" role="status">Feedback guardado. Gracias.</p>'
        if feedback_recorded
        else ""
    )
    feedback = _feedback_form(run["run_id"], csrf_token) if csrf_token else ""
    return f"""<section id="run-status" data-state="succeeded" aria-live="polite">
<section class="panel status"><p class="eyebrow">Respuesta terminada</p><h2>Respuesta</h2>{warning_html}
<div class="answer">{_escape(answer.get("text", ""))}</div><p><strong>Citas:</strong> {citation_links}</p>
<p class="meta">Premisa: {_escape(answer.get("premise_status", "unknown"))} · {_escape(result.get("elapsed_ms"))} ms</p></section>
<section class="panel"><h2>Evidencia verificable</h2><h3>Documentos consultados</h3>{documents}
<h3 style="margin-top:1.5rem">Pasajes leídos</h3>{evidence}</section>
<section class="panel"><h2>Transparencia de la ejecución</h2>
<details><summary>Búsquedas, lecturas y verificaciones</summary><pre>{trace}</pre></details>
<details><summary>Versión de código, índice, modelo y configuración</summary><pre>{provenance}</pre></details></section>
{saved}{feedback}</section>"""


def _feedback_form(run_id: str, csrf_token: str) -> str:
    checks = "".join(
        f'<label class="check"><input type="checkbox" name="problem_types" value="{key}"> {_escape(label)}</label>'
        for key, label in PROBLEM_LABELS.items()
    )
    return f"""<section class="panel"><h2>Evalúa la respuesta</h2>
<p class="lede">Tu evaluación se guarda como un registro nuevo. No cambia esta respuesta ni el conjunto v4.</p>
<form method="post" action="/runs/{_escape(run_id)}/feedback">
<input type="hidden" name="csrf_token" value="{_escape(csrf_token)}">
<div class="field"><label for="rating">Evaluación general</label><select id="rating" name="rating" required>
<option value="helpful">Útil</option><option value="partially_helpful">Parcialmente útil</option>
<option value="not_helpful">No útil</option></select></div>
<fieldset class="field"><legend>¿Qué problema encontraste?</legend><div class="checks">{checks}</div></fieldset>
<div class="field"><label for="comment">Explicación breve <span class="meta">(opcional)</span></label>
<textarea id="comment" name="comment" maxlength="2000" style="min-height:6rem"></textarea></div>
<button type="submit">Guardar evaluación</button></form></section>"""


def create_app(
    service: EvaluationService,
    settings: WebSettings,
    provenance_factory: Callable[[], dict[str, Any]],
) -> Any:
    """Build an Air app around an injected service so UI behavior is testable."""

    @asynccontextmanager
    async def lifespan(_: Any):
        service.start()
        try:
            yield
        finally:
            service.close()

    app = air.Air(lifespan=lifespan)
    authenticator = TokenAuthenticator(settings.evaluator_tokens)
    limiter = HourlyRateLimiter(settings.rate_limit_per_hour)
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
            "script-src 'self' 'unsafe-inline'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def login_redirect() -> RedirectResponse:
        return RedirectResponse("/login", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request) -> Response:
        if _evaluator(request):
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(_login_page(_csrf(request)))

    @app.post("/login", response_class=HTMLResponse)
    async def login(request: Request) -> Response:
        try:
            form = await _form(request)
        except ContractError as exc:
            return HTMLResponse(_login_page(_csrf(request), str(exc)), status_code=400)
        if not _csrf_valid(request, form.get("csrf_token")):
            return HTMLResponse(
                _login_page(_csrf(request), "La sesión del formulario venció."),
                status_code=403,
            )
        evaluator_hash = authenticator.authenticate(str(form.get("token", "")))
        if evaluator_hash is None:
            return HTMLResponse(
                _login_page(_csrf(request), "Token de invitación inválido."),
                status_code=401,
            )
        request.session.clear()
        request.session["evaluator_hash"] = evaluator_hash
        request.session["csrf_token"] = secrets.token_urlsafe(32)
        return RedirectResponse("/", status_code=303)

    @app.post("/logout")
    async def logout(request: Request) -> Response:
        form = await _form(request)
        if not _csrf_valid(request, form.get("csrf_token")):
            return HTMLResponse("Solicitud inválida.", status_code=403)
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> Response:
        evaluator_hash = _evaluator(request)
        if evaluator_hash is None:
            return login_redirect()
        runs = service.store.runs_for_evaluator(evaluator_hash)
        return HTMLResponse(_home_page(request, runs))

    @app.post("/runs", response_class=HTMLResponse)
    async def create_run(request: Request) -> Response:
        evaluator_hash = _evaluator(request)
        if evaluator_hash is None:
            return login_redirect()
        try:
            form = await _form(request)
        except ContractError as exc:
            return HTMLResponse(
                _home_page(request, [], error=str(exc)), status_code=400
            )
        values = {
            "question": form.get("question", ""),
            "as_of": form.get("as_of", ""),
            "required_hops": form.get("required_hops", "1"),
            "client_request_id": form.get("client_request_id", ""),
        }
        if not _csrf_valid(request, form.get("csrf_token")):
            return HTMLResponse(
                _home_page(
                    request, [], error="La sesión del formulario venció.", values=values
                ),
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
            existing = service.idempotent_run(
                run_request, evaluator_hash=evaluator_hash
            )
            if existing is not None:
                run = existing
            else:
                if not limiter.consume(evaluator_hash):
                    raise ContractError("Se alcanzó el límite de ejecuciones por hora.")
                run = service.submit(run_request, evaluator_hash=evaluator_hash)
        except (ContractError, ValueError) as exc:
            runs = service.store.runs_for_evaluator(evaluator_hash)
            return HTMLResponse(
                _home_page(request, runs, error=str(exc), values=values),
                status_code=422,
            )
        except ActiveRunError:
            message = "Ya existe una ejecución activa para este evaluador."
            runs = service.store.runs_for_evaluator(evaluator_hash)
            return HTMLResponse(
                _home_page(request, runs, error=message, values=values), status_code=409
            )
        except IdempotencyConflictError:
            message = "El identificador del formulario ya se usó para otra pregunta."
            runs = service.store.runs_for_evaluator(evaluator_hash)
            return HTMLResponse(
                _home_page(request, runs, error=message, values=values), status_code=409
            )
        except QueueFullError:
            message = "La cola local está llena; intenta más tarde."
            runs = service.store.runs_for_evaluator(evaluator_hash)
            return HTMLResponse(
                _home_page(request, runs, error=message, values=values), status_code=503
            )
        return RedirectResponse(f"/runs/{run['run_id']}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_page(request: Request, run_id: str) -> Response:
        evaluator_hash = _evaluator(request)
        if evaluator_hash is None:
            return login_redirect()
        try:
            run = service.public_run(run_id, evaluator_hash=evaluator_hash)
        except KeyError:
            return HTMLResponse("Ejecución no encontrada.", status_code=404)
        csrf_token = _csrf(request)
        feedback_recorded = request.query_params.get("feedback") == "recorded"
        fragment = _status_fragment(
            run,
            csrf_token=csrf_token,
            feedback_recorded=feedback_recorded,
        )
        body = f"""<p><a href="/">← Nueva pregunta</a></p><section><p class="eyebrow">Ejecución</p>
<h1>{_escape(run["question"])}</h1><p class="lede">Fecha de corte: {_escape(run.get("as_of") or "sin fecha")} · Documentos mínimos: {_escape(run["required_hops"])}</p></section>
{fragment}"""
        return HTMLResponse(
            _page("Ejecución", body, authenticated=True, csrf_token=csrf_token)
        )

    @app.get("/runs/{run_id}/status", response_class=HTMLResponse)
    async def run_status(request: Request, run_id: str) -> Response:
        evaluator_hash = _evaluator(request)
        if evaluator_hash is None:
            return HTMLResponse("Sesión requerida.", status_code=401)
        try:
            run = service.public_run(run_id, evaluator_hash=evaluator_hash)
        except KeyError:
            return HTMLResponse("Ejecución no encontrada.", status_code=404)
        return HTMLResponse(_status_fragment(run, csrf_token=_csrf(request)))

    @app.post("/runs/{run_id}/feedback", response_class=HTMLResponse)
    async def submit_feedback(request: Request, run_id: str) -> Response:
        evaluator_hash = _evaluator(request)
        if evaluator_hash is None:
            return login_redirect()
        form = await _form(request)
        if not _csrf_valid(request, form.get("csrf_token")):
            return HTMLResponse("La sesión del formulario venció.", status_code=403)
        try:
            feedback = FeedbackRequest.from_dict(
                {
                    "rating": form.get("rating"),
                    "problem_types": form.getlist("problem_types"),
                    "comment": form.get("comment", ""),
                }
            )
            service.submit_feedback(run_id, feedback, evaluator_hash=evaluator_hash)
        except ContractError as exc:
            return HTMLResponse(_escape(str(exc)), status_code=422)
        except KeyError:
            return HTMLResponse("Ejecución no encontrada.", status_code=404)
        return RedirectResponse(f"/runs/{run_id}?feedback=recorded", status_code=303)

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
                    "runs_per_hour": settings.rate_limit_per_hour,
                    "active_runs_per_evaluator": 1,
                },
            }
        )

    return app


def build_default_app(repo_root: Path | None = None) -> tuple[Any, WebSettings]:
    root = (repo_root or Path(__file__).resolve().parent.parent).resolve()
    settings = WebSettings.from_env(root)
    executor = AgentRunExecutor(AgentExecutorConfig.from_env(root))
    service = EvaluationService(
        EvaluationStore(settings.db_path),
        executor,
        executor.provenance,
        queue_capacity=settings.queue_capacity,
    )
    return create_app(service, settings, executor.provenance), settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the DOF human-evaluation site")
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    args = parser.parse_args()
    app, settings = build_default_app(args.repo_root)
    uvicorn.run(app, host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
