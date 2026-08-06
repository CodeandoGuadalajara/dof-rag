"""Build eval set v3: real titles + anchor-carrying thematic/paraphrase queries.

v2 defects being fixed (see docs/full-corpus-build.md):
1. 271/499 verbatim_title queries are filename slugs -> titles re-extracted
   programmatically from markdown heading/bold header blocks (free, no LLM).
2. thematic/paraphrase queries are non-identifying: their gold doc is one
   of hundreds-to-thousands of equivalent answers. v3 regenerates them
   instructing the LLM to include identifying anchors (dates, amounts,
   entity names, agreement numbers), and VALIDATES that each query carries
   at least one rare token (df < 0.1% of corpus, via documents_fts_vocab)
   that appears in the document.

factual/article_specific queries are kept from v2 (they are chunk-anchored
and were the best-performing type at full corpus scale).

Token budget: one call per doc with title + first ~6k chars (~2k tokens
in, ~150 out) => ~1M input / 75k output tokens for the 499 docs.

Usage:
    export KIMI_API_KEY="***"   # or LLM_API_KEY / ANTHROPIC_API_KEY in .env
    uv run python scripts/build_eval_v3.py [--limit 5] [--workers 4]

Resumable: docs already present in the v3 output are skipped.
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from extract_title import SLUG_RE, extract_title  # noqa: E402

V2_PATH = Path("eval/dof_queries_v2.jsonl")
V3_PATH = Path("eval/dof_queries_v3.jsonl")
CORPUS = Path("../dof_md")
CORPUS_DB = "dof_db/dof_corpus_l3.sqlite"

DEFAULT_BASE_URL = "https://api.kimi.com/coding"
DEFAULT_MODEL = "kimi-for-coding"
DOC_CHARS = 6_000

PROMPT = """\
Estás mejorando un set de evaluación de búsqueda para el Diario Oficial de \
la Federación (DOF) de México. El problema a evitar: el DOF tiene miles de \
documentos casi idénticos entre sí (el tipo de cambio se publica a diario, \
hay cientos de convocatorias del mismo trámite, convenios con el mismo \
formato). Una consulta genérica NO sirve para encontrar ESTE documento \
entre 657,867.

TÍTULO: {title}

DOCUMENTO (extracto):
{excerpt}

Genera consultas que una persona real haría para encontrar ESTE documento \
específico. Cada consulta DEBE incluir al menos un detalle identificador \
del documento: fechas, montos, nombres de dependencias/empresas/personas, \
números de acuerdo/convocatoria/licitación, municipios o estados, algún \
dato que lo distinga de documentos similares.

Responde SOLO con un array JSON (sin markdown, sin explicaciones):
1. {{"type": "thematic", "question": "..."}}: pregunta ciudadana sobre el \
tema, SIN jerga legal, pero CON al menos un detalle identificador. Ej: \
"¿cuánto se transfirió a Aguascalientes para obra pública en el convenio \
de reasignación de 2006?"
2. {{"type": "paraphrase", "question": "..."}}: el tema del documento \
reformulado con otras palabras (no copies 5+ palabras consecutivas del \
título), conservando los detalles identificadores clave.

Reglas:
- Español natural, como lo escribiría una persona en un buscador.
- Ambas terminan en "?". Máximo 40 palabras cada una.
"""


def fold(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t.lower())
                   if not unicodedata.combining(c))


class AnchorChecker:
    """df lookups against documents_fts_vocab to verify identifying anchors.

    Uses one connection per thread (SQLite connections are thread-bound).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.local = __import__("threading").local()
        conn = self._conn()
        self.n_docs = conn.execute(
            "SELECT COUNT(*) FROM documents").fetchone()[0]
        self.max_df = self.n_docs * 0.001  # token must appear in <0.1% of docs
        self.cache: dict[str, int] = {}

    def _conn(self):
        conn = getattr(self.local, "conn", None)
        if conn is None:
            from corpus_store.db import connect
            conn = connect(self.db_path)
            self.local.conn = conn
        return conn

    def df(self, token: str) -> int:
        f = fold(token)
        if f not in self.cache:
            r = self._conn().execute(
                "SELECT doc FROM documents_fts_vocab WHERE term = ?",
                (f,)).fetchone()
            self.cache[f] = r[0] if r else 0
        return self.cache[f]

    def has_anchor(self, question: str, doc_text: str) -> bool:
        doc_folded = fold(doc_text)
        for tok in re.findall(r"\w+", question):
            if len(tok) < 4:
                continue
            if re.match(r"^\d{4,}$", tok):  # year/number anchor
                if tok in doc_text:
                    return True
            elif self.df(tok) < self.max_df and fold(tok) in doc_folded:
                return True
        return False


def call_llm(prompt: str, base_url: str, model: str, api_key: str,
             api: str = "anthropic", max_tokens: int = 800,
             retries: int = 3) -> str:
    """LLM call: anthropic-messages (Kimi) or openai-completions (opencode)."""
    body = {
        "model": model, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if api != "openai":
        # kimi-for-coding is a thinking model; disable thinking — the task
        # is simple and thinking tokens are billed as output tokens.
        body["thinking"] = {"type": "disabled"}
    body = json.dumps(body).encode()
    if api == "openai":
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # opencode.ai sits behind Cloudflare; a browser-like UA helps.
            "User-Agent": "Mozilla/5.0",
        }
    else:
        url = f"{base_url.rstrip('/')}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "User-Agent": "KimiCLI/1.5",
        }
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.load(resp)
            if api == "openai":
                msg = payload["choices"][0]["message"]
                text = msg.get("content") or msg.get("reasoning_content") or ""
                if text:
                    return text
                raise ValueError(f"empty content: {str(payload)[:200]}")
            for block in payload.get("content", []):
                if block.get("type") == "text":
                    return block["text"]
            raise ValueError(f"empty content: {str(payload)[:200]}")
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, ValueError, KeyError) as exc:
            code = getattr(exc, "code", None)
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1) * (5 if code in (402, 429) else 1))
    return ""


def parse_and_validate(raw: str, doc_text: str,
                       anchor: AnchorChecker) -> tuple[list[dict], list[str]]:
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return [], ["unparseable JSON"]
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [], ["bad JSON"]
    valid, rejected, seen = [], [], set()
    for item in data if isinstance(data, list) else []:
        if isinstance(item, str):  # model stringified the objects
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                rejected.append(f"not a dict: {item[:40]}")
                continue
        if not isinstance(item, dict):
            rejected.append(f"not a dict: {str(item)[:40]}")
            continue
        qtype = item.get("type")
        q = str(item.get("question", "")).strip()
        if qtype not in ("thematic", "paraphrase"):
            rejected.append(f"bad type: {qtype}")
            continue
        if not (15 <= len(q) <= 300):
            rejected.append(f"length: {q[:40]}")
            continue
        if qtype != "paraphrase" and not q.endswith("?"):
            rejected.append(f"missing ?: {q[:40]}")
            continue
        norm = " ".join(fold(q).split())
        if norm in seen:
            rejected.append("duplicate")
            continue
        if not anchor.has_anchor(q, doc_text):
            rejected.append(f"no anchor: {q[:60]}")
            continue
        seen.add(norm)
        valid.append({"query": q, "type": qtype, "expected_chunk_index": None,
                      "gen": "v3"})
    return valid, rejected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=os.environ.get("EVAL_V3_MODEL",
                                                      DEFAULT_MODEL))
    ap.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL",
                                                         DEFAULT_BASE_URL))
    ap.add_argument("--api", default="anthropic",
                    choices=["anthropic", "openai"],
                    help="anthropic=Kimi, openai=opencode zen")
    ap.add_argument("--titles-only", action="store_true",
                    help="only fix titles, skip LLM regeneration (free)")
    args = ap.parse_args()

    api_key = (os.environ.get("KIMI_API_KEY") or os.environ.get("LLM_API_KEY")
               or os.environ.get("ANTHROPIC_API_KEY") or "")
    if args.api == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or api_key
    if not api_key and not args.titles_only:
        print("ERROR: export KIMI_API_KEY (or LLM_API_KEY / ANTHROPIC_API_KEY)",
              file=sys.stderr)
        return 1

    v2 = [json.loads(l) for l in open(V2_PATH)]
    docs = [r for r in v2 if not r.get("error") and r.get("queries")]
    done = set()
    if V3_PATH.exists():
        for line in open(V3_PATH):
            done.add(json.loads(line)["relpath"])
    todo = [r for r in docs if r["relpath"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(docs)} v2 docs, {len(done)} already in v3, {len(todo)} to go",
          flush=True)

    anchor = AnchorChecker(CORPUS_DB) if not args.titles_only else None

    def process(rec: dict) -> dict:
        out = dict(rec)
        text = (CORPUS / rec["relpath"]).read_text(encoding="utf-8",
                                                   errors="replace")
        # 1. title fix (free)
        if SLUG_RE.match(rec["title"]):
            t = extract_title(text)
            if t:
                out["title"] = t
                out["title_v3"] = True
        if args.titles_only:
            return out
        # 2. regenerate thematic + paraphrase with anchors
        prompt = PROMPT.format(title=out["title"], excerpt=text[:DOC_CHARS])
        new_q, rejected = [], []
        for _attempt in range(2):
            try:
                raw = call_llm(prompt, args.base_url, args.model, api_key,
                               api=args.api)
            except Exception as exc:
                return {**out, "error": f"{type(exc).__name__}: {exc}"}
            new_q, rejected = parse_and_validate(raw, text, anchor)
            if new_q:
                break
        kept = [q for q in rec["queries"]
                if q["type"] not in ("thematic", "paraphrase")]
        out["queries"] = kept + new_q
        out["rejected_v3"] = rejected
        return out

    t0 = time.time()
    n_ok = 0
    with open(V3_PATH, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, r): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            out = fut.result()
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            n_ok += not out.get("error")
            if i % 25 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"  [{i}/{len(todo)}] {rate:.1f} docs/s "
                      f"({(len(todo) - i) / rate / 60:.0f} min left)",
                      flush=True)
    print(f"done: {n_ok}/{len(todo)} ok -> {V3_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
