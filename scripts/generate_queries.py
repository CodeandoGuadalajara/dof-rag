"""Generate the v2 evaluation query set for the DOF retrieval benchmark.

Run from repo root:
    export KIMI_API_KEY="***"
    python scripts/generate_queries.py --corpus ../dof_md-local --docs 500 \
        --out eval/dof_queries_v2.jsonl

Pipeline:
1. Stratified sample of documents (year bucket × chunker pattern) so the
   eval is not dominated by tiny AVISOs and keeps giant tables/compound
   decrees represented.
2. For each document, call an Anthropic-Messages-compatible LLM (default:
   Kimi k3-256k via https://api.kimi.com/coding) with the numbered chunks and
   ask for a JSON array of natural-language queries:
   - 1 paraphrase   (subject reworded, no key-term overlap, doc-level)
   - 1 thematic     (citizen-style topic question, no legal jargon, doc-level)
   - 2 factual      (answerable from ONE chunk, chunk-level ground truth)
   - 1 article_specific (only if the doc has "Artículo" sections, chunk-level)
3. Programmatic validation: JSON parse, format, chunk-index bounds, 5-gram
   overlap filter for paraphrase/thematic, dedupe. One LLM retry on failure.
4. Incremental JSONL output (resumable: docs already in the output are
   skipped) + a .meta.json sidecar with run stats.

The output feeds scripts/evaluate_retrieval.py --queries <jsonl>.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from random import Random

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_retrieval import _iter_md_files  # noqa: E402

from rag_poc.chunker import classify, split_file  # noqa: E402

DEFAULT_BASE_URL = "https://api.kimi.com/coding"
DEFAULT_API = "anthropic"
DEFAULT_MODEL = "k3-256k"
MAX_CHARS_PER_DOC = 12_000
MAX_CHARS_PER_CHUNK = 2_500
YEAR_BUCKETS = [(1999, 2009), (2010, 2015), (2016, 2020), (2021, 2027)]

QUERY_TYPES = ["paraphrase", "thematic", "factual", "article_specific"]

PROMPT_TEMPLATE = """\
Eres parte de un equipo que construye un evaluador de búsqueda para el \
Diario Oficial de la Federación (DOF) de México. A continuación va un \
documento del DOF dividido en chunks numerados.

TÍTULO: {title}

{chunks_block}

Genera consultas de búsqueda REALISTAS que una persona (ciudadano, abogado, \
funcionario) haría para encontrar este documento. Responde SOLO con un array \
JSON (sin markdown, sin explicaciones) con estas consultas:

1. Un objeto {{"type": "paraphrase", "question": "...", "chunk": null}}: \
reformula el tema del documento con otras palabras. NO copies 5 o más \
palabras consecutivas del documento ni del título.
2. Un objeto {{"type": "thematic", "question": "...", "chunk": null}}: una \
pregunta ciudadana sobre el TEMA general, sin usar jerga legal ni el nombre \
del decreto (ej: "¿qué apoyos hay para pescadores afectados por huracanes?"). \
NO copies 5 o más palabras consecutivas del documento.
3. Dos objetos {{"type": "factual", "question": "...", "chunk": N}}: preguntas \
específicas cuya respuesta está en UN SOLO chunk; indica su número en "chunk". \
Ejemplos: plazos, montos, requisitos, vigencias, quién debe cumplir qué.
{article_instruction}

Reglas:
- Español natural, como lo escribiría una persona real en un buscador.
- Las preguntas factual/thematic/article_specific deben terminar en "?".
- Las factual deben ser respondibles SOLO con el contenido del chunk indicado.
- Varía la dificultad: una factual fácil y una que requiera entender el texto.
"""

ARTICLE_INSTRUCTION = """\
4. Un objeto {{"type": "article_specific", "question": "...", "chunk": N}}: \
pregunta qué establece un artículo o sección específica (ej: "¿Qué establece \
el artículo 5 de este decreto?"), indicando el chunk donde aparece.
"""

NO_ARTICLE_INSTRUCTION = """\
4. (Este documento no tiene artículos numerados; NO incluyas article_specific.)
"""


# ── Sampling ──────────────────────────────────────────────────────────────
def _year_of(relpath: str) -> int:
    m = re.match(r"(\d{4})/", relpath)
    return int(m.group(1)) if m else 0


def _year_bucket(year: int) -> str:
    for lo, hi in YEAR_BUCKETS:
        if lo <= year <= hi:
            return f"{lo}-{hi}"
    return "other"


def stratified_sample(corpus: Path, n_docs: int, seed: int, pool_mult: int = 3) -> list[Path]:
    """Stratified sample by (year bucket × chunker pattern).

    Draws a uniform pool, classifies each file (cheap, no tokenization),
    then round-robins across strata until n_docs are selected.
    """
    rng = Random(seed)
    all_files = sorted(_iter_md_files(corpus))
    pool = rng.sample(all_files, min(len(all_files), n_docs * pool_mult))

    strata: dict[tuple[str, str], list[Path]] = {}
    for f in pool:
        rel = f.relative_to(corpus)
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            pattern = classify(text, f.stat().st_size).value
        except Exception:
            continue
        strata.setdefault((_year_bucket(_year_of(str(rel))), pattern), []).append(f)

    keys = sorted(strata, key=lambda k: -len(strata[k]))
    selected: list[Path] = []
    while len(selected) < n_docs and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(selected) < n_docs:
                selected.append(strata[k].pop())
    return selected


# ── LLM client ────────────────────────────────────────────────────────────
def _extract_text(payload: dict, api: str) -> str:
    if api == "openai":
        msg = payload["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or ""
    # anthropic-messages
    for block in payload.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return ""


def call_llm(prompt: str, base_url: str, model: str, api_key: str,
             api: str = "anthropic", max_tokens: int = 4000, retries: int = 3) -> str:
    """Call an LLM endpoint (anthropic-messages or openai-completions)."""
    if api == "openai":
        body = json.dumps({
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # opencode.ai sits behind Cloudflare; a browser-like UA is required.
            "User-Agent": "Mozilla/5.0",
        }
    else:  # anthropic-messages
        body = json.dumps({
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
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
            with urllib.request.urlopen(req, timeout=300) as resp:
                payload = json.load(resp)
            text = _extract_text(payload, api)
            if text:
                return text
            raise ValueError(f"empty content: {str(payload)[:200]}")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            code = getattr(exc, "code", None)
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1) * (5 if code in (402, 429) else 1)
            time.sleep(wait)
    return ""  # unreachable


# ── Validation ────────────────────────────────────────────────────────────
def _norm_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"\w+", text)


def _has_ngram_overlap(question: str, doc_text: str, n: int = 5) -> bool:
    q_words = _norm_words(question)
    if len(q_words) < n:
        return False
    doc_words = _norm_words(doc_text)
    doc_ngrams = {tuple(doc_words[i : i + n]) for i in range(len(doc_words) - n + 1)}
    return any(tuple(q_words[i : i + n]) in doc_ngrams for i in range(len(q_words) - n + 1))


def _parse_queries(raw: str) -> list[dict] | None:
    """Extract and parse the JSON array from the LLM response."""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def validate_queries(data: list[dict], doc_text: str, n_chunks: int,
                     has_articles: bool) -> tuple[list[dict], list[str]]:
    """Return (valid queries, rejection reasons)."""
    valid: list[dict] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            rejected.append("not a dict")
            continue
        qtype = item.get("type")
        question = str(item.get("question", "")).strip()
        chunk = item.get("chunk")
        if qtype not in QUERY_TYPES:
            rejected.append(f"bad type: {qtype}")
            continue
        if qtype == "article_specific" and not has_articles:
            rejected.append("article_specific in doc without articles")
            continue
        if len(question) < 15 or len(question) > 300:
            rejected.append(f"bad length: {len(question)}")
            continue
        if qtype != "paraphrase" and not question.endswith("?"):
            rejected.append("missing ?")
            continue
        norm = " ".join(_norm_words(question))
        if norm in seen:
            rejected.append("duplicate")
            continue
        if qtype in ("paraphrase", "thematic"):
            if _has_ngram_overlap(question, doc_text):
                rejected.append(f"{qtype} copies >=5 words")
                continue
        expected_chunk = None
        if qtype in ("factual", "article_specific"):
            if isinstance(chunk, int) and 0 <= chunk < n_chunks:
                expected_chunk = chunk
            else:
                rejected.append(f"chunk out of range: {chunk}")
                continue
        seen.add(norm)
        valid.append({
            "query": question,
            "type": qtype,
            "expected_chunk_index": expected_chunk,
        })
    return valid, rejected


# ── Per-document generation ───────────────────────────────────────────────
def build_prompt(title: str, chunks: list[str]) -> str:
    parts: list[str] = []
    total = 0
    for i, ch in enumerate(chunks):
        snippet = ch[:MAX_CHARS_PER_CHUNK]
        block = f"[CHUNK {i}]\n{snippet}\n"
        if total + len(block) > MAX_CHARS_PER_DOC:
            parts.append(f"[... {len(chunks) - i} chunks más omitidos ...]\n")
            break
        parts.append(block)
        total += len(block)
    full = "\n".join(chunks)
    article_instruction = (
        ARTICLE_INSTRUCTION if re.search(r"Artículo|ARTÍCULO|Articulo", full)
        else NO_ARTICLE_INSTRUCTION
    )
    return PROMPT_TEMPLATE.format(
        title=title,
        chunks_block="\n".join(parts),
        article_instruction=article_instruction,
    )


def generate_for_doc(
    path: Path, corpus: Path, base_url: str, model: str, api_key: str, api: str = "anthropic"
) -> dict:
    """Generate validated queries for one document. Never raises."""
    rel = str(path.relative_to(corpus))
    result: dict = {
        "doc_id": path.stem,
        "relpath": rel,
        "year": _year_of(rel),
        "error": None,
    }
    try:
        chunks = list(split_file(path))
        if not chunks:
            result["error"] = "no chunks"
            return result
        title = chunks[0].heading_path[0] if chunks[0].heading_path else path.stem
        result["title"] = title
        result["pattern"] = chunks[0].pattern.value
        result["n_chunks"] = len(chunks)

        full_text = "\n".join(ch.text for ch in chunks)
        has_articles = bool(re.search(r"Artículo|ARTÍCULO|Articulo", full_text))
        prompt = build_prompt(title, [ch.text for ch in chunks])

        queries: list[dict] = []
        rejected: list[str] = []
        for _attempt in range(2):  # one retry on parse/validation failure
            raw = call_llm(prompt, base_url, model, api_key, api=api)
            data = _parse_queries(raw)
            if data is None:
                rejected.append("unparseable JSON")
                continue
            queries, rejected = validate_queries(data, full_text, len(chunks), has_articles)
            if queries:
                break
        result["queries"] = queries
        result["rejected"] = rejected
        if not queries:
            result["error"] = "no valid queries"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="./dof_md")
    parser.add_argument("--docs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval/dof_queries_v2.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api", default=DEFAULT_API, choices=["anthropic", "openai"],
                        help="Protocolo del endpoint (default: anthropic)")
    parser.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retry-errors", action="store_true",
                        help="Reintentar docs con error ya presentes en el JSONL")
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("KIMI_API_KEY") or ""
    if not api_key:
        print("ERROR: export LLM_API_KEY (or KIMI_API_KEY)", file=sys.stderr)
        return 1

    corpus = Path(args.corpus).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip docs already present in the output file.
    done: set[str] = set()
    error_rels: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("error"):
                    error_rels.add(rec["relpath"])
                    if not args.retry_errors:
                        done.add(rec["relpath"])
                else:
                    done.add(rec["relpath"])
            except Exception:
                continue
        print(f"Resuming: {len(done)} valid, {len(error_rels)} error in {out_path}")

    print(f"Sampling {args.docs} documents (stratified, seed {args.seed})...")
    files = stratified_sample(corpus, args.docs, args.seed)
    if args.retry_errors:
        todo = [f for f in files if str(f.relative_to(corpus)) in error_rels]
        print(f"  retrying {len(todo)} errored docs")
    else:
        todo = [f for f in files if str(f.relative_to(corpus)) not in done]
    print(f"  {len(files)} selected, {len(todo)} to generate")

    stats = {"docs_ok": 0, "docs_error": 0, "queries": 0, "rejected": 0,
             "by_type": {t: 0 for t in QUERY_TYPES}}
    t0 = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_for_doc, f, corpus, args.base_url, args.model, api_key, args.api): f
            for f in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            if r["error"]:
                stats["docs_error"] += 1
            else:
                stats["docs_ok"] += 1
            for q in r.get("queries", []):
                stats["queries"] += 1
                stats["by_type"][q["type"]] += 1
            stats["rejected"] += len(r.get("rejected", []))
            if i % 25 == 0 or i == len(todo):
                rate = i / (time.perf_counter() - t0)
                print(f"  [{i}/{len(todo)}] {stats['queries']} queries, "
                      f"{stats['docs_error']} errors, {rate:.1f} docs/s")

    meta = {
        "corpus": str(corpus),
        "docs_requested": args.docs,
        "docs_generated": stats["docs_ok"],
        "docs_error": stats["docs_error"],
        "queries": stats["queries"],
        "rejected_queries": stats["rejected"],
        "by_type": stats["by_type"],
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "date": time.strftime("%Y-%m-%d"),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone: {stats['queries']} queries from {stats['docs_ok']} docs "
          f"({stats['docs_error']} errors, {stats['rejected']} rejected)")
    print(f"Output: {out_path}\nMeta:   {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
