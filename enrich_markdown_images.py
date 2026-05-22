#!/usr/bin/env python3
"""
enrich_markdown_images.py
─────────────────────────
Scans a folder of markdown files, finds linked images, generates
retrieval-optimized VLM captions, and rewrites the markdown with the
description injected as an HTML comment — ready for RAG chunking.

The caption is optimized for RETRIEVAL (legal terms, article numbers,
exact values). The original image is stored alongside for use at
GENERATION time by the answer LLM.

Usage
─────
  export OPENROUTER_API_KEY="***"
  python enrich_markdown_images.py --docs ./docs --workers 15

  # Try a different model:
  python enrich_markdown_images.py --docs ./docs --model openai/gpt-4o-mini

  # Dry-run first to verify image discovery and context extraction:
  python enrich_markdown_images.py --docs ./docs --dry-run

Output format per image
───────────────────────
  <!-- IMAGE_DESCRIPTION: path/to/image.png
  Tabla de subsidios federales. Articulo 47 del Reglamento Nacional de
  Vivienda 2023. Categorias I-V segun valor de vivienda en UMAS (60-190).
  Rangos de puntaje: 0 a 1000. Montos maximos: 13-35 UMAS segun categoria
  y puntaje. Terminos: subsidio federal, CONAVI, politica habitacional.
  -->
  ![Tabla de subsidios — Art. 47 Reglamento Nacional de Vivienda](path/to/image.png)

Install
───────
  pip install openai

Prerequisites
─────────────
  WMF/EMF metafiles should be pre-converted to PNG using rasterize_metafiles.py:
    python rasterize_metafiles.py --docs ./docs --workers 8

Recommended models (cost for ~97k images, tested with prompt v3)
──────────────────────────────────────────────────────────────────
  google/gemini-2.5-flash-lite    ~$41     (tested: 100% success, 2.3s/img)
  google/gemini-2.5-flash         ~$62     (better OCR on dense tables)
  openai/gpt-4o-mini              ~$52     (solid vision, good Spanish)
"""

import argparse
import base64
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Cost estimate constants (from vlm_batch_100 run, Gemini Flash Lite pricing)
COST_AVG_INPUT_TOKENS = 2144
COST_AVG_OUTPUT_TOKENS = 171
COST_INPUT_PER_M = 0.10   # $/M tokens
COST_OUTPUT_PER_M = 0.40  # $/M tokens

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"
}
IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

# ─────────────────────────────────────────────────────────────────────────────
# Prompt — retrieval-optimized, context-aware
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
Eres un sistema de indexación para un motor RAG sobre documentos legales \
mexicanos (Diario Oficial de la Federación).

Tu tarea es generar una descripción de esta imagen optimizada para búsqueda \
semántica. La imagen original estará disponible al generar la respuesta final, \
así que no describas aspectos visuales como colores, bordes o diseño.

Si el contexto del documento incluye el título o caption de la figura \
(por ejemplo "FIGURA 1 Flexómetro"), úsalo como punto de partida — tiene \
más peso que tu interpretación visual.

Si la imagen es ambigua o de baja resolución, infiere el contenido a partir \
del contexto del documento.

Si el contexto del documento no parece relacionado con el contenido visual \
de la imagen, prioriza lo que ves en la imagen sobre el contexto.

Escribe un párrafo continuo en español de 4 a 6 oraciones que incluya:
- El tipo de imagen (tabla, diagrama, gráfica, mapa, logotipo, formato administrativo, etc.)
- Los identificadores legales que aparezcan en la imagen o se infieran del contexto: número de artículo, fracción, NOM, decreto, ley, DOF, fecha, nombre de dependencia
- Si no hay identificadores legales no menciones ninguno
- Todo el contenido literal relevante: valores numéricos, rangos, categorías, claves, abreviaturas, nombres propios exactamente como aparecen
- Los términos que un abogado, funcionario o investigador usaría para buscar este contenido

No listes elementos que ya aparecen en el texto circundante del documento.

No uses encabezados, etiquetas (TIPO:, CONTENIDO LITERAL:), viñetas, comillas ni markdown. Solo texto corrido.
"""


def build_user_prompt(surrounding_text: str = "") -> str:
    if not surrounding_text.strip():
        return "Describe esta imagen para indexacion RAG."
    return (
        f"Contexto del documento donde aparece esta imagen:\n"
        f'"""\n{surrounding_text[:700]}\n"""\n\n'
        "Con base en este contexto y en la imagen, genera la descripcion para indexacion RAG."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Context extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_surrounding_text(
    md_text: str, match_start: int, match_end: int, before: int = 800, after: int = 200
) -> str:
    """Grab text around an image reference, stripping other image tags."""
    start = max(0, match_start - before)
    end = min(len(md_text), match_end + after)
    snippet = md_text[start:end]
    snippet = IMAGE_RE.sub("", snippet)
    snippet = re.sub(r"\n{3,}", "\n\n", snippet).strip()
    return snippet


# ─────────────────────────────────────────────────────────────────────────────
# Backend: OpenRouter (OpenAI-compatible API)
# ─────────────────────────────────────────────────────────────────────────────
_client = None
_client_lock = Lock()


def _get_client(api_key: str):
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)
    return _client


def caption_image(
    img_path: Path,
    surrounding_text: str,
    api_key: str,
    model: str,
    max_retries: int = 5,
) -> str:
    client = _get_client(api_key)
    user_text = build_user_prompt(surrounding_text)

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    mime_type = mime_map.get(img_path.suffix.lower(), "image/png")
    data_url = f"data:{mime_type};base64,{b64}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_text},
                    ]},
                ],
                max_tokens=512,
                temperature=0.1,
            )
            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError(
                    f"Empty response from {model} for {img_path.name} — "
                    f"choices={len(response.choices) if response.choices else 0}"
                )
            return response.choices[0].message.content.strip()

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = (2 ** attempt) * 5
                log.warning(f"Rate limited — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "500" in err or "503" in err:
                time.sleep(3 * (attempt + 1))
            else:
                raise

    raise RuntimeError(f"API failed after {max_retries} retries for {img_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Work item
# ─────────────────────────────────────────────────────────────────────────────
class ImageTask:
    __slots__ = ("md_path", "img_ref", "img_path", "surrounding_text", "full_match", "alt_text")

    def __init__(self, md_path, img_ref, img_path, surrounding_text, full_match, alt_text):
        self.md_path = md_path
        self.img_ref = img_ref
        self.img_path = img_path
        self.surrounding_text = surrounding_text
        self.full_match = full_match
        self.alt_text = alt_text


# ─────────────────────────────────────────────────────────────────────────────
# Collect tasks
# ─────────────────────────────────────────────────────────────────────────────
def collect_tasks(md_files: list[Path], docs_dir: Path) -> list[ImageTask]:
    tasks: list[ImageTask] = []
    skipped_done = 0
    skipped_missing = 0

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8", errors="replace")
        for m in IMAGE_RE.finditer(text):
            full_match = m.group(0)
            alt_text = m.group(1)
            img_ref = m.group(2)

            if img_ref.startswith(("http://", "https://")):
                continue

            pos = m.start()
            preceding = text[max(0, pos - 250):pos]
            if "IMAGE_DESCRIPTION:" in preceding:
                skipped_done += 1
                continue

            img_path = (md_path.parent / img_ref).resolve()
            if not img_path.exists() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped_missing += 1
                continue

            # Reject paths outside docs_dir (path traversal safety)
            try:
                img_path.relative_to(docs_dir)
            except ValueError:
                log.warning(f"Skipping image outside docs_dir: {img_ref}")
                continue

            surrounding = extract_surrounding_text(text, m.start(), m.end())
            tasks.append(
                ImageTask(md_path, img_ref, img_path, surrounding, full_match, alt_text)
            )

    if skipped_done:
        log.info(f"Skipped {skipped_done:,} already-described images (idempotent).")
    if skipped_missing:
        log.warning(f"Skipped {skipped_missing:,} images with missing/unresolvable files.")

    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Process one task
# ─────────────────────────────────────────────────────────────────────────────
def process_task(
    task: ImageTask,
    api_key: str,
    model: str,
) -> tuple[ImageTask, str | None]:

    try:
        t0 = time.time()
        desc = caption_image(task.img_path, task.surrounding_text, api_key, model)
        log.debug(f"{task.img_path.name}: {time.time()-t0:.1f}s")
        return task, desc
    except Exception as e:
        log.error(f"FAILED {task.img_path.name}: {e}")
        return task, None


# ─────────────────────────────────────────────────────────────────────────────
# Apply descriptions back to markdown files (one write per file)
# ─────────────────────────────────────────────────────────────────────────────
def apply_descriptions(
    results: list[tuple[ImageTask, str | None]], dry_run: bool
) -> int:
    from collections import defaultdict
    by_file: dict[Path, list[tuple[ImageTask, str]]] = defaultdict(list)
    for task, desc in results:
        if desc is not None:
            by_file[task.md_path].append((task, desc))

    written = 0
    for md_path, file_results in by_file.items():
        text = md_path.read_text(encoding="utf-8", errors="replace")
        modified = False

        for task, description in file_results:
            if task.full_match not in text:
                continue
            short_title = description.split(".")[0].strip()[:120]
            enriched_alt = short_title if not task.alt_text else task.alt_text
            replacement = (
                f"<!-- IMAGE_DESCRIPTION: {task.img_ref}\n"
                f"{description}\n"
                f"-->\n"
                f"![{enriched_alt}]({task.img_ref})"
            )
            text = text.replace(task.full_match, replacement, 1)
            modified = True
            written += 1

        if modified and not dry_run:
            shutil.copy2(md_path, md_path.with_suffix(".md.bak"))
            md_path.write_text(text, encoding="utf-8")

    return written


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich markdown legal docs with retrieval-optimized VLM image captions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--docs", required=True,
                        help="Folder containing .md files and images.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenRouter model ID (default: {DEFAULT_MODEL}).")
    parser.add_argument("--workers", type=int, default=15,
                        help="Concurrent API calls (default: 15).")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key. Alternatively set OPENROUTER_API_KEY env var.")
    parser.add_argument("--glob", default="**/*.md",
                        help="Glob pattern for markdown files (default: **/*.md).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover and report tasks without calling APIs or writing files.")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="Log progress every N processed images (default: 100).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-image timing in logs.")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    docs_dir = Path(args.docs).resolve()
    if not docs_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {docs_dir}")

    # ── Validation ───────────────────────────────────────────────────────────
    if not args.api_key:
        sys.exit("ERROR: set OPENROUTER_API_KEY or pass --api-key.")
    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pip install openai")

    log.info(f"Model: {args.model}")

    # ── Discover files ───────────────────────────────────────────────────────
    md_files = sorted(docs_dir.glob(args.glob))
    if not md_files:
        sys.exit(f"No markdown files found with '{args.glob}' in {docs_dir}")
    log.info(f"Markdown files: {len(md_files):,}")

    log.info("Scanning for image references...")
    tasks = collect_tasks(md_files, docs_dir)
    log.info(f"Images to caption: {len(tasks):,}")

    if not tasks:
        log.info("Nothing to do — all images already captioned.")
        return

    # ── Dry-run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("[DRY-RUN] First 20 tasks:")
        for t in tasks[:20]:
            rel = t.img_path.relative_to(docs_dir)
            ctx_preview = t.surrounding_text[:80].replace("\n", " ")
            log.info(f"  {rel}  |  ctx: {ctx_preview!r}...")
        if len(tasks) > 20:
            log.info(f"  ...and {len(tasks) - 20:,} more.")
        # Rough cost estimate based on batch-100 results
        est_input_tok = len(tasks) * COST_AVG_INPUT_TOKENS
        est_output_tok = len(tasks) * COST_AVG_OUTPUT_TOKENS
        est_cost = est_input_tok * COST_INPUT_PER_M / 1e6 + est_output_tok * COST_OUTPUT_PER_M / 1e6
        log.info(f"[DRY-RUN] Estimated cost (Flash Lite): ${est_cost:.2f}")
        log.info("[DRY-RUN] No API calls made, no files written.")
        return

    # ── Process ──────────────────────────────────────────────────────────────
    results: list[tuple[ImageTask, str | None]] = []
    done_count = 0
    error_count = 0
    t_start = time.time()

    def _run(task: ImageTask) -> tuple[ImageTask, str | None]:
        return process_task(task, args.api_key, args.model)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, t): t for t in tasks}
        for future in as_completed(futures):
            task, desc = future.result()
            results.append((task, desc))

            if desc is None:
                error_count += 1
            else:
                done_count += 1

            total_done = done_count + error_count
            if total_done % args.progress_every == 0:
                elapsed = time.time() - t_start
                rate = total_done / elapsed if elapsed > 0 else 0
                remaining = (len(tasks) - total_done) / rate if rate > 0 else 0
                log.info(
                    f"Progress: {total_done:,}/{len(tasks):,} | "
                    f"{rate:.1f} img/s | ETA ~{remaining/60:.0f} min | "
                    f"errors: {error_count}"
                )

    # ── Write ────────────────────────────────────────────────────────────────
    log.info("Writing enriched markdown files...")
    written = apply_descriptions(results, dry_run=False)

    elapsed = time.time() - t_start
    log.info(
        f"\n✅ Done in {elapsed/60:.1f} min — "
        f"{written:,} images captioned | "
        f"{error_count} errors"
    )


if __name__ == "__main__":
    main()
