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
  export GEMINI_API_KEY="your-key"
  python enrich_markdown_images.py --docs ./docs --workers 15

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
  pip install google-genai pillow

Prerequisites
─────────────
  WMF/EMF metafiles should be pre-converted to PNG using rasterize_metafiles.py:
    python rasterize_metafiles.py --docs ./docs --workers 8

Rate limits (Gemini paid tier, as of May 2026)
───────────────────────────────────────────────
  2.5 Flash Lite: 4000 RPM (paid tier)
  Safe concurrency: --workers 15 (leaves headroom for retries)

Cost estimate (98k images, ~800 input tokens + image, ~200 output tokens)
──────────────────────────────────────────────────────────────────────────
  Flash Lite standard: ~$1.56  (input $0.78 + output $0.78)
  Flash Lite batch:    ~$0.78  (50% discount, async, 24h turnaround)
  Both essentially free for a one-time indexing run.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

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

# Change to "gemini-2.5-flash" for higher quality (~$11 total vs ~$1.56)
GEMINI_MODEL = "gemini-2.5-flash-lite"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"
}
IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

# ─────────────────────────────────────────────────────────────────────────────
# Prompt — retrieval-optimized, context-aware
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Eres un sistema de indexacion para un motor RAG (Retrieval Augmented Generation) "
    "sobre documentos legales mexicanos en espanol.\n\n"
    "Tu unica tarea es generar una descripcion de imagenes optimizada para busqueda semantica. "
    "La imagen original estara disponible en la fase de generacion de respuestas, "
    "por lo que NO debes describir aspectos visuales como colores, bordes o diseno grafico.\n\n"
    "Genera una descripcion que incluya obligatoriamente:\n"
    "1. TIPO: Indica si es tabla, diagrama, grafica, mapa, organigrama, figura, fotografia, etc.\n"
    "2. IDENTIFICADORES LEGALES: Numero de articulo, fraccion, inciso, nombre del reglamento, "
    "decreto, ley, norma oficial mexicana (NOM), DOF, fecha o cualquier referencia legal "
    "que aparezca en la imagen o se infiera del contexto del documento.\n"
    "3. CONTENIDO LITERAL: Todos los valores numericos, rangos, categorias, claves, "
    "abreviaturas y terminos tecnicos exactamente como aparecen "
    "(ej. UMAS, VSM, UMA, categoria I-V, puntajes, montos, porcentajes, plazos).\n"
    "4. VOCABULARIO DE BUSQUEDA: Los terminos legales y tecnicos en espanol que un abogado, "
    "funcionario publico, notario o investigador usaria al buscar este contenido especifico.\n\n"
    "NO incluyas:\n"
    "- Descripciones visuales (colores, fuentes, bordes, sombreado, diseno)\n"
    "- Frases introductorias como 'Esta imagen muestra...' o 'La tabla presenta...'\n"
    "- Comillas, markdown, listas con guiones, o cualquier formato especial\n"
    "- Repeticion innecesaria\n\n"
    "Responde UNICAMENTE con la descripcion en texto corrido, en espanol, entre 4 y 8 oraciones."
)


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
    md_text: str, img_ref: str, before: int = 800, after: int = 200
) -> str:
    """Grab text around an image reference, stripping other image tags."""
    pos = md_text.find(f"]({img_ref})")
    if pos == -1:
        return ""
    start = max(0, pos - before)
    end = min(len(md_text), pos + after)
    snippet = md_text[start:end]
    snippet = IMAGE_RE.sub("", snippet)
    snippet = re.sub(r"\n{3,}", "\n\n", snippet).strip()
    return snippet


# ─────────────────────────────────────────────────────────────────────────────
# Cache — SHA-256 keyed, thread-safe
# ─────────────────────────────────────────────────────────────────────────────
class DescriptionCache:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self._data: dict[str, str] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )

    @staticmethod
    def _hash(img_path: Path) -> str:
        h = hashlib.sha256()
        with open(img_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:20]

    def get(self, img_path: Path) -> Optional[str]:
        with self._lock:
            return self._data.get(self._hash(img_path))

    def set(self, img_path: Path, description: str) -> None:
        with self._lock:
            self._data[self._hash(img_path)] = description

    def save(self) -> None:
        with self._lock:
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def __len__(self) -> int:
        return len(self._data)


# ─────────────────────────────────────────────────────────────────────────────
# Backend: Gemini 2.5 Flash Lite  (official google-genai SDK)
# ─────────────────────────────────────────────────────────────────────────────
_gemini_client = None
_gemini_lock = Lock()


def _get_gemini_client(api_key: str):
    global _gemini_client
    with _gemini_lock:
        if _gemini_client is None:
            from google import genai
            _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def caption_gemini(
    img_path: Path,
    surrounding_text: str,
    api_key: str,
    max_retries: int = 5,
) -> str:
    from google.genai import types

    client = _get_gemini_client(api_key)

    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    mime_type = mime_map.get(img_path.suffix.lower(), "image/png")
    user_text = build_user_prompt(surrounding_text)

    with open(img_path, "rb") as f:
        image_bytes = f.read()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=mime_type,
                                    data=image_bytes,
                                )
                            ),
                            types.Part(text=user_text),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=512,
                    temperature=0.1,
                ),
            )
            return response.text.strip()

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = (2 ** attempt) * 5  # 5, 10, 20, 40, 80 seconds
                log.warning(f"Rate limited — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "500" in err or "503" in err:
                time.sleep(3 * (attempt + 1))
            else:
                raise

    raise RuntimeError(f"Gemini failed after {max_retries} retries for {img_path.name}")


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

            pos = text.find(full_match)
            preceding = text[max(0, pos - 250):pos]
            if "IMAGE_DESCRIPTION:" in preceding:
                skipped_done += 1
                continue

            img_path = (md_path.parent / img_ref).resolve()
            if not img_path.exists() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped_missing += 1
                continue

            surrounding = extract_surrounding_text(text, img_ref)
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
    cache: DescriptionCache,
    gemini_key: str,
) -> tuple[ImageTask, Optional[str]]:

    cached = cache.get(task.img_path)
    if cached:
        return task, cached

    try:
        t0 = time.time()
        desc = caption_gemini(task.img_path, task.surrounding_text, gemini_key)
        log.debug(f"{task.img_path.name}: {time.time()-t0:.1f}s")
        cache.set(task.img_path, desc)
        return task, desc
    except Exception as e:
        log.error(f"FAILED {task.img_path.name}: {e}")
        return task, None


# ─────────────────────────────────────────────────────────────────────────────
# Apply descriptions back to markdown files (one write per file)
# ─────────────────────────────────────────────────────────────────────────────
def apply_descriptions(
    results: list[tuple[ImageTask, Optional[str]]], dry_run: bool
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
    parser.add_argument("--workers", type=int, default=15,
                        help="Concurrent API calls (default: 15).")
    parser.add_argument("--gemini-key", default=os.environ.get("GEMINI_API_KEY"),
                        help="Gemini API key. Alternatively set GEMINI_API_KEY env var.")
    parser.add_argument("--glob", default="**/*.md",
                        help="Glob pattern for markdown files (default: **/*.md).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover and report tasks without calling APIs or writing files.")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save cache to disk every N processed images (default: 100).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-image timing in logs.")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    docs_dir = Path(args.docs).resolve()
    if not docs_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {docs_dir}")

    # ── Validation ───────────────────────────────────────────────────────────
    if not args.gemini_key:
        sys.exit("ERROR: set GEMINI_API_KEY or pass --gemini-key.")
    try:
        from google import genai  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pip install google-genai")

    # ── Discover files ───────────────────────────────────────────────────────
    md_files = sorted(docs_dir.glob(args.glob))
    if not md_files:
        sys.exit(f"No markdown files found with '{args.glob}' in {docs_dir}")
    log.info(f"Markdown files: {len(md_files):,}")

    cache = DescriptionCache(docs_dir / ".image_caption_cache.json")
    log.info(f"Cache: {len(cache):,} existing entries.")

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
        # Rough cost estimate
        est_input_tok = len(tasks) * 800
        est_output_tok = len(tasks) * 200
        est_cost = est_input_tok * 0.10 / 1e6 + est_output_tok * 0.40 / 1e6
        log.info(f"[DRY-RUN] Estimated cost (Flash Lite): ${est_cost:.2f}")
        log.info("[DRY-RUN] No API calls made, no files written.")
        return

    # ── Process ──────────────────────────────────────────────────────────────
    results: list[tuple[ImageTask, Optional[str]]] = []
    done_count = 0
    error_count = 0
    cached_count = 0
    t_start = time.time()

    def _run(task: ImageTask) -> tuple[ImageTask, Optional[str]]:
        return process_task(task, cache, args.gemini_key)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, t): t for t in tasks}
        for future in as_completed(futures):
            task, desc = future.result()
            results.append((task, desc))

            if desc is None:
                error_count += 1
            elif cache.get(task.img_path) == desc and done_count == 0:
                cached_count += 1
            else:
                done_count += 1

            total_done = done_count + error_count + cached_count
            if total_done % args.save_every == 0:
                cache.save()
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
    cache.save()

    elapsed = time.time() - t_start
    log.info(
        f"\n✅ Done in {elapsed/60:.1f} min — "
        f"{written:,} images captioned | "
        f"{error_count} errors | "
        f"cache: {docs_dir / '.image_caption_cache.json'}"
    )


if __name__ == "__main__":
    main()
