#!/usr/bin/env python3
"""
vlm_batch_100.py
────────────────
Run Gemini 2.5 Flash Lite on 100 randomly sampled images from DOF documents.
Tests the prompt v3 with context mismatch and anti-duplication instructions.

Usage
─────
  export OPENROUTER_API_KEY="***"
  python vlm_batch_100.py

  # Custom sample size:
  python vlm_batch_100.py --sample 50

  # Different model:
  python vlm_batch_100.py --model google/gemini-2.5-flash

Output: vlm_batch_100_results.json + vlm_batch_100_report.md
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DOF_MD_DIR = Path(__file__).parent / "dof_md"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

# ─────────────────────────────────────────────────────────────────────────────
# Prompt v3 — context mismatch + anti-duplication
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_V3 = """\
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


def build_user_prompt(context: str = "") -> str:
    if not context.strip():
        return "Describe esta imagen para indexación RAG."
    return (
        f"Contexto del documento donde aparece esta imagen:\n"
        f'"""\n{context[:700]}\n"""\n\n'
        "Con base en este contexto y en la imagen, genera la descripción para indexación RAG."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Context extraction (from enrich_markdown_images.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_surrounding_text(md_text: str, img_ref: str, before: int = 800, after: int = 200) -> str:
    """Grab text around an image reference, stripping other image tags."""
    pos = md_text.find(f"]({img_ref})")
    if pos == -1:
        return ""
    # match_end is the position right after the full ![alt](ref) syntax
    match_end = pos + 2 + len(img_ref) + 1  # ](img_ref)
    start = max(0, pos - before)
    end = min(len(md_text), match_end + after)
    snippet = md_text[start:end]
    snippet = IMAGE_RE.sub("", snippet)
    snippet = re.sub(r"\n{3,}", "\n\n", snippet).strip()
    return snippet


# ─────────────────────────────────────────────────────────────────────────────
# Find image references across all markdown files
# ─────────────────────────────────────────────────────────────────────────────

def discover_image_refs(dof_dir: Path, max_scan: int = 5000) -> list[dict]:
    """
    Scan markdown files for image references with context.
    Returns list of {img_path, md_path, img_ref, context, alt_text}.
    Samples max_scan md files to keep discovery fast.
    """
    md_files = sorted(dof_dir.rglob("*.md"))
    # Random sample of files to scan (there are tens of thousands)
    if len(md_files) > max_scan:
        random.shuffle(md_files)
        md_files = md_files[:max_scan]

    refs = []
    seen_images = set()

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8", errors="replace")
        for m in IMAGE_RE.finditer(text):
            img_ref = m.group(2)
            alt_text = m.group(1)

            # Skip external images
            if img_ref.startswith(("http://", "https://")):
                continue

            img_path = (md_path.parent / img_ref).resolve()
            if not img_path.exists() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Deduplicate by image path
            img_key = str(img_path)
            if img_key in seen_images:
                continue
            seen_images.add(img_key)

            # Skip very small images (< 5KB, likely logos/icons)
            if img_path.stat().st_size < 5120:
                continue

            context = extract_surrounding_text(text, img_ref)

            refs.append({
                "img_path": img_path,
                "md_path": md_path,
                "img_ref": img_ref,
                "context": context,
                "alt_text": alt_text,
            })

    return refs


# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────

def caption_image(client, img_path: Path, context: str, model: str) -> dict:
    user_text = build_user_prompt(context)

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    }
    mime_type = mime_map.get(img_path.suffix.lower(), "image/png")
    data_url = f"data:{mime_type};base64,{b64}"

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_V3},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        elapsed = time.time() - t0
        usage = response.usage
        return {
            "text": response.choices[0].message.content.strip(),
            "time": round(elapsed, 1),
            "model": model,
            "error": None,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "text": None,
            "time": round(elapsed, 1),
            "model": model,
            "error": str(e),
            "input_tokens": None,
            "output_tokens": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch VLM caption test on 100 DOF images")
    parser.add_argument("--sample", type=int, default=100, help="Number of images to test")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between API calls (seconds)")
    parser.add_argument("--max-scan", type=int, default=5000, help="Max markdown files to scan")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: set OPENROUTER_API_KEY")
        return

    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pip install openai")

    random.seed(args.seed)
    client = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_API_KEY)

    # ── Discover image references ──────────────────────────────────────────
    print(f"Scanning markdown files for images (max {args.max_scan} files)...")
    all_refs = discover_image_refs(DOF_MD_DIR, max_scan=args.max_scan)
    print(f"Found {len(all_refs)} unique images (>5KB)")

    if len(all_refs) < args.sample:
        print(f"WARNING: only found {len(all_refs)} images, using all of them")
        sample = all_refs
    else:
        sample = random.sample(all_refs, args.sample)

    print(f"\nSampled {len(sample)} images")
    print(f"Model: {args.model}")
    print(f"Prompt: v3 (context mismatch + anti-duplication)")
    print()

    # ── Run captions ───────────────────────────────────────────────────────
    results = []
    out_json = Path("vlm_batch_100_results.json")

    total_input_tokens = 0
    total_output_tokens = 0
    ok_count = 0
    error_count = 0

    for i, ref in enumerate(sample):
        img_rel = ref["img_path"].relative_to(DOF_MD_DIR)
        print(f"[{i+1}/{len(sample)}] {img_rel}", end=" ", flush=True)

        r = caption_image(client, ref["img_path"], ref["context"], args.model)

        if r["input_tokens"]:
            total_input_tokens += r["input_tokens"]
        if r["output_tokens"]:
            total_output_tokens += r["output_tokens"]

        if r["error"]:
            print(f"ERROR ({r['time']}s): {r['error'][:80]}", flush=True)
            error_count += 1
        else:
            print(f"OK ({r['time']}s, {r['output_tokens']}tok)", flush=True)
            ok_count += 1

        results.append({
            "img_path": str(img_rel),
            "md_path": str(ref["md_path"].relative_to(DOF_MD_DIR)),
            "alt_text": ref["alt_text"],
            "context_preview": ref["context"][:200] if ref["context"] else "",
            "caption": r["text"],
            "time": r["time"],
            "error": r["error"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "file_size_kb": round(ref["img_path"].stat().st_size / 1024, 1),
        })

        # Save partial results after each call
        intermediate = {
            "meta": {
                "model": args.model,
                "prompt_version": "v3",
                "seed": args.seed,
                "sample_size": len(sample),
                "completed": i + 1,
                "ok": ok_count,
                "errors": error_count,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
            },
            "results": results,
        }
        out_json.write_text(json.dumps(intermediate, ensure_ascii=False, indent=2))

        time.sleep(args.delay)

    # ── Final save ─────────────────────────────────────────────────────────
    final = {
        "meta": {
            "model": args.model,
            "prompt_version": "v3",
            "seed": args.seed,
            "sample_size": len(sample),
            "completed": len(sample),
            "ok": ok_count,
            "errors": error_count,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
        "results": results,
    }
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2))

    # ── Generate report ────────────────────────────────────────────────────
    report = generate_report(final)
    out_md = Path("vlm_batch_100_report.md")
    out_md.write_text(report, encoding="utf-8")

    # ── Summary ────────────────────────────────────────────────────────────
    avg_time = sum(r["time"] for r in results) / len(results) if results else 0
    print(f"\n{'='*60}")
    print(f"Done! {ok_count} OK, {error_count} errors")
    print(f"Avg time/image: {avg_time:.1f}s")
    print(f"Total tokens: {total_input_tokens:,} in + {total_output_tokens:,} out")
    print(f"Results: {out_json}")
    print(f"Report: {out_md}")


def generate_report(data: dict) -> str:
    meta = data["meta"]
    results = data["results"]
    lines = []

    lines.append("# VLM Batch Test: 100 imágenes DOF con Gemini 2.5 Flash Lite\n")
    lines.append(f"**Modelo:** `{meta['model']}`")
    lines.append(f"**Prompt:** v3 (context mismatch + anti-duplication)")
    lines.append(f"**Imágenes:** {meta['sample_size']}")
    lines.append(f"**Seed:** {meta['seed']}")
    lines.append(f"**Resultados:** {meta['ok']} OK, {meta['errors']} errores")
    lines.append("")

    # Timing stats
    times = [r["time"] for r in results if r["time"]]
    if times:
        avg_t = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)
        lines.append("## Estadísticas de tiempo\n")
        lines.append(f"| Métrica | Valor |")
        lines.append(f"|---------|-------|")
        lines.append(f"| Promedio | {avg_t:.1f}s |")
        lines.append(f"| Mínimo | {min_t:.1f}s |")
        lines.append(f"| Máximo | {max_t:.1f}s |")
        lines.append(f"| Tiempo total | {sum(times):.0f}s ({sum(times)/60:.1f} min) |")
        lines.append("")

    # Token stats
    lines.append("## Tokens\n")
    lines.append(f"| Tipo | Cantidad |")
    lines.append(f"|------|----------|")
    lines.append(f"| Input | {meta['total_input_tokens']:,} |")
    lines.append(f"| Output | {meta['total_output_tokens']:,} |")
    lines.append("")

    # Image size distribution
    sizes = [r["file_size_kb"] for r in results]
    if sizes:
        lines.append("## Distribución de tamaño de imagen\n")
        lines.append(f"| Métrica | Valor |")
        lines.append(f"|---------|-------|")
        lines.append(f"| Promedio | {sum(sizes)/len(sizes):.0f} KB |")
        lines.append(f"| Mínimo | {min(sizes):.0f} KB |")
        lines.append(f"| Máximo | {max(sizes):.0f} KB |")
        lines.append("")

    # Year distribution
    years = {}
    for r in results:
        parts = r["img_path"].split("/")
        year = parts[0] if parts else "unknown"
        years[year] = years.get(year, 0) + 1
    if years:
        lines.append("## Distribución por año\n")
        lines.append("| Año | Imágenes |")
        lines.append("|-----|----------|")
        for y in sorted(years.keys()):
            lines.append(f"| {y} | {years[y]} |")
        lines.append("")

    # Sample of results (first 20 with captions)
    lines.append("## Muestra de resultados (20 primeras imágenes con caption)\n")
    count = 0
    for r in results:
        if count >= 20:
            break
        if not r["caption"]:
            continue

        lines.append(f"### {r['img_path']}\n")
        lines.append(f"- **Tamaño:** {r['file_size_kb']:.0f} KB")
        lines.append(f"- **Tiempo:** {r['time']}s")
        lines.append(f"- **Alt text original:** {r['alt_text'][:80] if r['alt_text'] else '(vacío)'}")
        if r["context_preview"]:
            ctx = r["context_preview"].replace("\n", " ")[:150]
            lines.append(f"- **Contexto:** {ctx}...")
        lines.append(f"\n> {r['caption']}\n")
        count += 1

    lines.append("---\n")
    lines.append(f"*Generado por vlm_batch_100.py — {time.strftime('%Y-%m-%d %H:%M')}*")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
