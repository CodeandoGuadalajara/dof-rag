#!/usr/bin/env python3
"""
rasterize_metafiles.py
──────────────────────
Convert WMF/EMF metafiles to PNG using LibreOffice headless.
Designed to run in parallel for bulk conversion.

Usage
─────
  # Convert all WMF/EMF in a folder (8 parallel workers):
  python rasterize_metafiles.py --docs ./dof_md --workers 8

  # Dry-run to see how many files need conversion:
  python rasterize_metafiles.py --docs ./dof_md --dry-run

  # Clean up PNGs that were generated (keeps originals):
  python rasterize_metafiles.py --docs ./dof_md --cleanup

Requirements
────────────
  LibreOffice: brew install --cask libreoffice
"""

import argparse
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

METAFILE_EXTENSIONS = {".wmf", ".emf"}


def find_metafiles(docs_dir: Path) -> list[Path]:
    """Find all WMF/EMF files recursively."""
    files = []
    for ext in METAFILE_EXTENSIONS:
        files.extend(docs_dir.rglob(f"*{ext}"))
        files.extend(docs_dir.rglob(f"*{ext.upper()}"))
    return sorted(files)


def rasterize(src: Path, timeout: int = 30) -> tuple[Path, bool, str]:
    """
    Convert a single WMF/EMF to PNG via LibreOffice headless.
    Returns (source_path, success, message).
    """
    out_png = src.with_suffix(".png")
    if out_png.exists():
        return src, True, "already exists"

    try:
        result = subprocess.run(
            [
                "soffice", "--headless", "--convert-to", "png",
                "--outdir", str(src.parent), str(src),
            ],
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode == 0 and out_png.exists():
            return src, True, "converted"
        stderr = result.stderr.decode(errors="replace")[:200]
        return src, False, f"LibreOffice error: {stderr}"
    except FileNotFoundError:
        return src, False, "LibreOffice (soffice) not found"
    except subprocess.TimeoutExpired:
        return src, False, f"timed out ({timeout}s)"


def cleanup_pngs(docs_dir: Path) -> int:
    """Remove PNGs that were generated from WMF/EMF (keeps originals)."""
    metafiles = find_metafiles(docs_dir)
    removed = 0
    for mf in metafiles:
        png = mf.with_suffix(".png")
        if png.exists():
            png.unlink()
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WMF/EMF metafiles to PNG via LibreOffice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--docs", required=True,
                        help="Folder containing files to convert.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel LibreOffice instances (default: 8).")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout per file in seconds (default: 30).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be converted without doing it.")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove generated PNGs, keep original WMF/EMF.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-file results.")
    args = parser.parse_args()

    docs_dir = Path(args.docs).resolve()
    if not docs_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {docs_dir}")

    # ── Cleanup mode ─────────────────────────────────────────────────────────
    if args.cleanup:
        removed = cleanup_pngs(docs_dir)
        log.info(f"Removed {removed:,} generated PNGs.")
        return

    # ── Find files ───────────────────────────────────────────────────────────
    metafiles = find_metafiles(docs_dir)
    if not metafiles:
        log.info("No WMF/EMF files found.")
        return

    already_done = sum(1 for mf in metafiles if mf.with_suffix(".png").exists())
    to_convert = [mf for mf in metafiles if not mf.with_suffix(".png").exists()]

    log.info(f"WMF/EMF files: {len(metafiles):,}")
    log.info(f"Already converted: {already_done:,}")
    log.info(f"Need conversion: {to_convert:,}")

    if not to_convert:
        log.info("Nothing to do — all files already have PNGs.")
        return

    # ── Dry-run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        est_time = len(to_convert) * args.timeout / args.workers / 60
        log.info(f"[DRY-RUN] Would convert {len(to_convert):,} files "
                 f"with {args.workers} workers (~{est_time:.0f} min estimated)")
        for mf in to_convert[:20]:
            log.info(f"  {mf.relative_to(docs_dir)}")
        if len(to_convert) > 20:
            log.info(f"  ...and {len(to_convert) - 20:,} more.")
        return

    # ── Convert ──────────────────────────────────────────────────────────────
    converted = 0
    failed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(rasterize, mf, args.timeout): mf for mf in to_convert}
        for future in as_completed(futures):
            src, success, msg = future.result()
            if success:
                converted += 1
            else:
                failed += 1
                log.warning(f"FAILED {src.name}: {msg}")

            total_done = converted + failed
            if total_done % 100 == 0 or total_done == len(to_convert):
                elapsed = time.time() - t_start
                rate = total_done / elapsed if elapsed > 0 else 0
                remaining = (len(to_convert) - total_done) / rate if rate > 0 else 0
                log.info(
                    f"Progress: {total_done:,}/{len(to_convert):,} | "
                    f"{rate:.1f} files/s | ETA ~{remaining/60:.0f} min | "
                    f"failed: {failed}"
                )
            elif args.verbose:
                rel = src.relative_to(docs_dir)
                log.info(f"  {'OK' if success else 'FAIL'} {rel} — {msg}")

    elapsed = time.time() - t_start
    log.info(
        f"\n✅ Done in {elapsed/60:.1f} min — "
        f"{converted:,} converted | {failed} failed"
    )


if __name__ == "__main__":
    main()
