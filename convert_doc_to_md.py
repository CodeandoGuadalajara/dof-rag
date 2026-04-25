#!/usr/bin/env python3
"""
Convert DOF .doc files directly to Markdown.
Pipeline: .doc -> .docx (LibreOffice) -> .md (pandoc with lua filter)

Reads .doc files from input directory (default: ./dof_word).
Writes .md files to output directory (default: ./dof_md).

Uses parallel workers. Each file gets retried up to 3 times on LibreOffice failure.

Requirements:
    - LibreOffice (soffice) installed
    - pandoc installed
    - pandoc_filters/dof_headers.lua present

Usage:
    python convert_doc_to_md.py                          # Convert all years
    python convert_doc_to_md.py --years 2020 2021 2022   # Specific years
    python convert_doc_to_md.py --workers 4              # Limit parallelism
    python convert_doc_to_md.py --dry-run                # Just count files
    python convert_doc_to_md.py --retry-failed           # Retry only failed files
    python convert_doc_to_md.py --input-dir /path/to/docs --output-dir /path/to/md
"""

import argparse
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Configuration
PANDOC_FILTER = Path(__file__).parent / "pandoc_filters" / "dof_headers.lua"
LOG_FILE = Path(__file__).parent / "convert_doc_to_md.log"

# Defaults (overridden by CLI args)
INPUT_DIR = Path("./dof_word")
OUTPUT_DIR = Path("./dof_md")
FAILED_DIR = Path("./dof_failed")

# Timeout per file conversion (seconds)
LIBREOFFICE_TIMEOUT = 90
PANDOC_TIMEOUT = 120
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def setup_libreoffice_profile(worker_id: int) -> Path:
    """Create a unique LibreOffice user profile for this worker."""
    profile_dir = Path(tempfile.gettempdir()) / f"lo_profile_worker_{worker_id}"
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


def convert_single_doc(doc_path: Path, output_md_path: Path, worker_id: int = 0) -> dict:
    """
    Convert a single .doc file to .md with retries.
    Pipeline: .doc -> .docx (LibreOffice headless) -> .md (pandoc)
    """
    result = {
        "doc_path": str(doc_path),
        "output_path": str(output_md_path),
        "status": "unknown",
        "error": None,
        "size_bytes": 0,
    }

    if output_md_path.exists() and output_md_path.stat().st_size > 0:
        result["status"] = "skipped"
        return result

    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        # Add small delay between retries to reduce LO contention
        if attempt > 1:
            time.sleep(0.5 * attempt)

        with tempfile.TemporaryDirectory(prefix=f"dof_w{worker_id}_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            tmp_doc = tmp_path / doc_path.name
            tmp_docx = tmp_path / (doc_path.stem + ".docx")

            shutil.copy2(doc_path, tmp_doc)

            # Step 1: .doc -> .docx via LibreOffice
            lo_profile = setup_libreoffice_profile(worker_id)
            lo_cmd = [
                "soffice",
                "--headless",
                f"-env:UserInstallation=file://{lo_profile}",
                "--norestore",
                "--nologo",
                "--nolockcheck",
                "--convert-to", "docx",
                "--outdir", str(tmp_path),
                str(tmp_doc),
            ]

            try:
                lo_result = subprocess.run(
                    lo_cmd,
                    timeout=LIBREOFFICE_TIMEOUT,
                    capture_output=True,
                    text=True,
                )

                if lo_result.returncode != 0 or not tmp_docx.exists():
                    if attempt < MAX_RETRIES:
                        continue
                    result["status"] = "libreoffice_failed"
                    result["error"] = f"stderr: {(lo_result.stderr or '')[:200]} stdout: {(lo_result.stdout or '')[:200]}"
                    return result

            except subprocess.TimeoutExpired:
                subprocess.run(["pkill", "-f", f"lo_profile_worker_{worker_id}"],
                               capture_output=True)
                if attempt < MAX_RETRIES:
                    continue
                result["status"] = "libreoffice_timeout"
                result["error"] = f"Exceeded {LIBREOFFICE_TIMEOUT}s after {MAX_RETRIES} attempts"
                return result

            # Step 2: .docx -> .md via pandoc
            tmp_md = tmp_path / "output.md"
            tmp_media = tmp_path / "media_extract"
            pandoc_cmd = [
                "pandoc",
                str(tmp_docx),
                "-f", "docx+styles",
                "-t", "markdown",
                "--wrap=none",
                "--extract-media", str(tmp_media),
                "--lua-filter", str(PANDOC_FILTER),
                "--columns=120",
                "-o", str(tmp_md),
            ]

            try:
                p_result = subprocess.run(
                    pandoc_cmd,
                    timeout=PANDOC_TIMEOUT,
                    capture_output=True,
                    text=True,
                )

                if p_result.returncode == 0 and tmp_md.exists() and tmp_md.stat().st_size > 0:
                    # Copy media files if pandoc extracted any
                    media_src = tmp_media / "media"
                    if media_src.exists():
                        media_dst = output_md_path.parent / "media"
                        media_dst.mkdir(parents=True, exist_ok=True)
                        # Copy individual files (don't replace entire media dir —
                        # other .md files in the same dir may share it)
                        for img in media_src.iterdir():
                            shutil.copy2(img, media_dst / img.name)

                        # Rewrite image paths: pandoc writes absolute tmp paths
                        # like /tmp/xxx/media_extract/media/image1.png
                        # We want: media/image1.png
                        md_text = tmp_md.read_text(encoding="utf-8")
                        md_text = re.sub(
                            r"!\[([^\]]*)\]\([^)]*media_extract/media/([^)]+)\)",
                            r"![\1](media/\2)",
                            md_text,
                        )
                        # Also strip width/height attributes from images
                        md_text = re.sub(
                            r"(!\[[^\]]*\]\([^)]+\))\{[^}]*\}",
                            r"\1",
                            md_text,
                        )
                        tmp_md.write_text(md_text, encoding="utf-8")

                    shutil.copy2(tmp_md, output_md_path)
                    result["status"] = "success"
                    result["size_bytes"] = output_md_path.stat().st_size
                    return result
                else:
                    if attempt < MAX_RETRIES:
                        continue
                    result["status"] = "pandoc_failed"
                    result["error"] = (p_result.stderr or "")[:500]
                    return result

            except subprocess.TimeoutExpired:
                if attempt < MAX_RETRIES:
                    continue
                result["status"] = "pandoc_timeout"
                result["error"] = f"Exceeded {PANDOC_TIMEOUT}s"
                return result

    result["status"] = "max_retries_exceeded"
    return result


def get_output_path(doc_path: Path) -> Path:
    """
    Map input .doc path to output .md path.
    Input:  dof_word/2025/01/02012025/MAT/001_DOF_20250102_MAT_5746544.doc
    Output: dof_md/2025/01/02012025/MAT/001_DOF_20250102_MAT_5746544.md
    """
    try:
        rel = doc_path.relative_to(INPUT_DIR)
    except ValueError:
        rel = Path(doc_path.name)
    return OUTPUT_DIR / rel.with_suffix(".md")


def find_doc_files(years=None):
    """Find all .doc files, optionally filtered by years."""
    files = []
    if years:
        for year in years:
            year_dir = INPUT_DIR / str(year)
            if year_dir.exists():
                year_files = sorted(year_dir.rglob("*.doc"))
                files.extend(year_files)
                log.info(f"Year {year}: {len(year_files)} .doc files")
    else:
        files = sorted(INPUT_DIR.rglob("*.doc"))
        log.info(f"Total: {len(files)} .doc files")
    return files


def process_file(args):
    """Wrapper for ProcessPoolExecutor."""
    doc_path, output_path, worker_id = args
    return convert_single_doc(Path(doc_path), Path(output_path), worker_id)


def main():
    parser = argparse.ArgumentParser(description="Convert DOF .doc files to Markdown")
    parser.add_argument("--years", nargs="+", type=int, help="Years to process")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel LibreOffice workers (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just count files, don't convert")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry only files that previously failed")
    parser.add_argument("--input-dir", type=str, default="./dof_word",
                        help="Input directory with .doc files (default: ./dof_word)")
    parser.add_argument("--output-dir", type=str, default="./dof_md",
                        help="Output directory for .md files (default: ./dof_md)")
    args = parser.parse_args()

    global INPUT_DIR, OUTPUT_DIR, FAILED_DIR
    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    FAILED_DIR = OUTPUT_DIR.parent / "dof_failed"

    log.info("=" * 60)
    log.info("DOF .doc to Markdown converter")
    log.info(f"Input:  {INPUT_DIR}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info(f"Workers: {args.workers}")
    log.info(f"Max retries: {MAX_RETRIES}")
    if args.years:
        log.info(f"Years: {args.years}")
    else:
        log.info("Years: ALL")
    log.info("=" * 60)

    files = find_doc_files(args.years)
    if not files:
        log.warning("No .doc files found!")
        return

    output_paths = [get_output_path(f) for f in files]
    already_done = sum(1 for p in output_paths if p.exists() and p.stat().st_size > 0)
    todo = len(files) - already_done

    log.info(f"Total: {len(files)} | Already converted: {already_done} | To convert: {todo}")

    if args.dry_run:
        from collections import Counter
        year_counts = Counter()
        year_done = Counter()
        for f, o in zip(files, output_paths):
            year = f.relative_to(INPUT_DIR).parts[0]
            year_counts[year] += 1
            if o.exists() and o.stat().st_size > 0:
                year_done[year] += 1

        print(f"\n{'Year':<8} {'Total':>8} {'Done':>8} {'Pending':>8}")
        print("-" * 36)
        for year in sorted(year_counts.keys()):
            total = year_counts[year]
            done = year_done[year]
            print(f"{year:<8} {total:>8} {done:>8} {total - done:>8}")
        print("-" * 36)
        print(f"{'TOTAL':<8} {len(files):>8} {already_done:>8} {todo:>8}")
        return

    if todo == 0 and not args.retry_failed:
        log.info("All files already converted!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build work items
    if args.retry_failed:
        # Load failed list
        failed_list = FAILED_DIR / "failed_files.txt"
        if failed_list.exists():
            failed_paths = set()
            with open(failed_list) as f:
                for line in f:
                    failed_paths.add(line.strip())
            work_items = []
            for i, (doc_path, out_path) in enumerate(zip(files, output_paths)):
                if str(doc_path) in failed_paths or not out_path.exists() or out_path.stat().st_size == 0:
                    work_items.append((str(doc_path), str(out_path), i % args.workers))
            log.info(f"Retrying {len(work_items)} failed files")
        else:
            log.warning("No failed files list found")
            return
    else:
        work_items = []
        for i, (doc_path, out_path) in enumerate(zip(files, output_paths)):
            if not out_path.exists() or out_path.stat().st_size == 0:
                work_items.append((str(doc_path), str(out_path), i % args.workers))

    if not work_items:
        log.info("No files to convert!")
        return

    log.info(f"Starting conversion of {len(work_items)} files with {args.workers} workers...")

    start_time = time.time()
    success = 0
    failed = 0
    skipped = 0
    failed_paths = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, item): item for item in work_items}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status = result["status"]

            if status == "success":
                success += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failed_paths.append(result["doc_path"])
                log.warning(f"FAILED [{status}]: {result['doc_path']}")

            if i % 1000 == 0 or i == len(work_items):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(work_items) - i) / rate / 60 if rate > 0 else 0
                log.info(
                    f"Progress: {i}/{len(work_items)} "
                    f"(OK: {success}, Fail: {failed}, Skip: {skipped}) "
                    f"Rate: {rate:.1f}/s ETA: {eta:.0f}min"
                )

    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info(f"Conversion complete in {elapsed / 60:.1f} minutes")
    log.info(f"Success: {success} | Failed: {failed} | Skipped: {skipped}")
    log.info(f"Rate: {len(work_items) / elapsed:.1f} files/sec")
    log.info(f"Output: {OUTPUT_DIR}")

    # Save failed files list for retry
    if failed_paths:
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        failed_file = FAILED_DIR / "failed_files.txt"
        with open(failed_file, "w") as f:
            for p in failed_paths:
                f.write(p + "\n")
        log.info(f"Failed files saved to: {failed_file}")
    else:
        log.info("All files converted successfully!")

    # Cleanup temp profiles
    for wid in range(args.workers):
        profile = Path(tempfile.gettempdir()) / f"lo_profile_worker_{wid}"
        if profile.exists():
            shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
