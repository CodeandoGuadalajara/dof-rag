#!/usr/bin/env python3
# /// script
# dependencies = [
#   "typer"
# ]
# ///
"""
DOF DOCX to Markdown Converter

Converts DOF (Diario Oficial de la Federación) DOCX files to Markdown format
using Pandoc with custom LUA filters.

"""

import sys
import shutil
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from enum import Enum

import typer


class ProcessResult(Enum):
    SUCCESS = "success"
    FAILED = "failed"


def convert_docx_to_markdown(docx_path: Path, output_path: Path, 
                           images_dir: Path, lua_filter_headers: Path) -> bool:
    """
    Converts a DOCX file to Markdown using Pandoc with optimized configuration
    
    Args:
        docx_path: Path to input DOCX file
        output_path: Path to output Markdown file
        images_dir: Directory where to extract images
        lua_filter_headers: Path to custom LUA filter for headers
        
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        pandoc_cmd = [
            'pandoc',
            str(docx_path),
            '-f', 'docx+styles',
            '-t', 'markdown',
            '--wrap=preserve',
            '--extract-media', str(images_dir),
            '--lua-filter', str(lua_filter_headers),
            '-o', str(output_path)
        ]
        
        logging.info(f"Executing: {' '.join(pandoc_cmd)}")
        
        file_size_mb = docx_path.stat().st_size / (1024 * 1024)
        timeout_seconds = min(600, max(300, int(file_size_mb * 30)))
        
        logging.info(f"File size {file_size_mb:.2f} MB, timeout: {timeout_seconds} seconds")
        result = subprocess.run(pandoc_cmd, capture_output=True, text=True, timeout=timeout_seconds)
        
        if result.returncode == 0:
            logging.info(f"Successful conversion: {docx_path.name} -> {output_path.name}")
            
            if output_path.exists() and output_path.stat().st_size > 0:
                return True
            else:
                logging.error(f"Output file empty or doesn't exist: {output_path}")
                return False
        else:
            logging.error(f"Pandoc error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout in conversion of {docx_path}")
        return False
    except Exception as e:
        logging.error(f"Error converting {docx_path}: {e}")
        return False


def organize_images(images_dir: Path, target_images_dir: Path) -> int:
    """
    Organizes extracted images in the target directory
    
    Args:
        images_dir: Temporary directory with images extracted by Pandoc
        target_images_dir: Target directory for images
        
    Returns:
        Number of organized images
    """
    organized_count = 0
    
    try:
        if not images_dir.exists():
            return 0
        
        target_images_dir.mkdir(parents=True, exist_ok=True)
        
        for image_file in images_dir.rglob('*'):
            if image_file.is_file() and image_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.emf', '.wmf']:
                target_name = f"img_{organized_count + 1:03d}{image_file.suffix}"
                target_path = target_images_dir / target_name
                
                shutil.copy2(image_file, target_path)
                organized_count += 1
                logging.info(f"Image organized: {image_file.name} -> {target_name}")
        
        if images_dir.exists():
            shutil.rmtree(images_dir, ignore_errors=True)
        
        return organized_count
        
    except Exception as e:
        logging.error(f"Error organizing images: {e}")
        return organized_count


def process_docx_file(docx_path: Path, output_base_dir: Path, lua_filter_headers: Path) -> ProcessResult:
    """
    Processes an individual DOCX file. Always overwrites existing files.
    
    Args:
        docx_path: Path to DOCX file
        output_base_dir: Base output directory
        lua_filter_headers: Path to LUA filter for headers
        
    """
    try:
        path_parts = docx_path.parts
        logging.debug(f"Path parts: {path_parts}")
        
        if len(path_parts) < 5:
            logging.error(f"Invalid path structure: {docx_path}")
            return False
        
        year = path_parts[-5]
        month = path_parts[-4]
        date_dir = path_parts[-3]
        edition = path_parts[-2]  # MAT or VES
        
        logging.info(f"Extracted - Year: {year}, Month: {month}, Date: {date_dir}, Edition: {edition}")
        
        output_dir = output_base_dir / year / month / date_dir / edition
        output_dir.mkdir(parents=True, exist_ok=True)
        
        md_filename = docx_path.stem + '.md'
        md_path = output_dir / md_filename
        
        images_dir = output_dir / 'images'
        temp_media_dir = output_dir / 'media_temp'
        
        logging.info(f"Processing: {docx_path} -> {md_path}")
        
        if convert_docx_to_markdown(docx_path, md_path, temp_media_dir, lua_filter_headers):
            images_count = organize_images(temp_media_dir, images_dir)
            if images_count > 0:
                logging.info(f"Extracted {images_count} images")
            
            return ProcessResult.SUCCESS
        else:
            return ProcessResult.FAILED
            
    except Exception as e:
        logging.error(f"Error processing {docx_path}: {e}")
        return ProcessResult.FAILED


def find_docx_files(input_dir: Path, date_str: Optional[str] = None, 
                   end_date_str: Optional[str] = None) -> List[Path]:
    """
    Finds DOCX files to process. Always processes both MAT and VES editions.
    
    Args:
        input_dir: Input directory
        date_str: Specific date (DD/MM/YYYY) or None for all
        end_date_str: End date for range or None
        
    Returns:
        List of paths to DOCX files
    """
    docx_files = []
    
    try:
        if date_str:
            start_date = datetime.strptime(date_str, "%d/%m/%Y")
            end_date = datetime.strptime(end_date_str, "%d/%m/%Y") if end_date_str else start_date
            
            current_date = start_date
            while current_date <= end_date:
                year = current_date.strftime("%Y")
                month = current_date.strftime("%m")
                day = current_date.strftime("%d")
                date_folder = f"{day}{month}{year}"
                
                editions_to_process = ['MAT', 'VES']  # Always process both editions
                
                for edition in editions_to_process:
                    edition_dir = input_dir / year / month / date_folder / edition
                    if edition_dir.exists():
                        for docx_file in edition_dir.glob('*.docx'):
                            docx_files.append(docx_file)
                
                current_date += timedelta(days=1)
        else:
            for docx_file in input_dir.rglob('*.docx'):
                docx_files.append(docx_file)
        
        logging.info(f"Found {len(docx_files)} DOCX files to process")
        return docx_files
        
    except Exception as e:
        logging.error(f"Error searching DOCX files: {e}")
        return []


def main(
    date: Optional[str] = typer.Argument(None, help="Date (DD/MM/YYYY) or start date for range - optional"),
    end_date: Optional[str] = typer.Argument(None, help="End date (DD/MM/YYYY) - optional for date range"),
    input_dir: str = typer.Option("./dof_docx", help="Input directory with DOCX files"),
    output_dir: str = typer.Option("./dof_word_md", help="Output directory for Markdown files"),
    log_level: str = typer.Option("INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")
):
    """
    Converts DOF DOCX files to Markdown format using Pandoc
    with custom LUA filters and image extraction.
    Always processes both MAT and VES editions and overwrites existing files.
    
    Usage examples:
    # Process all files:
    uv run dof_docx_to_md.py
    
    # For a specific date:
    uv run dof_docx_to_md.py 22/01/2025
    
    # For a date range:
    uv run dof_docx_to_md.py 01/01/2025 31/01/2025
    """
    
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    
    logging.basicConfig(
        level=log_levels.get(log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('convert_docx_to_md.log'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=== Starting DOCX to Markdown conversion ===")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logging.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)
    
    if date:
        try:
            datetime.strptime(date, "%d/%m/%Y")
            if end_date:
                datetime.strptime(end_date, "%d/%m/%Y")
        except ValueError:
            logging.error("Dates must be in DD/MM/YYYY format")
            sys.exit(1)
    
    try:
        lua_filter_headers = Path("./pandoc_filters/dof_headers.lua")
        
        if not lua_filter_headers.exists():
            logging.error(f"LUA headers filter not found: {lua_filter_headers}")
            logging.error("Please ensure the LUA filter exists before running the script")
            sys.exit(1)
        
        docx_files = find_docx_files(input_path, date, end_date)
        
        if not docx_files:
            logging.warning("No DOCX files found to process")
            return
        
        successful_conversions = 0
        failed_conversions = 0
        
        for docx_file in docx_files:
            logging.info(f"Processing {docx_file.name}...")
            
            result = process_docx_file(docx_file, output_path, lua_filter_headers)
            
            if result == ProcessResult.SUCCESS:
                successful_conversions += 1
            elif result == ProcessResult.FAILED:
                failed_conversions += 1
        
        logging.info("=== Conversion summary ===")
        logging.info(f"Files processed successfully: {successful_conversions}")
        logging.info(f"Files with errors: {failed_conversions}")
        logging.info(f"Total files: {len(docx_files)}")
        
        if successful_conversions > 0:
            logging.info(f"Markdown files generated in: {output_path}")
        
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)