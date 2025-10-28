#!/usr/bin/env python3
# /// script
# dependencies = [
#   "typer",
#   "docxcompose",
#   "python-docx",
# ]
# ///
"""DOC to DOCX converter with automatic edition discovery

This script automatically discovers and processes all DOC files in MAT/VES folders,
converts them to DOCX using LibreOffice in headless mode and merges them into
unified documents per edition.

"""

import logging
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from docx import Document
from docxcompose.composer import Composer


CONVERSION_TIMEOUT_SECONDS = 90
UNIFIED_FILE_PATTERN = r'^\d{8}_(MAT|VES)$'


class ProcessStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NO_FILES = "no_files"


class ProcessManager:
    """Manages timeout and process operations for document conversion"""
    
    def __init__(self, timeout_seconds: int = CONVERSION_TIMEOUT_SECONDS):
        self.timeout_seconds = timeout_seconds
    
    def kill_libreoffice_processes(self) -> int:
        """Kills all active LibreOffice processes to prevent orphaned processes"""
        result = subprocess.run(
            ["pkill", "-f", "soffice"],
            shell=False,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logging.warning("Killing orphaned LibreOffice processes...")
            return 1
        else:
            return 0
    
    def run_conversion_with_timeout(self, cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
        """Runs a conversion command with timeout handling"""
        return subprocess.run(
            cmd,
            timeout=self.timeout_seconds,
            capture_output=True,
            text=True,
            cwd=cwd
        )


def convert_doc_to_docx(doc_path: Path, docx_path: Path, 
                       process_manager: ProcessManager) -> Path:
    """
    Converts a DOC file to DOCX using LibreOffice in headless mode
    
    Args:
        doc_path: Path to the input DOC file
        docx_path: Path where the output DOCX file will be saved
        process_manager: ProcessManager instance for timeout handling
        
    Returns:
        Path to the created DOCX file
    """
    
    if not doc_path.exists():
        raise FileNotFoundError(f"DOC file not found: {doc_path}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_doc = temp_path / doc_path.name
        temp_docx = temp_path / docx_path.name
        
        shutil.copy2(doc_path, temp_doc)
        
        cmd = [
            'soffice',
            '--headless',
            '--convert-to', 'docx',
            '--outdir', str(temp_path),
            str(temp_doc)
        ]
        
        logging.info(f"Starting conversion: {doc_path} -> {docx_path}")
        logging.debug(f"Command: {' '.join(cmd)}")
        
        result = process_manager.run_conversion_with_timeout(cmd, temp_path)
        
        if result.returncode == 0 and temp_docx.exists():
            shutil.copy2(temp_docx, docx_path)
            
            if result.stdout:
                logging.debug(f"LibreOffice output for {doc_path.name}: {result.stdout}")
            
            logging.info(f"Successful conversion: {doc_path} -> {docx_path}")
            return docx_path
        else:
            error_msg = f"LibreOffice failed to convert {doc_path}"
            if result.stderr:
                error_msg += f". Error: {result.stderr}"
            if result.stdout:
                error_msg += f". Output: {result.stdout}"
            
            raise subprocess.CalledProcessError(result.returncode, cmd, error_msg)

def save_problematic_files_report(output_dir: Path, problematic_list: List[str]) -> Optional[Path]:
    """
    Saves a report of problematic files to a file
    
    """
    if not problematic_list:
        return None
    
    report_path = output_dir / f"archivos_problematicos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Problematic Files Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total files that reached timeout ({CONVERSION_TIMEOUT_SECONDS}s): {len(problematic_list)}\n\n")
        
        for i, file_path in enumerate(problematic_list, 1):
            f.write(f"{i:3d}. {file_path}\n")
    
    logging.info(f"Problematic files report saved at: {report_path}")
    return report_path


def merge_docx_files(docx_files: List[Path], output_path: Path) -> None:
    """
    Merges multiple DOCX files into a single document using docxcompose
    
    """
    if not docx_files:
        raise ValueError("No DOCX files to merge")
    
    existing_files = []
    for docx_file in docx_files:
        if docx_file.exists():
            existing_files.append(docx_file)
    
    if not existing_files:
        raise ValueError("No valid DOCX files found to merge")
    
    logging.info(f"Starting merge of {len(existing_files)} DOCX files using docxcompose...")
    
    master_doc = Document(existing_files[0])
    composer = Composer(master_doc)
    
    logging.info(f"Master document: {existing_files[0].name}")
    
    for i, docx_file in enumerate(existing_files[1:], 1):
        doc_to_append = Document(docx_file)
        composer.append(doc_to_append)
        logging.info(f"File {i}/{len(existing_files)-1} added: {docx_file.name}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composer.save(output_path)
    
    logging.info(f"Merged document created successfully: {output_path.name}")


def cleanup_temp_files(directory: Path) -> int:
    """
    Cleans temporary and residual files from directory after document processing
    
    LibreOffice conversion process may leave behind temporary files that consume disk space:
    - Files starting with '~$' (LibreOffice lock/temp files)
    - Files with .tmp, .temp extensions
    - Intermediate DOCX files that are not the final unified document
    
    """
    if not directory.exists():
        return 0
    
    deleted_count = 0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            should_delete = False
            
            if file_path.suffix.lower() == '.docx':
                filename = file_path.stem
                unified_pattern = re.match(UNIFIED_FILE_PATTERN, filename)
                
                if not unified_pattern:
                    should_delete = True
            
            elif file_path.name.startswith('~$') or file_path.suffix.lower() in ['.tmp', '.temp']:
                should_delete = True
            
            if should_delete:
                file_path.unlink()
                deleted_count += 1
                logging.info(f"Temporary file deleted: {file_path}")
    
    logging.info(f"Cleanup completed. {deleted_count} temporary files deleted.")
    return deleted_count


def parse_date_to_path_components(date_str: str) -> Tuple[str, str, str]:
    """
    Parses a date string and returns components for path construction
    
    Args:
        date_str: Date string in DD/MM/YYYY format
        
    Returns:
        Tuple of (month, year, formatted_date) where formatted_date is DDMMYYYY
    """
    day, month, year = date_str.split('/')
    return month, year, f"{day}{month}{year}"


def find_all_edition_folders(base_dir: Path, date_str: str) -> List[str]:
    """Finds all available edition folders (MAT/VES) for a specific date"""
    month, year, formatted_date = parse_date_to_path_components(date_str)
    
    date_dir = base_dir / year / month / formatted_date
    editions = []
    
    if date_dir.exists():
        for edition_dir in date_dir.iterdir():
            if edition_dir.is_dir() and edition_dir.name in ['MAT', 'VES']:
                # Check if the edition folder contains any DOC files
                if list(edition_dir.glob('*.doc')):
                    editions.append(edition_dir.name)
                    logging.info(f"Found edition folder with DOC files: {edition_dir}")
    
    return editions


def find_doc_files_in_edition_folders(base_dir: Path, date_str: str, edition: str) -> List[Path]:
    """Searches for DOC files in MAT/VES folders for a specific date"""
    month, year, formatted_date = parse_date_to_path_components(date_str)
    
    edition_dir = base_dir / year / month / formatted_date / edition
    
    doc_files = []
    
    if edition_dir.exists():
        doc_files = sorted(edition_dir.glob('*.doc'))
        for doc_file in doc_files:
            logging.info(f"DOC file found: {doc_file}")
    else:
        logging.warning(f"Directory not found: {edition_dir}")
    
    return doc_files


def create_docx_output_structure(base_output_dir: Path, date_str: str, edition: str) -> Path:
    """Creates the output directory structure for DOCX files, separate from DOC files"""
    month, year, formatted_date = parse_date_to_path_components(date_str)
    
    output_dir = base_output_dir / year / month / formatted_date / edition
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Created DOCX output directory: {output_dir}")
    return output_dir


def process_date_edition(base_dir: Path, date_str: str, edition: str, 
                         process_manager: ProcessManager) -> Tuple[ProcessStatus, List[str]]:
    """
    Processes a specific date and edition: converts DOC to DOCX and merges
    """
    logging.info(f"Processing {date_str} - {edition}")
    problematic_files = []
    
    doc_files = find_doc_files_in_edition_folders(base_dir, date_str, edition)
    
    if not doc_files:
        logging.info(f"No DOC files found for {date_str} - {edition}")
        return ProcessStatus.NO_FILES, problematic_files
    
    logging.info(f"Found {len(doc_files)} DOC files")
    
    output_dir = base_dir.parent / "dof_docx"
    docx_output_dir = create_docx_output_structure(output_dir, date_str, edition)
    
    # Phase 1: Convert DOC files to DOCX in separate directory
    docx_files = []
    
    for doc_file in doc_files:
        try:
            docx_filename = doc_file.stem + '.docx'
            docx_output_path = docx_output_dir / docx_filename
            
            docx_file = convert_doc_to_docx(doc_file, docx_output_path, process_manager=process_manager)
            docx_files.append(docx_file)
            logging.info(f"Converted: {doc_file.name} -> {docx_file.name}")
            
        except subprocess.TimeoutExpired:
            problematic_files.append(str(doc_file))
            logging.warning(f"TIMEOUT: File {doc_file} exceeded {process_manager.timeout_seconds} seconds limit and will be marked as problematic")
            
            processes_killed = process_manager.kill_libreoffice_processes()
            if processes_killed > 0:
                logging.warning(f"LibreOffice process cleanup completed after timeout of {doc_file.name}")
            
            logging.warning(f"TIMEOUT detected in {doc_file.name}. Skipping complete day: {date_str} - {edition}")
            return ProcessStatus.TIMEOUT, problematic_files
            
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            logging.error(f"Error converting {doc_file}: {e}")
            continue
    
    if not docx_files:
        logging.error(f"Could not convert any DOC files for {date_str} - {edition}")
        return ProcessStatus.FAILED, problematic_files
    
    # Phase 2: Merge DOCX documents in output directory
    formatted_date = date_str.replace('/', '')
    
    unified_filename = f"{formatted_date}_{edition}.docx"
    unified_path = docx_output_dir / unified_filename
    
    logging.info(f"Merging {len(docx_files)} DOCX files...")
    
    try:
        merge_docx_files(docx_files, unified_path)
        logging.info(f"Merged document created: {unified_path}")
        
        # Phase 3: Cleanup temporary files in output directory only
        logging.info("Starting cleanup of temporary files...")
        cleanup_temp_files(docx_output_dir)
        
        return ProcessStatus.SUCCESS, problematic_files
        
    except (ValueError, OSError, Exception) as e:
        logging.error(f"Error during document merge for {date_str} - {edition}: {e}")
        return ProcessStatus.FAILED, problematic_files


def main(
    date: str = typer.Argument(..., help="Fecha (DD/MM/YYYY) o fecha de inicio para rango"),
    end_date: Optional[str] = typer.Argument(None, help="Fecha de fin (DD/MM/YYYY) - opcional para rango de fechas"),
    input_dir: str = typer.Option("./dof_word", help="Directorio de entrada donde buscar archivos DOC (solo lectura)"),
    log_level: str = typer.Option("INFO", help="Nivel de logging: DEBUG, INFO, WARNING, ERROR")
):
    """
    Converts DOC files to DOCX and merges them using docxcompose
    
    This script automatically discovers and processes all DOC files in MAT/VES folders,
    converts them to DOCX using LibreOffice in headless mode and merges them into 
    unified documents per edition.
    
    Key architectural principles:
    - DOC files are treated as a read-only library (never modified)
    - DOCX files are created in a separate directory structure
    - Original files are preserved for future processing
    - Automatically processes all available editions (MAT/VES)
    
    Main features:
    - Automatic edition discovery (no need to specify MAT/VES)
    - Conversion with 90-second timeout per file
    - Detailed logging with LibreOffice output capture
    - Automatic identification of problematic files
    - Generation of timeout file reports
    - Automatic merging and temporary file cleanup
    - Separate directory structure for processed files
    
    Usage examples:
    # For a specific date:
    uv run dof_processor.py 02/01/2023
    
    # For a date range:
    uv run dof_processor.py 01/01/2023 31/01/2023 --input-dir ./docs
    
    """
    
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    
    script_name = Path(__file__).stem
    log_filename = f"{script_name}.log"
    
    logging.basicConfig(
        level=log_levels.get(log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    try:
        start_dt = datetime.strptime(date, "%d/%m/%Y")
        
        if end_date is None:
            end_dt = start_dt
        else:
            end_dt = datetime.strptime(end_date, "%d/%m/%Y")
            
    except ValueError:
        logging.error("Dates must be in DD/MM/YYYY format")
        sys.exit(1)
    
    if start_dt > end_dt:
        logging.error("Start date must be before end date")
        sys.exit(1)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"Input directory not found: {input_path.absolute()}")
        sys.exit(1)
    
    output_path = input_path.parent / "dof_docx"
    
    logging.info("Starting DOC file conversion and merging")
    
    if end_date is None:
        logging.info(f"Date: {date}")
    else:
        logging.info(f"Period: {date} - {end_date}")
    
    logging.info(f"Input directory (read-only): {input_path.absolute()}")
    logging.info(f"Output directory (DOCX): {output_path.absolute()}")
    logging.info("-" * 60)
    
    # Initialize ProcessManager for conversion operations
    process_manager = ProcessManager()
    
    total_processed = 0
    successful_processed = 0
    problematic_days = []
    all_problematic_files = []
    current_date = start_dt
    
    while current_date <= end_dt:
        date_str = current_date.strftime("%d/%m/%Y")
        
        # Automatically discover all available editions for this date
        available_editions = find_all_edition_folders(input_path, date_str)
        
        if not available_editions:
            logging.info(f"No edition folders found for {date_str}")
        
        for edition in available_editions:
            result, problematic_files = process_date_edition(input_path, date_str, edition, process_manager)
            
            # Collect problematic files from this edition
            all_problematic_files.extend(problematic_files)
            
            if result == ProcessStatus.TIMEOUT:
                problematic_days.append(f"{date_str} - {edition}")
                logging.warning(f"Edition marked as problematic due to timeout: {date_str} - {edition}")
                total_processed += 1
                continue
                
            if result != ProcessStatus.NO_FILES:
                total_processed += 1
                if result == ProcessStatus.SUCCESS:
                    successful_processed += 1
        
        current_date += timedelta(days=1)
    
    logging.info("-" * 60)
    logging.info("Processing completed.")
    logging.info(f"Total processed: {total_processed}")
    logging.info(f"Successful: {successful_processed}")
    logging.info(f"Failed: {total_processed - successful_processed}")
    
    if problematic_days:
        logging.warning(f"Found {len(problematic_days)} problematic editions due to timeout:")
        for i, day_edition in enumerate(problematic_days, 1):
            logging.warning(f"  {i:3d}. {day_edition}")
    else:
        logging.info("No problematic editions found due to timeout")
    
    if all_problematic_files:
        logging.warning(f"Found {len(all_problematic_files)} problematic files that reached timeout ({CONVERSION_TIMEOUT_SECONDS}s)")
        
        save_problematic_files_report(output_path, all_problematic_files)
        
        logging.warning("Problematic files:")
        for i, file_path in enumerate(all_problematic_files, 1):
            logging.warning(f"  {i:3d}. {file_path}")
    else:
        logging.info("No problematic files found during processing")


if __name__ == "__main__":
    typer.run(main)