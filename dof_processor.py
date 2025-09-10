#!/usr/bin/env python3
# /// script
# dependencies = [
#   "typer",
#   "docxcompose",
#   "python-docx",
# ]
# ///
"""Script to convert DOC files to DOCX and merge them

This script searches for DOC files in MAT/VES folders, converts them to DOCX
using LibreOffice in headless mode and then merges them into a single document per edition.

"""

import re
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import typer
from docxcompose.composer import Composer
from docx import Document
import subprocess
import tempfile
import shutil


CONVERSION_TIMEOUT_SECONDS = 90
problematic_files = []

def kill_libreoffice_processes() -> int:
    """
    Kills all active LibreOffice (soffice) processes to prevent orphaned processes
    from affecting subsequent conversions after timeouts.
    
    Returns:
        int: Number of processes killed (0 if no processes were found)
    """
    try:
        # Execute pkill to kill all soffice processes
        result = subprocess.run(
            "pkill -f soffice",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logging.warning("Killing orphaned LibreOffice processes...")
            return 1
        else:
            return 0
            
    except Exception as e:
        logging.error(f"Unexpected error cleaning LibreOffice processes: {e}")
        return 0

def convert_doc_to_docx(doc_path: Path, docx_path: Optional[Path] = None) -> Optional[Path]:
    """
    Converts a DOC file to DOCX using LibreOffice in headless mode
    
    Args:
        doc_path: Path to the original DOC file
        docx_path: Destination path for the DOCX file (optional)
        
    Returns:
        Path to the converted DOCX file or None if conversion failed
    """
    global problematic_files
    
    def _read_conversion_log(log_file: Path, doc_name: str, log_level: str) -> None:
        """
        Helper function to read and log conversion log content
        
        Args:
            log_file: Path to the log file
            doc_name: Name of the document being converted
            log_level: Level for logging ('debug', 'error', 'warning')
        """
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
                if log_content.strip():
                    if log_level == 'debug':
                        logging.debug(f"Conversion log for {doc_name}:\n{log_content}")
                    elif log_level == 'error':
                        logging.error(f"Detailed error log for {doc_name}:\n{log_content}")
                    elif log_level == 'warning':
                        logging.warning(f"Partial log before timeout for {doc_name}:\n{log_content}")
    
    if not doc_path.exists():
        logging.error(f"DOC file not found: {doc_path}")
        return None
    
    if docx_path is None:
        docx_path = doc_path.with_suffix('.docx')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_doc = temp_path / doc_path.name
        temp_docx = temp_path / docx_path.name
        
        try:
            shutil.copy2(doc_path, temp_doc)
            
            cmd = [
                'soffice',
                '--headless',
                '--convert-to', 'docx',
                '--outdir', str(temp_path),
                str(temp_doc)
            ]
            
            # Create log file for this process
            log_file = temp_path / f"conversion_{doc_path.stem}.log"
            
            tee_cmd = f"({' '.join(cmd)}) 2>&1 | tee {log_file}"
            
            logging.info(f"Starting conversion: {doc_path} -> {docx_path}")
            logging.debug(f"Command: {tee_cmd}")
            
            try:
                result = subprocess.run(
                    tee_cmd,
                    shell=True,
                    timeout=CONVERSION_TIMEOUT_SECONDS,
                    capture_output=True,
                    text=True,
                    cwd=temp_path
                )
                
                if result.returncode == 0 and temp_docx.exists():
                    shutil.copy2(temp_docx, docx_path)
                    
                    _read_conversion_log(log_file, doc_path.name, 'debug')
                    
                    logging.info(f"Successful conversion: {doc_path} -> {docx_path}")
                    return docx_path
                else:
                    # Log the error
                    error_msg = f"LibreOffice failed to convert {doc_path}"
                    if result.stderr:
                        error_msg += f". Error: {result.stderr}"
                    if result.stdout:
                        error_msg += f". Output: {result.stdout}"
                    
                    logging.error(error_msg)
                    
                    _read_conversion_log(log_file, doc_path.name, 'error')
                    
                    return None
                    
            except subprocess.TimeoutExpired:
                problematic_files.append(str(doc_path))
                logging.warning(f"TIMEOUT: File {doc_path} exceeded {CONVERSION_TIMEOUT_SECONDS} seconds limit and will be marked as problematic")
                
                _read_conversion_log(log_file, doc_path.name, 'warning')
                
                processes_killed = kill_libreoffice_processes()
                if processes_killed > 0:
                    logging.warning(f"LibreOffice process cleanup completed after timeout of {doc_path.name}")
                
                return None
                
        except Exception as e:
            logging.error(f"Error during conversion of {doc_path}: {e}")
            return None

def get_problematic_files() -> List[str]:
    """
    Returns the list of files that reached timeout during conversion
    
    Returns:
        List of problematic file paths
    """
    return problematic_files.copy()

def clear_problematic_files() -> None:
    """
    Clears the list of problematic files
    """
    global problematic_files
    problematic_files.clear()

def save_problematic_files_report(output_dir: Path) -> Optional[Path]:
    """
    Saves a report of problematic files to a file
    
    Args:
        output_dir: Directory where to save the report
        
    Returns:
        Path to the created report file or None if no problematic files
    """
    global problematic_files
    
    if not problematic_files:
        return None
    
    report_path = output_dir / f"archivos_problematicos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Problematic Files Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total files that reached timeout ({CONVERSION_TIMEOUT_SECONDS}s): {len(problematic_files)}\n\n")
            
            for i, file_path in enumerate(problematic_files, 1):
                f.write(f"{i:3d}. {file_path}\n")
        
        logging.info(f"Problematic files report saved at: {report_path}")
        return report_path
        
    except Exception as e:
        logging.error(f"Error saving problematic files report: {e}")
        return None


def merge_docx_files(docx_files: List[Path], output_path: Path) -> bool:
    """
    Merges multiple DOCX files into a single document using docxcompose
    
    Args:
        docx_files: List of paths to DOCX files (can be strings or Path objects)
        output_path: Path to the merged output file (can be string or Path object)
        
    Returns:
        True if merge was successful, False otherwise
    """
    if not docx_files:
        logging.warning("No DOCX files to merge.")
        return False
    
    # Convert to Path objects and filter existing files
    existing_files = []
    for f in docx_files:
        path_obj = Path(f) if isinstance(f, str) else f
        if path_obj.exists():
            existing_files.append(path_obj)
    
    if not existing_files:
        logging.warning("No valid DOCX files found to merge.")
        return False
    
    # Convert output_path to Path object if string
    output_path = Path(output_path) if isinstance(output_path, str) else output_path
    
    try:
        logging.info(f"Starting merge of {len(existing_files)} DOCX files using docxcompose...")
        
        # Create master document with first file
        master_doc = Document(str(existing_files[0].absolute()))
        composer = Composer(master_doc)
        
        logging.info(f"Master document: {existing_files[0].name}")
        
        # Add remaining documents
        for i, docx_file in enumerate(existing_files[1:], 1):
            try:
                # Load document to append
                doc_to_append = Document(str(docx_file.absolute()))
                
                # Add document to composer
                composer.append(doc_to_append)
                
                logging.info(f"File {i}/{len(existing_files)-1} added: {docx_file.name}")
                
            except Exception as e:
                logging.error(f"Error adding {docx_file.name} to merged document: {e}")
                continue
        
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save merged document
        composer.save(str(output_path.absolute()))
        
        logging.info(f"Merged document created successfully: {output_path.name}")
        return True
        
    except Exception as e:
        logging.error(f"Error during document merge with docxcompose: {e}")
        return False


def cleanup_temp_files(directory: Path) -> int:
    """
    Cleans temporary and residual files from directory
    
    Args:
        directory: Directory to clean
        
    Returns:
        Number of files deleted
    """
    if not directory.exists():
        return 0
    
    deleted_count = 0
    
    try:
        # Check if unified files exist (do this once before the loop)
        unified_files_exist = any(directory.glob('*_MAT.docx')) or any(directory.glob('*_VES.docx'))
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                # Determine if file should be deleted
                should_delete = False
                
                # Delete original DOC files after conversion
                if file_path.suffix.lower() == '.doc':
                    # Check if unified file exists in directory
                    # Unified file has format: DDMMYYYY_EDITION.docx
                    if unified_files_exist:
                        # If at least one unified file exists, delete all DOC files
                        should_delete = True
                
                # Delete individual DOCX files if unified file exists
                elif file_path.suffix.lower() == '.docx':
                    # Check if it's an individual file (not unified)
                    # Unified file has format: DDMMYYYY_EDITION.docx
                    # Individual files have longer names or different patterns
                    filename = file_path.stem
                    # Pattern for unified file: 8 digits + _ + 3 letters (MAT/VES)
                    unified_pattern = re.match(r'^\d{8}_(MAT|VES)$', filename)
                    
                    # If it doesn't match unified file pattern, it's an individual file
                    if not unified_pattern:
                        should_delete = True
                
                # Delete Word temporary files
                elif file_path.name.startswith('~$') or file_path.suffix.lower() in ['.tmp', '.temp']:
                    should_delete = True
                
                if should_delete:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logging.info(f"File deleted: {file_path}")
                    except Exception as e:
                        logging.warning(f"Could not delete {file_path}: {e}")
        
        logging.info(f"Cleanup completed. {deleted_count} files deleted.")
        return deleted_count
        
    except Exception as e:
        logging.error(f"Error during cleanup of {directory}: {e}")
        return deleted_count


def find_doc_files_in_edition_folders(base_dir: Path, date_str: str, edition: str) -> List[Path]:
    """
    Searches for DOC files in MAT/VES folders for a specific date
    
    Args:
        base_dir: Base directory to search in
        date_str: Date in DD/MM/YYYY format
        edition: Edition ('MAT' or 'VES')
        
    Returns:
        List of paths to found DOC files
    """
    # Build expected path: base_dir/YYYY/MM/DDMMYYYY/EDITION/
    date_parts = date_str.split('/')
    day, month, year = date_parts[0], date_parts[1], date_parts[2]
    
    edition_dir = base_dir / year / month / f"{day}{month}{year}" / edition
    
    doc_files = []
    
    if edition_dir.exists():
        # Search for all .doc files in directory
        doc_files = list(edition_dir.glob('*.doc'))
        for doc_file in doc_files:
            logging.info(f"DOC file found: {doc_file}")
    else:
        logging.warning(f"Directory not found: {edition_dir}")
    
    return doc_files


def process_date_edition(base_dir: Path, date_str: str, edition: str) -> Optional[bool]:
    """
    Processes a specific date and edition: converts DOC to DOCX and merges
    
    Args:
        base_dir: Base directory to search for files
        date_str: Date in DD/MM/YYYY format
        edition: Edition ('MAT' or 'VES')
        
    Returns:
        True if processing was successful, False if failed, None if no files to process
        'TIMEOUT' if any document reached timeout (problematic day)
    """
    logging.info(f"Processing {date_str} - {edition}")
    
    # Search for DOC files
    doc_files = find_doc_files_in_edition_folders(base_dir, date_str, edition)
    
    if not doc_files:
        logging.info(f"No DOC files found for {date_str} - {edition}")
        return None  
    
    logging.info(f"Found {len(doc_files)} DOC files")
    
    # Phase 1: Convert DOC files to DOCX
    docx_files = []
    initial_problematic_count = len(get_problematic_files())
    
    for doc_file in doc_files:
        docx_file = convert_doc_to_docx(doc_file)
        if docx_file and docx_file.exists():
            docx_files.append(docx_file)
            logging.info(f"Converted: {doc_file.name} -> {docx_file.name}")
        else:
            # Check if failure was due to timeout
            current_problematic_count = len(get_problematic_files())
            if current_problematic_count > initial_problematic_count:
                logging.warning(f"TIMEOUT detected in {doc_file.name}. Skipping complete day: {date_str} - {edition}")
                return 'TIMEOUT'
            logging.error(f"✗ Error converting: {doc_file.name}")
    
    if not docx_files:
        logging.error(f"Could not convert DOC files for {date_str} - {edition}")
        return False
    
    # Phase 2: Merge DOCX documents
    date_parts = date_str.split('/')
    day, month, year = date_parts[0], date_parts[1], date_parts[2]
    
    # Create unified filename with format: DDMMYYYY_EDITION.docx
    unified_filename = f"{day}{month}{year}_{edition}.docx"
    edition_dir = base_dir / year / month / f"{day}{month}{year}" / edition
    unified_path = edition_dir / unified_filename
    
    logging.info(f"Merging {len(docx_files)} DOCX files...")
    
    if merge_docx_files(docx_files, unified_path):
        logging.info(f"Merged document created: {unified_path}")
        
        # Phase 3: Automatic cleanup of residual files
        logging.info("Starting cleanup of residual files...")
        deleted_count = cleanup_temp_files(edition_dir)
        
        return True
    else:
        logging.error(f"✗ Error creating merged document for {date_str} - {edition}")
        return False


def main(
    date: str = typer.Argument(..., help="Fecha (DD/MM/YYYY) o fecha de inicio para rango"),
    end_date: Optional[str] = typer.Argument(None, help="Fecha de fin (DD/MM/YYYY) - opcional para rango de fechas"),
    input_dir: str = typer.Option("./dof_word", help="Directorio de entrada donde buscar archivos DOC"),
    editions: str = typer.Option("both", help="Ediciones a procesar: 'mat', 'ves', o 'both'"),
    log_level: str = typer.Option("INFO", help="Nivel de logging: DEBUG, INFO, WARNING, ERROR")
):
    """
    Converts DOC files to DOCX and merges them using docxcompose
    
    This script searches for DOC files in MAT/VES folders, converts them to DOCX
    using LibreOffice in headless mode and then merges them into a single document per edition.
    
    Main features:
    - Conversion with 90-second timeout per file
    - Detailed logging with LibreOffice output capture
    - Automatic identification of problematic files
    - Generation of timeout file reports
    - Automatic merging and temporary file cleanup
    
    Usage examples:
    # For a specific date:
    uv run dof_processor.py 02/01/2023 --editions both
    
    # For a date range:
    uv run dof_processor.py 01/01/2023 31/01/2023 --editions both
    
    # Specifying custom directory:
    uv run dof_processor.py 02/01/2023 --input-dir ./my_folder --editions both
    """
    
    # Configure logging
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
            logging.FileHandler('dof_processor.log'),
            logging.StreamHandler()
        ]
    )
    
    # Validate dates
    try:
        start_dt = datetime.strptime(date, "%d/%m/%Y")
        
        # If end_date is not provided, use same date (single date)
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
    
    # Validate editions
    editions = editions.lower()
    if editions not in ['mat', 'ves', 'both']:
        logging.error("Editions must be 'mat', 'ves', or 'both'")
        sys.exit(1)
    
    # Create input directory
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"Input directory not found: {input_path.absolute()}")
        sys.exit(1)
    
    logging.info(f"Starting DOC file conversion and merging")
    
    # Show appropriate period based on single date or range
    if end_date is None:
        logging.info(f"Date: {date}")
    else:
        logging.info(f"Period: {date} - {end_date}")
    
    logging.info(f"Editions: {editions}")
    logging.info(f"Input directory: {input_path.absolute()}")
    logging.info("-" * 60)
    
    total_processed = 0
    successful_processed = 0
    problematic_days = []
    current_date = start_dt
    
    while current_date <= end_dt:
        date_str = current_date.strftime("%d/%m/%Y")
        
        # Determine which editions to process
        editions_to_process = []
        if editions in ['mat', 'both']:
            editions_to_process.append('MAT')
        if editions in ['ves', 'both']:
            editions_to_process.append('VES')
        
        for edition in editions_to_process:
            result = process_date_edition(input_path, date_str, edition)
            
            if result == 'TIMEOUT':
                # Mark only this edition as problematic, not the whole day
                problematic_days.append(f"{date_str} - {edition}")
                logging.warning(f"Edition marked as problematic due to timeout: {date_str} - {edition}")
                continue
                
            if result is not None:  # Only count if there are files to process
                total_processed += 1
                if result:  # True = successful, False = failed
                    successful_processed += 1
        
        current_date += timedelta(days=1)
    
    logging.info("-" * 60)
    logging.info(f"Processing completed.")
    logging.info(f"Total processed: {total_processed}")
    logging.info(f"Successful: {successful_processed}")
    logging.info(f"Failed: {total_processed - successful_processed}")
    
    # Report of problematic editions due to timeout
    if problematic_days:
        logging.warning(f"Found {len(problematic_days)} problematic editions due to timeout:")
        for i, day_edition in enumerate(problematic_days, 1):
            logging.warning(f"  {i:3d}. {day_edition}")
    else:
        logging.info("No problematic editions found due to timeout")
    
    # Generate problematic files report if any
    problematic_list = get_problematic_files()
    if problematic_list:
        logging.warning(f"Found {len(problematic_list)} problematic files that reached timeout ({CONVERSION_TIMEOUT_SECONDS}s)")
        
        # Save report in input directory
        report_path = save_problematic_files_report(input_path)
        
        # Show list in log
        logging.warning("Problematic files:")
        for i, file_path in enumerate(problematic_list, 1):
            logging.warning(f"  {i:3d}. {file_path}")
    else:
        logging.info("No problematic files found during processing")
    
    # Clear list for future executions
    clear_problematic_files()


if __name__ == "__main__":
    typer.run(main)