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
from enum import Enum

import typer
from docxcompose.composer import Composer
from docx import Document
import subprocess
import tempfile
import shutil


CONVERSION_TIMEOUT_SECONDS = 90
UNIFIED_FILE_PATTERN = r'^\d{8}_(MAT|VES)$'


class ProcessStatus(Enum):
    """Enum for process status results"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NO_FILES = "no_files"


class ProblematicFilesManager:
    """
    Manages the state of problematic files that have reached timeout during conversion.
    Encapsulates global state to improve testability and maintainability.
    """
    
    def __init__(self):
        self._problematic_files: List[str] = []
    
    def add_problematic_file(self, file_path: str) -> None:
        """Add a file to the problematic files list."""
        self._problematic_files.append(file_path)
    
    def get_problematic_files(self) -> List[str]:
        """Get a copy of the problematic files list."""
        return self._problematic_files.copy()
    
    def clear_problematic_files(self) -> None:
        """Clear the problematic files list."""
        self._problematic_files.clear()
    
    def count(self) -> int:
        """Get the count of problematic files."""
        return len(self._problematic_files)


class ProcessManager:
    """
    Manages timeout and process-related operations for document conversion.
    Encapsulates process management logic for better organization and testability.
    """
    
    def __init__(self, timeout_seconds: int = CONVERSION_TIMEOUT_SECONDS):
        self.timeout_seconds = timeout_seconds
    
    def kill_libreoffice_processes(self) -> int:
        """
        Kills all active LibreOffice (soffice) processes to prevent orphaned processes
        from affecting subsequent conversions after timeouts.
        
        Returns:
            int: Number of processes killed (0 if no processes were found)
        """
        try:
            # Execute pkill to kill all soffice processes
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
                
        except Exception as e:
            logging.error(f"Unexpected error cleaning LibreOffice processes: {e}")
            return 0
    
    def run_conversion_with_timeout(self, cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
        """
        Runs a conversion command with timeout handling.
        
        Args:
            cmd: Command to execute
            cwd: Working directory for the command
            
        Returns:
            subprocess.CompletedProcess: Result of the command execution
            
        Raises:
            subprocess.TimeoutExpired: If the command exceeds the timeout
        """
        return subprocess.run(
            cmd,
            timeout=self.timeout_seconds,
            capture_output=True,
            text=True,
            cwd=cwd
        )


# Global instances
_problematic_files_manager = ProblematicFilesManager()
_process_manager = ProcessManager()

def convert_doc_to_docx(doc_path: Path, docx_path: Optional[Path] = None, 
                       problematic_files_manager: Optional[ProblematicFilesManager] = None,
                       process_manager: Optional[ProcessManager] = None) -> Optional[Path]:
    """
    Converts a DOC file to DOCX using LibreOffice in headless mode
    
    Args:
        doc_path: Path to the original DOC file
        docx_path: Destination path for the DOCX file (optional)
        problematic_files_manager: Manager for tracking problematic files (optional)
        process_manager: Manager for process operations (optional)
        
    Returns:
        Path to the converted DOCX file or None if conversion failed
    """
    if problematic_files_manager is None:
        problematic_files_manager = _problematic_files_manager
    if process_manager is None:
        process_manager = _process_manager
    
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
            
            logging.info(f"Starting conversion: {doc_path} -> {docx_path}")
            logging.debug(f"Command: {' '.join(cmd)}")
            
            try:
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
                    
                    logging.error(error_msg)
                    return None
                    
            except subprocess.TimeoutExpired:
                problematic_files_manager.add_problematic_file(str(doc_path))
                logging.warning(f"TIMEOUT: File {doc_path} exceeded {process_manager.timeout_seconds} seconds limit and will be marked as problematic")
                
                processes_killed = process_manager.kill_libreoffice_processes()
                if processes_killed > 0:
                    logging.warning(f"LibreOffice process cleanup completed after timeout of {doc_path.name}")
                
                return None
                
        except Exception as e:
            logging.error(f"Error during conversion of {doc_path}: {e}")
            return None

def get_problematic_files(problematic_files_manager: Optional[ProblematicFilesManager] = None) -> List[str]:
    """
    Returns the list of files that reached timeout during conversion
    
    Args:
        problematic_files_manager: Manager for tracking problematic files (optional)
        
    Returns:
        List of problematic file paths
    """
    if problematic_files_manager is None:
        problematic_files_manager = _problematic_files_manager
    return problematic_files_manager.get_problematic_files()

def clear_problematic_files(problematic_files_manager: Optional[ProblematicFilesManager] = None) -> None:
    """
    Clears the list of problematic files
    
    Args:
        problematic_files_manager: Manager for tracking problematic files (optional)
    """
    if problematic_files_manager is None:
        problematic_files_manager = _problematic_files_manager
    problematic_files_manager.clear_problematic_files()

def save_problematic_files_report(output_dir: Path, 
                                 problematic_files_manager: Optional[ProblematicFilesManager] = None) -> Optional[Path]:
    """
    Saves a report of problematic files to a file
    
    Args:
        output_dir: Directory where to save the report
        problematic_files_manager: Manager for tracking problematic files (optional)
        
    Returns:
        Path to the created report file or None if no problematic files
    """
    if problematic_files_manager is None:
        problematic_files_manager = _problematic_files_manager
    
    problematic_files = problematic_files_manager.get_problematic_files()
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
    
    existing_files = []
    for f in docx_files:
        path_obj = Path(f) if isinstance(f, str) else f
        if path_obj.exists():
            existing_files.append(path_obj)
    
    if not existing_files:
        logging.warning("No valid DOCX files found to merge.")
        return False
    
    output_path = Path(output_path) if isinstance(output_path, str) else output_path
    
    try:
        logging.info(f"Starting merge of {len(existing_files)} DOCX files using docxcompose...")
        
        master_doc = Document(existing_files[0])
        composer = Composer(master_doc)
        
        logging.info(f"Master document: {existing_files[0].name}")
        
        for i, docx_file in enumerate(existing_files[1:], 1):
            try:
                doc_to_append = Document(docx_file)
                composer.append(doc_to_append)
                
                logging.info(f"File {i}/{len(existing_files)-1} added: {docx_file.name}")
                
            except Exception as e:
                logging.error(f"Error adding {docx_file.name} to merged document: {e}")
                continue
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composer.save(output_path)
        
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
        unified_files_exist = any(directory.glob('*_MAT.docx')) or any(directory.glob('*_VES.docx'))
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                should_delete = False
                
                if file_path.suffix.lower() == '.doc':
                    if unified_files_exist:
                        should_delete = True
                
                elif file_path.suffix.lower() == '.docx':
                    filename = file_path.stem
                    unified_pattern = re.match(UNIFIED_FILE_PATTERN, filename)
                    
                    if not unified_pattern:
                        should_delete = True
                
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
    date_parts = date_str.split('/')
    day, month, year = date_parts[0], date_parts[1], date_parts[2]
    
    edition_dir = base_dir / year / month / f"{day}{month}{year}" / edition
    
    doc_files = []
    
    if edition_dir.exists():
        doc_files = list(edition_dir.glob('*.doc'))
        for doc_file in doc_files:
            logging.info(f"DOC file found: {doc_file}")
    else:
        logging.warning(f"Directory not found: {edition_dir}")
    
    return doc_files


def process_date_edition(base_dir: Path, date_str: str, edition: str, 
                        problematic_files_manager: Optional[ProblematicFilesManager] = None) -> ProcessStatus:
    """
    Processes a specific date and edition: converts DOC to DOCX and merges
    
    Args:
        base_dir: Base directory to search for files
        date_str: Date in DD/MM/YYYY format
        edition: Edition ('MAT' or 'VES')
        problematic_files_manager: Manager for tracking problematic files (optional)
        
    Returns:
        ProcessStatus: SUCCESS if processing was successful, FAILED if failed, 
        NO_FILES if no files to process, TIMEOUT if any document reached timeout
    """
    if problematic_files_manager is None:
        problematic_files_manager = _problematic_files_manager
    logging.info(f"Processing {date_str} - {edition}")
    
    # Search for DOC files
    doc_files = find_doc_files_in_edition_folders(base_dir, date_str, edition)
    
    if not doc_files:
        logging.info(f"No DOC files found for {date_str} - {edition}")
        return ProcessStatus.NO_FILES  
    
    logging.info(f"Found {len(doc_files)} DOC files")
    
    # Phase 1: Convert DOC files to DOCX
    docx_files = []
    initial_problematic_count = len(get_problematic_files(problematic_files_manager))
    
    for doc_file in doc_files:
        docx_file = convert_doc_to_docx(doc_file, problematic_files_manager=problematic_files_manager, process_manager=_process_manager)
        if docx_file and docx_file.exists():
            docx_files.append(docx_file)
            logging.info(f"Converted: {doc_file.name} -> {docx_file.name}")
        else:
            # Check if failure was due to timeout
            current_problematic_count = len(get_problematic_files(problematic_files_manager))
            if current_problematic_count > initial_problematic_count:
                logging.warning(f"TIMEOUT detected in {doc_file.name}. Skipping complete day: {date_str} - {edition}")
                return ProcessStatus.TIMEOUT
            logging.error(f"✗ Error converting: {doc_file.name}")
    
    if not docx_files:
        logging.error(f"Could not convert DOC files for {date_str} - {edition}")
        return ProcessStatus.FAILED
    
    # Phase 2: Merge DOCX documents
    date_parts = date_str.split('/')
    day, month, year = date_parts[0], date_parts[1], date_parts[2]
    
    unified_filename = f"{day}{month}{year}_{edition}.docx"
    edition_dir = base_dir / year / month / f"{day}{month}{year}" / edition
    unified_path = edition_dir / unified_filename
    
    logging.info(f"Merging {len(docx_files)} DOCX files...")
    
    if merge_docx_files(docx_files, unified_path):
        logging.info(f"Merged document created: {unified_path}")
        
        # Phase 3: Automatic cleanup of residual files
        logging.info("Starting cleanup of residual files...")
        cleanup_temp_files(edition_dir)
        
        return ProcessStatus.SUCCESS
    else:
        logging.error(f"✗ Error creating merged document for {date_str} - {edition}")
        return ProcessStatus.FAILED


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
    
    editions = editions.lower()
    if editions not in ['mat', 'ves', 'both']:
        logging.error("Editions must be 'mat', 'ves', or 'both'")
        sys.exit(1)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"Input directory not found: {input_path.absolute()}")
        sys.exit(1)
    
    logging.info("Starting DOC file conversion and merging")
    
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
        
        editions_to_process = []
        if editions in ['mat', 'both']:
            editions_to_process.append('MAT')
        if editions in ['ves', 'both']:
            editions_to_process.append('VES')
        
        for edition in editions_to_process:
            result = process_date_edition(input_path, date_str, edition, _problematic_files_manager)
            
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
    
    problematic_list = get_problematic_files(_problematic_files_manager)
    if problematic_list:
        logging.warning(f"Found {len(problematic_list)} problematic files that reached timeout ({CONVERSION_TIMEOUT_SECONDS}s)")
        
        save_problematic_files_report(input_path, _problematic_files_manager)
        
        logging.warning("Problematic files:")
        for i, file_path in enumerate(problematic_list, 1):
            logging.warning(f"  {i:3d}. {file_path}")
    else:
        logging.info("No problematic files found during processing")
    
    clear_problematic_files(_problematic_files_manager)


if __name__ == "__main__":
    typer.run(main)