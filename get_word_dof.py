#!/usr/bin/env python3
# /// script
# requires-python = ">=3.7"
# dependencies = [
#     "requests",
#     "typer",
#     "beautifulsoup4",
#     "urllib3",
# ]
# ///
"""
Script to download WORD files from the Official Gazette of the Federation (DOF)
Simplified version - Only downloads WORD files

"""

import re
import ssl
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests
import typer
import urllib3
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Constants
MIN_FILE_SIZE = 1024  # Minimum file size in bytes for validation


class TLSAdapter(HTTPAdapter):
    """Custom adapter to force TLS 1.2 or 1.3"""

    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context or ssl.create_default_context()
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self.ssl_context
        return super().init_poolmanager(*args, **kwargs)


def setup_session() -> requests.Session:
    """
    Configures a requests session with custom TLS adapter
    
    This function disables SSL certificate verification for compatibility
    with the DOF website.
    """
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False  # Disables hostname verification
    ssl_context.verify_mode = ssl.CERT_NONE  # Disables certificate verification
    ssl_context.set_ciphers("DEFAULT@SECLEVEL=1")
    
    session = requests.Session()
    session.mount('https://', TLSAdapter(ssl_context=ssl_context))
    session.verify = False  # Disables requests SSL verification - security risk
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session


def extract_word_links(html_content: str, base_url: str = 'https://www.dof.gob.mx') -> List[Tuple[str, str]]:
    """
    Extracts WORD file links from HTML content
    
    Args:
        html_content: HTML content of the page
        base_url: Base URL to build absolute links
        
    Returns:
        List of tuples (url_word, codnota)
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    word_links = []
    
    # Find all links pointing to nota_to_doc.php
    word_anchors = soup.find_all('a', href=re.compile(r'/nota_to_doc\.php\?codnota=\d+'))
    
    for anchor in word_anchors:
        href = anchor.get('href')
        if href:
            # Extract the note code
            match = re.search(r'codnota=(\d+)', href)
            if match:
                codnota = match.group(1)
                full_url = urljoin(base_url, href)
                word_links.append((full_url, codnota))
    
    return word_links


def download_word_file(session: requests.Session, url: str, output_path: Path) -> bool:
    """
    Downloads a WORD file from the specified URL
    
    Args:
        session: Configured requests session
        url: URL of the WORD file
        output_path: Path where to save the file
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        logging.info(f"Downloading: {url}")
        
        response = session.get(url, timeout=30)
        response.raise_for_status()
        
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        # Basic validation: check if file exists and has minimum size
        if output_path.exists() and output_path.stat().st_size >= MIN_FILE_SIZE:
            logging.info(f"Downloaded successfully: {output_path}")
            return True
        else:
            logging.warning(f"Invalid file (missing or too small), deleting: {output_path}")
            if output_path.exists():
                output_path.unlink()
            return False
            
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def process_dof_page(session: requests.Session, date_str: str, edition: str, output_dir: Path, sleep_delay: float = 1.0) -> int:
    """
    Processes a DOF page to download WORD files only
    
    Args:
        session: Configured requests session
        date_str: Date in DD/MM/YYYY format
        edition: Edition ('MAT' or 'VES')
        output_dir: Base directory to save files
        sleep_delay: Time to wait between downloads in seconds (default: 1.0)
        
    Returns:
        Number of files downloaded successfully
    """
    # Build DOF page URL
    date_parts = date_str.split('/')
    day, month, year = date_parts[0], date_parts[1], date_parts[2]
    
    edition_param = 'MAT' if edition == 'MAT' else 'VES'
    dof_url = f"https://www.dof.gob.mx/index.php?year={year}&month={month}&day={day}&edicion={edition_param}"
    
    try:
        logging.info(f"Processing page: {dof_url}")
        response = session.get(dof_url, timeout=30)
        response.raise_for_status()
        
        # Extract WORD file links
        word_links = extract_word_links(response.text)
        
        if not word_links:
            logging.info(f"No WORD files found for {date_str} - {edition}")
            return 0
        
        logging.info(f"Found {len(word_links)} WORD files")
        
        # Create directory for this date and edition
        # Structure: 2023 > 01 > 02012023 > MAT/VES
        base_date_dir = output_dir / year / month / f"{day}{month}{year}"
        date_dir = base_date_dir / edition
        date_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_count = 0
        
        # Download DOC files
        for word_url, codnota in word_links:
            # Generate filename
            filename = f"DOF_{year}{month}{day}_{edition}_{codnota}.doc"
            output_path = date_dir / filename
            
            # Skip if file already exists and has minimum size
            if output_path.exists() and output_path.stat().st_size >= MIN_FILE_SIZE:
                logging.warning(f"File already exists: {output_path}")
                continue
            
            # Download file
            if download_word_file(session, word_url, output_path):
                downloaded_count += 1
            
            # Brief pause between downloads to not overload the server
            time.sleep(sleep_delay)
        
        return downloaded_count
        
    except Exception as e:
        logging.error(f"Error processing page {dof_url}: {e}")
        return 0


def main(
    date: str = typer.Argument(..., help="Fecha (DD/MM/YYYY) o fecha de inicio para rango"),
    end_date: Optional[str] = typer.Argument(None, help="Fecha de fin (DD/MM/YYYY) - opcional para rango de fechas"),
    output_dir: str = typer.Option("./dof_word", help="Directorio de salida"),
    editions: str = typer.Option("both", help="Ediciones a descargar: 'mat', 'ves', o 'both'"),
    log_level: str = typer.Option("INFO", help="Nivel de logging: DEBUG, INFO, WARNING, ERROR"),
    sleep_delay: float = typer.Option(1.0, help="Tiempo de espera en segundos entre descargas")
):
    """
    Downloads WORD files from the Official Gazette of the Federation
    Simplified version - Only downloads WORD files
    
    Usage examples:
    # For a specific date (uses ./dof_word by default):
    python get_word_dof.py 02/01/2023 --editions both
    
    # For a date range:
    python get_word_dof.py 01/01/2023 31/01/2023 --editions both
    
    # Specifying custom directory:
    python get_word_dof.py 02/01/2023 --output-dir ./my_folder --editions both
    
    # Controlling download speed with sleep_delay:
    python get_word_dof.py 02/01/2023 --sleep-delay 0.5   # Fast downloads (0.5s between files)
    python get_word_dof.py 02/01/2023 --sleep-delay 2.0   # Slow downloads (2s between files)
    python get_word_dof.py 02/01/2023 --sleep-delay 0.1   # Very fast (0.1s - use carefully)
    
    # Complete example:
    python get_word_dof.py 01/01/2023 31/01/2023 --output-dir ./dof --editions both --sleep-delay 1.5
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
            logging.FileHandler('word_download.log'),
            logging.StreamHandler()
        ]
    )
    
    # Validate dates
    try:
        start_dt = datetime.strptime(date, "%d/%m/%Y")
        
        # If end_date is not provided, use the same date (single date)
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
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure session
    session = setup_session()
    
    logging.info(f"Starting DOF WORD files download")
    
    # Show appropriate period depending on whether it's a single date or range
    if end_date is None:
        logging.info(f"Date: {date}")
    else:
        logging.info(f"Period: {date} - {end_date}")
    
    logging.info(f"Editions: {editions}")
    logging.info(f"Output directory: {output_path.absolute()}")
    logging.info("-" * 60)
    
    total_downloaded = 0
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
            downloaded = process_dof_page(session, date_str, edition, output_path, sleep_delay)
            total_downloaded += downloaded
        
        current_date += timedelta(days=1)
    
    logging.info("-" * 60)
    logging.info(f"Download completed. Total files downloaded: {total_downloaded}")


if __name__ == "__main__":
    typer.run(main)