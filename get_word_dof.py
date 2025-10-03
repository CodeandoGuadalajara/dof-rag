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
from typing import List, Optional
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
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    ssl_context.set_ciphers("DEFAULT@SECLEVEL=1")
    
    session = requests.Session()
    session.mount('https://', TLSAdapter(ssl_context=ssl_context))
    session.verify = False
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session


def extract_word_links(html_content: str, base_url: str = 'https://www.dof.gob.mx') -> List[tuple[str, str]]:
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
    
    word_anchors = soup.find_all('a', href=re.compile(r'/nota_to_doc\.php\?codnota=\d+'))
    
    for anchor in word_anchors:
        href = anchor.get('href')
        if href:
            match = re.search(r'codnota=(\d+)', href)
            if match:
                codnota = match.group(1)
                full_url = urljoin(base_url, href)
                word_links.append((full_url, codnota))
    
    return word_links


def extract_notice_links(html_content: str) -> List[tuple[str, str]]:
    """
    Extracts notice links from SIDOF HTML content for all AVISOS subsections
    Detects edition (MAT/VES) based on tab-pane container
    
    Args:
        html_content: HTML content of the SIDOF page
        
    Returns:
        List of tuples (note_id, edition) where edition is 'MAT' or 'VES'
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    notice_links = []
    
    # Find all spans with class "txt-traduct" that contain " AVISOS "
    avisos_spans = soup.find_all('span', class_='txt-traduct', string=re.compile(r'^\s*AVISOS\s*$'))
    
    for avisos_span in avisos_spans:
        # Detect edition by finding the tab-pane container
        tab_pane = avisos_span.find_parent('div', class_='tab-pane')
        if not tab_pane:
            continue
            
        tab_id = tab_pane.get('id')
        edition = None
        if tab_id == 'resp-tab2':  # Vespertina
            edition = 'VES'
        elif tab_id == 'resp-tab3':  # Matutina
            edition = 'MAT'
        
        if not edition:
            continue
        
        panel_heading = avisos_span.find_parent('div', class_='panel-heading')
        if not panel_heading:
            continue
        
        parent_panel = panel_heading.find_parent('div', class_='panel-default')
        if not parent_panel:
            continue
        
        collapse_divs = parent_panel.find_all('div', class_=re.compile(r'panel-collapse'))
        
        for collapse_div in collapse_divs:
            note_links = collapse_div.find_all('a', href=re.compile(r'/notas/\d+'))
            
            for link in note_links:
                href = link.get('href')
                if href:
                    match_id = re.search(r'/notas/(\d+)', href)
                    if match_id:
                        note_id = match_id.group(1)
                        notice_links.append((note_id, edition))
    
    return notice_links


def _download_file(session: requests.Session, url: str, output_path: Path, file_type: str = "file") -> bool:
    """
    Internal function to download a file from a URL
    
    Args:
        session: Configured requests session
        url: URL of the file to download
        output_path: Path where to save the file
        file_type: Type of file for logging purposes (default: "file")
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        logging.info(f"Downloading {file_type}: {url}")
        
        response = session.get(url, timeout=30)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
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
    return _download_file(session, url, output_path, file_type="WORD file")


def download_notice_file(session: requests.Session, note_id: str, output_path: Path) -> bool:
    """
    Downloads a notice file from SIDOF
    
    Args:
        session: Configured requests session
        note_id: Note ID from SIDOF
        output_path: Path where to save the file
        
    Returns:
        True if download was successful, False otherwise
    """
    url = f"https://sidof.segob.gob.mx/notas/getDoc/{note_id}"
    return _download_file(session, url, output_path, file_type="notice")


def process_sidof_notices(session: requests.Session, day: str, month: str, year: str, edition: str, output_dir: Path, sleep_delay: float = 1.0) -> int:
    """
    Processes SIDOF page to download AVISOS notices for specific edition
    
    Args:
        session: Configured requests session
        day: Day component (DD)
        month: Month component (MM)
        year: Year component (YYYY)
        edition: Edition ('MAT' or 'VES') to filter notices
        output_dir: Base directory to save files
        sleep_delay: Time to wait between downloads in seconds (default: 1.0)
        
    Returns:
        Number of notice files downloaded successfully
    """
    sidof_date = f"{day}-{month}-{year}"
    date_str = f"{day}/{month}/{year}"
    
    sidof_url = f"https://sidof.segob.gob.mx/welcome/{sidof_date}"
    
    try:
        logging.info(f"Processing SIDOF page: {sidof_url}")
        response = session.get(sidof_url, timeout=30)
        response.raise_for_status()
        
        notice_links = extract_notice_links(response.text)
        
        if not notice_links:
            logging.info(f"No notices found for {date_str} in SIDOF")
            return 0
        
        filtered_notices = [(nid, ed) for nid, ed in notice_links if ed == edition]
        
        logging.info(f"Found {len(notice_links)} total notices in SIDOF, {len(filtered_notices)} for {edition} edition")
        
        if not filtered_notices:
            logging.info(f"No notices found for {edition} edition")
            return 0
        
        base_date_dir = output_dir / year / month / f"{day}{month}{year}"
        date_dir = base_date_dir / edition
        date_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_count = 0
        
        for note_id, _ in filtered_notices:
            filename = f"AVISO_{year}{month}{day}_{edition}_{note_id}.doc"
            output_path = date_dir / filename
            
            if output_path.exists() and output_path.stat().st_size >= MIN_FILE_SIZE:
                logging.warning(f"File already exists: {output_path}")
                continue
            
            if download_notice_file(session, note_id, output_path):
                downloaded_count += 1
            
            time.sleep(sleep_delay)
        
        return downloaded_count
        
    except Exception as e:
        logging.error(f"Error processing SIDOF page {sidof_url}: {e}")
        return 0


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
    date_parts = date_str.split('/')
    day, month, year = date_parts[0], date_parts[1], date_parts[2]
    
    dof_url = f"https://www.dof.gob.mx/index.php?year={year}&month={month}&day={day}&edicion={edition}"
    
    try:
        logging.info(f"Processing page: {dof_url}")
        response = session.get(dof_url, timeout=30)
        response.raise_for_status()
        
        word_links = extract_word_links(response.text)
        
        if not word_links:
            logging.info(f"No WORD files found for {date_str} - {edition}")
            return 0
        
        logging.info(f"Found {len(word_links)} WORD files")
        
        base_date_dir = output_dir / year / month / f"{day}{month}{year}"
        date_dir = base_date_dir / edition
        date_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_count = 0
        
        for word_url, codnota in word_links:
            filename = f"DOF_{year}{month}{day}_{edition}_{codnota}.doc"
            output_path = date_dir / filename
            
            if output_path.exists() and output_path.stat().st_size >= MIN_FILE_SIZE:
                logging.warning(f"File already exists: {output_path}")
                continue
            
            if download_word_file(session, word_url, output_path):
                downloaded_count += 1
            
            time.sleep(sleep_delay)
        
        logging.info(f"Now processing SIDOF notices for {date_str} - {edition}")
        notices_downloaded = process_sidof_notices(session, day, month, year, edition, output_dir, sleep_delay)
        downloaded_count += notices_downloaded
        
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
    Includes regular WORD files from dof.gob.mx and AVISOS notices from sidof.segob.gob.mx
    
    Usage examples:
    # For a specific date (uses ./dof_word by default):
    python get_word_dof_new.py 02/01/2023 --editions both
    
    # For a date range:
    python get_word_dof_new.py 01/01/2023 31/01/2023 --editions both
    
    # Specifying custom directory:
    python get_word_dof_new.py 02/01/2023 --output-dir ./my_folder --editions both
    
    # Controlling download speed with sleep_delay:
    python get_word_dof_new.py 02/01/2023 --sleep-delay 0.5   # Fast downloads (0.5s between files)
    python get_word_dof_new.py 02/01/2023 --sleep-delay 2.0   # Slow downloads (2s between files)
    python get_word_dof_new.py 02/01/2023 --sleep-delay 0.1   # Very fast (0.1s - use carefully)
    
    # Complete example:
    python get_word_dof_new.py 01/01/2023 31/01/2023 --output-dir ./dof --editions both --sleep-delay 1.5
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
            logging.FileHandler('word_download.log'),
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
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    session = setup_session()
    
    logging.info("Starting DOF WORD files download")
    
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