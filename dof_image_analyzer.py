# /// script
# dependencies = [
#   "typer",
#   "google-genai",
#   "python-dotenv",
#   "pillow"
# ]
# ///

"""
DOF Image Analyzer with Gemini 2.5 Flash-Lite.

Generates image descriptions and inserts them as alt text
in the corresponding Markdown files within the DOF structure.

Help and examples: `uv run dof_image_analyzer.py --help`
"""

import sys
import time
import logging
import os
import typer
import re

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
from google import genai
from dotenv import load_dotenv
from PIL import Image

class ProcessResult(Enum):
    SUCCESS = "success"
    FAILED = "failed"


class RateLimiter:
    """
    Rate limiter manager for Google Gemini API
    
    By default respects the free tier limit of 15 requests per minute (RPM),
    waiting 4 seconds between requests to avoid quota errors.
    """

    def __init__(self, requests_per_minute: int = 15):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self.min_interval = 60.0 / requests_per_minute

    def wait_if_needed(self):
        """
        Wait the necessary time to respect API limits
        
        Implements two strategies:
        1. Per-minute limit: maximum 15 requests in 60-second window
        2. Minimum interval: 4 seconds between consecutive requests
        """
        current_time = time.time()

        self.request_times = [t for t in self.request_times if current_time - t < 60]

        if len(self.request_times) >= self.requests_per_minute:
            oldest_request = min(self.request_times)
            wait_time = 60 - (current_time - oldest_request) + 1
            if wait_time > 0:
                logging.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        if self.request_times:
            time_since_last = current_time - max(self.request_times)
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logging.debug(f"Waiting {wait_time:.1f}s for minimum interval...")
                time.sleep(wait_time)

        self.request_times.append(time.time())

def find_markdown_file(image_path: Path) -> Optional[Path]:
    """
    Find the corresponding Markdown file for an image
    
    Expected structure:
    - Image: /year/month/date/edition/media_temp/media/img_001.png
    - Markdown: /year/month/date/edition/DDMMYYYY_EDITION.md

    """
    edition_dir = image_path.parent.parent.parent

    for md_file in edition_dir.glob('*.md'):
        return md_file

    return None


def needs_description(md_file: Path, image_name: str) -> bool:
    """
    Check if the Markdown has at least one reference to the image
    with empty alt text (meaning a description is needed).

    Returns True only when an image reference exists and its alt text is empty.
    Returns False when the image is not referenced or alt text is already present.
    """
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = rf'!\[([^\]]*)\]\([^)]*{re.escape(image_name)}[^)]*\)'
        matches = list(re.finditer(pattern, content))

        if not matches:
            logging.debug(f"Image reference for {image_name} not found in {md_file}")
            return False

        # Check only first occurrence to avoid redundancy in embeddings
        first_match = matches[0]
        first_alt = first_match.group(1)
        
        return not first_alt.strip()

    except Exception as e:
        logging.error(f"Error checking alt text in {md_file}: {e}")
        return False


def insert_description_in_markdown(md_file: Path, image_name: str, description: str) -> bool:
    """
    Insert the image description as alt text in the Markdown reference
    
    Searches for patterns like ![](path/image.png) and converts them to ![description](path/image.png).
    
    Args:
        md_file: Markdown file where to insert the description
        image_name: Name of the image file to search for
        description: AI-generated description to insert        
    """
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = rf'!\[([^\]]*)\]\([^)]*{re.escape(image_name)}[^)]*\)'

        match = re.search(pattern, content)
        if not match:
            logging.debug(f"Image reference for {image_name} not found in {md_file}")
            return False

        full_match = match.group(0)
        current_alt = match.group(1)

        if not current_alt.strip():
            new_image_ref = full_match.replace(f'![{current_alt}]', f'![{description}]')
            new_content = content.replace(full_match, new_image_ref, 1)

            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return True
        else:
            logging.debug(f"First occurrence of {image_name} already has alt text in {md_file}")
            return False

    except Exception as e:
        logging.error(f"Error inserting description in {md_file}: {e}")
        return False

class ImageAnalyzer:
    """
    Main image analyzer using Google Gemini 2.5 Flash-Lite
    
    Handles communication with Google's API, respects rate limits, and coordinates image analysis with insertion into Markdown files.
    """

    def __init__(self, api_key: str, no_limits: bool = False):
        self.client = genai.Client(api_key=api_key)
        # If no_limits is True, use a very high RPM (1000) to effectively disable rate limiting
        rpm = 1000 if no_limits else 15
        self.rate_limiter = RateLimiter(requests_per_minute=rpm)
        if no_limits:
            logging.warning("NO LIMITS MODE ACTIVE - Rate limiting disabled. Use with caution!")

    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze a single image and return description and metadata

        Args:
            image_path: Path to the image file

        """
        try:
            # Find corresponding Markdown file first and bail early if missing
            md_file = find_markdown_file(image_path)
            if not md_file:
                logging.warning(f"Markdown file not found for image: {image_path}")
                return {"status": ProcessResult.FAILED.value, "error": "Markdown file not found"}

            # Check if description is needed (alt text missing). If not, skip API call
            if not needs_description(md_file, image_path.name):
                logging.info(f"Skipping analysis for {image_path.name} in {md_file.name}: alt text already present or reference missing")
                return {"status": ProcessResult.SUCCESS.value, "image_name": image_path.name}

            self.rate_limiter.wait_if_needed()

            with Image.open(image_path) as image:
                prompt = """Eres un experto en documentos del Diario Oficial de la Federación (DOF). Analiza la imagen y sigue estas instrucciones:

                    **IMPORTANTE - DETECCIÓN DE IMÁGENES VACÍAS O DECORATIVAS:**
                    Si la imagen está completamente vacía, en blanco, es solo un punto, responde ÚNICAMENTE con esta frase corta:
                    "Elemento decorativo sin contenido informativo."
                    
                    NO generes la estructura completa si la imagen no tiene contenido útil.

                    ---

                    **DETECCIÓN DE FÓRMULAS MATEMÁTICAS:**
                    Si la imagen contiene ÚNICAMENTE una o varias fórmulas matemáticas (ecuaciones, expresiones algebraicas, cálculos), responde con este formato conciso:
                    "Fórmula: [reproduce la fórmula exactamente como aparece, usando notación matemática estándar con *, /, ^, subíndices y superíndices]"
                    
                    Ejemplo: "Fórmula: FCpⱼ = 0.3 * FCⱼ + 0.7 * FCⱼ²"
                    
                    NO uses la estructura completa para fórmulas solas.

                    ---

                    **SOLO SI LA IMAGEN CONTIENE INFORMACIÓN ÚTIL **, genera una descripción estructurada en español (máx. 800 tokens) con el siguiente formato:
                    
                    **TIPO DE CONTENIDO:** (Tabla | Gráfica | Mapa | Diagrama | Logo | Formato | Esquema | Lista normativa | Otro)

                    **ENTIDADES CLAVE:** [Dependencias, secretarías, organismos, fechas exactas, folios, códigos o identificadores visibles]

                    **CONTENIDO PRINCIPAL:**  
                    - Si el texto o la tabla son legibles, **reprodúcelos literalmente** respetando su orden y estructura.  
                    - Si la imagen contiene **tablas numéricas extensas, coordenadas o listados repetitivos sin texto explicativo**, escribe una **descripción breve y factual de su contenido y finalidad** (por ejemplo: “Tabla con coordenadas geográficas por municipio”).  
                    - Si contiene tanto texto como tabla, incluye **solo la parte legible y relevante para contexto administrativo o legal**.  
                    - Usa tablas Markdown solo si la estructura es visible.

                    **DATOS ESPECÍFICOS:** [Cifras, fechas, porcentajes, ubicaciones o conceptos cuantitativos que aparezcan explícitamente]

                    **CONTEXTO LEGAL/ADMINISTRATIVO:** [Textos legales, artículos, referencias normativas o reglamentos visibles]

                    Reglas:
                    - No inventes, interpretes ni completes información.  
                    - No digas “ilegible” ni “parece ser”.  
                    - No repitas ni reformules texto ya incluido.  
                    - No incluyas descripciones visuales (colores, logotipos, íconos).  
                    - Prioriza siempre el **texto literal** sobre la explicación.  
                    - Si el texto visible es poco útil o puramente numérico, **resume su propósito con lenguaje administrativo preciso.**
                    """

                response = self.client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[prompt, image]
                )

            description = response.text

            success = insert_description_in_markdown(md_file, image_path.name, description)

            if success:
                logging.info(f"Successfully analyzed and inserted description for {image_path.name} in {md_file.name}")
                return {"status": ProcessResult.SUCCESS.value, "image_name": image_path.name}
            else:
                return {"status": ProcessResult.FAILED.value, "error": "Failed to insert description"}

        except Exception as e:
            logging.error(f"Error analyzing image {image_path}: {e}")
            return {"status": ProcessResult.FAILED.value, "error": str(e), "image_name": image_path.name}


def find_image_files(input_dir: Path, date_str: Optional[str] = None,
                     end_date_str: Optional[str] = None) -> List[Path]:
    """
    Search for image files to analyze following DOF structure    
    
    Args:
        input_dir: Root directory (e.g.: ./dof_word_md)
        date_str: Specific date (DD/MM/YYYY) or None to process all
        end_date_str: End date for range (DD/MM/YYYY) or None for single date
    """
    image_files = []
    supported_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}

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

                for edition in ['MAT', 'VES']:
                    images_dir = input_dir / year / month / date_folder / edition / 'media_temp' / 'media'
                    if images_dir.exists():
                        for image_file in images_dir.iterdir():
                            if image_file.is_file() and image_file.suffix.lower() in supported_extensions:
                                image_files.append(image_file)

                current_date += timedelta(days=1)
        else:
            for image_file in input_dir.rglob('media_temp/media/*'):
                if image_file.is_file() and image_file.suffix.lower() in supported_extensions:
                    image_files.append(image_file)
        return sorted(image_files)

    except Exception as e:
        logging.error(f"Error searching for image files: {e}")
        return []

def main(
    date: Optional[str] = typer.Argument(None, help="Specific date (DD/MM/YYYY) or start date for range. Optional - if not specified processes all images"),
    end_date: Optional[str] = typer.Argument(None, help="End date for range processing (DD/MM/YYYY). Only use with start date"),
    input_dir: str = typer.Option("./dof_word_md", help="Input directory with extracted images (expected DOF structure)"),
    log_level: str = typer.Option("INFO", help="Logging level: DEBUG (very detailed), INFO (normal), WARNING (only warnings), ERROR (only errors)"),
    no_limits: bool = typer.Option(False, "--no-limits", help="DANGEROUS: Disables API speed limits. Faster but may exceed Google quotas")
):
    """
    DOF IMAGES ANALYZER WITH GEMINI 2.5 FLASH-LITE
    
    Processes images extracted from DOF documents and generates detailed descriptions
    using AI, inserting them directly into the corresponding Markdown files.
    
    WHAT DOES THIS SCRIPT DO?
    - Finds images in DOF directory structure
    - Analyzes each image with Gemini AI to extract text and content
    - Inserts descriptions as alt text in Markdown files
    - Respects API limits (4 seconds between requests by default)
    - Generates detailed process logs
    
    TYPICAL USE CASES:
    • Process images from ONE SPECIFIC DAY of DOF
    • Process images from ONE COMPLETE MONTH (date range)
    • Process ALL available images (no date)
    • Reprocess images with better speed (--no-limits mode)
    
    REQUIRED CONFIGURATION:
    1. .env file with: GEMINI_API_KEY=your_api_key_here
    2. Images already extracted by dof_docx_to_md.py
    3. Directory structure: year/month/date/edition/media_temp/media/
    
    EXPECTED PERFORMANCE:
    • With limits: ~15 images/minute (Google free tier)
    • Without limits: ~60+ images/minute (requires paid tier)
    
    WARNINGS:
    - --no-limits may exceed Google's free quotas
    - Script only processes images that DO NOT have alt text
    - Requires stable internet connection for API calls
    
    DETAILED EXAMPLES:
    
    # Process everything
    uv run dof_image_analyzer.py
    
    # Specific day:
    uv run dof_image_analyzer.py 15/03/2024
    
    # Date range (e.g.: whole week):
    uv run dof_image_analyzer.py 01/03/2024 07/03/2024
    
    # Fast but consumes more API (only with paid tier):
    uv run dof_image_analyzer.py 01/03/2024 31/03/2024 --no-limits
    """

    # Setup logging
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
            logging.FileHandler('dof_image_analyzer.log'),
            logging.StreamHandler()
        ]
    )

    logging.info("=== Starting DOF Image Analysis ===")

    input_path = Path(input_dir)

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
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logging.error("GEMINI_API_KEY not found in .env file")
            sys.exit(1)

        analyzer = ImageAnalyzer(api_key, no_limits=no_limits)

        image_files = find_image_files(input_path, date, end_date)

        if not image_files:
            logging.warning("No image files found to process")
            return

        logging.info(f"Found {len(image_files)} images to analyze")
        if no_limits:
            logging.info("No-limits active: processing will be faster than with rate limiting.")
        else:
            logging.info(f"Estimated processing time: {len(image_files) * 4 / 60:.1f} minutes (with rate limiting)")

        successful_count = 0
        failed_count = 0

        for i, image_file in enumerate(image_files, 1):
            logging.info(f"Processing image {i}/{len(image_files)}: {image_file.name}")

            result = analyzer.analyze_image(image_file)

            if result.get('status') == ProcessResult.SUCCESS.value:
                successful_count += 1
            else:
                failed_count += 1

        logging.info("=== Analysis Summary ===")
        logging.info(f"Total images processed: {len(image_files)}")
        logging.info(f"Successful analyses: {successful_count}")
        logging.info(f"Failed analyses: {failed_count}")
        logging.info("Descriptions have been inserted directly into the corresponding Markdown files.")

        if successful_count > 0:
            logging.info("Image analysis completed successfully")

    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    typer.run(main)