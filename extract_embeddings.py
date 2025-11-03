# %%
"""DOF Document Embedding Extraction with Structured Headers and Character-Based Chunking

This script processes markdown files from the Mexican Official Gazette (DOF),
extracts their content with advanced header structure recognition, splits them 
into intelligent character-based chunks (to respect model token limits) while
maintaining hierarchical context, and generates vector embeddings for storage.
The embeddings are stored in a DuckDB database for
later use by search and retrieval systems.

FEATURES:
- Structured header detection and hierarchical context maintenance
- Text cleaning and multi-space normalization
- Contextual headers for improved embedding quality
- Efficient embedding storage using DuckDB with FLOAT[] arrays
- Debug output files for manual inspection
- Optimized model sequence length (1024 tokens — enforced embedding input limit) for better performance


CHUNKING MANAGEMENT:
The script uses semantic-aware Markdown splitting (via `MarkdownSplitter`) rather than raw
token-counting. MarkdownSplitter preserves semantic structure (headings, code blocks,
lists and paragraphs) and produces chunks that respect document structure while keeping
the total characters per chunk within a configured capacity. The character capacity is
chosen empirically so the resulting chunks map safely below the model embedding input
limit (1024 tokens).

MarkdownSplitter splitting priority (applies in order):
1. Headings (#, ##, ###) — preserve section context
2. Code blocks (``` fenced code) — keep code intact
3. Lists (ordered/unordered) — keep list items together
4. Paragraph blocks (double newlines) — default fallback


DATABASE:
The system uses DuckDB for efficient embedding storage:
- Native support for FLOAT[] arrays for embeddings
- Unified database (dof_db/db.duckdb) for all data

Usage:
python extract_embeddings.py /path/to/markdown/files [--verbose]
"""

import os
import re
import torch
import logging
from datetime import datetime
from typing import Union, Tuple, Dict, List
from pathlib import Path

import typer
import duckdb
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from semantic_text_splitter import MarkdownSplitter 

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
IS_CPU_ONLY = not IS_CUDA_AVAILABLE and not IS_MPS_AVAILABLE

device = torch.device(
    "cuda" if IS_CUDA_AVAILABLE else
    ("mps" if IS_MPS_AVAILABLE else "cpu")
)

# Model configuration constants
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024
MODEL_MAX_SEQ_LENGTH = 1024  # Optimized from 32k default for better performance and memory usage

# Character management constants
CAPACITY = 2000      # capacity in characters per chunk (approx. ~500 tokens)
OVERLAP_CHARS = 500    # overlap in characters between chunks (25% of CAPACITY)

# PRE-COMPILED REGEX PATTERNS
UNDERSCORES_PATTERN = re.compile(r'(\\_){2,}')
PATTERN_UNIVERSAL = re.compile(r'^[\s\|\+\-:=]{10,}$', re.MULTILINE)
MULTIPLE_SPACES_PATTERN = re.compile(r' {3,}')
HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$')

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dof_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dof_embeddings")

# =======================
# TEXT CLEANING
# =======================

def clean_multiple_spaces(text: str) -> str:
    """
    Clean text before chunk processing.
    
    Operations applied once per file:
    1. Normalize long underscore patterns (≥2 characters)
    2. Simplify long table separators 
    3. Reduce multiple spaces (≥3) to single space
    4. Normalize excessive line breaks
    
    Args:
        text: Complete file text to clean
        
    Returns:
        Clean text ready for chunking
    """
    if not text:
        return text
    
    # 1. Normalize patterns of long underscores
    text = UNDERSCORES_PATTERN.sub(' __________ ', text)
    
    # 2. Simplify long table separators
    def count_columns_and_simplify_universal(match):
        """Count columns and create simplified separator for any type of separator"""
        line = match.group(0).strip()

        # Count column separators based on the type of separator
        if '+' in line:
            column_count = max(1, line.count('+') - 1)
        elif '|' in line:
            column_count = max(1, line.count('|') - 1)
        else:
            column_count = 1
        
        # Create simplified separator
        if column_count <= 1:
            return '|---|'
        else:
            return '|' + '---|' * column_count

    text = PATTERN_UNIVERSAL.sub(count_columns_and_simplify_universal, text)
    
    # 3. Reduce multiple spaces
    cleaned_text = MULTIPLE_SPACES_PATTERN.sub(' ', text)
    
    # 4. Normalize excessive line breaks
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text

# %%
logger.info(f"Loading model: {MODEL_NAME}")

model = SentenceTransformer(
    MODEL_NAME, 
    truncate_dim=EMBEDDING_DIM,
    model_kwargs={"device_map": device},
    trust_remote_code=True
)

# Configure sequence length limits for optimal performance
model.max_seq_length = MODEL_MAX_SEQ_LENGTH
if hasattr(model.tokenizer, 'model_max_length'):
    model.tokenizer.model_max_length = MODEL_MAX_SEQ_LENGTH
if hasattr(model[0], 'max_position_embeddings'):
    model[0].max_position_embeddings = MODEL_MAX_SEQ_LENGTH

# Optimize model for inference
model.to(device)
# Set model to evaluation mode to save memory
# Disables training-specific layers such as dropout and batch norm
# Reference: https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval
model.eval()

# Disable gradient computation to save memory
# Globally disables gradient computation and significantly reduces memory consumption
# Reference: https://discuss.pytorch.org/t/does-model-eval-with-torch-set-grad-enabled-is-train-have-the-same-effect-for-grad-history/17183
torch.set_grad_enabled(False)

logger.info(f"Model loaded successfully with max_seq_length: {model.max_seq_length}")
logger.info(f"Device configuration: device_map='{device}' + .to({device})")
logger.info(f"Performance flags: CUDA={IS_CUDA_AVAILABLE}, MPS={IS_MPS_AVAILABLE}, CPU_ONLY={IS_CPU_ONLY}")

# %%
# =============================================================================
# TEXT SPLITTER INITIALIZATION
# =============================================================================

logger.info(f"Initializing MarkdownSplitter with model's tokenizer. CAPACITY={CAPACITY}, OVERLAP_CHARS={OVERLAP_CHARS}")
try:
    text_splitter = MarkdownSplitter(
        CAPACITY,
        overlap=OVERLAP_CHARS,
        trim=True
    )
    logger.info("MarkdownSplitter initialized successfully for structured Markdown splitting.")
except Exception as e:
    logger.error(f"Failed to initialize MarkdownSplitter: {e}")
    logger.error("Ensure 'text-splitter' and 'tokenizers' (HuggingFace) are correctly installed.")
    exit(1)
    
# %%
# Database paths configuration
DB_FILE = "dof_db/db.duckdb"

# Ensure the database directory exists
db_dir = os.path.dirname(DB_FILE)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)
    logger.info(f"Ensuring database directory exists at: {db_dir}")

# Database initialization and schema setup
db = duckdb.connect(DB_FILE)

# Create sequences for auto-incrementing primary keys
db.execute("CREATE SEQUENCE IF NOT EXISTS documents_id_seq START 1")
db.execute("CREATE SEQUENCE IF NOT EXISTS chunks_id_seq START 1")

# Documents table: stores metadata about each document
db.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY DEFAULT nextval('documents_id_seq'),
        title VARCHAR,
        url VARCHAR UNIQUE,
        file_path VARCHAR,
        created_at TIMESTAMP
    )
""")

# Chunks table: stores document chunks with embeddings
db.execute(f"""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
        document_id INTEGER,
        text VARCHAR,
        header VARCHAR,
        chunk_number INTEGER,
        embedding FLOAT[{EMBEDDING_DIM}],
        created_at TIMESTAMP,
        FOREIGN KEY (document_id) REFERENCES documents(id)
    )
""")

def get_heading_level(line: str) -> Union[Tuple[int, str], Tuple[None, None]]:
    """
    Returns the heading level and text,
    or (None, None) if the line is not a heading.
    """
    match = HEADING_PATTERN.match(line)
    if match:
        hashes = match.group(1)
        heading_text = match.group(2).strip()
        level = len(hashes)
        return level, heading_text
    return None, None

def update_open_headings_dict(open_headings: Dict[int, str], line: str) -> Dict[int, str]:
    """
    Updates the dictionary of open headings according to the line.
    
    - If an H1 is found, all previous headings are removed.
    - If a heading of any level is found, all lower priority headings (higher numbers) 
      are removed and the current one is added or updated.
    - If no heading is found, the dictionary remains unchanged.
    
    Args:
        open_headings: Current dictionary of headings (level -> text)
        line: The line to check for headings
        
    Returns:
        Updated dictionary of headings
    """
    lvl, txt = get_heading_level(line)
    if lvl is None:
        # No heading in this line
        return open_headings
    
    if lvl == 1:
        # H1 resets all context
        logger.debug(f"H1 heading found: '{txt}'. Resetting heading context")
        return {1: txt}
    else:
        # Create copy to avoid modifying the original
        new_headings = {k: v for k, v in open_headings.items() if k < lvl}
        new_headings[lvl] = txt
        logger.debug(f"H{lvl} heading found: '{txt}'. Updating context")
        return new_headings

def build_header_dict(doc_title: str, open_headings: Dict[int, str]) -> str:
    """
    Builds the chunk header with the format:
    # Document: <Document Name>
    ## Heading Level 2
    ### Heading Level 3
    ...
    
    The headings are listed in order of level (H1, H2, H3, etc.)
    
    Args:
        doc_title: Title of the document
        open_headings: Dictionary of open headings (level -> text)
        
    Returns:
        Formatted header as a string
    """
    header_lines = [f"# Document: {doc_title}"]
    
    for level in sorted(open_headings.keys()):
        text = open_headings[level]
        header_lines.append(f"{'#' * level} {text}")
    
    return "\n".join(header_lines)

# %%

def get_url_from_filename(filename: str) -> str:
    """
    Generate the URL based on the filename pattern.
    
    Expected filename format: DDMMYYYY_XXX.md (e.g., 23012025_MAT.md, 23012025_VES.md)
    The function converts the underscore to hyphen for the PDF URL as required by DOF.
    
    Args:
        filename (str): Filename of the .md file

    Returns:
        str: The URL to the original PDF document

    Raises:
        ValueError: If the filename doesn't match the expected format
    """
    # Extract just the base filename in case the full path was passed
    base_filename = os.path.basename(filename).replace(".md", "")
    
    # Convert underscore to hyphen for PDF URL (DDMMYYYY_XXX -> DDMMYYYY-XXX)
    if '_' in base_filename:
        base_filename = base_filename.replace('_', '-')
    
    # Extract year and construct URL (similar to improved.py logic)
    if len(base_filename) >= 8:
        year = base_filename[4:8]  # Extract year (2025 from 23012025)
        pdf_filename = f"{base_filename}.pdf"
        url = f"https://diariooficial.gob.mx/abrirPDF.php?archivo={pdf_filename}&anio={year}&repo=repositorio/"
        return url
    else:
        raise ValueError(f"Expected filename like 23012025_MAT.md but got {filename}")

def _prepare_document_metadata(file_path: Union[str, Path]) -> tuple:
    """
    Extract metadata from the file.
    
    Args:
        file_path: Path to the markdown file (str or Path object)
        
    Returns:
        tuple: (content, title, url)
    """
    # Convert to Path object for consistent handling
    path_obj = Path(file_path)
    
    try:
        with path_obj.open("r", encoding="utf-8") as file:
            content = file.read()
    except (OSError, UnicodeDecodeError) as e:
        logger.error(f"Error reading file {path_obj}: {e}")
        raise

    title = path_obj.stem  # Get filename without extension
    url = get_url_from_filename(str(path_obj.name))
    logger.debug(f"Processing file '{path_obj}' with title '{title}'")
    
    return content, title, url

def _setup_database_document(title: str, url: str, file_path: Union[str, Path]) -> dict:
    """
    Configure document in database, cleaning up existing chunks to avoid foreign key violations.
    
    Args:
        title (str): Document title
        url (str): Document URL  
        file_path: Path to the document file (str or Path object)
        
    Returns:
        dict: Document record with explicit type conversions
        
    Raises:
        ValueError: If any required parameter is None or empty
        Exception: If database operation fails
    """
    if not all([title, url, file_path]):
        raise ValueError("All parameters (title, url, file_path) must be non-empty")
    
    # Convert Path to string for database storage
    file_path_str = str(file_path)
    
    try:
        existing_doc_df = db.execute("SELECT id FROM documents WHERE url = ?", [url]).pl()
        
        if not existing_doc_df.is_empty():
            doc_id = int(existing_doc_df.row(0, named=True)['id'])
            db.execute("DELETE FROM chunks WHERE document_id = ?", [doc_id])
            logger.info('') # Ensure clean separation in logs
            logger.info(f"Cleaned up existing chunks for document: {title}")
            
            db.execute("""
                UPDATE documents 
                SET title = ?, file_path = ?, created_at = ?
                WHERE url = ?
            """, [title, file_path_str, datetime.now(), url])
            
            result_df = db.execute(
                "SELECT id, title, url, file_path, created_at FROM documents WHERE url = ?", 
                [url]
            ).pl()
            action = "updated"
        else:
            result_df = db.execute("""
                INSERT INTO documents (title, url, file_path, created_at) 
                VALUES (?, ?, ?, ?)
                RETURNING id, title, url, file_path, created_at
            """, [title, url, file_path_str, datetime.now()]).pl()
            action = "created"
        
        if result_df.is_empty():
            raise Exception("Failed to insert/update document record")
        
        doc_row = result_df.row(0, named=True)
        doc = {
            "id": int(doc_row['id']),  
            "title": str(doc_row['title']), 
            "url": str(doc_row['url']),
            "file_path": str(doc_row['file_path']),
            "created_at": doc_row['created_at']
        }
        
        logger.info(f"Document {action} successfully: {title} (ID: {doc['id']})")
        return doc
        
    except Exception as e:
        logger.error(f"Failed to configure document '{title}': {str(e)}")
        raise

def _prepare_document_chunks(content: str) -> List:
    """
    Divide content into semantic chunks using MarkdownSplitter and character
    capacity limits.

    The splitter will prioritize structural boundaries (in order):
      1) Headings (#, ##, ...)
      2) Fenced code blocks (```)
      3) Lists (ordered/unordered)
      4) Paragraphs (double newlines)

    Args:
        content (str): Document content

    Returns:
        list: List of document chunks with chunk numbers
    """
    # Clean the content first
    cleaned_content = clean_multiple_spaces(content)
    
    # Split using MarkdownSplitter for semantic structure retention
    try:
        chunk_texts = list(text_splitter.chunks(cleaned_content))
    except Exception as e:
        logger.error(f"Error during chunking with MarkdownSplitter: {e}")
        return [] # Return empty list on failure

    # Convert to chunk format with sequential numbering
    document_chunks = []
    for i, chunk_text in enumerate(chunk_texts, 1):
        # Ensure the chunk is not empty or whitespace-only
        if chunk_text and chunk_text.strip():
            document_chunks.append({"text": chunk_text, "chunk": str(i)})
    
    logger.debug(f"Document split into {len(document_chunks)} semantic, character-based chunks using MarkdownSplitter")
    return document_chunks

def _initialize_chunk_processing(file_path: Union[str, Path], verbose: bool) -> tuple: 
    """
    Initialize variables for chunk processing.
    
    Args:
        file_path: Path to the file (str or Path object)
        verbose (bool): If True, creates debug chunks file
        
    Returns:
        tuple: (open_headings_dict, chunks_file, chunks_file_path)
    """
    open_headings_dict = {}
    chunks_file = None
    chunks_file_path = None
    
    if verbose:
        # Create debug chunks file with robust path handling
        try:
            path_obj = Path(file_path)
            # Use string manipulation for consistency with Path object usage throughout the codebase
            base_name = path_obj.stem  # filename without extension
            parent_dir = path_obj.parent
            chunks_file_path = parent_dir / f"{base_name}_chunks.txt"
            chunks_file = chunks_file_path.open("w", encoding="utf-8")
        except (OSError, ValueError) as e:
            logger.warning(f"Could not create debug file for {file_path}: {e}")
            chunks_file = None
            chunks_file_path = None
    
    return open_headings_dict, chunks_file, chunks_file_path


def _generate_chunk_embedding(header: str, chunk_text: str, file_path: Union[str, Path], chunk_counter: int):
    """
    Generate embedding for a chunk.
    
    Args:
        header (str): Chunk header
        chunk_text (str): Chunk text
        file_path: Path to the file (for error logging, str or Path object)
        chunk_counter (int): Chunk counter (for error logging)
       
    Returns:
        numpy.ndarray: Embedding vector
    """
    text_for_embedding = f"{header}\n\n{chunk_text}"
    
    try:
        # Using torch.inference_mode() for better performance and lower autograd overhead
        # Recommended by PyTorch core devs for inference-only operations
        # Reference: https://discuss.pytorch.org/t/pytorch-torch-no-grad-vs-torch-inference-mode/134099/3
        with torch.inference_mode():
            embedding = model.encode(text_for_embedding, show_progress_bar=False)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding for chunk #{chunk_counter} of {file_path}: {str(e)}")
        raise

def _write_debug_chunk(chunks_file, chunk_counter: int, header: str, chunk_text: str):
    """
    Write chunk details to debug file.
    
    Args:
        chunks_file: File object for writing debug info
        chunk_counter (int): Chunk counter
        header (str): Chunk header
        chunk_text (str): Chunk text
    """
    if chunks_file:
          
        chunks_file.write(f"--- CHUNK #{chunk_counter} ---\n")
        chunks_file.write(f"Header:\n{header}\n\n")
        chunks_file.write(f"Text:\n{chunk_text}\n")
        chunks_file.write("\n" + "-"*50 + "\n\n")

def _save_chunk_to_database(doc_id: int, chunk_text: str, header: str, chunk_number: int, embedding):
    """
    Save chunk in database.
    
    Args:
        doc_id (int): Document ID
        chunk_text (str): Chunk text
        header (str): Chunk header
        chunk_number (int): Chunk number
        embedding: Chunk embedding (numpy array)
    """
    # Convert numpy array to list for DuckDB FLOAT[] type
    embedding_list = embedding.tolist()
    
    db.execute("""
        INSERT INTO chunks (document_id, text, header, chunk_number, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [doc_id, chunk_text, header, chunk_number, embedding_list, datetime.now()])

def _update_headers_state(chunk_text: str, open_headings_dict: dict) -> dict:
    """
    Update state of headers after processing chunk.
    
    Args:
        chunk_text (str): Chunk text
        open_headings_dict (dict): Dictionary of open headings
        
    Returns:
        dict: Updated open_headings_dict
    """
    lines = chunk_text.splitlines()
    for line in lines:
        open_headings_dict = update_open_headings_dict(open_headings_dict, line)
    return open_headings_dict

def _process_single_chunk(chunk: dict, chunk_counter: int, title: str, open_headings_dict: dict, doc_id: int, chunks_file, verbose: bool, file_path: str) -> dict:
    """
    Process a single chunk.
    
    Args:
        chunk (dict): Chunk data
        chunk_counter (int): Chunk counter
        title (str): Document title
        open_headings_dict (dict): Dictionary of open headings
        doc_id (int): Document ID
        chunks_file: File object for debug info
        verbose (bool): If True, writes debug info
        file_path: Path to the file (for error logging, str or Path object)
        
    Returns:
        dict: Updated open_headings_dict
    """
    chunk_text = chunk["text"]
    chunk_number = chunk["chunk"]

    # Build the header using the "open" headings at the beginning of the chunk.
    header = build_header_dict(title, open_headings_dict)

    # Generate embedding
    embedding = _generate_chunk_embedding(header, chunk_text, file_path, chunk_counter)

    if verbose:
        _write_debug_chunk(chunks_file, chunk_counter, header, chunk_text)

    # Save to database
    _save_chunk_to_database(doc_id, chunk_text, header, chunk_number, embedding)

    # Update headers state
    open_headings_dict = _update_headers_state(chunk_text, open_headings_dict)
    
    return open_headings_dict

# =============================================================================
# DIRECTORY PROCESSING AND FILE DISCOVERY FUNCTIONS
# =============================================================================

def find_markdown_files(directory: Path) -> List[Path]:
    """
    Find and sort all .md files in a directory and subdirectories.
    
    Args:
        directory: Path object of the directory to search
        
    Returns:
        List[Path]: List of sorted Path objects for each .md file found
    """
    md_files = []
    for md_file in directory.rglob("*.md"):
        if md_file.is_file():
            md_files.append(md_file)
    
    # Sort files by their full path for consistent, chronological processing
    # This ensures dates are processed in order: 01012025, 02012025, etc.
    md_files.sort()
    
    return md_files

def process_file(file_path: Union[str, Path], verbose: bool = False):
    """
    Process a markdown file with structured headers and semantic character-based chunking support.
    This function:
    1. Reads file content
    2. Extracts metadata from filename
    3. Deletes any previous version of this document
    4. Splits content using semantic Markdown splitting (MarkdownSplitter)
    5. Maintains hierarchical context with open headings
    6. Generates contextual headers for each chunk
    7. Generates vector embeddings for each chunk (including header)
    8. Stores chunks and embeddings in the database
    9. Creates a debug file for manual inspection (only if verbose=True)

    Args:
        file_path: Path to the markdown file to process (str or Path object)
        verbose (bool): If True, creates debug chunks file
    """
    
    # Convert to Path object for consistent handling
    path_obj = Path(file_path)
    
    try:
        # 1. Prepare metadata
        content, title, url = _prepare_document_metadata(path_obj)
        
        # 2. Setup document in database
        doc = _setup_database_document(title, url, path_obj)

        # 3. Initialize state variables BEFORE chunking and entering the loop.
        open_headings_dict, chunks_file, chunks_file_path = _initialize_chunk_processing(path_obj, verbose)

        # 4. Prepare the chunks (semantic markdown splitting)
        document_chunks = _prepare_document_chunks(content)


        # Track processing time
        start_time = datetime.now()
        
        try:
            # 5. Process each chunk with progress bar
            total_chunks = len(document_chunks)
            chunk_counter = 0
            
            with tqdm(total=total_chunks, desc=f"Processing {title}...", unit="chunk", position=1, leave=False) as pbar:
                for chunk in document_chunks:
                    chunk_counter += 1
                    open_headings_dict = _process_single_chunk(
                        chunk, chunk_counter, title, open_headings_dict, 
                        doc["id"], chunks_file, verbose, path_obj
                    )
                    
                    # Update progress bar
                    pbar.update(1)
        finally:
            if chunks_file:
                chunks_file.close()
        
        # Calculate processing time for file
        processing_time = datetime.now() - start_time
        
        # 6. Final logging
        logger.info(f"File '{title}': Completed {len(document_chunks)} chunks in {processing_time.total_seconds():.2f}s")
        if verbose and chunks_file_path:
            logger.info(f"Chunks file generated at: {chunks_file_path}")
        
    except Exception as e:
        logger.error(f"Error processing file {path_obj}: {str(e)}")
        raise

def process_directory(directory_path: Union[str, Path], verbose: bool = False):
    """
    Process all markdown files in a directory and its subdirectories using optimized file discovery.

    Args:
        directory_path: Path to the directory to process (str or Path object)
        verbose (bool): If True, creates debug chunks files
    """
    # Convert to Path object for consistent handling
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        logger.error(f"Directory does not exist: {dir_path}")
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")
    
    if not dir_path.is_dir():
        logger.error(f"Path is not a directory: {dir_path}")
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
    
    # Find and collect all .md files with error handling and sorting
    try:
        md_files = find_markdown_files(dir_path)
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not access directory {dir_path}: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error scanning directory {dir_path}: {e}")
        return
    
    logger.info(f"Found {len(md_files)} markdown files in {dir_path}")
    
    if not md_files:
        logger.warning(f"No markdown files found in {dir_path}")
        return
    
    # Show sample of files to be processed for verification
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Sample files to process:")
        for i, md_file in enumerate(md_files[:5], 1):
            relative_path = md_file.relative_to(dir_path)
            logger.debug(f"  {i}. {relative_path}")
        if len(md_files) > 5:
            logger.debug(f"  ... and {len(md_files) - 5} more files")
    
    for md_file in tqdm(md_files, desc=f"Processing {dir_path.name}", position=0, leave=True):
        try:
            process_file(md_file, verbose=verbose)
        except Exception as e:
            logger.error(f"Failed to process file {md_file}: {e}")
            continue  # Continue with next file instead of stopping entirely
            
    # Final summary
    logger.info(f"Processing completed for directory: {dir_path}")

def main(root_dir: str, verbose: bool = False):
    """
    Process all markdown files in a directory and its subdirectories.
    
    Args:
        root_dir (str): Root directory to search for markdown files
        verbose (bool, optional): If True, shows detailed debug messages. Default is False.
    """
    # Configure logging level based on verbose parameter
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode activated - showing detailed debug messages")
    else:
        logger.setLevel(logging.INFO)
        
    logger.info(f"Starting document processing in: {root_dir}")
    start_time = datetime.now()
    
    try:
        process_directory(root_dir, verbose=verbose)
        
        elapsed_time = datetime.now() - start_time
        logger.info(f"Processing completed in {elapsed_time}")
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        raise


if __name__ == "__main__":
    typer.run(main)