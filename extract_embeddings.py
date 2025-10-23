# %%
"""DOF Document Embedding Extraction with Structured Headers and Token-Based Processing

This script processes markdown files from the Mexican Official Gazette (DOF),
extracts their content with advanced header structure recognition, splits them 
into intelligent token-based chunks maintaining hierarchical context, and generates 
vector embeddings for storage. The embeddings are stored in a DuckDB database for
later use by search and retrieval systems.

FEATURES:
- Structured header detection and hierarchical context maintenance
- Token-based chunking with ultra-fast token estimation
- Text cleaning and multi-space normalization
- Contextual headers for improved embedding quality
- Efficient embedding storage using DuckDB with FLOAT[] arrays
- Debug output files for manual inspection
- Optimized model sequence length (1024 tokens vs default 32k) for better performance


TOKEN MANAGEMENT:
The script uses intelligent token-based processing:
- Ultra-fast token estimation using character-based approximation
- Optimized text splitting respecting sentence boundaries
- Dynamic chunk sizing based on token limits
- Maintains context coherence across token boundaries


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

import typer
import duckdb
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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

# Token management constants
MAX_TOKENS = 512
OVERLAP_TOKENS = 128
MIN_CHUNK_TOKENS = 300  # Desired minimum for chunks (except final ones)
OVERESTIMATION_FACTOR = 1.1

# PRE-COMPILED REGEX PATTERNS
WORD_PATTERN = re.compile(r'\b\w+\b')
PARAGRAPH_PATTERN = re.compile(r'\n\s*\n')
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

# =============================================================================
# TEXT CLEANING AND TOKEN MANAGEMENT FUNCTIONS
# =============================================================================

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

def estimate_tokens_ultra_fast(text: str) -> int:
    """
    Fast token estimation using word counting.
    
    Optimized method that uses word counting with overestimation factor
    to avoid expensive real tokenization that is computationally costly.
    
    Args:
        text: Input text
        
    Returns:
        int: Estimated number of tokens
    """
    if not text:
        return 0
    
    # Ultra-fast word counting using pre-compiled regex
    word_count = len(WORD_PATTERN.findall(text))
    
    # Conservative overestimation for safety (avoids exceeding limits)
    estimated_tokens = int(word_count * OVERESTIMATION_FACTOR) + 10
    
    return estimated_tokens

def split_text_by_tokens_optimized(text: str, max_tokens: int = MAX_TOKENS, 
                                 overlap_tokens: int = OVERLAP_TOKENS, min_tokens: int = MIN_CHUNK_TOKENS) -> List[Dict]:
    """
    Split text into chunks based on token estimation, respecting paragraph boundaries.
    
    Args:
        text (str): Input text to split
        max_tokens (int): Maximum tokens per chunk
        overlap_tokens (int): Overlap tokens between chunks
        min_tokens (int): Minimum tokens per chunk
        
    Returns:
        List[Dict]: List of chunks with text and token count
    """
    if not text or not text.strip():
        return []
    
    # If text is short enough, return as single chunk
    total_tokens = estimate_tokens_ultra_fast(text)
    if total_tokens <= max_tokens:
        return [{"text": text, "token_count": total_tokens}]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    # Split by paragraphs first (using the compiled pattern)
    paragraphs = [p.strip() for p in PARAGRAPH_PATTERN.split(text) if p.strip()]
    
    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens_ultra_fast(paragraph)
        
        # If paragraph alone is too big, force split (simplified approach)
        if paragraph_tokens > max_tokens:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append({"text": current_chunk.strip(), "token_count": current_tokens})
                current_chunk = ""
                current_tokens = 0
            
            # Split giant paragraph by sentences (simplified)
            sentences = [s.strip() for s in re.split(r'[.!?]+', paragraph) if s.strip()]
            for sentence in sentences:
                sentence_tokens = estimate_tokens_ultra_fast(sentence)
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    chunks.append({"text": current_chunk.strip(), "token_count": current_tokens})
                    current_chunk = sentence
                    current_tokens = sentence_tokens
                else:
                    current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
                    current_tokens += sentence_tokens
            continue
        
        # Test if adding this paragraph would exceed token limit
        if current_tokens + paragraph_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append({"text": current_chunk.strip(), "token_count": current_tokens})
            current_chunk = paragraph
            current_tokens = paragraph_tokens
        else:
            current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
            current_tokens += paragraph_tokens
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append({"text": current_chunk.strip(), "token_count": current_tokens})
    
    return chunks

def split_by_sentences(text: str) -> List[str]:
    """
    Split text by sentence boundaries, handling Spanish punctuation.
    
    Args:
        text (str): Input text to split
        
    Returns:
        List[str]: List of sentences
    """
    sentences = []
    current_sentence = ""
    
    sentence_boundary_chars = '.!?'
    
    i = 0
    while i < len(text):
        char = text[i]
        current_sentence += char
        
        # Check for sentence boundary
        if char in sentence_boundary_chars:
            # Look ahead to see if this is really the end of a sentence
            if i + 1 < len(text):
                next_char = text[i + 1]
                # If next character is whitespace or newline, it's likely end of sentence
                if next_char.isspace():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
            else:
                # End of text
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        i += 1
    
    # Add remaining text as final sentence
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return [s for s in sentences if s.strip()]

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
        chunk_number INTEGER,  -- Chunk number for token-based processing
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

def split_text_by_page_break(text: str):
    """
    Splits the text into chunks based on the page pattern:
    {number} followed by at least 5 hyphens.
    """
    page_pattern = re.compile(r'\{(\d+)\}\s*-{5,}')
    chunks = []
    last_index = 0
    last_page = None

    for match in page_pattern.finditer(text):
        page_num = match.group(1)
        chunk_text = text[last_index:match.start()].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "page": page_num})
        last_index = match.end()
        last_page = page_num

    # Last fragment after the last page mark
    remaining = text[last_index:].strip()
    if remaining:
        final_page = str(int(last_page) + 1) if last_page else "1"
        chunks.append({"text": remaining, "page": final_page})

    logger.debug(f"Document split into {len(chunks)} chunks based on page markers")
    return chunks

# %%

def get_url_from_filename(filename: str):
    """
    Generate the URL based on the filename pattern.

    Expected filename format: DDMMYYYY-XXX.md
    Example: 23012025-MAT.md (representing January 23, 2025, MAT section)

    The generated URL points to the PDF document in the DOF repository.

    Args:
        filename (str): Filename of the .md file

    Returns:
        str: The URL to the original PDF document

    Raises:
        ValueError: If the filename doesn't match the expected format
    """
    # Extract just the base filename in case the full path was passed
    base_filename = os.path.basename(filename).replace(".md", "")
    
    # The year should be extracted from the filename (positions 4-8 in 23012025-MAT.pdf)
    # This assumes the format is consistent
    if len(base_filename) >= 8:
        year = base_filename[4:8]  # Extract year (2025 from 23012025-MAT.pdf)
        pdf_filename = f"{base_filename}.pdf"  # Add .pdf extension back
        
        # Construct the URL
        url = f"https://diariooficial.gob.mx/abrirPDF.php?archivo={pdf_filename}&anio={year}&repo=repositorio/"
        return url
    else:
        # Return None or an error message if the filename doesn't match expected format
        raise ValueError(f"Expected filename like 23012025-MAT.md but got {filename}")

def _prepare_document_metadata(file_path: str) -> tuple:
    """
    Extract metadata from the file.
    
    Args:
        file_path (str): Path to the markdown file
        
    Returns:
        tuple: (content, title, url)
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    title = os.path.splitext(os.path.basename(file_path))[0]
    url = get_url_from_filename(file_path)
    logger.debug(f"Processing file '{file_path}' with title '{title}'")
    
    return content, title, url

def _setup_database_document(title: str, url: str, file_path: str) -> dict:
    """
    Configure document in database, cleaning up existing chunks to avoid foreign key violations.
    
    Args:
        title (str): Document title
        url (str): Document URL  
        file_path (str): Path to the document file
        
    Returns:
        dict: Document record with explicit type conversions
        
    Raises:
        ValueError: If any required parameter is None or empty
        Exception: If database operation fails
    """
    if not all([title, url, file_path]):
        raise ValueError("All parameters (title, url, file_path) must be non-empty")
    
    try:
        existing_doc_df = db.execute("SELECT id FROM documents WHERE url = ?", [url]).pl()
        
        if not existing_doc_df.is_empty():
            doc_id = int(existing_doc_df.row(0, named=True)['id'])
            db.execute("DELETE FROM chunks WHERE document_id = ?", [doc_id])
            logger.info(f"Cleaned up existing chunks for document: {title}")
            
            db.execute("""
                UPDATE documents 
                SET title = ?, file_path = ?, created_at = ?
                WHERE url = ?
            """, [title, file_path, datetime.now(), url])
            
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
            """, [title, url, file_path, datetime.now()]).pl()
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

def _prepare_token_chunks(content: str) -> list:
    """
    Divide content into chunks based on token limits using intelligent splitting.
    
    Args:
        content (str): Document content
        
    Returns:
        list: List of token-based chunks with chunk numbers
    """
    # Clean the content first
    cleaned_content = clean_multiple_spaces(content)
    
    # Split using token-based approach
    chunk_dicts = split_text_by_tokens_optimized(cleaned_content, MAX_TOKENS)
    
    # Convert to chunk format with sequential numbering
    token_chunks = []
    for i, chunk_dict in enumerate(chunk_dicts, 1):
        token_chunks.append({"text": chunk_dict["text"], "chunk": str(i)})
    
    logger.debug(f"Document split into {len(token_chunks)} token-based chunks")
    return token_chunks

def _initialize_chunk_processing(file_path: str, verbose: bool) -> tuple:
    """
    Initialize variables for chunk processing.
    
    Args:
        file_path (str): Path to the file
        verbose (bool): If True, creates debug chunks file
        
    Returns:
        tuple: (open_headings_dict, chunks_file, chunks_file_path)
    """
    open_headings_dict = {}
    chunks_file = None
    chunks_file_path = None
    
    if verbose:
        chunks_file_path = os.path.splitext(file_path)[0] + "_chunks.txt"
        chunks_file = open(chunks_file_path, "w", encoding="utf-8")
    
    return open_headings_dict, chunks_file, chunks_file_path


def _generate_chunk_embedding(header: str, chunk_text: str, file_path: str, chunk_counter: int):
    """
    Generate embedding for a chunk.
    
    Args:
        header (str): Chunk header
        chunk_text (str): Chunk text
        file_path (str): Path to the file (for error logging)
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
            embedding = model.encode(f"search_document: {text_for_embedding}", show_progress_bar=False)
        
        del text_for_embedding

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
        # Add token estimation info to debug output
        estimated_tokens = estimate_tokens_ultra_fast(chunk_text)
        
        chunks_file.write(f"--- CHUNK #{chunk_counter} ---\n")
        chunks_file.write(f"Estimated tokens: {estimated_tokens}\n")
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
        file_path (str): Path to the file (for error logging)
        
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

def process_file(file_path, verbose: bool = False):
    """
    Process a markdown file with structured headers and token-based chunking support.
    This function:
    1. Reads file content
    2. Extracts metadata from filename
    3. Deletes any previous version of this document
    4. Splits content by token limits using intelligent splitting
    5. Maintains hierarchical context with open headings
    6. Generates contextual headers for each chunk
    7. Generates vector embeddings for each chunk (including header)
    8. Stores chunks and embeddings in the database
    9. Creates a debug file for manual inspection (only if verbose=True)

    Args:
        file_path (str): Path to the markdown file to process
        verbose (bool): If True, creates debug chunks file
    """
    try:
        # 1. Prepare metadata
        content, title, url = _prepare_document_metadata(file_path)
        
        # 2. Setup document in database
        doc = _setup_database_document(title, url, file_path)
        
        # 3. Prepare token-based chunks
        token_chunks = _prepare_token_chunks(content)
        
        # 4. Initialize chunk processing
        open_headings_dict, chunks_file, chunks_file_path = _initialize_chunk_processing(file_path, verbose)
        
        try:
            # 5. Process each chunk with progress bar
            total_chunks = len(token_chunks)
            chunk_counter = 0
            
            with tqdm(total=total_chunks, desc=f"Processing {title[:30]}...", unit="chunk") as pbar:
                for chunk in token_chunks:
                    chunk_counter += 1
                    open_headings_dict = _process_single_chunk(
                        chunk, chunk_counter, title, open_headings_dict, 
                        doc["id"], chunks_file, verbose, file_path
                    )
                    
                    # Update progress bar
                    pbar.set_postfix({"chunk": f"{chunk['chunk']}"})
                    pbar.update(1)
        finally:
            if chunks_file:
                chunks_file.close()
        
        # 6. Final logging
        logger.info(f"Processing completed for: {file_path}")
        if verbose and chunks_file_path:
            logger.info(f"Chunks file generated at: {chunks_file_path}")
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise

def process_directory(directory_path, verbose: bool = False):
    """
    Recursively process all files in a directory and its subdirectories.

    Args:
        directory_path (str): Path to the directory to process
        verbose (bool): If True, creates debug chunks files
    """
    try:
        # Get the list of files in the directory
        entries = os.listdir(directory_path)
        md_files = [entry for entry in entries if entry.lower().endswith(".md")]
        logger.info(f"Found {len(md_files)} markdown files in {directory_path}")
        
        # Loop through all entries in the directory
        for entry in tqdm(md_files, desc=f"Processing {directory_path}"):
            # Create full path
            entry_path = os.path.join(directory_path, entry)

            # Process the markdown file
            process_file(entry_path, verbose=verbose)
            
        # Process subdirectories separately without progress bar
        for entry in entries:
            entry_path = os.path.join(directory_path, entry)
            # If it's a directory, recursively process it
            if os.path.isdir(entry_path):
                process_directory(entry_path, verbose=verbose)
                
        logger.info(f"Processing completed for directory: {directory_path}")
        
    except Exception as e:
        logger.error(f"Error processing directory {directory_path}: {str(e)}")
        raise


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