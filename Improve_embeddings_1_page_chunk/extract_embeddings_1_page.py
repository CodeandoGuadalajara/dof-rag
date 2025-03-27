import os
import re
from datetime import datetime
import logging  # Added logging module

import typer
from fastlite import database
from sqlite_vec import load, serialize_float32
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from tqdm import tqdm
from os import getenv
import numpy as np
from dotenv import load_dotenv

# -----------------------------------------------------
# Logging system configuration
# -----------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dof_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dof_embeddings")

# -----------------------------------------------------
# Models and tokenizers configuration
# -----------------------------------------------------
tokenizer = Tokenizer.from_pretrained("nomic-ai/modernbert-embed-base")
model = SentenceTransformer("nomic-ai/modernbert-embed-base", trust_remote_code=True)

# -----------------------------------------------------
# Initialize database with sqlite-vec
# -----------------------------------------------------
db = database("dof_db/db.sqlite")
db.conn.enable_load_extension(True)
load(db.conn)
db.conn.enable_load_extension(False)

# -----------------------------------------------------
# Create/update table schema
# -----------------------------------------------------
db.t.documents.create(
    id=int,
    title=str,
    url=str,
    file_path=str,
    created_at=datetime,
    pk="id",
    ignore=True
)
db.t.documents.create_index(["url"], unique=True, if_not_exists=True)

db.t.chunks.create(
    id=int,
    document_id=int,
    text=str,
    header=str,
    embedding=bytes,
    created_at=datetime,
    pk="id",
    foreign_keys=[("document_id", "documents")],
    ignore=True,
)

# -----------------------------------------------------
# Document processing functions
# -----------------------------------------------------

# Pattern to detect headings (titles/subtitles) in Markdown
HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$')

def split_text_by_page_break(text: str):
    """
    Divides the complete text into chunks based on the page marker.
    The separator is expected to have the following format:
      {number} followed by at least 5 dashes.
    Returns a list of dictionaries with:
      - "text": complete content of the page
      - "page": page number extracted from the marker (as string)
    """
    page_pattern = re.compile(r'\{(\d+)\}\s*-{5,}')
    chunks = []
    last_index = 0
    last_page = None
    for match in page_pattern.finditer(text):
        page_num = match.group(1)
        chunk_text = text[last_index:match.start()].strip()
        logger.debug(f"Found page separator {page_num} at position {match.start()} - {match.end()}")
        if chunk_text:
            chunks.append({"text": chunk_text, "page": page_num})
            logger.debug(f"Added chunk for page {page_num} with length {len(chunk_text)}")
        last_index = match.end()
        last_page = page_num
    remaining = text[last_index:].strip()
    if remaining:
        final_page = last_page if last_page else "1"
        chunks.append({"text": remaining, "page": final_page})
        logger.debug(f"Added final chunk for page {final_page} with length {len(remaining)}")
    return chunks

def extract_headings(text: str) -> list:
    """
    Simply extracts all headings (titles and subtitles) present in the text.
    Only the headings that appear in the current chunk are returned.
    """
    headings = []
    for line in text.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            headings.append(heading_text)
            logger.debug(f"Heading found - Level: {level}, Text: '{heading_text}'")
    return headings

def build_chunk_header(doc_title: str, heading_list: list):
    """
    Builds the contextual header using the document title and the list of extracted headings.
    """
    if not heading_list:
        return f"Document: {doc_title}"
    hierarchy_str = " > ".join(heading_list)
    return f"Document: {doc_title} | Section: {hierarchy_str}"

def get_url_from_filename(filename: str) -> str:
    """
    Generates the URL based on the filename pattern.
    Expected format: DDMMYYYY-XXX.md
    Example: 23012025-MAT.md
    """
    base_filename = os.path.basename(filename).replace(".md", "")
    if len(base_filename) >= 8:
        year = base_filename[4:8]
        pdf_filename = f"{base_filename}.pdf"
        url = f"https://diariooficial.gob.mx/abrirPDF.php?archivo={pdf_filename}&anio={year}&repo=repositorio/"
        return url
    else:
        raise ValueError(f"Expected filename like 23012025-MAT.md but got {filename}")

def process_file(file_path: str):
    """
    Processes a complete markdown file, separating it into pages based on the page marker,
    extracting only the headings from the current chunk (and inheriting the last title if it was cut)
    and generating embeddings for each chunk. Debug prints are added to track the process.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    title = os.path.splitext(os.path.basename(file_path))[0]
    url = get_url_from_filename(file_path)
    logger.debug(f"Processing file '{file_path}' with title '{title}'")

    db.t.documents.delete_where("url = ?", [url])
    doc = db.t.documents.insert(
        title=title,
        url=url,
        file_path=file_path,
        created_at=datetime.now()
    )

    if re.search(r'\{\d+\}\s*-{5,}', content):
        page_chunks = split_text_by_page_break(content)
    else:
        page_chunks = [{"text": content, "page": "1"}]
        logger.debug("No page separator found; using all content as a single chunk.")

    chunks_file_path = os.path.splitext(file_path)[0] + "_chunks.txt"
    global_chunk_counter = 1
    last_heading = None

    with open(chunks_file_path, "w", encoding="utf-8") as chunks_file:
        for chunk in page_chunks:
            chunk_text = chunk["text"]
            page = chunk["page"]
            logger.debug(f"Processing chunk for page {page} with length {len(chunk_text)}")

            current_headings = extract_headings(chunk_text)
            logger.debug(f"Current chunk headings: {current_headings}")

            # Inherit last title if previous chunk was cut
            if last_heading:
                header_headings = [last_heading] + current_headings
            else:
                header_headings = current_headings

            logger.debug(f"Final heading for header (chunk {global_chunk_counter}): {header_headings}")
            header = f"{build_chunk_header(title, header_headings)} | Page: {page}"
            logger.debug(f"Header built for page {page}: {header}")

            # Update last_heading only if there's a new heading
            if current_headings:
                last_heading = current_headings[-1]

            text_for_embedding = f"{header}\n\n{chunk_text}"
            embedding = model.encode(f"search_document: {text_for_embedding}")

            chunks_file.write(f"--- CHUNK #{global_chunk_counter} ---\n")
            chunks_file.write(f"Header: {header}\n")
            chunks_file.write(f"Text:\n{chunk_text}\n")
            chunks_file.write("\n" + "-"*50 + "\n\n")
            logger.debug(f"Chunk #{global_chunk_counter} processed for page {page}")
            global_chunk_counter += 1

            db.t.chunks.insert(
                document_id=doc["id"],
                text=chunk_text,
                header=header,
                embedding=embedding,
                created_at=datetime.now(),
            )

    logger.info(f"Processing completed for: {file_path}")
    logger.info(f"Chunks file generated at: {chunks_file_path}")


def main(directory: str):
    """
    Processes all .md files in a directory.
    """
    md_files = [f for f in os.listdir(directory) if f.endswith(".md")]
    for f in md_files:
        file_path = os.path.join(directory, f)
        process_file(file_path)

if __name__ == "__main__":
    typer.run(main)
