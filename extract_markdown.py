# encoding: utf-8

# Description: Script para extraer markdown del Diario Oficial de la Federación (DOF) de México
# Uses Gemini 2.0 Flash to OCR scanned PDFs (pre-1999 DOF documents)
# Based on: https://www.sergey.fyi/articles/gemini-flash-2

import os
import pathlib
from pathlib import Path

import pikepdf
import typer
from google import genai
from google.genai import types

from get_dof import is_file_valid

API_KEY = os.environ.get("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)


def count_pages(pdf_path):
    with pikepdf.open(pdf_path) as pdf:
        return len(pdf.pages)


def extract_markdown_from_pdf(pdf_path: str) -> str:
    """
    Uses Gemini 2.0 Flash to extract markdown from a scanned PDF.
    """
    page_count = count_pages(pdf_path)
    filepath = pathlib.Path(pdf_path)

    prompt = f"""\
    OCR the following page into Markdown.

    Chunk the document into sections of roughly 250 - 1000 words. Our goal is
    to identify parts of the page with same semantic theme. These chunks will
    be embedded and used in a RAG pipeline.

    Surround the chunks with <chunk pages="page_numbers"> </chunk> html tags,
    if multiple pages separate them by commas like <chunk pages="2,3"> </chunk>.

    Please output each and every of the {page_count} pages in the PDF.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=filepath.read_bytes(),
                mime_type="application/pdf",
            ),
            prompt,
        ],
    )
    return response.text


def extract_from_to(pdf_path: str, md_path: str) -> None:
    """
    Extract markdown from a PDF file and save it to a Markdown file.
    """
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    if not is_file_valid(pdf_path):
        typer.echo(f"Skipping invalid PDF: {pdf_path}")
        return

    # Create the target directory if it doesn't exist
    Path(os.path.dirname(md_path)).mkdir(parents=True, exist_ok=True)

    markdown_text = extract_markdown_from_pdf(pdf_path)
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_text)


def main(pdf_path: str, md_path: str) -> None:
    extract_from_to(pdf_path, md_path)


if __name__ == "__main__":
    typer.run(main)
