import pymupdf  # PyMuPDF library
from pdf2image import convert_from_path
import pathlib
import os
import time
from typing import List

# Start the timer
start_time = time.time()

# Define Poppler path (adjust based on your system)
poppler_path = r"C:\poppler-24.08.0\Library\bin"

# Define input and output paths
input_pdf = "../_pdf/tables/11012023-MAT-20pages.pdf"
output_md = "./markdown/test/11012023-MAT-20pages.md"

# Ensure the output directory exists
pathlib.Path(output_md).parent.mkdir(parents=True, exist_ok=True)

# Convert PDF to images (optional, only if you need images for some reason)
try:
    images = convert_from_path(input_pdf, poppler_path=poppler_path)
    page_count = len(images)
    print(f"📄 Total pages detected: {page_count}")
except Exception as e:
    print(f"❌ Error converting PDF to images: {e}")
    exit(1)

# Open the PDF with PyMuPDF
try:
    doc = pymupdf.open(input_pdf)
    if doc.page_count != page_count:
        print(f"⚠️ Warning: Page count mismatch. pdf2image: {page_count}, PyMuPDF: {doc.page_count}")
    page_count = doc.page_count  # Use PyMuPDF's page count
except Exception as e:
    print(f"❌ Error opening PDF with PyMuPDF: {e}")
    exit(1)

# Function to extract Markdown from a page using PyMuPDF
def extract_markdown_from_page(page, page_num: int) -> str:
    try:
        # Extract plain text from the page
        text = page.get_text("text")
        if not text.strip():
            return f"Página: {page_num}\n\n*No text content extracted*"

        # Split text into sections of approximately 250-1000 words
        words = text.split()
        section_size_min, section_size_max = 250, 1000
        sections = []
        current_section = []
        word_count = 0

        for word in words:
            current_section.append(word)
            word_count += 1
            if word_count >= section_size_min and (word_count >= section_size_max or word.endswith('.')):
                sections.append(" ".join(current_section))
                current_section = []
                word_count = 0
        
        if current_section:  # Add any remaining words
            sections.append(" ".join(current_section))

        # Format as Markdown with page number
        markdown = f"Página: {page_num}\n\n"
        for i, section in enumerate(sections, 1):
            markdown += f"### Section {i}\n\n{section}\n\n"
        
        return markdown
    except Exception as e:
        print(f"⚠️ Error processing page {page_num}: {e}")
        return f"Página: {page_num}\n\n*Content not processed*"

# Extract Markdown for each page
markdown_pages = []
for i in range(page_count):
    page = doc.load_page(i)  # Load page (0-based index)
    markdown = extract_markdown_from_page(page, i + 1)  # Page number is 1-based
    markdown_pages.append(markdown)

# Join all pages with separators
final_markdown = "\n\n---\n\n".join(markdown_pages)

# Save the Markdown file
try:
    pathlib.Path(output_md).write_text(final_markdown, encoding="utf-8")
    print(f"✅ Conversion completed. Markdown saved to: {output_md}")
except Exception as e:
    print(f"❌ Error saving Markdown file: {e}")

# Calculate and display total processing time
end_time = time.time()
total_time = end_time - start_time
print(f"Total processing time: {total_time:.2f} seconds")

# Close the document
doc.close()