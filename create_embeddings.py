
import os
import json
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from extract_markdown import extract_markdown_from_pdf

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv("GENAI")
genai.configure(api_key=GOOGLE_API_KEY)

# Model configurations
FAISS_EMBEDDING_MODEL = SentenceTransformer("hkunlp/instructor-xl")
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"  
FAISS_INDEX_PATH = "faiss_index.idx"
METADATA_PATH = "faiss_index.json"



def extract_pdfs_to_markdown(source_dir: str, target_dir: str):
    """
    Iterates over the PDF files in the source directory, extracts their content as Markdown,
    and saves them in a structured folder within the target directory.

    Args:
        source_dir (str): Root directory containing PDFs organized in subfolders.
        target_dir (str): Root directory where the extracted Markdown files will be saved.
    """
    for month in sorted(os.listdir(source_dir)):  # Iterate over subdirectories (months)
        month_path = os.path.join(source_dir, month)

        if not os.path.isdir(month_path):  # Skip files, only process directories
            continue  

        # Create the corresponding directory in the target location
        target_month_path = os.path.join(target_dir, month)
        os.makedirs(target_month_path, exist_ok=True)

        for pdf_file in sorted(os.listdir(month_path)):  # Iterate over PDF files in the subdirectory
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(month_path, pdf_file)  # Full path to the PDF file
                md_file_name = pdf_file.replace(".pdf", ".md")  # Change file extension
                md_path = os.path.join(target_month_path, md_file_name)  # Output file path

                #  Extract content from the PDF
                markdown_text = extract_markdown_from_pdf(pdf_path)

                #  Validate that the extracted text is not None before writing to the file
                if markdown_text is None:
                    print(f"⚠️ Warning: Could not extract content from {pdf_path}. Skipping file.")
                    continue  # Skip this file and continue with the next one

                # Save extracted content as a Markdown file
                with open(md_path, "w", encoding="utf-8") as md_file:
                    md_file.write(markdown_text)

                print(f"✅ Processed: {pdf_path} → {md_path}")



# function to extract file's metadata
def parse_metadata_from_filename(file_name):
    """
    Extracts the date and category from the file name.

    Args:
        file_name (str): The name of the file from which metadata will be extracted.

    Returns:
        dict: A dictionary containing the extracted date and category.
    """
    parts = file_name.split('-')  # Split the filename by dashes
    if len(parts) >= 2:
        date_part = parts[0]  # Extract the first part as the date component
        try:
            # Extract day, month, and year from the date part
            day, month, year = date_part[:2], date_part[2:4], date_part[4:]
            date = f"{year}-{month}-{day}"  # Format the date as YYYY-MM-DD
        except:
            date = "Unknown"  # Assign "Unknown" if parsing fails
        
        # Extract category from the second part of the filename (before the first dot)
        category = parts[1].split('.')[0] if len(parts) > 1 else "Unknown"
    else:
        # Default to "Unknown" if the filename does not match the expected format
        date, category = "Unknown", "Unknown"

    return {"date": date, "category": category}




def generate_faiss_embeddings(md_dir: str, faiss_index_path: str, metadata_path: str, model_name="hkunlp/instructor-xl"):
    """
    Generates embeddings from Markdown files and stores them in a FAISS index.
    Also saves metadata in a JSON file.

    Args:
        md_dir (str): Directory containing Markdown files.
        faiss_index_path (str): Path to save the FAISS index.
        metadata_path (str): Path to save the metadata in JSON format.
        model_name (str): Embedding model to use.

    Returns:
        index (faiss.Index): The generated FAISS index.
        metadata (list): List of metadata from the processed files.
    """

    # Load the embedding model
    model = SentenceTransformer(model_name)

    embeddings = []
    metadata = []

    # Iterate over Markdown files in the specified directory
    for root, _, files in os.walk(md_dir):
        for file in sorted(files):
            if file.endswith(".md"):
                md_path = os.path.join(root, file)

                # Read the content of the Markdown file
                with open(md_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                # Generate an embedding for the extracted text
                embedding = model.encode(text).astype("float32")
                embeddings.append(embedding)

                # Store metadata for the file
                metadata.append({"file": file, "path": md_path})

                print(f"✅ Processed: {md_path} → Embedding generated")

    # Convert embeddings list to a NumPy array
    embeddings = np.array(embeddings, dtype="float32")

    # Check if any embeddings were generated
    if len(embeddings) == 0:
        raise ValueError("❌ No embeddings were generated. Ensure that Markdown files exist.")

    # Determine the embedding dimension
    dimension = embeddings.shape[1]

    # Create a FAISS index using L2 distance
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the FAISS index to a file
    faiss.write_index(index, faiss_index_path)
    print(f"✅ FAISS index saved at: {faiss_index_path}")

    # Save metadata to a JSON file
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Metadata saved at: {metadata_path}")

    return index, metadata


def search_faiss(query, index, metadata, top_k=5, filter_date=None, filter_category=None, filter_page=None):
    """
    Performs a search in FAISS with the option to filter by date, category, or page.

    Args:
        query (str): The search query.
        index (faiss.Index): The preloaded FAISS index.
        metadata (list): List of metadata associated with the indexed documents.
        top_k (int): Number of results to return.
        filter_date (str, optional): Filter results by date.
        filter_category (str, optional): Filter results by category.
        filter_page (str, optional): Filter results by page number.

    Returns:
        list: A list of results sorted by relevance.
    """
    if index is None or metadata is None:
        raise ValueError("FAISS index or metadata is not properly loaded.")

    # Generate an embedding for the query
    query_embedding = FAISS_EMBEDDING_MODEL.encode(query).astype("float32").reshape(1, -1)

    # Ensure the query embedding dimension matches the FAISS index dimension
    if query_embedding.shape[1] != index.d:
        raise ValueError(f"Embedding dimension ({query_embedding.shape[1]}) does not match FAISS ({index.d}).")

    # Perform the FAISS search
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue  # Skip invalid indices

        doc_metadata = metadata[idx]

        # Apply optional filters
        if filter_date and doc_metadata.get("date") != filter_date:
            continue
        if filter_category and doc_metadata.get("category") != filter_category:
            continue
        if filter_page and filter_page not in doc_metadata.get("pages", ""):
            continue

        # Store the search result
        results.append({
            "rank": len(results) + 1,
            "file": doc_metadata.get("file", "Unknown"),
            "path": doc_metadata.get("path", "Unknown"),
            "distance": float(distances[0][i])
        })

    # Return results sorted by distance (lower is better)
    return sorted(results, key=lambda x: x["distance"])



def generate_rag_response(query, index, metadata, top_k=3):
    """
    Uses FAISS to retrieve relevant documents and generates a response using Google Gemini API.
    """
    results = search_faiss(query, index, metadata, top_k=top_k)
    
    if not results:
        gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        response = gemini_model.generate_content(query)
        return response.text.strip()
    
    context = "\n\n".join([
        f"Document: {res['file']}\nContent:\n{open(res['path'], 'r', encoding='utf-8').read()[:2000]}"
        for res in results
    ])
    
    prompt = f"""
    Use the retrieved information to answer the question:
    {context}
    Question: {query}
    Answer:
    """
    
    gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
    response = gemini_model.generate_content(prompt)
    
    return response.text.strip()
