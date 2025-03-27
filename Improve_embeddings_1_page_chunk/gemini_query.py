"""
DOF Document Query System with Gemini 2.0

This module implements an advanced semantic search and query system using
Gemini 2.0 for documents from the Official Journal of the Federation (DOF).

Features:
- Vector search with Euclidean distance
- Expanded context retrieval (adjacent chunks)
- Queries to Gemini 2.0 with specific instructions
- Interactive mode for continuous queries

Usage:
python gemini_query.py [--query "query"] [--interactive]
"""

import os
import numpy as np
import typer
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
from fastlite import database
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not configured in the environment variables")

genai.configure(api_key=api_key)

# Embeddings model configuration
model = SentenceTransformer("nomic-ai/modernbert-embed-base", trust_remote_code=True)

# Database initialization
db = database("dof_db/db.sqlite")


def deserialize_embedding(blob):
    """
    Converts a BLOB stored in the database to a NumPy vector of type float32.
    """
    return np.frombuffer(blob, dtype=np.float32)


def get_all_chunks():
    """
    Gets all chunks stored in the database.
    
    Returns:
        list: List of dictionaries with each chunk's information
    """
    results = db.query(
        """
        SELECT 
            c.id as chunk_id,
            c.text as chunk_text,
            c.document_id as document_id,
            d.title as doc_title,
            d.url as doc_url,
            c.embedding as embedding_blob
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        ORDER BY c.id;
        """
    )
    return list(results)


def find_relevant_chunks(query, all_chunks, top_k=10):
    """
    Finds the most relevant chunks for the query.
    
    Args:
        query (str): User query
        all_chunks (list): List of all chunks
        top_k (int): Maximum number of chunks to return
        
    Returns:
        list: Sorted list of most relevant chunks
    """
    # Generate embedding for the query
    query_embedding = model.encode(f"search_query: {query}")
    
    # Calculate distances
    scored_results = []
    for chunk in all_chunks:
        chunk_embedding = deserialize_embedding(chunk["embedding_blob"])
        distance = np.linalg.norm(query_embedding - chunk_embedding)
        scored_results.append({
            "chunk_id": chunk["chunk_id"],
            "chunk_text": chunk["chunk_text"],
            "document_id": chunk["document_id"],
            "doc_title": chunk["doc_title"],
            "doc_url": chunk["doc_url"],
            "distance": distance,
            "similarity": 1.0 / (1.0 + distance)  # Convert distance to similarity
        })
    
    # Sort by lowest distance (highest similarity)
    return sorted(scored_results, key=lambda x: x["distance"])[:top_k]


def get_extended_context(best_chunk, all_chunks):
    """
    Gets extended context including adjacent chunks.
    
    Args:
        best_chunk (dict): The most relevant chunk
        all_chunks (list): List of all chunks
        
    Returns:
        str: Extended context with adjacent chunks
    """
    # Filter chunks from the same document
    same_doc_chunks = [c for c in all_chunks if c["document_id"] == best_chunk["document_id"]]
    
    # Sort by ID to maintain the original order
    same_doc_chunks = sorted(same_doc_chunks, key=lambda c: c["chunk_id"])
    
    # Find the position of the best chunk
    best_index = None
    for i, chunk in enumerate(same_doc_chunks):
        if chunk["chunk_id"] == best_chunk["chunk_id"]:
            best_index = i
            break
    
    # Build extended context
    context = ""
    
    if best_index is not None:
        # Add previous chunk if it exists
        if best_index > 0:
            context += same_doc_chunks[best_index - 1]["chunk_text"] + "\n\n"
            
        # Add the best chunk
        context += best_chunk["chunk_text"] + "\n\n"
        
        # Add next chunk if it exists
        if best_index < len(same_doc_chunks) - 1:
            context += same_doc_chunks[best_index + 1]["chunk_text"]
    else:
        # If not found (rare case), use only the best chunk
        context = best_chunk["chunk_text"]
    
    return context


def query_gemini(query, verbose=True):
    """
    Performs a complete query: search for relevant chunks and generate response with Gemini.
    
    Args:
        query (str): User query
        verbose (bool): Whether to display detailed information
        
    Returns:
        dict: Dictionary with response and metadata
    """
    try:
        if verbose:
            print(f"\n🔍 Processing query: '{query}'")
            print("Getting chunks from database...")
        
        # Get all chunks
        all_chunks = get_all_chunks()
        
        if not all_chunks:
            return {
                "success": False,
                "error": "No documents found in the database."
            }
            
        if verbose:
            print(f"Retrieved {len(all_chunks)} total chunks.")
            print("Searching for relevant chunks...")
        
        # Find relevant chunks
        top_results = find_relevant_chunks(query, all_chunks)
        
        if not top_results:
            return {
                "success": False,
                "error": "No relevant chunks found for this query."
            }
            
        # Get the best result
        best_chunk = top_results[0]
        
        if verbose:
            print(f"Found {len(top_results)} relevant chunks.")
            print(f"Best chunk: ID={best_chunk['chunk_id']}, Distance={best_chunk['distance']:.4f}")
            print("Getting extended context...")
        
        # Get extended context (with adjacent chunks)
        extended_context = get_extended_context(best_chunk, all_chunks)
        
        if verbose:
            print("Generating prompt for Gemini...")
        
        # Build prompt for Gemini
        prompt = (
            f"Using the following context extracted from an Official Journal of the Federation document:\n\n"
            f"---\n{extended_context}\n---\n\n"
            f"Instructions:\n"
            f"- If the query is directly related to the content, answer based on the provided information.\n"
            f"- If the query is a greeting, farewell, or common social interaction, respond in a cordial and appropriate manner.\n"
            f"- If information is requested that is not found in the context, clearly indicate that there is no relevant information available in the document.\n"
            f"- Make sure to correctly interpret the request, answering only what is asked and avoiding adding irrelevant information.\n\n"
            f"Question:\n{query}"
        )
        
        if verbose:
            print("Sending query to Gemini...")
        
        # Generate response with Gemini
        model_gemini = genai.GenerativeModel('gemini-2.0-flash')
        response = model_gemini.generate_content(prompt)
        
        # Prepare result
        result = {
            "success": True,
            "query": query,
            "response": response.text,
            "source": {
                "title": best_chunk["doc_title"],
                "url": best_chunk["doc_url"],
                "distance": best_chunk["distance"],
                "similarity": best_chunk["similarity"]
            },
            "context": extended_context
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error during process: {str(e)}"
        }


def display_result(result):
    """
    Displays the query result in a user-friendly way.
    
    Args:
        result (dict): Query result
    """
    if not result["success"]:
        print(f"\n❌ Error: {result['error']}")
        return
    
    print("\n" + "="*60)
    print(f"ANSWER TO: '{result['query']}'")
    print("="*60)
    print(result["response"])
    print("\n" + "-"*60)
    print(f"Source: {result['source']['title']}")
    print(f"URL: {result['source']['url']}")
    print(f"Relevance (distance): {result['source']['distance']:.4f}")
    print(f"Similarity: {result['source']['similarity']:.4f}")
    print("-"*60)
    print("\nCONTEXT USED:")
    print("-"*60)
    print(result["context"])
    print("="*60)


def interactive_mode():
    """
    Starts an interactive mode for continuous queries.
    """
    print("\n" + "="*60)
    print("Interactive Query System with Gemini 2.0")
    print("Type 'exit' to quit")
    print("="*60 + "\n")
    
    while True:
        query = input("\nQuery: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not query:
            continue
            
        result = query_gemini(query)
        display_result(result)


def main(
    query: str = typer.Option(None, "--query", "-q", help="Direct query without interactive mode"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Start interactive mode")
):
    """
    DOF documents query system with Gemini 2.0.
    """
    if query:
        # Direct query mode
        result = query_gemini(query)
        display_result(result)
    elif interactive:
        # Interactive mode
        interactive_mode()
    else:
        print("You must specify a query (--query) or use interactive mode (--interactive)")


if __name__ == "__main__":
    typer.run(main) 