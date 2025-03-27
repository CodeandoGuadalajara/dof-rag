# DOF Documents Processing and Query System

## Table of Contents
- [Introduction](#introduction)
- [Main Features](#main-features)
- [Requirements and Dependencies](#requirements-and-dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Document Processing](#document-processing)
  - [Document Querying](#document-querying)
- [Code Structure](#code-structure)
  - [Initial Configuration](#initial-configuration)
  - [Database Structure](#database-structure)
  - [Text Processing](#text-processing)
  - [Embeddings Generation](#embeddings-generation)
  - [Database Storage](#database-storage)
- [Technical Operation](#technical-operation)
  - [Header Detection](#header-detection)
  - [Page-based Text Division](#page-based-text-division)
  - [Contextual Headers Construction](#contextual-headers-construction)
- [Gemini Query System](#gemini-query-system)
  - [Vector Search](#vector-search)
  - [Expanded Context](#expanded-context)
  - [Gemini 2.0 Query](#gemini-20-query)
  - [Operation Modes](#operation-modes)
- [Workflow](#workflow)
  - [Document Processing](#document-processing-1)
  - [Information Querying](#information-querying)
- [Limitations and Considerations](#limitations-and-considerations)

## Introduction

This system processes documents from the Federal Official Gazette (DOF) stored in Markdown format. Its main objective is to extract content, detect the hierarchical structure of documents, divide the text into page-based fragments, and generate vector embeddings to enable efficient semantic searches.

The processed documents are stored in a SQLite database optimized for vector searches, allowing semantic similarity queries on DOF content.

The system also includes a query module that enables semantic searches and provides contextualized answers using Google's Gemini 2.0 model.

## Main Features

- Processing of DOF Markdown files
- Intelligent extraction of hierarchical structure through header detection
- Text division into chunks based on page markers
- Vector embeddings generation using SentenceTransformer
- Storage in SQLite database with vector capabilities via sqlite-vec
- Creation of contextual headers to improve semantic search
- Generation of debug files to verify text segmentation
- Semantic search using Euclidean distance between embeddings
- Obtaining expanded context (adjacent chunks) to improve answers
- Queries to Gemini 2.0 with specific instructions for processing DOF documents
- Interactive mode for continuous querying
- Logging system for processing tracking

## Requirements and Dependencies

The system depends on the following libraries and technologies:

- typer: for command-line interface
- fastlite: for simplified SQLite database manipulation
- sqlite-vec: to enable vector capabilities in SQLite
- sentence-transformers: for embeddings generation
- tokenizers: for text tokenization
- tqdm: for progress bars
- google.generativeai: for integration with Gemini 2.0 model
- numpy: for numerical processing
- python-dotenv: for loading environment variables

## Installation

1. Clone the repository or download the scripts
2. Install the necessary dependencies using uv (faster and more modern package installer):

```bash
# Install uv if you don't have it
curl -sSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
uv pip install typer fastlite sqlite-vec sentence-transformers tokenizers tqdm google-generativeai numpy python-dotenv
```

3. Configure a Google Gemini API key in a .env file:

```
GOOGLE_API_KEY=your_api_key_here
```

4. Make sure you have an appropriate directory structure to store DOF files in Markdown format

## Usage

### Document Processing

To process all Markdown files in a specific directory:

```bash
python extract_embeddings_1_page.py /path/to/directory/with/md/files
```

The script will perform the following actions:
1. Process each Markdown file in the directory
2. Extract the hierarchical structure of the documents
3. Divide the text into chunks based on page markers
4. Generate embeddings for each chunk
5. Store everything in the SQLite database
6. Create debug files named "filename_chunks.txt" showing how the text was divided

## Code Structure

### Initial Configuration

The script begins by initializing the necessary models and configurations:

- Loading the tokenizer and embeddings model (nomic-ai/modernbert-embed-base)
- Configuring the logging system for debugging and tracking
- Initializing the SQLite database with vector extensions

### Database Structure

The database consists of two main tables:

1. **documents**: Stores document metadata
   - id: Unique identifier
   - title: Document title
   - url: URL to the original document in the DOF
   - file_path: Path to the local file
   - created_at: Processing date and time

2. **chunks**: Stores text fragments and their embeddings
   - id: Unique identifier
   - document_id: Relationship with the original document
   - text: Fragment text
   - header: Contextual header to improve search
   - embedding: Serialized embeddings vector
   - created_at: Processing date and time

### Text Processing

The code implements several key functions to process the documents:

- `split_text_by_page_break`: Divides the text into sections based on page markers
- `extract_headings`: Extracts the Markdown headings present in the text
- `build_chunk_header`: Builds contextual headers for each chunk
- `get_url_from_filename`: Generates the DOF URL from the filename

### Embeddings Generation

Embeddings are generated using the SentenceTransformer model (nomic-ai/modernbert-embed-base). For each chunk, a text composed of the contextual header and content is built and encoded using the model.

### Database Storage

Processed documents and chunks are stored in the SQLite database, which has been extended with vector capabilities through sqlite-vec. This allows for efficient semantic searches.

## Technical Operation

### Header Detection

The system uses regular expressions to detect headers in Markdown format:
```python
HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$')
```

This expression identifies lines that begin with between 1 and 6 '#' symbols, followed by a space and text, capturing both the header level and its content.

### Page-based Text Division

The `split_text_by_page_break` algorithm is central to the system. It divides the text into fragments based on page markers with the following format:

```
{page_number}-----------------------------------------------------
```

The system detects these markers and creates separate chunks for each page, preserving the page number for reference. This allows processing DOF documents while maintaining the original page structure, which facilitates referencing and citation.

### Contextual Headers Construction

For each chunk, a contextual header is built that includes:
- The document title
- The headings found in the current chunk
- The page number

Example: "Document: 23012025-MAT | Section: Main Title > Subtitle | Page: 3"

This header helps the embeddings capture the hierarchical context and location in the document, improving the quality of searches.

## Gemini Query System

The `gemini_query.py` module implements an advanced semantic search and query system using Gemini 2.0 for Federal Official Gazette (DOF) documents.

### Vector Search

The system uses vector embeddings and Euclidean distance to find the most relevant fragments for a query:

```python
def find_relevant_chunks(query, all_chunks, top_k=10):
    # Generate embedding for the query
    query_embedding = model.encode(f"search_query: {query}")
    
    # Calculate distances
    scored_results = []
    for chunk in all_chunks:
        chunk_embedding = deserialize_embedding(chunk["embedding_blob"])
        distance = np.linalg.norm(query_embedding - chunk_embedding)
        scored_results.append({
            # ... chunk metadata ...
            "distance": distance,
            "similarity": 1.0 / (1.0 + distance)
        })
    
    # Sort by smallest distance (highest similarity)
    return sorted(scored_results, key=lambda x: x["distance"])[:top_k]
```

This function:
1. Converts the text query to an embeddings vector using the same model as in processing
2. Calculates the Euclidean distance between the query and each chunk in the database
3. Sorts the results by relevance (smallest distance)
4. Returns the `top_k` most relevant results

### Expanded Context

To improve the quality of responses, the system obtains an expanded context including chunks adjacent to the most relevant one:

```python
def get_extended_context(best_chunk, all_chunks):
    # Filter chunks from the same document
    same_doc_chunks = [c for c in all_chunks if c["document_id"] == best_chunk["document_id"]]
    
    # ... find position of the best chunk ...
    
    # Build extended context with previous chunk + best chunk + next chunk
    context = ""
    if best_index > 0:
        context += same_doc_chunks[best_index - 1]["chunk_text"] + "\n\n"
    context += best_chunk["chunk_text"] + "\n\n"
    if best_index < len(same_doc_chunks) - 1:
        context += same_doc_chunks[best_index + 1]["chunk_text"]
```

This strategy:
1. Identifies all chunks from the same document as the most relevant chunk
2. Determines the position of the most relevant chunk in the original sequence
3. Builds a context that includes the previous chunk, the most relevant chunk, and the next chunk

### Gemini 2.0 Query

The system uses Google's Gemini 2.0 model to generate contextualized responses:

```python
# Build prompt for Gemini
prompt = (
    f"Using the following context extracted from a Federal Official Gazette document:\n\n"
    f"---\n{extended_context}\n---\n\n"
    f"Instructions:\n"
    # ... specific instructions ...
    f"Question:\n{query}"
)

# Generate response with Gemini
model_gemini = genai.GenerativeModel('gemini-2.0-flash')
response = model_gemini.generate_content(prompt)
```

The prompt includes:
1. The expanded context extracted from the documents
2. Specific instructions for the model on how it should respond
3. The user's original query

### Operation Modes

The system offers two operation modes:

1. **Direct query**: Performs a single query and displays the result
   ```bash
   python gemini_query.py --query "your query here"
   ```

2. **Interactive mode**: Allows for multiple queries in a session
   ```bash
   python gemini_query.py --interactive
   ```

In interactive mode, the system will display a prompt to enter queries and will continue until the user types 'exit', 'quit', or 'salir'.

## Workflow

### Document Processing

1. For each Markdown file in the specified directory:
   - Document metadata is extracted
   - Any previous version of the document in the database is deleted
   - The new document is inserted into the 'documents' table
   - It checks if the document contains page markers
   - If it finds markers, it divides the text by pages; otherwise, it treats all content as a single page
   - For each page:
     - Present headings are extracted
     - The contextual header is built
     - The embedding is generated by combining the header and content
     - It's saved in the database and in a debug file

### Information Querying

1. The user makes a query in natural language
2. The system retrieves all chunks stored in the database
3. The query is converted into an embeddings vector
4. Similarity is calculated between the query and all chunks
5. The most relevant chunk is selected and expanded context (adjacent chunks) is obtained
6. A prompt is built for Gemini 2.0 with the context and query
7. Gemini 2.0 generates a response based on the context
8. The response is displayed along with metadata from the source document

## Limitations and Considerations

- The system depends on the correct identification of page markers in {number}----- format
- If no page markers are found, the entire document is processed as a single page
- The filename must follow a specific format (DDMMYYYY-XXX.md) to correctly generate URLs
- The quality of responses depends on both the accuracy of the vector search and the Gemini model
- Requires a valid Google API key to use Gemini 2.0
- Result visualization is optimized for terminal with text format
- Performance may be affected when the database contains many documents
- The logging system facilitates debugging and tracking of processing
