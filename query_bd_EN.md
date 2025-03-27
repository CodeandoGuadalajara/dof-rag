# Documentation: query_bd.py

## Overview
This script implements a semantic search engine that uses vector embeddings to find relevant text fragments in a SQLite database. It allows searching for documents by semantic similarity rather than simple keyword matches.

## Dependencies
- `sqlite3`: For database connection and manipulation
- `sqlite_vec`: Extension for vector operations in SQLite
- `sentence_transformers`: For generating text embeddings
- `typer`: For creating the command-line interface

## Embedding Model
- **Model used**: "nomic-ai/modernbert-embed-base"
- The model is loaded once at script startup to optimize performance

## Functionalities

### Main command: `search`
Performs a vector search in the `chunks` table and returns the most relevant results.

#### Parameters:
- `query` (required): Text or question to search for
- `limit` (optional): Maximum number of results (default: 5)
- `db_path` (optional): Path to the SQLite database (default: "dof_db/db.sqlite")

#### Process:
1. Connects to the SQLite database
2. Loads the vector extension
3. Converts the query into a vector embedding
4. Executes an SQL query that:
   - Searches in the `chunks` table
   - Joins with the `documents` table
   - Calculates the cosine distance between embeddings
   - Sorts by similarity (lower distance)
   - Limits the number of results
5. Displays formatted results

## Results Format
For each result shows:
- Document title
- Document URL
- Fragment header
- Similarity score (1 - cosine_distance)
- Fragment text (first 200 characters)

## Command-line Usage
```bash
python query_bd.py "your query here" --limit 10 --db path/to/your/database.sqlite
```

## Database Structure
The script assumes a database with at least:
- `chunks` table with fields: `id`, `document_id`, `header`, `text`, `embedding`
- `documents` table with fields: `id`, `title`, `url` 