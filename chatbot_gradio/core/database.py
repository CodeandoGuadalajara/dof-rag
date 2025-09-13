"""Database operations for RAG chat system using DuckDB with VSS extension."""

import logging
from typing import Any, Dict, List, Optional
import duckdb
from .embeddings import embedding_manager


logger = logging.getLogger(__name__)


def connect_duckdb(path: str) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB database with VSS extension for vector operations.
    
    Args:
        path: Database file path
        
    Returns:
        DuckDB connection with VSS extension loaded
        
    Raises:
        duckdb.Error: Connection or extension loading failure
    """
    try:
        conn = duckdb.connect(path)
        
        # Install VSS extension for vector similarity operations
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        
        # Verify required tables exist (read-only check)
        verify_database_schema(conn)
        
        logger.info(f"Connected to DuckDB at {path} with VSS extension")
        return conn
        
    except duckdb.Error as e:
        logger.error(f"Failed to connect to DuckDB or load VSS extension: {e}")
        raise


def verify_database_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Verify that required tables exist in the database (read-only check).
    
    Args:
        conn: DuckDB connection with VSS extension loaded
        
    Raises:
        duckdb.Error: If required tables are missing
    """
    try:
        # Check if required tables exist
        required_tables = ['documents', 'chunks']
        
        for table in required_tables:
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", 
                [table]
            ).fetchone()
            
            if result[0] == 0:
                raise duckdb.Error(f"Required table '{table}' not found in database")
        
        # Verify embedding dimension matches
        embedding_dim = embedding_manager._embedder.get_sentence_embedding_dimension()
        logger.info(f"Database schema verified - embedding dimension: {embedding_dim}")
        
    except duckdb.Error as e:
        logger.error(f"Database schema verification failed: {e}")
        raise


def query_context(
    conn: duckdb.DuckDBPyConnection, 
    embedding: List[float], 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve chunks most similar to query embedding.
    
    Args:
        conn: DuckDB connection
        embedding: Query vector
        top_k: Maximum results to return
        
    Returns:
        Chunks ranked by similarity score
        
    Raises:
        duckdb.Error: Query execution failure
    """
    try:
        # Execute vector similarity search using array_distance
        expected_dim = embedding_manager._embedder.get_sentence_embedding_dimension()
        query = f"""
            SELECT 
                c.id,
                c.text,
                c.header,
                c.document_id,
                d.title,
                d.url,
                d.file_path,
                array_distance(c.embedding, ?::FLOAT[{expected_dim}]) as distance
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT ?
        """
        result = conn.execute(query, [embedding, top_k]).fetchall()
        
        chunks = []
        for row in result:
            # Convert distance to similarity score for ranking
            distance = row[7]
            similarity = 1.0 / (1.0 + distance) if distance >= 0 else 1.0
            
            chunk = {
                "id": row[0],
                "text": row[1],
                "header": row[2],
                "document_id": row[3],
                "title": row[4],
                "url": row[5],
                "file_path": row[6],
                "distance": distance,
                "similarity": similarity
            }
            chunks.append(chunk)
        
        return chunks
        
    except duckdb.Error as e:
        logger.error(f"Failed to query context: {e}")
        raise


def get_image_descriptions(
    conn: duckdb.DuckDBPyConnection,
    document_name: str,
    page_number: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Get image descriptions for a document and optionally a specific page.
    
    Args:
        conn: DuckDB connection object
        document_name: Name of the document
        page_number: Optional page number to filter by
        
    Returns:
        List of image descriptions
        
    Raises:
        duckdb.Error: If query fails
    """
    try:
        if page_number is not None:
            query = """
                SELECT id, document_name, page_number, image_filename, 
                       description, created_at, updated_at
                FROM image_descriptions
                WHERE document_name = ? AND page_number = ?
                ORDER BY page_number, image_filename
            """
            result = conn.execute(query, [document_name, page_number]).fetchall()
        else:
            query = """
                SELECT id, document_name, page_number, image_filename, 
                       description, created_at, updated_at
                FROM image_descriptions
                WHERE document_name = ?
                ORDER BY page_number, image_filename
            """
            result = conn.execute(query, [document_name]).fetchall()
        
        images = []
        for row in result:
            images.append({
                "id": row[0],
                "document_name": row[1],
                "page_number": row[2],
                "image_filename": row[3],
                "description": row[4],
                "created_at": row[5],
                "updated_at": row[6]
            })
        
        return images
        
    except duckdb.Error as e:
        logger.error(f"Failed to get image descriptions: {e}")
        raise

def close_connection(conn: duckdb.DuckDBPyConnection) -> None:
    """Close DuckDB connection safely.
    
    Args:
        conn: DuckDB connection object to close
    """
    try:
        conn.close()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")