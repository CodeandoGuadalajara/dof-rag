import sqlite3
from sqlite_vec import load, serialize_float32
from sentence_transformers import SentenceTransformer
import typer

app = typer.Typer()

MODEL_NAME = "nomic-ai/modernbert-embed-base"
DEFAULT_DB = "dof_db/db.sqlite"

# Load the model only once
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

@app.command()
def search(
    query: str = typer.Argument(..., help="Question or text to search for"),
    limit: int = typer.Option(5, "--limit", "-n", help="Maximum number of results"),
    db_path: str = typer.Option(DEFAULT_DB, "--db", help="Path to SQLite database")
):
    """
    Performs a vector search on the `chunks` table and returns the X most relevant results.
    """
    # Connection and loading of the vector extension
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    load(conn)
    conn.enable_load_extension(False)

    # Query embedding
    embedding = model.encode(f"search_document: {query}")
    blob = serialize_float32(embedding)

    sql = """
    SELECT
        chunks.id,
        documents.title,
        documents.url,
        chunks.header,
        chunks.text,
        vec_distance_cosine(chunks.embedding, ?) AS distance
    FROM chunks
    JOIN documents ON chunks.document_id = documents.id
    ORDER BY distance ASC
    LIMIT ?
    """
    cursor = conn.execute(sql, (blob, limit))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        typer.echo("❌ No results found.")
        raise typer.Exit()

    for i, (chunk_id, title, url, header, text, distance) in enumerate(rows, start=1):
        similarity = 1 - distance  # cosine_distance = 1 - cosine_similarity
        typer.echo(f"🔎 Result #{i}")
        typer.echo(f" • Title     : {title}")
        typer.echo(f" • URL       : {url}")
        typer.echo(f" • Header    : {header}")
        typer.echo(f" • Similarity: {similarity:.4f}")
        typer.echo(f" • Text      : {text[:200]}{'...' if len(text) > 200 else ''}")
        typer.echo("-" * 60)

if __name__ == "__main__":
    app()
