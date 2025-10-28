import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import numpy as np

    # Database and embedding imports
    import duckdb
    from sentence_transformers import SentenceTransformer
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from google.genai import errors

    _ = load_dotenv()

    return SentenceTransformer, duckdb, mo, np, os, genai, types, errors


@app.cell
def _(SentenceTransformer, duckdb):
    # Initialize the same model and database as in extract_embeddings.py
    model = SentenceTransformer(
        "nomic-ai/modernbert-embed-base", trust_remote_code=True
    )

    # Connect to the existing DuckDB vector database with VSS extension
    db = duckdb.connect("dof_db/db.duckdb")
    db.execute("INSTALL vss")
    db.execute("LOAD vss")

    return db, model


@app.cell
def _(mo):
    """UI for Gemini API Key input with instructions."""
    
    mo.md(
        """
        ## 🔑 Configuración de API Key de Gemini
        
        Para usar este chat, necesitas una API key de Google Gemini.
        
        **¿Cómo obtener tu API key?**
        1. Visita [Google AI Studio](https://aistudio.google.com/app/apikey)
        2. Inicia sesión con tu cuenta de Google
        3. Haz clic en "Get API Key" o "Create API Key"
        4. Copia la clave y pégala aquí abajo
        
        **Nota:** Tu API key se mantiene en tu sesión actual y NO se guarda en ninguna base de datos.
        """
    )


@app.cell
def _(mo):
    """Text input for API key."""
    api_key_input = mo.ui.text(
        label="Gemini API Key",
        placeholder="Pega tu API key aquí...",
        kind="password",
        full_width=True
    )
    api_key_input


@app.cell
def _(genai, api_key_input):
    """Initialize Gemini client with user-provided API key."""
    
    model_id = "gemini-2.5-flash"
    
    if not api_key_input.value or api_key_input.value.strip() == "":
        client = None
    else:
        try:
            client = genai.Client(api_key=api_key_input.value.strip())
            print("✅ Gemini client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Gemini client: {e}")
            client = None

    return client, model_id


@app.cell
def _(db, model, np):
    def search_similar_chunks(query: str, limit: int = 5):
        """
        Search for similar chunks in the DuckDB vector database using pure semantic similarity.
        Results are ordered by vector distance without temporal reranking or document grouping.
        """
        try:
            # Generate embedding for the query with search prefix (like chatbot_gradio)
            query_with_prefix = f"search_document: {query}"
            query_embedding = model.encode(query_with_prefix)

            # Convert numpy array to list for DuckDB compatibility
            query_embedding_list = query_embedding.tolist()

            # Vector search using optimized array_distance
            search_sql = """
            SELECT 
                c.id,
                c.text,
                c.header,
                c.document_id,
                d.title,
                d.url,
                d.file_path,
                d.created_at,
                array_distance(c.embedding, ?::FLOAT[768]) as distance_score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY distance_score ASC
            LIMIT ?
            """

            results = db.execute(search_sql, [query_embedding_list, limit]).fetchall()

            # Convert results to list of dictionaries
            search_results = [
                {
                    "id": row[0],
                    "text": row[1],
                    "header": row[2],
                    "document_id": row[3],
                    "document_title": row[4],
                    "url": row[5],
                    "file_path": row[6],
                    "created_at": row[7],
                    "distance_score": row[8],
                }
                for row in results
            ]
            
            return search_results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    return (search_similar_chunks,)


@app.cell
def _(mo):
    # Display title and description
    mo.md(
        """
        # Chat with DOF Database

        **Intelligent Assistant for Official Gazette Queries**

        **Instructions:**
        1. Write your question in the chat
        2. The system will search the vector database
        3. You will receive answers with relevant sources and links
        """
    )


@app.cell
def _(search_similar_chunks, client, model_id, mo, types, errors):
    def rag_model(messages, config) -> str:
        """RAG model with Gemini 2.5-flash and collapsible sources.

        Args:
            messages: List of chat messages
            config: Marimo chat configuration (contains temperature, max_tokens,
                   top_p, top_k, frequency_penalty, presence_penalty, layout, etc.)

        Returns:
            str: Model response with sources in Markdown format
        """

        if not messages:
            return "¡Hola! Soy tu asistente para consultar documentos del Diario Oficial de la Federación (DOF). ¿En qué puedo ayudarte?"

        # Get the last message and validate it has content
        last_message = messages[-1]

        latest_message = last_message.content.strip()

        search_results = search_similar_chunks(latest_message, limit=3)

        context_chunks, fuentes_md = [], []
        for i, res in enumerate(search_results, 1):
            # Validate and sanitize fields that may be None
            doc_title = res.get("document_title") or "Sin título"
            header = res.get("header") or "Sin sección"
            text_content = res.get("text") or "Sin contenido"
            url = res.get("url") or "Sin URL"
            distance = res.get("distance_score", 0)

            context_chunks.append(
                f"Documento: {doc_title}\nSección: {header}\nContenido: {text_content}"
            )
            fuentes_md.append(
                f"**Fuente {i}** (Distancia: {distance:.4f})  \n"
                f"📄 **Documento:** {doc_title}  \n"
                f"📋 **Sección:** {header}  \n"
                f"🔗 **URL:** {url}"
            )

        context_text = "\n\n".join(context_chunks)

        # Verify if Gemini client is available
        if client is None:
            # Fallback mode: manual response without LLM
            answer = (
                f"**Información encontrada sobre '{latest_message}':**\n\n"
                f"He encontrado {len(search_results)} documentos relevantes en la base de datos del DOF. "
                f"Aquí tienes un resumen basado en los documentos más similares:\n\n"
                f"**Contexto principal:**\n{context_text[:500]}...\n\n"
                f"⚠️ **Nota:** Respuesta generada automáticamente. Para respuestas más elaboradas, "
                f"configura GEMINI_API_KEY en tu archivo .env"
            )
        else:
            # Normal mode: use Gemini to generate response
            prompt = (
                "Responde a la pregunta usando exclusivamente la información provista en el Contexto. "
                "Si la respuesta no se encuentra allí, indícalo explícitamente. "
                "Responde en español y en formato Markdown.\n\n"
                f"Contexto:\n{context_text}\n\n"
                f"Pregunta: {latest_message}\n\nRespuesta:"
            )

            try:
                # Hardcoded configuration values for Gemini
                gemini_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=-1),
                    response_mime_type="text/plain",
                    temperature=0.5,
                    max_output_tokens=4096,
                    top_p=0.9,
                    top_k=40,
                )

                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=prompt)]
                    )
                ]
                resp = client.models.generate_content(
                    model=model_id, contents=contents, config=gemini_config
                )

                # Handle response according to finish_reason
                if resp.text:
                    answer = resp.text
                elif resp.candidates and len(resp.candidates) > 0:
                    # Access partial response when MAX_TOKENS is reached
                    candidate = resp.candidates[0]
                    if candidate.content and candidate.content.parts:
                        # Get text from first part
                        answer = "".join(
                            part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text
                        )
                        if not answer:
                            answer = "No se pudo generar una respuesta."
                    else:
                        answer = "No se pudo generar una respuesta."
                else:
                    answer = "No se pudo generar una respuesta."

            except errors.APIError as e:
                answer = f"⚠️ Error de API: {e.message}\n\n**Información encontrada (modo fallback):**\n{context_text[:500]}..."
            except Exception as e:
                answer = f"⚠️ Error inesperado: {str(e)}\n\n**Información encontrada (modo fallback):**\n{context_text[:500]}..."

        fuentes_md_block = "\n\n".join(fuentes_md)

        # Ensure answer is string
        answer = str(answer)

        # Get layout from config or use default value
        layout = getattr(config, "layout", "details") if config else "details"

        if layout == "accordion":
            fuentes_widget = mo.accordion(
                {f"Fuentes ({len(search_results)})": mo.md(fuentes_md_block)}
            )
            return mo.vstack([mo.md(answer), fuentes_widget])

        # Default layout: details
        details_block = (
            f"\n/// details | Fuentes ({len(search_results)})\n{fuentes_md_block}\n///"
        )
        return mo.md(answer + details_block)

    return (rag_model,)


@app.cell
def _(mo, rag_model):
    """Chat interface with DOF database."""
    
    # Create and display the chat interface
    mo.ui.chat(
        rag_model,
        prompts=[
            "¿Qué información hay sobre regulaciones ambientales?",
            "Buscar decretos sobre {{tema}}",
            "¿Cuáles son las últimas modificaciones en {{área_legal}}?",
            "Información sobre impuestos y contribuciones",
            "Regulaciones sobre salud pública",
            "Normativas de educación",
            "¿Qué dice sobre {{concepto_específico}}?",
        ],
        show_configuration_controls=False,
    )


if __name__ == "__main__":
    app.run()