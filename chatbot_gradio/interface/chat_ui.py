# -*- coding: utf-8 -*-
"""Modern Gradio chat interface for RAG chat system.

This module provides a clean, modern ChatInterface implementation using
Gradio Blocks with horizontal layout and responsive design.
"""

import logging
from typing import Dict, List, Tuple

import gradio as gr

from core.rag_pipeline import RAGPipeline
from interface.embedded_history import embedded_chat_history

logger = logging.getLogger(__name__)

# UI Configuration
APP_TITLE = "Chat RAG - Sistema de Consulta Avanzado"
APP_DESCRIPTION = """
🤖 **Sistema de Chat Inteligente con Tecnología RAG**

Haz preguntas y obtén respuestas respaldadas por fuentes documentales. 
El sistema utiliza técnicas avanzadas de recuperación para proporcionar respuestas precisas 
y contextuales basadas en tu base de conocimientos de documentos.
"""
CHAT_PLACEHOLDER = "Escribe tu pregunta aquí..."
MAX_INPUT_LENGTH = 2000


def create_chat_interface(rag_pipeline: RAGPipeline) -> gr.Blocks:
    """Create the main chat interface using Gradio Blocks with horizontal layout.
    
    Args:
        rag_pipeline: Configured RAG pipeline instance
        
    Returns:
        Configured Gradio Blocks interface
    """
    # Custom styles are loaded from external CSS file for better maintainability
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate"
        ),
        fill_height=True,
        css="/gradio_api/file=chat_styles.css"
    ) as interface:
        
        # Title section (fixed height)
        with gr.Row(elem_classes=["header-section"]):
            with gr.Column():
                gr.Markdown(f"# {APP_TITLE}")
                gr.Markdown(APP_DESCRIPTION)
        
        # Chat interface (dynamic height - main area)
        chatbot = gr.Chatbot(
            value=[],
            height=450,
            elem_classes=["custom-chatbot"],
            label="Conversación",
            show_label=False,
            type="messages",
            show_copy_button=True,
            avatar_images=("👤", "🤖"),
            elem_id="main_chatbot"
        )
        
        # Input and configuration row (fixed height)
        with gr.Row(elem_classes=["input-config-row"]):
            with gr.Column(scale=6):
                chat_input = gr.Textbox(
                    placeholder=CHAT_PLACEHOLDER,
                    label="Mensaje",
                    show_label=False,
                    lines=2,
                    max_lines=4
                )
            with gr.Column(scale=2):
                submit_btn = gr.Button("Enviar", variant="primary")
                clear_btn = gr.Button("Limpiar", variant="secondary")
            with gr.Column(scale=2):
                max_sources_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Máx. fuentes",
                    interactive=True
                )
        
        # Examples section (fixed height)
        with gr.Row(elem_classes=["examples-section"]):
            with gr.Column():
                gr.Markdown("### 💡 Ejemplos de preguntas:")
                gr.Examples(
                    examples=[
                        ["¿Cuáles son los temas principales cubiertos en los documentos?"],
                        ["¿Puedes explicar los conceptos clave mencionados?"],
                        ["¿Qué metodología se discute en las fuentes?"],
                        ["Resume los hallazgos más importantes."]
                    ],
                    inputs=chat_input,
                    label=""
                )
        
        # Event handlers
        def process_message(message: str, history: List[Dict[str, str]], max_sources: int) -> Tuple:
            """Process user message and return updated components."""
            message = message.strip()
            if not message:
                return history, ""
            
            try:
                # Validate and truncate input length
                if len(message) > MAX_INPUT_LENGTH:
                    message = message[:MAX_INPUT_LENGTH]
                
                # Add user message to embedded history and get query ID
                query_id = embedded_chat_history.add_user_message(message)
                
                # Get response from RAG pipeline with dynamic top_k
                response, sources = rag_pipeline.query(message, top_k=max_sources)
                
                # Add assistant response with sources to embedded history
                embedded_chat_history.add_assistant_response(response, sources, query_id)
                
                # Get formatted history for display (now includes embedded sources)
                updated_history = embedded_chat_history.get_formatted_history()
                
                return updated_history, ""
                
            except Exception as e:
                error_msg = f"Error al procesar tu pregunta: {str(e)}"
                # Add error to history
                embedded_chat_history.add_user_message(message)
                embedded_chat_history.add_assistant_response(error_msg, [], f"error_{embedded_chat_history.query_counter}")
                updated_history = embedded_chat_history.get_formatted_history()
                
                return updated_history, ""
        
        def clear_chat() -> Tuple:
            """Clear chat history and reset interface."""
            embedded_chat_history.clear_history()
            return [], ""
        
        # Wire up events with scroll to bottom
        submit_btn.click(
            fn=process_message,
            inputs=[chat_input, chatbot, max_sources_slider],
            outputs=[chatbot, chat_input]
        )
        
        chat_input.submit(
            fn=process_message,
            inputs=[chat_input, chatbot, max_sources_slider],
            outputs=[chatbot, chat_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, chat_input]
        )
    
    return interface


def launch_ui(
    rag_pipeline: RAGPipeline,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False
) -> None:
    """Launch the Gradio chat interface.
    
    Args:
        rag_pipeline: Configured RAG pipeline instance
        server_name: Server host name
        server_port: Server port number
        share: Whether to create a public link
    """
    try:
        # Create the interface
        interface = create_chat_interface(rag_pipeline)
        
        # Launch server
        logger.info(f"Starting server on {server_name}:{server_port}")
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_error=True,
            inbrowser=True
        )
        
    except Exception as e:
        logger.error(f"Failed to launch UI: {e}")
        raise