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

# Avatar configuration for better UX - Using web images
USER_AVATAR = "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/svg/1f468-200d-1f4bb.svg"  # User with laptop
BOT_AVATAR = "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/svg/1f916.svg"  # Robot face

# Example questions for better UX
EXAMPLE_QUESTIONS = [
    ["¿Cuáles son los temas principales cubiertos en los documentos?"],
    ["¿Puedes explicar los conceptos clave mencionados?"],
    ["¿Qué metodología se discute en las fuentes?"],
    ["Resume los hallazgos más importantes."]
]


def _create_simplified_theme() -> gr.themes.Soft:
    """Create a simplified theme with essential overrides only."""
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate"
    ).set(
        # Only essential overrides for better contrast
        background_fill_secondary="#f0f4f8",
        background_fill_secondary_dark="#2d3748",
        border_color_primary="#cbd5e0",
        border_color_primary_dark="#4a5568"
    )


def _handle_error(message: str, error: Exception) -> Tuple:
    """Handle errors consistently and add to chat history."""
    error_msg = f"Error al procesar tu pregunta: {str(error)}"
    embedded_chat_history.add_user_message(message)
    embedded_chat_history.add_assistant_response(error_msg, [], f"error_{embedded_chat_history.query_counter}")
    return embedded_chat_history.get_formatted_history(), ""


def create_chat_interface(rag_pipeline: RAGPipeline) -> gr.Blocks:
    """Create the main chat interface using Gradio Blocks with horizontal layout.
    
    Args:
        rag_pipeline: Configured RAG pipeline instance
        
    Returns:
        Configured Gradio Blocks interface
    """
    # Use simplified theme
    custom_theme = _create_simplified_theme()
    
    # Custom styles are loaded from external CSS file for better maintainability
    with gr.Blocks(
        title=APP_TITLE,
        theme=custom_theme,
        fill_height=True,
        css="/gradio_api/file=chat_styles.css"
    ) as interface:
        
        # Title section (fixed height)
        with gr.Row():
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
            avatar_images=(USER_AVATAR, BOT_AVATAR),
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
                    examples=EXAMPLE_QUESTIONS,
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
                return _handle_error(message, e)
        
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