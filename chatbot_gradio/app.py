#!/usr/bin/env python3
"""
RAG Chat Application - Main Entry Point

This is the primary entry point for the RAG Chat system.
Run this file to start the chat interface with RAG capabilities.

Usage:
    python app.py
    
Or with UV:
    uv run python app.py

Environment Variables:
    See .env.example for configuration options
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    from config.config import (
        _app_config,
        PROJECT_ROOT,
        validate_environment
    )
    from core.database import connect_duckdb, close_connection
    from core.embeddings import initialize_embeddings, embedding_manager
    from core.llm_client import UniversalLLMClient
    from core.rag_pipeline import RAGPipeline
    from interface.chat_ui import launch_ui
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('chatbot_gradio.log', mode='a')]
    )
    
    # Suppress verbose logging from third-party libraries
    for module in ['sentence_transformers', 'urllib3', 'httpx']:
        logging.getLogger(module).setLevel(logging.WARNING)


def _initialize_components() -> Tuple[RAGPipeline, Optional[object]]:
    """Initialize all system components.
    
    Returns:
        Tuple of (RAGPipeline, database_connection)
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing system components...")
        
        # Initialize embedding model
        initialize_embeddings()
        
        # Initialize database connection
        db_conn = connect_duckdb(_app_config.duckdb_path)
        logger.info(f"Database connected: {_app_config.duckdb_path}")
        
        # Initialize LLM client
        provider_config = _app_config.get_active_provider_config()
        llm_client = UniversalLLMClient(
            api_key=provider_config["api_key"],
            base_url=provider_config["base_url"],
            model=provider_config["model"],
            timeout=_app_config.timeout,
            max_retries=_app_config.max_retries
        )
        
        # Test LLM connectivity
        if llm_client.test_connection():
            logger.info("LLM client initialized and tested successfully")
        else:
            logger.warning("LLM client initialized but connection test failed")
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            llm_client=llm_client,
            db_conn=db_conn,
            embedding_manager=embedding_manager,
            system_prompt=_app_config.system_prompt,
            top_k=_app_config.top_k
        )
        
        logger.info("All components initialized successfully")
        return rag_pipeline, db_conn
        
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        raise


def main() -> None:
    """Main application entry point."""
    logger = logging.getLogger(__name__)
    db_conn = None
    
    try:
        # Configure application logging
        setup_logging()
        
        print("🚀 Starting RAG Chat System...")
        logger.info("Starting RAG chat application...")
        
        # Ensure environment is properly configured
        try:
            validation_result = validate_environment()
            if not validation_result["valid"]:
                print("❌ Environment validation failed:")
                for error in validation_result["errors"]:
                    print(f"   - {error}")
                return
            
            available_providers = validation_result["available_providers"]
            if available_providers:
                print(f"🤖 Available providers: {', '.join(available_providers)}")
            else:
                print("❌ No LLM providers available")
                return
                
        except Exception as e:
            print(f"⚠️  Environment validation failed: {e}")
            return
        
        # Build RAG system components
        rag_pipeline, db_conn = _initialize_components()
        
        # Warn if environment file is missing
        env_file = PROJECT_ROOT / ".env"
        if not env_file.exists():
            print("⚠️  No .env file found. Using environment variables and defaults.")
        
        # Launch web interface
        config = _app_config
        print(f"🌐 Launching web interface at http://{config.app_host}:{config.app_port}")
        print("   Press Ctrl+C to stop the application")
        print()
        
        logger.info("Launching chat interface...")
        launch_ui(
            rag_pipeline=rag_pipeline,
            server_name=config.app_host,
            server_port=config.app_port,
            share=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        logger.info("Application interrupted by user")
    except Exception as e:
        print(f"❌ Application failed: {e}")
        logger.error(f"Application failed to start: {e}")
    finally:
        # Clean up database connection
        if db_conn:
            close_connection(db_conn)


if __name__ == "__main__":
    main()
