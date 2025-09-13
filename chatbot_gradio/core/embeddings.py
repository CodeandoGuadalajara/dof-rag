"""Embedding generation for RAG chat system using SentenceTransformers."""

import logging
from typing import List, Optional

import numpy as np
from config.config import _app_config
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Simple embedding manager using SentenceTransformers."""
    
    def __init__(self) -> None:
        """Initialize empty manager."""
        self._embedder: Optional[SentenceTransformer] = None
        self._query_prefix: str = ""
        self._document_prefix: str = ""
    
    def initialize(self, model_name: str, query_prefix: str = "", document_prefix: str = "") -> None:
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model to load
            query_prefix: Prefix to add to query texts
            document_prefix: Prefix to add to document texts
        """
        self._embedder = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        logger.info(f"Loaded embedding model: {model_name}")
    
    def encode_text(self, text: str, is_query: bool = False) -> List[float]:
        """Generate embedding for a text string.
        
        Args:
            text: Input text to encode
            is_query: Whether this is a query (applies query prefix) or document text
            
        Returns:
            List of float values representing the text embedding
        """
        if self._embedder is None:
            raise RuntimeError("Embedding model not initialized. Call initialize() first.")
        
        # Apply appropriate prefix
        if is_query and self._query_prefix:
            text = f"{self._query_prefix}{text}"
        elif not is_query and self._document_prefix:
            text = f"{self._document_prefix}{text}"
        
        # Generate embedding and convert to list
        embedding = self._embedder.encode(text, convert_to_tensor=False)
        return embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
    
# Global instance
embedding_manager = EmbeddingManager()

def initialize_embeddings() -> None:
    """Initialize embedding manager with configuration."""
    embedding_manager.initialize(
        model_name=_app_config.embeddings_model,
        query_prefix=_app_config.query_prefix,
        document_prefix=_app_config.document_prefix
    )