"""RAG pipeline orchestrator for RAG chat system."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import duckdb
from config.config import CONTEXT_FORMAT_TEMPLATE, PROMPT_TEMPLATE, calculate_document_age
from .database import query_context
from .embeddings import EmbeddingManager
from .llm_client import UniversalLLMClient


logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates embedding generation, retrieval, and response generation."""
    
    def __init__(
        self,
        llm_client: UniversalLLMClient,
        db_conn: duckdb.DuckDBPyConnection,
        embedding_manager: EmbeddingManager,
        system_prompt: str,
        top_k: int = 5
    ):
        """Initialize RAG pipeline components.
        
        Args:
            llm_client: Response generation client
            db_conn: Database connection for retrieval
            embedding_manager: Text embedding generator
            system_prompt: Base prompt template
            top_k: Default number of fragments to retrieve
        """
        self.llm_client = llm_client
        self.db_conn = db_conn
        self.embedding_manager = embedding_manager
        self.system_prompt = system_prompt
        self.top_k = top_k
    
    def query(self, user_question: str, top_k: int = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Execute complete RAG pipeline for user question.
        
        Args:
            user_question: User's input question
            top_k: Override default fragment count
            
        Returns:
            Tuple of (response_text, sources_list)
            
        Raises:
            Exception: If any step in the pipeline fails
        """
        try:
            user_question = user_question.strip()
            if not user_question:
                raise ValueError("User question cannot be empty")
            
            effective_top_k = top_k or self.top_k
            
            # Convert user question to vector representation
            question_embedding = self.embedding_manager.encode_text(user_question, is_query=True)
            
            # Retrieve candidate fragments with extra buffer for reranking
            retrieval_top_k = min(effective_top_k * 2, 20)
            fragments = query_context(self.db_conn, question_embedding, retrieval_top_k)
            
            # Boost relevance of recent documents while preserving similarity
            fragments = self._apply_temporal_reranking(fragments)
            
            # Select final fragment set after reranking
            fragments = fragments[:effective_top_k]
            
            # Organize fragments by source document for context building
            grouped_fragments = self._group_fragments_by_document(fragments)
            
            # Construct formatted context from retrieved fragments
            document_context = self._build_context(grouped_fragments)
            
            # Generate LLM response with system prompt and context
            user_prompt = PROMPT_TEMPLATE.format(context=document_context, query=user_question)
            complete_prompt = f"{self.system_prompt}\n\n{user_prompt}"
            response = self.llm_client.chat(complete_prompt)
            
            # Format source information for frontend display
            sources = self._prepare_sources(grouped_fragments)
            
            return response, sources
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            raise
    
    def _apply_temporal_reranking(self, fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply temporal reranking to prioritize recent content while preserving relevance.
        
        Args:
            fragments: List of retrieved fragments
            
        Returns:
            Reranked list of fragments
        """
        try:
            for fragment in fragments:
                created_at = fragment.get("created_at")
                if created_at:
                    age_info = calculate_document_age(created_at)
                    recency_factor = age_info["recency_factor"]
                    
                    # Apply modest recency boost to preserve similarity-based ranking
                    original_similarity = fragment["similarity"]
                    boosted_similarity = original_similarity * (0.8 + 0.2 * recency_factor)
                    fragment["temporal_score"] = boosted_similarity
                else:
                    # Fallback to semantic similarity when date unavailable
                    fragment["temporal_score"] = fragment["similarity"]
            
            # Rank by combined similarity and recency score
            fragments.sort(key=lambda x: x["temporal_score"], reverse=True)
            
            return fragments
            
        except Exception as e:
            logger.error(f"Failed to apply temporal reranking: {e}")
            return fragments  # Return original order on error

    def _group_fragments_by_document(
        self, 
        fragments: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """Group chunks by source document.
        
        Args:
            fragments: Retrieved chunk data
            
        Returns:
            Dictionary grouped by document_id with document info and chunks
        """
        try:
            grouped = defaultdict(lambda: {"chunks": [], "metadata": {}})
            
            for chunk in fragments:
                doc_id = chunk["document_id"]
                
                grouped[doc_id]["chunks"].append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "header": chunk.get("header", ""),
                    "similarity": chunk.get("similarity", 0)
                })
                
                # Document metadata is consistent across chunks
                grouped[doc_id]["metadata"] = {
                    "title": chunk.get("title", "Unknown Document"),
                    "url": chunk.get("url", ""),
                    "file_path": chunk.get("file_path", ""),
                    "created_at": chunk.get("created_at")
                }
            
            # Order chunks by relevance within each document
            for doc_data in grouped.values():
                doc_data["chunks"].sort(
                    key=lambda x: x["similarity"], 
                    reverse=True
                )
            
            return dict(grouped)
            
        except Exception as e:
            logger.error(f"Failed to group chunks by document: {e}")
            raise
    
    def _build_context(
        self, 
        grouped_fragments: Dict[int, Dict[str, Any]]
    ) -> str:
        """Build the document context string from grouped chunks using template.
        
        Args:
            grouped_fragments: Chunks grouped by document
            
        Returns:
            Formatted context string for the LLM
        """
        try:
            context_parts = []
            
            for doc_num, (doc_id, doc_data) in enumerate(grouped_fragments.items(), 1):
                title = doc_data["metadata"]["title"]
                url = doc_data["metadata"].get("url", "N/A")
                file_path = doc_data["metadata"].get("file_path", "N/A")
                created_at = doc_data["metadata"].get("created_at")
                chunks = doc_data["chunks"]
                
                # Compute document relevance from chunk similarities
                avg_similarity = sum(chunk["similarity"] for chunk in chunks) / len(chunks)
                
                # Generate temporal metadata for context display
                age_info = {}
                if created_at:
                    age_info = calculate_document_age(created_at)
                    publication_date = age_info["formatted_date"]
                    age_description = age_info["age_description"]
                    age_emoji = age_info["emoji"]
                    recency_label = age_info["label"]
                else:
                    publication_date = "Fecha no disponible"
                    age_description = "N/A"
                    age_emoji = "❓"
                    recency_label = "Desconocida"
                
                # Combine all chunk texts for this document
                content_parts = []
                for i, chunk in enumerate(chunks, 1):
                    text = chunk["text"]
                    header = chunk.get("header", "")
                    similarity = chunk["similarity"]
                    
                    # Include document section headers for better context
                    if header:
                        content_parts.append(f"--- {header} ---")
                    
                    content_parts.append(f"Fragmento {i} (similitud: {similarity:.3f}):")
                    content_parts.append(text)
                    content_parts.append("")  # Empty line between chunks
                
                combined_content = "\n".join(content_parts)
                
                # Apply standard context formatting template
                formatted_section = CONTEXT_FORMAT_TEMPLATE.format(
                    doc_num=doc_num,
                    title=title,
                    publication_date=publication_date,
                    age_description=age_description,
                    age_emoji=age_emoji,
                    similarity=avg_similarity,
                    recency_label=recency_label,
                    url=url,
                    file_path=file_path,
                    content=combined_content
                )
                
                context_parts.append(formatted_section)
            
            context = "\n".join(context_parts)
            return context
            
        except Exception as e:
            logger.error(f"Failed to build context: {e}")
            raise
    
    def _prepare_sources(
        self, 
        grouped_fragments: Dict[int, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare sources information for the UI.
        
        Args:
            grouped_fragments: Chunks grouped by document
            
        Returns:
            List of source documents with their chunks
        """
        try:
            sources = []
            
            for doc_id, doc_data in grouped_fragments.items():
                source = {
                    "document_id": doc_id,
                    "title": doc_data["metadata"]["title"],
                    "url": doc_data["metadata"].get("url", ""),
                    "file_path": doc_data["metadata"].get("file_path", ""),
                    "created_at": doc_data["metadata"].get("created_at"),
                    "chunks": doc_data["chunks"]
                }
                sources.append(source)
            
            # Order sources by highest chunk relevance
            sources.sort(
                key=lambda x: max(f["similarity"] for f in x["chunks"]),
                reverse=True
            )
            
            return sources
            
        except Exception as e:
            logger.error(f"Failed to prepare sources: {e}")
            raise