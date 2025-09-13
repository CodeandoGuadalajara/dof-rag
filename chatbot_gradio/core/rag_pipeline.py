"""RAG pipeline orchestrator for RAG chat system."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import duckdb
from config.config import CONTEXT_FORMAT_TEMPLATE, PROMPT_TEMPLATE, calculate_document_age, extract_date_from_title
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
            llm_client: Client for AI response generation
            db_conn: DuckDB connection for embeddings query
            embedding_manager: Text vectorization manager
            system_prompt: Base prompt template for LLM
            top_k: Default fragments to retrieve (default: 5)
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
            top_k: Override default fragment count (uses self.top_k if None)
            
        Returns:
            Tuple of (response_text, sources_list)
                
        Raises:
            ValueError: If user_question is empty
            Exception: If pipeline step fails
        """
        try:
            user_question = user_question.strip()
            if not user_question:
                raise ValueError("User question cannot be empty")
            
            effective_top_k = top_k or self.top_k
            
            # Execute pipeline steps
            fragments = self._retrieve_fragments(user_question, effective_top_k)
            grouped_fragments = self._group_fragments_by_document(fragments)
            document_context = self._build_context(grouped_fragments)
            response = self._generate_response(user_question, document_context)
            sources = self._prepare_sources(grouped_fragments)
            
            return response, sources
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            raise
    
    def _retrieve_fragments(self, user_question: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve and rerank fragments for the user question.
        
        Args:
            user_question: Question to search for fragments
            top_k: Number of fragments to retrieve
            
        Returns:
            List of fragment dicts with: id, text, title, similarity, temporal_score,
            document_id, header, url, file_path
        """
        # Convert user question to vector representation
        question_embedding = self.embedding_manager.encode_text(user_question, is_query=True)
        
        # Retrieve candidate fragments with extra buffer for reranking
        retrieval_top_k = min(top_k * 2, 20)
        fragments = query_context(self.db_conn, question_embedding, retrieval_top_k)
        
        # Apply temporal reranking and select final set
        fragments = self._apply_temporal_reranking(fragments)
        return fragments[:top_k]
    
    def _apply_temporal_reranking(self, fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply temporal reranking to prioritize recent content.
        
        Args:
            fragments: List of fragment dicts with similarity scores and titles
            
        Returns:
            Reranked fragments with temporal_score, sorted by combined similarity+recency
        """
        try:
            for fragment in fragments:
                title = fragment.get("title")
                if title:
                    try:
                        doc_date = extract_date_from_title(title)
                        age_info = calculate_document_age(doc_date)
                        recency_factor = age_info["recency_factor"]
                        
                        # Apply modest recency boost to preserve similarity-based ranking
                        original_similarity = fragment["similarity"]
                        boosted_similarity = original_similarity * (0.8 + 0.2 * recency_factor)
                        fragment["temporal_score"] = boosted_similarity
                    except Exception:
                        fragment["temporal_score"] = fragment["similarity"]
                else:
                    fragment["temporal_score"] = fragment["similarity"]
            
            # Rank by combined similarity and recency score
            fragments.sort(key=lambda x: x["temporal_score"], reverse=True)
            return fragments
            
        except Exception as e:
            logger.error(f"Failed to apply temporal reranking: {e}")
            return fragments  # Return original order on error

    def _group_fragments_by_document(self, fragments: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Group fragments by source document.
        
        Args:
            fragments: List of fragment dicts from retrieval
            
        Returns:
            Dict with document_id as key, containing chunks list and metadata dict.
            Chunks sorted by similarity (highest first)
        """
        try:
            grouped = defaultdict(lambda: {"chunks": [], "metadata": {}})
            
            for chunk in fragments:
                doc_id = chunk["document_id"]
                
                # Add chunk data
                grouped[doc_id]["chunks"].append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "header": chunk.get("header", ""),
                    "similarity": chunk.get("similarity", 0)
                })
                
                # Set document metadata (consistent across chunks)
                if not grouped[doc_id]["metadata"]:
                    grouped[doc_id]["metadata"] = {
                        "title": chunk.get("title", "Unknown Document"),
                        "url": chunk.get("url", ""),
                        "file_path": chunk.get("file_path", "")
                    }
            
            # Order chunks by relevance within each document
            for doc_data in grouped.values():
                doc_data["chunks"].sort(key=lambda x: x["similarity"], reverse=True)
            
            return dict(grouped)
            
        except Exception as e:
            logger.error(f"Failed to group chunks by document: {e}")
            raise
    
    def _get_temporal_metadata(self, title: str) -> Dict[str, str]:
        """Extract temporal metadata from document title.
        
        Args:
            title: Document title (format: DDMMYYYY-...)
            
        Returns:
            Dict with: publication_date, age_description, age_emoji, recency_label.
            Returns defaults if title empty or extraction fails
        """
        default_metadata = {
            "publication_date": "Fecha no disponible",
            "age_description": "N/A",
            "age_emoji": "❓",
            "recency_label": "Desconocida"
        }
        
        if not title:
            return default_metadata
        
        try:
            doc_date = extract_date_from_title(title)
            age_info = calculate_document_age(doc_date)
            return {
                "publication_date": age_info["formatted_date"],
                "age_description": age_info["age_description"],
                "age_emoji": age_info["emoji"],
                "recency_label": age_info["label"]
            }
        except Exception:
            return default_metadata
    
    def _format_chunk_content(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunk content for LLM context.
        
        Args:
            chunks: List of chunk dicts with text, header, similarity
            
        Returns:
            Formatted string with headers, fragment numbers, text, and separators
        """
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
        
        return "\n".join(content_parts)
    
    def _build_context(self, grouped_fragments: Dict[int, Dict[str, Any]]) -> str:
        """Build formatted context string from grouped fragments.
        
        Args:
            grouped_fragments: Dict of documents with chunks and metadata
            
        Returns:
            Formatted context string using CONTEXT_FORMAT_TEMPLATE
                
        Raises:
            Exception: If context building fails
        """
        try:
            context_parts = []
            
            for doc_num, (doc_id, doc_data) in enumerate(grouped_fragments.items(), 1):
                title = doc_data["metadata"]["title"]
                url = doc_data["metadata"].get("url", "N/A")
                file_path = doc_data["metadata"].get("file_path", "N/A")
                chunks = doc_data["chunks"]
                
                # Compute document relevance from chunk similarities
                avg_similarity = sum(chunk["similarity"] for chunk in chunks) / len(chunks)
                
                # Get temporal metadata
                temporal_metadata = self._get_temporal_metadata(title)
                
                # Format chunk content
                combined_content = self._format_chunk_content(chunks)
                
                # Apply standard context formatting template
                formatted_section = CONTEXT_FORMAT_TEMPLATE.format(
                    doc_num=doc_num,
                    title=title,
                    publication_date=temporal_metadata["publication_date"],
                    age_description=temporal_metadata["age_description"],
                    age_emoji=temporal_metadata["age_emoji"],
                    similarity=avg_similarity,
                    recency_label=temporal_metadata["recency_label"],
                    url=url,
                    file_path=file_path,
                    content=combined_content
                )
                
                context_parts.append(formatted_section)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to build context: {e}")
            raise
    
    def _generate_response(self, user_question: str, document_context: str) -> str:
        """Generate AI response using LLM with system prompt and context.
        
        Args:
            user_question: Original user question
            document_context: Formatted context from retrieved documents
            
        Returns:
            Generated AI response string
        """
        user_prompt = PROMPT_TEMPLATE.format(context=document_context, query=user_question)
        complete_prompt = f"{self.system_prompt}\n\n{user_prompt}"
        return self.llm_client.chat(complete_prompt)
    
    def _prepare_sources(self, grouped_fragments: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information for UI display.
        
        Args:
            grouped_fragments: Dict of documents with chunks and metadata
            
        Returns:
            List of source dicts with: document_id, title, url, file_path, chunks.
            Sorted by highest chunk relevance
                
        Raises:
            Exception: If source preparation fails
        """
        try:
            sources = []
            
            for doc_id, doc_data in grouped_fragments.items():
                source = {
                    "document_id": doc_id,
                    "title": doc_data["metadata"]["title"],
                    "url": doc_data["metadata"].get("url", ""),
                    "file_path": doc_data["metadata"].get("file_path", ""),
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