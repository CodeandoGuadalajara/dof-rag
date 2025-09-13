# -*- coding: utf-8 -*-
"""Enhanced chat history management with embedded sources.

This module handles chat history with integrated source context,
maintaining a rolling history of up to 10 query-response pairs
with their associated sources.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Configuration
MAX_HISTORY_ENTRIES = 10
QUERY_ID_PREFIX = "q"


class EmbeddedChatHistory:
    """Manages chat history with embedded sources for each query-response pair."""
    
    def __init__(self):
        """Initialize the chat history manager."""
        self.history: List[Dict[str, Any]] = []
        self.query_counter = 0
    
    def add_user_message(self, message: str) -> str:
        """Add a user message and return query ID.
        
        Args:
            message: User's question
            
        Returns:
            Query ID for this interaction
        """
        self.query_counter += 1
        query_id = f"{QUERY_ID_PREFIX}{self.query_counter}"
        
        user_entry = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now(),
            "query_id": query_id
        }
        
        self.history.append(user_entry)
        self._trim_history()
        
        logger.info(f"Added user message with query_id: {query_id}")
        return query_id
    
    def add_assistant_response(
        self, 
        response: str, 
        sources: List[Dict[str, Any]], 
        query_id: str
    ) -> None:
        """Add assistant response with embedded sources.
        
        Args:
            response: Assistant's response text
            sources: List of source documents used
            query_id: ID linking to the user question
        """
        assistant_entry = {
            "role": "assistant",
            "content": response,
            "sources": sources,
            "timestamp": datetime.now(),
            "query_id": query_id
        }
        
        self.history.append(assistant_entry)
        self._trim_history()
        
        logger.info(f"Added assistant response for query_id: {query_id} with {len(sources)} sources")
    
    def get_formatted_history(self) -> List[Dict[str, str]]:
        """Get history formatted for Gradio chatbot component.
        
        Returns:
            List of message dictionaries for Gradio
        """
        formatted_history = []
        
        for entry in self.history:
            if entry["role"] == "user":
                formatted_history.append({
                    "role": "user",
                    "content": entry["content"]
                })
            elif entry["role"] == "assistant":
                # For assistant messages, embed sources in the content
                content_with_sources = self._format_response_with_sources(
                    entry["content"],
                    entry.get("sources", []),
                    entry["query_id"]
                )
                formatted_history.append({
                    "role": "assistant", 
                    "content": content_with_sources
                })
        
        return formatted_history
    
    def clear_history(self) -> None:
        """Clear all chat history."""
        self.history.clear()
        self.query_counter = 0
        logger.info("Chat history cleared")
    
    def get_sources_for_query(self, query_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get sources for a specific query ID.
        
        Args:
            query_id: Query identifier
            
        Returns:
            List of sources or None if not found
        """
        for entry in self.history:
            if entry.get("query_id") == query_id and entry["role"] == "assistant":
                return entry.get("sources", [])
        return None
    
    def _trim_history(self) -> None:
        """Trim history to maintain maximum entries limit."""
        while len(self.history) > MAX_HISTORY_ENTRIES * 2:  # *2 for user+assistant pairs
            removed = self.history.pop(0)
            logger.debug(f"Removed old history entry: {removed.get('query_id', 'unknown')}")
    
    def _format_response_with_sources(
        self, 
        response: str, 
        sources: List[Dict[str, Any]], 
        query_id: str
    ) -> str:
        """Format assistant response with embedded sources.
        
        Args:
            response: Original assistant response
            sources: Associated source documents
            query_id: Query identifier for visual connection
            
        Returns:
            HTML formatted response with embedded sources
        """
        from interface.render_context import render_embedded_sources
        
        if not sources:
            return response
        
        # Create embedded sources HTML
        sources_html = render_embedded_sources(sources, query_id)
        
        # Combine response with sources (simplified format for Gradio)
        formatted_content = f"""{response}

{sources_html}"""
        
        return formatted_content

# Global instance for the application
embedded_chat_history = EmbeddedChatHistory()
