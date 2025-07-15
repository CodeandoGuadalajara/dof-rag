"""Universal LLM client for RAG chat system supporting multiple providers."""

import logging
import time
from typing import Any, Dict
import openai

logger = logging.getLogger(__name__)

class UniversalLLMClient:
    """Universal client for various LLM providers using OpenAI-compatible API."""
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str, 
        model: str,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """Initialize the LLM client.
        
        Args:
            api_key: API key for the LLM provider
            base_url: Base URL for the API endpoint
            model: Model name to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def chat(self, prompt: str) -> str:
        """Send a chat completion request to the LLM.
        
        Args:
            prompt: The prompt/message to send to the LLM
            
        Returns:
            The response text from the LLM
            
        Raises:
            Exception: If the request fails after all retries
        """
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout=self.timeout
                )
                
                response_text = self._handle_response(response)
                return response_text
                
            except openai.APITimeoutError as e:
                logger.warning(f"Request timeout (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"Request timed out after {self.max_retries} attempts")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except openai.APIError as e:
                logger.error(f"API error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"API error after {self.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"Request failed after {self.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
    
    def _handle_response(self, response) -> str:
        """Extract text from the LLM response.
        
        Args:
            response: Raw response from the LLM API
            
        Returns:
            Extracted response text
            
        Raises:
            Exception: If response format is invalid
        """
        try:
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except (AttributeError, IndexError) as e:
            logger.error(f"Invalid response format: {e}")
            raise Exception("Invalid response format: no content found")
    
    def test_connection(self) -> bool:
        """Test the connection to the LLM provider.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_prompt = "Hello, this is a test message. Please respond with 'OK'."
            response = self.chat(test_prompt)
            logger.info(f"Connection test successful: {response[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }