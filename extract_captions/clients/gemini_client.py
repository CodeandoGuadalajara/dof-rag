import os
from google import genai
from google.genai import types
import time
import threading
from typing import Dict, Any, Optional, Tuple

from .AbstractClient import AbstractAIClient

api_lock = threading.Lock()

class GeminiClient(AbstractAIClient):
    """
    Concrete implementation for interacting with Google's Gemini API.

    This class provides methods to process images using Google's Gemini models,
    handling authentication, API communication, and result processing. It 
    implements the AbstractAIClient interface to ensure consistency with
    other model implementations.
    
    Attributes:
        model (str): Gemini model identifier (e.g., "gemini-2.0-flash")
        max_tokens (int): Maximum token limit for generated responses
        temperature (float): Controls randomness in generation (0.0 to 1.0)
        top_p (float): Controls diversity via nucleus sampling (0.0 to 1.0)
        top_k (int): Controls diversity via vocabulary restriction
        response_mime_type (str): MIME type for the response format
        question (str): Prompt text to send with images
        api_key (str): Authentication key for Gemini API
        _client (genai.Client): Gemini client instance
        use_streaming (bool): Whether to use streaming API for content generation
    """
    def __init__(self, 
                 model: str = "gemini-2.0-flash", 
                 max_tokens: int = 256, 
                 temperature: float = 0.6,
                 top_p: float = 0.6,
                 top_k: int = 20,
                 response_mime_type: str = "text/plain",
                 api_key: Optional[str] = None,
                 use_streaming: bool = False):
        """
        Initializes the Gemini client configuration.

        Args:
            model: Gemini model identifier to use for image processing.
            max_tokens: Maximum number of tokens in the output response.
            temperature: Controls generation creativity (0.0 to 1.0).
            top_p: Top_p value for nucleus sampling (0.0 to 1.0).
            top_k: Top_k value for vocabulary restriction.
            response_mime_type: MIME type for the response format.
            api_key: API key to access Gemini. If not provided,
                     it's taken from the 'GEMINI_API_KEY' environment variable.
            use_streaming: Whether to use streaming API for content generation.
                     
        Raises:
            ValueError: If api_key is later found to be invalid during set_api_key.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.response_mime_type = response_mime_type
        self.question = "¿Qué se observa en esta imagen?, respóndelo en español, por favor."
        self.use_streaming = use_streaming
                
        if api_key:
            self.set_api_key(api_key)
        
    def set_question(self, question: str) -> None:
        """
        Configures the question (prompt) in Spanish that will be used for generation.

        Args:
            question: Question or prompt text in Spanish that will be sent to Gemini
                     along with the image.
        """
        self.question = question

    def set_api_key(self, api_key: str) -> None:
        """
        Updates the API key and reinitializes the Gemini client.

        Args:
            api_key: New API key for Google Gemini service.
            
        Raises:
            ValueError: If the provided API key is empty or invalid.
        """
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found. Please configure it.")
        self.api_key = api_key
        self._client = genai.Client(api_key=self.api_key)

    def process_image_to_bytes(self, image_path: str) -> Tuple[bytes, str]:
        """
        Load an image and convert it to the format needed by Gemini API.
        
        Args:
            image_path: Path of the image file to process.
            
        Returns:
            Tuple[bytes, str]: Tuple containing (image_bytes, mime_type) for use with Part.from_bytes().
            
        Raises:
            ValueError: If the image file cannot be processed.
        """
        # Verify the image exists and is readable
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        try:
            # Get the mime type based on file extension
            _, ext = os.path.splitext(image_path)
            mime_type = f"image/{ext[1:].lower()}"
            if ext.lower() in ('.jpg', '.jpeg'):
                mime_type = "image/jpeg"
            
            # Read the image file as bytes
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                
            return image_bytes, mime_type
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

    def process_imagen(self, image_path: str, force_overwrite: bool = False) -> Dict[str, Any]:
        """
        Processes an image using Google's Gemini models.
        
        If a description file already exists and force_overwrite is False,
        the already processed result is returned. Otherwise, the description
        is generated through the API, the output is validated, and if correct,
        a TXT file with the description is written in the same location as the image.
        
        Args:
            image_path: Path of the image file to process.
            force_overwrite: If True, the image is reprocessed even if a description exists.
        
        Returns:
            Dictionary with processing information including:
                - image_path: Path of the processed image
                - description: Generated description (if successful)
                - status: Processing status ("processed", "already_processed")
                - process_time: Time taken for processing (if applicable)
                - error: Error message (if an error occurred)
                - error_type: Type of error encountered (if applicable)
        """
        result: Dict[str, Any] = {"image_path": image_path}
        output_file = f"{os.path.splitext(image_path)[0]}.txt"

        # If the description already exists and reprocessing is not forced, return the result.
        if os.path.exists(output_file) and not force_overwrite:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content and not content.startswith("ERROR:"):
                        result["description"] = content
                        result["status"] = "already_processed"
                        return result
            except Exception:
                pass

        start = time.time()

        try:
            # Process the image to get bytes and mime type
            image_bytes, mime_type = self.process_image_to_bytes(image_path)
            
            # Prepare content parts
            contents = [
                # First part is the text prompt
                types.Part.from_text(text=self.question),
                # Second part is the image
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ]
            
            description = ""
            
            with api_lock:
                if self.use_streaming:
                    # Use streaming API
                    for chunk in self._client.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=self.get_generate_config()
                    ):
                        description += chunk.text
                else:
                    # Use non-streaming API
                    response = self._client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=self.get_generate_config()
                    )
                    description = response.text if hasattr(response, 'text') else str(response)
                    
            process_time = time.time() - start
            result["description"] = description
            result["process_time"] = process_time
            result["status"] = "processed"
        except Exception as e:
            error_msg = str(e)
            result["error"] = f"Error processing image with Gemini: {error_msg}"
            if "429" in error_msg:
                result["error_type"] = "rate_limit"
            elif "404" in error_msg:
                result["error_type"] = "not_found"
            else:
                result["error_type"] = "other"
            self.log_error(image_path)
            return result

        # Verify the generated output
        if not self.verify_output(result):
            return result

        # If there are no errors, create the file with the generated description.
        self.create_output_description(image_path, result["description"])

        return result

    def verify_output(self, result: Dict[str, Any]) -> bool:
        """
        Verifies that the generated output meets quality criteria.
        
        The verification ensures:
        - The description must not be empty.
        - No errors should have occurred during processing.
        
        If an error is found, the image path is logged for debugging.
        
        Args:
            result: Processing result dictionary to validate.
        
        Returns:
            True if the output is valid and meets quality standards, False otherwise.
        """
        if "error" in result:
            self.log_error(result["image_path"])
            return False
        if "description" not in result or not result["description"].strip():
            self.log_error(result["image_path"])
            return False
        return True

    def create_output_description(self, image_path: str, description: str) -> None:
        """
        Creates a TXT file with the description generated by the model.
        
        The file is created in the same path as the image, using the same base name
        but with a .txt extension.
        
        Args:
            image_path: Path of the processed image.
            description: Description text generated by the Gemini API.
            
        Note:
            Errors during file creation are logged but don't interrupt processing.
        """
        output_file = f"{os.path.splitext(image_path)[0]}.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(description)
        except Exception as e:
            # If there's an error writing the file, it's logged in the error log.
            self.log_error(image_path, extra_info=f"Error writing description: {str(e)}")

    def log_error(self, image_path: str, extra_info: str = "") -> None:
        """
        Logs the path of the image that produced an error for later analysis.
        
        Writes error information to the 'error_images.txt' file including the image path
        and optional additional details about the error.
        
        Args:
            image_path: Path of the image with error.
            extra_info: Additional information about the error.
            
        Note:
            If error logging itself fails, it's silently skipped to avoid
            interrupting the processing flow.
        """
        log_file = "error_images.txt"
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                log_entry = image_path
                if extra_info:
                    log_entry += f" | {extra_info}"
                f.write(log_entry + "\n")
        except Exception:
            # If error logging fails, it's skipped to avoid interrupting the flow.
            pass

    def get_generate_config(self) -> types.GenerateContentConfig:
        """
        Configures and returns the parameters for content generation through the API.
        
        Creates a configuration object with the client's current settings for
        temperature, top_p, etc.

        Returns:
            Configuration object for Gemini content generation.
        """
        return types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_tokens,
            response_mime_type=self.response_mime_type,
        )

    def get_client(self) -> genai.Client:
        """
        Returns the Gemini API client instance.

        Returns:
            Current Gemini client instance for direct API access if needed.
        """
        return self._client

    def update_config(self, 
                     max_tokens: Optional[int] = None, 
                     model: Optional[str] = None, 
                     temperature: Optional[float] = None, 
                     top_p: Optional[float] = None, 
                     top_k: Optional[int] = None, 
                     response_mime_type: Optional[str] = None,
                     use_streaming: Optional[bool] = None) -> 'GeminiClient':
        """
        Allows updating the client configuration parameters.

        Args:
            max_tokens: New maximum number of tokens for response generation.
            model: New Gemini model identifier to use.
            temperature: New temperature value for controlling randomness.
            top_p: New top_p value for nucleus sampling.
            top_k: New top_k value for vocabulary restriction.
            response_mime_type: New MIME type for the response format.
            use_streaming: Whether to use streaming API for content generation.

        Returns:
            The current instance to allow method chaining.
        """
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if top_k is not None:
            self.top_k = top_k
        if response_mime_type is not None:
            self.response_mime_type = response_mime_type
        if use_streaming is not None:
            self.use_streaming = use_streaming
            
        return self
