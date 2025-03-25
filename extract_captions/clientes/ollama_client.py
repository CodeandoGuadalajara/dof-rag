import os
import time
from PIL import Image
import threading
import base64
import mimetypes
import llm
from llm import UnknownModelError
from typing import Dict, Any, Optional, Union, Tuple, List

from .AbstractClient import AbstractAIClient

api_lock = threading.Lock()

class LocalImageAttachment:
    """
    Class to wrap a local image file and provide its content in base64,
    as well as resolve its MIME type for API consumption.
    
    Attributes:
        file_path (str): Path to the local image file.
    """
    def __init__(self, file_path: str):
        """
        Initialize with the path to a local image file.
        
        Args:
            file_path: Absolute or relative path to an image file.
        """
        self.file_path = file_path

    def base64_content(self) -> str:
        """
        Read the file and return its content encoded in base64.
        
        Returns:
            Base64 encoded content of the file as a string.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            IOError: If there's an error reading the file.
        """
        with open(self.file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def resolve_type(self) -> str:
        """
        Determine the MIME type of the file based on its extension.
        
        Returns:
            MIME type string, or "application/octet-stream" if unknown.
        """
        mime_type, _ = mimetypes.guess_type(self.file_path)
        return mime_type or "application/octet-stream"

class OllamaClient(AbstractAIClient):
    """
    Concrete implementation for interacting with local Ollama models.

    This class manages connections to locally-hosted Ollama models,
    providing methods to generate image descriptions through the LLM package
    interface. It implements the AbstractAIClient interface for
    compatibility with other model implementations.
    
    Attributes:
        model_name (str): Name of the Ollama model to use
        max_tokens (int): Maximum token length for generated responses
        temperature (float): Controls randomness in generation
        top_p (float): Controls diversity via nucleus sampling
        top_k (int): Controls diversity via vocabulary restriction
        num_ctx (int): Context window size for the model
        question (str): Prompt text to send with images
        model (llm.Model): The loaded Ollama model instance
        model_lock (threading.Lock): Lock for thread-safe model access
    """
    def __init__(self, 
                 model: str = "gemma3:4b", 
                 max_tokens: int = 512, 
                 temperature: float = 0.5,
                 top_p: float = 0.5,
                 top_k: int = 20,
                 num_ctx: int = 8192,
                 api_key: Optional[str] = None):
        """
        Initializes the Ollama client configuration.

        Args:
            model: Ollama model identifier to use (e.g., "gemma3:4b").
            max_tokens: Maximum number of tokens in the output response.
            temperature: Controls generation creativity (0.0 to 1.0).
            top_p: Top_p value for nucleus sampling (0.0 to 1.0).
            top_k: Top_k value for vocabulary restriction.
            num_ctx: Context window size for the model.
            api_key: Not used for Ollama, but kept for compatibility.
            
        Raises:
            ValueError: If the specified model cannot be loaded.
        """
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_ctx = num_ctx
        self.question = "¿Qué se observa en esta imagen?, respóndelo en español, por favor."
        
        # Try to load the model
        try:
            self.model = llm.get_model(model)
        except UnknownModelError as e:
            raise ValueError(f"Error getting model {model}: {e}")
        
        # Lock to protect model access in multi-threaded environments
        self.model_lock = threading.Lock()
        
    def set_question(self, question: str) -> None:
        """
        Configures the question (prompt) in Spanish that will be used for generation.

        Args:
            question: Question or prompt text in Spanish that will be sent to the model
                     along with the image.
        """
        self.question = question

    def set_api_key(self, api_key: str) -> None:
        """
        Method implemented for compatibility with the abstract interface.
        Ollama does not require an API key since it runs locally.

        Args:
            api_key: Not used for Ollama, included for interface compatibility.
        """
        # Ollama does not require API key, but we implement the method for compatibility
        pass

    def process_imagen(self, image_path: str, force_overwrite: bool = False) -> Dict[str, Any]:
        """
        Processes an image using local Ollama models.
        
        If a description file already exists and force_overwrite is False,
        the already processed result is returned. Otherwise, the description
        is generated through the local model, the output is validated, and if correct,
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

        # Extract basic image information
        try:
            image_info = self._extract_basic_image_info(image_path)
            # Prepare the image as attachment
            attachment = LocalImageAttachment(os.path.abspath(image_path))
        except Exception as e:
            result["error"] = f"Error opening image: {str(e)}"
            self.log_error(image_path)
            return result

        # Create an effective prompt for the description
        prompt = self._create_description_prompt(image_info)

        try:
            with self.model_lock:
                base_params: Dict[str, Any] = {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_ctx": self.num_ctx,
                    "top_k": self.top_k,
                    "num_predict": self.max_tokens,
                    "stop": ["###"]
                }
                response = self.model.prompt(prompt, attachments=[attachment], **base_params)
                description = response.text()
            
            process_time = time.time() - start
            result["description"] = description
            result["process_time"] = process_time
            result["status"] = "processed"
        except Exception as e:
            error_msg = str(e)
            result["error"] = f"Error processing image with Ollama: {error_msg}"
            self.log_error(image_path)
            return result

        # Verify the generated output
        if not self.verify_output(result):
            return result

        # If there are no errors, create the file with the generated description.
        self.create_output_description(image_path, result["description"])

        return result

    def _extract_basic_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        Extracts basic metadata from an image file.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary containing:
                - filename: Base name of the image file
                - path: Full path to the image
                - format: Image format (JPEG, PNG, etc.)
                - mode: Color mode (RGB, RGBA, etc.)
                - size: Tuple of (width, height) in pixels
                
        Raises:
            IOError: If the image cannot be opened or read.
            PIL.UnidentifiedImageError: If the file is not a valid image.
        """
        info: Dict[str, Any] = {
            "filename": os.path.basename(image_path),
            "path": image_path
        }
        with Image.open(image_path) as img:
            info["format"] = img.format
            info["mode"] = img.mode
            info["size"] = img.size
        return info

    def _create_description_prompt(self, image_info: Dict[str, Any]) -> str:
        """
        Creates an effective prompt for the model to generate a detailed image description.
        
        Args:
            image_info: Dictionary containing image metadata.
            
        Returns:
            Formatted prompt string that includes the user question and image context.
        """
        prompt = f"""{self.question}
            
            The image is a {image_info.get('format', 'unknown')} file of 
            {image_info.get('size', (0, 0))[0]}x{image_info.get('size', (0, 0))[1]} pixels.

            Your response must be in Spanish.
            """
        return prompt

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
            description: Description text generated by the model.
            
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

    def update_config(self, 
                     max_tokens: Optional[int] = None, 
                     model: Optional[str] = None, 
                     temperature: Optional[float] = None, 
                     top_p: Optional[float] = None, 
                     top_k: Optional[int] = None, 
                     num_ctx: Optional[int] = None) -> 'OllamaClient':
        """
        Allows updating the client configuration parameters.

        Args:
            max_tokens: New maximum number of tokens for response generation.
            model: New Ollama model identifier to use.
            temperature: New temperature value for controlling randomness.
            top_p: New top_p value for nucleus sampling.
            top_k: New top_k value for vocabulary restriction.
            num_ctx: New context window size for the model.

        Returns:
            The current instance to allow method chaining.
            
        Raises:
            ValueError: If the specified new model cannot be loaded.
        """
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if model is not None:
            # Try to load the new model
            try:
                self.model = llm.get_model(model)
                self.model_name = model
            except UnknownModelError as e:
                raise ValueError(f"Error getting model {model}: {e}")
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if top_k is not None:
            self.top_k = top_k
        if num_ctx is not None:
            self.num_ctx = num_ctx
            
        return self