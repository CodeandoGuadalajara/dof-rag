import os
import time
from PIL import Image
import threading
import base64
import io
from openai import OpenAI
from typing import Dict, Any, Optional, Union, List, Tuple

from .AbstractClient import AbstractAIClient

api_lock = threading.Lock()

class OpenAIClient(AbstractAIClient):
    """
    Concrete implementation for interacting with the OpenAI API.

    This class provides methods to process images using OpenAI's vision models,
    handling authentication, API communication, and result processing. It 
    implements the AbstractAIClient interface to ensure consistency with
    other model implementations.
    
    Attributes:
        model (str): OpenAI model identifier (e.g., "gpt-4o")
        max_tokens (int): Maximum token limit for generated responses
        temperature (float): Controls randomness in generation (0.0 to 1.0)
        top_p (float): Controls diversity via nucleus sampling (0.0 to 1.0)
        question (str): Prompt text to send with images
        api_key (str): Authentication key for OpenAI API
        _client (OpenAI): OpenAI client instance
    """
    def __init__(self, 
                 model: str = "gpt-4o", 
                 max_tokens: int = 256, 
                 temperature: float = 0.6,
                 top_p: float = 0.6,
                 api_key: Optional[str] = None):
        """
        Initializes the OpenAI client configuration.

        Args:
            model: OpenAI model identifier to use for image processing.
            max_tokens: Maximum number of tokens in the output response.
            temperature: Controls generation creativity (0.0 to 1.0).
            top_p: Top_p value for nucleus sampling (0.0 to 1.0).
            api_key: API key to access OpenAI. If not provided,
                     it's taken from the 'OPENAI_API_KEY' environment variable.
                     
        Raises:
            ValueError: If api_key is later found to be invalid during set_api_key.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.question = "¿Qué se observa en esta imagen?, respóndelo en español, por favor."
                
        if api_key:
            self.set_api_key(api_key)
        
    def set_question(self, question: str) -> None:
        """
        Configures the question (prompt) in Spanish that will be used for generation.

        Args:
            question: Question or prompt text in Spanish that will be sent to OpenAI
                     along with the image.
        """
        self.question = question

    def set_api_key(self, api_key: str) -> None:
        """
        Updates the API key and reinitializes the OpenAI client.

        Args:
            api_key: New API key for OpenAI service.
            
        Raises:
            ValueError: If the provided API key is empty or invalid.
        """
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found. Please configure it.")
        self.api_key = api_key
        self._client = OpenAI(api_key=self.api_key)

    def _encode_image(self, image_path: str) -> str:
        """
        Encodes an image in base64 to send it to the OpenAI API.
        
        Args:
            image_path: Path of the image file to encode.
            
        Returns:
            Base64 encoded image string ready for API submission.
            
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            IOError: If there's an error reading the image file.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_imagen(self, image_path: str, force_overwrite: bool = False) -> Dict[str, Any]:
        """
        Processes an image using OpenAI's vision models.
        
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

        # Open the image
        try:
            # We only need to verify the image can be opened correctly
            _ = Image.open(image_path)
            # Encode the image in base64
            base64_image = self._encode_image(image_path)
        except Exception as e:
            result["error"] = f"Error opening image: {str(e)}"
            self.log_error(image_path)
            return result

        try:
            with api_lock:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an assistant specialized in describing images in Spanish in a clear and concise manner."},
                        {"role": "user", "content": [
                            {"type": "text", "text": self.question},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                description = response.choices[0].message.content
            process_time = time.time() - start
            result["description"] = description
            result["process_time"] = process_time
            result["status"] = "processed"
        except Exception as e:
            error_msg = str(e)
            result["error"] = f"Error processing image with OpenAI: {error_msg}"
            if "429" in error_msg:
                result["error_type"] = "rate_limit"
            elif "404" in error_msg:
                result["error_type"] = "not_found"
            else:
                result["error_type"] = "other"
            self.log_error(image_path)
            return result

        # Verify the generated output.
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
            description: Description text generated by the OpenAI API.
            
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
                     top_p: Optional[float] = None) -> 'OpenAIClient':
        """
        Allows updating the client configuration parameters.

        Args:
            max_tokens: New maximum number of tokens for response generation.
            model: New OpenAI model identifier to use.
            temperature: New temperature value for controlling randomness.
            top_p: New top_p value for nucleus sampling.

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
            
        return self