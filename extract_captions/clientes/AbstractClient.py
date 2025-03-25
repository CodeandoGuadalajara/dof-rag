from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Bool

class AbstractAIClient(ABC):
    """
    Abstract class that defines the interface for clients of different AI models.
    
    This interface standardizes the methods that all AI client implementations
    must provide, ensuring consistency across different model providers.
    
    Methods that must be implemented in each model are:
        - set_api_key: Configure or update the API key.
        - process_imagen: Process an input image.
        - verify_output: Validate the generated output.
        - create_output_description: Generate a description of the output.
        - set_question: Configure the question or prompt in Spanish.
        - update_config: Update the client configuration.
    """

    @abstractmethod
    def set_api_key(self, api_key: str) -> None:
        """
        Configures the API key for the client.
        
        Args:
            api_key: Authentication key for the specific AI service.
        
        Raises:
            ValueError: If the API key is invalid or empty.
        """
        pass

    @abstractmethod
    def process_imagen(self, image_path: str, force_overwrite: bool = False) -> Dict[str, Any]:
        """
        Processes the input image using the AI model to generate a description.
        
        Args:
            image_path: Path to the image file to be processed.
            force_overwrite: If True, reprocess even if a description already exists.
            
        Returns:
            Dict containing processing results with these possible keys:
                - image_path: Path of the processed image.
                - description: Generated description text (if successful).
                - status: Processing status ("processed", "already_processed", etc.).
                - process_time: Time taken to process the image (if applicable).
                - error: Error message (if any error occurred).
                - error_type: Type of error encountered (if applicable).
        """
        pass

    @abstractmethod
    def verify_output(self, result: Dict[str, Any]) -> bool:
        """
        Verifies and validates the generated output to ensure quality.
        
        Args:
            result: Dictionary with processing result information.
            
        Returns:
            True if the output is valid and meets quality standards, False otherwise.
        """
        pass

    @abstractmethod
    def create_output_description(self, image_path: str, description: str) -> None:
        """
        Creates a file with the description generated by the model.
        
        Args:
            image_path: Path of the processed image.
            description: Description text generated by the model.
            
        Note:
            The output file is typically saved with the same name as the image
            but with a .txt extension in the same directory.
        """
        pass
    
    @abstractmethod
    def set_question(self, question: str) -> None:
        """
        Configures the question (prompt) in Spanish that will be used for generation.
        
        Args:
            question: Question or prompt text in Spanish that will be sent to the model
                     along with the image to generate a description.
        """
        pass
    
    @abstractmethod
    def log_error(self, image_path: str, extra_info: str = "") -> None:
        """
        Logs information about errors during processing to facilitate debugging.
        
        Args:
            image_path: Path of the image that produced an error.
            extra_info: Additional context or error details to include in the log.
        """
        pass
    
    @abstractmethod
    def update_config(self, **kwargs: Any) -> 'AbstractAIClient':
        """
        Updates the configuration parameters of the client.
        
        Args:
            **kwargs: Configuration parameters to update, which may include:
                     - model: Name of the AI model to use
                     - max_tokens: Maximum response length
                     - temperature: Controls randomness in generation
                     - top_p: Controls diversity via nucleus sampling
                     - top_k: Controls diversity via vocabulary restriction
            
        Returns:
            The current instance for method chaining.
        """
        pass