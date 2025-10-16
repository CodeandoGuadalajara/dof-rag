import base64
import logging
import os
from threading import Lock
import time
from typing import Optional

try:
    import openai
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI library not found. Install with: pip install openai")

# Thread lock for API calls to prevent rate limit issues
api_lock = Lock()

class OpenAIClient:
    """
    OpenAI API client for image description generation.
    
    This client provides a unified interface for generating image descriptions
    using OpenAI's vision models. It includes automatic rate limiting,
    error handling, and retry logic.
    """
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 max_tokens: int = 300,
                 temperature: float = 0.3,
                 top_p: float = 0.9,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            model: Model name to use for image description.
            max_tokens: Maximum number of tokens in the output response.
            temperature: Controls generation creativity (0.0 to 1.0).
            top_p: Top_p value for nucleus sampling (0.0 to 1.0).
            base_url: Optional base URL for API requests (for custom endpoints).
            api_key: API key for OpenAI. If not provided, will try environment variable.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = base_url
        
        # Default prompt will be set via set_prompt() method
        self.prompt = None
        
        self._client = None
        self.error_handler = None
        self.logger = logging.getLogger(__name__)
        self.debug_mode = False  # Will be set by CaptionExtractor if debug mode is enabled
        
        # Rate limiting system
        self.requests_per_minute = None
        self.request_timestamps = []
        self.rate_limit_enabled = False
        
        # Reference to file processor for interrupt handling
        self.file_processor = None
        
        if api_key:
            self.set_api_key(api_key)
        elif os.getenv("OPENAI_API_KEY"):
            self.set_api_key(os.getenv("OPENAI_API_KEY"))
    
    def set_error_handler(self, error_handler) -> None:
        """
        Set the error handler for the client.
        
        Args:
            error_handler: Error handler instance for logging and managing errors.
        """
        self.error_handler = error_handler
    
    def set_file_processor(self, file_processor) -> None:
        """
        Set the file processor reference for interrupt handling.
        
        Args:
            file_processor: FileProcessor instance for checking interrupt status.
        """
        self.file_processor = file_processor
    
    def set_rate_limits(self, requests_per_minute: Optional[int] = None) -> None:
        """
        Configure rate limiting for the client.
        
        Args:
            requests_per_minute: Maximum requests per minute allowed. If None, rate limiting is disabled.
        """
        self.requests_per_minute = requests_per_minute
        self.rate_limit_enabled = requests_per_minute is not None and requests_per_minute > 0
        
        if self.rate_limit_enabled:
            self.logger.info(f"Rate limiting enabled: {requests_per_minute} requests per minute")
        else:
            self.logger.info("Rate limiting disabled")
    
    def _clean_old_timestamps(self) -> None:
        """
        Remove timestamps older than 1 minute from the request history.
        """
        current_time = time.time()
        cutoff_time = current_time - 60
        old_count = len(self.request_timestamps)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff_time]
        
        # Reset warning flag when timestamps are cleaned (new minute period)
        if len(self.request_timestamps) < old_count:
            self.logger.debug(f"Cleaned {old_count - len(self.request_timestamps)} old timestamps")
            # Reset warning flag for new minute period
            if hasattr(self, '_warning_shown_this_minute'):
                delattr(self, '_warning_shown_this_minute')
            if hasattr(self, '_last_warning_time'):
                delattr(self, '_last_warning_time')
    
    def _check_rate_limit(self) -> bool:
        """
        Check if we've reached the rate limit and apply cooling if necessary.
        
        Returns:
            bool: True if request can proceed, False if rate limit reached.
        """
        if not self.rate_limit_enabled:
            return True
        
        self._clean_old_timestamps()
        current_requests = len(self.request_timestamps)
        
        # Show warning exactly one request before the actual rate limit
        warning_threshold = self.requests_per_minute - 1
        
        # Warning one request before rate limit (only show once per minute)
        current_time = time.time()
        if current_requests >= warning_threshold and not hasattr(self, '_warning_shown_this_minute'):
            # Only show warning if we haven't shown it in the last 60 seconds
            if not hasattr(self, '_last_warning_time') or (current_time - self._last_warning_time) >= 60:
                self._warning_shown_this_minute = True
                self._last_warning_time = current_time
                warning_msg = f"\n🟡 Rate limit warning: {current_requests}/{self.requests_per_minute} requests"
                next_msg = "   └─ Next request will trigger 60s cooling period"
                self.logger.warning(warning_msg)
                self.logger.info(next_msg)
        
        # Apply cooling when reaching the actual rate limit
        if current_requests >= self.requests_per_minute:
            cooling_msg = f"\n🔴 Rate limit reached: {current_requests}/{self.requests_per_minute} requests per minute"
            self.logger.warning(cooling_msg)
            self._apply_cooling()
            return False
        
        return True
    
    def _apply_cooling(self) -> None:
        """
        Apply automatic cooling period when rate limit is reached.
        """
        cooling_seconds = 62  # Cool for 62 seconds (with 2s buffer for safety)
        self.logger.info(f"❄️ Applying automatic cooling for {cooling_seconds} seconds to avoid rate limiting...")
        
        try:
            from tqdm import tqdm
            
            # Enhanced visual cooling progress with better formatting
            cooling_desc = "🧊 Rate limit cooling - Please wait"
            with tqdm(total=cooling_seconds, desc=cooling_desc, unit="s",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]',
                     ncols=100, leave=False, colour='blue') as pbar:
                
                for second in range(cooling_seconds):
                    # Check for interruption during cooling
                    if self.file_processor and hasattr(self.file_processor, 'interrupted') and self.file_processor.interrupted:
                        pbar.clear()
                        self.logger.info("\n⚠️ Cooling interrupted by user")
                        break
                    
                    time.sleep(1)
                    pbar.update(1)
                    
                    # Update description with remaining time
                    remaining = cooling_seconds - second - 1
                    if remaining > 0:
                        pbar.set_description(f"🧊 Rate limit cooling - {remaining}s remaining")
                    else:
                        pbar.set_description("🧊 Cooling complete")
                        
            self.logger.info("\n✅ Rate limit cooling completed - Ready to continue")
            
        except ImportError:
            # Fallback to original method if tqdm is not available
            self.logger.info(f"🧊 Cooling for {cooling_seconds} seconds...")
            
            for second in range(cooling_seconds):
                # Check for interruption during cooling
                if self.file_processor and hasattr(self.file_processor, 'interrupted') and self.file_processor.interrupted:
                    self.logger.info("⚠️ Cooling interrupted by user")
                    break
                time.sleep(1)
                
                # Show progress every 10 seconds
                if (second + 1) % 10 == 0:
                    remaining = cooling_seconds - second - 1
                    self.logger.info(f"   └─ {remaining}s remaining...")
                
            self.logger.info("✅ Rate limit cooling completed - Ready to continue")
        
        # Clear old timestamps after cooling and log the cleanup
        timestamps_before = len(self.request_timestamps)
        self._clean_old_timestamps()
        timestamps_after = len(self.request_timestamps)
        
        self.logger.info(f"✅ Cooling period completed - Cleaned {timestamps_before - timestamps_after} old timestamps")
        self.logger.info("✅ Ready to resume normal processing")
    
    def _record_request(self) -> None:
        """
        Record a successful API request timestamp.
        """
        if self.rate_limit_enabled:
            self.request_timestamps.append(time.time())
     
    @staticmethod
    def clean_api_key(api_key: str) -> str:
        """
        Clean and validate API key by removing quotes and whitespace.
        
        Args:
            api_key: Raw API key that may contain quotes or extra whitespace.
            
        Returns:
            str: Cleaned API key without quotes or extra whitespace.
            
        Raises:
            ValueError: If the API key is empty or invalid after cleaning.
        """
        if not api_key:
            raise ValueError("API key cannot be empty or None")
        
        # Remove leading/trailing whitespace
        cleaned_key = api_key.strip()
        
        # Remove surrounding quotes (both single and double)
        if len(cleaned_key) >= 2:
            if (cleaned_key.startswith('"') and cleaned_key.endswith('"')) or \
               (cleaned_key.startswith("'") and cleaned_key.endswith("'")):
                cleaned_key = cleaned_key[1:-1]
        
        # Remove any remaining whitespace after quote removal
        cleaned_key = cleaned_key.strip()
        
        if not cleaned_key:
            raise ValueError("API key is empty after cleaning quotes and whitespace")
        
        return cleaned_key
    
    def set_api_key(self, api_key: str) -> None:
        """
        Configure the API key and initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (may contain quotes or extra whitespace).
            
        Raises:
            ValueError: If the API key is empty or invalid.
        """
        if not api_key:
            raise ValueError("OpenAI API key cannot be empty")
        
        # Clean the API key to remove quotes and whitespace
        self.api_key = self.clean_api_key(api_key)
        
        # Initialize client with optional base_url
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self._client = OpenAI(**client_kwargs)
    
    def set_prompt(self, prompt: str) -> None:
        """
        Configure the prompt used for image description generation.
        
        Args:
            prompt: Prompt text in Spanish.
        """
        self.prompt = prompt
    
    def describe(self, image_path: str) -> str:
        """
        Generate a description for the given image using OpenAI.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            str: Generated description in Spanish.
            
        Raises:
            ValueError: If image file doesn't exist or can't be processed.
            Exception: For API errors or other processing issues.
        """
        if not self._client:
            raise ValueError("API key not configured. Call set_api_key() first.")
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        # Debug mode logging
        debug_mode = getattr(self, 'debug_mode', False)
        if debug_mode:
            self.logger.debug(f"🔍 Starting image description for: {os.path.basename(image_path)}")
            self.logger.debug(f"Model configuration: {self.model}, max_tokens: {self.max_tokens}, temp: {self.temperature}")
            if self.base_url:
                self.logger.debug(f"Using custom base URL: {self.base_url}")
        else:
            # Show API requests in normal mode like regular processing
            self.logger.info(f"🤖 Requesting description for: {os.path.basename(image_path)}")
        
        try:
            # Check rate limit before making request
            if not self._check_rate_limit():
                # Rate limit was reached and cooling was applied, try again with explicit cleanup
                self.logger.debug("Rate limit reached, cooling applied. Performing second verification...")
                
                # Ensure timestamps are cleaned before second check
                self._clean_old_timestamps()
                
                # Add small buffer to ensure timing precision
                time.sleep(1)
                
                if not self._check_rate_limit():
                    current_requests = len(self.request_timestamps)
                    self.logger.error(f"Rate limit still exceeded after cooling: {current_requests}/{self.requests_per_minute} requests")
                    raise Exception(f"Rate limit still exceeded after cooling period: {current_requests}/{self.requests_per_minute} requests in last minute")
            
            # Record start time for API call timing
            api_start_time = time.time()
            
            # Process image to base64
            if debug_mode:
                self.logger.debug("Converting image to base64...")
            image_base64 = self._process_image_to_base64(image_path)
            
            if debug_mode:
                image_size_kb = len(image_base64) * 3 / 4 / 1024  # Approximate size in KB
                self.logger.debug(f"Image converted to base64 (~{image_size_kb:.1f} KB)")
            
            # Prepare message for the API
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ]
            
            if debug_mode:
                self.logger.debug(f"Prompt length: {len(self.prompt)} characters")
                self.logger.debug("Sending request to OpenAI API...")
            
            # Generate description with thread safety
            with api_lock:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                
                description = response.choices[0].message.content if response.choices else ""
                
            # Calculate API response time
            api_response_time = time.time() - api_start_time
            
            if debug_mode:
                self.logger.debug(f"✓ API response received in {api_response_time:.2f}s")
                if hasattr(response, 'usage') and response.usage:
                    self.logger.debug(f"Token usage - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
                self.logger.debug(f"Response length: {len(description)} characters")
            else:
                # Show API response in normal mode like regular processing
                self.logger.info(f"✅ Description received ({api_response_time:.1f}s) - {len(description)} chars")
                
            if not description or description.strip() == "":
                raise Exception("Empty response from OpenAI API")
            
            # Record successful request for rate limiting
            self._record_request()
            
            # Reset consecutive API errors counter after successful request
            if self.error_handler:
                self.error_handler.reset_consecutive_api_errors()
                
            return description.strip()
            
        except openai.APIError as e:
            # Handle OpenAI API specific errors
            error_msg = f"OpenAI API Error: {str(e)}"
            http_status = getattr(e, 'status_code', None)
            error_type = type(e).__name__
            
            # Debug mode error logging
            if debug_mode:
                self.logger.debug("❌ OpenAI API Error details:")
                self.logger.debug(f"  - Error type: {error_type}")
                self.logger.debug(f"  - HTTP status: {http_status}")
                self.logger.debug(f"  - Error message: {str(e)}")
                self.logger.debug(f"  - Image path: {image_path}")
                self.logger.debug(f"  - Model: {self.model}")
            
            # Categorize API errors
            is_server_error = http_status and (500 <= http_status < 600)
            is_rate_limit = isinstance(e, openai.RateLimitError) or http_status == 429
            is_auth_error = isinstance(e, openai.AuthenticationError)
            is_permission_error = isinstance(e, openai.PermissionDeniedError)
            
            if self.error_handler:
                error_context = {
                    'image_path': image_path,
                    'model': self.model,
                    'api_response_time': locals().get('api_response_time', 0),
                    'http_status': http_status,
                    'error_type': error_type,
                    'is_server_error': is_server_error,
                    'is_rate_limit': is_rate_limit,
                    'is_auth_error': is_auth_error,
                    'is_permission_error': is_permission_error,
                    'error_category': self._categorize_api_error(e, http_status)
                }
                self.error_handler.handle_api_error(e, error_context)
            
            raise Exception(error_msg)
            
        except ConnectionError as e:
            # Handle network connection errors
            error_msg = f"Network connection error: {str(e)}"
            
            if debug_mode:
                self.logger.debug("❌ Connection Error details:")
                self.logger.debug(f"  - Error type: {type(e).__name__}")
                self.logger.debug(f"  - Error message: {str(e)}")
                self.logger.debug(f"  - Image path: {image_path}")
                self.logger.debug(f"  - Model: {self.model}")
            
            if self.error_handler:
                error_context = {
                    'image_path': image_path,
                    'model': self.model,
                    'api_response_time': locals().get('api_response_time', 0),
                    'error_category': 'network_error',
                    'is_network_error': True
                }
                self.error_handler.handle_api_error(e, error_context)
            
            raise Exception(error_msg)
            
        except TimeoutError as e:
            # Handle timeout errors
            error_msg = f"Request timeout error: {str(e)}"
            
            if debug_mode:
                self.logger.debug("❌ Timeout Error details:")
                self.logger.debug(f"  - Error type: {type(e).__name__}")
                self.logger.debug(f"  - Error message: {str(e)}")
                self.logger.debug(f"  - Image path: {image_path}")
                self.logger.debug(f"  - Model: {self.model}")
            
            if self.error_handler:
                error_context = {
                    'image_path': image_path,
                    'model': self.model,
                    'api_response_time': locals().get('api_response_time', 0),
                    'error_category': 'timeout_error',
                    'is_timeout_error': True
                }
                self.error_handler.handle_api_error(e, error_context)
            
            raise Exception(error_msg)
            
        except Exception as e:
            # Handle any other unexpected errors
            error_msg = f"Unexpected error processing image with OpenAI: {str(e)}"
            
            # Debug mode error logging
            if debug_mode:
                self.logger.debug("❌ Unexpected Error details:")
                self.logger.debug(f"  - Error type: {type(e).__name__}")
                self.logger.debug(f"  - Error message: {str(e)}")
                self.logger.debug(f"  - Image path: {image_path}")
                self.logger.debug(f"  - Model: {self.model}")
                if hasattr(e, 'response') and e.response:
                    self.logger.debug(f"  - HTTP status: {getattr(e.response, 'status_code', 'Unknown')}")
            
            # Si tenemos acceso al error_handler, registrar el error
            if self.error_handler:
                # Verificar si es un error de servidor (503, 500, etc.)
                is_server_error = any(term in str(e).lower() for term in ['503', '500', 'unavailable', 'overloaded'])
                
                # Registrar el error con el error_handler
                error_context = {
                    'image_path': image_path,
                    'is_server_error': is_server_error,
                    'model': self.model,
                    'api_response_time': locals().get('api_response_time', 0),
                    'error_category': 'unknown_error'
                }
                
                self.error_handler.handle_api_error(e, error_context)
                
            # Re-raise the exception to be handled by the caller
            raise Exception(error_msg)
    
    def _categorize_api_error(self, error: Exception, http_status: Optional[int]) -> str:
        """
        Categorize API errors for better error handling and reporting.
        
        Args:
            error: The API exception.
            http_status: HTTP status code if available.
            
        Returns:
            str: Error category for classification.
        """
        if isinstance(error, openai.RateLimitError):
            return 'api_rate_limit'
        elif isinstance(error, openai.AuthenticationError):
            return 'api_authentication'
        elif isinstance(error, openai.PermissionDeniedError):
            return 'api_permission_denied'
        elif isinstance(error, openai.BadRequestError):
            return 'api_bad_request'
        elif http_status and 500 <= http_status < 600:
            return 'api_server_error'
        elif http_status and 400 <= http_status < 500:
            return 'api_client_error'
        else:
            return 'api_communication'
    
    def get_model_info(self) -> str:
        """
        Get current model configuration information formatted for display.
        
        Returns:
            Formatted string with model configuration details.
        """
        info_lines = []
        info_lines.append(f"📋 Max Tokens: {self.max_tokens}")
        info_lines.append(f"📋 Temperature: {self.temperature}")
        info_lines.append(f"📋 Top P: {self.top_p}")
        if self.base_url:
            info_lines.append(f"📋 Rate Limiting: {'Enabled' if hasattr(self, 'rate_limit_enabled') and self.rate_limit_enabled else 'Disabled'}")
        
        return "\n".join(info_lines)
    
    def _process_image_to_base64(self, image_path: str) -> str:
        """
        Load an image and convert it to base64 encoding.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            str: Base64 encoded image.
            
        Raises:
            ValueError: If image cannot be processed.
        """
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                
            return base64.b64encode(image_bytes).decode('utf-8')
            
        except Exception as e:
            raise ValueError(f"Error processing image file: {str(e)}")