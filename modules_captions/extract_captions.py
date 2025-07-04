#!/usr/bin/env python3
"""Extract Captions - Enhanced Image Description Extraction System

This script provides an improved version of the caption extraction system
with SQLite database storage, better error handling, and enhanced processing
capabilities. It supports multiple AI providers through direct command-line flags.

The system automatically handles path resolution for database files:
- Paths starting with '../' are resolved relative to the project root
- Simple filenames are placed in modules_captions/db/
- Absolute paths are used as-is

Usage:
    python extract_captions.py --root-dir /path/to/images --db-path ../dof_db/db.sqlite
    python extract_captions.py --root-dir /path/to/images --openai
    python extract_captions.py --root-dir /path/to/images --gemini
    python extract_captions.py --root-dir /path/to/images --claude
"""

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

# Import centralized color management
try:
    from .utils.colors import ColorManager
    COLORAMA_AVAILABLE = True
except ImportError:
    # Fallback if ColorManager is not available
    class MockColorManager:
        @staticmethod
        def colorize(text, color):
            return text
    
    ColorManager = MockColorManager()
    COLORAMA_AVAILABLE = False

try:
    from .db.manager import DatabaseManager
    from .clients import create_client
    from .utils.file_processor import FileProcessor
    from .utils.error_handler import ErrorHandler
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent))
    from db.manager import DatabaseManager
    from clients import create_client
    from utils.file_processor import FileProcessor
    from utils.error_handler import ErrorHandler

def print_header(text: str, color: str = 'cyan'):
    """Print a formatted header with colors."""
    separator = "=" * len(text)
    header_text = f"\n{separator}\n{text}\n{separator}"
    print(ColorManager.colorize(header_text, color))

def print_info(label: str, value: str, color: str = 'green'):
    """Print formatted information with colors."""
    info_text = f"📋 {label}: {value}"
    print(ColorManager.colorize(info_text, color))

def print_success(message: str):
    """Print success message with green color."""
    success_text = f"✅ {message}"
    print(ColorManager.colorize(success_text, 'green'))

def print_warning(message: str):
    """Print warning message with yellow color."""
    warning_text = f"⚠️  {message}"
    print(ColorManager.colorize(warning_text, 'yellow'))

def print_error(message: str):
    """Print error message with red color."""
    error_text = f"❌ {message}"
    print(ColorManager.colorize(error_text, 'red'))

def print_stats(stats: Dict[str, Any]):
    """Print essential processing statistics with colors and formatting."""
    print_header("📊 Processing Summary", 'cyan')
    
    # Essential statistics only
    processed = stats.get('total_processed', 0)
    total = stats.get('total_images', 0)
    errors = stats.get('total_errors', 0)
    
    print_info("Processed", f"{processed}/{total} images", 'green' if processed > 0 else 'yellow')
    
    if errors > 0:
        print_info("Errors", str(errors), 'red')
    
    # Success rate with color coding
    success_rate = stats.get('success_rate', 0)
    if success_rate >= 90:
        color = 'green'
    elif success_rate >= 70:
        color = 'yellow'
    else:
        color = 'red'
    print_info("Success rate", f"{success_rate:.1f}%", color)
    
    # Time information (simplified)
    total_time = stats.get('total_time_seconds', 0)
    if total_time > 0:
        print_info("Total time", f"{total_time:.1f}s", 'magenta')
    
    # Database summary (simplified)
    db_stats = stats.get('database_stats', {})
    if db_stats and db_stats.get('total_descriptions', 0) > 0:
        print_info("Total descriptions in DB", str(db_stats.get('total_descriptions', 0)), 'cyan')

class CaptionExtractor:
    """
    Main caption extraction orchestrator.
    
    This class coordinates the entire caption extraction process,
    managing the database, AI client, file processor, and error handling.
    """
    
    def __init__(self, config: Dict[str, Any], status_only: bool = False):
        """
        Initialize the caption extractor.
        
        Args:
            config: Configuration dictionary with all necessary parameters.
            status_only: If True, skip API key validation for status-only operations.
        """
        self.config = config
        self.interrupted = False
        self.status_only = status_only
        self.debug_mode = config.get('debug_mode', False)
        
        # Initialize error handler first
        self.error_handler = ErrorHandler(
            log_dir=config.get('log_dir', 'logs'),
            log_level=config.get('log_level', 20),  # INFO level
            debug_mode=config.get('debug_mode', False)
        )
        
        # Initialize database manager
        self.db_manager = DatabaseManager(config['db_path'])
        
        # Initialize AI client
        self.ai_client = self._create_ai_client()
        
        # Pass debug mode to AI client if supported
        if hasattr(self.ai_client, 'debug_mode'):
            self.ai_client.debug_mode = config.get('debug_mode', False)
        
        # Initialize file processor (always needed for status and processing)
        self.file_processor = FileProcessor(
            root_directory=config['root_directory'],
            db_manager=self.db_manager,
            ai_client=self.ai_client,
            log_dir=config.get('log_directory', 'logs'),
            commit_interval=config.get('commit_interval', 10),
            cooldown_seconds=config.get('cooldown_seconds', 0),
            debug_mode=config.get('debug_mode', False),
            checkpoint_dir=config.get('checkpoint_dir')
        )
        
        # Set file processor reference in AI client for interrupt handling (only if not status-only)
        if not status_only and hasattr(self.ai_client, 'set_file_processor'):
            self.ai_client.set_file_processor(self.file_processor)
        
        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _create_ai_client(self):
        """
        Create and configure the AI client.
        
        Returns:
            Configured AI client instance.
        """
        try:
            # Get provider configuration
            provider = self.config.get('provider', 'openai')
            client_config = self.config.get('client_config', {})
            
            # Create client
            client = create_client(provider, **client_config)
            
            # Set API key
            api_key = self.config.get('api_key') or os.getenv(f"{provider.upper()}_API_KEY")
            if not api_key:
                error_msg = (f"API key not found for provider {provider}. "
                           f"Set {provider.upper()}_API_KEY environment variable or provide in config.")
                if self.status_only:
                    self.error_handler.logger.warning(error_msg + " (Status-only mode, continuing without API key)")
                    # Don't set API key, client will be in limited mode
                else:
                    self.error_handler.logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                client.set_api_key(api_key)
            
            # Set prompt from config - required field
            prompt = self.config.get('prompt')
            if not prompt:
                raise ValueError("Prompt must be defined in config.json")
            client.set_prompt(prompt)
            
            # Set error handler for the client
            client.set_error_handler(self.error_handler)
            
            # Configure rate limiting if available
            rate_limits = self.config.get('rate_limits', {})
            requests_per_minute = rate_limits.get('requests_per_minute')
            if requests_per_minute and hasattr(client, 'set_rate_limits'):
                client.set_rate_limits(requests_per_minute)
            
            # Show client configuration only in debug mode
            if self.debug_mode:
                print(client.get_model_info())
            return client
            
        except Exception as e:
            self.error_handler.handle_error(e, {'provider': provider, 'config': client_config}, 'api')
            raise
    
    def _signal_handler(self, signum, frame):
        """
        Handle interrupt signals for graceful shutdown.
        
        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        self.error_handler.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.interrupted = True
        self.file_processor.interrupt()
    
    def extract_captions(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Main method to extract captions from images.
        
        Args:
            force_reprocess: If True, reprocess images even if descriptions exist.
            
        Returns:
            Dict with processing results and statistics.
        """
        try:
            # Validate API key configuration before processing
            if not hasattr(self.ai_client, '_client') or self.ai_client._client is None:
                error_msg = "AI client not properly configured. API key validation failed."
                self.error_handler.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log processing start
            self.error_handler.log_processing_start(self.config)
            
            # Find images to process
            self.error_handler.logger.info("Discovering images to process...")
            images = self.file_processor.find_images(skip_existing=not force_reprocess)
            
            if not images:
                self.error_handler.logger.info("No images found to process")
                return {'status': 'completed', 'message': 'No images to process'}
            
            # Process images
            self.error_handler.logger.info(f"Starting processing of {len(images)} images")
            results = self.file_processor.process_images(images, force_reprocess)
            
            # Process error images with priority if any exist
            self.error_handler.logger.info("🔍 Attempting to process images with errors...")
            error_images = self.file_processor._extract_error_images_for_processing()
            
            if error_images:
                self.error_handler.logger.info(f"📋 Found {len(error_images)} error images to retry")
                self.file_processor._process_error_images_with_priority(error_images)
            else:
                self.error_handler.logger.info("✅ No error images found to process")
            
            # Check for missing images from the provided directory
            missing_images = self._check_missing_images_from_directory(images)
            if missing_images:
                self.error_handler.logger.info(f"⚠️ Found {len(missing_images)} missing images from provided directory")
                for i, missing_img in enumerate(missing_images[:5], 1):  # Show first 5
                    self.error_handler.logger.info(f"   {i}. {missing_img}")
                if len(missing_images) > 5:
                    self.error_handler.logger.info(f"   ... and {len(missing_images) - 5} more missing images")
            else:
                self.error_handler.logger.info("✅ All images from directory are present")
            
            # Add database statistics
            db_stats = self.db_manager.get_statistics()
            results['database_stats'] = db_stats
            
            # Add error summary
            error_summary = self.error_handler.get_error_summary()
            results['error_summary'] = error_summary
            
            # Log processing end
            self.error_handler.log_processing_end(results)
            
            return {
                'status': 'completed' if not results.get('interrupted', False) else 'interrupted',
                'results': results
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, {'operation': 'extract_captions'}, 'unknown')
            return {
                'status': 'error',
                'error': str(e),
                'error_summary': self.error_handler.get_error_summary()
            }
    
    def _check_missing_images_from_directory(self, processed_images: List) -> List[str]:
        """
        Check for images that exist in the directory but were not processed.
        
        Args:
            processed_images: List of image tuples that were processed
            
        Returns:
            List of missing image filenames
        """
        try:
            # Get all image files from the root directory
            all_images = set()
            root_path = Path(self.config.get('root_directory', '.'))
            
            if root_path.exists():
                # Find all image files in the directory
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff']:
                    all_images.update(f.name for f in root_path.rglob(ext))
            
            # Get processed image filenames
            processed_filenames = set()
            for image_tuple in processed_images:
                if len(image_tuple) >= 4:
                    processed_filenames.add(image_tuple[3])  # image_filename is 4th element
            
            # Find missing images
            missing_images = list(all_images - processed_filenames)
            return sorted(missing_images)
            
        except Exception as e:
            self.error_handler.logger.error(f"Error checking missing images: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status and statistics.
        
        Returns:
            Dict with system status information.
        """
        try:
            db_stats = self.db_manager.get_statistics()
            error_summary = self.error_handler.get_error_summary()
            
            # Count available images if file_processor is available
            total_images = 0
            processed_images = 0
            completion_percentage = 0
            directory_status = "unknown"
            current_directory = ""
            
            if self.file_processor:
                try:
                    images = self.file_processor.find_images(skip_existing=False)
                    total_images = len(images)
                    
                    # Count how many images actually have descriptions in database
                    processed_images = self._count_processed_images(images)
                    
                    # Calculate completion percentage
                    if total_images > 0:
                        completion_percentage = round((processed_images / total_images) * 100, 1)
                    
                    # Check if current directory is completed using hierarchical validation
                    completed_dirs = self.file_processor._load_completed_directories()
                    
                    # Get relative path of current directory
                    try:
                        current_directory = str(self.file_processor.root_directory.relative_to(Path.cwd()))
                        if current_directory == ".":
                            current_directory = ""
                    except ValueError:
                        current_directory = str(self.file_processor.root_directory.name)
                    
                    # Use hierarchical validation to determine if directory is truly completed
                    is_hierarchically_complete = self.file_processor._validate_directory_completion_by_hierarchy(current_directory)
                    
                    # Determine directory status with hierarchical precision
                    if total_images == 0:
                        directory_status = "no_images"
                    elif completion_percentage == 100 and is_hierarchically_complete:
                        directory_status = "fully_completed"
                    elif current_directory in completed_dirs:
                        # Directory is marked as completed, but verify with hierarchical validation
                        if is_hierarchically_complete and completion_percentage == 100:
                            directory_status = "fully_completed"
                        elif processed_images > 0:
                            directory_status = "partially_completed"
                        else:
                            # Directory marked as completed but validation shows it's not truly complete
                            directory_status = "marked_completed"
                    elif processed_images > 0:
                        directory_status = "partially_completed"
                    else:
                        directory_status = "pending"
                        
                except Exception as e:
                    self.error_handler.logger.warning(f"Could not count images: {e}")
            
            # Try to get model information, but don't fail if not possible
            try:
                model_info = self.ai_client.get_model_info()
            except Exception as model_error:
                self.error_handler.logger.warning(f"Could not get model information: {model_error}")
                model_info = {"status": "unavailable", "error": str(model_error)}
            
            # Filter sensitive information from configuration
            safe_config = self.config.copy()
            if 'api_key' in safe_config:
                safe_config['api_key'] = '***REDACTED***'
            
            return {
                'status': 'ready',
                'total_images': total_images,
                'processed_images': processed_images,
                'completion_percentage': completion_percentage,
                'directory_status': directory_status,
                'current_directory': current_directory,
                'database_stats': db_stats,
                'error_summary': error_summary,
                'model_info': model_info,
                'config': safe_config
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, {'operation': 'get_status'}, 'unknown')
            return {'error': str(e)}
    
    def _count_processed_images(self, images: List[tuple]) -> int:
        """
        Count how many images in the list have descriptions in the database.
        
        Args:
            images: List of tuples (document_name, image_path, page_number, image_filename, parent_dir)
            
        Returns:
            int: Number of images that have descriptions in the database
        """
        processed_count = 0
        
        for image_tuple in images:
            try:
                # Unpack the tuple (document_name, image_path, page_number, image_filename, parent_dir)
                document_name, image_path, page_number, image_filename, parent_dir = image_tuple
                    
                # Check if description exists in database
                if self.db_manager.description_exists(document_name, page_number, image_filename):
                    processed_count += 1
                        
            except Exception as e:
                self.error_handler.logger.warning(f"Error checking processed status for {image_tuple}: {e}")
                continue
                
        return processed_count


def load_provider_config(provider: str) -> Dict[str, Any]:
    """
    Load configuration for the specified provider from config.json.
    
    Args:
        provider: Provider name (openai, gemini, claude, ollama, azure)
        
    Returns:
        Complete configuration for the specified provider
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            unified_config = json.load(f)
    except Exception as e:
        print_error(f"Error loading config.json: {e}")
        print_warning("Using default configuration...")
        return create_default_config()
    
    # Get base configuration (excluding metadata sections)
    excluded_keys = {"providers", "_comment", "_instructions", "_configuration_notes", "_environment_variables", "_usage_examples"}
    config = {k: v for k, v in unified_config.items() if k not in excluded_keys}
    
    # Check if provider exists
    providers = unified_config.get("providers", {})
    if provider not in providers:
        print_error(f"Provider '{provider}' not found in configuration. Available providers:")
        for available_provider in providers.keys():
            provider_text = f"  - {available_provider}"
            print(ColorManager.colorize(provider_text, 'cyan'))
        print_warning("Using default configuration...")
        return config
    
    # Update with provider-specific configuration
    provider_config = providers[provider]
    config["provider"] = provider
    
    # Update client_config and rate_limits if present
    for key in ["client_config", "rate_limits"]:
        if key in provider_config:
            config[key] = provider_config[key]
    
    # Resolve paths
    _resolve_config_paths(config, os.path.dirname(os.path.abspath(__file__)))
    
    # Display rate limits if available
    _display_rate_limits(provider, provider_config.get("rate_limits"))
    
    # Handle API key from environment
    _handle_api_key(config, provider_config)
    
    return config

def _resolve_config_paths(config: Dict[str, Any], module_dir: str) -> None:
    """Resolve relative paths in configuration to absolute paths."""
    path_configs = {
        "log_directory": "log_dir",
        "checkpoint_directory": "checkpoint_dir"
    }
    
    for path_key, alias_key in path_configs.items():
        if path_key in config and not os.path.isabs(config[path_key]):
            config[path_key] = os.path.join(module_dir, config[path_key])
            config[alias_key] = config[path_key]  # Alias for compatibility
            os.makedirs(config[path_key], exist_ok=True)
    
    # Handle db_path separately due to special logic
    if "db_path" in config and not os.path.isabs(config["db_path"]):
        if config["db_path"].startswith("../"):
            relative_path = config["db_path"][3:]
            config["db_path"] = os.path.normpath(os.path.join(module_dir, "..", relative_path))
        else:
            db_filename = os.path.basename(config["db_path"])
            config["db_path"] = os.path.join(module_dir, "db", db_filename)
        
        db_dir = os.path.dirname(config["db_path"])
        os.makedirs(db_dir, exist_ok=True)

def _display_rate_limits(provider: str, rate_limits: Optional[Dict[str, Any]]) -> None:
    """Display rate limit information for the provider."""
    if not rate_limits:
        return
        
    print_header(f"⚡ Rate Limits for {provider}", 'yellow')
    for limit_type, limit_value in rate_limits.items():
        display_name = limit_type.replace('_', ' ').title()
        display_value = str(limit_value) if limit_value is not None else 'Not specified'
        print_info(display_name, display_value, 'cyan')

def _handle_api_key(config: Dict[str, Any], provider_config: Dict[str, Any]) -> None:
    """Handle API key configuration from environment variables."""
    env_var = provider_config.get("env_var")
    if not env_var:
        return
        
    api_key = os.getenv(env_var)
    if api_key:
        print_success(f"Using API key from environment variable {env_var}")
        from clients.openai import OpenAIClient
        config["api_key"] = OpenAIClient.clean_api_key(api_key)
    else:
        print_warning(f"Environment variable {env_var} not found")
        print_warning(f"Set {env_var} or provide an API key with --api-key")

def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Returns:
        Default configuration.
    """
    # Path to modules_captions directory for logs and checkpoints
    module_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(module_dir, 'logs')
    
    # Create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    
    return {
        'provider': 'openai',

        'log_dir': logs_dir,
        'log_level': 20,  # INFO
        'client_config': {
            'model': 'gpt-4o',
            'max_tokens': 256,
            'temperature': 0.6,
            'top_p': 0.6
        }
        # Prompt will be loaded from config.json
    }

# Load environment variables from .env file
try:
    load_dotenv()
    print("Environment variables loaded from .env")
except Exception as e:
    print(f"Error loading environment variables: {e}")

def get_default_provider() -> str:
    """
    Get the default provider from config.json.
    
    Returns:
        Default provider name
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('provider', 'gemini')
    except Exception:
        return 'gemini'  # Fallback if config can't be read

def main():
    """
    Main entry point for the caption extraction script.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Image Caption Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_captions.py --root-dir ./images --db-path ../../dof_db/db.sqlite
  python extract_captions.py --force-reprocess
  python extract_captions.py --root-dir ./images --openai
  python extract_captions.py --root-dir ./images --gemini
  python extract_captions.py --root-dir ./images --claude
        """
    )
    
    # Configuration options
    parser.add_argument('--root-dir', type=str, help='Root directory containing images')
    parser.add_argument('--db-path', type=str, default='../../dof_db/db.sqlite', help='Path to SQLite database')
    
    # Provider selection (mutually exclusive)
    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument('--openai', action='store_true', help='Use OpenAI provider')
    provider_group.add_argument('--gemini', action='store_true', help='Use Google Gemini provider')
    provider_group.add_argument('--claude', action='store_true', help='Use Anthropic Claude provider')
    provider_group.add_argument('--ollama', action='store_true', help='Use Ollama provider')
    provider_group.add_argument('--azure', action='store_true', help='Use Azure OpenAI provider')
    
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Reprocess images even if descriptions exist')
    parser.add_argument('--status', action='store_true', help='Show system status and exit')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with verbose logging and detailed API information')
    
    args = parser.parse_args()
    
    try:
        # Determine provider from command line arguments or use config default
        if args.gemini:
            provider = 'gemini'
        elif args.claude:
            provider = 'claude'
        elif args.ollama:
            provider = 'ollama'
        elif args.azure:
            provider = 'azure'
        elif args.openai:
            provider = 'openai'
        else:
            # Get default provider from config.json
            provider = get_default_provider()
        
        # Load provider-specific configuration from config.json
        config = load_provider_config(provider)
        
        # Set default root directory to dof_markdown if not specified
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_root_dir = os.path.join(project_root, 'dof_markdown')
        
        # Override config with command line arguments
        if args.root_dir:
            config['root_directory'] = args.root_dir
        else:
            # Use dof_markdown as default root directory
            config['root_directory'] = default_root_dir
            
        if args.db_path and args.db_path != '../../dof_db/db.sqlite':  # Only override if explicitly set by user
            # Apply same path resolution logic as load_provider_config
            if not os.path.isabs(args.db_path):
                module_dir = os.path.dirname(os.path.abspath(__file__))
                if args.db_path.startswith("../"):
                    # Resolve '../' paths relative to module directory
                    relative_path = args.db_path[3:]
                    config['db_path'] = os.path.normpath(os.path.join(module_dir, "..", relative_path))
                else:
                    # Place simple filenames in modules_captions/db/
                    db_filename = os.path.basename(args.db_path)
                    config['db_path'] = os.path.join(module_dir, "db", db_filename)
                # Create directory if it doesn't exist
                db_dir = os.path.dirname(config['db_path'])
                os.makedirs(db_dir, exist_ok=True)
            else:
                config['db_path'] = args.db_path

        if args.log_level:
            log_levels = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40}
            config['log_level'] = log_levels[args.log_level]
        
        # Enable debug mode if requested
        if args.debug:
            config['log_level'] = 10  # DEBUG level
            config['debug_mode'] = True
            print("🔍 Debug mode enabled - verbose logging activated")
        else:
            config['debug_mode'] = False
        
        # Validate required parameters
        if 'root_directory' not in config:
            parser.error("Root directory is required. Use --root-dir or provide in config file.")
        
        if not os.path.exists(config['root_directory']):
            parser.error(f"Root directory does not exist: {config['root_directory']}")
        
        # Handle status request specially to avoid API key requirement
        if args.status:
            try:
                # Initialize extractor in status-only mode
                extractor = CaptionExtractor(config, status_only=True)
                status = extractor.get_status()
                
                # Display status with colored formatting instead of JSON
                print_header("📊 System Status", 'blue')
                print_info("Status", status.get('status', 'Unknown'), 'green')
                
                # Directory status information
                directory_status = status.get('directory_status', 'unknown')
                current_dir = status.get('current_directory', '')
                total_images = status.get('total_images', 0)
                processed_images = status.get('processed_images', 0)
                completion_percentage = status.get('completion_percentage', 0)
                
                dir_display = current_dir if current_dir else 'root'
                
                if directory_status == 'fully_completed':
                    print_info("Directory", f"{dir_display} (✅ FULLY COMPLETED)", 'green')
                    print_info("Images found", f"{total_images} ({processed_images}/{total_images} processed - 100%)", 'green')
                elif directory_status == 'partially_completed':
                    print_info("Directory", f"{dir_display} (⚠️ PARTIALLY COMPLETED)", 'yellow')
                    print_info("Images found", f"{total_images} ({processed_images}/{total_images} processed - {completion_percentage}%)", 'yellow')
                elif directory_status == 'marked_completed':
                    print_info("Directory", f"{dir_display} (🔄 MARKED COMPLETED)", 'magenta')
                    print_info("Images found", f"{total_images} (marked as completed but {processed_images} actually processed)", 'magenta')
                elif directory_status == 'no_images':
                    print_info("Directory", f"{dir_display} (📁 NO IMAGES)", 'cyan')
                    print_info("Images found", "0 (no images in directory)", 'cyan')
                elif directory_status == 'pending':
                    print_info("Directory", f"{dir_display} (🔄 PENDING)", 'white')
                    if processed_images > 0:
                        print_info("Images found", f"{total_images} ({processed_images}/{total_images} processed - {completion_percentage}%)", 'white')
                    else:
                        print_info("Images found", f"{total_images} (0 processed - ready to start)", 'white')
                else:
                    print_info("Directory", f"{dir_display} (❓ UNKNOWN)", 'red')
                    print_info("Images found", str(total_images), 'red')
                
                # Database statistics
                db_stats = status.get('database_stats', {})
                if db_stats:
                    print_header("💾 Database Statistics", 'blue')
                    print_info("Database path", os.path.abspath(config.get('db_path', '../../dof_db/db.sqlite')), 'yellow')
                    print_info("Total descriptions", str(db_stats.get('total_descriptions', 0)), 'cyan')
                    print_info("Unique documents", str(db_stats.get('unique_documents', 0)), 'cyan')
                    print_info("Recent descriptions", str(db_stats.get('recent_descriptions', 0)), 'cyan')
                
                return
            except Exception as e:
                print_error(f"Error getting status: {e}")
                sys.exit(1)
        
        # Initialize extractor for normal operation
        print_header("🚀 Starting Caption Extraction", 'green')
        print_info("Provider", provider, 'cyan')
        print_info("Model", config.get('client_config', {}).get('model', 'Not specified'), 'cyan')
        print_info("Base URL", config.get('client_config', {}).get('base_url', 'Default'), 'cyan')
        
        if config.get('debug_mode', False):
            print_warning("🔍 Debug mode: ENABLED")
            print_info("Root directory", config['root_directory'], 'yellow')
            print_info("Database path", os.path.abspath(config.get('db_path', '../../dof_db/db.sqlite')), 'yellow')
        
        extractor = CaptionExtractor(config)
        
        # Extract captions
        print_info("Starting caption extraction", "Processing images...", 'green')
        results = extractor.extract_captions(force_reprocess=args.force_reprocess)
        
        # Display results with appropriate formatting based on status
        if results['status'] in ['completed', 'interrupted']:
            # Display statistics if processing results are available
            if 'results' in results:
                print_stats(results['results'])
            else:
                print_info("Processing completed", results.get('message', 'No additional information'), 'green')
            if results['status'] == 'interrupted':
                print_warning("Processing was interrupted")
        else:
            print_error(f"Error during processing: {results.get('error', 'Unknown error')}")
            # Display error summary for troubleshooting
            if 'error_summary' in results:
                error_summary = results['error_summary']
                if error_summary.get('total_errors', 0) > 0:
                    print_error(f"Total errors encountered: {error_summary['total_errors']}")
                    print_info("Error log", error_summary.get('error_log_file', 'Not available'), 'yellow')
        
    except KeyboardInterrupt:
        print_warning("Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()