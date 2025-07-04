"""Enhanced Image Caption Extraction System

This module provides an improved version of the caption extraction system
with SQLite database storage, better error handling, and enhanced processing
capabilities. It replaces the file-based approach with a more robust
database-driven solution.

Main components:
- DatabaseManager: SQLite database operations
- OpenAIClient: AI client for image description generation with support for multiple providers
- FileProcessor: Batch image processing with checkpoints and error prioritization
- ErrorHandler: Comprehensive error handling and logging
- ImagePriorityManager: Intelligent error image prioritization system

Usage:
    from modules_captions import DatabaseManager, FileProcessor, ErrorHandler
    
    db_manager = DatabaseManager('path/to/db.sqlite')
    file_processor = FileProcessor(root_directory='path/to/images', db_manager=db_manager)
    results = file_processor.process_images(images)
"""

__version__ = "3.0.0"
__author__ = "DOF-RAG Project"
__description__ = "Enhanced Image Caption Extraction System with SQLite Storage and Error Prioritization"

# Import main classes for easy access
from .clients import OpenAIClient, create_client
from .db.manager import DatabaseManager
from .utils.error_handler import ErrorHandler
from .utils.error_log_manager import ErrorLogManager
from .utils.file_processor import FileProcessor
from .utils.prioritize_error_images import ImagePriorityManager

__all__ = [
    'DatabaseManager',
    'create_client',
    'OpenAIClient',
    'FileProcessor',
    'ErrorHandler',
    'ErrorLogManager',
    'ImagePriorityManager',
]