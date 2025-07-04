#!/usr/bin/env python3
"""
Centralized Error Log Manager for the caption extraction system.

This module provides a unified interface for managing error_images.json,
eliminating duplicated logic across FileProcessor, ErrorHandler, and
ImagePriorityManager classes.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from .logging_config import setup_logging


class ErrorLogManager:
    """Manages error_images.json with support for legacy and modern formats."""
    
    def __init__(self, log_dir: str = "logs", debug_mode: bool = False) -> None:
        """
        Initialize error log manager.
        
        Args:
            log_dir: Directory for error log file
            debug_mode: Enable debug logging
        """
        self.log_dir = Path(log_dir)
        self.debug_mode = debug_mode
        self.error_log_file = self.log_dir / "error_images.json"
        
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        log_level = "DEBUG" if debug_mode else "INFO"
        self.logger = setup_logging(
            log_level=log_level,
            log_file="error_log_manager.log",
            log_dir=str(self.log_dir),
            enable_colors=True
        )
        self.logger.name = 'error_log_manager'
        
        if self.debug_mode:
            self.logger.debug("🔍 Debug mode activated in ErrorLogManager")
            self.logger.debug(f"Error log file: {self.error_log_file}")
    
    def load_error_images(self) -> List[Dict[str, Any]]:
        """
        Load error records from JSON file.
        
        Returns:
            List of error records, empty if file missing or corrupted
        """
        if not self.error_log_file.exists():
            self.logger.debug("Error log file does not exist, returning empty list")
            return []
        
        try:
            with open(self.error_log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Standard format: list of error objects
                self.logger.debug(f"Loaded {len(data)} error records from standard list format")
                return data
            elif isinstance(data, dict):
                # Legacy format: dictionary of errors - convert to list
                self.logger.warning("Legacy dictionary format detected in error_images.json. Converting to list format.")
                converted_list = self._convert_dict_to_list(data)
                # Save converted format back to file
                self._save_error_list(converted_list)
                return converted_list
            else:
                self.logger.warning(f"Unexpected data format in error_images.json: {type(data)}. Returning empty list.")
                return []
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Could not decode JSON from {self.error_log_file}: {e}. File might be corrupted.")
            return []
        except Exception as e:
            self.logger.error(f"Error loading error images: {e}")
            return []
    
    def get_error_keys_set(self) -> Set[str]:
        """
        Get a set of normalized image paths that have errors.
        
        Returns:
            Set of normalized image paths from error records.
        """
        error_records = self.load_error_images()
        error_keys = set()
        
        for error_record in error_records:
            # Handle both direct image_path and context-based paths
            image_path = error_record.get('image_path')
            if not image_path:
                # Try to get from context
                context = error_record.get('context', {})
                image_path = context.get('image_path')
            
            if image_path:
                normalized_path = self._normalize_image_path(image_path)
                error_keys.add(normalized_path)
        
        self.logger.debug(f"Generated error keys set with {len(error_keys)} entries")
        return error_keys
    
    def save_error_image(self, directory: str, filename: str, error_msg: str, 
                        error_type: str = "unknown", context: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a new error record to the error log.
        
        Args:
            directory: Directory containing the image.
            filename: Name of the image file.
            error_msg: Error message describing the issue.
            error_type: Type of error (api, file, database, etc.).
            context: Additional context information.
        """
        # Normalize paths for consistency
        full_path = self._normalize_image_path(str(Path(directory) / filename))
        normalized_directory = self._normalize_image_path(directory)
        
        # Prepare the new error record
        new_error_record = {
            'image_path': full_path,
            'error_message': error_msg,
            'error_type': error_type,
            'timestamp': time.time(),
            'context': {
                'directory': normalized_directory,
                'filename': filename,
                'image_path': full_path,
                **(context or {})
            }
        }
        
        # Load existing errors and append new one
        error_list = self.load_error_images()
        error_list.append(new_error_record)
        
        # Save updated list
        self._save_error_list(error_list)
        
        self.logger.debug(f"Saved error for image: {full_path} (type: {error_type})")
    
    def remove_error_image(self, image_path: str) -> bool:
        """
        Remove a successfully processed image from the error log.
        
        Args:
            image_path: Path of the image to remove from error log.
            
        Returns:
            True if any entries were removed, False otherwise.
        """
        error_list = self.load_error_images()
        if not error_list:
            return False
        
        # Normalize the image path for consistent lookup
        normalized_path = self._normalize_image_path(image_path)
        filename = Path(image_path).name
        
        # Find entries to remove
        entries_to_remove = []
        for i, error_entry in enumerate(error_list):
            if self._matches_image(error_entry, image_path, normalized_path, filename):
                entries_to_remove.append(i)
                entry_path = error_entry.get('image_path', 'unknown')
                self.logger.info(f"✅ Found error entry to remove: {entry_path} (matches: {image_path})")
        
        # Remove entries (in reverse order to maintain indices)
        for i in reversed(entries_to_remove):
            del error_list[i]
        
        if entries_to_remove:
            # Save updated list back to file
            self._save_error_list(error_list)
            self.logger.info(f"🧹 Cleaned {len(entries_to_remove)} error entries for image: {filename}")
            return True
        else:
            self.logger.debug(f"Image path not found in error log: {image_path}")
            return False
    
    def get_error_count(self) -> int:
        """
        Get the total number of error records.
        
        Returns:
            Number of error records in the log.
        """
        return len(self.load_error_images())
    
    def clear_error_log(self) -> None:
        """
        Clear all error records from the log.
        """
        self._save_error_list([])
        self.logger.info("🧹 Cleared all error records from log")
    
    def _convert_dict_to_list(self, error_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert legacy dictionary format to modern list format.
        
        Args:
            error_dict: Dictionary in legacy format.
            
        Returns:
            List of error records in modern format.
        """
        converted_list = []
        
        for image_path, error_data in error_dict.items():
            if isinstance(error_data, dict):
                # Use existing data and ensure required fields
                record = error_data.copy()
                record['image_path'] = image_path
                
                # Ensure required fields exist
                if 'timestamp' not in record:
                    record['timestamp'] = time.time()
                if 'error_type' not in record:
                    record['error_type'] = 'unknown'
                if 'context' not in record:
                    record['context'] = {'image_path': image_path}
                
                converted_list.append(record)
            else:
                # Simple string error message
                record = {
                    'image_path': image_path,
                    'error_message': str(error_data),
                    'error_type': 'unknown',
                    'timestamp': time.time(),
                    'context': {'image_path': image_path}
                }
                converted_list.append(record)
        
        self.logger.info(f"Converted {len(converted_list)} records from dictionary to list format")
        return converted_list
    
    def _save_error_list(self, error_list: List[Dict[str, Any]]) -> None:
        """
        Save error list to the JSON file.
        
        Args:
            error_list: List of error records to save.
        """
        try:
            with open(self.error_log_file, 'w', encoding='utf-8') as f:
                json.dump(error_list, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved {len(error_list)} error records to {self.error_log_file}")
        except Exception as e:
            self.logger.error(f"Error saving error list to {self.error_log_file}: {e}")
            raise
    
    def _normalize_image_path(self, path: str, root_directory: Optional[str] = None) -> str:
        """
        Normalize image path for consistent comparison.
        
        Args:
            path: Image path to normalize.
            root_directory: Optional root directory to make path relative to.
            
        Returns:
            Normalized path with forward slashes.
        """
        if not path:
            return ""
        
        # Convert to Path object for proper handling
        path_obj = Path(path)
        
        # If root_directory is provided, try to make path relative to it
        if root_directory:
            try:
                root_path = Path(root_directory)
                if path_obj.is_absolute() and path_obj.is_relative_to(root_path):
                    relative_path = path_obj.relative_to(root_path)
                    return str(relative_path).replace('\\', '/')
            except (ValueError, OSError):
                # If relative_to fails, continue with other normalization
                pass
        
        # Try to make relative to a common base if it's absolute
        try:
            # If path contains 'dof_markdown', extract from there
            path_str = str(path_obj)
            if 'dof_markdown' in path_str:
                parts = path_str.replace('\\', '/').split('/')
                if 'dof_markdown' in parts:
                    idx = parts.index('dof_markdown')
                    return '/'.join(parts[idx:])
        except ValueError:
            pass
        
        # Fallback: use the path as-is but normalize separators
        return str(path_obj).replace('\\', '/')
    
    def _matches_image(self, error_entry: Dict[str, Any], image_path: str, 
                      normalized_path: str, filename: str) -> bool:
        """
        Check if an error entry matches the given image path.
        
        Args:
            error_entry: Error record to check.
            image_path: Original image path.
            normalized_path: Normalized image path.
            filename: Image filename.
            
        Returns:
            True if the entry matches the image.
        """
        # Get paths from error entry
        entry_image_path = error_entry.get('image_path', '')
        context = error_entry.get('context', {})
        context_image_path = context.get('image_path', '')
        context_filename = context.get('filename', '')
        
        if not entry_image_path and not context_image_path:
            return False
        
        # Use the most reliable path
        primary_path = entry_image_path or context_image_path
        normalized_entry_path = self._normalize_image_path(primary_path)
        
        # Check multiple matching criteria
        # 1. Exact normalized path match
        if normalized_entry_path == normalized_path:
            return True
        
        # 2. Original path match
        if primary_path == image_path:
            return True
        
        # 3. Filename match
        if context_filename == filename or primary_path.endswith('/' + filename):
            return True
        
        # 4. Handle truncated paths (e.g., "01/31012025-MAT/filename")
        if 'dof_markdown' in image_path and 'dof_markdown' in primary_path:
            # Extract relative parts for comparison
            image_parts = image_path.replace('\\', '/').split('/')
            entry_parts = primary_path.replace('\\', '/').split('/')
            
            if 'dof_markdown' in image_parts and 'dof_markdown' in entry_parts:
                img_idx = image_parts.index('dof_markdown')
                entry_idx = entry_parts.index('dof_markdown')
                
                # Compare from dof_markdown onwards
                img_relative = '/'.join(image_parts[img_idx:])
                entry_relative = '/'.join(entry_parts[entry_idx:])
                
                if img_relative == entry_relative:
                    return True
        
        return False