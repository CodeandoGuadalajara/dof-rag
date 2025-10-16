import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .error_log_manager import ErrorLogManager
from .logging_config import setup_logging

class ErrorHandler:
    """Centralized error handling with tracking and recovery for image processing."""
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO, debug_mode: bool = False) -> None:
        """
        Initialize error handler.
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            debug_mode: Enable debug mode with enhanced logging
        """
        self.log_dir = Path(log_dir)
        self.debug_mode = debug_mode
        
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.error_counts = {
            'api_errors': 0,
            'file_errors': 0,
            'database_errors': 0,
            'validation_errors': 0,
            'unknown_errors': 0
        }
        
        self.error_details = []
        self.consecutive_api_errors = 0
        self._last_rate_limit_warning = 0
        self._rate_limit_warning_interval = 60
        
        log_level_name = logging.getLevelName(log_level)
        self.logger = setup_logging(
            log_level=log_level_name,
            log_file="error_handler.log",
            log_dir=str(self.log_dir),
            enable_colors=True
        )
        self.logger.name = 'error_handler'
        
        self.error_log_file = self.log_dir / "error_images.json"
        self.error_log_manager = ErrorLogManager(log_dir=str(self.log_dir), debug_mode=self.debug_mode)
        
        if self.debug_mode:
            self.logger.debug("🔍 Debug mode activated in ErrorHandler")
            self.logger.debug(f"Log directory: {self.log_dir}")
            self.logger.debug(f"Log level: {log_level}")
            self.logger.debug(f"Error log file: {self.error_log_file}")
    
   
    def handle_error(self, 
                    error: Exception, 
                    context: Dict[str, Any], 
                    error_type: str = "unknown") -> None:
        """
        Handle and log error with context.
        
        Args:
            error: Exception that occurred
            context: Context information about error location
            error_type: Error category (api, file, database, validation, unknown)
        """
        error_key = f"{error_type}_errors"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        normalized_context = self._normalize_context(context)
        error_record = self._create_error_record(error, error_type, normalized_context)
        
        self.error_details.append(error_record)
        self._log_error(error, error_type, normalized_context)
        
        self._save_error_log(error_record)
    
    def handle_api_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Handle API-related errors with improved categorization and logging.
        
        Args:
            error: The API exception.
            context: Additional context information about the error.
        """
        context = context or {}
        error_message = str(error)
        
        error_info = self._categorize_api_error(error_message, context)
        context.update(error_info)
        
        self._log_api_error(error_message, error_info)
        self._update_consecutive_errors(error_info)
        
        self.handle_error(error, context, 'api')
    
    def handle_file_error(self, error: Exception, file_path: str, operation: str) -> None:
        """
        Handle file-related errors.
        
        Args:
            error: The file exception.
            file_path: Path to the file that caused the error.
            operation: The file operation being performed (read, write, delete, etc.).
        """
        context = {
            'file_path': file_path,
            'operation': operation,
            'file_exists': os.path.exists(file_path),
            'error_category': 'file_operation'
        }
        self.handle_error(error, context, 'file')
    
    def handle_database_error(self, error: Exception, operation: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle database-related errors.
        
        Args:
            error: The database exception.
            operation: The database operation being performed.
            data: Data involved in the operation (optional).
        """
        context = {
            'operation': operation,
            'data': data,
            'error_category': 'database_operation'
        }
        self.handle_error(error, context, 'database')
    
    def handle_validation_error(self, error: Exception, validation_type: str, data: Any) -> None:
        """
        Handle validation-related errors.
        
        Args:
            error: The validation exception.
            validation_type: Type of validation that failed.
            data: Data that failed validation.
        """
        context = {
            'validation_type': validation_type,
            'data': str(data)[:500],
            'error_category': 'data_validation'
        }
        self.handle_error(error, context, 'validation')
    
    def _save_error_log(self, error_record: Dict[str, Any]) -> None:
        """
        Appends an error record to the error log file (error_images.json).
        
        Legacy wrapper method that uses ErrorLogManager for centralized handling.

        Args:
            error_record: The error dictionary to append.
        """
        image_path = error_record.get('context', {}).get('image_path', '')
        error_message = error_record.get('error_message', str(error_record.get('error', '')))
        directory = error_record.get('context', {}).get('directory', '')
        filename = error_record.get('context', {}).get('filename', '')
        error_type = error_record.get('error_type', 'unknown')
        
        # Extract directory from image_path if directory is not available
        if not directory and image_path:
            directory = str(Path(image_path).parent)
        
        # Extract filename from image_path if filename is not available
        if not filename and image_path:
            filename = Path(image_path).name
        
        self.error_log_manager.save_error_image(
            directory=directory,
            filename=filename,
            error_msg=error_message,
            error_type=error_type,
            context=error_record.get('context', {})
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors encountered.
        
        Returns:
            Dict with error statistics and recent errors.
        """
        return {
            'error_counts': self.error_counts.copy(),
            'total_errors': sum(self.error_counts.values()),
            'recent_errors': self.error_details[-10:],
            'error_log_file': str(self.error_log_file)
        }
    
    def get_error_report(self) -> str:
        """
        Generate a human-readable error report.
        
        Returns:
            Formatted string with error summary.
        """
        summary = self.get_error_summary()
        error_categories = self._count_error_categories()
        
        report_lines = [
            "=== Error Report ===",
            f"Total Errors: {summary['total_errors']}",
            f"API Errors: {self.error_counts['api_errors']}",
            f"  - Rate Limit Errors: {error_categories['rate_limit']}",
            f"  - Authentication Errors: {error_categories['auth']}",
            f"  - Server Errors (5xx): {error_categories['server']}",
            f"  - Network Errors: {error_categories['network']}",
            f"  - Timeout Errors: {error_categories['timeout']}",
            f"File Errors: {self.error_counts['file_errors']}",
            f"Database Errors: {self.error_counts['database_errors']}",
            f"Validation Errors: {self.error_counts['validation_errors']}",
            f"Unknown Errors: {self.error_counts['unknown_errors']}",
            "",
            "Recent Errors:"
        ]
        
        for i, error in enumerate(summary['recent_errors'][-5:], 1):
            report_lines.extend([
                f"{i}. [{error['error_type'].upper()}] {error['error_class']}: {error['error_message'][:100]}...",
                f"   Time: {error['timestamp']}",
                f"   Context: {error['context']}"
            ])
        
        report_lines.extend([
            "",
            f"Detailed error log: {summary['error_log_file']}",
            "=================="
        ])
        
        return "\n".join(report_lines)
    
    def clear_errors(self) -> None:
        """
        Clear the current error tracking (not the log files).
        """
        self.error_counts = {
            'api_errors': 0,
            'file_errors': 0,
            'database_errors': 0,
            'validation_errors': 0,
            'unknown_errors': 0
        }
        self.error_details.clear()
        self.logger.info("Error tracking cleared - All error counters and details have been reset for new processing session")
        
        # Reset consecutive API errors counter
        self.consecutive_api_errors = 0
    
    def should_continue_processing(self, max_consecutive_errors: int = 5) -> bool:
        """
        Determine if processing should continue based on error patterns.
        
        Args:
            max_consecutive_errors: Maximum consecutive errors before stopping.
            
        Returns:
            bool: True if processing should continue, False otherwise.
        """
        if not self.error_details:
            return True
            
        # Check for critical server errors
        if self._has_critical_server_error():
            return False
            
        # Check for consecutive API errors
        if self._has_too_many_consecutive_errors(max_consecutive_errors):
            return False
            
        # Check total error threshold
        total_errors = sum(self.error_counts.values())
        if total_errors > 50:
            self.logger.warning(f"High error count detected: {total_errors}")
            return False
        
        return True
    
    def log_processing_start(self, config: Dict[str, Any]) -> None:
        """
        Log the start of a processing session.
        
        Args:
            config: Processing configuration parameters.
        """
        self.logger.info("=== Starting Caption Extraction Session ===")
        self.clear_errors()
    
    def log_processing_end(self, stats: Dict[str, Any]) -> None:
        """
        Log the end of a processing session.
        
        Args:
            stats: Processing statistics.
        """
        self.logger.info("=== Caption Extraction Session Complete ===")
        
        if sum(self.error_counts.values()) > 0:
            self.logger.warning(f"Errors encountered: {sum(self.error_counts.values())} total")
        else:
            self.logger.info("No errors encountered during processing")
    
    def close(self) -> None:
        """
        Close all logging handlers to release file handles.
        This should be called when the ErrorHandler is no longer needed.
        """
        for handler in self.logger.handlers[:]:  # Create a copy of the list to avoid modification during iteration
            handler.close()
            self.logger.removeHandler(handler)
        self.logger.info("Logging handlers closed")
        logging.shutdown()
    
    def reset_consecutive_api_errors(self) -> None:
        """Reset the consecutive API errors counter.
        
        This method is called after a successful API operation to reset
        the consecutive error tracking.
        """
        if self.consecutive_api_errors > 0:
            self.logger.debug(f"Resetting consecutive API errors counter (was {self.consecutive_api_errors})")
            self.consecutive_api_errors = 0
    
    def _normalize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize context data for consistency."""
        if not context:
            return {}
        
        normalized = context.copy()
        if 'image_path' in normalized:
            normalized['image_path'] = normalized['image_path'].replace('\\', '/')
        return normalized
    
    def _create_error_record(self, error: Exception, error_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized error record."""
        return {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_class': error.__class__.__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context
        }
    
    def _log_error(self, error: Exception, error_type: str, context: Dict[str, Any]) -> None:
        """Log error with appropriate detail level."""
        # Check if it's a rate limit error and simplify the message
        if context.get('is_rate_limit', False) or 'RateLimitError' in error.__class__.__name__:
            retry_delay = self._extract_retry_delay(str(error))
            error_msg = f"[{error_type.upper()}] Rate limit exceeded. Retry in {retry_delay}."
        else:
            error_msg = f"[{error_type.upper()}] {error.__class__.__name__}: {str(error)}"
        
        if self.debug_mode:
            self.logger.error(error_msg)
            self.logger.debug(f"Error context: {json.dumps(context, indent=2, default=str)}")
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        else:
            # For rate limit errors, don't include the verbose context
            if context.get('is_rate_limit', False) or 'RateLimitError' in error.__class__.__name__:
                self.logger.error(error_msg)
            else:
                self.logger.error(f"{error_msg} | Context: {context}")
    
    def _categorize_api_error(self, error_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize API error and return classification info."""
        error_category = context.get('error_category', 'api_communication')
        http_status = context.get('http_status')
        error_lower = error_message.lower()
        
        return {
            'error_category': error_category,
            'is_server_error': (
                context.get('is_server_error', False) or
                error_category == 'api_server_error' or
                (http_status and 500 <= http_status < 600) or
                any(term in error_lower for term in ['503', '500', 'unavailable', 'overloaded'])
            ),
            'is_rate_limit': (
                context.get('is_rate_limit', False) or
                error_category == 'api_rate_limit' or
                'rate limit' in error_lower or
                (http_status == 429)
            ),
            'is_auth_error': (
                context.get('is_auth_error', False) or
                error_category == 'api_authentication' or
                any(term in error_lower for term in ['authentication', 'unauthorized']) or
                (http_status == 401)
            ),
            'is_network_error': (
                context.get('is_network_error', False) or
                error_category == 'network_error' or
                any(term in error_lower for term in ['connection', 'network'])
            ),
            'is_timeout_error': (
                context.get('is_timeout_error', False) or
                error_category == 'timeout_error' or
                'timeout' in error_lower
            ),
            'http_status': http_status
        }
    
    def _extract_retry_delay(self, error_message: str) -> str:
        """Extract retry delay from rate limit error message."""
        import re
        delay_match = re.search(r"'retryDelay': '(\d+)s'", error_message)
        if delay_match:
            return delay_match.group(1) + "s"
        return "unknown"
    
    def _log_api_error(self, error_message: str, error_info: Dict[str, Any]) -> None:
        """Log API error with appropriate severity."""
        http_status = error_info.get('http_status')
        
        if error_info['is_server_error']:
            self.logger.critical(f"Server error detected (HTTP {http_status}): {error_message}. This will stop processing.")
        elif error_info['is_rate_limit']:
            # Throttle rate limit messages to avoid spam
            import time
            current_time = time.time()
            if (current_time - self._last_rate_limit_warning) >= self._rate_limit_warning_interval:
                retry_delay = self._extract_retry_delay(error_message)
                self.logger.warning(f"Rate limit exceeded (HTTP {http_status}). Retry in {retry_delay}.")
                self._last_rate_limit_warning = current_time
            # If within throttle interval, log at debug level only
            else:
                retry_delay = self._extract_retry_delay(error_message)
                self.logger.debug(f"Rate limit exceeded (HTTP {http_status}). Retry in {retry_delay}. [Throttled]")
        elif error_info['is_auth_error']:
            self.logger.error(f"Authentication error detected (HTTP {http_status}): {error_message}. Check API credentials.")
        elif error_info['is_network_error']:
            self.logger.warning(f"Network error detected: {error_message}. Check internet connection.")
        elif error_info['is_timeout_error']:
            self.logger.warning(f"Timeout error detected: {error_message}. API response took too long.")
        else:
            self.logger.error(f"API communication error (HTTP {http_status}): {error_message}")
    
    def _update_consecutive_errors(self, error_info: Dict[str, Any]) -> None:
        """Update consecutive error counter based on error type."""
        if error_info['is_server_error']:
            self.consecutive_api_errors += 3  # Force processing stop
        elif error_info['is_rate_limit']:
            pass  # Don't increment for rate limits
        elif error_info['is_auth_error']:
            self.consecutive_api_errors += 2
        else:
            self.consecutive_api_errors += 1
    
    def _count_error_categories(self) -> Dict[str, int]:
        """Count errors by category for reporting."""
        categories = {'rate_limit': 0, 'auth': 0, 'server': 0, 'network': 0, 'timeout': 0}
        
        for error in self.error_details:
            context = error.get('context', {})
            if context.get('is_rate_limit'):
                categories['rate_limit'] += 1
            if context.get('is_auth_error'):
                categories['auth'] += 1
            if context.get('is_server_error'):
                categories['server'] += 1
            if context.get('is_network_error'):
                categories['network'] += 1
            if context.get('is_timeout_error'):
                categories['timeout'] += 1
        
        return categories
    
    def _has_critical_server_error(self) -> bool:
        """Check if latest error is a critical server error."""
        latest_error = self.error_details[-1]
        error_message = latest_error.get('error_message', '')
        error_context = latest_error.get('context', {})
        
        is_server_error = (
            error_context.get('is_server_error', False) or
            error_context.get('error_category') == 'api_server_error' or
            any(term in error_message for term in ['503 UNAVAILABLE', '500 ']) or
            any(term in error_message.lower() for term in ['unavailable', 'overloaded'])
        )
        
        if is_server_error:
            self.logger.critical(f"Detected server error: {error_message}. Stopping processing immediately.")
            return True
        return False
    
    def _has_too_many_consecutive_errors(self, max_consecutive_errors: int) -> bool:
        """Check if there are too many consecutive API errors."""
        if len(self.error_details) < max_consecutive_errors:
            return False
            
        recent_errors = self.error_details[-max_consecutive_errors:]
        api_errors = sum(1 for e in recent_errors if e['error_type'] == 'api')
        
        if api_errors == max_consecutive_errors:
            self.logger.warning(f"Detected {max_consecutive_errors} consecutive API errors")
            return True
        return False