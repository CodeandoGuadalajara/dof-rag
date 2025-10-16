#!/usr/bin/env python3
"""Centralized logging configuration with file/console handlers and color support."""

import logging
from pathlib import Path
from typing import Optional

from .colors import ColorManager

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Setup centralized logging with file and console handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name
        log_dir: Log directory path
        enable_colors: Enable colored console output
        
    Returns:
        Configured logger instance
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logger = logging.getLogger('modules_captions')
    logger.setLevel(numeric_level)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    
    if enable_colors and ColorManager.is_available():
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        if log_dir:
            log_path = Path(log_dir) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_path = Path(log_file)
            
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support."""
    
    COLORS = {
        'DEBUG': 'CYAN',
        'INFO': 'GREEN',
        'WARNING': 'YELLOW',
        'ERROR': 'RED',
        'CRITICAL': 'MAGENTA'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, 'WHITE')
        record.levelname = ColorManager.colorize(
            record.levelname, log_color, 'BRIGHT'
        )
        return super().format(record)