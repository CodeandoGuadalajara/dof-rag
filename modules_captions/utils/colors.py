#!/usr/bin/env python3
"""Centralized color management with colorama fallback support."""

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    
    # Fallback color definitions
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = ""
    
    class Style:
        BRIGHT = RESET_ALL = ""

class ColorManager:
    """Centralized color management with fallback support."""
    
    @staticmethod
    def is_available() -> bool:
        """Check if colorama is available."""
        return COLORAMA_AVAILABLE
    
    @staticmethod
    def get_colors():
        """Get color classes (Fore, Style)."""
        return Fore, Style
    
    @staticmethod
    def colorize(text: str, color: str, style: str = "") -> str:
        """Apply color and style to text if colorama available."""
        if not COLORAMA_AVAILABLE:
            return text
        return f"{getattr(Fore, color.upper(), '')}{getattr(Style, style.upper(), '')}{text}{Style.RESET_ALL}"