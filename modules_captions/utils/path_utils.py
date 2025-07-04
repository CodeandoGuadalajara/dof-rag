#!/usr/bin/env python3
"""Cross-platform path utilities with hierarchical directory support."""

import re
from pathlib import Path
from typing import Generator, Tuple

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')


class PathManager:
    """Cross-platform path management for hierarchical directories."""
    
    @staticmethod
    def find_images_recursive(
        root_path: str,
        image_extensions: Tuple[str, ...] = IMAGE_EXTENSIONS
    ) -> Generator[Tuple[str, str, int, str, str], None, None]:
        """
        Find images recursively in hierarchical directory structure.
        
        Args:
            root_path: Root directory to search from
            image_extensions: Tuple of valid image extensions
            
        Yields:
            Tuple of (document_name, img_path, page_number, filename, parent_dir)
        """
        root = Path(root_path)
        
        if not root.exists():
            return
            
        for img_path in root.rglob('*'):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                parent_dir = str(img_path.parent)
                filename = img_path.name
                
                document_name = img_path.parent.name
                page_number = PathManager.extract_page_number(filename)
                
                yield (document_name, str(img_path), page_number, filename, parent_dir)
    
    @staticmethod
    def extract_page_number(filename: str) -> int:
        """
        Extract page number from filename.
        
        Args:
            filename: Image filename
            
        Returns:
            Page number (0 if not found)
        """
        page_match = re.search(r'_page_(\d+)_', filename)
        return int(page_match.group(1)) if page_match else 0