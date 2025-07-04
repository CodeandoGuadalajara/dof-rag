#!/usr/bin/env python3
"""Image prioritization system for error image reprocessing."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .error_log_manager import ErrorLogManager
from .path_utils import PathManager

class ImagePriorityManager:
    """Manages prioritization of error images for reprocessing."""
    
    def __init__(self, modules_captions_dir: str) -> None:
        self.modules_captions_dir = Path(modules_captions_dir)
        self.error_images_file = self.modules_captions_dir / "logs" / "error_images.json"
        self.priority_images = []
        self.pending_images = []
        
        # Initialize centralized error log manager
        log_dir = str(self.modules_captions_dir / "logs")
        self.error_log_manager = ErrorLogManager(log_dir=log_dir, debug_mode=False)
        
    def load_priority_list(self, error_data: List[Dict[str, Any]] = None) -> bool:
        """
        Load priority list from error_images data.
        
        Args:
            error_data: List of error images data. If None, loads from file.
        
        Returns:
            True if priority list loaded successfully, False otherwise.
        """
        try:
            if error_data is None:
                error_data = self.error_log_manager.load_error_images()
                if not error_data:
                    print("ℹ️ No error images found")
                    return False
            
            self.priority_images = []
            for error_entry in error_data:
                context = error_entry.get('context', {})
                image_path = context.get('image_path', '')
                
                if not image_path:
                    continue
                    
                path_parts = image_path.replace('\\', '/').split('/')
                if len(path_parts) >= 1:
                    filename = context.get('filename', path_parts[-1])
                    directory = context.get('directory', '/'.join(path_parts[:-1]))
                    
                    page_number = PathManager.extract_page_number(filename)
                    
                    if directory:
                        doc_parts = directory.split('/')
                        document_name = next((part for part in reversed(doc_parts) if part), 'unknown')
                    else:
                        document_name = filename.split('_')[0] if '_' in filename else 'unknown'
                    
                    self.priority_images.append({
                        'document': document_name,
                        'page': page_number,
                        'filename': filename,
                        'directory': directory,
                        'error_type': error_entry.get('error_type', 'unknown'),
                        'timestamp': error_entry.get('timestamp', ''),
                        'original_path': image_path  # Keep original path for debugging
                    })
            
            if len(self.priority_images) > 0:
                print(f"✅ Loaded {len(self.priority_images)} error images for priority processing")
                # Show sample of priority images for debugging
                print("📋 Priority images sample:")
                for i, img in enumerate(self.priority_images[:3]):
                    print(f"   {i+1}. {img['document']}/{img['filename']} (page {img['page']})")
                if len(self.priority_images) > 3:
                    print(f"   ... and {len(self.priority_images) - 3} more")
            else:
                print("ℹ️ No error images found to prioritize")
            
            return len(self.priority_images) > 0
            
        except Exception as e:
            print(f"❌ Error loading error images: {e}")
            return False
    
    def create_prioritized_image_list(self, found_images: List[Tuple[str, str, int, str, str]]) -> List[Tuple[str, str, int, str, str]]:
        """
        Reorganize image list to prioritize error images.
        
        Args:
            found_images: Original list of found images
            
        Returns:
            Reorganized list with priority images first
        """
        if not self.priority_images:
            print("ℹ️ No priority images to process")
            return found_images
            
        prioritized_list = []
        remaining_images = []
        priority_found = 0
              
        exact_match_set = set()
        filename_match_set = set()
        path_match_dict = {}
        
        for priority_img in self.priority_images:
            exact_key = (priority_img['document'], priority_img['page'], priority_img['filename'])
            exact_match_set.add(exact_key)
            
            filename_match_set.add(priority_img['filename'])
            path_match_dict[priority_img['filename']] = priority_img
        
        print(f"🔍 Matching {len(exact_match_set)} priority images against {len(found_images)} total images")
        
        matched_priority_images = set()
        
        for img_tuple in found_images:
            document_name, img_path, page_number, filename, parent_dir = img_tuple
            
            # Strategy 1: Try exact match first
            img_identifier = (document_name, page_number, filename)
            
            if img_identifier in exact_match_set:
                prioritized_list.append(img_tuple)
                priority_found += 1
                matched_priority_images.add(filename)
                # Reduce console output - only show significant matches
                if priority_found <= 2:
                    print(f"🔄 Found: {document_name}/{filename} (page {page_number})")
            # Strategy 2: Try filename-only match if exact match failed
            elif filename in filename_match_set and filename not in matched_priority_images:
                prioritized_list.append(img_tuple)
                priority_found += 1
                matched_priority_images.add(filename)
                priority_info = path_match_dict.get(filename, {})
                expected_doc = priority_info.get('document', 'unknown')
                # Only show document mismatches and first few matches
                if priority_found <= 2:
                    print(f"🔄 Found: {document_name}/{filename} (page {page_number})")
                if expected_doc != document_name and priority_found <= 2:
                    print(f"   ⚠️ Document mismatch: expected '{expected_doc}'")
            else:
                remaining_images.append(img_tuple)
        
        # Combine priority images first, then remaining images
        final_list = prioritized_list + remaining_images
        
        # Show summary with indication of hidden messages
        if priority_found > 2:
            print(f"   ... and {priority_found - 2} more priority images found")
        
        print("\n📊 Image prioritization summary:")
        print(f"   🔄 Priority images found: {priority_found}/{len(self.priority_images)}")
        print(f"   📝 Regular images: {len(remaining_images)}")
        print(f"   📋 Total images: {len(final_list)}")
        
        if priority_found < len(self.priority_images):
            missing_count = len(self.priority_images) - priority_found
            print(f"   ⚠️ Missing priority images: {missing_count}")
            
            # Show only first few missing images to avoid console spam
            missing_images = [img for img in self.priority_images if img['filename'] not in matched_priority_images]
            if missing_images:
                print("\n🔍 Missing priority images (showing first 3):")
                for i, priority_img in enumerate(missing_images[:3]):
                    print(f"   ❌ {priority_img['document']}/{priority_img['filename']} (page {priority_img['page']})")
                if len(missing_images) > 3:
                    print(f"   ... and {len(missing_images) - 3} more missing images")
        
        return final_list
    
    def remove_from_priority_list(self, document_name: str, filename: str) -> bool:
        """
        Remove an image from the error_images.json file.
        
        Legacy wrapper method that uses ErrorLogManager for centralized handling.
        
        Args:
            document_name: Name of the document
            filename: Name of the image file
            
        Returns:
            True if successfully removed, False otherwise
        """
        try:
            # Construct image path for removal
            image_path = str(Path(document_name) / filename)
            
            # Use ErrorLogManager to remove the error
            success = self.error_log_manager.remove_error_image(image_path)
            
            if success:
                # Update internal priority list
                self.priority_images = [img for img in self.priority_images 
                                      if not (img['document'] == document_name and img['filename'] == filename)]
            
            return success
            
        except Exception as e:
            print(f"Error removing from priority list: {e}")
            return False
    
    def _matches_image(self, entry: Dict[str, Any], document_name: str, filename: str) -> bool:
        """
        Check if an error entry matches the given document and filename.

        Args:
            entry: Error entry from error_images.json.
            document_name: Document name to match.
            filename: Filename to match.

        Returns:
            True if the entry matches, False otherwise.
        """
        # Prioritize direct path matching for robustness
        image_path = entry.get('image_path')
        if image_path:
            # Normalize both paths for a reliable comparison
            normalized_entry_path = Path(image_path).as_posix()
            # Reconstruct the path from the provided document name and filename to compare
            # This is less ideal, but necessary if we don't have the full path of the processed image
            normalized_comparison_path = Path(document_name, filename).as_posix()

            # Check if the reconstructed path is a suffix of the stored path
            # This handles cases where the base directory might differ
            if normalized_entry_path.endswith(normalized_comparison_path):
                return True

        # Fallback to context-based matching if direct path match fails
        context = entry.get('context', {})
        entry_filename = context.get('filename')
        entry_directory = context.get('directory')

        if filename == entry_filename:
            # If directory is also available, check if it matches
            if entry_directory and document_name in Path(entry_directory).as_posix():
                return True
            # If no directory, match by filename alone (less precise)
            elif not entry_directory:
                return True

        return False

    def get_priority_images(self) -> List[Dict[str, Any]]:
        """
        Get the list of priority images.
        
        Returns:
            List of priority image dictionaries
        """
        return self.priority_images
    