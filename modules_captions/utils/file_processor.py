#!/usr/bin/env python3
"""
Batch image processor with database storage and error recovery.

Handles image discovery, AI processing, database storage, and error management
with checkpoint support for interrupted processing recovery.
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .colors import ColorManager
from .path_utils import PathManager

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
    class tqdm:
        """Fallback progress bar when tqdm is unavailable."""
        
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **kwargs):
            self.iterable = iterable or []
            self.total = total or len(self.iterable) if hasattr(self.iterable, '__len__') else 0
            self.desc = desc or ''
            self.n = 0
            
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
                
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def update(self, n=1):
            self.n += n
            
        def set_description(self, desc):
            self.desc = desc
            
        def close(self):
            pass

try:
    from ..db.manager import DatabaseManager
    from .prioritize_error_images import ImagePriorityManager
    from .error_log_manager import ErrorLogManager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db.manager import DatabaseManager
    from utils.prioritize_error_images import ImagePriorityManager
    from utils.error_log_manager import ErrorLogManager

class FileProcessor:
    """Processes images for caption extraction with database storage and error recovery."""
    
    def __init__(self, 
                 root_directory: str,
                 db_manager: DatabaseManager,
                 ai_client: Any,
                 log_dir: str = "logs",
                 commit_interval: int = 10,
                 cooldown_seconds: int = 0,
                 debug_mode: bool = False,
                 checkpoint_dir: Optional[str] = None) -> None:
        """
        Initialize the file processor.
        
        Args:
            root_directory: Root directory containing image files.
            db_manager: Database manager for storing descriptions.
            ai_client: AI client for generating descriptions.
            log_dir: Directory for log files.
            commit_interval: Images to process before database commit.
            cooldown_seconds: Wait time between processing batches.
            debug_mode: Enable enhanced logging.
            checkpoint_dir: Checkpoint directory (defaults to log_dir).
        """
        self.root_directory = Path(root_directory)
        self.db_manager = db_manager
        self.ai_client = ai_client
        self.log_dir = Path(log_dir)
        self.commit_interval = commit_interval
        self.cooldown_seconds = cooldown_seconds
        self.debug_mode = debug_mode
        
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.log_dir
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.completed_checkpoints_file = self.log_dir / "completed_directories.json"
        self.pending_checkpoint_file = self.log_dir / "pending_directory.json"
        self.error_images_file = self.log_dir / "error_images.json"
        
        self.stats = {
            'total_found': 0,
            'total_processed': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'commits_made': 0,
            'start_time': None,
            'end_time': None
        }
        
        self.interrupted = False
        self._last_rate_limit_message = 0
        self._rate_limit_message_interval = 60
        
        from .logging_config import setup_logging
        log_level = "DEBUG" if self.debug_mode else "INFO"
        self.logger = setup_logging(
            log_level=log_level,
            log_file="file_processor.log",
            log_dir=str(self.log_dir),
            enable_colors=True
        )
        self.logger.name = 'file_processor'
        
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        self.priority_manager = ImagePriorityManager(str(Path(__file__).parent.parent))
        self.error_log_manager = ErrorLogManager(log_dir=str(self.log_dir), debug_mode=self.debug_mode)
        
        if self.debug_mode:
            self.logger.debug("🔍 Debug mode activated in FileProcessor")
            self.logger.debug(f"Root directory: {self.root_directory}")
            self.logger.debug(f"Log directory: {self.log_dir}")
            self.logger.debug(f"Commit interval: {self.commit_interval} images")
            self.logger.debug(f"Supported extensions: {self.image_extensions}")
    
    def find_images(self, skip_existing: bool = True) -> List[Tuple[str, str, int, str]]:
        """
        Find image files requiring processing.
        
        Args:
            skip_existing: Skip images with existing database descriptions
            
        Returns:
            List of (document_name, image_path, page_number, image_filename)
        """
        images_to_process = []
        completed_dirs = self._load_completed_directories()
        error_images = self.error_log_manager.get_error_keys_set()
        
        self.logger.info(f"Scanning for images in {self.root_directory}")
        
        directories_with_images = {}
        
        if not self.root_directory.exists():
            self.logger.error(f"Root directory does not exist: {self.root_directory}")
            return images_to_process
        
        all_images = self._discover_images_in_directory()
        
        for image_path in all_images:
            try:
                relative_parent = image_path.parent.relative_to(self.root_directory)
                parent_dir = str(relative_parent) if str(relative_parent) != '.' else ''
            except ValueError:
                continue
            
            if skip_existing and parent_dir in completed_dirs:
                self.stats['total_skipped'] += 1
                continue
            
            image_key = f"{parent_dir}/{image_path.name}" if parent_dir else image_path.name
            if skip_existing and image_key in error_images:
                self.stats['total_skipped'] += 1
                continue
            
            document_name, page_number = self._extract_document_info(image_path)
            image_filename = image_path.name
            
            if skip_existing and self.db_manager.description_exists(
                document_name, page_number, image_filename
            ):
                self.stats['total_skipped'] += 1
                continue
            
            if parent_dir not in directories_with_images:
                directories_with_images[parent_dir] = []
            
            directories_with_images[parent_dir].append((
                document_name, 
                str(image_path), 
                page_number, 
                image_filename,
                parent_dir
            ))
            
            if self.debug_mode:
                self.logger.debug(f"🔍 Found image: {image_path} -> {document_name}, page {page_number}")
        
        pending_dir = self._load_pending_directory()
        if pending_dir and pending_dir in directories_with_images:
            images_to_process.extend(directories_with_images[pending_dir])
            del directories_with_images[pending_dir]
        
        for dir_images in directories_with_images.values():
            images_to_process.extend(dir_images)
        
        self.stats['total_found'] = len(images_to_process)
        self.logger.info(f"Found {len(images_to_process)} images to process across {len(directories_with_images) + (1 if pending_dir else 0)} directories")
        
        if self.debug_mode:
            self.logger.debug(f"🔍 Total files scanned: {self.stats['total_found'] + self.stats['total_skipped']}")
            self.logger.debug(f"🔍 Directories with images: {list(directories_with_images.keys())}")
            self.logger.debug(f"🔍 Completed directories: {completed_dirs}")
            self.logger.debug(f"🔍 Error images: {len(error_images)}")
        
        
        if hasattr(self, 'priority_manager') and self.priority_manager:
            self.priority_manager.load_priority_list()
            images_to_process = self.priority_manager.create_prioritized_image_list(images_to_process)
        
        return images_to_process
    
    def _discover_images_in_directory(self) -> List[Path]:
        """Discover all image files using PathManager."""
        all_images = []
        
        # Use PathManager for consistent image discovery
        for document_name, img_path, page_number, filename, parent_dir in PathManager.find_images_recursive(str(self.root_directory)):
            all_images.append(Path(img_path))
        
        return all_images
    
    def process_images(self, 
                      images: Optional[List[Tuple[str, str, int, str, str]]] = None,
                      force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process images to generate descriptions.
        
        Args:
            images: Image tuples to process, auto-find if None
            force_reprocess: Reprocess existing descriptions
            
        Returns:
            Processing results and statistics
        """
        if images is None:
            images = self.find_images(skip_existing=not force_reprocess)
        else:
            # Update total_found when images are provided directly
            self.stats['total_found'] = len(images)
        
        if not images:
            self.logger.info("No images to process")
            return self._get_final_stats()
        
        self.stats['start_time'] = time.time()
        
        # Enhanced processing start message with statistics
        start_msg = ColorManager.colorize(f"🚀 Starting directory-based processing of {len(images)} images", "green")
        config_msg = ColorManager.colorize(f"   └─ Commit interval: {self.commit_interval} images | Debug: {self.debug_mode}", "cyan")
        
        # Force start messages to be visible
        self.logger.warning(f"\n{start_msg}")
        self.logger.warning(config_msg)
        print(f"\n{start_msg}")
        print(config_msg)
        
        if self.debug_mode:
            self.logger.debug("Processing configuration:")
            self.logger.debug(f"  - Total images: {len(images)}")
            self.logger.debug(f"  - Commit interval: {self.commit_interval}")
            self.logger.debug(f"  - AI client: {type(self.ai_client).__name__}")
            self.logger.debug(f"  - Database path: {self.db_manager.db_path if hasattr(self.db_manager, 'db_path') else 'Unknown'}")
            
            # Log first few images for debugging
            sample_images = images[:3] if len(images) > 3 else images
            self.logger.debug("Sample images to process:")
            for i, (doc, path, page, filename, parent_dir) in enumerate(sample_images):
                self.logger.debug(f"  {i+1}. {filename} (doc: {doc}, page: {page}, dir: {parent_dir})")
            if len(images) > 3:
                self.logger.debug(f"  ... and {len(images) - 3} more images")
        
        # Group images by directory for processing
        images_by_directory = {}
        for image_tuple in images:
            parent_dir = image_tuple[4]  # parent_dir is the 5th element
            if parent_dir not in images_by_directory:
                images_by_directory[parent_dir] = []
            images_by_directory[parent_dir].append(image_tuple)
        
        # Process directories one by one
        for current_dir, dir_images in images_by_directory.items():
            if self.interrupted:
                break
                
            # Enhanced directory processing logs with forced visibility
            dir_start_msg = ColorManager.colorize(f"📂 Starting directory: {current_dir}", 'blue')
            dir_info_msg = ColorManager.colorize(f"   └─ Found {len(dir_images)} images to process", 'cyan')
            
            # Force directory messages to be visible
            self.logger.warning(f"\n{dir_start_msg}")
            self.logger.warning(dir_info_msg)
            print(f"\n{dir_start_msg}")
            print(dir_info_msg)
            self._save_pending_directory(current_dir)
            
            success = self._process_directory_images(dir_images)
            
            if success and not self.interrupted:
                # Mark directory as completed
                self._mark_directory_completed(current_dir)
                self._clear_pending_directory()
                
                # Enhanced completion message with forced visibility
                completion_msg = ColorManager.colorize(f"✅ Directory completed: {current_dir}", 'green')
                stats_msg = ColorManager.colorize(f"   └─ Processed: {len(dir_images)} images | Errors: {self.stats.get('directory_errors', 0)}", 'cyan')
                
                # Force completion messages to be visible
                self.logger.warning(f"\n{completion_msg}")
                self.logger.warning(stats_msg)
                print(f"\n{completion_msg}")
                print(stats_msg)
            elif self.interrupted:
                interrupt_msg = ColorManager.colorize(f"⚠️ Processing interrupted in directory: {current_dir}", 'yellow')
                self.logger.warning(f"\n{interrupt_msg}")
                break
        
        self.stats['end_time'] = time.time()
        
        # Clean up checkpoint if processing completed successfully
        if not self.interrupted:
            self._cleanup_checkpoint()
        
        return self._get_final_stats()
    
    def _process_directory_images(self, dir_images: List[Tuple[str, str, int, str, str]]) -> bool:
        """
        Process all images in a specific directory.
        
        Args:
            dir_images: List of image tuples for the directory.
            
        Returns:
            True if all images were processed successfully, False otherwise.
        """
        pending_descriptions = []
        individual_failures = 0
        
        tqdm_config = self._create_tqdm_config(dir_images)
        
        with tqdm(**tqdm_config) as pbar:
            for image_tuple in dir_images:
                if self.interrupted:
                    return False
                
                success = self._process_single_image(image_tuple, pending_descriptions, pbar)
                
                if not success:
                    individual_failures += 1
                    if self._is_critical_error_detected():
                        return False
                
                pbar.update(1)
                
                # Commit batch if needed
                if len(pending_descriptions) >= self.commit_interval:
                    self._commit_batch(pending_descriptions, pbar)
                    pending_descriptions = []
        
        # Commit remaining descriptions
        if pending_descriptions:
            self._commit_descriptions(pending_descriptions)
        
        # Directory is successful if no failures and not interrupted
        final_success = not self.interrupted and individual_failures == 0
        
        if individual_failures > 0 and not self.interrupted:
            self.logger.warning(f"Directory completed with {individual_failures} individual failures")
        
        return final_success
    
    def _create_tqdm_config(self, dir_images: List[Tuple]) -> Dict[str, Any]:
        """Create tqdm configuration for progress bar with enhanced visibility."""
        directory_name = dir_images[0][4] if dir_images else 'directory'
        config = {
            'total': len(dir_images),
            'desc': ColorManager.colorize(f"🔄 Processing {directory_name}", "cyan"),
            'unit': "img",
            'ncols': 120,  # Wider progress bar
            'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            'leave': True,
            'position': 0,
            'colour': 'green'  # Green progress bar
        }
        
        if TQDM_AVAILABLE:
            config['file'] = None  # Use stdout
            config['dynamic_ncols'] = True
            config['miniters'] = 1  # Update on every iteration
        
        return config
    
    def _process_single_image(self, image_tuple: Tuple, pending_descriptions: List, pbar) -> bool:
        """Process a single image and handle errors."""
        document_name, image_path, page_number, image_filename, parent_dir = image_tuple
        
        try:
            # Update progress bar with detailed information
            current_progress = f"{pbar.n + 1}/{pbar.total}"
            pbar.set_description(ColorManager.colorize(f"🔄 Processing {parent_dir} - {image_filename[:20]}...", "cyan"))
            pbar.set_postfix_str(f"Total: {self.stats['total_processed']} | Errors: {self.stats['total_errors']}")
            
            # Log progress every 10 images or for important milestones
            if (pbar.n + 1) % 10 == 0 or pbar.n + 1 == pbar.total:
                progress_msg = ColorManager.colorize(f"📊 Progress: {current_progress} images processed", "yellow")
                print(f"\r{progress_msg}", end="", flush=True)
            
            # Generate description
            description = self.ai_client.describe(image_path)
            
            if description:
                pending_descriptions.append((document_name, page_number, image_filename, description))
                self.stats['total_processed'] += 1
                self.logger.debug(f"Successfully processed {image_filename}")
                return True
            else:
                raise ValueError("Empty description returned")
                
        except Exception as e:
            return self._handle_image_error(e, image_tuple, pending_descriptions, pbar)
    
    def _handle_image_error(self, error: Exception, image_tuple: Tuple, pending_descriptions: List, pbar) -> bool:
        """Handle errors during image processing."""
        document_name, image_path, page_number, image_filename, parent_dir = image_tuple
        error_msg = str(error)
        
        # Handle rate limit errors with retry
        if self._is_rate_limit_error(error_msg):
            return self._handle_rate_limit(image_tuple, pending_descriptions, pbar, error_msg)
        
        # Handle normal errors
        self.stats['total_errors'] += 1
        
        pbar.clear()
        self.logger.error(ColorManager.colorize(f"❌ Error processing {image_filename}: {error_msg}", "red"))
        pbar.refresh()
        
        # Save error directly using ErrorLogManager
        normalized_directory = self.error_log_manager._normalize_image_path(parent_dir, self.root_directory)
        self.error_log_manager.save_error_image(
            directory=normalized_directory,
            filename=image_filename,
            error_msg=error_msg,
            error_type="processing_error"
        )
        
        # Check for critical server errors
        if self._is_critical_server_error(error_msg):
            pbar.clear()
            self.logger.critical(ColorManager.colorize(f"🚨 Detected server error: {error_msg}. Stopping directory processing.", "red"))
            self.interrupted = True
        
        return False
    
    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if error is a rate limit error."""
        return '429' in error_msg or 'rate limit' in error_msg.lower() or 'quota' in error_msg.lower()
    
    def _is_critical_server_error(self, error_msg: str) -> bool:
        """Check if error is a critical server error."""
        return any(code in error_msg for code in ['400', '500', '503', 'UNAVAILABLE', 'overloaded'])
    
    def _is_critical_error_detected(self) -> bool:
        """Check if a critical error was detected and processing should stop."""
        return self.interrupted
    
    def _handle_rate_limit(self, image_tuple: Tuple, pending_descriptions: List, pbar, error_msg: str) -> bool:
        """Handle rate limit errors with cooling and retry."""
        document_name, image_path, page_number, image_filename, parent_dir = image_tuple
        
        # Throttle rate limit messages to avoid spam
        import time
        current_time = time.time()
        
        pbar.clear()
        if (current_time - self._last_rate_limit_message) >= self._rate_limit_message_interval:
            self.logger.warning(ColorManager.colorize(f"⚠️ Rate limit detected for {image_filename}. Applying cooling and retry...", "yellow"))
            self._last_rate_limit_message = current_time
        else:
            # Log at debug level if within throttle interval
            self.logger.debug(f"Rate limit detected for {image_filename}. Applying cooling and retry... [Throttled]")
        pbar.refresh()
        
        # Extract retry delay
        retry_delay = self._extract_retry_delay(error_msg)
        
        # Apply cooling period
        if not self._apply_cooling_period(retry_delay):
            return False
        
        # Retry processing
        return self._retry_image_processing(image_tuple, pending_descriptions, pbar)
    
    def _extract_retry_delay(self, error_msg: str) -> int:
        """Extract retry delay from error message."""
        retry_delay = 60  # Default
        if 'retryDelay' in error_msg:
            import re
            delay_match = re.search(r"'retryDelay': '(\d+)s'", error_msg)
            if delay_match:
                retry_delay = int(delay_match.group(1)) + 5  # Add buffer
        return retry_delay
    
    def _apply_cooling_period(self, retry_delay: int) -> bool:
        """Apply cooling period with interruption support and progress bar."""
        # Throttle cooling messages to avoid spam
        import time
        current_time = time.time()
        
        if (current_time - self._last_rate_limit_message) >= self._rate_limit_message_interval:
            self.logger.info(f"🧊 Cooling for {retry_delay} seconds due to rate limit...")
            self._last_rate_limit_message = current_time
        else:
            self.logger.debug(f"Cooling for {retry_delay} seconds due to rate limit... [Throttled]")
        
        if TQDM_AVAILABLE:
            # Use tqdm progress bar for cooling period
            with tqdm(total=retry_delay, desc="🧊 Rate limit cooling", unit="s", 
                     bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total}s [{elapsed}<{remaining}]") as pbar:
                for second in range(retry_delay):
                    if self.interrupted:
                        pbar.close()
                        self.logger.info("⚠️ Cooling interrupted by user")
                        return False
                    time.sleep(1)
                    pbar.update(1)
        else:
            # Fallback to original logging method if tqdm not available
            for second in range(retry_delay):
                if self.interrupted:
                    self.logger.info("⚠️ Cooling interrupted by user")
                    return False
                time.sleep(1)
                if second % 10 == 0:
                    remaining = retry_delay - second
                    self.logger.info(f"🧊 Cooling... {remaining}s remaining")
        
        self.logger.info("✅ Cooling period completed")
        return True
    
    def _retry_image_processing(self, image_tuple: Tuple, pending_descriptions: List, pbar) -> bool:
        """Retry image processing after cooling."""
        document_name, image_path, page_number, image_filename, parent_dir = image_tuple
        
        if self.interrupted:
            return False
            
        self.logger.info(f"✅ Cooling completed. Retrying {image_filename}...")
        
        try:
            description = self.ai_client.describe(image_path)
            if description:
                pending_descriptions.append((document_name, page_number, image_filename, description))
                self.stats['total_processed'] += 1
                self.logger.debug(f"Successfully processed {image_filename} after retry")
                return True
            else:
                raise ValueError("Empty description returned after retry")
        except Exception as retry_e:
            retry_error_msg = str(retry_e)
            if self._is_rate_limit_error(retry_error_msg):
                # Throttle critical rate limit messages
                import time
                current_time = time.time()
                
                if (current_time - self._last_rate_limit_message) >= self._rate_limit_message_interval:
                    self.logger.critical(ColorManager.colorize("🚨 Rate limit persists after cooling. Stopping directory processing.", "red"))
                    self._last_rate_limit_message = current_time
                else:
                    self.logger.debug("Rate limit persists after cooling. Stopping directory processing. [Throttled]")
                self.interrupted = True
                return False
            else:
                # Different error after retry, handle normally
                self.stats['total_errors'] += 1
                self.logger.error(ColorManager.colorize(f"❌ Error processing {image_filename} after retry: {retry_error_msg}", "red"))
                # Save error directly using ErrorLogManager
                normalized_directory = self.error_log_manager._normalize_image_path(parent_dir, self.root_directory)
                self.error_log_manager.save_error_image(
                    directory=normalized_directory,
                    filename=image_filename,
                    error_msg=retry_error_msg,
                    error_type="retry_failed"
                )
                return False
    
    def _commit_batch(self, pending_descriptions: List, pbar) -> None:
        """Commit a batch of descriptions with progress bar management."""
        pbar.clear()
        self._commit_descriptions(pending_descriptions)
        pbar.refresh()
    
    # Checkpoint management methods for directory-based processing
    def _load_completed_directories(self) -> set:
        """Load the set of completed directories."""
        if not self.completed_checkpoints_file.exists():
            return set()
        
        try:
            with open(self.completed_checkpoints_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('completed_directories', []))
        except Exception as e:
            self.logger.error(f"Error loading completed directories: {e}")
            return set()
    
    def _mark_directory_completed(self, directory: str) -> None:
        """Mark a directory as completed."""
        completed_dirs = self._load_completed_directories()
        completed_dirs.add(directory)
        
        data = {
            'completed_directories': list(completed_dirs),
            'last_updated': time.time()
        }
        
        try:
            with open(self.completed_checkpoints_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Marked directory as completed: {directory}")
        except Exception as e:
            self.logger.error(f"Error marking directory as completed: {e}")
    
    def _load_pending_directory(self) -> Optional[str]:
        """Load the currently pending directory."""
        if not self.pending_checkpoint_file.exists():
            return None
        
        try:
            with open(self.pending_checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('pending_directory')
        except Exception as e:
            self.logger.error(f"Error loading pending directory: {e}")
            return None
    
    def _save_pending_directory(self, directory: str) -> None:
        """Save the currently processing directory as pending."""
        data = {
            'pending_directory': directory,
            'timestamp': time.time()
        }
        
        try:
            with open(self.pending_checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved pending directory: {directory}")
        except Exception as e:
            self.logger.error(f"Error saving pending directory: {e}")
    
    def _clear_pending_directory(self) -> None:
        """Clear the pending directory file."""
        try:
            if self.pending_checkpoint_file.exists():
                self.pending_checkpoint_file.unlink()
                self.logger.debug("Cleared pending directory")
        except Exception as e:
            self.logger.error(f"Error clearing pending directory: {e}")
    
    def _commit_descriptions(self, descriptions: List[Tuple[str, int, str, str]]) -> None:
        """
        Commit a batch of descriptions to the database.
        
        Args:
            descriptions: List of description tuples to commit.
        """
        if not descriptions:
            return
            
        try:
            inserted_count = self.db_manager.batch_insert_descriptions(descriptions)
            if inserted_count != len(descriptions):
                self.logger.warning(f"Expected to insert {len(descriptions)} descriptions, but only {inserted_count} were inserted")
                self.stats['total_errors'] += len(descriptions) - inserted_count
            
            self.stats['commits_made'] += 1
            
            # Enhanced commit message with better formatting and forced visibility
            commit_msg = ColorManager.colorize(f"💾 Committed {len(descriptions)} descriptions to database", "green")
            commit_details = ColorManager.colorize(f"   └─ Commit #{self.stats['commits_made']} | Total processed: {self.stats['total_processed']}", "cyan")
            
            # Force commit messages to be visible by using WARNING level
            self.logger.warning(f"\n{commit_msg}")
            self.logger.warning(commit_details)
            
            # Also print directly to ensure visibility
            print(f"\n{commit_msg}")
            print(commit_details)   
           
        except Exception as e:
            error_msg = ColorManager.colorize(f"❌ Failed to commit {len(descriptions)} descriptions: {str(e)}", "red")
            self.logger.error(error_msg)
            self.stats['total_errors'] += len(descriptions)
    
    
    def _extract_document_info(self, image_path: Path) -> Tuple[str, int]:
        """
        Extract document name and page number from image path using PathManager.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Tuple of (document_name, page_number).
        """
        # Use PathManager for consistent page number extraction
        page_number = PathManager.extract_page_number(image_path.name)
        
        # Use parent directory as document name, or filename if no clear structure
        if image_path.parent != self.root_directory:
            document_name = image_path.parent.name
        else:
            # Extract document name from filename (remove page info)
            filename = image_path.stem
            document_name = re.sub(r'[_-]?page[_-]?\d+.*$', '', filename, flags=re.IGNORECASE)
            if not document_name:
                document_name = filename
        
        return document_name, page_number
    
    def _cooldown(self) -> None:
        """
        Wait for the cooldown period between batches.
        """
        if self.cooldown_seconds <= 0:
            return
        
        self.logger.info(f"Cooling down for {self.cooldown_seconds} seconds...")
        
        # Interruptible sleep
        for _ in range(self.cooldown_seconds):
            if self.interrupted:
                break
            time.sleep(1)
    
    def _save_checkpoint(self, processed_count: int, total_count: int) -> None:
        """
        Save processing checkpoint.
        
        Args:
            processed_count: Number of images processed so far.
            total_count: Total number of images to process.
        """
        checkpoint_data = {
            'processed_count': processed_count,
            'total_count': total_count,
            'stats': self.stats.copy(),
            'timestamp': time.time()
        }
        
        checkpoint_file = self.checkpoint_dir / 'processing_checkpoint.json'
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Checkpoint saved: {processed_count}/{total_count} processed")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the last processing checkpoint.
        
        Returns:
            Dict with checkpoint data if available, None otherwise.
        """
        checkpoint_file = self.checkpoint_dir / 'processing_checkpoint.json'
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def interrupt(self) -> None:
        """
        Signal the processor to stop at the next safe point.
        """
        self.interrupted = True
        self.logger.info("Interrupt signal received")
    
    def _cleanup_checkpoint(self) -> None:
        """
        Remove checkpoint file after successful completion.
        """
        checkpoint_file = self.checkpoint_dir / 'processing_checkpoint.json'
        
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                self.logger.debug("Checkpoint file cleaned up")
        except Exception as e:
            self.logger.warning(f"Could not clean up checkpoint file: {e}")
    
    def _get_final_stats(self) -> Dict[str, Any]:
        """
        Calculate and return final processing statistics.
        
        Returns:
            Dictionary containing processing statistics.
        """
        if self.stats['start_time'] and self.stats['end_time']:
            total_time = self.stats['end_time'] - self.stats['start_time']
        else:
            total_time = 0
        
        return {
            'total_images': self.stats['total_found'],
            'total_processed': self.stats['total_processed'],
            'total_skipped': self.stats['total_skipped'],
            'total_errors': self.stats['total_errors'],
            'commits_made': self.stats.get('commits_made', 0),
            'total_time_seconds': round(total_time, 2),
            'average_time_per_image': round(total_time / max(self.stats['total_processed'], 1), 2),
            'success_rate': round((self.stats['total_processed'] / max(self.stats['total_found'], 1)) * 100, 2),
            'interrupted': self.interrupted
        }
    
    def get_processing_summary(self) -> str:
        """
        Get a human-readable summary of processing results.
        
        Returns:
            Formatted string with processing summary.
        """
        stats = self._get_final_stats()
        
        summary = f"""
                        === Processing Summary ===
                        Images found: {stats['total_images']}
                        Images processed: {stats['total_processed']}
                        Images skipped: {stats['total_skipped']}
                        Errors: {stats['total_errors']}
                        Commits made: {stats['commits_made']}
                        Total time: {stats['total_time_seconds']:.2f} seconds
                        Average time per image: {stats['average_time_per_image']:.2f} seconds
                        Success rate: {stats['success_rate']:.1f}%
                        Interrupted: {stats['interrupted']}
                        =========================
                    """
        
        return summary
    
    def _validate_directory_completion_by_hierarchy(self, directory_path: str) -> bool:
        """
        Validate if a directory is truly completed based on its hierarchical level.
        
        Args:
            directory_path: Path to the directory to validate
            
        Returns:
            bool: True if directory is genuinely completed, False otherwise
        """
        dir_path = Path(directory_path)
        relative_path = dir_path.relative_to(self.root_directory) if dir_path.is_absolute() else Path(directory_path)
        
        # Determine hierarchy level based on path depth
        path_parts = relative_path.parts
        
        if len(path_parts) == 0:  # Root directory
            return self._is_root_directory_complete()
        elif len(path_parts) == 1:  # Year directory
            return self._is_year_directory_complete(path_parts[0])
        elif len(path_parts) == 2:  # Month directory
            return self._is_month_directory_complete(path_parts[0], path_parts[1])
        elif len(path_parts) == 3:  # Day directory
            return self._is_day_directory_complete(path_parts[0], path_parts[1], path_parts[2])
        else:
            # Deeper levels - validate as day directory
            return self._is_day_directory_complete(path_parts[0], path_parts[1], path_parts[2])
    
    def _is_day_directory_complete(self, year: str, month: str, day: str) -> bool:
        """
        Check if a day directory is completely processed.
        
        Args:
            year: Year part of the path
            month: Month part of the path
            day: Day part of the path
            
        Returns:
            bool: True if all images in the day directory are processed
        """
        day_path = self.root_directory / year / month / day
        if not day_path.exists():
            return True
            
        # Check if all images in the directory have descriptions
        for image_file in day_path.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in self.image_extensions:
                document_name, page_number = self._extract_document_info(image_file)
                if not self.db_manager.description_exists(document_name, page_number, image_file.name):
                    return False
                    
        return True
    
    def _load_error_log_entries(self) -> List[Dict[str, Any]]:
        """
        Load error log entries from the JSON file.
        
        Returns:
            List of error entries converted from the error images dictionary
        """
        try:
            error_images = self.error_log_manager.get_error_keys_set()
            # Convert dictionary format to list format for compatibility
            entries = []
            for image_path, error_data in error_images.items():
                # Preserve full context if available, otherwise create basic context
                context = error_data.get('context', {})
                if not context:
                    context = {
                        'image_path': image_path,
                        'directory': error_data.get('directory', ''),
                        'filename': error_data.get('filename', '')
                    }
                else:
                    # Ensure image_path is in context
                    context['image_path'] = image_path
                
                entry = {
                    'context': context,
                    'error_message': error_data.get('error_message', ''),
                    'timestamp': error_data.get('timestamp', 0)
                }
                
                # Preserve additional fields if they exist
                for field in ['error_type', 'error_class']:
                    if field in error_data:
                        entry[field] = error_data[field]
                        
                entries.append(entry)
            return entries
        except Exception as e:
            self.logger.warning(f"Could not load error log entries: {e}")
            return []
    
    def _extract_error_images_for_processing(self) -> List[Tuple[str, str, int, str, str]]:
        """
        Extract images from error log for priority processing.
        
        Returns:
            List of tuples (document_name, image_path, page_number, image_filename, parent_dir)
        """
        error_images = []
        
        try:
            if not self.error_images_file.exists():
                self.logger.info("No error images file found")
                return []
                
            # Load error data directly as list
            with open(self.error_images_file, 'r', encoding='utf-8') as f:
                error_data = json.load(f)
            
            if not isinstance(error_data, list):
                self.logger.warning(f"Expected list format in error_images.json, got {type(error_data)}")
                return []
            
            for entry in error_data:
                context = entry.get('context', {})
                image_path_str = context.get('image_path', '')
                
                if image_path_str:
                    image_path = Path(image_path_str)
                    
                    # Check if image file still exists
                    if image_path.exists():
                        # Extract document and page information
                        document_name, page_number = self._extract_document_info(image_path)
                        
                        # Calculate parent directory relative to root
                        try:
                            if self.root_directory in image_path.parents:
                                relative_parent = image_path.parent.relative_to(self.root_directory)
                                parent_dir = str(relative_parent) if str(relative_parent) != '.' else ''
                            else:
                                parent_dir = str(image_path.parent)
                        except ValueError:
                            parent_dir = str(image_path.parent)
                        
                        error_images.append((
                            document_name,
                            str(image_path),
                            page_number,
                            image_path.name,
                            parent_dir
                        ))
                    else:
                        # Remove non-existent images from error log
                        self.error_log_manager.remove_error_image(image_path_str)
            
            self.logger.info(f"🔄 Found {len(error_images)} error images for priority processing")
            return error_images
            
        except Exception as e:
            self.logger.error(f"Error extracting error images for processing: {e}")
            return []
    
    def _process_error_images_with_priority(self, error_images: List[Tuple[str, str, int, str, str]]) -> None:
        """
        Process images that previously caused errors with priority.
        
        Args:
            error_images: List of error image tuples to process
        """
        if not error_images:
            self.logger.info("📋 No error images to process")
            return
        
        self.logger.info(f"🚀 Starting priority processing of {len(error_images)} error images")
        
        # Show summary of error images instead of individual details
        if len(error_images) <= 5:
            self.logger.info("📋 Error images to process:")
            for i, (doc_name, image_path, page_num, filename, parent_dir) in enumerate(error_images, 1):
                self.logger.info(f"   {i}. {filename} (doc: {doc_name}, page: {page_num})")
        else:
            self.logger.info(f"📋 Processing {len(error_images)} error images (showing first 3):")
            for i, (doc_name, image_path, page_num, filename, parent_dir) in enumerate(error_images[:3], 1):
                self.logger.info(f"   {i}. {filename} (doc: {doc_name}, page: {page_num})")
            self.logger.info(f"   ... and {len(error_images) - 3} more error images")
        
        pending_descriptions = []
        processed_count = 0
        
        # Use tqdm for progress tracking if available
        try:
            from tqdm import tqdm
            iterator = tqdm(error_images, desc="🔄 Processing error images", 
                          unit="img", colour="yellow", ncols=100,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        except ImportError:
            iterator = error_images
        
        for doc_name, image_path, page_num, filename, parent_dir in iterator:
            try:
                # Process the image
                description = self.ai_client.describe(image_path)
                
                if description:
                    pending_descriptions.append((doc_name, page_num, filename, description))
                    
                    processed_count += 1
                    self.logger.info(f"✅ Successfully reprocessed: {filename}")
                    
                    # Remove from error log after successful processing
                    self.error_log_manager.remove_error_image(image_path)
                    
                    # Commit in batches
                    if len(pending_descriptions) >= self.commit_interval:
                        self._commit_descriptions(pending_descriptions)
                        pending_descriptions.clear()
                        
                else:
                    self.logger.warning(f"⚠️ Empty description for error image: {filename}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to reprocess error image {filename}: {str(e)[:100]}..." if len(str(e)) > 100 else f"❌ Failed to reprocess error image {filename}: {e}")
                # Keep the error in the log for future retry
                continue
        
        # Commit any remaining descriptions
        if pending_descriptions:
            self._commit_descriptions(pending_descriptions)
        
        # Final summary
        success_rate = (processed_count / len(error_images) * 100) if error_images else 0
        self.logger.info(f"\n📊 Priority processing completed:")
        self.logger.info(f"   ✅ Successfully processed: {processed_count}/{len(error_images)} images ({success_rate:.1f}%)")
        if processed_count < len(error_images):
            failed_count = len(error_images) - processed_count
            self.logger.info(f"   ❌ Failed to process: {failed_count} images (will retry in next run)")
        self.logger.info(f"   💾 Total commits made: {self.stats.get('commits_made', 0)}")
    
    def _is_month_directory_complete(self, year: str, month: str) -> bool:
        """
        Check if a month directory is completely processed.
        
        Args:
            year: Year part of the path
            month: Month part of the path
            
        Returns:
            bool: True if all day directories in the month are complete
        """
        month_path = self.root_directory / year / month
        if not month_path.exists():
            return True  # Non-existent directory is considered complete
            
        # Check all day directories in the month
        for day_dir in month_path.iterdir():
            if day_dir.is_dir():
                if not self._is_day_directory_complete(year, month, day_dir.name):
                    return False
                    
        return True
    
    def _is_year_directory_complete(self, year: str) -> bool:
        """
        Check if a year directory is completely processed.
        
        Args:
            year: Year part of the path
            
        Returns:
            bool: True if all month directories in the year are complete
        """
        year_path = self.root_directory / year
        if not year_path.exists():
            return True  # Non-existent directory is considered complete
            
        # Check all month directories in the year
        for month_dir in year_path.iterdir():
            if month_dir.is_dir():
                if not self._is_month_directory_complete(year, month_dir.name):
                    return False
                    
        return True
    
    def _is_root_directory_complete(self) -> bool:
        """
        Check if the root directory is completely processed.
        
        Returns:
            bool: True if all year directories are complete
        """
        if not self.root_directory.exists():
            return True  # Non-existent directory is considered complete
            
        # Check all year directories in the root
        for year_dir in self.root_directory.iterdir():
            if year_dir.is_dir():
                if not self._is_year_directory_complete(year_dir.name):
                    return False
                    
        return True