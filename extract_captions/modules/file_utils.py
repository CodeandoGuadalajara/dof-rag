import os
import time
import json
import glob
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, TypeVar, cast

# Define a generic type for the client
ClientType = TypeVar('ClientType')

class FileUtil:
    """
    Class for managing image file processing and tracking progress.
    
    This utility handles:
    - Searching for image files in directory trees
    - Processing images in batches with cooling periods
    - Checkpoint management for resumable operations
    - Tracking of successful and failed operations
    - Retry mechanisms for failed images
    """
    
    def __init__(self, root_directory: str = ".", client: Optional[ClientType] = None, 
                 batch_size: int = 10, cooling_period: int = 5) -> None:
        """
        Initializes the file utility with processing parameters.
        
        Parameters:
            root_directory (str): Root directory where images will be searched.
            client (Optional[ClientType]): AI client instance for processing images.
            batch_size (int): Number of images to process in each batch.
            cooling_period (int): Wait time between batches in seconds.
        """
        self.root_directory = root_directory
        self.client = client
        self.batch_size = batch_size
        self.cooling_period = cooling_period
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_file = "processing_checkpoint.json"
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            try:
                os.makedirs(self.checkpoint_dir)
            except Exception as e:
                print(f"Warning: Could not create checkpoint directory: {e}")
    
    def process_images(self, force_overwrite: bool = False, 
                       interrupt_flag_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes all images in the root directory and subdirectories.
        
        Parameters:
            force_overwrite (bool): If True, already processed images will be reprocessed.
            interrupt_flag_file (Optional[str]): Path to the flag file for interruption detection.
            
        Returns:
            Dict[str, Any]: Processing statistics dictionary.
        """
        # Start time for statistics
        start_time = time.time()
        
        # Search for images and load checkpoint if it exists
        image_paths = self._get_all_images()
        checkpoint_data = self._load_checkpoint()
        processed_images = checkpoint_data.get("processed_images", [])
        last_index = checkpoint_data.get("last_index", -1)
        stats = checkpoint_data.get("stats", self._init_stats())
        
        # Update initial statistics
        stats["total_images"] = len(image_paths)
        stats["start_time"] = start_time
        stats["last_resumed"] = time.time()
        
        # Optimization: synchronize checkpoint with existing .txt files
        if not force_overwrite:
            processed_images, newly_found = self._sync_checkpoint_with_existing_files(image_paths, processed_images)
            if newly_found > 0:
                stats["auto_detected"] += newly_found
        
        print(f"Total images found: {len(image_paths)}")
        print(f"Images already processed: {len(processed_images)}")
        
        if not image_paths:
            print("No images found to process.")
            return stats
        
        # Continue from where it left off
        start_index = last_index + 1 if last_index >= 0 else 0
        
        if start_index >= len(image_paths):
            print("All images have already been processed.")
            return stats
        
        # Create/update checkpoint before starting processing
        self._save_checkpoint(processed_images, last_index, stats)
        
        # Variables for statistics during this process
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        # Process images in batches
        for batch_start in range(start_index, len(image_paths), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(image_paths))
            batch = image_paths[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//self.batch_size + 1} ({batch_end}/{len(image_paths)} images)")
            
            # Check for interruption before starting the batch
            if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                print("\n⚠️ Processing interrupted by user request before starting the batch.")
                print(f"Progress saved in checkpoint: {batch_start}/{len(image_paths)} images")
                
                # Update final statistics
                stats["end_time"] = time.time()
                stats["processing_time"] += stats["end_time"] - stats["last_resumed"]
                stats["interrupted"] = True
                self._save_checkpoint(processed_images, last_index, stats)
                
                return stats
            
            # Process each image in the batch
            for i, img_path in enumerate(batch):
                # Check for interruption before processing each image
                if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                    print("\n⚠️ Processing interrupted by user request.")
                    print(f"Progress saved in checkpoint: {batch_start + i}/{len(image_paths)} images")
                    
                    # Update final statistics
                    stats["end_time"] = time.time()
                    stats["processing_time"] += stats["end_time"] - stats["last_resumed"]
                    stats["interrupted"] = True
                    
                    # Save checkpoint with the last completely processed image
                    self._save_checkpoint(processed_images, batch_start + i - 1, stats)
                    
                    return stats
                
                if img_path in processed_images and not force_overwrite:
                    print(f"Skipping (already processed): {img_path}")
                    skipped_count += 1
                    continue
                
                print(f"Processing: {img_path}")
                result = self.client.process_imagen(img_path, force_overwrite)
                
                if result.get("status") == "processed":
                    processed_images.append(img_path)
                    success_count += 1
                    stats["successful"] += 1
                    print(f"✓ Processed in {result.get('process_time', 0):.2f}s")
                elif result.get("status") == "already_processed":
                    processed_images.append(img_path)
                    stats["already_processed"] += 1
                    print("✓ Already processed previously")
                else:
                    error_count += 1
                    stats["errors"] += 1
                    print(f"✗ Error: {result.get('error', 'Unknown error')}")
            
            # Save checkpoint after each batch
            last_index = batch_end - 1
            stats["last_batch"] = batch_start//self.batch_size + 1
            stats["last_image_index"] = last_index
            self._save_checkpoint(processed_images, last_index, stats)
            
            # Check if the interrupt flag file exists
            if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                print("\n⚠️ Processing paused by user request.")
                print(f"Progress saved in checkpoint: {batch_end}/{len(image_paths)} images")
                
                # Update final statistics
                stats["end_time"] = time.time()
                stats["processing_time"] += stats["end_time"] - stats["last_resumed"]
                stats["interrupted"] = True
                self._save_checkpoint(processed_images, last_index, stats)
                
                return stats
            
            # Cooling period between batches, except if it's the last batch
            if batch_end < len(image_paths):
                print(f"\nCooling down for {self.cooling_period} seconds...")
                # Check for interruption during cooling every second
                for i in range(self.cooling_period):
                    if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                        print("\n⚠️ Processing interrupted during cooling.")
                        print(f"Progress saved in checkpoint: {batch_end}/{len(image_paths)} images")
                        
                        # Update final statistics
                        stats["end_time"] = time.time()
                        stats["processing_time"] += stats["end_time"] - stats["last_resumed"]
                        stats["interrupted"] = True
                        self._save_checkpoint(processed_images, last_index, stats)
                        
                        return stats
                    time.sleep(1)
                
        # Finalize statistics
        stats["end_time"] = time.time()
        stats["processing_time"] += stats["end_time"] - stats["last_resumed"]
        stats["complete"] = True
        stats["interrupted"] = False
        
        # Save final checkpoint
        self._save_checkpoint(processed_images, last_index, stats, is_final=True)
        
        print("\n✅ Processing completed")
        print(f"Summary: {success_count} images processed, {error_count} errors, {skipped_count} skipped")
        
        return stats
    
    def retry_failed_images(self, interrupt_flag_file: Optional[str] = None) -> None:
        """
        Retries processing images that previously failed.
        
        Parameters:
            interrupt_flag_file (Optional[str]): Path to the flag file for interruption detection.
        """
        failed_images = self._load_failed_images()
        
        if not failed_images:
            print("No failed images to retry.")
            return
        
        print(f"Retrying {len(failed_images)} failed images...")
        successfully_processed = 0
        
        # Process failed images in batches
        for batch_start in range(0, len(failed_images), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(failed_images))
            batch = failed_images[batch_start:batch_end]
            
            # Check for interruption before starting the batch
            if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                print("\n⚠️ Retry interrupted by user request.")
                return
            
            for i, img_path in enumerate(batch):
                # Check for interruption before processing each image
                if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                    print("\n⚠️ Retry interrupted by user request.")
                    return
                
                print(f"Retrying: {img_path}")
                result = self.client.process_imagen(img_path, force_overwrite=True)
                
                if result.get("status") == "processed":
                    self._remove_from_failed_list(img_path)
                    successfully_processed += 1
                    print("✓ Successfully processed on second attempt")
                else:
                    print(f"✗ Still failing: {result.get('error', 'Unknown error')}")
            
            # Check if the interrupt flag file exists
            if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                print("\n⚠️ Retry paused by user request.")
                break
            
            # Cooling period between batches, except if it's the last
            if batch_end < len(failed_images):
                print(f"\nCooling down for {self.cooling_period} seconds...")
                # Check for interruption during cooling every second
                for i in range(self.cooling_period):
                    if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                        print("\n⚠️ Retry interrupted during cooling.")
                        return
                    time.sleep(1)
        
        if successfully_processed > 0:
            print(f"\nSuccessfully processed {successfully_processed} of {len(failed_images)} failed images.")
        else:
            print("\nCould not process any failed images.")
    
    def _get_all_images(self) -> List[str]:
        """
        Gets all image paths in the root directory and subdirectories.
        
        Returns:
            List[str]: List of paths to valid image files.
        """
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        image_paths = []
        
        for root, _, files in os.walk(self.root_directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(root, file))
        
        return sorted(image_paths)
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """
        Loads checkpoint from previous processing, if it exists.
        
        Returns:
            Dict[str, Any]: Checkpoint data with processed images and statistics.
        """
        # Try to load from the main file
        full_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    print(f"Loaded checkpoint from {checkpoint.get('timestamp_formatted', 'unknown date')}")
                    return checkpoint
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                
                # Try to load any backup
                backups = self._get_checkpoint_backups()
                if backups:
                    latest_backup = backups[-1]
                    print(f"Trying to load most recent backup: {os.path.basename(latest_backup)}")
                    try:
                        with open(latest_backup, 'r', encoding='utf-8') as f:
                            checkpoint = json.load(f)
                            print(f"Loaded backup from {checkpoint.get('timestamp_formatted', 'unknown date')}")
                            return checkpoint
                    except Exception as e2:
                        print(f"Error loading backup: {str(e2)}")
        
        return {"processed_images": [], "last_index": -1, "stats": self._init_stats()}
    
    def _save_checkpoint(self, processed_images: List[str], last_index: int, 
                        stats: Dict[str, Any], is_final: bool = False) -> None:
        """
        Saves a checkpoint of the current processing state.
        
        Parameters:
            processed_images (List[str]): List of already processed images.
            last_index (int): Index of the last processed image.
            stats (Dict[str, Any]): Processing statistics.
            is_final (bool): If True, generates a checkpoint with a special identifier.
        """
        # Create directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            try:
                os.makedirs(self.checkpoint_dir)
            except Exception as e:
                print(f"Error creating checkpoint directory: {str(e)}")
                return
        
        # Current data
        timestamp = time.time()
        timestamp_formatted = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Make sure processing time is updated
        if stats.get("end_time") is None:
            stats["end_time"] = time.time()
            stats["processing_time"] += stats["end_time"] - stats.get("last_resumed", stats.get("start_time", stats["end_time"]))
        
        checkpoint = {
            "processed_images": processed_images,
            "last_index": last_index,
            "timestamp": timestamp,
            "timestamp_formatted": timestamp_formatted,
            "stats": stats
        }
        
        # Save main checkpoint
        full_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        try:
            # Backup the previous one
            if os.path.exists(full_path):
                # Limit number of backups
                self._manage_checkpoint_backups()
                
                # New name for the backup
                backup_name = f"checkpoint_{int(timestamp)}.json"
                backup_path = os.path.join(self.checkpoint_dir, backup_name)
                shutil.copy2(full_path, backup_path)
            
            # Save the new checkpoint
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            
            # If it's the final checkpoint, create a copy with special format
            if is_final:
                final_name = f"checkpoint_FINAL_{int(timestamp)}.json"
                final_path = os.path.join(self.checkpoint_dir, final_name)
                with open(final_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                print(f"Saved final checkpoint: {final_name}")
            
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
    
    def _manage_checkpoint_backups(self, max_backups: int = 10) -> None:
        """
        Manages the number of checkpoint backup files.
        Deletes the oldest ones if the maximum number is exceeded.
        
        Parameters:
            max_backups (int): Maximum number of backups to keep.
        """
        backups = self._get_checkpoint_backups()
        
        # If there are more backups than the maximum, delete the oldest ones
        if len(backups) > max_backups:
            backups_to_delete = backups[:(len(backups) - max_backups)]
            for backup in backups_to_delete:
                try:
                    os.remove(backup)
                except Exception:
                    pass
    
    def _get_checkpoint_backups(self) -> List[str]:
        """
        Gets the list of checkpoint backup files sorted by date.
        
        Returns:
            List[str]: List of paths to backup files, oldest first.
        """
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_*.json")
        backups = glob.glob(pattern)
        # Sort by modification date (oldest first)
        return sorted(backups, key=os.path.getmtime)
    
    def _sync_checkpoint_with_existing_files(self, image_paths: List[str], 
                                           processed_images: List[str]) -> Tuple[List[str], int]:
        """
        Synchronizes the list of processed images with existing .txt files.
        
        Parameters:
            image_paths (List[str]): List of found image paths.
            processed_images (List[str]): Current list of processed images from checkpoint.
            
        Returns:
            Tuple[List[str], int]: Updated list and number of newly detected processed files.
        """
        # Convert to set for more efficient operations
        processed_set = set(processed_images)
        newly_found = 0
        
        for img_path in image_paths:
            # If already marked as processed, skip it
            if img_path in processed_set:
                continue
                
            # Check if the description file exists
            txt_file = f"{os.path.splitext(img_path)[0]}.txt"
            if os.path.exists(txt_file):
                try:
                    # Check that the file is not empty or contains an error
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content and not content.startswith("ERROR:"):
                            processed_set.add(img_path)
                            newly_found += 1
                except Exception:
                    # If there's an error reading, assume it's not properly processed
                    pass
        
        # If new processed files were found, update the list
        if newly_found > 0:
            print(f"Found {newly_found} additional already processed images that were not in the checkpoint.")
            updated_list = list(processed_set)
            return updated_list, newly_found
            
        return processed_images, 0
    
    def get_available_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Gets the list of available checkpoints with metadata.
        
        Returns:
            List[Dict[str, Any]]: List of checkpoints with their basic information.
        """
        checkpoints = []
        
        # Check the main checkpoint
        main_checkpoint = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        if os.path.exists(main_checkpoint):
            try:
                with open(main_checkpoint, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    checkpoints.append({
                        "path": main_checkpoint,
                        "timestamp": data.get("timestamp"),
                        "timestamp_formatted": data.get("timestamp_formatted"),
                        "processed": len(data.get("processed_images", [])),
                        "name": "Current",
                        "is_final": data.get("stats", {}).get("complete", False),
                        "interrupted": data.get("stats", {}).get("interrupted", False)
                    })
            except Exception:
                pass
        
        # Get all backups
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_*.json")
        backup_files = glob.glob(pattern)
        
        for backup in backup_files:
            try:
                with open(backup, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    is_final = "FINAL" in os.path.basename(backup)
                    checkpoints.append({
                        "path": backup,
                        "timestamp": data.get("timestamp"),
                        "timestamp_formatted": data.get("timestamp_formatted"),
                        "processed": len(data.get("processed_images", [])),
                        "name": os.path.basename(backup),
                        "is_final": is_final or data.get("stats", {}).get("complete", False),
                        "interrupted": data.get("stats", {}).get("interrupted", False)
                    })
            except Exception:
                pass
        
        # Sort by timestamp (most recent first)
        return sorted(checkpoints, key=lambda x: x.get("timestamp", 0), reverse=True)
    
    def load_specific_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Loads a specific checkpoint and sets it as the current checkpoint.
        
        Parameters:
            checkpoint_path (str): Path to the checkpoint file.
            
        Returns:
            Optional[Dict[str, Any]]: Checkpoint data or None if there's an error.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file {checkpoint_path} does not exist.")
            return None
        
        try:
            # Read the selected checkpoint
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # Save as current checkpoint
            main_checkpoint = os.path.join(self.checkpoint_dir, self.checkpoint_file)
            with open(main_checkpoint, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            print(f"Checkpoint loaded and set as current: {checkpoint_data.get('timestamp_formatted')}")
            return checkpoint_data
        
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return None
    
    def get_checkpoint_summary(self) -> Optional[Dict[str, Any]]:
        """
        Generates a summary of the current checkpoint.
        
        Returns:
            Optional[Dict[str, Any]]: Checkpoint summary or None if there's no checkpoint.
        """
        checkpoint_data = self._load_checkpoint()
        
        if not checkpoint_data or "processed_images" not in checkpoint_data:
            return None
        
        processed = checkpoint_data.get("processed_images", [])
        stats = checkpoint_data.get("stats", {})
        
        # Calculate progress percentage
        total = stats.get("total_images", 0)
        percentage = (len(processed) / total * 100) if total > 0 else 0
        
        # Calculate total processing time formatted
        processing_time = stats.get("processing_time", 0)
        hours, remainder = divmod(int(processing_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_formatted = f"{hours:02}:{minutes:02}:{seconds:02}"
        
        # Prepare summary
        summary = {
            "timestamp": checkpoint_data.get("timestamp_formatted", "Unknown"),
            "processed_count": len(processed),
            "total_images": total,
            "percentage": percentage,
            "processing_time": time_formatted,
            "is_complete": stats.get("complete", False),
            "was_interrupted": stats.get("interrupted", False),
            "successful": stats.get("successful", 0),
            "errors": stats.get("errors", 0),
            "already_processed": stats.get("already_processed", 0),
            "auto_detected": stats.get("auto_detected", 0)
        }
        
        return summary
    
    def _init_stats(self) -> Dict[str, Any]:
        """
        Initializes processing statistics dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary with initial statistics values.
        """
        return {
            "start_time": time.time(),
            "last_resumed": time.time(),
            "end_time": None,
            "processing_time": 0,
            "total_images": 0,
            "successful": 0,
            "errors": 0,
            "already_processed": 0,
            "auto_detected": 0,
            "last_batch": 0,
            "last_image_index": -1,
            "complete": False,
            "interrupted": False
        }
    
    def _load_failed_images(self) -> List[str]:
        """
        Loads the list of images that failed processing.
        
        Returns:
            List[str]: List of paths to failed images.
        """
        failed_images = []
        if os.path.exists("error_images.txt"):
            try:
                with open("error_images.txt", 'r', encoding='utf-8') as f:
                    for line in f:
                        img_path = line.strip().split(" | ")[0]
                        if img_path and os.path.exists(img_path):
                            failed_images.append(img_path)
            except Exception as e:
                print(f"Error loading failed images list: {str(e)}")
        
        return failed_images
    
    def _remove_from_failed_list(self, image_path: str) -> None:
        """
        Removes an image from the failed list after successfully processing it.
        
        Parameters:
            image_path (str): Path of the image to remove from the list.
        """
        if not os.path.exists("error_images.txt"):
            return
        
        try:
            with open("error_images.txt", 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            with open("error_images.txt", 'w', encoding='utf-8') as f:
                for line in lines:
                    if not line.strip().startswith(image_path):
                        f.write(line)
        except Exception as e:
            print(f"Error updating failed images list: {str(e)}")
    
    def show_failed_images(self) -> None:
        """Displays the paths of images that failed processing."""
        failed_images = self._load_failed_images()
        
        if not failed_images:
            print("No failed images recorded.")
            return
        
        print(f"Failed images ({len(failed_images)}):")
        for img_path in failed_images:
            print(f"  - {img_path}")