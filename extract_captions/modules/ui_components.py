import ipywidgets as widgets
from IPython.display import display, HTML
import os
import threading
import time
from typing import Dict, Any, List, Callable, Optional, Union, TypeVar, cast

# Define generic types for flexibility
ClientType = TypeVar('ClientType')
FileUtilType = TypeVar('FileUtilType')

class ProcessingControls:
    """Interface controls for processing images with Gemini Vision API.
    
    This class provides a complete UI for handling image processing tasks including:
    - Running batch processing of images
    - Pausing/resuming operations
    - Retrying failed images
    - Monitoring progress
    - Managing cooling periods between batches
    """
    
    def __init__(self, client_instance: ClientType, file_util_instance: FileUtilType) -> None:
        """
        Initializes the interface with all required components.
        
        Parameters:
            client_instance (ClientType): Instance of GeminiClient used for image processing.
            file_util_instance (FileUtilType): Instance of FileUtil for file management.
        """
        self.client = client_instance
        self.file_util = file_util_instance
        
        # Define interrupt flag file path in the notebook directory
        self.interrupt_flag_file = "interrupt.flag"  # Will be saved in the current working directory
        
        # Processing state
        self.is_processing = False
        self.is_cooling = False  # New state for cooling period
        
        # Processing statistics
        self.stats = {
            "total_images": 0,
            "processed_images": 0,
            "current_batch": 0,
            "total_batches": 0,
            "start_time": None
        }
        
        # Create widgets
        self._create_widgets()
        self._setup_callbacks()
        
        # Make sure there are no active interrupt flags at startup
        self._remove_interrupt_flag()
    
    def _create_widgets(self) -> None:
        """Creates and configures all UI widgets for the processing interface."""
        # Common styles
        button_layout = widgets.Layout(width='auto', height='40px')
        
        # Directory widget
        self.dir_input = widgets.Text(
            value=self.file_util.root_directory,
            placeholder='Enter directory path',
            description='Directory:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='90%', margin='5px')
        )
        
        # Sliders
        self.batch_slider = widgets.IntSlider(
            value=self.file_util.batch_size,
            min=1,
            max=20,
            step=1,
            description='Batch size:',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=widgets.Layout(width='45%', margin='5px')
        )
        
        self.cooling_slider = widgets.IntSlider(
            value=self.file_util.cooling_period,
            min=1,
            max=30,
            step=1,
            description='Cooling (sec):',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=widgets.Layout(width='45%', margin='5px')
        )
        
        # Buttons
        self.process_button = widgets.Button(
            description='▶️ Process Images',
            button_style='success',
            tooltip='Process images',
            layout=button_layout
        )
        
        self.interrupt_button = widgets.Button(
            description='⏹️ Stop',
            button_style='danger',
            tooltip='Stop processing',
            disabled=True,
            layout=button_layout
        )
        
        self.resume_button = widgets.Button(
            description='⏯️ Resume',
            button_style='info',
            tooltip='Resume from checkpoint',
            layout=button_layout
        )
        
        self.retry_button = widgets.Button(
            description='🔄 Retry Failed',
            button_style='warning',
            tooltip='Retry failed images',
            layout=button_layout
        )
        
        self.show_failed_button = widgets.Button(
            description='📋 View Failed',
            button_style='',
            tooltip='Show failed image paths',
            layout=button_layout
        )
        
        # Stats panel with improved style
        self.stats_panel = widgets.HTML(
            value=self._generate_stats_html(),
            layout=widgets.Layout(
                width='100%',
                margin='10px 0px',
                padding='10px',
                border='1px solid #ddd',
                border_radius='5px'
            )
        )
        
        # Status indicator
        self.status_indicator = widgets.HTML(
            value='<span style="color:gray; font-size: 16px;">Ready to process</span>',
            layout=widgets.Layout(margin='10px 0px')
        )
        
        # Output area with scroll for messages and logs
        self.output_area = widgets.Output(
            layout=widgets.Layout(
                height='300px',
                max_width='100%',
                overflow='auto',
                border='1px solid #ddd',
                border_radius='5px',
                padding='10px'
            )
        )
    
    def _generate_stats_html(self) -> str:
        """
        Generates HTML to display current statistics.
        
        Returns:
            str: Formatted HTML with progress bar and statistics.
        """
        # Calculate elapsed time
        elapsed_time = "00:00:00"
        if self.stats["start_time"]:
            seconds = int(time.time() - self.stats["start_time"])
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed_time = f"{hours:02}:{minutes:02}:{seconds:02}"
        
        # Apply style to time depending on cooling state
        time_style = "color:#2196F3; font-weight:bold;" if self.is_cooling else ""
        
        # Calculate completion percentage
        percent_complete = 0
        if self.stats["total_images"] > 0:
            percent_complete = (self.stats["processed_images"] / self.stats["total_images"]) * 100
        
        # Calculate remaining images
        remaining_images = max(0, self.stats["total_images"] - self.stats["processed_images"])
        
        # Calculate remaining batches
        remaining_batches = max(0, self.stats["total_batches"] - self.stats["current_batch"])
        
        # Generate HTML
        html = f"""
        <div style="font-family: Arial, sans-serif; padding: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <div style="width: 48%;">
                    <h4 style="margin: 0 0 5px 0;">Progress</h4>
                    <div style="background-color: #f0f0f0; border-radius: 10px; height: 20px; width: 100%; overflow: hidden;">
                        <div style="background-color: #4CAF50; height: 100%; width: {min(percent_complete, 100)}%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 3px;">
                        <span>0%</span>
                        <span>{percent_complete:.1f}%</span>
                        <span>100%</span>
                    </div>
                </div>
                <div style="width: 48%;">
                    <h4 style="margin: 0 0 5px 0;">Information</h4>
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        <tr>
                            <td style="padding: 2px;">⏱️ Time:</td>
                            <td style="padding: 2px; text-align: right; {time_style}">{elapsed_time}{' ⏸️' if self.is_cooling else ''}</td>
                        </tr>
                        <tr>
                            <td style="padding: 2px;">📊 Images:</td>
                            <td style="padding: 2px; text-align: right;">{self.stats["processed_images"]}/{self.stats["total_images"]} (Remaining: {remaining_images})</td>
                        </tr>
                        <tr>
                            <td style="padding: 2px;">📦 Batches:</td>
                            <td style="padding: 2px; text-align: right;">{self.stats["current_batch"]}/{self.stats["total_batches"]} (Remaining: {remaining_batches})</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        """
        return html
    
    def _update_stats(self, update_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Updates statistics and refreshes the information panel.
        
        Parameters:
            update_dict (Optional[Dict[str, Any]]): Dictionary with values to update.
        """
        if update_dict:
            for key, value in update_dict.items():
                if key in self.stats:
                    self.stats[key] = value
        
        # Update statistics panel
        self.stats_panel.value = self._generate_stats_html()
    
    def _setup_callbacks(self) -> None:
        """Sets up callback functions for all interactive widgets."""
        self.process_button.on_click(self._on_process_clicked)
        self.interrupt_button.on_click(self._on_interrupt_clicked)
        self.resume_button.on_click(self._on_resume_clicked)
        self.retry_button.on_click(self._on_retry_clicked)
        self.show_failed_button.on_click(self._on_show_failed_clicked)
    
    def _update_button_states(self, processing: bool = False) -> None:
        """
        Updates button states based on processing state.
        
        Parameters:
            processing (bool): True if processing is active, False otherwise.
        """
        self.is_processing = processing
        
        # Enable/disable buttons according to state
        self.process_button.disabled = processing
        self.resume_button.disabled = processing
        self.retry_button.disabled = processing
        self.show_failed_button.disabled = processing
        self.interrupt_button.disabled = not processing
        
        # Update status indicator
        if processing:
            self.status_indicator.value = '<span style="color:green; font-size: 16px; font-weight:bold;">⚙️ Processing in progress...</span>'
            # Start timer
            if not self.stats["start_time"]:
                self.stats["start_time"] = time.time()
        else:
            if os.path.exists(self.interrupt_flag_file):
                self.status_indicator.value = '<span style="color:red; font-size: 16px; font-weight:bold;">⚠️ Processing interrupted</span>'
            else:
                self.status_indicator.value = '<span style="color:gray; font-size: 16px;">Ready to process</span>'
            # Stop timer
            self.stats["start_time"] = None
        
        # Update statistics panel
        self._update_stats()
    
    def _create_interrupt_flag(self) -> bool:
        """
        Creates flag file to request interruption.
        
        Returns:
            bool: True if flag was created successfully, False otherwise.
        """
        try:
            with open(self.interrupt_flag_file, 'w') as f:
                f.write('interrupt')
            self.status_indicator.value = '<span style="color:orange;font-weight:bold;">Stopping processing...</span>'
            return True
        except Exception as e:
            with self.output_area:
                print(f"Error creating interrupt file: {str(e)}")
            return False
    
    def _remove_interrupt_flag(self) -> bool:
        """
        Removes the interrupt flag file if it exists.
        
        Returns:
            bool: True if flag was removed or didn't exist, False if removal failed.
        """
        if os.path.exists(self.interrupt_flag_file):
            try:
                os.remove(self.interrupt_flag_file)
                return True
            except Exception as e:
                with self.output_area:
                    print(f"Error removing interrupt file: {str(e)}")
                return False
        return True
    
    def _process_thread_function(self, force_overwrite: bool = False) -> None:
        """
        Function to run image processing in a separate thread.
        
        Parameters:
            force_overwrite (bool): If True, overwrite existing caption files.
        """
        try:
            # Get total images information
            checkpoint_data = self.file_util._load_checkpoint()
            processed_images = checkpoint_data.get("processed_images", [])
            stats = checkpoint_data.get("stats", {})
            image_paths = self.file_util._get_all_images()
            total_images = len(image_paths)
            total_batches = (total_images + self.file_util.batch_size - 1) // self.file_util.batch_size
            
            # Update initial statistics
            self._update_stats({
                "total_images": total_images,
                "processed_images": len(processed_images),
                "current_batch": 0,
                "total_batches": total_batches
            })
            
            # Create a FileUtil subclass that reports progress
            file_util = self.file_util
            original_process_images = file_util.process_images
            
            def process_images_with_progress(force_overwrite: bool = False, interrupt_flag_file: Optional[str] = None) -> None:
                # Initial call to configure
                image_paths = file_util._get_all_images()
                checkpoint_data = file_util._load_checkpoint()
                processed_images = checkpoint_data.get("processed_images", [])
                last_index = checkpoint_data.get("last_index", -1)
                stats = checkpoint_data.get("stats", {})
                
                with self.output_area:
                    print(f"Total images found: {len(image_paths)}")
                    print(f"Images already processed: {len(processed_images)}")
                
                if not image_paths:
                    with self.output_area:
                        print("No images found to process.")
                    return
                
                # Continue from where it left off
                start_index = last_index + 1 if last_index >= 0 else 0
                
                if start_index >= len(image_paths):
                    with self.output_area:
                        print("All images have already been processed.")
                    return
                
                # Update statistics
                self._update_stats({
                    "processed_images": len(processed_images),
                })
                
                # Process images in batches
                for batch_start in range(start_index, len(image_paths), file_util.batch_size):
                    batch_end = min(batch_start + file_util.batch_size, len(image_paths))
                    batch = image_paths[batch_start:batch_end]
                    current_batch = batch_start//file_util.batch_size + 1
                    
                    # Update batch statistics
                    self._update_stats({
                        "current_batch": current_batch,
                    })
                    
                    with self.output_area:
                        print(f"\nProcessing batch {current_batch} ({batch_end}/{len(image_paths)} images)")
                    
                    # Check for interruption before starting the batch
                    if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                        with self.output_area:
                            print("\n⚠️ Processing interrupted before starting the batch.")
                            print(f"Progress saved: {batch_start}/{len(image_paths)} images")
                        return
                    
                    # Process each image in the batch
                    for i, img_path in enumerate(batch):
                        # Check for interruption before processing each image
                        if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                            with self.output_area:
                                print("\n⚠️ Processing interrupted by user request.")
                                print(f"Progress saved: {batch_start + i}/{len(image_paths)} images")
                            # Save checkpoint
                            file_util._save_checkpoint(processed_images, batch_start + i - 1, stats)
                            return
                        
                        if img_path in processed_images and not force_overwrite:
                            with self.output_area:
                                print(f"Skipping (already processed): {img_path}")
                            continue
                        
                        with self.output_area:
                            print(f"Processing: {img_path}")
                        
                        result = file_util.client.process_imagen(img_path, force_overwrite)
                        
                        if result.get("status") == "processed":
                            processed_images.append(img_path)
                            self._update_stats({"processed_images": len(processed_images)})
                            with self.output_area:
                                print(f"✓ Processed in {result.get('process_time', 0):.2f}s")
                        elif result.get("status") == "already_processed":
                            processed_images.append(img_path)
                            self._update_stats({"processed_images": len(processed_images)})
                            with self.output_area:
                                print("✓ Already processed previously")
                        else:
                            with self.output_area:
                                print(f"✗ Error: {result.get('error', 'Unknown error')}")
                    
                    # Save checkpoint after each batch
                    file_util._save_checkpoint(processed_images, batch_end - 1, stats)
                    
                    # Check for interruption after the batch
                    if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                        with self.output_area:
                            print("\n⚠️ Processing paused after the batch.")
                            print(f"Progress saved: {batch_end}/{len(image_paths)} images")
                        return
                    
                    # Cooling period between batches
                    if batch_end < len(image_paths):
                        with self.output_area:
                            print(f"\nCooling down for {file_util.cooling_period} seconds...")
                        
                        # Activate cooling state
                        self.is_cooling = True
                        self._update_stats()  # Update UI with new state
                        
                        # Check for interruption during cooling every second
                        # and update interface every second
                        cooling_start = time.time()
                        remaining_seconds = file_util.cooling_period
                        
                        while remaining_seconds > 0:
                            if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                                with self.output_area:
                                    print("\n⚠️ Processing interrupted during cooling.")
                                self.is_cooling = False
                                self._update_stats()
                                return
                            
                            # Update the interface
                            self._update_stats()
                            
                            # Wait one second
                            time.sleep(1)
                            
                            # Update remaining time
                            elapsed = int(time.time() - cooling_start)
                            remaining_seconds = file_util.cooling_period - elapsed
                            
                            # If less than 0 seconds remain, finish
                            if remaining_seconds <= 0:
                                break
                        
                        # Deactivate cooling state
                        self.is_cooling = False
                        self._update_stats()
                
                with self.output_area:
                    print("\n✅ Processing completed")
            
            # Temporarily replace the method to provide progress
            file_util.process_images = process_images_with_progress
            
            # Execute processing
            file_util.process_images(force_overwrite=force_overwrite, interrupt_flag_file=self.interrupt_flag_file)
            
            # Restore original method
            file_util.process_images = original_process_images
            
        finally:
            # Make sure buttons return to normal state
            self._update_button_states(processing=False)
            # If there was an interruption, update the indicator
            if os.path.exists(self.interrupt_flag_file):
                self.status_indicator.value = '<span style="color:red; font-size: 16px; font-weight:bold;">⚠️ Processing interrupted</span>'
                self._remove_interrupt_flag()
            else:
                self.status_indicator.value = '<span style="color:blue; font-size: 16px; font-weight:bold;">✅ Processing completed</span>'
    
    def _retry_thread_function(self) -> None:
        """
        Function to retry processing failed images in a separate thread.
        """
        try:
            # Get information about failed images
            failed_images = self.file_util._load_failed_images()
            total_images = len(failed_images)
            total_batches = (total_images + self.file_util.batch_size - 1) // self.file_util.batch_size if total_images > 0 else 0
            
            # Update initial statistics
            self._update_stats({
                "total_images": total_images,
                "processed_images": 0,
                "current_batch": 0,
                "total_batches": total_batches
            })
            
            # Create a FileUtil subclass that reports progress
            file_util = self.file_util
            original_retry_failed = file_util.retry_failed_images
            
            def retry_failed_with_progress(interrupt_flag_file: Optional[str] = None) -> None:
                failed_images = file_util._load_failed_images()
                
                if not failed_images:
                    with self.output_area:
                        print("No failed images to retry.")
                    return
                
                with self.output_area:
                    print(f"Retrying {len(failed_images)} failed images...")
                
                successfully_processed = 0
                
                # Process failed images in batches
                for batch_start in range(0, len(failed_images), file_util.batch_size):
                    batch_end = min(batch_start + file_util.batch_size, len(failed_images))
                    batch = failed_images[batch_start:batch_end]
                    current_batch = batch_start//file_util.batch_size + 1
                    
                    # Update batch statistics
                    self._update_stats({
                        "current_batch": current_batch,
                    })
                    
                    with self.output_area:
                        print(f"\nProcessing batch {current_batch} ({batch_end}/{len(failed_images)} images)")
                    
                    # Check for interruption before starting the batch
                    if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                        with self.output_area:
                            print("\n⚠️ Retry interrupted before starting the batch.")
                        return
                    
                    for i, img_path in enumerate(batch):
                        # Check for interruption before processing each image
                        if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                            with self.output_area:
                                print("\n⚠️ Retry interrupted by user request.")
                            return
                        
                        with self.output_area:
                            print(f"Retrying: {img_path}")
                        
                        result = file_util.client.process_imagen(img_path, force_overwrite=True)
                        
                        if result.get("status") == "processed":
                            file_util._remove_from_failed_list(img_path)
                            successfully_processed += 1
                            self._update_stats({"processed_images": successfully_processed})
                            with self.output_area:
                                print("✓ Successfully processed on second attempt")
                        else:
                            with self.output_area:
                                print(f"✗ Still failing: {result.get('error', 'Unknown error')}")
                    
                    # Check for interruption after the batch
                    if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                        with self.output_area:
                            print("\n⚠️ Retry paused after the batch.")
                        return
                    
                    # Cooling period between batches
                    if batch_end < len(failed_images):
                        with self.output_area:
                            print(f"\nCooling down for {file_util.cooling_period} seconds...")
                        
                        # Activate cooling state
                        self.is_cooling = True
                        self._update_stats()  # Update UI with new state
                        
                        # Check for interruption during cooling every second
                        # and update interface every second
                        cooling_start = time.time()
                        remaining_seconds = file_util.cooling_period
                        
                        while remaining_seconds > 0:
                            if interrupt_flag_file and os.path.exists(interrupt_flag_file):
                                with self.output_area:
                                    print("\n⚠️ Retry interrupted during cooling.")
                                self.is_cooling = False
                                self._update_stats()
                                return
                            
                            # Update the interface
                            self._update_stats()
                            
                            # Wait one second
                            time.sleep(1)
                            
                            # Update remaining time
                            elapsed = int(time.time() - cooling_start)
                            remaining_seconds = file_util.cooling_period - elapsed
                            
                            # If less than 0 seconds remain, finish
                            if remaining_seconds <= 0:
                                break
                        
                        # Deactivate cooling state
                        self.is_cooling = False
                        self._update_stats()
                
                with self.output_area:
                    if successfully_processed > 0:
                        print(f"\n✅ Successfully processed {successfully_processed} of {len(failed_images)} failed images.")
                    else:
                        print("\n⚠️ Could not process any failed images.")
            
            # Temporarily replace the method to provide progress
            file_util.retry_failed_images = retry_failed_with_progress
            
            # Execute retry
            file_util.retry_failed_images(interrupt_flag_file=self.interrupt_flag_file)
            
            # Restore original method
            file_util.retry_failed_images = original_retry_failed
            
        finally:
            # Make sure buttons return to normal state
            self._update_button_states(processing=False)
            # If there was an interruption, update the indicator
            if os.path.exists(self.interrupt_flag_file):
                self.status_indicator.value = '<span style="color:red; font-size: 16px; font-weight:bold;">⚠️ Retry interrupted</span>'
                self._remove_interrupt_flag()
            else:
                self.status_indicator.value = '<span style="color:blue; font-size: 16px; font-weight:bold;">✅ Retry completed</span>'
    
    def _on_process_clicked(self, b: widgets.Button) -> None:
        """
        Callback to start image processing.
        
        Parameters:
            b (widgets.Button): Button widget that triggered the callback.
        """
        if self.is_processing:
            return
        
        self.file_util.root_directory = self.dir_input.value.strip()
        self.file_util.batch_size = self.batch_slider.value
        self.file_util.cooling_period = self.cooling_slider.value
        
        # Clear output area
        self.output_area.clear_output()
        
        # Make sure there are no active interrupt flags
        self._remove_interrupt_flag()
        
        # Update button states
        self._update_button_states(processing=True)
        
        # Start processing in a separate thread
        with self.output_area:
            print("Starting processing...")
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self._process_thread_function, args=(False,))
        thread.daemon = True
        thread.start()
    
    def _on_interrupt_clicked(self, b: widgets.Button) -> None:
        """
        Callback to request stopping of processing.
        
        Parameters:
            b (widgets.Button): Button widget that triggered the callback.
        """
        if not self.is_processing:
            return
            
        # Don't clear output to keep message visible
        if self._create_interrupt_flag():
            with self.output_area:
                print("\n⚠️ Requesting to stop processing. The process will stop after finishing the current image.")
        else:
            with self.output_area:
                print("Error requesting interruption.")
    
    def _on_resume_clicked(self, b: widgets.Button) -> None:
        """
        Callback to resume processing from checkpoint.
        
        Parameters:
            b (widgets.Button): Button widget that triggered the callback.
        """
        if self.is_processing:
            return
            
        # Clear output area
        self.output_area.clear_output()
        
        # Make sure there are no active interrupt flags
        self._remove_interrupt_flag()
        
        # Update button states
        self._update_button_states(processing=True)
        
        with self.output_area:
            print("Resuming processing from checkpoint (if it exists)...")
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self._process_thread_function, args=(False,))
        thread.daemon = True
        thread.start()
    
    def _on_retry_clicked(self, b: widgets.Button) -> None:
        """
        Callback to retry processing previously failed images.
        
        Parameters:
            b (widgets.Button): Button widget that triggered the callback.
        """
        if self.is_processing:
            return
            
        # Clear output area
        self.output_area.clear_output()
        
        # Make sure there are no active interrupt flags
        self._remove_interrupt_flag()
        
        # Update button states
        self._update_button_states(processing=True)
        
        with self.output_area:
            print("Retrying failed images...")
        
        # Start retry in a separate thread
        thread = threading.Thread(target=self._retry_thread_function)
        thread.daemon = True
        thread.start()
    
    def _on_show_failed_clicked(self, b: widgets.Button) -> None:
        """
        Callback to display a list of failed image paths.
        
        Parameters:
            b (widgets.Button): Button widget that triggered the callback.
        """
        # Clear output area to show only failed images
        self.output_area.clear_output()
        
        with self.output_area:
            self.file_util.show_failed_images()
    
    def display(self) -> None:
        """Displays the complete user interface with all panels and widgets."""
        # Main title
        display(HTML("<h2 style='color:#3498db; font-family:Arial,sans-serif;'>Image Processing with Gemini</h2>"))
        
        # Configuration panel
        config_panel = widgets.VBox([
            widgets.HTML("<b style='font-size:16px;'>⚙️ Configuration</b>"),
            self.dir_input,
            widgets.HBox([self.batch_slider, self.cooling_slider], layout=widgets.Layout(justify_content='space-between'))
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0px',
            border_radius='5px'
        ))
        display(config_panel)
        
        # Control panel
        control_panel = widgets.VBox([
            widgets.HTML("<b style='font-size:16px;'>🎮 Controls</b>"),
            widgets.HBox([
                self.process_button, self.interrupt_button, self.resume_button, 
                self.retry_button, self.show_failed_button
            ], layout=widgets.Layout(justify_content='space-between'))
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0px',
            border_radius='5px'
        ))
        display(control_panel)
        
        # Status and statistics panel
        status_panel = widgets.VBox([
            widgets.HTML("<b style='font-size:16px;'>📊 Status and Progress</b>"),
            self.status_indicator,
            self.stats_panel,
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0px',
            border_radius='5px'
        ))
        display(status_panel)
        
        # Output panel
        output_panel = widgets.VBox([
            widgets.HTML("<b style='font-size:16px;'>📋 Processing Log</b>"),
            self.output_area
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0px',
            border_radius='5px'
        ))
        display(output_panel)

def create_processing_interface(client_instance: ClientType, file_util_instance: FileUtilType) -> ProcessingControls:
    """
    Creates and displays the processing interface.
    
    Parameters:
        client_instance (ClientType): Instance of Gemini client.
        file_util_instance (FileUtilType): Instance of FileUtil.
    
    Returns:
        ProcessingControls: Instance of the UI controls.
    """
    controls = ProcessingControls(client_instance, file_util_instance)
    controls.display()
    return controls
