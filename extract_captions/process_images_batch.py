# /// script
# dependencies = [
#   "python-dotenv",
#   "google>=0.3.0",
#   "google-genai>=1.3.0",
#   "pillow"
# ]
# ///

"""
Script to process images in batch using Gemini API.
Non-interactive version based on process_images.ipynb.

This script allows for batch processing of images found in a specified directory,
generating text descriptions for each image using Google's Gemini API.
It supports processing in batches with cooling periods to avoid API rate limits.

Usage:
    python process_images_batch.py [options]

Options:
    --root-dir       Directory containing images to process
    --batch-size     Number of images to process in each batch
    --cooldown       Seconds to wait between batches
    --max-tokens     Maximum tokens for Gemini API response
    --api-key        Gemini API key (falls back to environment variable)
    --output-dir     Directory to save results (defaults to images location)
    --recursive      Search for images in subdirectories
"""

import os
import time
import argparse
import shutil
from typing import Dict, Any

from dotenv import load_dotenv
from modules.file_utils import FileUtil
from clientes.gemini_client import GeminiClient

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the image processing script.
    
    Returns:
        argparse.Namespace: Object containing all parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Process images in batch with Gemini API')
    
    # Standard options
    parser.add_argument('--root-dir', type=str, default="dof_markdown",
                        help='Root directory to search for images (default: dof_markdown)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for processing images (default: 10)')
    parser.add_argument('--cooldown', type=int, default=20,
                        help='Cooling time between batches in seconds (default: 20)')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Maximum number of tokens for response (default: 512)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (defaults to the same as images)')
    parser.add_argument('--recursive', action='store_true',
                        help='Search for images recursively in subdirectories')
    
    return parser.parse_args()

def main() -> None:
    """
    Main function that orchestrates the batch processing of images.
    
    This function:
    1. Loads environment variables
    2. Parses command-line arguments
    3. Configures and initializes the Gemini client
    4. Processes images in batches
    5. Handles output file management
    6. Provides processing statistics
    
    Raises:
        ValueError: If no API key is provided
        KeyboardInterrupt: When user interrupts the process (handled gracefully)
    """
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_arguments()
    
    # Configure the prompt for Gemini (prompt is kept in Spanish as per original requirements)
    question = """Resume brevemente la imagen en español (máximo 3-4 oraciones por categoría):  
                - **Texto:** Menciona solo el título y 2-3 puntos clave si hay texto.
                - **Mapas:** Identifica la región principal y máximo 2-3 ubicaciones relevantes.
                - **Diagramas:** Resume el concepto central en 1-2 oraciones.
                - **Logos:** Identifica la entidad y sus características distintivas.
                - **Datos visuales:** Menciona solo los 2-3 valores o tendencias más importantes.
                Prioriza la información esencial sobre los detalles, manteniendo la descripción breve y directa."""
    
    # Configure Gemini client
    gemini_client = GeminiClient(max_tokens=args.max_tokens)
    
    # Set API key (first from argument, then from environment)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found. Please configure it.")
    
    gemini_client.set_api_key(api_key)
    gemini_client.set_question(question)
    
    # Configure FileUtil with appropriate parameters
    file_util = FileUtil(
        root_directory=args.root_dir, 
        client=gemini_client, 
        batch_size=args.batch_size, 
        cooling_period=args.cooldown
    )
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    
    # Process images using FileUtil functionality
    print(f"Starting image processing in: {args.root_dir}")
    print(f"Batch size: {args.batch_size}, Cooling time: {args.cooldown}s")
    
    # Create a temporary interruption file to be able to stop the process
    interrupt_flag_file = "stop_processing.flag"
    if os.path.exists(interrupt_flag_file):
        os.remove(interrupt_flag_file)
    
    # Process images and get statistics
    stats: Dict[str, Any] = {}
    try:
        stats = file_util.process_images(force_overwrite=False, interrupt_flag_file=interrupt_flag_file)
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user (Ctrl+C)")
        print("Saving checkpoint with current progress...")
        # Load the last checkpoint to get current information
        checkpoint_data = file_util._load_checkpoint()
        print(f"Progress saved in checkpoint: {checkpoint_data.get('last_index', 0) + 1}/{checkpoint_data.get('stats', {}).get('total_images', 0)} images")
        stats = checkpoint_data.get('stats', {})
        stats['interrupted'] = True
        
        # Update processing time in case of interruption
        if 'start_time' in stats and not 'processing_time' in stats:
            stats['processing_time'] = time.time() - stats['start_time']
        elif 'last_resumed' in stats and 'processing_time' in stats:
            stats['processing_time'] += time.time() - stats['last_resumed']
            
        print("Checkpoint saved successfully.")
    
    # If a different output directory was specified, move the generated files
    if args.output_dir:
        print("\nMoving result files to output directory...")
        image_files = file_util._get_all_images()
        for img_path in image_files:
            # Check if there is an associated text file
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                # Get the base name of the file
                base_name = os.path.basename(txt_path)
                # Create the destination path
                dest_path = os.path.join(args.output_dir, base_name)
                # Copy the file if it doesn't exist in the destination
                if not os.path.exists(dest_path):
                    try:
                        shutil.copy2(txt_path, dest_path)
                        print(f"  Copied: {base_name}")
                    except Exception as e:
                        print(f"  Error copying {base_name}: {str(e)}")
    
    # Show final summary
    print("\n=== Processing Summary ===")
    print(f"Total images found: {stats.get('total_images', 0)}")
    print(f"Images processed successfully: {stats.get('successful', 0)}")
    print(f"Images already processed previously: {stats.get('already_processed', 0)}")
    print(f"Errors: {stats.get('errors', 0)}")
    
    # Show total processing time (not including cooling)
    processing_time = stats.get('processing_time', 0)
    if processing_time > 0:
        hours, remainder = divmod(processing_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            print(f"Total processing time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        elif minutes > 0:
            print(f"Total processing time: {int(minutes)}m {seconds:.1f}s")
        else:
            print(f"Total processing time: {seconds:.1f}s")
    
    # Remove the interruption file if it exists
    if os.path.exists(interrupt_flag_file):
        os.remove(interrupt_flag_file)
    
    print("\nProcessing completed.")
    if stats.get('interrupted', False):
        print("⚠️ Processing was interrupted before completion.")

if __name__ == "__main__":
    main()