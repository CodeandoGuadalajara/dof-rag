# /// script
# dependencies = [
#   "moondream>=0.0.6",
#   "tokenizers>=0.20.1,<0.21.0",
#   "sentencepiece>=0.2.0",
#   "sacremoses>=0.1.1",
#   "transformers>=4.46.3",
#   "python-dotenv"
# ]
# ///

import moondream as md
from PIL import Image
import time
import datetime
import os
import glob
import gc
import threading
import concurrent.futures
import argparse
import logging
import colorama
from tqdm import tqdm
from colorama import Fore

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Logging configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)



# Import libraries for translation with MarianMT
from transformers import MarianMTModel, MarianTokenizer

# Load translation model (from English to Spanish) globally for better efficiency
translation_model_name = "Helsinki-NLP/opus-mt-en-es"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

# Define default question for images
DEFAULT_QUESTION = "¿Qué se muestra en esta imagen? Describe todos los elementos visibles con detalle."

# Define available models
AVAILABLE_MODELS = {
    "small": "model/moondream-0_5b-int8.mf",  # Small model (0.5b)
    "big": "model/moondream-2b-int8.mf"       # Big model (2b)
}

def translate_text(text):
    """Translates text from English to Spanish using MarianMT."""
    if not text:
        return ""
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = translation_model.generate(**inputs)
    translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

class MoondreamImageProcessor:
    """Class for processing images with Moondream."""
    
    def __init__(self, model_path="model/moondream-0_5b-int8.mf", output_file=None, mode="single"):
        """Initialize the Moondream image processor.
        
        Args:
            model_path: Path to the Moondream model to use
            output_file: File to save results to
            mode: Processing mode ("batch" or "single")
        """
        self.model_path = model_path
        self.mode = mode
        
        # Extract model name from path to use in filename
        model_name = os.path.basename(model_path).split('.')[0]
        
        # Generate output filename if not specified
        if output_file is None:
            mode_suffix = "single" if mode == "single" else "batch"
            self.output_file = f"tests/results/{model_name}_{mode_suffix}.txt"
        else:
            self.output_file = output_file
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Lock to synchronize model access
        self.model_lock = threading.Lock()
        self.model = None
    
    def initialize_model(self):
        """Initialize the Moondream model."""
        try:
            init_start = time.time()
            self.model = md.vl(model=self.model_path)
            init_time = time.time() - init_start
            logger.info(f"{Fore.GREEN}✓ Model {os.path.basename(self.model_path)} initialized in {init_time:.2f}s")
            return init_time
        except Exception as e:
            logger.error(f"{Fore.RED}✗ Error initializing model: {str(e)}")
            raise
    
    def close_model(self):
        """Close the model and free resources."""
        try:
            if self.model and hasattr(self.model, "close"):
                self.model.close()
            del self.model
            self.model = None
            gc.collect()
            logger.info(f"{Fore.CYAN}ℹ Model released")
        except Exception as e:
            logger.error(f"{Fore.RED}✗ Error closing model: {str(e)}")
    
    def process_image(self, image_path, question):
        """Process a single image using the shared model."""
        result = {"image_path": image_path}
        start = time.time()
        
        try:
            image = Image.open(image_path)
            result["image_size"] = f"{image.width}x{image.height}"
            result["image_format"] = image.format
        except Exception as e:
            result["error"] = f"Error opening image: {str(e)}"
            return result
        
        try:
            with self.model_lock:
                # Encode the image
                encode_start = time.time()
                encoded_image = self.model.encode_image(image)
                encode_time = time.time() - encode_start

                # Generate caption
                caption_start = time.time()
                caption_result = self.model.caption(encoded_image)
                caption_time = time.time() - caption_start
                caption = caption_result.get("caption", "")

                # Execute query
                query_start = time.time()
                answer_result = self.model.query(encoded_image, question)
                query_time = time.time() - query_start
                answer = answer_result.get("answer", "")
            
            total_time = time.time() - start
            result["encode_time"] = encode_time
            result["caption_time"] = caption_time
            result["query_time"] = query_time
            result["total_time"] = total_time
            result["caption"] = caption
            result["answer"] = answer
            result["question"] = question

            # Translate responses (caption and answer) from English to Spanish
            result["caption_trad"] = translate_text(caption)
            result["answer_trad"] = translate_text(answer)
        except Exception as e:
            result["error"] = f"Error processing image: {str(e)}"
        
        return result
    
    def save_image_result(self, img_res):
        """Save the result of an image to the results file."""
        separator = "\n" + "-" * 60 + "\n"
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(separator)
            f.write("IMAGE RESULTS\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Model: {os.path.basename(self.model_path)}\n")
            f.write(f"Image: {os.path.basename(img_res.get('image_path', 'N/A'))}\n")
            
            if "error" in img_res:
                f.write("Error: " + img_res["error"] + "\n")
            else:
                f.write("\nMETRICS:\n")
                f.write(f"- Encode: {img_res.get('encode_time', 0):.4f}s | ")
                f.write(f"Caption: {img_res.get('caption_time', 0):.4f}s | ")
                f.write(f"Query: {img_res.get('query_time', 0):.4f}s | ")
                f.write(f"Total: {img_res.get('total_time', 0):.4f}s\n")
                
                f.write("\nRESULTS:\n")
                f.write(f"Question: {img_res.get('question', '')}\n\n")
                f.write(f"Description: {img_res.get('caption_trad', '')}\n\n")
                f.write(f"Answer: {img_res.get('answer_trad', '')}\n")
            f.write(separator)

    def process_interactive(self, image_path, question):
        """Process an image in interactive mode."""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}Processing: {Fore.YELLOW}{os.path.basename(image_path)}")
        print(f"{Fore.CYAN}Model: {Fore.YELLOW}{os.path.basename(self.model_path)}")
        
        try:
            # Initialize model if not initialized
            if self.model is None:
                print(f"{Fore.CYAN}Initializing model...")
                self.initialize_model()
            
            # Process image
            print(f"{Fore.CYAN}Processing image...")
            result = self.process_image(image_path, question)
            
            if "error" in result:
                print(f"{Fore.RED}Error: {result['error']}")
            else:
                # Show results in compact form
                print(f"\n{Fore.GREEN}Times: ", end="")
                print(f"Encode={result['encode_time']:.2f}s | ", end="")
                print(f"Caption={result['caption_time']:.2f}s | ", end="")
                print(f"Query={result['query_time']:.2f}s | ", end="")
                print(f"Total={result['total_time']:.2f}s")
                
                print(f"\n{Fore.GREEN}Description: {Fore.WHITE}{result['caption_trad']}")
                print(f"\n{Fore.GREEN}Question: {Fore.WHITE}{question}")
                print(f"{Fore.GREEN}Answer: {Fore.WHITE}{result['answer_trad']}")
            
            # Save result
            self.save_image_result(result)
            print(f"\n{Fore.CYAN}Results saved: {os.path.basename(self.output_file)}")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}")

    def process_batch(self, image_files, question, max_workers=None):
        """Process multiple images in parallel."""
        results = {
            "model_path": self.model_path,
            "images": [],
            "start_time": datetime.datetime.now().isoformat()
        }
        
        start_total = time.time()
        total_images = len(image_files)
        
        # Calculate number of workers
        if max_workers is None:
            num_cpu = os.cpu_count() or 1
            max_workers = max(1, int(num_cpu * 0.7))
        
        # Initialize results file
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("TEST MOONDREAM - BATCH\n")
            f.write(f"Model: {os.path.basename(self.model_path)}\n")
            f.write(f"Start: {results['start_time']}\n")
            f.write(f"Images: {total_images}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Workers: {max_workers}\n")
        
        try:
            # Initialize model
            init_time = self.initialize_model()
            results["init_time"] = init_time
            
            print(f"{Fore.CYAN}Processing {total_images} images with {max_workers} workers...")
            
            # Use tqdm to show a progress bar
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_image, img_path, question): img_path 
                           for img_path in image_files}
                
                # Progress bar with tqdm
                with tqdm(total=total_images, desc="Processing") as progress_bar:
                    for future in concurrent.futures.as_completed(futures):
                        image_result = future.result()
                        results["images"].append(image_result)
                        
                        # Save result
                        self.save_image_result(image_result)
                        
                        # Update progress bar
                        progress_bar.update(1)
                        
                        # Show minimal info
                        img_name = os.path.basename(image_result['image_path'])
                        if "error" in image_result:
                            tqdm.write(f"{Fore.RED}Error: {img_name}")
                        else:
                            tqdm.write(f"{Fore.GREEN}OK: {img_name} ({image_result.get('total_time', 0):.2f}s)")
            
            # Add final statistics
            total_time = time.time() - start_total
            results["total_time"] = total_time
            results["end_time"] = datetime.datetime.now().isoformat()
            
            # Save final summary
            self._save_summary(results)
            
            # Show compact summary
            successful = sum(1 for img in results.get('images', []) if 'error' not in img)
            failed = len(results.get('images', [])) - successful
            print(f"\n{Fore.GREEN}Completed: {successful} OK, {failed} errors")
            print(f"{Fore.GREEN}Total time: {results.get('total_time', 0):.2f} seconds")
            print(f"{Fore.GREEN}Results: {os.path.basename(self.output_file)}")
            
        except Exception as e:
            results["error"] = f"Error in processing: {str(e)}"
            logger.error(f"{Fore.RED}{results['error']}")
        finally:
            # Close model
            self.close_model()
        
    def _save_summary(self, results):
        """Save a summary of the results."""
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write("\n" + "-" * 60 + "\n")
            f.write("FINAL PROCESSING SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Model: {os.path.basename(self.model_path)}\n")
            f.write(f"Completion timestamp: {datetime.datetime.now().isoformat()}\n")
            
            # Image statistics
            total_images = len(results.get('images', []))
            successful_images = sum(1 for img in results.get('images', []) if 'error' not in img)
            failed_images = total_images - successful_images
            
            f.write(f"Total images processed: {total_images}\n")
            f.write(f"Successful: {successful_images}, Errors: {failed_images}\n")
            
            # Highlighted total time
            f.write(f"Total processing time: {results.get('total_time', 0):.2f} seconds\n")
            f.write(f"Initialization time: {results.get('init_time', 0)::.2f} seconds\n")
            
            if successful_images > 0:
                # Calculate average times in compact form
                avg_encode = sum(img.get('encode_time', 0) for img in results.get('images', []) 
                               if 'error' not in img) / successful_images
                avg_caption = sum(img.get('caption_time', 0) for img in results.get('images', []) 
                                if 'error' not in img) / successful_images
                avg_query = sum(img.get('query_time', 0) for img in results.get('images', []) 
                              if 'error' not in img) / successful_images
                avg_total = sum(img.get('total_time', 0) for img in results.get('images', []) 
                              if 'error' not in img) / successful_images
                
                f.write("\nAverage time per image:\n")
                f.write(f"- Encode: {avg_encode:.4f}s | ")
                f.write(f"Caption: {avg_caption:.4f}s | ")
                f.write(f"Query: {avg_query:.4f}s | ")
                f.write(f"Total: {avg_total:.4f}s\n")
            
            f.write("-" * 60 + "\n")


def main():
    """Main function that handles command line parameters."""
    # Simple banner
    print(f"\n{Fore.CYAN}TEST MOONDREAM")
    print(f"{Fore.CYAN}{'-' * 30}")
    
    parser = argparse.ArgumentParser(
        description="Image processor with Moondream"
    )
    
    # Processing mode
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                      help="Mode: 'single' (one image) or 'batch' (all)")
    
    # Model size
    parser.add_argument("--model", choices=["small", "big"], default="small",
                      help="Model: 'small' (0.5b) or 'big' (2b)")
    
    # Other parameters
    parser.add_argument("-q", "--question", type=str, default=DEFAULT_QUESTION,
                      help="Question for the images")
    parser.add_argument("-d", "--dir", type=str, default="imagenes_prueba",
                      help="Image directory")
    parser.add_argument("-o", "--output", type=str, default=None,
                      help="Output file")
    parser.add_argument("-w", "--workers", type=int, default=None,
                      help="Number of workers (batch)")
 
    args = parser.parse_args()
    
    # Get model path
    model_path = AVAILABLE_MODELS[args.model]
    
    # Get images to process
    image_dir = os.path.abspath(args.dir)
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                        glob.glob(os.path.join(image_dir, "*.jpeg")) + 
                        glob.glob(os.path.join(image_dir, "*.png")))
    
    if not image_files:
        print(f"{Fore.RED}Error: No images found in {image_dir}")
        return
    
    # Create image processor
    processor = MoondreamImageProcessor(model_path=model_path, output_file=args.output, mode=args.mode)
    
    if args.mode == "single":
        # Process only the first image
        image_path = image_files[0]
        print(f"{Fore.YELLOW}Mode: Single | Image: {os.path.basename(image_path)}")
        print(f"{Fore.YELLOW}Model: {args.model} | Question: {args.question[:30]}...")
        
        processor.process_interactive(image_path, args.question)
    else:
        # Process all images
        print(f"{Fore.YELLOW}Mode: Batch | Images: {len(image_files)}")
        print(f"{Fore.YELLOW}Model: {args.model} | Workers: {args.workers or 'Auto'}")
        
        processor.process_batch(image_files, args.question, max_workers=args.workers)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Interrupted by user.")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}")
        raise
