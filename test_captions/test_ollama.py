# /// script
# dependencies = [
#   "llm-ollama>=0.9.1",
#   "llm>=0.23",
#   "imagehash>=4.3.2",
#   "Pillow",
#   "python-dotenv",
#   "tqdm"
# ]
# ///

"""
Script for processing images using a local LLM (Ollama) with focus on image description.
Uses threads for parallel processing and ensures model access through threading.Lock().
"""

import os
import glob
import time
import datetime
import re
import argparse
import logging
import base64
import mimetypes
import llm
from llm import UnknownModelError
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Basic logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalImageAttachment:
    """
    Class to wrap a local image and provide its content in base64,
    as well as resolve its MIME type.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def base64_content(self):
        with open(self.file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def resolve_type(self):
        mime_type, _ = mimetypes.guess_type(self.file_path)
        return mime_type or "application/octet-stream"


class OllamaImageProcessor:
    """
    Class for processing images using a local Ollama model.
    Provides methods to get basic image information, generate a prompt,
    and process the image with the model in a multi-threaded environment.
    """
    def __init__(self, model="gemma3:4b", temperature=0.5, max_tokens=4096,
                top_p=0.5, num_ctx=8192, top_k=20, output_file=None, mode="single"):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.num_ctx = num_ctx
        self.top_k = top_k  
        self.mode = mode
        
        # Output file management
        if output_file is None:
            model_clean = model.replace(":", "_")
            mode_suffix = "single" if mode == "single" else "batch"
            self.output_file = f"./tests/results/ollama__{model_clean}_{mode_suffix}_tokens_{max_tokens}.txt"
        else:
            self.output_file = output_file
            
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
        self.token_factor = 1.3  # Factor to estimate tokens

        # Try to load the model once
        try:
            self.model = llm.get_model(model)
        except UnknownModelError as e:
            logger.error(f"Error getting model {model}: {e}")
            raise

        # Lock to protect model access in multi-threaded environments
        self.model_lock = threading.Lock()

    def extract_basic_image_info(self, image_path):
        """
        Extracts basic information from an image: name, format, mode, and dimensions.
        """
        try:
            info = {
                "filename": os.path.basename(image_path),
                "path": image_path
            }
            with Image.open(image_path) as img:
                info["format"] = img.format
                info["mode"] = img.mode
                info["size"] = img.size
            return info
        except Exception as e:
            logger.warning(f"Error extracting information from {image_path}: {e}")
            return {"filename": os.path.basename(image_path),
                    "path": image_path,
                    "error": str(e)}

    def create_description_prompt(self, image_info):
        """
        Creates an effective prompt to request a detailed description of the image.
        """
        prompt = f"""Necesito que analices esta imagen y proporciones un resumen conciso de su contenido principal.
            
            La imagen es un archivo {image_info.get('format', 'desconocido')} de 
            {image_info.get('size', (0, 0))[0]}x{image_info.get('size', (0, 0))[1]} píxeles.

            Tu tarea:
            1. Resumir brevemente qué contiene la imagen en 2-3 frases.
            2. Si hay texto visible, proporcionar un resumen de su contenido principal (no transcripción completa).
            3. Si es una gráfica o diagrama, explicar qué tipo de datos o información representa.
            4. Si es un mapa, identificar qué región o lugar muestra y cualquier elemento destacado.
            5. Identificar el propósito o contexto probable de la imagen (educativo, informativo, promocional, etc).

            Tu respuesta debe ser clara y directa, entre 100-200 palabras, en español.
            Enfócate en IDENTIFICAR y RESUMIR, no en describir detalles visuales menores.
            """
        return prompt

    def estimate_tokens(self, text):
        """
        Estimates the number of tokens in a text (approximation based on words and signs).
        """
        if not text:
            return 0
        words = re.findall(r'\w+|[^\w\s]', text)
        return int(len(words) * self.token_factor)

    def process_image(self, image_path, max_retries=3):
        """
        Processes an image and gets its description using the local model.
        Uses threading.Lock() to protect model access. In case of very short responses,
        retries by adding instructions to the prompt.
        """
        start_time = time.time()
        image_info = self.extract_basic_image_info(image_path)
        base_prompt = self.create_description_prompt(image_info)
        prompt = base_prompt

        # Prepare the image as attachment
        attachment = LocalImageAttachment(os.path.abspath(image_path))

        retries = 0
        while retries <= max_retries:
            try:
                with self.model_lock:
                    base_params = {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_ctx": self.num_ctx,
                        "top_k": self.top_k,
                        "num_predict": self.max_tokens,
                        "stop": ["###"]
                    }
                    response = self.model.prompt(prompt, attachments=[attachment], **base_params)
                response_text = response.text()

                # If the response is too short (less than 100 characters), retry
                if len(response_text.strip()) < 100:
                    if retries < max_retries:
                        retries += 1
                        logger.warning(f"Response too short for {image_path}, retry {retries}/{max_retries}")
                        # Enhanced prompt for retry that emphasizes the need for more details
                        prompt = base_prompt + """
                            NOTA IMPORTANTE: Tu respuesta anterior fue demasiado breve o incompleta.

                            Por favor, proporciona una respuesta más completa siguiendo estas instrucciones específicas:

                            1. IDENTIFICA CLARAMENTE el tipo de imagen: fotografía, gráfico, documento, mapa, etc.

                            2. DESCRIBE DE MANERA ESPECÍFICA:
                            - Si contiene texto: resume su contenido principal y menciona títulos o encabezados importantes
                            - Si es un gráfico/tabla: indica qué tipo de datos muestra y conclusiones principales
                            - Si es un mapa: especifica la región/lugar y elementos geográficos destacados
                            - Si es una fotografía: menciona objetos, personas, colores predominantes y composición general

                            3. CONTEXTUALIZA la imagen: indica su propósito probable (educativo, informativo, comercial, etc.)

                            Tu respuesta debe tener entre 150-200 palabras y ser ESTRUCTURADA según los puntos anteriores.
                            No te extiendas en detalles visuales menores, pero asegúrate de IDENTIFICAR todos los elementos importantes.
                            """
                        continue

                duration = time.time() - start_time
                tokens = self.estimate_tokens(response_text)
                logger.info(f"Processed {image_path} in {duration:.2f}s, estimated tokens: {tokens}")

                return {
                    "image_path": image_path,
                    "response": response_text.strip(),
                    "processing_time": duration,
                    "token_count": tokens,
                    "prompt_used": prompt
                }

            except Exception as e:
                if retries == max_retries:
                    duration = time.time() - start_time
                    logger.error(f"Error processing {image_path} after {max_retries} attempts: {e}")
                    return {"image_path": image_path,
                            "error": str(e),
                            "processing_time": duration}
                retries += 1
                logger.warning(f"Error in attempt {retries}/{max_retries} for {image_path}: {e}")
                time.sleep(1)  # Pause between retries

        duration = time.time() - start_time
        return {"image_path": image_path,
                "error": "Could not get a valid response after multiple attempts",
                "processing_time": duration}

    def process_batch(self, image_files, max_workers=None):
        """
        Processes a list of images in parallel using ThreadPoolExecutor.
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_image, img): img for img in image_files}
            for future in as_completed(futures):
                img_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed: {img_path}")
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    results.append({"image_path": img_path,
                                    "error": str(e),
                                    "processing_time": 0})
        return results

    def save_results(self, results):
        """
        Saves the processing results to a text file with standardized format.
        Uses append to accumulate results in multiple executions.
        """
        # Check if the file exists to decide whether to add header
        file_exists = os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0
        
        with open(self.output_file, "a", encoding="utf-8") as f:
            # Only add header if the file is new
            if not file_exists:
                f.write(f"Image description results with Ollama - Model: {self.model_name} - max_tokens: {self.max_tokens}\n\n")
            
            # Execution mark
            separator = "\n" + "=" * 80 + "\n"
            f.write(separator)
            f.write(f"NEW EXECUTION: {datetime.datetime.now().isoformat()}\n")
            f.write(separator)
            
            # Write each image result
            for res in results:
                f.write(separator)
                f.write("IMAGE RESULTS\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Model: {self.model_name}, Max tokens: {self.max_tokens}\n")
                f.write(f"Image: {os.path.basename(res.get('image_path', 'N/A'))}\n")
                
                if "error" in res:
                    f.write(f"Error: {res['error']}\n")
                else:
                    f.write("\nMETRICS:\n")
                    f.write(f"- Description: {res.get('processing_time', 0):.4f}s | ")
                    f.write(f"Total: {res.get('processing_time', 0):.4f}s | ")
                    f.write(f"Tokens: {res.get('token_count', 0)}\n")
                    
                    f.write("\nRESULTS:\n")
                    f.write("Question: What can be observed in this image?\n\n")
                    f.write(f"Description: {res.get('response', '')}\n")
                
                f.write(separator + "\n")
            
            # Final summary
            self._save_final_summary(results)
                
        logger.info(f"Results added in: {self.output_file}")
    

    def _save_final_summary(self, results):
        """
        Saves a final summary of all results at the end of the file.
        """
        separator = "\n" + "=" * 80 + "\n"
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(separator)
            f.write("FINAL PROCESSING SUMMARY\n")
            f.write(f"Completion timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Model: {self.model_name}, Max tokens: {self.max_tokens}\n")
            
            # Basic statistics
            total_images = len(results)
            successful_images = sum(1 for res in results if "error" not in res)
            failed_images = total_images - successful_images
            
            f.write(f"Total images processed: {total_images}\n")
            f.write(f"Successful: {successful_images}, Errors: {failed_images}\n")
            
            # Total time and tokens
            total_time = sum(res.get("processing_time", 0) for res in results if "processing_time" in res)
            total_tokens = sum(res.get("token_count", 0) for res in results if "token_count" in res)
            
            f.write(f"Total processing time: {total_time:.2f} seconds\n")
            
            if successful_images > 0:
                # Calculate averages
                avg_time = total_time / successful_images
                avg_tokens = total_tokens / successful_images
                
                f.write("\nAverage time per image:\n")
                f.write(f"- Total: {avg_time:.4f}s | ")
                f.write(f"Tokens: {avg_tokens:.1f}\n")
            
            f.write(separator)


    def process_interactive(self, image_path):
        """
        Processes an image in interactive mode (similar to moondream).
        """
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"Model: {self.model_name}")
        
        try:
            # Process image
            print("Processing image...")
            result = self.process_image(image_path)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                # Show results in compact form
                print("\nTimes: ", end="")
                print(f"Processing={result['processing_time']:.2f}s | ", end="")
                print(f"Estimated tokens={result['token_count']}")
                
                print("\nDescription:")
                print(f"{result['response']}")
            
            # Save result in common format
            self.save_single_result(result)
            print(f"\nResults saved: {os.path.basename(self.output_file)}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def save_single_result(self, img_res):
        """
        Saves the result of a single image processed in interactive mode.
        Uses append to accumulate results in multiple executions.
        """
        # Check if the file exists to decide whether to add header
        file_exists = os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0
        
        with open(self.output_file, "a", encoding="utf-8") as f:
            # Only add header if the file is new
            if not file_exists:
                f.write(f"Image description results with Ollama - Model: {self.model_name} - max_tokens: {self.max_tokens}\n\n")
            
            separator = "=" * 80 + "\n"
            f.write(separator)
            f.write("IMAGE RESULTS\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Model: {self.model_name}, Max tokens: {self.max_tokens}\n")
            f.write(f"Image: {os.path.basename(img_res.get('image_path', 'N/A'))}\n")
            
            if "error" in img_res:
                f.write(f"Error: {img_res['error']}\n")
            else:
                f.write("\nMETRICS:\n")
                f.write(f"- Description: {img_res.get('processing_time', 0):.4f}s | ")
                f.write(f"Total: {img_res.get('processing_time', 0):.4f}s | ")
                f.write(f"Tokens: {img_res.get('token_count', 0)}\n")
                
                f.write("\nRESULTS:\n")
                f.write("Question: What can be observed in this image?\n\n")
                f.write(f"Description: {img_res.get('response', '')}\n")
            
            f.write("\n" + separator)


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Image processing with local LLM (Ollama) using threads"
    )
    parser.add_argument("--image_dir", "-d", type=str, default="./imagenes_prueba",
                        help="Directory with images to process (default: ./imagenes_prueba)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output filename for results")
    parser.add_argument("--model", "-m", default="gemma3:4b",
                        help="Model to use (default: gemma3:4b)")
    parser.add_argument("--temperature", "-t", type=float, default=0.5,
                        help="Base temperature for the model (default: 0.5)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum number of tokens to generate (default: 4096)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Maximum number of threads to use (default: automatic)")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Limit token selection to the k most likely (default: 40)")
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                        help="Mode: 'single' (one image) or 'batch' (all) (default: single)")
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = os.path.abspath(args.image_dir)
    logger.info(f"Image directory: {image_dir}")

    # Find images in the directory with valid extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    image_files = glob.glob(os.path.join(image_dir, "*"))
    image_files = [img for img in image_files if img.lower().endswith(valid_extensions)]

    if not image_files:
        logger.error(f"No images found in directory: {image_dir}")
        return
    
    # Sort images for consistency
    image_files = sorted(image_files)

    # Simplify output filename generation
    model_clean = args.model.replace(":", "_")
    output_file = args.output
    if output_file is None:
        mode_suffix = "single" if args.mode == "single" else "batch"
        output_file = f"./tests/results/ollama__{model_clean}__{mode_suffix}__tokens_{args.max_tokens}.txt"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize the processor
    processor = OllamaImageProcessor(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.5,
        num_ctx=4096,
        top_k=20,
        output_file=output_file,
        mode=args.mode
    )

    # Calculate number of threads if not specified (default 70% of cores)
    if args.threads is None:
        cpu_count = os.cpu_count() or 1
        threads = max(1, int(cpu_count * 0.7))
    else:
        threads = args.threads

    # Logic based on selected mode
    if args.mode == "single":
        # Process only the first image
        image_path = image_files[0]
        print(f"Mode: Single | Image: {os.path.basename(image_path)}")
        print(f"Model: {args.model} | Max tokens: {args.max_tokens}")
        
        processor.process_interactive(image_path)
    else:
        # Process all images in batch
        print(f"Mode: Batch | Images: {len(image_files)}")
        print(f"Model: {args.model} | Workers: {threads}")
        
        start_time = time.time()
        results = processor.process_batch(image_files, max_workers=threads)
        total_time = time.time() - start_time

        # Display results summary
        success_count = sum(1 for res in results if "error" not in res)
        total_tokens = sum(res.get("token_count", 0) for res in results if "token_count" in res)

        print(f"Processing completed in {total_time:.2f}s")
        print(f"Images successfully processed: {success_count}/{len(results)}")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Results saved in: {os.path.basename(output_file)}")

        # Save results
        processor.save_results(results)


if __name__ == "__main__":
    main()
