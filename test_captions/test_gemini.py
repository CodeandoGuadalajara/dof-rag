# /// script
# dependencies = [
#   "python-dotenv>=0.9.9",
#   "google>=0.3.0",
#   "google-genai>=1.3.0",
#   "pillow"
# ]
# ///

from PIL import Image
import time
import datetime
import os
import glob
import threading
import concurrent.futures
from google import genai
from google.genai import types
import re
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a global lock to synchronize API access
api_lock = threading.Lock()


class GeminiConfig:
    """
    Class to handle Gemini API configuration.
    """

    def __init__(
        self,
        model="gemini-2.0-flash",
        max_tokens=256,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        response_mime_type="text/plain",
    ):
        # Get API key from environment variables
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not found. Please configure it."
            )

        self.model = model
        self.max_tokens = max_tokens

        # Generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.response_mime_type = response_mime_type

    def get_generate_config(self):
        """Returns the configuration for content generation"""
        return types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_tokens,
            response_mime_type=self.response_mime_type,
        )

    def get_client(self):
        """Returns a client initialized with the configured API key"""
        return genai.Client(api_key=self.api_key)

    def get_output_filename(self):
        """Generates the output filename based on model and tokens"""

        # Use the full model name, replacing problematic characters for filenames
        model_name = self.model.replace(".", "-").replace(":", "_")
        return f"./tests/results/{model_name}_tokens_{self.max_tokens}.txt"


# Variable for results file
def initialize_config(token_limit=256):
    """Initializes configuration with specified token limit"""
    return GeminiConfig(max_tokens=token_limit)


# Initialize default configuration
GEMINI_CONFIG = initialize_config()
GEMINI_MODEL = GEMINI_CONFIG.model
MAX_TOKENS = GEMINI_CONFIG.max_tokens
output_file = GEMINI_CONFIG.get_output_filename()


def estimate_tokens(text):
    """
    Estimates the number of tokens in a text.
    This is an approximate estimation, as exact tokenization
    depends on the specific tokenizer used by Gemini.
    """
    if not text:
        return 0

    # Simple method based on words (approximate)
    words = re.findall(r"\w+|[^\w\s]", text)
    num_words = len(words)

    # Some tokens are subwords, others are multiple words
    # Approximate adjustment factor: 1.3 tokens per word
    return round(num_words * 1.3)


def initialize_gemini_client():
    """
    Initializes and returns a Gemini client.
    """
    return GEMINI_CONFIG.get_client()


def process_image_with_gemini(client, image_path, question, stream=False):
    """
    Processes a single image using Gemini.
    Each API call is protected with a lock to avoid concurrency issues.
    Returns a dictionary with times and responses, or an error in case of failure.

    If stream=True, uses streaming mode for response generation.
    """
    result = {"image_path": image_path}
    start = time.time()

    try:
        image = Image.open(image_path)
    except Exception as e:
        result["error"] = f"Error opening image: {str(e)}"
        return result

    try:
        with api_lock:
            # Generate image description
            caption_start = time.time()

            if not stream:
                caption_response = client.models.generate_content(
                    model=GEMINI_CONFIG.model,
                    contents=[
                        """Describe detalladamente la imagen en español, priorizando la información clave:  
                            - **Texto:** Si contiene texto (como en diagramas, infografías o documentos oficiales), transcribe el contenido principal.  
                            - **Mapas:** Menciona lugares, nombres geográficos y símbolos relevantes.  
                            - **Esquemas o diagramas:** Explica las relaciones o procesos principales.  
                            - **Logos:** Describe el diseño, colores y, si es reconocible, a qué entidad pertenece.  
                            - **Datos visuales:** Si hay gráficos o estadísticas, resume los valores más importantes.  
                            Sé claro y preciso, priorizando la información más relevante.""",
                        image,
                    ],
                    config=GEMINI_CONFIG.get_generate_config(),
                )
                caption = (
                    caption_response.text
                    if hasattr(caption_response, "text")
                    else str(caption_response)
                )
            else:
                # Streaming mode for description
                caption = ""
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_CONFIG.model,
                    contents=[
                        """Describe detalladamente la imagen en español, priorizando la información clave:  
            -               - **Texto:** Si contiene texto (como en diagramas, infografías o documentos oficiales), transcribe el contenido principal.  
                            - **Mapas:** Menciona lugares, nombres geográficos y símbolos relevantes.  
                            - **Esquemas o diagramas:** Explica las relaciones o procesos principales.  
                            - **Logos:** Describe el diseño, colores y, si es reconocible, a qué entidad pertenece.  
                            - **Datos visuales:** Si hay gráficos o estadísticas, resume los valores más importantes.  
                            Sé claro y preciso, priorizando la información más relevante.""",
                        image,
                    ],
                    config=GEMINI_CONFIG.get_generate_config(),
                ):
                    caption += chunk.text if hasattr(chunk, "text") else ""

            caption_time = time.time() - caption_start
            caption_tokens = estimate_tokens(caption)

            # Answer question about the image
            query_start = time.time()

            if not stream:
                answer_response = client.models.generate_content(
                    model=GEMINI_CONFIG.model,
                    contents=[
                        question
                        + " Sé muy breve y conciso. Responde COMPLETAMENTE en español. Máximo 100 palabras.",
                        image,
                    ],
                    config=GEMINI_CONFIG.get_generate_config(),
                )
                answer = (
                    answer_response.text
                    if hasattr(answer_response, "text")
                    else str(answer_response)
                )
            else:
                # Streaming mode for answer
                answer = ""
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_CONFIG.model,
                    contents=[
                        question
                        + " Sé muy breve y conciso. Responde COMPLETAMENTE en español. Máximo 100 palabras.",
                        image,
                    ],
                    config=GEMINI_CONFIG.get_generate_config(),
                ):
                    answer += chunk.text if hasattr(chunk, "text") else ""

            query_time = time.time() - query_start
            answer_tokens = estimate_tokens(answer)

        total_time = time.time() - start
        result["caption_time"] = caption_time
        result["query_time"] = query_time
        result["total_time"] = total_time
        result["caption"] = caption
        result["answer"] = answer
        result["caption_tokens"] = caption_tokens
        result["answer_tokens"] = answer_tokens
        result["total_tokens"] = caption_tokens + answer_tokens
        result["streaming"] = stream
    except Exception as e:
        result["error"] = f"Error processing image with Gemini: {str(e)}"

    return result


def process_text_with_gemini(client, prompt, stream=False):
    """
    Processes text using Gemini.
    """
    result = {"prompt": prompt}
    start = time.time()

    try:
        with api_lock:
            if not stream:
                response = client.models.generate_content(
                    model=GEMINI_CONFIG.model,
                    contents=prompt,
                    config=GEMINI_CONFIG.get_generate_config(),
                )
                text_response = (
                    response.text if hasattr(response, "text") else str(response)
                )
            else:
                # Streaming mode
                text_response = ""
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_CONFIG.model,
                    contents=prompt,
                    config=GEMINI_CONFIG.get_generate_config(),
                ):
                    text_response += chunk.text if hasattr(chunk, "text") else ""
                    # If you want to see the response in real-time, uncomment:
                    # print(chunk.text, end="", flush=True)

        process_time = time.time() - start
        tokens = estimate_tokens(text_response)

        result["response"] = text_response
        result["process_time"] = process_time
        result["tokens"] = tokens
        result["streaming"] = stream
    except Exception as e:
        result["error"] = f"Error processing text with Gemini: {str(e)}"

    return result


def save_image_result(img_res, log_file=None):
    """
    Saves the result of an image to the results file (append mode).
    """
    if log_file is None:
        log_file = GEMINI_CONFIG.get_output_filename()

    separator = "\n" + "=" * 80 + "\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(separator)
        f.write("IMAGE RESULTS\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Model: {GEMINI_MODEL}, Max tokens: {MAX_TOKENS}\n")
        f.write(f"Image: {os.path.basename(img_res.get('image_path', 'N/A'))}\n")

        if "error" in img_res:
            f.write("Error: " + img_res["error"] + "\n")
        else:
            f.write("\nMETRICS:\n")
            f.write(f"- Description: {img_res.get('caption_time', 0):.4f}s | ")
            f.write(f"Query: {img_res.get('query_time', 0):.4f}s | ")
            f.write(f"Total: {img_res.get('total_time', 0):.4f}s | ")
            f.write(f"Tokens: {img_res.get('total_tokens', 0)}\n")

            f.write("\nRESULTS:\n")
            f.write("Question: What can be observed in this image?\n\n")
            f.write(f"Description: {img_res.get('caption', '')}\n\n")
            f.write(f"Answer: {img_res.get('answer', '')}\n")
        f.write(separator + "\n")


def save_final_summary(results, log_file=None):
    """
    Saves a final summary of all results at the end of the file.
    """
    if log_file is None:
        log_file = GEMINI_CONFIG.get_output_filename()

    separator = "\n" + "=" * 80 + "\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(separator)
        f.write("FINAL PROCESSING SUMMARY\n")
        f.write(f"Completion timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Model: {GEMINI_MODEL}, Max tokens: {MAX_TOKENS}\n")

        # Basic statistics
        total_images = len(results.get("images", []))
        successful_images = sum(
            1 for img in results.get("images", []) if "error" not in img
        )
        failed_images = total_images - successful_images

        f.write(f"Total images processed: {total_images}\n")
        f.write(f"Successful: {successful_images}, Errors: {failed_images}\n")

        # Total time
        total_time = results.get("total_time", 0)
        f.write(f"Total processing time: {total_time:.2f} seconds\n")

        if successful_images > 0:
            # Calculate averages
            avg_caption_time = (
                sum(
                    img.get("caption_time", 0)
                    for img in results.get("images", [])
                    if "error" not in img
                )
                / successful_images
            )
            avg_query_time = (
                sum(
                    img.get("query_time", 0)
                    for img in results.get("images", [])
                    if "error" not in img
                )
                / successful_images
            )
            avg_total_time = (
                sum(
                    img.get("total_time", 0)
                    for img in results.get("images", [])
                    if "error" not in img
                )
                / successful_images
            )
            avg_tokens = (
                sum(
                    img.get("total_tokens", 0)
                    for img in results.get("images", [])
                    if "error" not in img
                )
                / successful_images
            )

            f.write("\nAverage time per image:\n")
            f.write(f"- Description: {avg_caption_time:.4f}s | ")
            f.write(f"Query: {avg_query_time:.4f}s | ")
            f.write(f"Total: {avg_total_time:.4f}s | ")
            f.write(f"Tokens: {avg_tokens:.1f}\n")

        f.write(separator)


def process_images_with_threads(
    image_files, question, max_workers=None, use_streaming=False
):
    """
    Processes images in threads using ThreadPoolExecutor and the Gemini API.

    Parameters:
    - image_files: List of image paths to process
    - question: Question to ask about each image
    - max_workers: Maximum number of workers in the ThreadPool
    - use_streaming: If True, uses streaming mode for responses
    """
    results = {}
    results["model"] = GEMINI_CONFIG.model
    results["max_tokens"] = GEMINI_CONFIG.max_tokens
    start_total = time.time()

    try:
        init_start = time.time()
        client = initialize_gemini_client()
        init_time = time.time() - init_start
        results["init_time"] = init_time
    except Exception as e:
        results["error"] = f"Error initializing Gemini client: {str(e)}"
        return results

    results["images"] = []
    total_images = len(image_files)
    completed = 0

    # Calculate 70% of available cores
    if max_workers is None:
        num_cpu = os.cpu_count() or 1
        max_workers = max(1, int(num_cpu * 0.7))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_image_with_gemini, client, img_path, question, use_streaming
            ): img_path
            for img_path in image_files
        }
        for future in concurrent.futures.as_completed(futures):
            image_result = future.result()
            results["images"].append(image_result)
            completed += 1

            # Immediately save the image result to the file
            save_image_result(image_result)

            elapsed = time.time() - start_total
            avg_time = elapsed / completed
            remaining = total_images - completed
            estimated_remaining = remaining * avg_time
            print(
                f"[Gemini] Image {completed}/{total_images} processed in {image_result.get('total_time', 0):.2f} s. "
                f"Total time: {elapsed:.2f} s. Estimated remaining: {estimated_remaining:.2f} s",
                flush=True,
            )

    results["total_time"] = time.time() - start_total

    # Save final summary
    save_final_summary(results)

    return results


if __name__ == "__main__":
    # Command line arguments configuration
    parser = argparse.ArgumentParser(description="Process images with Gemini API")
    parser.add_argument(
        "--tokens",
        type=int,
        default=256,
        choices=[256, 512, 1024, 2048, 4096, 8192],
        help="Maximum number of tokens in the response (default: 256)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for responses",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./imagenes_prueba",
        help="Directory with test images (default: ./imagenes_prueba)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="¿Qué se observa en esta imagen?",
        help="Question to ask about each image",
    )

    args = parser.parse_args()

    # Initialize configuration with specified tokens
    GEMINI_CONFIG = initialize_config(args.tokens)
    GEMINI_MODEL = GEMINI_CONFIG.model
    MAX_TOKENS = GEMINI_CONFIG.max_tokens
    output_file = GEMINI_CONFIG.get_output_filename()

    # Directory with test images
    image_directory = args.dir
    image_files = glob.glob(os.path.join(image_directory, "*.*"))
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
    image_files = [img for img in image_files if img.lower().endswith(valid_extensions)]

    if not image_files:
        print("No images found in directory:", image_directory)
        exit(1)

    # Question in Spanish (Gemini is multilingual, so we don't need translation)
    question = args.question

    # Clear the results file at the beginning
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            f"Prediction results with Gemini ({GEMINI_MODEL}) - max_tokens: {MAX_TOKENS}\n"
        )

    print("\n" + "=" * 80)
    print(
        f"Starting predictions with Gemini ({GEMINI_MODEL}) - max_tokens: {MAX_TOKENS}"
    )
    print(f"Streaming mode: {'Enabled' if args.streaming else 'Disabled'}")
    print(f"Results file: {output_file}")
    results = process_images_with_threads(
        image_files, question, use_streaming=args.streaming
    )
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Processing completed in {results.get('total_time', 0):.2f} s")
        print(f"Summary saved in: {output_file}")
    print("=" * 80 + "\n")
