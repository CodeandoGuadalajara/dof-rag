from PIL import Image
import time
import datetime
import os
import glob
import gc
import threading
import concurrent.futures
from google import genai
from google.genai import types
import re
from dotenv import load_dotenv

load_dotenv()

# Variable para el archivo de resultados, indicando el maximo de tokens por salida
archivo = "test_results_gemini_tokens_256.txt"

# Creamos un lock global para sincronizar el acceso a la API
api_lock = threading.Lock()

# Configuración de la API de Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-2.0-flash"  # Modelo a utilizar

# Configuración para limitar tokens
MAX_TOKENS = 256

def estimar_tokens(texto):
    """
    Estima el número de tokens en un texto.
    Esta es una estimación aproximada, ya que la tokenización exacta 
    depende del tokenizador específico que usa Gemini.
    """
    if not texto:
        return 0
        
    # Método simple basado en palabras (aproximado)
    palabras = re.findall(r'\w+|[^\w\s]', texto)
    num_palabras = len(palabras)
    
    # Algunos tokens son subpalabras, otros son múltiples palabras
    # Factor de ajuste aproximado: 1.3 tokens por palabra
    return round(num_palabras * 1.3)

def initialize_gemini_client():
    """
    Inicializa y retorna un cliente de Gemini.
    """
    # La API correcta es usar directamente el constructor del cliente con api_key
    return genai.Client(api_key=GEMINI_API_KEY)

def process_image_with_gemini(client, image_path, question):
    """
    Procesa una única imagen usando Gemini.
    Se protege cada llamada a la API con un lock para evitar problemas de concurrencia.
    Retorna un diccionario con los tiempos y las respuestas, o un error en caso de fallo.
    """
    result = {"image_path": image_path}
    start = time.time()
    
    try:
        image = Image.open(image_path)
    except Exception as e:
        result["error"] = f"Error abriendo imagen: {str(e)}"
        return result
    
    try:
        with api_lock:
            # Generar descripción de la imagen
            caption_start = time.time()
            caption_response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=["Describe esta imagen detalladamente.", image]
            )
            caption_time = time.time() - caption_start
            caption = caption_response.text if hasattr(caption_response, 'text') else str(caption_response)
            caption_tokens = estimar_tokens(caption)

            # Responder a la pregunta sobre la imagen
            query_start = time.time()
            answer_response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[question, image]
            )
            query_time = time.time() - query_start
            answer = answer_response.text if hasattr(answer_response, 'text') else str(answer_response)
            answer_tokens = estimar_tokens(answer)
        
        total_time = time.time() - start
        result["caption_time"] = caption_time
        result["query_time"] = query_time
        result["total_time"] = total_time
        result["caption"] = caption
        result["answer"] = answer
        result["caption_tokens"] = caption_tokens
        result["answer_tokens"] = answer_tokens
        result["total_tokens"] = caption_tokens + answer_tokens
    except Exception as e:
        result["error"] = f"Error procesando imagen con Gemini: {str(e)}"
    
    return result

def guardar_resultado_imagen(img_res, log_file=archivo):
    """
    Guarda el resultado de una imagen en el archivo de resultados (modo append).
    """
    separator = "\n" + "=" * 80 + "\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(separator)
        f.write("Timestamp: " + datetime.datetime.now().isoformat() + "\n")
        f.write("Modelo: Gemini " + GEMINI_MODEL + "\n")
        f.write("Imagen: " + img_res.get("image_path", "N/A") + "\n")
        if "error" in img_res:
            f.write("Error: " + img_res["error"] + "\n")
        else:
            f.write("Tiempo de descripción: {:.4f} s\n".format(img_res.get("caption_time", 0)))
            f.write("Tiempo de query: {:.4f} s\n".format(img_res.get("query_time", 0)))
            f.write("Tiempo total de la imagen: {:.4f} s\n".format(img_res.get("total_time", 0)))
            f.write("Tokens en descripción: {}\n".format(img_res.get("caption_tokens", 0)))
            f.write("Tokens en respuesta: {}\n".format(img_res.get("answer_tokens", 0)))
            f.write("Total de tokens: {}\n".format(img_res.get("total_tokens", 0)))
            f.write("Descripción: " + img_res.get("caption", "") + "\n")
            f.write("Respuesta: " + img_res.get("answer", "") + "\n")
        f.write(separator + "\n")

def process_images_with_threads(image_files, question, max_workers=None):
    """
    Procesa las imágenes en hilos usando ThreadPoolExecutor y la API de Gemini.
    """
    results = {}
    results["model"] = GEMINI_MODEL
    start_total = time.time()
    
    try:
        init_start = time.time()
        client = initialize_gemini_client()
        init_time = time.time() - init_start
        results["init_time"] = init_time
    except Exception as e:
        results["error"] = f"Error inicializando el cliente de Gemini: {str(e)}"
        return results

    results["images"] = []
    total_images = len(image_files)
    completed = 0

    # Calcular el 70% de los núcleos disponibles
    if max_workers is None:
        num_cpu = os.cpu_count() or 1
        max_workers = max(1, int(num_cpu * 0.7))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image_with_gemini, client, img_path, question): img_path for img_path in image_files}
        for future in concurrent.futures.as_completed(futures):
            image_result = future.result()
            results["images"].append(image_result)
            completed += 1

            # Guardar inmediatamente el resultado de la imagen en el archivo
            guardar_resultado_imagen(image_result, log_file=archivo)

            elapsed = time.time() - start_total
            avg_time = elapsed / completed
            remaining = total_images - completed
            estimated_remaining = remaining * avg_time
            print(f"[Gemini] Imagen {completed}/{total_images} procesada en {image_result.get('total_time', 0):.2f} s. "
                  f"Tiempo total: {elapsed:.2f} s. Estimado restante: {estimated_remaining:.2f} s", flush=True)
    
    results["total_time"] = time.time() - start_total
    return results

if __name__ == "__main__":
    # Directorio con las imágenes de prueba
    image_directory = "./imagenes_prueba"
    image_files = glob.glob(os.path.join(image_directory, "*.*"))
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_files = [img for img in image_files if img.lower().endswith(valid_extensions)]
    
    if not image_files:
        print("No se encontraron imágenes en el directorio:", image_directory)
        exit(1)
    
    # Pregunta en español (Gemini es multilingüe, por lo que no necesitamos traducción)
    question = "¿Qué se observa en esta imagen?"
    
    # Limpiar el archivo de resultados al inicio
    with open(archivo, "w", encoding="utf-8") as f:
        f.write("Resultados de predicciones con Gemini:\n")
    
    print("\n" + "=" * 80)
    print(f"Iniciando predicciones con Gemini ({GEMINI_MODEL})")
    results = process_images_with_threads(image_files, question)
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Procesamiento finalizado en {results.get('total_time', 0):.2f} s")
    print("=" * 80 + "\n")
