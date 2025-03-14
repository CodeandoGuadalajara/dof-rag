import moondream as md
from PIL import Image
import time
import datetime
import os
import glob
import gc
import threading
import concurrent.futures

# Importamos las librerías para traducción con MarianMT
from transformers import MarianMTModel, MarianTokenizer

# Cargamos el modelo de traducción (del inglés al español) de forma global para mayor eficiencia
translation_model_name = "Helsinki-NLP/opus-mt-en-es"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

def translate_text(text):
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = translation_model.generate(**inputs)
    translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Variable para el modelo y archivo de resultados
model_paths = [
    "model/moondream-2b-int8.mf",
    # "model/moondream-0_5b-int8.mf"
]
# archivo = "test_results-0_5b.txt"
archivo = "test_results-2b.txt"

# Creamos un lock global para sincronizar el acceso al modelo compartido
model_lock = threading.Lock()

def process_image_shared(model, image_path, question):
    """
    Procesa una única imagen usando el modelo compartido.
    Se protege cada llamada al modelo con un lock para evitar accesos concurrentes.
    Retorna un diccionario con los tiempos (codificación, caption, query, total),
    además del caption y answer (y sus traducciones), o un error en caso de fallo.
    """
    result = {"image_path": image_path}
    start = time.time()
    
    try:
        image = Image.open(image_path)
    except Exception as e:
        result["error"] = f"Error abriendo imagen: {str(e)}"
        return result
    
    try:
        with model_lock:
            # Codificar la imagen
            encode_start = time.time()
            encoded_image = model.encode_image(image)
            encode_time = time.time() - encode_start

            # Generar caption
            caption_start = time.time()
            caption_result = model.caption(encoded_image)
            caption_time = time.time() - caption_start
            caption = caption_result.get("caption", "")

            # Ejecutar la query
            query_start = time.time()
            answer_result = model.query(encoded_image, question)
            query_time = time.time() - query_start
            answer = answer_result.get("answer", "")
        
        total_time = time.time() - start
        result["encode_time"] = encode_time
        result["caption_time"] = caption_time
        result["query_time"] = query_time
        result["total_time"] = total_time
        result["caption"] = caption
        result["answer"] = answer

        # Traducir las respuestas (caption y answer) del inglés al español
        result["caption_trad"] = translate_text(caption) if caption else ""
        result["answer_trad"] = translate_text(answer) if answer else ""
    except Exception as e:
        result["error"] = f"Error procesando imagen: {str(e)}"
    
    return result

def guardar_resultado_imagen(img_res, model_path, log_file=archivo):
    """
    Guarda el resultado de una imagen en el archivo de resultados (modo append).
    Se registra la información básica: tiempos, caption y answer, y sus traducciones.
    """
    separator = "\n" + "=" * 80 + "\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(separator)
        f.write("Timestamp: " + datetime.datetime.now().isoformat() + "\n")
        f.write("Modelo: " + model_path + "\n")
        f.write("Imagen: " + img_res.get("image_path", "N/A") + "\n")
        if "error" in img_res:
            f.write("Error: " + img_res["error"] + "\n")
        else:
            f.write("Tiempo de codificación: {:.4f} s\n".format(img_res.get("encode_time", 0)))
            f.write("Tiempo de caption: {:.4f} s\n".format(img_res.get("caption_time", 0)))
            f.write("Tiempo de query: {:.4f} s\n".format(img_res.get("query_time", 0)))
            f.write("Tiempo total de la imagen: {:.4f} s\n".format(img_res.get("total_time", 0)))
            f.write("Caption (inglés): " + img_res.get("caption", "") + "\n")
            f.write("Answer (inglés): " + img_res.get("answer", "") + "\n")
            f.write("Caption (traducido): " + img_res.get("caption_trad", "") + "\n")
            f.write("Answer (traducido): " + img_res.get("answer_trad", "") + "\n")
        f.write(separator + "\n")

def process_model_with_threads(model_path, image_files, question, max_workers=None):
    """
    Carga el modelo una única vez y procesa las imágenes en hilos usando ThreadPoolExecutor.
    Se calcula el número de workers a utilizar como el 70% de los núcleos disponibles,
    para no saturar el entorno en sistemas limitados.
    Se muestra el progreso en consola y se retornan los resultados.
    """
    results = {}
    results["model_path"] = model_path
    start_total = time.time()
    
    try:
        init_start = time.time()
        model = md.vl(model=model_path)
        init_time = time.time() - init_start
        results["init_time"] = init_time
    except Exception as e:
        results["error"] = f"Error inicializando el modelo: {str(e)}"
        return results

    results["images"] = []
    total_images = len(image_files)
    completed = 0

    # Calcular el 70% de los núcleos disponibles
    if max_workers is None:
        num_cpu = os.cpu_count() or 1
        max_workers = max(1, int(num_cpu * 0.7))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image_shared, model, img_path, question): img_path for img_path in image_files}
        for future in concurrent.futures.as_completed(futures):
            image_result = future.result()
            results["images"].append(image_result)
            completed += 1

            # Guardar inmediatamente el resultado de la imagen en el archivo (append)
            guardar_resultado_imagen(image_result, model_path, log_file=archivo)

            elapsed = time.time() - start_total
            avg_time = elapsed / completed
            remaining = total_images - completed
            estimated_remaining = remaining * avg_time
            print(f"[{model_path}] Imagen {completed}/{total_images} procesada en {image_result.get('total_time', 0):.2f} s. "
                  f"Tiempo total: {elapsed:.2f} s. Estimado restante: {estimated_remaining:.2f} s", flush=True)
    
    results["total_time"] = time.time() - start_total

    try:
        if hasattr(model, "close"):
            model.close()
    except Exception as e:
        print(f"Error cerrando el modelo {model_path}: {str(e)}")
    del model
    gc.collect()
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
    
    # Pregunta en español
    question = "¿Qué se observa en esta imagen?"
    
    # Limpiar el archivo de resultados al inicio
    with open(archivo, "w", encoding="utf-8") as f:
        f.write("Resultados de predicciones:\n")
    
    # Procesar cada modelo de forma secuencial
    for model_path in model_paths:
        print("\n" + "=" * 80)
        print(f"Iniciando predicciones con el modelo: {model_path}")
        results = process_model_with_threads(model_path, image_files, question)
        if "error" in results:
            print(f"Error en el modelo {model_path}: {results['error']}")
        else:
            print(f"Modelo {model_path} finalizado en {results.get('total_time', 0):.2f} s")
        print("=" * 80 + "\n")
