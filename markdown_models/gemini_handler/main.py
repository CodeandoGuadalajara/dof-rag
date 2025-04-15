import google.generativeai as genai
from pdf2image import convert_from_path
import pathlib
import os
import time
from typing import List

# Iniciar el temporizador
start_time = time.time()

# Configurar la API de Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Ruta de Poppler (ajusta según tu sistema)
poppler_path = r"C:\poppler-24.08.0\Library\bin"

# Definir rutas de entrada y salida
input_pdf = "../_pdf/tables/01042024-MAT_pages_53_to_54.pdf"
output_md = "./markdown/test/01042024-MAT_pages_53_to_54.md"

# Asegurar que la carpeta de salida exista
pathlib.Path(output_md).parent.mkdir(parents=True, exist_ok=True)

# Convertir el PDF a imágenes
try:
    images = convert_from_path(input_pdf, poppler_path=poppler_path)
    page_count = len(images)
    print(f"📄 Total de páginas detectadas: {page_count}")
except Exception as e:
    print(f"❌ Error al convertir el PDF a imágenes: {e}")
    exit(1)

# Función para extraer Markdown usando Gemini
def extract_markdown_from_image(image, page_num: int) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = (
        "Extrae el contenido de esta página y conviértelo en Markdown.\n\n"
        "Divide el documento en secciones de aproximadamente 250 a 1000 palabras. "
        "Nuestro objetivo es identificar partes de la página con el mismo tema semántico. "
        "Estos fragmentos serán incrustados y utilizados en una pipeline RAG.\n\n"
        "No uses etiquetas HTML. En su lugar, al inicio de cada página, escribe 'Página: [número de página]'."
    )
    
    try:
        response = model.generate_content([image, prompt])
        markdown = response.text if response and response.text else ""
        
        # Si no hay respuesta, reintentar una vez
        if not markdown and response.candidates:
            time.sleep(2)
            response = model.generate_content("Continúa por favor.")
            markdown = response.text if response and response.text else ""
        
        # Asegurar que el número de página esté al inicio
        if not markdown.startswith(f"Página: {page_num}"):
            markdown = f"Página: {page_num}\n\n{markdown}"
        
        return markdown
    except Exception as e:
        print(f"⚠️ Error al procesar la página {page_num}: {e}")
        return f"Página: {page_num}\n\n*Contenido no procesado*"

# Extraer Markdown de cada página con su número correspondiente
markdown_pages = [extract_markdown_from_image(img, i + 1) for i, img in enumerate(images)]

# Unir todas las páginas con separadores
final_markdown = "\n\n---\n\n".join(markdown_pages)

# Guardar el archivo Markdown
try:
    pathlib.Path(output_md).write_text(final_markdown, encoding="utf-8")
    print(f"✅ Conversión completada. Markdown guardado en: {output_md}")
except Exception as e:
    print(f"❌ Error al guardar el archivo Markdown: {e}")

# Calcular y mostrar el tiempo total
end_time = time.time()
total_time = end_time - start_time
print(f"Tiempo total de procesamiento: {total_time:.2f} segundos")