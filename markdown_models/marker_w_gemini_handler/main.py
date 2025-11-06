import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

# Rutas de entrada y salida
pdf_path = "../_pdf/tables/01042024-MAT_pages_53_to_54.pdf"
output_dir = "./markdown/test"

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Verificar que la clave API esté disponible en el entorno
if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("La variable de entorno GOOGLE_API_KEY no está definida. Por favor, configúrala antes de ejecutar el script.")

# Configuración para el LLM usando ConfigParser
config = {
    "llm_service": "marker.services.gemini.GoogleGeminiService",
    "gemini_api_key": os.environ["GOOGLE_API_KEY"],  # Pasar la clave desde el entorno
    "output_format": "markdown"  # Especificar formato de salida
}
config_parser = ConfigParser(config)

# Configurar el convertidor con soporte LLM
converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    llm_service=config_parser.get_llm_service()
)

try:
    # Realizar la conversión
    rendered = converter(pdf_path)
    
    # Extraer el texto markdown
    markdown_text, metadata, images = text_from_rendered(rendered)
    
    # Guardar el resultado en un archivo
    output_path = os.path.join(output_dir, "01042024-MAT_pages_53_to_54.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    
    print(f"Conversión completada. El archivo markdown se guardó en: {output_path}")
    print(f"Metadatos: {metadata}")
    print(f"Número de imágenes extraídas: {len(images)}")

except Exception as e:
    print(f"Ocurrió un error durante la conversión: {str(e)}")