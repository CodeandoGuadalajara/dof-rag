from docling.document_converter import DocumentConverter
import pathlib
import time

# Iniciar medición del tiempo
start_time = time.time()

# Definir rutas de entrada y salida
input_pdf = "../_pdf/tables/11012023-MAT-20pages.pdf"
output_md = "./markdown/test/11012023-MAT-20pages.md"

# Asegurar que la ruta de salida exista
output_path = pathlib.Path(output_md)
output_path.parent.mkdir(parents=True, exist_ok=True)

# Crear instancia del convertidor
converter = DocumentConverter()

# Convertir el PDF a un objeto DoclingDocument
docling_doc = converter.convert(input_pdf)

# Extraer el contenido en formato Markdown
markdown_text = docling_doc.document.export_to_markdown()

# Guardar el contenido en un archivo Markdown
output_path.write_text(markdown_text, encoding="utf-8")

# Calcular tiempo de procesamiento
end_time = time.time()
processing_time = end_time - start_time

print(f"✅ Conversión completada. Markdown guardado en: {output_md}")
print(f"Tiempo de procesamiento: {processing_time:.2f} segundos")