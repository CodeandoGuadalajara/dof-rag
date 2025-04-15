# 📝 Conversión de PDF a Markdown (DOF)

Este módulo forma parte del sistema RAG en desarrollo, y se enfoca **exclusivamente en la experimentación de distintas técnicas para convertir archivos PDF a formato Markdown**. El objetivo principal es evaluar cuál método ofrece una mejor extracción estructurada del contenido del Diario Oficial de la Federación (DOF) para su posterior uso en el sistema de recuperación aumentada por generación.

---

## 📄 Documento base utilizado

- Se utilizó un DOF del **11 de enero de 2023**.
- El documento original contenía **aproximadamente 700 páginas**.
- Para facilitar las pruebas, se **recortó a las primeras 20 páginas**, lo cual permite una iteración más rápida durante el desarrollo.

El archivo de entrada se encuentra en:

```
markdown_models/_pdf/11012023-MAT-20pages.pdf
```

---

## 📁 Estructura de carpetas

Cada carpeta contiene un enfoque distinto de conversión:

```
markdown_models/
├── _pdf/                        # PDF de entrada (20 páginas del DOF)
├── docling_handler/            # Conversión usando Docling
├── gemini_handler/             # Conversión usando Gemini
├── marker_w_gemini_handler/    # Gemini combinado con Marker
├── pymupdf_handler/            # Conversión usando PyMuPDF
├── pymupdf4llm_handler/        # Conversión usando PyMuPDF4LLM
```

Cada carpeta incluye su propio entorno virtual y un script `main.py` que realiza la conversión.

---

## ⚙️ Instrucciones para ejecutar los scripts

Cada handler puede ejecutarse de forma independiente. Para correr cualquier modelo:

1. Abre una terminal y navega al directorio del handler que deseas probar:
   ```bash
   cd markdown_models/<handler>
   # Ejemplo:
   cd markdown_models/docling_handler
   ```

2. Crea el entorno virtual con [`uv`](https://github.com/astral-sh/uv) y sincroniza las dependencias:
   ```bash
   uv venv
   uv sync
   ```

3. Ejecuta el script de conversión:
   ```bash
   python main.py
   ```

---

## 🧪 Sobre el script `main.py`

Cada script `main.py` sigue esta estructura básica:

- Define la ruta de entrada y salida:
  ```python
  input_pdf = "../_pdf/11012023-MAT-20pages.pdf"
  output_md = "./markdown/2024/11012023-MAT.md"
  ```

- Convierte el PDF a Markdown usando la herramienta específica.
- Guarda el contenido Markdown generado en un archivo local.
- Imprime el tiempo de procesamiento.

Ejemplo de salida:

```
✅ Conversión completada. Markdown guardado en: ./markdown/2024/11012023-MAT.md
Tiempo de procesamiento: 2.87 segundos
```

---

## 📌 Nota

Este módulo **no representa el README principal del proyecto**, sino que documenta únicamente los experimentos relacionados con la conversión de PDF a Markdown dentro del contexto del sistema RAG.
