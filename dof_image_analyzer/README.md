# DOF Image Analyzer (Gemini 2.5 Flash‑Lite)

Analiza imágenes extraídas de documentos del Diario Oficial de la Federación (DOF) usando la API de Google Gemini 2.5 Flash‑Lite y escribe descripciones directamente como texto alternativo (alt) en los archivos Markdown correspondientes.

## Flujo DOF
- `get_word_dof.py` → descarga documentos del DOF.
- `dof_processor.py` → convierte archivos DOC a DOCX.
- `dof_docx_to_md.py` → transforma DOCX a Markdown y extrae imágenes.
- `dof_image_analyzer.py` → analiza imágenes y agrega descripciones (este script).
 - Nota: Este es el último paso del flujo antes de convertir la información a embeddings.

## Requisitos
- `GEMINI_API_KEY` definido en `.env`.
- Imágenes extraídas previamente por `dof_docx_to_md.py`.
- Estructura de directorios: `year/month/DDMMYYYY/EDICIÓN/media_temp/media/`.

## Uso rápido
- Procesar todo: `uv run dof_image_analyzer.py`
- Día específico: `uv run dof_image_analyzer.py 15/03/2024`
- Rango de fechas: `uv run dof_image_analyzer.py 01/03/2024 07/03/2024`
- Sin límites de velocidad: `uv run dof_image_analyzer.py --no-limits`
- Ayuda detallada: `uv run dof_image_analyzer.py --help`

## Opciones principales
- `date` (argumento): fecha inicial en `DD/MM/YYYY`. Si se omite, procesa todo.
- `end_date` (argumento): fecha final del rango en `DD/MM/YYYY`.
- `--input-dir`: ruta base con la estructura e imágenes (por defecto `./dof_word_md`).
- `--log-level`: `DEBUG` | `INFO` | `WARNING` | `ERROR` (por defecto `INFO`).
- `--no-limits`: desactiva el control de velocidad (recomendado solo con cuota de pago).

## Comportamiento
- Respeta por defecto ~15 RPM (4 s entre peticiones).
- Solo inserta descripción en imágenes sin texto alternativo existente.
- Calcula tiempo estimado de procesamiento y genera bitácora en `dof_image_analyzer.log`.

## Advertencias
- `--no-limits` puede exceder cuotas gratuitas de Google.
- Requiere conexión estable a Internet para llamadas a la API.
- No sobreescribe alt text existente en los Markdown.

## Ejemplo de estructura esperada
```
./dof_word_md/
  2024/
    03/
      15032024/
        MAT/
          media_temp/
            media/
              img_001.png
        VES/
          media_temp/
            media/
              img_002.png
```