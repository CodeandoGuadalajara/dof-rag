# dof-rag

dof-rag es un chat y un sistema de consulta por generación aumentada para explorar las ediciones del Diario Oficial de la Federación de México.

# Requerimientos

Instala [uv](https://docs.astral.sh/uv/) para manejar las dependencias y ejecutar el proyecto.

Una vez instalado uv, ejecuta el siguiente comando para iniciar el proyecto:

```bash
uv venv # Crear un entorno virtual
uv sync # Sincronizar dependencias
```

## Bajar archivos del DOF

### Archivos PDF

Para bajar archivos PDF del DOF se usa el script `get_dof.py`:

```bash
uv run get_dof.py --help
uv run get_dof.py --start-year=2025 --end-year=2023
```

Esto crea directorios como:

```
dof/
├── 2025/
│   ├── 01/
│   │   ├── 02012025-MAT.pdf
│   │   ├── 03012025-MAT.pdf
...
```

### Archivos Word (.doc) — disponible desde 1999

Los archivos del DOF también están disponibles en formato Word (.doc), lo cual facilita la extracción de texto.

Para bajarlos se usa el script `get_word_dof.py`:

```bash
uv run get_word_dof.py --help
uv run get_word_dof.py --start-year=2025 --end-year=2023
```

Esto crea directorios como:

```
dof_word/
├── 2025/
│   ├── 01/
│   │   ├── 02012025/
│   │   │   ├── MAT/
│   │   │   │   └── 001_DOF_20250102_MAT_5746544.doc
│   │   │   └── VES/
│   │   │       └── 001_DOF_20250102_VES_5746544.doc
...
```

> **NOTA**: El script descarga un archivo .doc por cada documento legal individual (no por edición completa).

## Extraer markdown

Hay dos métodos de extracción dependiendo del tipo de archivo:

### Desde archivos Word (.doc) — 1999 en adelante

El script `convert_doc_to_md.py` convierte archivos `.doc` directamente a Markdown, manteniendo cada documento legal como un archivo individual — ideal para chunking y recuperación en RAG.

**Requisitos adicionales:**
- LibreOffice (`soffice`) — para conversión .doc → .docx
- pandoc — para conversión .docx → .md

```bash
# Convertir todos los años
python convert_doc_to_md.py --input-dir ./dof_word --output-dir ./dof_md

# Años específicos
python convert_doc_to_md.py --years 2020 2021 --workers 4

# Ver progreso sin convertir
python convert_doc_to_md.py --dry-run

# Reintentar archivos fallidos
python convert_doc_to_md.py --retry-failed
```

**Rendimiento:**
- ~9-10 archivos/segundo con 4 workers
- Tasa de fallo < 0.02% con reintentos automáticos
- Reanudable: omite archivos ya convertidos

**Estructura de salida:**

```
dof_md/
├── 2025/
│   ├── 01/
│   │   ├── 02012025/
│   │   │   ├── MAT/
│   │   │   │   └── 001_DOF_20250102_MAT_5746544.md
│   │   │   └── VES/
│   │   │       └── 001_DOF_20250102_VES_5746544.md
...
```

### Desde PDFs escaneados — antes de 1999

Los archivos del DOF anteriores a 1999 solo están disponibles como PDFs escaneados (imagen), por lo que requieren OCR. El script `extract_markdown.py` usa Gemini 2.0 Flash para extraer texto:

```bash
uv run extract_markdown.py --help
```

Los archivos Word (.doc) solo están disponibles desde 1999, por lo que los documentos anteriores requieren este método alternativo.

**Requisito:** Configurar la variable de entorno `GOOGLE_API_KEY` con una clave de Google AI.

### Desde PDFs digitales — alternativa

Para PDFs digitales (no escaneados), se puede usar [marker](https://github.com/VikParuchuri/marker):

```bash
marker --output_dir dof_markdown/2024/04/ \
  --paginate_output \
  --languages="es" \
  --skip_existing \
  --workers=1 \
  dof/2024/04/
```

## Extraer embeddings

Para extraer embeddings de un archivo específico:

```bash
python extract_embeddings.py dof_markdown/2024/04/
```

Puedes especificar la carpeta de un solo archivo, o la carpeta de un mes, o incluso la carpeta de un año.

## RAG PoC — Búsqueda híbrida con sqlite-vec + FTS5 (local ONNX)

En `rag_poc/` hay una prueba de concepto de motor RAG para el DOF usando:

- **Embeddings:** `pplx-embed-context-v1-0.6b` (Perplexity) corriendo localmente via ONNX Runtime — late chunking contextual
- **Vector search:** `sqlite-vec` — KNN en SQLite con virtual tables
- **Full-text search:** SQLite FTS5
- **Ranking híbrido:** Reciprocal Rank Fusion (RRF) sobre resultados vectoriales + texto

### Uso rápido

```bash
# Primera ejecución descarga el modelo ONNX (~1.2 GB) automáticamente

# Indexar un directorio de markdown
python -m rag_poc.cli index ./dof_md/2020/01/15012020/MAT

# Buscar
python -m rag_poc.cli search "subsidio federal articulo 47 vivienda"

# Ver estadísticas
python -m rag_poc.cli stats
```

### Chunking inteligente por patrón

El chunker clasifica cada documento en 5 patrones antes de dividir:

| Patrón | Trigger | Estrategia |
|---|---|---|
| `small` | < 10 KB | Un solo chunk |
| `h2_compound` | ≥2 H2 headings | Cada H2 = chunk atómico; si no cabe, partir por H3 |
| `bold_headers` | ≥2 líneas en negritas | Las negritas son metadato; split por párrafos |
| `plain_text` | Sin estructura | Split por párrafos con overlap |
| `giant_table` | >40% líneas son tabla | Cada tabla markdown = chunk; repetir header de columnas |

Ver `rag_poc/README.md` para detalles de arquitectura y late chunking.

## Estructura del proyecto

```
.
├── get_dof.py              # Descarga archivos PDF del DOF
├── get_word_dof.py         # Descarga archivos Word (.doc) del DOF (1999+)
├── convert_doc_to_md.py    # Convierte .doc → .md (pipeline individual)
├── extract_markdown.py     # Extrae texto de PDFs escaneados con Gemini (pre-1999)
├── extract_embeddings.py   # Extrae embeddings para RAG (pipeline anterior)
├── rag_poc/                # PoC de RAG híbrido (sqlite-vec + FTS5 + pplx-embed)
│   ├── cli.py              # CLI: index / search / stats
│   ├── chunker.py          # Chunking por patrón
│   ├── embedder.py         # Cliente de embeddings
│   ├── database.py         # SQLite + sqlite-vec + FTS5
│   ├── search.py           # Búsqueda híbrida con RRF
│   └── README.md           # Documentación del PoC
├── ai_agent.ipynb          # Notebook del agente de consulta
├── pandoc_filters/         # Filtros Lua para pandoc
├── modules_captions/       # Módulo de descripción de imágenes
├── Improve_embeddings_1_page_chunk/  # Mejoras de embeddings
├── pyproject.toml          # Dependencias del proyecto
└── README.md
```
