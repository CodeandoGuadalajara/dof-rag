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

### Extraer markdown desde PDFs

UV instala la dependencia [marker](https://github.com/VikParuchuri/marker) que contiene un ejecutable para convertir PDFs a formato markdown.

Para convertir un folder completo:

```bash
marker --output_dir dof_markdown/2024/04/ \
  --paginate_output \
  --languages="es" \
  --skip_existing \
  --workers=1 \
  dof/2024/04/
```

### Archivos Word (.doc)

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

## Extraer markdown desde archivos Word (.doc)

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

## Extraer embeddings

Para extraer embeddings de un archivo específico:

```bash
python extract_embeddings.py dof_markdown/2024/04/
```

Puedes especificar la carpeta de un solo archivo, o la carpeta de un mes, o incluso la carpeta de un año.

## Estructura del proyecto

```
.
├── get_dof.py              # Descarga archivos PDF del DOF
├── get_word_dof.py         # Descarga archivos Word (.doc) del DOF
├── convert_doc_to_md.py    # Convierte .doc → .md (pipeline individual)
├── extract_embeddings.py   # Extrae embeddings para RAG
├── ai_agent.ipynb          # Notebook del agente de consulta
├── pandoc_filters/         # Filtros Lua para pandoc
├── modules_captions/       # Módulo de descripción de imágenes
├── Improve_embeddings_1_page_chunk/  # Mejoras de embeddings
├── pyproject.toml          # Dependencias del proyecto
└── README.md
```
