# dof-rag
dog-raf es un chat y un sistema de consulta por generación aumentada para explorar las ediciones del Diario Oficial de la Federación de México.

# Requerimientos

Instala [uv](https://docs.astral.sh/uv/) para manejar las dependencias y ejecutar el proyecto.

Una vez instalado uv, ejecuta el siguiente comando para iniciar el proyecto:

```bash
uv venv # Crear un entorno virtual
uv sync # Sincronizar dependencias
```

## Bajar archivos del DOF

Para bajar archivos del DOF se usa el script get_dof.py de la siguiente manera:

```bash
uv run get_dof.py --help
uv run get_dof.py --start-year=2025 --end-year=2023
```

Esto crea directorios como:

```
$ tree --sort=mtime dof | head -n 10
dof
├── 2025
│   ├── 01
│   │   ├── 02012025-MAT.pdf
│   │   ├── 03012025-MAT.pdf
│   │   ├── 06012025-MAT.pdf
│   │   ├── 07012025-MAT.pdf
...
```

## Extraer markdown:

### Por folders

UV instala la dependencia [marker](https://github.com/VikParuchuri/marker) que contiene un ejecutable para convertir PDFs a formato markdown.

Para convertir un folder completo ejecuta este comando:

```bash
marker --output_dir dof_markdown/2024/04/ \
  --paginate_output \
  --languages="es" \
  --skip_existing \
  --workers=1 \
  dof/2024/04/
# este comando tardó 2h 31m 23s en una macbook pro M3 de 36GB RAM
```

**NOTA**: Si el comando queda a la mitad, conviene borrar las carpetas incompletas.
Para ver cuáles archivos están incompletos puedes revisar el archivo markdown que tiene separaciones por hojas gracias a la opcíon `--paginate_output`
con el formato `{pagina}------------------------------------------------`. Revisa que contenga todas las páginas del archivo PDF.

Una vez borradas las carpetas incompletas, puedes volver a ejecutar el comando anterior para que el `--skip_existing` se salte las carpetas que ya existen.

### Por archivos

Para extraer el markdown de un archivo específico:

```bash
marker_single --output_dir dof_markdown/2024/04/ \
  --paginate_output \
  --languages="es" \
  dof/2024/04/01042024-MAT.pdf
# este comando tardó 2m 7s en una macbook pro M3 de 36GB RAM
# Sorprendentemente, porque otros pueden tardar más de 10 minutos.
```

## Bajar archivos Word del DOF

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
│   │   ├── 03012025/
...
```

> **NOTA**: El script descarga un archivo .doc por cada documento legal individual (no por edición completa).

### Procesar los archivos Word

Antes de convertir a Markdown, los archivos Word pueden procesarse con `dof_processor.py`, que organiza y prepara los archivos:

```bash
uv run dof_processor.py --help
```

Ver `README_DOFDOCX.md` para más detalles sobre este flujo.

## Extraer markdown desde archivos Word (.doc)

### Conversión individual (recomendado para RAG)

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

### Conversión por edición (alternativa)

El flujo alternativo usa `dof_docx_to_md.py` para convertir archivos ya procesados con `dof_processor.py`. Este método une todos los documentos de un día en un solo archivo Markdown.

Ver `README_DOFDOCX.md` para instrucciones completas.

## Extraer embeddings

Para extraer embeddings de un archivo específico:

```bash
python extract_embeddings.py dof_markdown/2024/04/
```

Puedes especificar la carpeta de un solo archivo, o la carpeta de un mes, o incluso la carpeta de un año.
En una macbook pro M3 de 36GB RAM, este comando tardó 190 minutos en extraer los embeddings de enero del 2025.
