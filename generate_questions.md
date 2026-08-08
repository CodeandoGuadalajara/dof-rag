# Documentación: generate_questions.py

## Descripción General
Este script genera preguntas relacionadas con documentos del Diario Oficial de la Federación (DOF) utilizando el modelo de lenguaje Gemini de Google. Selecciona aleatoriamente un archivo markdown con contenido del DOF, genera preguntas basadas en su contenido y las guarda en un archivo CSV.

## Dependencias
- `os`: Para interacción con el sistema operativo
- `random`: Para selección aleatoria de archivos
- `typer`: Para crear la interfaz de línea de comandos
- `getpass`: Para entrada segura de API keys
- `pathlib`: Para manipulación de rutas de archivos
- `datetime`: Para marcas de tiempo en las preguntas generadas
- `dotenv`: Para cargar variables de entorno
- `google.generativeai`: Cliente de la API de Google AI (Gemini)
- `csv`: Para manipulación de archivos CSV
- `io`: Para operaciones de entrada/salida en memoria

## Configuración
- Usa un archivo `.env` para cargar la clave API de Google
- Configura el cliente de Gemini con la clave API

## Funcionalidades

### Resolución de directorios
La función `resolve_base_dir()` intenta encontrar el directorio especificado:
- Primero busca en el directorio de trabajo actual
- Si no lo encuentra, busca relativo a la ubicación del script
- Termina la ejecución si el directorio no existe

### Búsqueda de archivos markdown
La función `find_md_files()`:
- Recorre recursivamente el directorio especificado
- Busca archivos `.md` cuyo nombre sigue el formato DDMMYYYY-*
- Filtra solo aquellos archivos que corresponden a DOF 2024 o 2025

### Procesamiento de respuestas de Gemini
La función `parse_csv_from_gemini()`:
- Recibe el texto generado por Gemini
- Elimina líneas con marcadores de código (```)
- Parsea el contenido como CSV
- Devuelve una lista de filas

### Generación de preguntas
La función `generate_questions()`:
- Lee el contenido del archivo markdown
- Construye un prompt para Gemini solicitando preguntas en formato CSV
- Utiliza el modelo "gemini-2.0-flash"
- Procesa y limita el número de preguntas generadas

### Comando principal
El comando `main()`:
1. Resuelve el directorio base
2. Encuentra archivos markdown del DOF 2024/2025
3. Selecciona un archivo aleatorio
4. Genera preguntas basadas en ese archivo
5. Escribe las preguntas en un archivo CSV, añadiendo una marca de tiempo

## Parámetros
- `directory` (obligatorio): Ruta al directorio que contiene los archivos markdown del DOF
- `output_csv` (opcional): Ruta del archivo CSV de salida (predeterminado: "questions.csv")
- `num_questions` (opcional): Número de preguntas a generar (predeterminado: 5)

## Formato de salida
El archivo CSV generado contiene las siguientes columnas:
- `RunTimestamp`: Marca de tiempo ISO 8601 de la ejecución
- `Question`: Pregunta generada
- `File`: Nombre del archivo fuente
- `Page`: Número de página (si disponible)
- `Extract`: Extracto de texto relevante para la pregunta

## Notas de implementación
El script está diseñado para funcionar en modo de "agregación", añadiendo nuevas preguntas al archivo CSV existente sin duplicar los encabezados. Cada conjunto de preguntas se identifica con una marca de tiempo única (RunID). El modelo Gemini está configurado para generar preguntas simples basadas en el contenido del documento, junto con la información contextual necesaria para validar las respuestas.

## Uso desde línea de comandos
```bash
python generate_questions.py dof_markdown --output preguntas.csv --num 10
```
Esto generará 10 preguntas basadas en un archivo aleatorio de la carpeta dof_markdown y las guardará en preguntas.csv.