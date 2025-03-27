# Sistema de Procesamiento y Consulta de Documentos del DOF

## Índice
- [Introducción](#introducción)
- [Características principales](#características-principales)
- [Requisitos y dependencias](#requisitos-y-dependencias)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Procesamiento de documentos](#procesamiento-de-documentos)
  - [Consulta de documentos](#consulta-de-documentos)
- [Estructura del código](#estructura-del-código)
  - [Configuración inicial](#configuración-inicial)
  - [Estructura de la base de datos](#estructura-de-la-base-de-datos)
  - [Procesamiento de texto](#procesamiento-de-texto)
  - [Generación de embeddings](#generación-de-embeddings)
  - [Almacenamiento en base de datos](#almacenamiento-en-base-de-datos)
- [Funcionamiento técnico](#funcionamiento-técnico)
  - [Detección de encabezados](#detección-de-encabezados)
  - [División de texto por páginas](#división-de-texto-por-páginas)
  - [Construcción de encabezados contextuales](#construcción-de-encabezados-contextuales)
- [Sistema de consulta con Gemini](#sistema-de-consulta-con-gemini)
  - [Búsqueda vectorial](#búsqueda-vectorial)
  - [Contexto expandido](#contexto-expandido)
  - [Consulta a Gemini 2.0](#consulta-a-gemini-20)
  - [Modos de operación](#modos-de-operación)
- [Flujo de trabajo](#flujo-de-trabajo)
  - [Procesamiento de documentos](#procesamiento-de-documentos-1)
  - [Consulta de información](#consulta-de-información)
- [Limitaciones y consideraciones](#limitaciones-y-consideraciones)

## Introducción

Este sistema procesa documentos del Diario Oficial de la Federación (DOF) almacenados en formato Markdown. Su objetivo principal es extraer el contenido, detectar la estructura jerárquica de los documentos, dividir el texto en fragmentos basados en páginas y generar embeddings vectoriales para permitir búsquedas semánticas eficientes.

Los documentos procesados se almacenan en una base de datos SQLite optimizada para búsquedas vectoriales, permitiendo consultas por similitud semántica en el contenido del DOF.

El sistema también incluye un módulo de consulta que permite realizar búsquedas semánticas y obtener respuestas contextualizadas utilizando el modelo Gemini 2.0 de Google.

## Características principales

- Procesamiento de archivos Markdown del DOF
- Extracción inteligente de la estructura jerárquica mediante detección de encabezados
- División de texto en chunks basados en marcadores de página
- Generación de embeddings vectoriales usando SentenceTransformer
- Almacenamiento en base de datos SQLite con capacidades vectoriales mediante sqlite-vec
- Creación de encabezados contextuales para mejorar la búsqueda semántica
- Generación de archivos de depuración para verificar la segmentación del texto
- Búsqueda semántica mediante distancia euclidiana entre embeddings
- Obtención de contexto expandido (chunks adyacentes) para mejorar respuestas
- Consultas a Gemini 2.0 con instrucciones específicas para procesar documentos del DOF
- Modo interactivo para realizar consultas continuas
- Sistema de logging para seguimiento del procesamiento

## Requisitos y dependencias

El sistema depende de las siguientes bibliotecas y tecnologías:

- typer: para la interfaz de línea de comandos
- fastlite: para manipulación simplificada de bases de datos SQLite
- sqlite-vec: para habilitar capacidades vectoriales en SQLite
- sentence-transformers: para la generación de embeddings
- tokenizers: para tokenizar el texto
- tqdm: para mostrar barras de progreso
- google.generativeai: para integración con el modelo Gemini 2.0
- numpy: para procesamiento numérico
- python-dotenv: para cargar variables de entorno

## Instalación

1. Clona el repositorio o descarga los scripts
2. Instala las dependencias necesarias usando uv (instalador de paquetes más rápido y moderno):

```bash
# Instalar uv si no lo tienes
curl -sSf https://astral.sh/uv/install.sh | sh

# Instalar dependencias con uv
uv pip install typer fastlite sqlite-vec sentence-transformers tokenizers tqdm google-generativeai numpy python-dotenv
```

3. Configura una API key de Google Gemini en un archivo .env:

```
GOOGLE_API_KEY=tu_api_key_aquí
```

4. Asegúrate de tener una estructura de directorios adecuada para almacenar los archivos del DOF en formato Markdown

## Uso

### Procesamiento de documentos

Para procesar todos los archivos Markdown en un directorio específico:

```bash
python extract_embeddings_1_page.py /ruta/a/directorio/con/archivos/md
```

El script realizará las siguientes acciones:
1. Procesará cada archivo Markdown en el directorio
2. Extraerá la estructura jerárquica de los documentos
3. Dividirá el texto en chunks basados en marcadores de página
4. Generará embeddings para cada chunk
5. Almacenará todo en la base de datos SQLite
6. Creará archivos de depuración con el nombre "nombre_archivo_chunks.txt" que muestran cómo se dividió el texto

## Estructura del código

### Configuración inicial

El script comienza inicializando los modelos y configuraciones necesarias:

- Carga del tokenizador y modelo de embeddings (nomic-ai/modernbert-embed-base)
- Configuración del sistema de logging para depuración y seguimiento
- Inicialización de la base de datos SQLite con extensiones vectoriales

### Estructura de la base de datos

La base de datos consta de dos tablas principales:

1. **documents**: Almacena metadatos de los documentos
   - id: Identificador único
   - title: Título del documento
   - url: URL al documento original en el DOF
   - file_path: Ruta al archivo local
   - created_at: Fecha y hora de procesamiento

2. **chunks**: Almacena los fragmentos de texto y sus embeddings
   - id: Identificador único
   - document_id: Relación con el documento original
   - text: Texto del fragmento
   - header: Encabezado contextual para mejorar la búsqueda
   - embedding: Vector de embeddings serializado
   - created_at: Fecha y hora de procesamiento

### Procesamiento de texto

El código implementa varias funciones clave para procesar los documentos:

- `split_text_by_page_break`: Divide el texto en secciones basadas en marcadores de página
- `extract_headings`: Extrae los encabezados de Markdown presentes en el texto
- `build_chunk_header`: Construye encabezados contextuales para cada chunk
- `get_url_from_filename`: Genera la URL del DOF a partir del nombre de archivo

### Generación de embeddings

Los embeddings se generan utilizando el modelo SentenceTransformer (nomic-ai/modernbert-embed-base). Para cada chunk, se construye un texto compuesto por el encabezado contextual y el contenido, y se codifica usando el modelo.

### Almacenamiento en base de datos

Los documentos y chunks procesados se almacenan en la base de datos SQLite, que ha sido extendida con capacidades vectoriales mediante sqlite-vec. Esto permite realizar búsquedas semánticas eficientes.

## Funcionamiento técnico

### Detección de encabezados

El sistema utiliza expresiones regulares para detectar encabezados en formato Markdown:
```python
HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$')
```

Esta expresión identifica líneas que comienzan con entre 1 y 6 símbolos '#', seguidos de un espacio y texto, capturando tanto el nivel de encabezado como su contenido.

### División de texto por páginas

El algoritmo `split_text_by_page_break` es central en el sistema. Divide el texto en fragmentos basados en marcadores de página con el siguiente formato:

```
{número_página}-----------------------------------------------------
```

El sistema detecta estos marcadores y crea chunks separados para cada página, conservando el número de página para referencia. Esto permite procesar documentos del DOF manteniendo la estructura original de páginas, lo que facilita la referencia y citación.

### Construcción de encabezados contextuales

Para cada chunk, se construye un encabezado contextual que incluye:
- El título del documento
- Los encabezados encontrados en el chunk actual
- El número de página

Ejemplo: "Document: 23012025-MAT | Section: Título Principal > Subtítulo | Page: 3"

Este encabezado ayuda a que los embeddings capten el contexto jerárquico y la ubicación en el documento, mejorando la calidad de las búsquedas.

## Sistema de consulta con Gemini

El módulo `gemini_query.py` implementa un sistema avanzado de búsqueda semántica y consulta mediante Gemini 2.0 para documentos del Diario Oficial de la Federación (DOF).

### Búsqueda vectorial

El sistema utiliza embeddings vectoriales y la distancia euclidiana para encontrar los fragmentos más relevantes para una consulta:

```python
def find_relevant_chunks(query, all_chunks, top_k=10):
    # Generar embedding para la consulta
    query_embedding = model.encode(f"search_query: {query}")
    
    # Calcular distancias
    scored_results = []
    for chunk in all_chunks:
        chunk_embedding = deserialize_embedding(chunk["embedding_blob"])
        distance = np.linalg.norm(query_embedding - chunk_embedding)
        scored_results.append({
            # ... metadatos del chunk ...
            "distance": distance,
            "similarity": 1.0 / (1.0 + distance)
        })
    
    # Ordenar por menor distancia (mayor similitud)
    return sorted(scored_results, key=lambda x: x["distance"])[:top_k]
```

Esta función:
1. Convierte la consulta de texto a un vector de embeddings usando el mismo modelo que en el procesamiento
2. Calcula la distancia euclidiana entre la consulta y cada chunk de la base de datos
3. Ordena los resultados por relevancia (menor distancia)
4. Devuelve los `top_k` resultados más relevantes

### Contexto expandido

Para mejorar la calidad de las respuestas, el sistema obtiene un contexto expandido incluyendo chunks adyacentes al más relevante:

```python
def get_extended_context(best_chunk, all_chunks):
    # Filtrar chunks del mismo documento
    same_doc_chunks = [c for c in all_chunks if c["document_id"] == best_chunk["document_id"]]
    
    # ... encontrar posición del mejor chunk ...
    
    # Construir contexto extendido con chunk anterior + mejor chunk + chunk siguiente
    context = ""
    if best_index > 0:
        context += same_doc_chunks[best_index - 1]["chunk_text"] + "\n\n"
    context += best_chunk["chunk_text"] + "\n\n"
    if best_index < len(same_doc_chunks) - 1:
        context += same_doc_chunks[best_index + 1]["chunk_text"]
```

Esta estrategia:
1. Identifica todos los chunks del mismo documento que el chunk más relevante
2. Determina la posición del chunk más relevante en la secuencia original
3. Construye un contexto que incluye el chunk anterior, el chunk más relevante y el chunk siguiente

### Consulta a Gemini 2.0

El sistema utiliza el modelo Gemini 2.0 de Google para generar respuestas contextualizadas:

```python
# Construir prompt para Gemini
prompt = (
    f"Utilizando el siguiente contexto extraído de un documento del Diario Oficial de la Federación:\n\n"
    f"---\n{extended_context}\n---\n\n"
    f"Instrucciones:\n"
    # ... instrucciones específicas ...
    f"Pregunta:\n{query}"
)

# Generar respuesta con Gemini
model_gemini = genai.GenerativeModel('gemini-2.0-flash')
response = model_gemini.generate_content(prompt)
```

El prompt incluye:
1. El contexto expandido extraído de los documentos
2. Instrucciones específicas para el modelo sobre cómo debe responder
3. La consulta original del usuario

### Modos de operación

El sistema ofrece dos modos de operación:

1. **Consulta directa**: Realiza una única consulta y muestra el resultado
   ```bash
   python gemini_query.py --query "tu consulta aquí"
   ```

2. **Modo interactivo**: Permite realizar múltiples consultas en una sesión
   ```bash
   python gemini_query.py --interactive
   ```

En el modo interactivo, el sistema mostrará un prompt para ingresar consultas y continuará hasta que el usuario escriba 'salir', 'exit' o 'quit'.

## Flujo de trabajo

### Procesamiento de documentos

1. Por cada archivo Markdown en el directorio especificado:
   - Se extraen los metadatos del documento
   - Se elimina cualquier versión anterior del documento en la base de datos
   - Se inserta el nuevo documento en la tabla 'documents'
   - Se verifica si el documento contiene marcadores de página
   - Si encuentra marcadores, divide el texto por páginas; sino, trata todo el contenido como una sola página
   - Por cada página:
     - Se extraen los encabezados presentes
     - Se construye el encabezado contextual
     - Se genera el embedding combinando el encabezado y el contenido
     - Se guarda en la base de datos y en un archivo de depuración

### Consulta de información

1. El usuario realiza una consulta en lenguaje natural
2. El sistema obtiene todos los chunks almacenados en la base de datos
3. La consulta se convierte en un vector de embeddings
4. Se calcula la similitud entre la consulta y todos los chunks
5. Se selecciona el chunk más relevante y se obtiene contexto expandido (chunks adyacentes)
6. Se construye un prompt para Gemini 2.0 con el contexto y la consulta
7. Gemini 2.0 genera una respuesta basada en el contexto
8. Se muestra la respuesta junto con metadatos del documento fuente

## Limitaciones y consideraciones

- El sistema depende de la correcta identificación de marcadores de página en formato {número}-----
- Si no se encuentran marcadores de página, todo el documento se procesa como una sola página
- El nombre de archivo debe seguir un formato específico (DDMMYYYY-XXX.md) para generar correctamente las URLs
- La calidad de las respuestas depende tanto de la precisión de la búsqueda vectorial como del modelo Gemini
- Requiere una API key válida de Google para utilizar Gemini 2.0
- La visualización de resultados está optimizada para terminal con formato de texto
- El rendimiento puede verse afectado cuando la base de datos contiene muchos documentos
- El sistema de logging facilita la depuración y seguimiento del procesamiento