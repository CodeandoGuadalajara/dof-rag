# DOF-RAG

**DOF-RAG** es un sistema de consulta por generación aumentada (RAG) para explorar las ediciones del Diario Oficial de la Federación de México usando tecnologías de inteligencia artificial modernas.

## Características Principales

- **Múltiples formatos**: Soporte para archivos PDF y WORD del DOF
- **Extracción completa**: Incluye documentos principales, avisos y convocatorias
- **Procesamiento unificado**: Convierte y unifica múltiples archivos DOC en documentos DOCX únicos
- **Conversión inteligente**: Usa Pandoc con filtros LUA para conversión DOCX → Markdown
- **Embeddings avanzados**: Sistema con modelo Qwen3-Embedding-0.6B y almacenamiento en DuckDB
- **Chunking semántico**: Divisiones inteligentes respetando estructura de documentos
- **Procesamiento optimizado**: Soporte para GPU (CUDA), Apple Silicon (MPS) y CPU

## Requisitos

### Dependencias Python

Instala [uv](https://docs.astral.sh/uv/) para manejar las dependencias de Python:

```bash
uv venv # Crear entorno virtual
uv sync # Sincronizar dependencias
```

### Herramientas Externas

- **LibreOffice**: Para conversión DOC → DOCX
- **Pandoc**: Para conversión DOCX → Markdown

```bash
# Ubuntu/Debian
sudo apt install libreoffice pandoc

# macOS
brew install --cask libreoffice
brew install pandoc
```

## Flujo de Trabajo Principal

El sistema DOF-RAG utiliza un **único flujo principal** optimizado para procesamiento completo de documentos del DOF:

### **Flujo Completo: WORD → DOCX → Markdown → Embeddings**

#### 1. Descarga de Archivos WORD + Avisos
```bash
# Descargar archivos WORD de una fecha específica (incluye avisos y convocatorias)
uv run get_word_dof.py 02/01/2025 --editions both

# Descargar un rango de fechas
uv run get_word_dof.py 01/01/2025 31/01/2025 --editions both --sleep-delay 1.0

# Solo edición matutina
uv run get_word_dof.py 02/01/2025 --editions mat
```

**Estructura generada:**
```
dof_word/
├── 2025/
│   ├── 01/
│   │   ├── 02012025/
│   │   │   ├── MAT/
│   │   │   │   ├── 001_DOF_20250102_MAT_12345.doc
│   │   │   │   ├── 002_AVISO_20250102_MAT_67890.doc
│   │   │   │   └── 003_DOF_20250102_MAT_11111.doc
│   │   │   └── VES/
│   │   │       └── 001_DOF_20250102_VES_22222.doc
```

#### 2. Conversión DOC → DOCX + Unificación
```bash
# Procesar una fecha específica (convierte y unifica automáticamente)
uv run dof_processor.py 02/01/2025

# Procesar un rango de fechas
uv run dof_processor.py 01/01/2025 31/01/2025 --input-dir ./dof_word

# Con logging detallado
uv run dof_processor.py 02/01/2025 --log-level DEBUG
```

**Estructura generada:**
```
dof_docx/
├── 2025/
│   ├── 01/
│   │   ├── 02012025/
│   │   │   ├── MAT/
│   │   │   │   └── 02012025_MAT.docx  # ← Archivo unificado
│   │   │   └── VES/
│   │   │       └── 02012025_VES.docx  # ← Archivo unificado
```

#### 3. Conversión DOCX → Markdown (Sin Paginación)

> **⚠️ Nota sobre Paginación y Herramientas:**
> 
> La paginación original de los documentos DOF **solo se puede preservar usando Microsoft Word**. En entornos Linux o usando LibreOffice + Pandoc, se pierde la información de páginas durante la conversión. Por esta razón, nuestros flujos de trabajo se centran en **procesamiento de contenido sin paginación**, optimizado para embeddings y búsqueda semántica.
> 
> **Comparación con herramientas anteriores:**
> - **LibreOffice + Pandoc:** No preserva saltos de página pero **mucho más rápido** que marker-pdf
> - **marker-pdf (herramienta anterior):** Tenía problemas graves con documentos DOF:
>   - Calidad deficiente en encabezados
>   - Pérdida de palabras en tablas complejas  
>   - Generación de tablas corruptas con etiquetas `<br>` no válidas
>   - Rendimiento significativamente más lento
> - **Ventaja actual:** Contenido optimizado, procesamiento más rápido y sin corrupción

```bash
# Convertir archivos DOCX específicos usando Pandoc
uv run dof_docx_to_md.py 02/01/2025

# Convertir un rango de fechas
uv run dof_docx_to_md.py 01/01/2025 31/01/2025

# Procesar todos los archivos DOCX disponibles
uv run dof_docx_to_md.py

# Con directorio personalizado
uv run dof_docx_to_md.py --input-dir ./dof_docx --output-dir ./dof_word_md_custom
```

**Estructura generada:**
```
dof_word_md/
├── 2025/
│   ├── 01/
│   │   ├── 02012025/
│   │   │   ├── MAT/
│   │   │   │   ├── 02012025_MAT.md
│   │   │   │   └── media_temp/             # ← Imágenes extraídas
│   │   │   └── VES/
│   │   │       └── 02012025_VES.md
```

#### 4. Generación de Embeddings (Optimizado para Markdown sin Paginación)
```bash
# Procesar archivos Markdown generados
uv run extract_embeddings.py dof_word_md/2025/ --verbose

# Con control de memoria
uv run extract_embeddings.py dof_word_md/2025/ --memory-cleanup-interval 25

# Procesar un mes específico
uv run extract_embeddings.py dof_word_md/2025/01/ --verbose
```

---

## Flujo Alternativo: Solo PDFs (Limitado)

**Nota**: `get_dof.py` solo descarga PDFs del DOF (edición matutina únicamente). **No hay procesamiento automático posterior** - los PDFs requieren procesamiento manual adicional para generar embeddings.

#### Descarga de PDFs
```bash
# Descargar PDFs desde 2025 hacia atrás hasta 2024 (el script cuenta hacia atrás)
uv run get_dof.py --start-year=2025 --end-year=2024

# Descargar un rango específico (desde 2025 hacia atrás hasta 2020)
uv run get_dof.py --start-year=2025 --end-year=2020
```

**Estructura generada:**
```
dof/
├── 2025/
│   ├── 01/
│   │   ├── 02012025-MAT.pdf
│   │   ├── 03012025-MAT.pdf
│   │   └── ...
```

**⚠️ Limitaciones del flujo PDF:**
- Solo edición matutina
- No incluye avisos ni convocatorias  
- Requiere procesamiento manual adicional
- No está integrado con el sistema de embeddings

---

## 📚 Cobertura Histórica del DOF

### **Períodos Disponibles:**

#### **1999 - Actualidad**: Documentos Digitales (WORD/DOC)
- **Formato**: Archivos DOC descargables
- **Contenido**: Documentos principales + avisos + convocatorias
- **Ediciones**: Matutina y Vespertina
- **Procesamiento**: Flujo principal optimizado (DOC → DOCX → MD → Embeddings)

#### **1920 - 1999**: Documentos Escaneados (PDF únicamente)
- **Formato**: Solo PDFs (documentos escaneados)
- **Contenido**: Documentos principales escaneados
- **Procesamiento**: Requiere herramientas de OCR adicionales
- **Estado**: Disponible para descarga, requiere procesamiento especializado

### **Próximos Desarrollos:**
- Sistema de consultas y búsqueda semántica
- Procesamiento OCR para documentos históricos (1920-1999)
- Interfaz de usuario para consultas
- API de búsqueda y recuperación

## Uso del Sistema

### Sistema de Embeddings

Una vez procesados los archivos Markdown, el sistema genera embeddings y los almacena en una base de datos DuckDB.

**Base de Datos:** `dof_db/db.duckdb`

### ¿Cómo Verificar la Base de Datos?

Puedes inspeccionar la base de datos generada para verificar que los documentos y chunks se han guardado correctamente. Para ello, puedes usar el cliente de línea de comandos de DuckDB.

1.  **Instala DuckDB** (si no lo tienes, aunque debería estar incluido en las dependencias del proyecto):
    ```bash
    pip install duckdb
    ```

2.  **Abre la base de datos**:
    ```bash
    duckdb dof_db/db.duckdb
    ```

3.  **Ejecuta consultas SQL para explorar los datos**:

    *   **Contar el número total de documentos procesados**:
        ```sql
        SELECT COUNT(*) FROM documents;
        ```

    *   **Ver los 5 documentos más recientes**:
        ```sql
        SELECT * FROM documents ORDER BY created_at DESC LIMIT 5;
        ```

    *   **Contar el número total de chunks generados**:
        ```sql
        SELECT COUNT(*) FROM chunks;
        ```

    *   **Ver un chunk específico y su texto asociado**:
        ```sql
        SELECT id, document_id, header, chunk_number, text FROM chunks LIMIT 1;
        ```

    *   **Encontrar todos los chunks de un documento específico (ej. ID=10)**:
        ```sql
        SELECT chunk_number, header, text FROM chunks WHERE document_id = 10 ORDER BY chunk_number;
        ```

**Nota**: El sistema de consultas semánticas está en desarrollo. Los embeddings se generan y almacenan correctamente en la base de datos DuckDB para su uso posterior en futuras funcionalidades.

## Estructura del Proyecto

```
dof-rag/
├── Scripts principales
│   ├── get_dof.py              # Descarga PDFs del DOF (sin procesamiento posterior)
│   ├── get_word_dof.py         # Descarga archivos WORD + avisos/convocatorias
│   ├── dof_processor.py        # Convierte DOC → DOCX + unifica
│   ├── dof_docx_to_md.py       # Convierte DOCX → Markdown (Pandoc + filtros LUA)
│   └── extract_embeddings.py   # Sistema de embeddings (procesa Markdown sin paginación)
├── Datos - Flujo Principal (WORD)
│   ├── dof_word/              # Archivos DOC + avisos descargados
│   ├── dof_docx/              # Archivos DOCX unificados  
│   ├── dof_word_md/           # Markdown optimizado (sin paginación)
│   └── dof_db/                # Base de datos DuckDB con embeddings
├── Datos - Alternativo (PDF)
│   └── dof/                   # PDFs descargados (requiere procesamiento manual)
├── Herramientas
│   ├── pandoc_filters/         # Filtros LUA para Pandoc
│   │   └── dof_headers.lua    # Filtro para headers del DOF
│   └── modules_captions/       # Módulos de extracción de metadatos
└── Configuración
    └── pyproject.toml          # Dependencias del proyecto
```

## Características Técnicas

### Limitaciones de Paginación
> **Importante:** El sistema está diseñado para **contenido continuo sin paginación** debido a limitaciones técnicas de las herramientas open-source:
> 
> - **Microsoft Word**: Única herramienta que preserva paginación original
> - **LibreOffice + Pandoc**: Pierden información de saltos de página
> - **Enfoque del sistema**: Optimizado para búsqueda semántica y embeddings
> - **Ventaja**: Chunking más efectivo sin interrupciones artificiales de página

### Decisiones Arquitectónicas: Migración de marker-pdf
> **Cambio tecnológico documentado:** Se migró de `marker-pdf` al flujo actual por problemas críticos de calidad:
> 
> **Problemas con marker-pdf:**
> - ❌ Encabezados mal procesados o perdidos
> - ❌ Pérdida de palabras en tablas complejas
> - ❌ Tablas corruptas con etiquetas `<br>` inválidas
> - ❌ Rendimiento lento para documentos extensos
> 
> **Ventajas del flujo actual (LibreOffice + Pandoc):**
> - ✅ **Velocidad significativamente superior**
> - ✅ Preservación completa de encabezados
> - ✅ Tablas bien formateadas sin corrupción
> - ✅ Contenido limpio optimizado para embeddings

### Modelo de Embeddings: Qwen3-Embedding-0.6B

Este proyecto utiliza [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), un modelo de embeddings de texto optimizado para tareas de Recuperación de Información (RAG). A continuación se detallan sus características principales:

-   **Tamaño y Arquitectura**: Es un modelo basado en Transformers con **0.6 mil millones de parámetros**, lo que ofrece un excelente equilibrio entre rendimiento y eficiencia computacional.
-   **Longitud de Secuencia**: Soporta una longitud máxima de contexto de **32k tokens**, permitiendo procesar documentos extensos sin necesidad de truncarlos excesivamente.
-   **Dimensiones del Embedding**: Genera embeddings con una dimensión de **1024**, capturando una gran riqueza semántica.
-   **Modelo Matrioska (Matryoshka Representation)**: Este modelo implementa una técnica que permite que los embeddings generados sean efectivos incluso si se truncan a dimensiones más pequeñas (ej. 512, 256). Esto ofrece flexibilidad para adaptar el tamaño del embedding a los requisitos de almacenamiento o rendimiento sin necesidad de reentrenar.
-   **Last Token Pooling**: En lugar de promediar todos los tokens de la secuencia (mean pooling), el modelo utiliza la representación del último token como el embedding final para todo el texto. Esta estrategia está alineada con su entrenamiento y ha demostrado ser altamente efectiva.
-   **Uso de Instrucciones**: Para mejorar la relevancia en tareas de búsqueda, el modelo utiliza prefijos específicos (instrucciones) para diferenciar entre la codificación de pasajes (documentos) y la codificación de consultas (preguntas). Por ejemplo:
    -   **Para documentos (instruct)**: Se añade un prefijo que indica al modelo que genere una representación para ser almacenada y encontrada.
    -   **Para búsquedas (question)**: Se utiliza un prefijo diferente para indicar que el texto es una consulta, optimizando el embedding para la tarea de búsqueda.

    **Ejemplo de formato en código:**
    ```python
      # 1. Definir la tarea de recuperación
      task_description = "Retrieve relevant legal document fragments including text, image descriptions, and table content that match the query"

      # 2. Formatear la consulta con la instrucción
      user_query = "..." # Pregunta del usuario
      instructed_query = f'Instruct: {task_description}\nQuery: {user_query}'

      # 3. Generar el embedding (ejemplo conceptual de la implementación interna)
      with inference_mode():
          embedding = embedding_model.encode([instructed_query], show_progress_bar=False)
    ```

### Modelos y Tecnologías
- **Pandoc + Filtros LUA**: Conversión DOCX → Markdown sin paginación
- **LibreOffice**: Conversión DOC → DOCX en modo headless
- **Qwen3-Embedding-0.6B**: Generación de embeddings (1024 dimensiones)
- **DuckDB**: Almacenamiento de embeddings con FLOAT[] arrays
- **MarkdownSplitter**: Chunking semántico optimizado para Markdown unificado

### Optimizaciones
- **Soporte multi-plataforma**: CUDA, MPS (Apple Silicon), CPU
- **Gestión de memoria**: Limpieza automática cada N chunks
- **Chunking inteligente**: Preserva jerarquía de headers
- **Timeouts configurables**: Manejo robusto de archivos problemáticos (90s LibreOffice)
- **Unificación automática**: Múltiples DOCs → 1 DOCX por edición/fecha
- **Extracción de medios**: Imágenes y tablas preservadas en conversión

### Base de Datos
```sql
-- Estructura de tablas en DuckDB
documents (id, title, url, file_path, created_at)
chunks (id, document_id, text, header, chunk_number, embedding[1024], created_at)
```

## 🚨 Notas Importantes

### Dependencias Críticas
- **LibreOffice**: Necesario para conversión DOC → DOCX 
  ```bash
  # Ubuntu/Debian
  sudo apt install libreoffice
  
  # macOS  
  brew install --cask libreoffice
  ```
- **Pandoc**: Necesario para conversión DOCX → Markdown
  ```bash
  # Ubuntu/Debian
  sudo apt install pandoc
  
  # macOS
  brew install pandoc
  ```

### Consideraciones de Rendimiento
- **Memoria**: Los embeddings pueden requerir significativa RAM para datasets grandes
- **Timeouts**: LibreOffice tiene timeout de 90s por archivo DOC
- **Archivos problemáticos**: Se generan reportes automáticos de archivos que fallan por timeout
- **Limpieza automática**: Se eliminan archivos temporales tras unificación

## Logs y Debugging

El sistema genera logs detallados:
- `dof_processing.log`: Logs del sistema principal de embeddings y extracción
- `dof_processor.log`: Logs de conversión DOC/DOCX  
- `convert_docx_to_md.log`: Logs de conversión DOCX/Markdown
- `word_download.log`: Logs de descarga de archivos WORD
- `archivos_problematicos_*.txt`: Reportes de archivos con timeout

Para debugging detallado, usa el flag `--verbose` o `--log-level DEBUG` en los scripts compatibles.

## 🔄 Arquitectura del Sistema

### **Flujo Principal Optimizado** 
El sistema está diseñado específicamente para el procesamiento completo de documentos del DOF:

```
DOC + Avisos → DOCX Unificado → Markdown Sin Paginación → Embeddings Optimizados
```

### **Características Clave:**
- **Unificación**: Múltiples archivos DOC se consolidan en un DOCX por fecha/edición
- **Sin Paginación**: El Markdown generado no tiene separaciones de página (optimizado para embeddings)
- **Estructura Semántica**: Preserva jerarquía de headers y estructura de documentos
- **Contenido Completo**: Incluye documentos principales + avisos + convocatorias

### **Flujo Alternativo (Limitado):**
- `get_dof.py` descarga PDFs pero no está integrado con el sistema de embeddings
- Requiere procesamiento manual adicional para generar embeddings

## 📈 Rendimiento y Características

### Tiempos de Procesamiento

Los tiempos de procesamiento varían según el hardware y tamaño de los archivos. Como referencia general:

#### Flujo Principal (WORD → DOCX → MD → Embeddings):
- **Descarga**: Depende de la conexión de red y cantidad de archivos
- **DOC → DOCX**: Variable según tamaño del archivo (timeout configurado a 90s)
- **Unificación**: Rápida, consolida múltiples archivos en uno
- **DOCX → Markdown**: Variable según complejidad del documento
- **Embeddings**: Depende del hardware (GPU/CPU) y longitud del texto

### Consideraciones de Hardware
- **Mínimo**: 8GB RAM, LibreOffice, Pandoc
- **Recomendado**: 16GB+ RAM, GPU compatible con CUDA/MPS
- **Óptimo**: 32GB+ RAM, GPU dedicada, múltiples núcleos CPU

## Casos de Uso y Recomendaciones

### **Para Documentos Actuales (1999-2025)**

#### Usa el **Flujo Principal** (WORD) para:
- ✅ Análisis completo del DOF (documentos + avisos + convocatorias)
- ✅ Ambas ediciones (matutina y vespertina)
- ✅ Sistema de embeddings optimizado
- ✅ Preservar formato e imágenes originales
- ✅ Máxima completitud de datos

### **Para Documentos Históricos (1920-1999)**

#### Usa `get_dof.py` (PDFs) para:
- Descargar PDFs escaneados para archivo histórico
- Procesamiento con herramientas OCR externas
- Análisis de documentos históricos

**Nota**: Los documentos históricos requieren procesamiento especializado con herramientas OCR para extraer texto.

## Guía de Inicio Rápido

### **Documentos Actuales (1999-2025)** - Procesamiento Completo
```bash
# 1. Instalar dependencias
sudo apt install libreoffice pandoc  # Linux
brew install --cask libreoffice && brew install pandoc  # macOS

# 2. Descargar archivos WORD completos
uv run get_word_dof.py 01/01/2025 31/01/2025 --editions both

# 3. Convertir y unificar DOC → DOCX
uv run dof_processor.py 01/01/2025 31/01/2025

# 4. Convertir DOCX → Markdown (sin paginación)
uv run dof_docx_to_md.py 01/01/2025 31/01/2025

# 5. Generar embeddings
uv run extract_embeddings.py dof_word_md/2025/ --verbose
```

### **Documentos Históricos (1920-1999)** - Solo Descarga
```bash
# Descargar PDFs escaneados para archivo o procesamiento especializado
uv run get_dof.py --start-year=1995 --end-year=1990
```

**Nota**: Los documentos históricos son PDFs escaneados que requieren herramientas OCR adicionales para procesamiento de texto.