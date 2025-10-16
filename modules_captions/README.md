# Módulo de Extracción de Descripciones de Imágenes

Sistema modular para generar descripciones automáticas de imágenes utilizando IA, con almacenamiento en SQLite, soporte para múltiples proveedores y priorización inteligente de errores.

## Características

- **Base de datos SQLite**: Almacenamiento transaccional con operaciones CRUD
- **Múltiples proveedores de IA**: OpenAI, Gemini, Claude, Ollama, Azure OpenAI
- **Arquitectura modular**: Componentes especializados y reutilizables con gestión centralizada
- **Manejo de errores**: Sistema centralizado con logging estructurado y recuperación automática
- **Gestión centralizada de errores**: ErrorLogManager unificado para manejo de error_images.json
- **Priorización de errores**: Procesamiento automático prioritario de imágenes con errores previos
- **Procesamiento por lotes**: Optimización de rendimiento con checkpoints y cooldown inteligente
- **Configuración flexible**: CLI, archivos JSON y variables de entorno
- **Gestión centralizada de colores**: Sistema unificado de colores con fallback automático
- **Gestión de rutas**: Normalización automática de rutas multiplataforma
- **Logging modular**: Sistema de logging especializado por componente

## Instalación

```bash
pip install openai pillow python-dotenv colorama tqdm
```

## Configuración

### Variables de Entorno

```bash
export OPENAI_API_KEY="tu_clave_api"        # OpenAI
export GEMINI_API_KEY="tu_clave_api"        # Google Gemini  
export ANTHROPIC_API_KEY="tu_clave_api"     # Anthropic Claude
export AZURE_OPENAI_API_KEY="tu_clave_api"  # Azure OpenAI
```

### Estructura del Proyecto

```
modules_captions/
├── extract_captions.py      # Script principal
├── config.json              # Configuración de proveedores
├── clients/                 # Clientes de IA
│   ├── __init__.py         # Factory de clientes
│   └── openai.py           # Cliente OpenAI con rate limiting
├── db/                      # Gestión de base de datos
│   ├── __init__.py
│   └── manager.py          # DatabaseManager con transacciones
├── utils/                   # Utilidades principales
│   ├── __init__.py
│   ├── colors.py           # ColorManager centralizado
│   ├── error_handler.py    # ErrorHandler con recuperación
│   ├── error_log_manager.py # ErrorLogManager centralizado
│   ├── file_processor.py   # FileProcessor con priorización
│   ├── logging_config.py   # Configuración de logging
│   ├── path_utils.py       # PathManager multiplataforma
│   └── prioritize_error_images.py  # Gestión de prioridades
└── logs/                    # Archivos de log
```

## Proveedores Soportados

| Proveedor | Modelos Principales | Variable de Entorno |
|-----------|-------------------|--------------------|
| **OpenAI** | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |
| **Gemini** | gemini-2.0-flash, gemini-1.5-pro | `GEMINI_API_KEY` |
| **Claude** | claude-sonnet-4, claude-3-5-sonnet | `ANTHROPIC_API_KEY` |
| **Ollama** | llama3.2-vision (local) | No requerida |
| **Azure OpenAI** | gpt-4-vision | `AZURE_OPENAI_API_KEY` |

## Uso

### Comandos Básicos

```bash
# Usar proveedor por defecto (Gemini)
python extract_captions.py --root-dir ./imagenes

# Especificar proveedor
python extract_captions.py --root-dir ./imagenes --openai
python extract_captions.py --root-dir ./imagenes --gemini
python extract_captions.py --root-dir ./imagenes --claude

# Forzar reprocesamiento de imágenes existentes
python extract_captions.py --root-dir ./imagenes --force-reprocess

# Ver estado del sistema
python extract_captions.py --status
```

### Parámetros Disponibles

| Parámetro | Descripción | Ejemplo |
|-----------|-------------|----------|
| `--root-dir` | Directorio con imágenes | `--root-dir ./imagenes` |
| `--db-path` | Ruta de base de datos | `--db-path ../dof_db/db.sqlite` |
| `--openai/--gemini/--claude` | Seleccionar proveedor | `--gemini` |
| `--force-reprocess` | Reprocesar imágenes existentes | `--force-reprocess` |
| `--status` | Mostrar estado del sistema | `--status` |
| `--log-level` | Nivel de logging | `--log-level DEBUG` |
| `--debug` | Modo depuración detallado | `--debug` |

### Resolución de Rutas

El sistema maneja automáticamente la resolución de rutas para garantizar compatibilidad:

- **Rutas relativas con `../`**: Se resuelven relativas al directorio del proyecto
- **Nombres de archivo**: Se ubican en `modules_captions/db/`
- **Rutas absolutas**: Se usan tal como se especifican
- **Compatibilidad**: Funciona ejecutando desde cualquier ubicación del proyecto

### Configuración Personalizada

El archivo `config.json` permite personalizar el comportamiento del sistema:

```json
{
  "provider": "gemini",
  "root_directory": "./imagenes",
  "db_path": "../dof_db/db.sqlite",
  "commit_interval": 10,
  "log_level": 20,
  "providers": {
    "gemini": {
      "client_config": {
        "model": "gemini-2.0-flash",
        "max_tokens": 256,
        "temperature": 0.6
      }
    }
  }
}
```

**Parámetros principales:**
- `provider`: Proveedor de IA por defecto
- `commit_interval`: Número de imágenes procesadas antes de guardar en BD
- `log_level`: Nivel de logging (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR)
- `prompt`: Prompt del sistema para generar descripciones en español

## Uso Programático

### Procesamiento de Imágenes

```python
from modules_captions import DatabaseManager, create_client, FileProcessor
from modules_captions.utils import ColorManager, PathManager

# Configurar componentes con gestión centralizada
db_manager = DatabaseManager("../dof_db/db.sqlite")
client = create_client("gemini")
processor = FileProcessor(
    root_directory="./imagenes",
    db_manager=db_manager,
    ai_client=client,
    debug_mode=True  # Habilita logging detallado
)

# Procesar imágenes (incluye priorización automática de errores)
results = processor.process_images()
print(f"Procesadas: {results['total_processed']} imágenes")

# Usar gestión centralizada de colores
print(ColorManager.colorize("Procesamiento completado", "green"))
```

### Gestión de Priorización de Errores

```python
from modules_captions import ImagePriorityManager
from modules_captions.utils import ErrorHandler, ColorManager

# Inicializar gestor de prioridades
priority_manager = ImagePriorityManager("./modules_captions")

# Cargar lista de errores para priorización
if priority_manager.load_priority_list():
    print(ColorManager.colorize(f"Cargadas {len(priority_manager.priority_images)} imágenes con errores", "yellow"))
    
    # Crear lista priorizada de imágenes
    images_to_process = ["imagen1.jpg", "imagen2.png"]
    prioritized_list = priority_manager.create_prioritized_image_list(images_to_process)
    print(ColorManager.colorize(f"Lista priorizada: {len(prioritized_list)} imágenes", "green"))

# Gestión avanzada de errores
error_handler = ErrorHandler(log_dir="logs", debug_mode=True)
error_handler.log_error("test_error", "Error de prueba", {"context": "ejemplo"})
stats = error_handler.get_error_statistics()
print(f"Errores registrados: {stats['total_errors']}")
```

### Operaciones de Base de Datos

```python
from modules_captions.db import DatabaseManager
from modules_captions.utils import PathManager

# Usar PathManager para normalización de rutas
db_path = PathManager.normalize_path("../dof_db/db.sqlite")
db = DatabaseManager(str(db_path))

# Insertar descripción con manejo de transacciones
descriptions = [
    ("documento_001", 1, "imagen_001.png", "Descripción de la imagen"),
    ("documento_002", 1, "imagen_002.png", "Segunda descripción")
]
inserted_count = db.insert_descriptions_batch(descriptions)
print(f"Insertadas {inserted_count} descripciones")

# Consultar descripción existente
if db.description_exists("documento_001", 1, "imagen_001.png"):
    descripcion = db.get_description("documento_001", 1, "imagen_001.png")
    print(descripcion)

# Obtener estadísticas detalladas
stats = db.get_statistics()
print(f"Total descripciones: {stats['total_descriptions']}")
print(f"Documentos únicos: {stats['unique_documents']}")
print(f"Tamaño de BD: {stats['database_size_mb']:.2f} MB")
```

## Integración con Base de Datos de Embeddings

El sistema ahora utiliza directamente la base de datos de embeddings (`dof_db/db.sqlite`) para almacenar las descripciones de imágenes. Esto elimina la necesidad de migraciones y mantiene todos los datos centralizados.

### Ventajas de la Integración

- **Sin migraciones**: Las descripciones se almacenan directamente donde se necesitan
- **Datos centralizados**: Una sola base de datos para embeddings y descripciones
- **Procesos independientes**: La extracción de captions y embeddings siguen siendo procesos separados
- **Compatibilidad**: La tabla `image_descriptions` coexiste con las tablas de embeddings sin conflictos

## Sistema de Priorización de Errores

El sistema incluye un mecanismo inteligente de priorización que procesa automáticamente las imágenes que previamente causaron errores, mejorando la eficiencia y reduciendo la lista de errores pendientes.

### Características del Sistema de Priorización

- **Procesamiento automático**: Las imágenes con errores se procesan con prioridad durante cada ejecución
- **Gestión inteligente**: El `ImagePriorityManager` organiza las imágenes por tipo de error y timestamp
- **Integración transparente**: Funciona automáticamente sin configuración adicional
- **Logging detallado**: Seguimiento completo del progreso de procesamiento prioritario

### Funcionamiento

1. **Detección automática**: El sistema carga `logs/error_images.json` al iniciar
2. **Priorización**: Las imágenes con errores se procesan antes que las nuevas
3. **Limpieza automática**: Los errores resueltos se eliminan automáticamente del log
4. **Categorización**: Los errores se agrupan por tipo (503 Service Overloaded, 429 Quota Exceeded, etc.)

### Archivos de Error

- `logs/error_images.json`: Registro centralizado de imágenes con errores
- `logs/completed_directories.json`: Directorios completamente procesados
- `logs/pending_directory.json`: Directorios pendientes de procesamiento

## Base de Datos

El sistema utiliza SQLite con la siguiente estructura:

```sql
CREATE TABLE image_descriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    image_filename TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_name, page_number, image_filename)
);
```

## Logging

### Archivos Generados

- `logs/error_handler.log`: Registro del manejador de errores
- `logs/error_log_manager.log`: Registro del gestor centralizado de errores
- `logs/file_processor.log`: Registro del procesador de archivos
- `logs/error_images.json`: Registro centralizado de errores de imágenes
- `logs/completed_directories.json`: Directorios completamente procesados
- `logs/pending_directory.json`: Directorios pendientes de procesamiento

### Niveles de Log

| Nivel | Código | Descripción |
|-------|--------|-------------|
| DEBUG | 10 | Información detallada |
| INFO | 20 | Progreso general |
| WARNING | 30 | Advertencias |
| ERROR | 40 | Errores de operación |



## Solución de Problemas

### Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| API Key no encontrada | Variable de entorno no configurada | Configurar `PROVIDER_API_KEY` |
| Permisos de base de datos | Sin permisos de escritura | Verificar permisos del directorio |
| Memoria insuficiente | Imágenes muy grandes | Reducir `commit_interval` |
| Rate limit excedido | Demasiadas solicitudes | Usar `--debug` para ver límites |

### Comandos de Depuración

```bash
# Ver estado detallado
python extract_captions.py --status --debug

# Logging completo
python extract_captions.py --root-dir ./imagenes --log-level DEBUG

# Verificar configuración
python -c "from modules_captions import DatabaseManager; print('OK')"
```