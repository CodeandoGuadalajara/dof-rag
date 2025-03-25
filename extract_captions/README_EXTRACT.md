# Sistema de Extracción de Descripciones de Imágenes

Este sistema modular permite procesar imágenes con diferentes modelos de IA para generar descripciones automáticas. Está diseñado específicamente para trabajar con imágenes extraídas de documentos del Diario Oficial de la Federación (DOF).

## Arquitectura del Sistema

El sistema sigue un patrón de diseño basado en interfaces que permite cambiar fácilmente entre diferentes proveedores de IA sin modificar el código cliente.

### Módulos Principales

1. **AbstractAIClient (`modules/AbstractClient.py`)**: 
   - Clase abstracta que define la interfaz común para todos los proveedores de IA
   - Métodos principales:
     - `process_image()`: Procesa una única imagen y devuelve su descripción
     - `set_api_key()`: Configura la clave API necesaria para el servicio
     - `set_question()`: Define la pregunta o prompt utilizado para la descripción
   - Proporciona métodos para modificar parámetros como temperatura, max_tokens, etc.

2. **Implementaciones Concretas**:
   - **GeminiClient (`clientes/gemini_client.py`)**: 
     - Utiliza Google Gemini API para generar descripciones de alta calidad
     - Optimizado para documentos oficiales y extracción de texto visible
   
   - **OpenAIClient (`clientes/openai_client.py`)**: 
     - Implementa conexión con modelos GPT-4o y otros modelos multimodales de OpenAI
     - Excelente para análisis detallado de imágenes complejas
   
   - **OllamaClient (`clientes/ollama_client.py`)**:
     - Permite usar modelos locales a través de Ollama
     - No requiere conexión a internet ni API keys
     - Diversos modelos disponibles (gemma, llava, bakllava, etc.)

3. **FileUtil (`modules/file_utils.py`)**:
   - Gestiona la búsqueda recursiva de imágenes en directorios
   - Funcionalidades:
     - Procesamiento por lotes con periodos de enfriamiento entre lotes
     - Sistema de checkpoint para reanudar procesamiento interrumpido
     - Generación de archivos TXT con las descripciones junto a las imágenes
     - Registro detallado de éxitos, errores y tiempos de procesamiento

4. **UI Components (`modules/ui_components.py`)**:
   - Proporciona una interfaz gráfica interactiva para Jupyter Notebook
   - Controles:
     - Selección de directorio raíz y configuración de parámetros
     - Botones para iniciar, pausar y detener el procesamiento
     - Monitor de progreso en tiempo real
     - Visualización de resultados y estadísticas

## Casos de Uso

### 1. Procesamiento Masivo de Imágenes

Ideal para procesar grandes volúmenes de imágenes extraídas de documentos PDF:

```python
# Configuración del cliente
client = GeminiClient(model="gemini-2.0-flash", max_tokens=256)
client.set_api_key(os.getenv("GEMINI_API_KEY"))
client.set_question("Describe esta imagen detalladamente en español")

# Configuración del procesador de archivos
file_util = FileUtil(root_directory="/ruta/a/documentos", client=client)

# Iniciar procesamiento por lotes
file_util.process_images_in_batches(batch_size=20, cooldown_seconds=30)
```

### 2. Uso con Interfaz Gráfica

Para uso interactivo en Jupyter Notebook:

```python
# Configurar cliente y utilidades
client = OpenAIClient(model="gpt-4o", max_tokens=512)
file_util = FileUtil(root_directory="/ruta/a/documentos", client=client)

# Crear y mostrar la interfaz de usuario
controls = create_processing_interface(client, file_util)
```

## Añadir un Nuevo Proveedor

Para implementar un nuevo proveedor de IA:

1. Crear una nueva clase que herede de `AbstractAIClient`
2. Implementar los métodos requeridos:
   ```python
   class NuevoProveedor(AbstractAIClient):
       def process_image(self, image_path):
           # Lógica específica para este proveedor
           return {"description": resultado}
       
       def set_api_key(self, api_key):
           # Configuración de autenticación
           self.api_key = api_key
   ```
3. Registrar cualquier parámetro específico en el constructor

## Recomendaciones de Uso

- **Para documentos oficiales**: GeminiClient con temperature=0.2 para descripciones precisas
- **Para imágenes complejas**: OpenAIClient con GPT-4o para mayor detalle y contexto
- **Procesamiento local**: OllamaClient con modelos como gemma3:4b para independencia de APIs externas
- **Optimización de costos**: Ajuste el tamaño de los lotes y los tiempos de enfriamiento según los límites de la API

## Ejemplos de Prompts Efectivos

Para obtener mejores resultados con documentos oficiales:

```
Resume brevemente la imagen en español (máximo 3-4 oraciones por categoría):  
- **Texto:** Menciona solo el título y 2-3 puntos clave si hay texto.
- **Mapas:** Identifica la región principal y máximo 2-3 ubicaciones relevantes.
- **Diagramas:** Resume el concepto central en 1-2 oraciones.
- **Logos:** Identifica la entidad y sus características distintivas.
- **Datos visuales:** Menciona solo los 2-3 valores o tendencias más importantes.
Prioriza la información esencial sobre los detalles, manteniendo la descripción breve y directa.
```

Este sistema modular facilita el procesamiento masivo de imágenes con diferentes modelos de IA mientras proporciona control, visibilidad y robustez en el proceso.