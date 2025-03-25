# Pruebas de Visión por Computadora para DOF-RAG

Este directorio contiene scripts para evaluar diferentes modelos de visión por computadora con un enfoque en el análisis de documentos e imágenes del Diario Oficial de la Federación.

## Scripts Disponibles

### 1. test_gemini.py

Prueba el modelo Gemini de Google para análisis de imágenes.

```bash
# Ejemplo básico
uv run tests/test_gemini.py

# Opciones personalizadas
uv run tests/test_gemini.py --tokens 512 --dir ./imagenes_prueba --question "¿Qué información contiene esta imagen?"

# Opciones disponibles
--tokens       # Número de tokens máximos (256, 512, 1024, 2048, 4096, 8192)
--streaming    # Activa el modo streaming de respuestas
--dir          # Directorio con las imágenes (default: ./imagenes_prueba)
--question     # Pregunta personalizada
```

### 2. test_moondream.py

Prueba el modelo Moondream (versiones 0.5B y 2B) para análisis de imágenes.

```bash
# Ejemplo básico
uv run tests/test_moondream.py

# Opciones personalizadas
uv run tests/test_moondream.py --model-size big --mode batch --dir ./imagenes_prueba

# Opciones disponibles
--mode         # Modo de procesamiento: single (una imagen) o batch (todas las imágenes)
--model-size   # Tamaño del modelo: small (0.5B) o big (2B)
--question/-q  # Pregunta personalizada
--dir/-d       # Directorio de imágenes (default: ./imagenes_prueba)
--output/-o    # Ruta del archivo de salida
--workers/-w   # Número de workers para procesamiento en paralelo
```

### 3. test_ollama.py

Prueba modelos locales de Ollama para análisis de imágenes.

```bash
# Ejemplo básico
uv run tests/test_ollama.py --image_dir ./imagenes_prueba

# Opciones personalizadas
uv run tests/test_ollama.py --image_dir ./imagenes_prueba --model gemma3:4b --max_tokens 4096 --mode batch

# Opciones disponibles
--image_dir/-d     # Directorio con imágenes (default: ./imagenes_prueba)
--output/-o        # Nombre del archivo de salida
--model/-m         # Modelo a utilizar (default: gemma3:4b)
--temperature/-t   # Temperatura para generación (default: 0.5)
--max_tokens       # Tokens máximos a generar (default: 4096)
--threads          # Hilos de procesamiento
--mode             # Modo de procesamiento: single o batch (default: single)
--top_k            # Limitar selección de tokens (default: 20)
```

## Resultados de las Pruebas

Los resultados se almacenan en formato texto en el directorio `tests/results/`. El formato del nombre de archivo incluye:
- Nombre del modelo usado
- Sufijo del modo (_single o _batch)
- Configuración de tokens

Ejemplo: `gemini-2-0-flash_tokens_256.txt` o `ollama__gemma3_4b__single__tokens_4096.txt`

### Formato de Resultados

Cada archivo contiene:

1. **Cabecera**: Información sobre el modelo y configuración
2. **Resultados por imagen**:
   - Timestamp del procesamiento
   - Información del modelo y tokens
   - Nombre de la imagen
   - Métricas de tiempo de procesamiento
   - Descripción y respuesta generada
3. **Resumen final** con estadísticas de rendimiento

### Comparación de Modelos

Basado en las pruebas realizadas, podemos observar:

1. **Google Gemini**: Ofrece respuestas de alta calidad y muy relevantes al contexto, con tiempos de respuesta rápidos (~15s por imagen). Las descripciones son precisas y detecta correctamente textos, diagramas y elementos visuales. Desventaja: requiere conexión a internet y una API key.

2. **Moondream**: Los modelos pequeños (0.5B) son rápidos pero generan descripciones más genéricas, en la algunas veces sin sentido o en la gran mayoria no es capaz de realizar la predicción. El modelo grande (2B) ofrece respuestas un poco más detalladas pero tarda significativamente (~200s por imagen). La traducción del inglés al español agrega un paso adicional.

3. **Ollama (local)**: Ofrece una amplia variedad de modelos. Los resultados varían por modelo:
   - gemma3:4b: Respuestas de calidad intermedia con tiempos moderados (~120s)
   - granite3.2-vision: Mejor calidad pero más lento (~140s)
   - moondream:latest (1.8B): Respuestas más limitadas y a veces incompletas, con tiempos moderados (~10-30s). Aunque es más rápido que otros modelos locales, las respuestas suelen ser demasiado escuetas o no logran identificar correctamente el contenido principal de las imágenes complejas o con texto.

## Visualización de Resultados

Para comparar resultados entre modelos, revisa los archivos individuales en `tests/results/`. Los tiempos de procesamiento y tokens generados dan una idea de la eficiencia, mientras que las descripciones generadas muestran la calidad.

## Procesamiento de Imágenes con Módulos

El proyecto incluye un sistema modular para procesar imágenes usando Google Gemini, diseñado especialmente para Jupyter Notebooks con una interfaz interactiva:

**Notas Adicionales**:

1. **Dependencias**: Todos los scripts usan el formato TOML para declarar sus dependencias, por lo que se pueden ejecutar directamente con `uv run`.

2. **Imágenes**: Las pruebas están diseñadas para trabajar con imágenes de ejemplo en el directorio `./imagenes_prueba/`.
