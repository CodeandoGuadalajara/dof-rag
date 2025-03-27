# Documentación: query_bd.py

## Descripción General
Este script implementa un buscador semántico que utiliza embeddings vectoriales para encontrar fragmentos de texto relevantes en una base de datos SQLite. Permite buscar documentos por similitud semántica en lugar de simples coincidencias de palabras clave.

## Dependencias
- `sqlite3`: Para la conexión y manipulación de la base de datos
- `sqlite_vec`: Extensión para operaciones vectoriales en SQLite
- `sentence_transformers`: Para generar embeddings de texto
- `typer`: Para crear la interfaz de línea de comandos

## Modelo de Embedding
- **Modelo utilizado**: "nomic-ai/modernbert-embed-base"
- El modelo se carga una sola vez al inicio del script para optimizar el rendimiento

## Funcionalidades

### Comando principal: `search`
Realiza una búsqueda vectorial en la tabla `chunks` y devuelve los resultados más relevantes.

#### Parámetros:
- `query` (obligatorio): Texto o pregunta para buscar
- `limit` (opcional): Número máximo de resultados (predeterminado: 5)
- `db_path` (opcional): Ruta a la base de datos SQLite (predeterminado: "dof_db/db.sqlite")

#### Proceso:
1. Conecta a la base de datos SQLite
2. Carga la extensión vectorial
3. Convierte la consulta en un embedding vectorial
4. Ejecuta una consulta SQL que:
   - Busca en la tabla `chunks`
   - Une con la tabla `documents`
   - Calcula la distancia coseno entre embeddings
   - Ordena por similitud (menor distancia)
   - Limita el número de resultados
5. Muestra los resultados formateados

## Formato de resultados
Para cada resultado muestra:
- Título del documento
- URL del documento
- Encabezado del fragmento
- Puntuación de similitud (1 - distancia_coseno)
- Texto del fragmento (primeros 200 caracteres)

## Uso desde línea de comandos
```bash
python query_bd.py "tu consulta aquí" --limit 10 --db ruta/a/tu/base.sqlite
```

## Estructura de la base de datos
El script asume una base de datos con al menos:
- Tabla `chunks` con campos: `id`, `document_id`, `header`, `text`, `embedding`
- Tabla `documents` con campos: `id`, `title`, `url`