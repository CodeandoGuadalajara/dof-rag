# Análisis de Overhead de Almacenamiento por Dimensión de Embeddings

## Resumen Ejecutivo

Este documento analiza el impacto del tamaño de los vectores de embedding en el almacenamiento de la base de datos DuckDB. Se utilizaron **tres bases de datos idénticas** que contienen los mismos documentos y chunks del DOF, diferenciándose únicamente en la **dimensión de los embeddings** (512d, 768d, 1024d).

**Fecha de análisis:** 12 de septiembre de 2025  
**Bases de datos analizadas:** 
- `db_qwen_512.duckdb` (embeddings de 512 dimensiones)
- `db_qwen_768.duckdb` (embeddings de 768 dimensiones)  
- `db_qwen_1024.duckdb` (embeddings de 1024 dimensiones)

**Contenido idéntico:** Las tres bases contienen exactamente los mismos 10,090 chunks de documentos del DOF. Solo varía la dimensión del vector embedding.

## Resultados del Análisis

### Mediciones Reales

| Dimensión | Tamaño DB | Chunks | Tamaño/Chunk | Overhead/Chunk | Factor Overhead |
|-----------|-----------|--------|--------------|----------------|-----------------|
| **512d**  | 216.3 MB  | 10,090 | 21.95 KB     | 14.95 KB       | 3.06x          |
| **768d**  | 224.8 MB  | 10,090 | 22.81 KB     | 14.81 KB       | 2.78x          |
| **1024d** | 249.0 MB  | 10,090 | 25.27 KB     | 16.27 KB       | 2.74x          |

### Desglose Técnico

**Tamaño base por chunk (sin embedding):** 5,120 bytes
- id: 4 bytes  
- document_id: 4 bytes
- text: ~5,000 bytes
- header: ~100 bytes  
- page_number: 4 bytes
- created_at: 8 bytes

**Tamaño del embedding por dimensión:**
- 512d: 2,048 bytes (512 × 4 bytes)
- 768d: 3,072 bytes (768 × 4 bytes)  
- 1024d: 4,096 bytes (1024 × 4 bytes)

**Tamaño teórico total:**
- 512d: 7,168 bytes (5,120 + 2,048)
- 768d: 8,192 bytes (5,120 + 3,072)
- 1024d: 9,216 bytes (5,120 + 4,096)

## Proyecciones de Almacenamiento (25 años)

### Escenario DOF
- 1 documento por día × 365 días × 25 años = 9,125 documentos
- 300 chunks por documento = 2,737,500 chunks totales
- 15 imágenes por documento = 136,875 imágenes totales

### Proyecciones por Dimensión

| Dimensión | Chunks (GB) | Documents (MB) | Images (MB) | **Total (GB)** |
|-----------|-------------|----------------|-------------|----------------|
| **512d**  | 57.30       | 3.56           | 267.33      | **57.56**      |
| **768d**  | 59.55       | 3.56           | 267.33      | **59.82**      |
| **1024d** | 65.98       | 3.56           | 267.33      | **66.24**      |

**Diferencias:**
- 768d vs 512d: +2.26 GB (+4%)
- 1024d vs 512d: +8.68 GB (+15%)

## Análisis de Escalabilidad

### Proyecciones para Diferentes Volúmenes

| Chunks | 512d | 768d | 1024d |
|--------|------|------|-------|
| 10,000 | 214.8 MB | 223.4 MB | 247.5 MB |
| 100,000 | 2.09 GB | 2.17 GB | 2.41 GB |
| 1,000,000 | 20.95 GB | 21.74 GB | 24.12 GB |

### Dimensiones Teóricas

**1536d (Proyección teórica):**
- Tamaño teórico: ~11.26 KB/chunk
- Tamaño estimado: ~31 KB/chunk (factor 2.75x)

**2048d (Proyección teórica):**
- Tamaño teórico: ~13.31 KB/chunk  
- Tamaño estimado: ~37 KB/chunk (factor 2.8x)

## Conclusiones

### Hallazgos Clave
1. **Overhead consistente:** Factor promedio de ~2.9x entre dimensiones
2. **Diferencias manejables:** Solo 8.68 GB adicionales (1024d vs 512d) en 25 años
3. **Escalabilidad lineal:** El crecimiento es proporcional al número de chunks
4. **Viabilidad:** Todas las dimensiones son técnicamente factibles

### Recomendaciones
- **512d:** Máxima eficiencia de almacenamiento
- **768d:** Mejor balance eficiencia/dimensión (factor 2.78x)
- **1024d:** Mayor capacidad de representación con overhead razonable
- **1536d/2048d:** Solo si se requiere máxima precisión (estimaciones teóricas)

### Consideración de Calidad
En pruebas con el modelo generador de embeddings se observaron diferencias de calidad entre dimensiones, donde dimensiones superiores proporcionan mejor precisión en recuperación.

## Metodología

### Bases de Datos Utilizadas
- **Contenido idéntico:** 10,090 chunks de documentos DOF
- **Variable:** Solo la dimensión del embedding (512d, 768d, 1024d)  
- **Modelo:** Embeddings generados con Qwen-0.6B

### Proceso de Cálculo
1. **Medición real:** Tamaño de archivo en disco
2. **Cálculo teórico:** Esquema + embedding  
3. **Factor overhead:** Real / Teórico
4. **Proyecciones:** Extrapolación lineal

### Archivos de Soporte
- `overhead_analysis/embedding_overhead_analysis.py`: Script principal
- `overhead_analysis/embedding_overhead_analysis_results.json`: Datos completos
- `overhead_analysis/verify_databases.py`: Verificación de bases de datos

---

**Documento generado:** 12 de septiembre de 2025  
**Script utilizado:** `overhead_analysis/embedding_overhead_analysis.py`