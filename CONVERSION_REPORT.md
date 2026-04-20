# Reporte de Conversión: DOF .doc → Markdown

**Fecha:** Abril 2026  
**Proyecto:** DOF-RAG  
**Script:** `convert_doc_to_md.py`

## Resumen

| Métrica | Valor |
|---------|-------|
| Total de archivos .doc | 647,017 |
| Total de archivos .md generados | **647,017** |
| Cobertura | **100%** |
| Tamaño original (.doc) | 71 GB |
| Tamaño convertido (.md) | 33 GB |
| Años cubiertos | 1999–2025 (27 años) |
| Reducción de tamaño | 53% |

## Pipeline de Conversión

El proceso utilizó un pipeline de dos etapas con herramientas de código abierto:

```
.doc → [LibreOffice headless] → .docx → [pandoc + filtro Lua] → .md
```

### Herramientas principales

1. **LibreOffice** (`soffice --headless`): Conversión de .doc (formato binario OLE) a .docx (formato XML Office Open)
2. **pandoc** con filtro Lua personalizado (`dof_headers.lua`): Conversión de .docx a Markdown limpio, preservando encabezados y estructura

### Configuración

- **Workers paralelos:** 4 (bulk), 2 (retries), 1 (último intento)
- **Timeout por archivo:** 600 segundos (LibreOffice + pandoc)
- **Reintentos máximos:** 3 por archivo
- **Filtro Lua:** Normaliza encabezados del DOF para mejor estructura

## Resultados por Etapa

### Etapa 1: Conversión masiva (LibreOffice + pandoc)

| Parámetro | Valor |
|-----------|-------|
| Archivos procesados | 647,017 |
| Exitosos | 646,986 (99.995%) |
| Fallidos | 31 (0.005%) |
| Duración total | ~20 horas (bulk) + retries |
| Velocidad promedio | ~8.7 archivos/segundo |

### Etapa 2: Recuperación de archivos fallidos

Los 31 archivos que LibreOffice no pudo convertir se clasificaron en dos categorías:

| Tipo de falla | Causa | Cantidad |
|---------------|-------|----------|
| `libreoffice_timeout` | Archivos complejos/grandes que exceden 600s | 26 |
| `libreoffice_failed` | Formatos corruptos o no reconocidos | 5 |

### Etapa 3: Métodos alternativos

Para los 31 archivos restantes se utilizaron herramientas complementarias:

| Método | Tipo de archivo | Archivos | Descripción |
|--------|----------------|----------|-------------|
| `catdoc` | Binary OLE .doc | 25 | Extrae texto directamente del formato binario Word 97-2003 |
| `python-docx` | DOCX disfrazado de .doc | 5 | Archivos ZIP (Office 2007+) con extensión .doc |

**Resultado:** 30 de 31 archivos recuperados exitosamente. El último archivo ya había sido convertido en una ejecución previa.

## Distribución de Archivos por Año

| Año | Documentos | Año | Documentos |
|-----|-----------|-----|-----------|
| 1999 | 24,958 | 2013 | 30,582 |
| 2000 | 22,533 | 2014 | 31,620 |
| 2001 | 20,441 | 2015 | 28,833 |
| 2002 | 20,468 | 2016 | 26,826 |
| 2003 | 20,511 | 2017 | 26,514 |
| 2004 | 20,637 | 2018 | 25,009 |
| 2005 | 21,274 | 2019 | 21,890 |
| 2006 | 23,419 | 2020 | 16,733 |
| 2007 | 23,779 | 2021 | 20,224 |
| 2008 | 26,719 | 2022 | 23,536 |
| 2009 | 24,757 | 2023 | 24,712 |
| 2010 | 25,427 | 2024 | 20,337 |
| 2011 | 29,623 | 2025 | 15,643 |
| 2012 | 30,012 | | |

**Pico:** 2014 con 31,620 documentos  
**Mínimo:** 2025 con 15,643 (año en curso, datos parciales)

## Distribución de Tamaños de Archivos .md

| Categoría | Tamaño | Cantidad | Porcentaje |
|-----------|--------|----------|------------|
| Muy pequeños | < 1 KB | 18,479 | 2.9% |
| Pequeños | 1–10 KB | 461,528 | 71.3% |
| Medianos | 10–100 KB | 128,640 | 19.9% |
| Grandes | 100 KB–1 MB | 33,062 | 5.1% |
| Muy grandes | > 1 MB | 5,308 | 0.8% |

**Nota:** Los 227 archivos con menos de 100 bytes corresponden a documentos con mínima información (ej. portadas, correcciones).

## Lecciones Aprendidas

1. **LibreOffice maneja el 99.995% de los archivos .doc** del DOF de manera confiable, pero tiene problemas con archivos extremadamente grandes o con formatos internos inusuales.

2. **`catdoc` es un complemento excelente** para archivos OLE binarios que LibreOffice no puede procesar. Es rápido (< 1 segundo por archivo) y directo.

3. **Algunos archivos .doc son en realidad .docx** (formato ZIP). Verificar los primeros bytes (`PK` = ZIP) permite enrutarlos al extractor correcto.

4. **El timeout de 600 segundos por archivo** es suficiente para la gran mayoría, pero algunos archivos del DOF (ej. tarifas arancelarias completas) pueden contener cientos de miles de líneas y requerir extracción alternativa.

5. **Procesamiento paralelo es clave**: Con 4 workers se alcanzan ~8.7 archivos/segundo. El cuello de botella es LibreOffice, no I/O de disco.

## Estructura de Salida

```
dof_md/
└── {año}/
    └── {mes}/
        └── {fecha}/
            └── {sección}/
                └── {documento}.md
```

**Ejemplo:**
```
dof_md/2025/01/02012025/MAT/001_DOF_20250102_MAT_5746544.md
```

La estructura espeja exactamente la del directorio fuente `dof_word/`, facilitando la trazabilidad entre archivos originales y convertidos.

## Lo que falta: los PDFs escaneados

La conversión de .doc a Markdown cubre los documentos individuales publicados desde 1999. Pero el DOF tiene un acervo mucho más amplio en formato PDF escaneado que todavía no podemos procesar con este pipeline.

### El paisaje de los PDFs del DOF

El sitio del DOF (diariooficial.gob.mx) ofrece PDFs de las ediciones completas. Nuestro análisis reveló:

| Periodo | Tipo de PDF | Tamaño promedio | Texto extraíble |
|---------|------------|-----------------|-----------------|
| 1920-1922 | Ediciones aisladas | Desconocido | No (escaneados) |
| 1990-2004 | Escaneado (imágenes) | 65-72 MB/edición | **No** |
| 2005-2008 | Escaneado | 2-5 MB/edición | **No** |
| 2009-2011 | Transición (mixto) | 4-5 MB/edición | Parcial |
| 2012-2025 | Mayormente digital | 5-9 MB/edición | Sí |

### Estado actual de la descarga

| Componente | Estado |
|------------|--------|
| PDFs 2002-2025 | Descargados (6,079 ediciones, 102 GB) |
| PDFs 1990-2001 | **No descargados** (~3,000 ediciones estimadas) |
| PDFs 1920-1922 | **No descargados** (ediciones aisladas) |

### Escala del reto

| Métrica | Valor |
|---------|-------|
| Ediciones descargadas | 6,079 |
| Ediciones pendientes de descargar | ~3,000+ (1990-2001) + ediciones aisladas 1920s |
| Páginas escaneadas estimadas (total) | **~650,000-850,000** |
| Período con documentos solo en formato escaneado | 1990-2004 (~15 años) |

### Próximo paso: OCR con modelos VLM

Los PDFs escaneados contienen décadas de documentos legales de México que actualmente son inaccesibles para búsqueda de texto. El plan es:

1. **Descargar** los PDFs faltantes (1990-2001)
2. **Ejecutar OCR** con modelos de visión-lenguaje (VLM) como LightOnOCR-2, GLM-OCR o DeepSeek-OCR
3. **Generar Markdown** limpio a partir de las imágenes escaneadas
4. **Integrar** con los 647,017 archivos .md ya existentes

Costo estimado del OCR: ~$0.002/página con GPU en la nube, totalizando ~$800-1,500 USD para toda la colección.

Este paso no solo beneficiará al sistema RAG, sino que representará una digitalización del patrimonio documental de México: décadas de leyes, decretos y resoluciones que actualmente solo existen como imágenes escaneadas inaccesibles.

---

*Generado automáticamente como parte del proyecto [DOF-RAG](https://github.com/CodeandoGuadalajara/dof-rag).*
