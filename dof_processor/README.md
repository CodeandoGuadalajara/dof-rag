# DOF Processor

Script para convertir archivos DOC a DOCX y unificarlos del Diario Oficial de la Federación (DOF).

## Descripción

Este script automatiza el proceso de conversión de archivos DOC a DOCX usando LibreOffice y los unifica en un solo documento por edición (MAT/VES). Está diseñado específicamente para procesar archivos del DOF organizados por fecha y edición.

## Características

- Conversión DOC a DOCX usando LibreOffice en modo headless
- **Procesamiento automático de ediciones** - procesa automáticamente todas las ediciones (MAT/VES) disponibles
- **Arquitectura de directorios separados** - DOC files como biblioteca de solo lectura, DOCX en directorio separado
- Timeout de 90 segundos por archivo para evitar bloqueos
- Identificación automática de archivos problemáticos
- Generación de reportes de archivos con timeout

## Requisitos

### Sistema
- LibreOffice headless instalado y accesible vía comando `soffice`
- Python 3.7+

### Instalación de LibreOffice Headless

**Ubuntu/Debian:**
```bash
# Solo instalar los componentes necesarios para conversión headless
sudo apt-get update
sudo apt-get install libreoffice-core libreoffice-writer

# Verificar instalación
soffice --headless --version
```

**CentOS/RHEL/Fedora:**
```bash
# Instalar solo el núcleo y writer para conversión
sudo yum install libreoffice-core libreoffice-writer
# o para sistemas más nuevos:
sudo dnf install libreoffice-core libreoffice-writer

# Verificar instalación
soffice --headless --version
```


### Dependencias Python
```bash
uv add typer docxcompose python-docx
```

## Estructura de Directorios

### Estructura de Entrada (Solo Lectura)
```
dof_word/
└── YYYY/
    └── MM/
        └── DDMMYYYY/
            ├── MAT/
            │   ├── archivo1.doc
            │   └── archivo2.doc
            └── VES/
                ├── archivo3.doc
                └── archivo4.doc
```

### Estructura de Salida (DOCX)
```
dof_docx/
└── YYYY/
    └── MM/
        └── DDMMYYYY/
            ├── MAT/
            │   ├── archivo1.docx
            │   ├── archivo2.docx
            │   └── DDMMYYYY_MAT.docx (documento unificado)
            └── VES/
                ├── archivo3.docx
                ├── archivo4.docx
                └── DDMMYYYY_VES.docx (documento unificado)
```

### Principios Arquitectónicos
- **DOC files**: Tratados como biblioteca de solo lectura (nunca se modifican)
- **DOCX files**: Creados en estructura de directorios separada
- **Preservación**: Archivos originales siempre se conservan

## Uso

### Procesar una fecha específica
```bash
uv run dof_processor.py 02/01/2023 --log-level INFO
```

### Procesar un rango de fechas
```bash
uv run dof_processor.py 01/01/2023 31/01/2023 --log-level INFO
```

### Especificar directorio personalizado
```bash
uv run dof_processor.py 02/01/2023 --input-dir ./mi_carpeta --log-level INFO
```

## Parámetros

| Parámetro | Descripción | Valores | Por defecto |
|-----------|-------------|---------|-------------|
| `date` | Fecha inicial (DD/MM/YYYY) | Formato DD/MM/YYYY | Requerido |
| `end_date` | Fecha final para rango (opcional) | Formato DD/MM/YYYY | None |
| `--input-dir` | Directorio de entrada | Ruta del directorio | `./dof_word` |
| `--log-level` | Nivel de logging | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |

## Proceso de Trabajo

2. **Conversión**: Convierte cada archivo DOC a DOCX usando LibreOffice en modo headless
3. **Almacenamiento separado**: Crea archivos DOCX en directorio `dof_docx` separado del original
4. **Unificación**: Combina todos los archivos DOCX de cada edición en un documento único
5. **Limpieza**: Elimina archivos temporales manteniendo archivos originales y unificados
6. **Reporte**: Genera reporte de archivos problemáticos si los hay

## Archivos de Salida

### Directorio de Salida: `dof_docx/`
- **Archivos DOCX individuales**: Conversiones individuales de cada archivo DOC
- **Documento unificado**: `DDMMYYYY_EDITION.docx` (ej: `02012023_MAT.docx`)
- **Reporte de problemas**: `archivos_problematicos_YYYYMMDD_HHMMSS.txt` (si hay archivos con timeout)
- **Log del proceso**: Salida en consola y archivo `dof_processor.log`


## Manejo de Errores

- **Timeout**: Archivos que excedan 90 segundos se marcan como problemáticos
- **Procesos huérfanos**: Limpieza automática de procesos LibreOffice bloqueados
- **Archivos corruptos**: Se omiten y se registran en el log
- **Directorios faltantes**: Se reportan como advertencias

## Notas Importantes

### Requisitos del Sistema
- El script requiere que LibreOffice esté instalado y disponible en el PATH del sistema
- Se recomienda tener suficiente espacio en disco para los archivos DOCX duplicados
