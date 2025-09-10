# DOF Processor

Script para convertir archivos DOC a DOCX y unificarlos del Diario Oficial de la Federación (DOF).

## Descripción

Este script automatiza el proceso de conversión de archivos DOC a DOCX usando LibreOffice y los unifica en un solo documento por edición (MAT/VES). Está diseñado específicamente para procesar archivos del DOF organizados por fecha y edición.

## Características

- Conversión DOC a DOCX usando LibreOffice en modo headless
- Timeout de 90 segundos por archivo para evitar bloqueos
- Identificación automática de archivos problemáticos
- Generación de reportes de archivos con timeout
- Limpieza automática de archivos temporales
- Logging detallado del proceso

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

## Estructura de Directorios Esperada

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

## Uso

### Procesar una fecha específica
```bash
uv run dof_processor.py 02/01/2023 --editions both --log-level INFO
```

### Procesar un rango de fechas
```bash
uv run dof_processor.py 01/01/2023 31/01/2023 --editions both --log-level INFO
```

### Especificar directorio personalizado
```bash
uv run dof_processor.py 02/01/2023 --input-dir ./mi_carpeta --editions both --log-level INFO
```

## Parámetros

| Parámetro | Descripción | Valores | Por defecto |
|-----------|-------------|---------|-------------|
| `date` | Fecha inicial (DD/MM/YYYY) | Formato DD/MM/YYYY | Requerido |
| `end_date` | Fecha final para rango (opcional) | Formato DD/MM/YYYY | None |
| `--input-dir` | Directorio de entrada | Ruta del directorio | `./dof_word` |
| `--editions` | Ediciones a procesar | `mat`, `ves`, `both` | `both` |
| `--log-level` | Nivel de logging | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |

## Proceso de Trabajo

1. **Búsqueda**: Localiza archivos DOC en las carpetas MAT/VES
2. **Conversión**: Convierte cada archivo DOC a DOCX usando LibreOffice
3. **Unificación**: Combina todos los archivos DOCX en un documento único
4. **Limpieza**: Elimina archivos temporales y residuales
5. **Reporte**: Genera reporte de archivos problemáticos si los hay

## Archivos de Salida

- **Documento unificado**: `DDMMYYYY_EDITION.docx` (ej: `02012023_MAT.docx`)
- **Reporte de problemas**: `archivos_problematicos_YYYYMMDD_HHMMSS.txt` (si hay archivos con timeout)
- **Log del proceso**: Salida en consola y archivo de log

## Manejo de Errores

- **Timeout**: Archivos que excedan 90 segundos se marcan como problemáticos
- **Procesos huérfanos**: Limpieza automática de procesos LibreOffice bloqueados
- **Archivos corruptos**: Se omiten y se registran en el log
- **Directorios faltantes**: Se reportan como advertencias

## Notas Importantes

- El script requiere que LibreOffice esté instalado y disponible en el PATH del sistema
- La limpieza automática elimina archivos temporales
