# DOF Word Downloader

Script para descargar archivos WORD del Diario Oficial de la Federación (DOF) de México.

## Descripción

Este proyecto contiene una herramienta de línea de comandos que permite descargar archivos WORD (.doc) del sitio web oficial del Diario Oficial de la Federación.

## Características

- Descarga archivos WORD (.doc) del DOF
- Soporte para fechas individuales o rangos de fechas
- Descarga de ediciones matutina (MAT) y vespertina (VES)
- Validación de integridad de archivos descargados
- Organización automática de archivos por fecha y edición

## Requisitos

### Python
- Python 3.7 o superior

### Dependencias

Este script tiene las dependencias definidas inline usando el formato de uv script:
- requests
- typer
- beautifulsoup4
- urllib3

## Instalación

Este script utiliza `uv` con dependencias inline

```bash
# Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Luego puedes ejecutar el script directamente y uv manejará automáticamente las dependencias:

```bash
uv run get_word_dof.py [FECHA] [OPCIONES]
```

## Uso

### Sintaxis básica
```bash
uv run get_word_dof.py [FECHA] [FECHA_FIN] [OPCIONES]
```

### Parámetros

- **FECHA** (requerido): Fecha en formato DD/MM/YYYY
- **FECHA_FIN** (opcional): Fecha de fin para descargar un rango

### Opciones

- `--output-dir`: Directorio de salida (por defecto: `./dof_word`)
- `--editions`: Ediciones a descargar (`mat`, `ves`, o `both`) (por defecto: `both`)
- `--log-level`: Nivel de logging (`DEBUG`, `INFO`, `WARNING`, `ERROR`) (por defecto: `INFO`)
- `--sleep-delay`: Tiempo de espera entre descargas en segundos (por defecto: `1.0`)

## Ejemplos de uso

### 1. Descargar una fecha específica
```bash
# Descargar ambas ediciones del 2 de enero de 2023
uv run get_word_dof.py 02/01/2023 --editions both --log-level INFO
```

### 2. Descargar un rango de fechas
```bash
# Descargar todo enero de 2023
uv run get_word_dof.py 01/01/2023 31/01/2023 --editions both --log-level INFO
```

### 3. Descargar solo edición matutina
```bash
# Solo edición matutina del 15 de febrero de 2023
uv run get_word_dof.py 15/02/2023 --editions mat
```

### 4. Especificar directorio personalizado
```bash
# Guardar en directorio personalizado
uv run get_word_dof.py 02/01/2023 --output-dir ./mi_carpeta --editions both
```

### 5. Ajustar velocidad de descarga
```bash
# Descargar más lento (2 segundos entre archivos)
uv run get_word_dof.py 02/01/2023 --sleep-delay 2.0

# Descargar más rápido (0.5 segundos entre archivos)
uv run get_word_dof.py 02/01/2023 --sleep-delay 0.5
```

## Estructura de archivos

Los archivos se organizan automáticamente en la siguiente estructura:

```
dof_word/
└── [AÑO]/
    └── [MES]/
        └── [DDMMAAAA]/
            ├── MAT/
            │   ├── DOF_20230102_MAT_123456.doc
            │   └── DOF_20230102_MAT_789012.doc
            └── VES/
                ├── DOF_20230102_VES_345678.doc
                └── DOF_20230102_VES_901234.doc
```

### Ejemplo:
```
dof_word/
└── 2023/
    └── 01/
        └── 02012023/
            ├── MAT/
            │   ├── DOF_20230102_MAT_123456.doc
            │   └── DOF_20230102_MAT_789012.doc
            └── VES/
                ├── DOF_20230102_VES_345678.doc
                └── DOF_20230102_VES_901234.doc
```

## Logging

El script genera logs detallados que se guardan en:
- **Archivo**: `word_download.log`
- **Consola**: Salida estándar

### Niveles de logging:
- **DEBUG**: Información muy detallada
- **INFO**: Información general del progreso
- **WARNING**: Advertencias (archivos ya existentes, etc.)
- **ERROR**: Errores durante la descarga

## Características técnicas

### Validación de archivos
El script valida automáticamente que los archivos descargados sean válidos:
- Verifica el tamaño mínimo (>1KB)
- Comprueba las firmas de archivo WORD
- Elimina archivos corruptos automáticamente

### Configuración SSL
- Adaptador TLS personalizado para compatibilidad con el sitio del DOF
- Deshabilitación de verificación SSL para evitar errores de certificado
- User-Agent personalizado para evitar bloqueos

### Manejo de errores
- Reintentos automáticos en caso de errores de red
- Validación de fechas y parámetros
- Manejo graceful de interrupciones
