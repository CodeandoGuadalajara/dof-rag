# DOF DOCX to Markdown Converter

Convierte archivos DOCX del DOF a Markdown usando Pandoc.

## Prerrequisitos

Antes de usar este script, **DEBES** haber ejecutado los siguientes scripts en orden:

1. `get_word_dof.py` - Descarga archivos DOCX del DOF
2. `dof_processor.py` - Organiza los archivos descargados

### Estructura esperada

Los scripts anteriores crean esta estructura:

```
dof_docx/
├── 2023/
│   ├── 01/
│   │   ├── 02012023/
│   │   │   ├── MAT/
│   │   │   │   └── archivo1.docx
│   │   │   └── VES/
│   │   │       └── archivo2.docx
│   │   ├── 03012023/
│   │   └── ...
│   └── 02/
└── 2024/
    └── 01/
```

## Requisitos

- Python 3.7+
- Pandoc
- Filtros LUA personalizados (incluidos en el repositorio)
- Dependencias Python:
  - typer

## Instalación de dependencias

### 1. Instalar Pandoc

#### Windows
- Descarga el instalador desde la [página oficial de Pandoc](https://github.com/jgm/pandoc/releases/latest) y ejecútalo.


#### Linux
- Usando el gestor de paquetes (puede estar desactualizado):
  ```bash
  sudo apt-get install pandoc
  # o en Fedora
  sudo dnf install pandoc
  ```
- Para la última versión, descarga el paquete desde la [página oficial](https://github.com/jgm/pandoc/releases/latest) y sigue las instrucciones:
  ```bash
  sudo dpkg -i <archivo.deb>
  # o
  tar xvzf <archivo.tar.gz> --strip-components 1 -C /usr/local/
  ```

#### macOS
```bash
brew install pandoc
```

### 2. Instalar typer
```bash
uv add typer
```

## Uso

Ejecuta el script desde la terminal usando UV:

- Procesar todos los archivos:
  ```bash
  uv run dof_docx_to_md.py
  ```
- Procesar archivos de una fecha específica:
  ```bash
  uv run dof_docx_to_md.py 22/01/2025
  ```
- Procesar archivos en un rango de fechas:
  ```bash
  uv run dof_docx_to_md.py 01/01/2025 31/01/2025
  ```

## Estructura de Salida

El script crea la siguiente estructura de salida:

```
dof_word_md/
├── 2023/
│   ├── 01/
│   │   ├── 02012023/
│   │   │   ├── MAT/
│   │   │   │   ├── archivo1.md
│   │   │   │   └── media_temp/
│   │   │   │       └── media/
│   │   │   │           ├── image1.png
│   │   │   │           └── image2.jpg
│   │   │   └── VES/
│   │   │       ├── archivo2.md
│   │   │       └── media_temp/
│   │   │           └── media/
│   │   └── ...
│   └── 02/
└── 2024/
```


### Opciones
```bash
uv run dof_docx_to_md.py 02/01/2020 --input-dir dof_word --output-dir dof_word_md --log-level DEBUG
```

## Notas
- Procesa siempre ambas ediciones (MAT y VES).
- Revisa el archivo de log `convert_docx_to_md.log` para detalles y errores.
- Asegúrate de que el filtro LUA `pandoc_filters/dof_headers.lua` exista antes de ejecutar el script.

## Recursos
- [Documentación oficial de Pandoc](https://pandoc.org/installing.html)
- [Repositorio de Pandoc](https://github.com/jgm/pandoc)
