# dof-rag
dog-raf es un chat y un sistema de consulta por generación aumentada para explorar las ediciones del Diario Oficial de la Federación de México.

# Requerimientos

Instala [uv](https://docs.astral.sh/uv/) para manejar las dependencias y ejecutar el proyecto.

Una vez instalado uv, ejecuta el siguiente comando para iniciar el proyecto:

```bash
uv venv # Crear un entorno virtual
uv sync # Sincronizar dependencias
```
## Activacion del Entorno Virtual
Cada vez que trabajes en el proyecto, activa el entorno virtual:

* ### **Linux/macOs:**
```bash
source .venv/bin/activate
```

* ### **Windows:**
```bash
.venv\Scripts\activate
```

Para desactivar el entorno virtual, ejecuta:
```bash
deactivate
```

## Bajar archivos del DOF

Para bajar archivos del DOF se usa el script get_dof.py de la siguiente manera:

```bash
python get_dof.py --help
python get_dof.py --start-year=2025 --end-year=2023
```

Esto crea directorios como:

```
$ tree --sort=mtime dof | head -n 10
dof
├── 2025
│   ├── 01
│   │   ├── 02012025-MAT.pdf
│   │   ├── 03012025-MAT.pdf
│   │   ├── 06012025-MAT.pdf
│   │   ├── 07012025-MAT.pdf
...
```

## Extraer markdown:

### Por folders

UV instala la dependencia [marker](https://github.com/VikParuchuri/marker) que contiene un ejecutable para convertir PDFs a formato markdown.

Para convertir un folder completo ejecuta este comando:

```bash
marker --output_dir dof_markdown/2024/04/ \
  --paginate_output \
  --languages="es" \
  --skip_existing \
  --workers=1 \
  dof/2024/04/
# este comando tardó 2h 31m 23s en una macbook pro M3 de 36GB RAM
```

**NOTA**: Si el comando queda a la mitad, conviene borrar las carpetas incompletas.
Para ver cuáles archivos están incompletos puedes revisar el archivo markdown que tiene separaciones por hojas gracias a la opcíon `--paginate_output`
con el formato `{pagina}------------------------------------------------`. Revisa que contenga todas las páginas del archivo PDF.

Una vez borradas las carpetas incompletas, puedes volver a ejecutar el comando anterior para que el `--skip_existing` se salte las carpetas que ya existen.

### Por archivos

Para extraer el markdown de un archivo específico:

```bash
marker_single --output_dir dof_markdown/2024/04/ \
  --paginate_output \
  --languages="es" \
  dof/2024/04/01042024-MAT.pdf
# este comando tardó 2m 7s en una macbook pro M3 de 36GB RAM
# Sorprendentemente, porque otros pueden tardar más de 10 minutos.
```

## Extraer embeddings

Para extraer embeddings de un archivo específico:

```bash
python extract_embeddings.py dof_markdown/2024/04/
```

Puedes especificar la carpeta de un solo archivo, o la carpeta de un mes, o incluso la carpeta de un año.
En una macbook pro M3 de 36GB RAM, este comando tardó 190 minutos en extraer los embeddings de enero del 2025.
