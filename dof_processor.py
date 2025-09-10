#!/usr/bin/env python3
# /// script
# dependencies = [
#   "typer",
#   "docxcompose",
#   "python-docx",
# ]
# ///
"""
Script para convertir archivos DOC a DOCX y unificarlos

Este script busca archivos DOC en las carpetas MAT/VES, los convierte a DOCX
usando LibreOffice en modo headless y luego los unifica en un solo documento por edición.

Características:
- Conversión DOC a DOCX usando LibreOffice (compatible con Linux)
- Timeout de 90 segundos por archivo para evitar bloqueos
- Identificación archivos problemáticos que lleguen a timeout
- Limpieza de archivos temporales

Requisitos:
- LibreOffice instalado y accesible via comando 'soffice'
- Python 3.7+
- Dependencias: typer, docxcompose, python-docx

Ejemplos de uso:
# Para procesar una fecha específica:
uv run dof_processor.py 02/01/2023 --editions both --log-level INFO

# Para un rango de fechas:
uv run dof_processor.py 01/01/2023 31/01/2023 --editions both --log-level INFO

# Especificando directorio personalizado:
uv run dof_processor.py 02/01/2023 --input-dir ./mi_carpeta --editions both --log-level INFO
"""

import re
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import typer

# Importaciones para docxcompose y python-docx
from docxcompose.composer import Composer
from docx import Document

# Importaciones para conversión DOC a DOCX con LibreOffice
import subprocess
import tempfile
import shutil


# Constante para timeout de conversión
CONVERSION_TIMEOUT_SECONDS = 90

# Lista global para archivos problemáticos que llegaron a timeout
problematic_files = []

def kill_libreoffice_processes() -> int:
    """
    Mata todos los procesos LibreOffice (soffice) activos para evitar que procesos
    huérfanos afecten conversiones posteriores después de timeouts.
    
    Returns:
        int: Número de procesos eliminados (0 si no había procesos)
    """
    try:
        # Ejecutar pkill para matar todos los procesos soffice
        result = subprocess.run(
            "pkill -f soffice",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logging.warning("Eliminando procesos LibreOffice huérfanos...")
            return 1  # Al menos un proceso fue eliminado
        else:
            # pkill retorna 1 si no encuentra procesos para matar
            return 0
            
    except Exception as e:
        logging.error(f"Error inesperado al limpiar procesos LibreOffice: {e}")
        return 0

def convert_doc_to_docx(doc_path: Path, docx_path: Optional[Path] = None) -> Optional[Path]:
    """
    Convierte un archivo DOC a DOCX usando LibreOffice en modo headless
    
    Args:
        doc_path: Ruta al archivo DOC original
        docx_path: Ruta de destino para el archivo DOCX (opcional)
        
    Returns:
        Ruta al archivo DOCX convertido o None si falló la conversión
    """
    global problematic_files
    
    if not doc_path.exists():
        logging.error(f"Archivo DOC no encontrado: {doc_path}")
        return None
    
    # Generar ruta de destino si no se proporciona
    if docx_path is None:
        docx_path = doc_path.with_suffix('.docx')
    
    # Crear directorio temporal para la conversión
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_doc = temp_path / doc_path.name
        temp_docx = temp_path / docx_path.name
        
        try:
            # Copiar archivo DOC al directorio temporal
            shutil.copy2(doc_path, temp_doc)
            
            # Comando LibreOffice para conversión headless
            cmd = [
                'soffice',
                '--headless',
                '--convert-to', 'docx',
                '--outdir', str(temp_path),
                str(temp_doc)
            ]
            
            # Crear archivo de log para este proceso
            log_file = temp_path / f"conversion_{doc_path.stem}.log"
            
            # Ejecutar comando con timeout de 90 segundos y logging con tee
            tee_cmd = f"({' '.join(cmd)}) 2>&1 | tee {log_file}"
            
            logging.info(f"Iniciando conversión: {doc_path} -> {docx_path}")
            logging.debug(f"Comando: {tee_cmd}")
            
            try:
                # Usar subprocess con timeout
                result = subprocess.run(
                    tee_cmd,
                    shell=True,
                    timeout=CONVERSION_TIMEOUT_SECONDS,
                    capture_output=True,
                    text=True,
                    cwd=temp_path
                )
                
                # Verificar si la conversión fue exitosa
                if result.returncode == 0 and temp_docx.exists():
                    # Copiar archivo convertido al destino final
                    shutil.copy2(temp_docx, docx_path)
                    
                    # Leer y registrar el log de conversión
                    if log_file.exists():
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            log_content = f.read()
                            if log_content.strip():
                                logging.debug(f"Log de conversión para {doc_path.name}:\n{log_content}")
                    
                    logging.info(f"Conversión exitosa: {doc_path} -> {docx_path}")
                    return docx_path
                else:
                    # Log del error
                    error_msg = f"LibreOffice falló al convertir {doc_path}"
                    if result.stderr:
                        error_msg += f". Error: {result.stderr}"
                    if result.stdout:
                        error_msg += f". Output: {result.stdout}"
                    
                    logging.error(error_msg)
                    
                    # Leer log file si existe para más detalles
                    if log_file.exists():
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            log_content = f.read()
                            if log_content.strip():
                                logging.error(f"Log detallado de error para {doc_path.name}:\n{log_content}")
                    
                    return None
                    
            except subprocess.TimeoutExpired:
                # Archivo llegó a timeout - agregarlo a lista de problemáticos
                problematic_files.append(str(doc_path))
                logging.warning(f"TIMEOUT: El archivo {doc_path} excedió el límite de {CONVERSION_TIMEOUT_SECONDS} segundos y será marcado como problemático")
                
                # Leer log parcial si existe
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        log_content = f.read()
                        if log_content.strip():
                            logging.warning(f"Log parcial antes del timeout para {doc_path.name}:\n{log_content}")
                
                # Limpiar procesos LibreOffice huérfanos
                processes_killed = kill_libreoffice_processes()
                if processes_killed > 0:
                    logging.warning(f"Limpieza de procesos LibreOffice completada después del timeout de {doc_path.name}")
                
                return None
                
        except Exception as e:
            logging.error(f"Error durante la conversión de {doc_path}: {e}")
            return None

def get_problematic_files() -> List[str]:
    """
    Retorna la lista de archivos que llegaron a timeout durante la conversión
    
    Returns:
        Lista de rutas de archivos problemáticos
    """
    return problematic_files.copy()

def clear_problematic_files() -> None:
    """
    Limpia la lista de archivos problemáticos
    """
    global problematic_files
    problematic_files.clear()

def save_problematic_files_report(output_dir: Path) -> Optional[Path]:
    """
    Guarda un reporte de archivos problemáticos en un archivo
    
    Args:
        output_dir: Directorio donde guardar el reporte
        
    Returns:
        Ruta al archivo de reporte creado o None si no hay archivos problemáticos
    """
    global problematic_files
    
    if not problematic_files:
        return None
    
    report_path = output_dir / f"archivos_problematicos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Reporte de Archivos Problemáticos - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total de archivos que llegaron a timeout ({CONVERSION_TIMEOUT_SECONDS}s): {len(problematic_files)}\n\n")
            
            for i, file_path in enumerate(problematic_files, 1):
                f.write(f"{i:3d}. {file_path}\n")
        
        logging.info(f"Reporte de archivos problemáticos guardado en: {report_path}")
        return report_path
        
    except Exception as e:
        logging.error(f"Error al guardar reporte de archivos problemáticos: {e}")
        return None


def merge_docx_files(docx_files: List[Path], output_path: Path) -> bool:
    """
    Unifica múltiples archivos DOCX en un solo documento usando docxcompose
    
    Args:
        docx_files: Lista de rutas a archivos DOCX (pueden ser strings o Path objects)
        output_path: Ruta del archivo unificado de salida (puede ser string o Path object)
        
    Returns:
        True si la unificación fue exitosa, False en caso contrario
    """
    if not docx_files:
        logging.warning("No hay archivos DOCX para unificar.")
        return False
    
    # Convertir a Path objects y filtrar archivos que existen
    existing_files = []
    for f in docx_files:
        path_obj = Path(f) if isinstance(f, str) else f
        if path_obj.exists():
            existing_files.append(path_obj)
    
    if not existing_files:
        logging.warning("No se encontraron archivos DOCX válidos para unificar.")
        return False
    
    # Convertir output_path a Path object si es string
    output_path = Path(output_path) if isinstance(output_path, str) else output_path
    
    try:
        logging.info(f"Iniciando unificación de {len(existing_files)} archivos DOCX usando docxcompose...")
        
        # Crear documento maestro con el primer archivo
        master_doc = Document(str(existing_files[0].absolute()))
        composer = Composer(master_doc)
        
        logging.info(f"Documento maestro: {existing_files[0].name}")
        
        # Agregar los documentos restantes
        for i, docx_file in enumerate(existing_files[1:], 1):
            try:
                # Cargar documento a agregar
                doc_to_append = Document(str(docx_file.absolute()))
                
                # Agregar documento al compositor
                composer.append(doc_to_append)
                
                logging.info(f"Archivo {i}/{len(existing_files)-1} agregado: {docx_file.name}")
                
            except Exception as e:
                logging.error(f"Error agregando {docx_file.name} al documento unificado: {e}")
                continue
        
        # Crear directorio padre si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar documento unificado
        composer.save(str(output_path.absolute()))
        
        logging.info(f"Documento unificado creado exitosamente: {output_path.name}")
        return True
        
    except Exception as e:
        logging.error(f"Error durante la unificación de documentos con docxcompose: {e}")
        return False


def cleanup_temp_files(directory: Path, keep_unified: bool = True) -> int:
    """
    Limpia archivos temporales y residuales del directorio
    
    Args:
        directory: Directorio a limpiar
        keep_unified: Si mantener archivos unificados (True por defecto)
        
    Returns:
        Número de archivos eliminados
    """
    if not directory.exists():
        return 0
    
    deleted_count = 0
    
    try:
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                # Determinar si el archivo debe ser eliminado
                should_delete = False
                
                # Eliminar archivos DOC originales después de conversión
                if file_path.suffix.lower() == '.doc':
                    # Verificar si existe un archivo unificado en el directorio
                    # El archivo unificado tiene formato: DDMMYYYY_EDITION.docx
                    unified_files = list(directory.glob('*_MAT.docx')) + list(directory.glob('*_VES.docx'))
                    if unified_files:
                        # Si existe al menos un archivo unificado, eliminar todos los DOC
                        should_delete = True
                
                # Eliminar archivos DOCX individuales si existe archivo unificado
                elif file_path.suffix.lower() == '.docx' and keep_unified:
                    # Verificar si es un archivo individual (no unificado)
                    # El archivo unificado tiene formato: DDMMYYYY_EDITION.docx
                    # Los archivos individuales tienen nombres más largos o diferentes patrones
                    filename = file_path.stem
                    # Patrón para archivo unificado: 8 dígitos + _ + 3 letras (MAT/VES)
                    unified_pattern = re.match(r'^\d{8}_(MAT|VES)$', filename)
                    
                    # Si NO coincide con el patrón de archivo unificado, es un archivo individual
                    if not unified_pattern:
                        should_delete = True
                
                # Eliminar archivos temporales de Word
                elif file_path.name.startswith('~$') or file_path.suffix.lower() in ['.tmp', '.temp']:
                    should_delete = True
                
                if should_delete:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logging.info(f"Archivo eliminado: {file_path}")
                    except Exception as e:
                        logging.warning(f"No se pudo eliminar {file_path}: {e}")
        
        logging.info(f"Limpieza completada. {deleted_count} archivos eliminados.")
        return deleted_count
        
    except Exception as e:
        logging.error(f"Error durante la limpieza de {directory}: {e}")
        return deleted_count


def find_doc_files_in_edition_folders(base_dir: Path, date_str: str, edition: str) -> List[Path]:
    """
    Busca archivos DOC en las carpetas MAT/VES para una fecha específica
    
    Args:
        base_dir: Directorio base donde buscar
        date_str: Fecha en formato DD/MM/YYYY
        edition: Edición ('MAT' o 'VES')
        
    Returns:
        Lista de rutas a archivos DOC encontrados
    """
    # Construir ruta esperada: base_dir/YYYY/MM/DDMMYYYY/EDITION/
    date_parts = date_str.split('/')
    day, month, year = date_parts[0], date_parts[1], date_parts[2]
    
    edition_dir = base_dir / year / month / f"{day}{month}{year}" / edition
    
    doc_files = []
    
    if edition_dir.exists():
        # Buscar todos los archivos .doc en el directorio
        doc_files = list(edition_dir.glob('*.doc'))
        for doc_file in doc_files:
            logging.info(f"Archivo DOC encontrado: {doc_file}")
    else:
        logging.warning(f"Directorio no encontrado: {edition_dir}")
    
    return doc_files


def process_date_edition(base_dir: Path, date_str: str, edition: str) -> Optional[bool]:
    """
    Procesa una fecha y edición específica: convierte DOC a DOCX y unifica
    
    Args:
        base_dir: Directorio base donde buscar archivos
        date_str: Fecha en formato DD/MM/YYYY
        edition: Edición ('MAT' o 'VES')
        
    Returns:
        True si el procesamiento fue exitoso, False si falló, None si no hay archivos para procesar
        'TIMEOUT' si algún documento alcanzó timeout (día problemático)
    """
    logging.info(f"Procesando {date_str} - {edition}")
    
    # Buscar archivos DOC
    doc_files = find_doc_files_in_edition_folders(base_dir, date_str, edition)
    
    if not doc_files:
        logging.info(f"No se encontraron archivos DOC para {date_str} - {edition}")
        return None  
    
    logging.info(f"Encontrados {len(doc_files)} archivos DOC")
    
    # Fase 1: Convertir archivos DOC a DOCX
    docx_files = []
    initial_problematic_count = len(get_problematic_files())
    
    for doc_file in doc_files:
        docx_file = convert_doc_to_docx(doc_file)
        if docx_file and docx_file.exists():
            docx_files.append(docx_file)
            logging.info(f"Convertido: {doc_file.name} -> {docx_file.name}")
        else:
            # Verificar si el fallo fue por timeout
            current_problematic_count = len(get_problematic_files())
            if current_problematic_count > initial_problematic_count:
                logging.warning(f"TIMEOUT detectado en {doc_file.name}. Omitiendo día completo: {date_str} - {edition}")
                return 'TIMEOUT'
            logging.error(f"✗ Error convirtiendo: {doc_file.name}")
    
    if not docx_files:
        logging.error(f"No se pudieron convertir archivos DOC para {date_str} - {edition}")
        return False
    
    # Fase 2: Unificar documentos DOCX
    date_parts = date_str.split('/')
    day, month, year = date_parts[0], date_parts[1], date_parts[2]
    
    # Crear nombre del archivo unificado con formato: DDMMYYYY_EDITION.docx
    unified_filename = f"{day}{month}{year}_{edition}.docx"
    edition_dir = base_dir / year / month / f"{day}{month}{year}" / edition
    unified_path = edition_dir / unified_filename
    
    logging.info(f"Unificando {len(docx_files)} archivos DOCX...")
    
    if merge_docx_files(docx_files, unified_path):
        logging.info(f"Documento unificado creado: {unified_path}")
        
        # Fase 3: Limpieza automática de archivos residuales
        logging.info("Iniciando limpieza de archivos residuales...")
        deleted_count = cleanup_temp_files(edition_dir, keep_unified=True)
        logging.info(f"Limpieza completada: {deleted_count} archivos eliminados")
        
        return True
    else:
        logging.error(f"✗ Error creando documento unificado para {date_str} - {edition}")
        return False


def main(
    date: str = typer.Argument(..., help="Fecha (DD/MM/YYYY) o fecha de inicio para rango"),
    end_date: Optional[str] = typer.Argument(None, help="Fecha de fin (DD/MM/YYYY) - opcional para rango de fechas"),
    input_dir: str = typer.Option("./dof_word", help="Directorio de entrada donde buscar archivos DOC"),
    editions: str = typer.Option("both", help="Ediciones a procesar: 'mat', 'ves', o 'both'"),
    log_level: str = typer.Option("INFO", help="Nivel de logging: DEBUG, INFO, WARNING, ERROR")
):
    """
    Convierte archivos DOC a DOCX y los unifica usando docxcompose
    
    Este script busca archivos DOC en las carpetas MAT/VES, los convierte a DOCX
    usando LibreOffice en modo headless y luego los unifica en un solo documento por edición.
    
    Características principales:
    - Conversión con timeout de 90 segundos por archivo
    - Logging detallado con captura de salida de LibreOffice
    - Identificación automática de archivos problemáticos
    - Generación de reportes de archivos que lleguen a timeout
    - Unificación automática y limpieza de archivos temporales
    
    Ejemplos de uso:
    # Para una fecha específica:
    python dof_processor.py 02/01/2023 --editions both
    
    # Para un rango de fechas:
    python dof_processor.py 01/01/2023 31/01/2023 --editions both
    
    # Especificando directorio personalizado:
    python dof_processor.py 02/01/2023 --input-dir ./mi_carpeta --editions both
    """
    
    # Configurar logging
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    
    logging.basicConfig(
        level=log_levels.get(log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dof_processor.log'),
            logging.StreamHandler()
        ]
    )
    
    # Validar fechas
    try:
        start_dt = datetime.strptime(date, "%d/%m/%Y")
        
        # Si no se proporciona end_date, usar la misma fecha (fecha única)
        if end_date is None:
            end_dt = start_dt
        else:
            end_dt = datetime.strptime(end_date, "%d/%m/%Y")
            
    except ValueError:
        logging.error("Las fechas deben estar en formato DD/MM/YYYY")
        sys.exit(1)
    
    if start_dt > end_dt:
        logging.error("La fecha de inicio debe ser anterior a la fecha de fin")
        sys.exit(1)
    
    # Validar ediciones
    editions = editions.lower()
    if editions not in ['mat', 'ves', 'both']:
        logging.error("Las ediciones deben ser 'mat', 'ves', o 'both'")
        sys.exit(1)
    
    # Crear directorio de entrada
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"Directorio de entrada no encontrado: {input_path.absolute()}")
        sys.exit(1)
    
    logging.info(f"Iniciando conversión y unificación de archivos DOC")
    
    # Mostrar período apropiado según si es fecha única o rango
    if end_date is None:
        logging.info(f"Fecha: {date}")
    else:
        logging.info(f"Período: {date} - {end_date}")
    
    logging.info(f"Ediciones: {editions}")
    logging.info(f"Directorio de entrada: {input_path.absolute()}")
    logging.info("-" * 60)
    
    total_processed = 0
    successful_processed = 0
    problematic_days = []
    current_date = start_dt
    
    while current_date <= end_dt:
        date_str = current_date.strftime("%d/%m/%Y")
        
        # Determinar qué ediciones procesar
        editions_to_process = []
        if editions in ['mat', 'both']:
            editions_to_process.append('MAT')
        if editions in ['ves', 'both']:
            editions_to_process.append('VES')
        
        for edition in editions_to_process:
            result = process_date_edition(input_path, date_str, edition)
            
            if result == 'TIMEOUT':
                # Marcar solo esta edición como problemática, no todo el día
                problematic_days.append(f"{date_str} - {edition}")
                logging.warning(f"Edición marcada como problemática por timeout: {date_str} - {edition}")
                continue
                
            if result is not None:  # Solo contar si hay archivos para procesar
                total_processed += 1
                if result:  # True = exitoso, False = fallido
                    successful_processed += 1
        
        current_date += timedelta(days=1)
    
    logging.info("-" * 60)
    logging.info(f"Procesamiento completado.")
    logging.info(f"Total procesados: {total_processed}")
    logging.info(f"Exitosos: {successful_processed}")
    logging.info(f"Fallidos: {total_processed - successful_processed}")
    
    # Reporte de ediciones problemáticas por timeout
    if problematic_days:
        logging.warning(f"Se encontraron {len(problematic_days)} ediciones problemáticas por timeout:")
        for i, day_edition in enumerate(problematic_days, 1):
            logging.warning(f"  {i:3d}. {day_edition}")
    else:
        logging.info("No se encontraron ediciones problemáticas por timeout")
    
    # Generar reporte de archivos problemáticos si los hay
    problematic_list = get_problematic_files()
    if problematic_list:
        logging.warning(f"Se encontraron {len(problematic_list)} archivos problemáticos que llegaron a timeout ({CONVERSION_TIMEOUT_SECONDS}s)")
        
        # Guardar reporte en el directorio de entrada
        report_path = save_problematic_files_report(input_path)
        if report_path:
            logging.info(f"Reporte de archivos problemáticos guardado en: {report_path}")
        
        # Mostrar lista en el log
        logging.warning("Archivos problemáticos:")
        for i, file_path in enumerate(problematic_list, 1):
            logging.warning(f"  {i:3d}. {file_path}")
    else:
        logging.info("No se encontraron archivos problemáticos durante el procesamiento")
    
    # Limpiar lista para futuras ejecuciones
    clear_problematic_files()


if __name__ == "__main__":
    typer.run(main)