#!/usr/bin/env python3
"""
unify_json_entities.py - Unificador de Archivos JSON
=====================================================

Lee todos los archivos JSON de una carpeta origen y crea un diccionario maestro
donde la clave es el nombre del archivo (sin extensión) y el valor es su contenido.

USO:
    python unify_json_entities.py
    python unify_json_entities.py --source corpus/ANTIGUO/entidades --output dataset_unificado.json

OUTPUT:
    {
        "doc_001": [ {...}, {...} ],
        "doc_002": [ {...}, {...} ],
        ...
    }
"""

import json
import argparse
import logging
from pathlib import Path
from glob import glob

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)-8s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "corpus" / "ANTIGUO" / "entidades"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset_unificado.json"


def unify_json_files(source_dir: Path, output_path: Path) -> dict:
    """
    Lee todos los archivos JSON de una carpeta y los unifica en un diccionario.
    
    Args:
        source_dir: Ruta a la carpeta con archivos JSON
        output_path: Ruta donde guardar el archivo unificado
    
    Returns:
        Diccionario con los datos unificados
    """
    
    if not source_dir.exists():
        logger.error(f"La carpeta de origen no existe: {source_dir}")
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    logger.info(f"Buscando archivos JSON en: {source_dir}")
    
    # Buscar todos los archivos .json
    json_files = sorted(source_dir.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No se encontraron archivos JSON en {source_dir}")
        return {}
    
    # Limitar a los primeros 50 archivos
    json_files = json_files[:50]
    
    logger.info(f"Procesando primeros 50 archivos JSON de {len(json_files)} encontrados")
    
    # Diccionario maestro
    master_dict = {}
    errors = []
    
    for json_file in json_files:
        file_key = json_file.stem  # Nombre sin extensión
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            master_dict[file_key] = content
            logger.debug(f"✓ {file_key}")
            
        except json.JSONDecodeError as e:
            error_msg = f"Error JSON en {json_file}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"Error leyendo {json_file}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Guardar diccionario maestro
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(master_dict, f, indent=4, ensure_ascii=False)
    
    logger.info(f"\n✅ Archivos procesados: {len(master_dict)}/{len(json_files)}")
    logger.info(f"📁 Archivo unificado guardado en: {output_path}")
    
    if errors:
        logger.warning(f"⚠️  Se encontraron {len(errors)} errores:")
        for error in errors:
            logger.warning(f"  - {error}")
    
    return master_dict


def main():
    parser = argparse.ArgumentParser(
        description="Unifica múltiples archivos JSON en un diccionario maestro"
    )
    parser.add_argument(
        '--source', '-s',
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Carpeta origen (default: {DEFAULT_SOURCE})"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Archivo de salida (default: {DEFAULT_OUTPUT})"
    )
    
    args = parser.parse_args()
    
    master_dict = unify_json_files(args.source, args.output)
    
    # Mostrar resumen
    if master_dict:
        print(f"\n📊 RESUMEN:")
        print(f"  Documentos procesados: {len(master_dict)}")
        if len(master_dict) <= 10:
            print(f"  Claves: {list(master_dict.keys())}")
        else:
            print(f"  Primeras 5 claves: {list(master_dict.keys())[:5]}")
            print(f"  Últimas 5 claves: {list(master_dict.keys())[-5:]}")
        
        # Estadísticas
        total_items = sum(
            len(v) if isinstance(v, list) else 1 
            for v in master_dict.values()
        )
        print(f"  Total de items: {total_items}")
    
    return 0


if __name__ == "__main__":
    exit(main())
