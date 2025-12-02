"""
io_json/api.py - API de alto nivel para operaciones de entrada/salida
=====================================================================

Este módulo proporciona funciones de alto nivel para cargar y guardar
datos del pipeline de anonimización de forma sencilla.

FUNCIONES PRINCIPALES:
    - load_pipeline_input: Carga datos de entrada para el pipeline
    - save_pipeline_output: Guarda resultados del pipeline
    - convert_and_save: Convierte formato y guarda

USO:
    from io_json import (
        load_pipeline_input,
        save_pipeline_output,
        convert_and_save
    )
    
    # Cargar entrada
    entities, metadata = load_pipeline_input("input.json")
    
    # ... procesar ...
    
    # Guardar salida
    save_pipeline_output(results, "output.json", metadata)
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from .loaders import load_json, load_entities, load_ner_results
from .savers import save_json, save_entities, save_pipeline_results
from .converters import normalize_entity, convert_to_standard_format


logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def load_pipeline_input(
    input_path: str,
    format_type: str = "auto"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Carga datos de entrada para el pipeline.
    
    Detecta automáticamente el formato y normaliza las entidades.
    
    Args:
        input_path: Ruta al archivo de entrada
        format_type: Tipo de formato ("auto", "entities", "ner_results", "raw")
        
    Returns:
        Tuple de (entidades, metadata)
        
    Ejemplo:
        entities, metadata = load_pipeline_input("input.json")
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")
    
    # Cargar JSON
    raw_data = load_json(input_path)
    
    metadata = {}
    entities = []
    
    # Detectar formato
    if format_type == "auto":
        if isinstance(raw_data, list):
            # Lista directa de entidades
            entities = raw_data
        elif isinstance(raw_data, dict):
            if "entities" in raw_data:
                entities = raw_data["entities"]
                metadata = raw_data.get("metadata", {})
            elif "results" in raw_data:
                entities = raw_data["results"]
                metadata = raw_data.get("metadata", {})
            else:
                # NER results format
                entities = load_ner_results(input_path)
        else:
            raise ValueError(f"Formato no reconocido: {type(raw_data)}")
    
    elif format_type == "entities":
        entities = load_entities(input_path)
        
    elif format_type == "ner_results":
        entities = load_ner_results(input_path)
        
    elif format_type == "raw":
        if isinstance(raw_data, list):
            entities = raw_data
        else:
            entities = [raw_data]
    
    # Normalizar entidades
    normalized = []
    for entity in entities:
        try:
            normalized.append(normalize_entity(entity))
        except Exception as e:
            logger.warning(f"Error normalizando entidad: {e}")
            normalized.append(entity)
    
    logger.info(f"Cargadas {len(normalized)} entidades desde {input_path}")
    
    return normalized, metadata


def load_documents_and_entities(
    entities_path: str,
    documents_dir: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Carga entidades y sus documentos correspondientes.
    
    Args:
        entities_path: Ruta al archivo de entidades
        documents_dir: Directorio con documentos .txt (opcional)
        
    Returns:
        Tuple de (entidades, {doc_id: texto})
        
    Ejemplo:
        entities, docs = load_documents_and_entities(
            "entities.json",
            "documents/"
        )
    """
    entities, _ = load_pipeline_input(entities_path)
    
    documents = {}
    
    if documents_dir:
        docs_path = Path(documents_dir)
        
        if docs_path.exists():
            # Cargar documentos referenciados
            doc_ids = set(e.get("doc_id") for e in entities if e.get("doc_id"))
            
            for doc_id in doc_ids:
                # Buscar archivo .txt
                txt_file = docs_path / f"{doc_id}.txt"
                
                if txt_file.exists():
                    documents[doc_id] = txt_file.read_text(encoding="utf-8")
                else:
                    logger.warning(f"Documento no encontrado: {txt_file}")
    
    logger.info(f"Cargados {len(documents)} documentos")
    
    return entities, documents


# ============================================================================
# FUNCIONES DE GUARDADO
# ============================================================================

def save_pipeline_output(
    entities: List[Dict[str, Any]],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    stats: Optional[Dict[str, Any]] = None,
    include_filtered: bool = False
) -> str:
    """
    Guarda los resultados del pipeline.
    
    Args:
        entities: Lista de entidades procesadas
        output_path: Ruta del archivo de salida
        metadata: Metadatos adicionales
        stats: Estadísticas del pipeline
        include_filtered: Incluir entidades filtradas
        
    Returns:
        Ruta del archivo guardado
        
    Ejemplo:
        save_pipeline_output(results, "output.json", metadata={"version": "1.0"})
    """
    # Filtrar si es necesario
    if not include_filtered:
        to_save = [e for e in entities if e.get("decision") != "FILTER"]
    else:
        to_save = entities
    
    save_pipeline_results(to_save, output_path, metadata, stats)
    
    logger.info(f"Guardadas {len(to_save)} entidades en {output_path}")
    
    return output_path


def save_intermediate_results(
    entities: List[Dict[str, Any]],
    output_dir: str,
    stage: str,
    timestamp: Optional[str] = None
) -> str:
    """
    Guarda resultados intermedios de una etapa del pipeline.
    
    Args:
        entities: Lista de entidades
        output_dir: Directorio de salida
        stage: Nombre de la etapa ("setfit", "dict", "llm")
        timestamp: Timestamp para el nombre del archivo
        
    Returns:
        Ruta del archivo guardado
    """
    from datetime import datetime
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"intermediate_{stage}_{timestamp}.json"
    filepath = output_path / filename
    
    save_entities(entities, str(filepath))
    
    logger.info(f"Guardados resultados intermedios: {filepath}")
    
    return str(filepath)


# ============================================================================
# FUNCIONES DE CONVERSIÓN
# ============================================================================

def convert_and_save(
    input_path: str,
    output_path: str,
    target_format: str = "standard"
) -> str:
    """
    Convierte archivo de entidades a otro formato.
    
    Args:
        input_path: Ruta del archivo de entrada
        output_path: Ruta del archivo de salida
        target_format: Formato de salida ("standard", "meddocan")
        
    Returns:
        Ruta del archivo guardado
    """
    entities, metadata = load_pipeline_input(input_path)
    
    if target_format == "standard":
        converted = [normalize_entity(e) for e in entities]
    elif target_format == "meddocan":
        converted = convert_to_standard_format(entities, "meddocan")
    else:
        converted = entities
    
    save_pipeline_results(converted, output_path, metadata)
    
    logger.info(f"Convertidas {len(converted)} entidades a formato {target_format}")
    
    return output_path


def quick_load_json(path: str) -> Any:
    """
    Carga rápida de un archivo JSON.
    
    Args:
        path: Ruta al archivo
        
    Returns:
        Contenido del JSON
    """
    return load_json(path)


def quick_save_json(data: Any, path: str) -> str:
    """
    Guardado rápido de datos a JSON.
    
    Args:
        data: Datos a guardar
        path: Ruta del archivo
        
    Returns:
        Ruta del archivo guardado
    """
    save_json(data, path)
    return path


# ============================================================================
# UTILIDADES
# ============================================================================

def get_entity_summary(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Genera un resumen de las entidades.
    
    Args:
        entities: Lista de entidades
        
    Returns:
        Diccionario con estadísticas
    """
    from collections import Counter
    
    summary = {
        "total": len(entities),
        "by_label": dict(Counter(e.get("label", "unknown") for e in entities)),
        "by_decision": dict(Counter(e.get("decision", "none") for e in entities)),
        "by_source": dict(Counter(e.get("decision_source", "none") for e in entities)),
    }
    
    return summary


def validate_entities(entities: List[Dict[str, Any]]) -> Tuple[List[Dict], List[str]]:
    """
    Valida una lista de entidades.
    
    Args:
        entities: Lista de entidades
        
    Returns:
        Tuple de (entidades válidas, lista de errores)
    """
    valid = []
    errors = []
    
    required_fields = ["text", "label"]
    
    for i, entity in enumerate(entities):
        missing = [f for f in required_fields if f not in entity]
        
        if missing:
            errors.append(f"Entidad {i}: campos faltantes {missing}")
        else:
            valid.append(entity)
    
    return valid, errors
