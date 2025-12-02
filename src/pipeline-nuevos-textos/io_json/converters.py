#!/usr/bin/env python3
"""
Conversores de formato - Normalización de entidades.

Convierte entre diferentes formatos de entidades usados
en el pipeline y sistemas externos.
"""

import logging
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class EntityFormat(Enum):
    """Formatos de entidades soportados."""
    STANDARD = "standard"      # Formato interno del pipeline
    MEDDOCAN = "meddocan"      # Formato de salida MEDDOCAN
    NER_JSON = "ner_json"      # Formato genérico NER
    SPACY = "spacy"            # Formato spaCy
    BRAT = "brat"              # Formato BRAT


# Mapeo de campos por formato
FIELD_MAPPINGS = {
    EntityFormat.STANDARD: {
        "text": ["entity_text", "text", "keyword"],
        "label": ["label", "entity_label", "etiqueta", "ner_label"],
        "start": ["start", "posicion_inicio", "start_offset"],
        "end": ["end", "posicion_fin", "end_offset"],
        "document_id": ["document_id", "doc_id", "file_id"],
        "confidence": ["confidence", "confianza", "score"],
        "context": ["context", "sentence_context", "contexto"],
    },
    EntityFormat.MEDDOCAN: {
        "text": ["texto_detectado", "text"],
        "label": ["etiqueta", "label"],
        "start": ["posicion_inicio", "start"],
        "end": ["posicion_fin", "end"],
        "document_id": ["doc_id", "document_id"],
        "confidence": ["confianza", "confidence"],
    },
}


def normalize_entity(
    entity: Dict[str, Any],
    source_format: EntityFormat = EntityFormat.NER_JSON
) -> Dict[str, Any]:
    """
    Normaliza una entidad al formato estándar del pipeline.
    
    Formato estándar:
    {
        "document_id": "doc1",
        "entity_text": "Juan García",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "start": 10,
        "end": 21,
        "confidence": 0.95,
        "context": "El paciente Juan García acudió..."
    }
    
    Args:
        entity: Entidad en formato original
        source_format: Formato de origen
        
    Returns:
        Entidad normalizada
    """
    normalized = {}
    
    # Obtener mapeo
    mapping = FIELD_MAPPINGS.get(source_format, FIELD_MAPPINGS[EntityFormat.STANDARD])
    
    # Procesar cada campo estándar
    for standard_field, possible_keys in FIELD_MAPPINGS[EntityFormat.STANDARD].items():
        value = None
        
        # Buscar en las claves posibles
        for key in possible_keys:
            if key in entity:
                value = entity[key]
                break
        
        # Mapear al nombre estándar
        if standard_field == "text":
            normalized["entity_text"] = value or ""
        elif standard_field == "label":
            normalized["label"] = value or ""
        elif standard_field == "start":
            normalized["start"] = int(value) if value is not None else -1
        elif standard_field == "end":
            normalized["end"] = int(value) if value is not None else -1
        elif standard_field == "document_id":
            normalized["document_id"] = value or ""
        elif standard_field == "confidence":
            normalized["confidence"] = float(value) if value is not None else 0.0
        elif standard_field == "context":
            normalized["context"] = value or ""
    
    # Calcular end si no está presente
    if normalized["end"] == -1 and normalized["start"] >= 0:
        normalized["end"] = normalized["start"] + len(normalized["entity_text"])
    
    # Copiar campos adicionales que no están en el mapeo
    for key, value in entity.items():
        if key not in normalized:
            normalized[key] = value
    
    return normalized


def convert_to_standard_format(
    entities: List[Dict[str, Any]],
    source_format: EntityFormat = EntityFormat.NER_JSON
) -> List[Dict[str, Any]]:
    """
    Convierte una lista de entidades al formato estándar.
    
    Args:
        entities: Lista de entidades en formato original
        source_format: Formato de origen
        
    Returns:
        Lista de entidades normalizadas
    """
    return [normalize_entity(e, source_format) for e in entities]


def convert_from_meddocan(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convierte entidades desde formato MEDDOCAN al estándar.
    
    Args:
        entities: Lista de entidades MEDDOCAN
        
    Returns:
        Lista de entidades en formato estándar
    """
    return convert_to_standard_format(entities, EntityFormat.MEDDOCAN)


def convert_to_output_format(
    entities: List[Dict[str, Any]],
    include_fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Convierte entidades al formato de salida del pipeline.
    
    Formato de salida:
    {
        "document_id": "doc1",
        "entity_text": "Juan García",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "start": 10,
        "end": 21,
        "decision": "KEEP",  # KEEP, FILTER, o ESCALATE
        "decision_source": "setfit",  # setfit, dict_filter, llm
        "confidence": 0.95,
    }
    
    Args:
        entities: Lista de entidades procesadas
        include_fields: Campos a incluir en la salida (None = campos estándar)
        
    Returns:
        Lista de entidades en formato de salida
    """
    default_fields = [
        "document_id", "entity_text", "label", "start", "end",
        "decision", "decision_source", "confidence"
    ]
    
    fields = include_fields or default_fields
    
    output = []
    for entity in entities:
        out_entity = {}
        for field in fields:
            if field in entity:
                out_entity[field] = entity[field]
        output.append(out_entity)
    
    return output


def group_entities_by_document(
    entities: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Agrupa entidades por documento.
    
    Args:
        entities: Lista de entidades
        
    Returns:
        Diccionario {document_id: [entidades]}
    """
    grouped = {}
    
    for entity in entities:
        doc_id = entity.get("document_id", "unknown")
        if doc_id not in grouped:
            grouped[doc_id] = []
        grouped[doc_id].append(entity)
    
    # Ordenar por posición
    for doc_id in grouped:
        grouped[doc_id].sort(key=lambda e: e.get("start", 0))
    
    return grouped
