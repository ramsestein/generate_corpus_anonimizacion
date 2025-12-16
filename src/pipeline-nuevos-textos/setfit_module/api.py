#!/usr/bin/env python3
"""
API simplificada para el módulo SetFit.

Proporciona una función de alto nivel para filtrar entidades
sin necesidad de instanciar clases directamente.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .gatekeeper import SetFitGatekeeper, ClassificationResult

logger = logging.getLogger(__name__)


def run_setfit_filter(
    entities: List[Dict[str, Any]],
    document_text: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Filtra entidades usando el clasificador SetFit mejorado.
    
    Esta es la función principal de la API del módulo SetFit.
    Toma una lista de entidades y devuelve las clasificaciones.
    
    Args:
        entities: Lista de entidades con formato:
            [
                {
                    "text": "Juan García",
                    "label": "NOMBRE_SUJETO_ASISTENCIA",
                    "start": 10,
                    "end": 21,
                    "context": "El paciente Juan García acudió..."
                },
                ...
            ]
        document_text: Texto completo del documento (opcional)
        config: Configuración opcional:
            {
                "model_path": "models/gatekeeper_setfit",
                "confidence_threshold": 0.75,
                "enable_noise_filter": False,
                "enable_pii_detector": False,
                "enable_fragment_filter": False,
                "enable_low_confidence_filter": False
            }
    
    Returns:
        Lista de resultados con formato:
            [
                {
                    "entity_text": "Juan García",
                    "label": "NOMBRE_SUJETO_ASISTENCIA",
                    "start": 10,
                    "end": 21,
                    "is_pii": True,
                    "confidence": 0.92,
                    "classification_method": "setfit",
                    "classification": "PII",  # PII = es PII real, RUIDO = no es PII
                    "classification_source": "setfit",  # Trazabilidad
                    "details": {...}
                },
                ...
            ]
    """
    if not entities:
        return []
    
    # Configuración por defecto - SIMPLIFICADA
    # SetFit clasifica TODO por igual (sin filtros de ruido predefinidos)
    default_config = {
        "model_path": "models/gatekeeper_setfit",
        "confidence_threshold": 0.85,  # Subido para mejorar precisión
        "enable_pii_detector": False,  # Detector regex deshabilitado
        "enable_low_confidence_filter": True,  # Filtrar baja confianza=True
    }
    
    if config:
        default_config.update(config)
    
    # Crear gatekeeper (simplificado)
    gatekeeper = SetFitGatekeeper(
        model_path=default_config["model_path"],
        confidence_threshold=default_config["confidence_threshold"],
        enable_pii_detector=default_config.get("enable_pii_detector", False),
        enable_low_confidence_filter=default_config.get("enable_low_confidence_filter", True),
    )
    
    # Extraer todos los textos de entidades para detección de fragmentos
    all_entity_texts = [
        e.get('text', e.get('entity_text', ''))
        for e in entities
    ]
    
    # Procesar cada entidad
    results = []
    for entity in entities:
        # Normalizar campos de entrada
        entity_text = entity.get('text', entity.get('entity_text', ''))
        entity_label = entity.get('label', entity.get('entity_label', ''))
        context = entity.get('context', entity.get('sentence_context', ''))
        start = entity.get('start', entity.get('posicion_inicio', -1))
        end = entity.get('end', entity.get('posicion_fin', -1))
        doc_id = entity.get('document_id', entity.get('doc_id', ''))
        
        # Clasificar
        classification = gatekeeper.classify(
            entity_text=entity_text,
            entity_label=entity_label,
            sentence_context=context,
            full_document=document_text,
            other_entities=all_entity_texts,
        )
        
        # Construir resultado
        # SetFit actúa como filtro binario de primera línea:
        # - PII: Modelo cree que es información sensible → va directo a salida final
        # - RUIDO: Modelo cree que NO es PII → pasa a dict_filters/LLM para posible rescate
        result = {
            # Datos originales
            "document_id": doc_id,
            "entity_text": entity_text,
            "label": entity_label,
            "start": start,
            "end": end,
            "context": context,
            # Clasificación SetFit (binaria)
            "is_pii": classification.is_pii,
            "confidence": classification.confidence,
            "classification_method": classification.classification_method,
            "classification": "PII" if classification.is_pii else "RUIDO",
            "classification_source": "setfit",  # Trazabilidad: quién clasificó
            "details": classification.details,
        }
        
        results.append(result)
    
    # Log estadísticas
    stats = gatekeeper.get_statistics()
    logger.info(
        f"SetFit processed {stats['total_processed']} entities: "
        f"{stats.get('percentages', {}).get('noise_filtered', 0):.1f}% noise, "
        f"{stats.get('percentages', {}).get('pii_detected', 0):.1f}% obvious PII"
    )
    
    return results


def filter_pii_entities(
    entities: List[Dict[str, Any]],
    document_text: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Filtra y devuelve SOLO las entidades clasificadas como PII.
    
    Es un wrapper sobre run_setfit_filter que devuelve solo
    las entidades con decision="KEEP".
    
    Args:
        entities: Lista de entidades
        document_text: Texto del documento (opcional)
        config: Configuración (opcional)
    
    Returns:
        Lista de entidades clasificadas como PII
    """
    all_results = run_setfit_filter(entities, document_text, config)
    return [r for r in all_results if r["decision"] == "KEEP"]


def filter_noise_entities(
    entities: List[Dict[str, Any]],
    document_text: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Filtra y devuelve SOLO las entidades clasificadas como ruido.
    
    Es un wrapper sobre run_setfit_filter que devuelve solo
    las entidades con decision="FILTER".
    
    Args:
        entities: Lista de entidades
        document_text: Texto del documento (opcional)
        config: Configuración (opcional)
    
    Returns:
        Lista de entidades clasificadas como ruido
    """
    all_results = run_setfit_filter(entities, document_text, config)
    return [r for r in all_results if r["decision"] == "FILTER"]
