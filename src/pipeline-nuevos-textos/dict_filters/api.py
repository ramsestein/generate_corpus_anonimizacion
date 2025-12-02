#!/usr/bin/env python3
"""
API simplificada para el módulo de filtros de diccionario.

Proporciona una función de alto nivel para aplicar filtros
de whitelist/blacklist sin necesidad de instanciar clases.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .list_loader import ListLoader
from .filter import DictFilter, FilterDecision

logger = logging.getLogger(__name__)


def apply_dict_filters(
    entities: List[Dict[str, Any]],
    document_text: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Aplica filtros de diccionario (whitelist/blacklist) a las entidades.
    
    Esta es la función principal de la API del módulo dict_filters.
    
    Args:
        entities: Lista de entidades con formato:
            [
                {
                    "entity_text": "paracetamol",
                    "label": "MEDICAMENTO",
                    ...
                },
                ...
            ]
        document_text: Texto completo del documento (no usado, para compatibilidad)
        config: Configuración opcional:
            {
                "json_whitelist_paths": ["data/hospitales.json", ...],
                "json_blacklist_paths": ["data/medicamentos.json", ...],
                "csv_base_path": "LISTAS/",
                "cie10_path": "LISTAS/cie10.xls",
                "min_length_per_label": {"NUMERO_IDENTIF": 3},
                "ignore_single_char": True,
                "ignore_numeric_only": False
            }
    
    Returns:
        Lista de resultados con formato:
            [
                {
                    "entity_text": "paracetamol",
                    "label": "MEDICAMENTO",
                    "filter_decision": "FORCE_IGNORE",
                    "matched_list": "blacklist",
                    "matched_term": "paracetamol",
                    "decision": "FILTER",  # KEEP, FILTER, o ESCALATE
                    ...
                },
                ...
            ]
    """
    if not entities:
        return []
    
    # Configuración por defecto
    default_config = {
        "json_whitelist_paths": [],
        "json_blacklist_paths": [],
        "csv_base_path": None,
        "cie10_path": None,
        "lowercase_whitelist": False,
        "lowercase_blacklist": True,
        "min_length_per_label": {},
        "ignore_single_char": True,
        "ignore_numeric_only": False,
    }
    
    if config:
        default_config.update(config)
    
    # Crear ListLoader
    list_loader = ListLoader(
        json_whitelist_paths=default_config["json_whitelist_paths"],
        json_blacklist_paths=default_config["json_blacklist_paths"],
        csv_base_path=default_config["csv_base_path"],
        cie10_path=default_config["cie10_path"],
        lowercase_whitelist=default_config["lowercase_whitelist"],
        lowercase_blacklist=default_config["lowercase_blacklist"],
    )
    
    # Crear filtro
    dict_filter = DictFilter(
        list_loader=list_loader,
        min_length_per_label=default_config["min_length_per_label"],
        ignore_single_char=default_config["ignore_single_char"],
        ignore_numeric_only=default_config["ignore_numeric_only"],
    )
    
    # Procesar entidades
    results = []
    for entity in entities:
        # Normalizar campos
        entity_text = entity.get('entity_text', entity.get('text', ''))
        entity_label = entity.get('label', entity.get('entity_label', ''))
        
        # Evaluar
        filter_result = dict_filter.evaluate(entity_text, entity_label)
        
        # Determinar decisión final
        if filter_result.decision == FilterDecision.FORCE_ANONYMIZE:
            decision = "KEEP"
        elif filter_result.decision == FilterDecision.FORCE_IGNORE:
            decision = "FILTER"
        else:
            decision = "ESCALATE"
        
        # Construir resultado
        result = {
            **entity,  # Mantener datos originales
            "filter_decision": filter_result.decision.name,
            "filter_reason": filter_result.reason,
            "matched_list": filter_result.matched_list,
            "matched_term": filter_result.matched_term,
            "heuristic_applied": filter_result.heuristic_applied,
            "decision": decision,
        }
        
        results.append(result)
    
    # Log estadísticas
    stats = dict_filter.get_stats()
    total = stats['total_evaluated']
    if total > 0:
        logger.info(
            f"DictFilter processed {total} entities: "
            f"{stats['force_anonymize']} ANONYMIZE, "
            f"{stats['force_ignore']} IGNORE, "
            f"{stats['escalate']} ESCALATE"
        )
    
    return results


def filter_with_whitelist(
    entities: List[Dict[str, Any]],
    whitelist_paths: List[str],
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Aplica solo filtro de whitelist.
    
    Args:
        entities: Lista de entidades
        whitelist_paths: Rutas a archivos de whitelist
        config: Configuración adicional
    
    Returns:
        Lista de entidades que matchean whitelist
    """
    cfg = config or {}
    cfg["json_whitelist_paths"] = whitelist_paths
    cfg["json_blacklist_paths"] = []
    cfg["cie10_path"] = None
    
    all_results = apply_dict_filters(entities, None, cfg)
    return [r for r in all_results if r["decision"] == "KEEP"]


def filter_with_blacklist(
    entities: List[Dict[str, Any]],
    blacklist_paths: List[str],
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Aplica solo filtro de blacklist.
    
    Args:
        entities: Lista de entidades
        blacklist_paths: Rutas a archivos de blacklist
        config: Configuración adicional
    
    Returns:
        Lista de entidades que matchean blacklist
    """
    cfg = config or {}
    cfg["json_whitelist_paths"] = []
    cfg["json_blacklist_paths"] = blacklist_paths
    cfg["cie10_path"] = None
    
    all_results = apply_dict_filters(entities, None, cfg)
    return [r for r in all_results if r["decision"] == "FILTER"]
