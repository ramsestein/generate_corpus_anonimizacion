#!/usr/bin/env python3
"""
API simplificada para el módulo LLM Judge.

Proporciona una función de alto nivel para evaluar entidades
con el LLM sin necesidad de instanciar clases.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .judge import LLMJudge

logger = logging.getLogger(__name__)


def run_llm_judge(
    entities: List[Dict[str, Any]],
    document_text: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Evalúa entidades usando el juez LLM.
    
    Esta es la función principal de la API del módulo LLM Judge.
    Solo debe recibir entidades que requieren validación (ESCALATE del filtro).
    
    Args:
        entities: Lista de entidades con formato:
            [
                {
                    "entity_text": "Juan García",
                    "label": "NOMBRE_SUJETO_ASISTENCIA",
                    "context": "El paciente Juan García acudió...",
                    ...
                },
                ...
            ]
        document_text: Texto completo del documento (opcional, para contexto)
        config: Configuración opcional:
            {
                "model": "gemma3:270m",
                "rules_path": "guias-anotacion.json",
                "template_name": "default",
                "timeout": 120,
                "max_retries": 2,
                "debug": False
            }
    
    Returns:
        Lista de resultados con formato:
            [
                {
                    "entity_text": "Juan García",
                    "label": "NOMBRE_SUJETO_ASISTENCIA",
                    "llm_decision": True,  # True = válido, False = falso positivo
                    "llm_confidence": 1.0,
                    "llm_response": "TRUE",
                    "llm_status": "success",
                    "decision": "KEEP",  # KEEP = es PII, FILTER = no es PII
                    ...
                },
                ...
            ]
    """
    if not entities:
        return []
    
    # Configuración por defecto
    default_config = {
        "model": "gemma3:270m",
        "rules_path": None,
        "template_name": "default",
        "timeout": 120,
        "max_retries": 2,
        "debug": False,
    }
    
    if config:
        default_config.update(config)
    
    # Buscar archivo de reglas si no se especifica
    if default_config["rules_path"] is None:
        # Buscar en ubicaciones comunes
        possible_paths = [
            "guias-anotacion.json",
            Path(__file__).parent.parent.parent.parent / "guias-anotacion.json",
        ]
        for path in possible_paths:
            if Path(path).exists():
                default_config["rules_path"] = str(path)
                break
    
    # Crear juez
    judge = LLMJudge(
        model=default_config["model"],
        rules_path=default_config["rules_path"],
        template_name=default_config["template_name"],
        timeout=default_config["timeout"],
        max_retries=default_config["max_retries"],
    )
    
    # Procesar entidades
    results = []
    total = len(entities)
    
    for i, entity in enumerate(entities, 1):
        # Normalizar campos
        entity_text = entity.get('entity_text', entity.get('text', ''))
        entity_label = entity.get('label', entity.get('entity_label', ''))
        context = entity.get('context', entity.get('sentence_context', ''))
        
        # Si no hay contexto y tenemos el documento, extraerlo
        if not context and document_text:
            start = entity.get('start', 0)
            end = entity.get('end', len(entity_text))
            left = max(0, start - 80)
            right = min(len(document_text), end + 80)
            context = document_text[left:right]
        
        if default_config["debug"]:
            logger.info(f"[{i}/{total}] Evaluating: {entity_text} ({entity_label})")
        
        # Evaluar
        judge_result = judge.evaluate(
            entity_text=entity_text,
            entity_label=entity_label,
            context=context,
            debug=default_config["debug"]
        )
        
        # Determinar decisión final
        if judge_result.is_valid is True:
            decision = "KEEP"
        elif judge_result.is_valid is False:
            decision = "FILTER"
        else:
            # En caso de error, ser conservador y mantener
            decision = "KEEP"
        
        # Construir resultado
        result = {
            **entity,  # Mantener datos originales
            "llm_decision": judge_result.is_valid,
            "llm_confidence": judge_result.confidence,
            "llm_response": judge_result.raw_response,
            "llm_status": judge_result.status,
            "llm_processing_time": judge_result.processing_time,
            "decision": decision,
        }
        
        results.append(result)
    
    # Log estadísticas
    stats = judge.get_stats()
    logger.info(
        f"LLM Judge processed {stats['total_evaluated']} entities: "
        f"{stats['valid']} valid, {stats['invalid']} invalid, "
        f"{stats['errors']} errors "
        f"(avg time: {stats.get('avg_time', 0):.2f}s)"
    )
    
    return results


def evaluate_single_entity(
    entity_text: str,
    entity_label: str,
    context: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evalúa una única entidad con el LLM.
    
    Función de conveniencia para evaluar una sola entidad.
    
    Args:
        entity_text: Texto de la entidad
        entity_label: Etiqueta de la entidad
        context: Contexto donde aparece
        config: Configuración opcional
    
    Returns:
        Resultado de la evaluación
    """
    entities = [{
        "entity_text": entity_text,
        "label": entity_label,
        "context": context,
    }]
    
    results = run_llm_judge(entities, None, config)
    return results[0] if results else {}
