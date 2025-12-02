"""
LLM Judge Module - Validación con LLM
======================================

Este módulo contiene toda la lógica relacionada con la validación LLM:
- Plantillas de prompts (system y user)
- Construcción de prompts con contexto
- Llamadas al LLM (Ollama)
- Parseo de respuestas

API Principal:
    from llm_judge import run_llm_judge, LLMJudge
    
    # Opción 1: Función simple
    results = run_llm_judge(entities, document_text, config)
    
    # Opción 2: Clase con más control
    judge = LLMJudge(model="gemma3:270m", rules_path="guias-anotacion.json")
    result = judge.evaluate(entity, context)
"""

from .prompts import (
    PromptBuilder,
    PROMPT_TEMPLATES,
    load_entity_rules,
    get_entity_rules_for_label,
)

from .judge import LLMJudge, JudgeResult

from .api import run_llm_judge

__all__ = [
    # Clases principales
    "LLMJudge",
    "JudgeResult",
    "PromptBuilder",
    # Constantes
    "PROMPT_TEMPLATES",
    # Funciones de utilidad
    "load_entity_rules",
    "get_entity_rules_for_label",
    # API simplificada
    "run_llm_judge",
]
