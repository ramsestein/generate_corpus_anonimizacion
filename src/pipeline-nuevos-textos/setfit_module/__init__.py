"""
SetFit Module - Clasificador PII vs Ruido
==========================================

Este módulo contiene toda la lógica relacionada con el clasificador SetFit:
- Carga del modelo SetFit
- Inferencia y clasificación de entidades
- Filtros pre/post clasificación (ruido obvio, PII obvio, fragmentos)

API Principal:
    from setfit_module import run_setfit_filter, SetFitGatekeeper
    
    # Opción 1: Función simple
    results = run_setfit_filter(entities, document_text, config)
    
    # Opción 2: Clase con más control
    gatekeeper = SetFitGatekeeper(model_path="models/gatekeeper_setfit")
    result = gatekeeper.classify(entity_text, entity_label, context)
"""

from .gatekeeper import (
    SetFitGatekeeper,
    ClassificationResult,
    create_gatekeeper,
)

from .filters import (
    NoiseFilter,
    PIIDetector,
    FragmentDetector,
)

from .api import run_setfit_filter

__all__ = [
    # Clase principal
    "SetFitGatekeeper",
    "ClassificationResult",
    "create_gatekeeper",
    # Filtros
    "NoiseFilter",
    "PIIDetector",
    "FragmentDetector",
    # API simplificada
    "run_setfit_filter",
]
