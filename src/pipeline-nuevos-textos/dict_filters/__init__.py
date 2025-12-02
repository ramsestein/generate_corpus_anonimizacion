"""
Dict Filters Module - Listas Negras y Blancas
==============================================

Este módulo contiene toda la lógica relacionada con filtros de diccionario:
- Carga de listas desde JSON, CSV y Excel
- Búsqueda en whitelist (forzar anonimización)
- Búsqueda en blacklist (ignorar entidades)
- Búsqueda en CIE10 (términos médicos)

API Principal:
    from dict_filters import apply_dict_filters, ListLoader, DictFilter
    
    # Opción 1: Función simple
    results = apply_dict_filters(entities, document_text, config)
    
    # Opción 2: Clases con más control
    loader = ListLoader(whitelist_paths=[...], blacklist_paths=[...])
    filter = DictFilter(loader)
    result = filter.evaluate(entity_text, entity_label)
"""

from .list_loader import ListLoader
from .filter import DictFilter, FilterDecision, FilterResult
from .api import apply_dict_filters

__all__ = [
    # Clases principales
    "ListLoader",
    "DictFilter",
    "FilterDecision",
    "FilterResult",
    # API simplificada
    "apply_dict_filters",
]
