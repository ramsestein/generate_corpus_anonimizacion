"""
IO JSON Module - Conversión de datos a JSON
============================================

Este módulo contiene toda la lógica de lectura/escritura de datos:
- Carga de datos desde CSV/Excel/JSON
- Conversión al formato estándar de entidades
- Guardado de resultados en JSON
- Carga de documentos de texto

API Principal:
    from io_json import load_entities, save_results, load_document
    
    # Cargar entidades
    entities = load_entities("entidades.json")
    
    # Guardar resultados
    save_results(results, "output.json")
    
    # Cargar documento
    text = load_document("corpus/doc1.txt")
    
    # API de alto nivel
    from io_json import load_pipeline_input, save_pipeline_output
    
    entities, metadata = load_pipeline_input("input.json")
    save_pipeline_output(results, "output.json", metadata)
"""

from .loaders import (
    load_entities,
    load_document,
    load_ner_results,
    load_json,
    load_csv,
    load_excel,
)

from .savers import (
    save_results,
    save_json,
    save_pipeline_results,
    save_entities,
)

from .converters import (
    normalize_entity,
    convert_to_standard_format,
    EntityFormat,
)

from .api import (
    load_pipeline_input,
    load_documents_and_entities,
    save_pipeline_output,
    save_intermediate_results,
    convert_and_save,
    quick_load_json,
    quick_save_json,
    get_entity_summary,
    validate_entities,
)

__all__ = [
    # Cargadores
    "load_entities",
    "load_document",
    "load_ner_results",
    "load_json",
    "load_csv",
    "load_excel",
    # Guardadores
    "save_results",
    "save_json",
    "save_pipeline_results",
    "save_entities",
    # Conversores
    "normalize_entity",
    "convert_to_standard_format",
    "EntityFormat",
    # API de alto nivel
    "load_pipeline_input",
    "load_documents_and_entities",
    "save_pipeline_output",
    "save_intermediate_results",
    "convert_and_save",
    "quick_load_json",
    "quick_save_json",
    "get_entity_summary",
    "validate_entities",
]
