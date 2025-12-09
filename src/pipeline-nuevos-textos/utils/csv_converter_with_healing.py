#!/usr/bin/env python3
"""
Integración de Token Healing en csv_converter.py

Este módulo proporciona funciones para mejorar la calidad de los datos
al convertir CSV a JSON, reparando las fronteras de las entidades.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

# Intentar importar desde el mismo directorio
try:
    from token_healing import fix_entity_boundaries, batch_fix_entity_boundaries
except ImportError:
    # Fallback si la importación falla
    from .token_healing import fix_entity_boundaries, batch_fix_entity_boundaries

logger = logging.getLogger(__name__)


def apply_token_healing_to_csv_entities(
    entities: List[Dict[str, Any]],
    document_texts: Dict[str, str],
    smart_mode: bool = True,
    max_expansion: int = 50,
    verbose: bool = False
) -> tuple:
    """
    Aplica token healing (reparación de fronteras) a entidades extraídas de CSV.
    
    Args:
        entities: Lista de entidades con campos: doc_id, text, start, end, label, etc.
        document_texts: Dict {doc_id: texto_completo}
        smart_mode: Si True, evita expansiones innecesarias (respeta puntuación)
        max_expansion: Máximo de caracteres a expandir en cada dirección
        verbose: Si True, imprime información de reparaciones
        
    Returns:
        Tupla (entities_healed, stats)
    """
    
    healed_entities = []
    stats = {
        "total_processed": len(entities),
        "total_healed": 0,
        "errors": 0,
        "skipped_no_doc": 0,
        "by_label": {},
        "examples": []
    }
    
    for entity in entities:
        doc_id = entity.get("doc_id")
        
        # Si no hay documento, saltar
        if not doc_id or doc_id not in document_texts:
            stats["skipped_no_doc"] += 1
            healed_entities.append(entity)
            continue
        
        original_text = document_texts[doc_id]
        
        try:
            # Aplicar token healing
            healed = fix_entity_boundaries(
                entity,
                original_text,
                max_expansion_chars=max_expansion,
                smart_mode=smart_mode
            )
            
            # Estadísticas
            if healed.get("boundary_fixed", False):
                stats["total_healed"] += 1
                
                label = healed.get("label", "N/A")
                if label not in stats["by_label"]:
                    stats["by_label"][label] = 0
                stats["by_label"][label] += 1
                
                # Guardar ejemplo
                if len(stats["examples"]) < 10:
                    stats["examples"].append({
                        "label": label,
                        "original_text": entity.get("text"),
                        "healed_text": healed.get("text"),
                        "original_span": [entity.get("start"), entity.get("end")],
                        "healed_span": [healed.get("start"), healed.get("end")],
                        "doc_id": doc_id
                    })
                
                if verbose:
                    print(f"  ✓ {label}: '{entity.get('text')}' → '{healed.get('text')}'")
            
            healed_entities.append(healed)
            
        except Exception as e:
            stats["errors"] += 1
            if verbose:
                print(f"  ✗ Error en {doc_id}: {e}")
            healed_entities.append(entity)
    
    stats["pct_healed"] = (stats["total_healed"] / stats["total_processed"] * 100) if stats["total_processed"] > 0 else 0
    
    return healed_entities, stats


def load_document_texts(
    documents_dir: Path,
    doc_ids_to_load: List[str] = None
) -> Dict[str, str]:
    """
    Carga textos de documentos desde archivos .txt en un directorio.
    
    Args:
        documents_dir: Ruta al directorio con archivos .txt
        doc_ids_to_load: Lista de doc_ids específicos (None = cargar todos)
        
    Returns:
        Dict {doc_id: texto_completo}
    """
    
    documents = {}
    
    if not documents_dir.exists():
        logger.warning(f"Directorio no encontrado: {documents_dir}")
        return documents
    
    for txt_file in documents_dir.glob("*.txt"):
        doc_id = txt_file.stem
        
        # Si hay lista de IDs específicos, filtrar
        if doc_ids_to_load and doc_id not in doc_ids_to_load:
            continue
        
        try:
            text = txt_file.read_text(encoding="utf-8")
            documents[doc_id] = text
        except Exception as e:
            logger.warning(f"Error leyendo {txt_file}: {e}")
    
    logger.info(f"Cargados {len(documents)} documentos desde {documents_dir}")
    return documents


def integrate_token_healing_in_pipeline(
    entities: List[Dict[str, Any]],
    corpus_dir: Path = None,
    smart_mode: bool = True,
    verbose: bool = False
) -> tuple:
    """
    Integración completa de token healing en el pipeline.
    
    Busca automáticamente los documentos y aplica healing.
    
    Args:
        entities: Entidades con doc_id y coordenadas
        corpus_dir: Directorio base del corpus (None = buscar automáticamente)
        smart_mode: Usar expansión inteligente
        verbose: Mostrar detalles
        
    Returns:
        Tupla (entities_healed, stats)
    """
    
    # Buscar directorio de documentos
    if corpus_dir is None:
        # Intentar rutas comunes
        possible_paths = [
            Path("corpus/documents"),
            Path("corpus/ANTIGUO/documents"),
            Path("documents"),
            Path("../corpus/documents"),
        ]
        
        for path in possible_paths:
            if path.exists():
                corpus_dir = path
                break
        
        if corpus_dir is None:
            logger.warning("No se encontró directorio de documentos")
            return entities, {"error": "No document directory found"}
    
    # Extraer doc_ids únicos
    doc_ids = set(e.get("doc_id") for e in entities if e.get("doc_id"))
    
    # Cargar documentos
    logger.info(f"Cargando {len(doc_ids)} documentos...")
    documents = load_document_texts(corpus_dir, list(doc_ids))
    
    if not documents:
        logger.warning("No se cargaron documentos")
        return entities, {"error": "No documents loaded"}
    
    # Aplicar token healing
    logger.info(f"Aplicando token healing a {len(entities)} entidades...")
    healed_entities, stats = apply_token_healing_to_csv_entities(
        entities,
        documents,
        smart_mode=smart_mode,
        verbose=verbose
    )
    
    logger.info(f"Token healing completado:")
    logger.info(f"  - Total procesadas: {stats['total_processed']}")
    logger.info(f"  - Total reparadas: {stats['total_healed']} ({stats['pct_healed']:.1f}%)")
    logger.info(f"  - Errores: {stats['errors']}")
    logger.info(f"  - Sin documento: {stats['skipped_no_doc']}")
    
    return healed_entities, stats


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TOKEN HEALING - Integración en CSV Converter")
    print("=" * 80)
    
    # Simulación de entidades con coordenadas cortadas
    sample_entities = [
        {
            "doc_id": "doc_001",
            "label": "NOMBRE",
            "text": "uan",
            "start": 8,
            "end": 11,
            "confidence": 0.95
        },
        {
            "doc_id": "doc_001",
            "label": "INSTITUCION",
            "text": "ospital",
            "start": 37,
            "end": 44,
            "confidence": 0.89
        }
    ]
    
    # Simulación de textos de documentos
    sample_documents = {
        "doc_001": """El Dr. Juan García trabaja en el Hospital San Carlos.
        Su teléfono es +34 91 876 54 32 y vive en Paseo Castellana 100."""
    }
    
    print("\n1. Entidades originales (con coordenadas cortadas):")
    for e in sample_entities:
        doc_text = sample_documents[e["doc_id"]]
        current_text = doc_text[e["start"]:e["end"]]
        print(f"   - {e['label']:20} '{e['text']:15}' [coords: {e['start']}:{e['end']}] "
              f"(actual: '{current_text}')")
    
    print("\n2. Aplicando token healing...")
    healed_entities, stats = apply_token_healing_to_csv_entities(
        sample_entities,
        sample_documents,
        smart_mode=True,
        verbose=True
    )
    
    print("\n3. Entidades reparadas:")
    for e in healed_entities:
        doc_text = sample_documents[e["doc_id"]]
        actual_text = doc_text[e["start"]:e["end"]]
        print(f"   - {e['label']:20} '{e['text']:15}' [coords: {e['start']}:{e['end']}] "
              f"(actual: '{actual_text}')")
    
    print("\n4. Estadísticas:")
    print(f"   Total reparadas: {stats['total_healed']}/{stats['total_processed']} "
          f"({stats['pct_healed']:.1f}%)")
    
    print("\n5. Reparaciones detalladas:")
    for i, ex in enumerate(stats['examples'], 1):
        print(f"\n   {i}. {ex['label']}")
        print(f"      Original: '{ex['original_text']}' {ex['original_span']}")
        print(f"      Reparada: '{ex['healed_text']}' {ex['healed_span']}")
