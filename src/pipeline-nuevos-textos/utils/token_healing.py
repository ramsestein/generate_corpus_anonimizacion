#!/usr/bin/env python3
"""
Token Healing - Reparación de fronteras de entidades.

Expande las coordenadas (start, end) de una entidad para capturar palabras completas
en lugar de fragmentos. Soluciona el problema de tokenización donde las coordenadas
cortan palabras a la mitad.

Ejemplo:
    Input:  start=5, end=8, text="ola" (debería ser "Hola")
    Output: start=4, end=9, text="Hola"
"""

from typing import Dict, Any, Tuple


def fix_entity_boundaries(
    entity: Dict[str, Any],
    original_text: str,
    max_expansion_chars: int = 50,
    smart_mode: bool = True
) -> Dict[str, Any]:
    """
    Repara las fronteras (start, end) de una entidad expandiendo a palabra completa.
    
    Expande hacia la izquierda y derecha hasta encontrar espacios en blanco o límites
    del documento. Tiene límite de expansión máxima para evitar capturas excesivas.
    
    Args:
        entity: Diccionario con la entidad:
            {
                "text": "ola",
                "entity": "NOMBRE",
                "start": 5,
                "end": 8,
                "doc_id": "doc_123",
                ...
            }
        original_text: Texto completo del documento del cual se extrajo la entidad
        max_expansion_chars: Número máximo de caracteres a expandir en cada dirección
        smart_mode: Si True, intenta no expandir más allá de ciertos límites inteligentes
        
    Returns:
        Diccionario con la entidad actualizada:
            {
                "text": "Hola",
                "entity": "NOMBRE",
                "start": 4,          # Expandido hacia izquierda
                "end": 8,            # Original (ya estaba en límite)
                "doc_id": "doc_123",
                "boundary_fixed": True,
                "original_start": 5,
                "original_end": 8,
                ...
            }
    
    Raises:
        ValueError: Si start o end están fuera de rango
    """
    
    # Hacer una copia para no modificar el original
    fixed_entity = entity.copy()
    
    # Extraer coordenadas
    start = entity.get("start")
    end = entity.get("end")
    
    # Validación de coordenadas
    if start is None or end is None:
        return fixed_entity
    
    start = int(start)
    end = int(end)
    
    # Validar límites
    if start < 0 or end > len(original_text) or start >= end:
        raise ValueError(
            f"Coordenadas inválidas: start={start}, end={end}, "
            f"text_length={len(original_text)}"
        )
    
    # Guardar coordenadas originales para trazabilidad
    original_start = start
    original_end = end
    
    # ========================================================================
    # EXPANSIÓN A IZQUIERDA
    # ========================================================================
    # Expandir hacia atrás hasta encontrar un límite (espacio, salto línea, inicio)
    new_start = start
    expansion_count = 0
    
    while new_start > 0 and expansion_count < max_expansion_chars:
        # Mirar el carácter anterior
        prev_char = original_text[new_start - 1]
        
        # Detener si encontramos un delimitador
        if prev_char.isspace():
            # Espacio, tabulación, salto de línea, etc.
            break
        
        # En modo smart: si encontramos puntuación de límite, parar
        if smart_mode and prev_char in '.,;:!?([{':
            break
        
        # Expandir un carácter más hacia la izquierda
        new_start -= 1
        expansion_count += 1
    
    # ========================================================================
    # EXPANSIÓN A DERECHA
    # ========================================================================
    # Expandir hacia adelante hasta encontrar un límite (espacio, salto línea, fin)
    new_end = end
    expansion_count = 0
    
    while new_end < len(original_text) and expansion_count < max_expansion_chars:
        # Mirar el carácter actual en la posición end
        curr_char = original_text[new_end]
        
        # Detener si encontramos un delimitador
        if curr_char.isspace():
            # Espacio, tabulación, salto de línea, etc.
            break
        
        # En modo smart: si encontramos puntuación de límite, parar
        if smart_mode and curr_char in '.,;:!?)]}':
            break
        
        # Expandir un carácter más hacia la derecha
        new_end += 1
        expansion_count += 1
    
    # ========================================================================
    # EXTRACCIÓN DEL NUEVO TEXTO
    # ========================================================================
    new_text = original_text[new_start:new_end].strip()
    
    # ========================================================================
    # ACTUALIZACIÓN DE LA ENTIDAD
    # ========================================================================
    fixed_entity["start"] = new_start
    fixed_entity["end"] = new_end
    fixed_entity["text"] = new_text
    fixed_entity["entity_text"] = new_text  # Para compatibilidad con otros formatos
    
    # Metadatos de reparación
    fixed_entity["boundary_fixed"] = (original_start != new_start) or (original_end != new_end)
    fixed_entity["original_start"] = original_start
    fixed_entity["original_end"] = original_end
    fixed_entity["original_text"] = entity.get("text", "")
    
    return fixed_entity


def batch_fix_entity_boundaries(
    entities: list,
    original_text: str,
    verbose: bool = False
) -> Tuple[list, dict]:
    """
    Repara las fronteras de múltiples entidades.
    
    Args:
        entities: Lista de diccionarios de entidades
        original_text: Texto completo del documento
        verbose: Si True, imprime información sobre las reparaciones
        
    Returns:
        Tupla (entidades_reparadas, estadísticas)
    """
    
    fixed_entities = []
    stats = {
        "total_processed": len(entities),
        "total_fixed": 0,
        "examples": [],
    }
    
    for i, entity in enumerate(entities):
        try:
            fixed = fix_entity_boundaries(entity, original_text)
            fixed_entities.append(fixed)
            
            if fixed.get("boundary_fixed", False):
                stats["total_fixed"] += 1
                
                # Guardar ejemplos de reparaciones
                if len(stats["examples"]) < 10:
                    stats["examples"].append({
                        "original": {
                            "text": entity.get("text"),
                            "start": entity.get("start"),
                            "end": entity.get("end"),
                        },
                        "fixed": {
                            "text": fixed.get("text"),
                            "start": fixed.get("start"),
                            "end": fixed.get("end"),
                        },
                        "label": entity.get("entity", entity.get("label", "N/A")),
                    })
                
                if verbose:
                    print(f"  ✓ Reparada: '{entity.get('text')}' → '{fixed.get('text')}'")
        
        except Exception as e:
            if verbose:
                print(f"  ✗ Error en entidad {i}: {e}")
            # En caso de error, mantener la entidad original
            fixed_entities.append(entity)
    
    stats["pct_fixed"] = (stats["total_fixed"] / len(entities) * 100) if entities else 0
    
    return fixed_entities, stats


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Ejemplo 1: Entidad simple cortada a la mitad
    print("=" * 80)
    print("EJEMPLO 1: Entidad cortada a la mitad")
    print("=" * 80)
    
    original_doc = "Mi nombre es Hola mundo, y vivo en Barcelona."
    
    # Entidad detectada incorrectamente: solo "ola" en lugar de "Hola"
    dirty_entity = {
        "text": "ola",
        "entity": "NOMBRE",
        "start": 15,
        "end": 18,
        "doc_id": "doc_001"
    }
    
    fixed = fix_entity_boundaries(dirty_entity, original_doc)
    
    print(f"\nTexto original: '{original_doc}'")
    print(f"\nEntidad sucia:")
    print(f"  text:   '{dirty_entity['text']}'")
    print(f"  start:  {dirty_entity['start']}")
    print(f"  end:    {dirty_entity['end']}")
    print(f"  snippet: '{original_doc[dirty_entity['start']:dirty_entity['end']]}'")
    
    print(f"\nEntidad reparada:")
    print(f"  text:   '{fixed['text']}'")
    print(f"  start:  {fixed['start']}")
    print(f"  end:    {fixed['end']}")
    print(f"  snippet: '{original_doc[fixed['start']:fixed['end']]}'")
    print(f"  boundary_fixed: {fixed['boundary_fixed']}")
    
    # Ejemplo 2: Múltiples entidades
    print(f"\n{'=' * 80}")
    print("EJEMPLO 2: Múltiples entidades")
    print("=" * 80)
    
    original_doc2 = """El Dr. Juan García trabaja en el Hospital San Carlos.
    Su teléfono es +34 91 876 54 32 y vive en Paseo Castellana 100."""
    
    dirty_entities = [
        {"text": "uan", "entity": "NOMBRE", "start": 8, "end": 11},  # "Juan" cortado
        {"text": "arcía", "entity": "NOMBRE", "start": 16, "end": 21},  # "García" cortado
        {"text": "ospital", "entity": "INSTITUCION", "start": 37, "end": 44},  # "Hospital" cortado
        {"text": "4 91", "entity": "NUMERO_TELEFONO", "start": 88, "end": 92},  # "+34 91" cortado
    ]
    
    print(f"\nTexto original:\n{original_doc2}\n")
    print(f"Reparando {len(dirty_entities)} entidades...")
    
    fixed_entities, stats = batch_fix_entity_boundaries(dirty_entities, original_doc2, verbose=True)
    
    print(f"\n{'─' * 80}")
    print(f"ESTADÍSTICAS:")
    print(f"  Total procesadas: {stats['total_processed']}")
    print(f"  Total reparadas:  {stats['total_fixed']} ({stats['pct_fixed']:.1f}%)")
    
    print(f"\n{'─' * 80}")
    print(f"EJEMPLOS DE REPARACIONES:")
    for i, example in enumerate(stats["examples"], 1):
        print(f"\n  {i}. {example['label']}")
        print(f"     Original: '{example['original']['text']}' "
              f"[{example['original']['start']}:{example['original']['end']}]")
        print(f"     Reparada: '{example['fixed']['text']}' "
              f"[{example['fixed']['start']}:{example['fixed']['end']}]")
    
    # Ejemplo 3: Entidad ya correcta (sin reparación)
    print(f"\n{'=' * 80}")
    print("EJEMPLO 3: Entidad ya correcta (sin reparación)")
    print("=" * 80)
    
    original_doc3 = "Mi nombre es Laura Pérez García."
    correct_entity = {
        "text": "Laura",
        "entity": "NOMBRE",
        "start": 11,
        "end": 16,
    }
    
    fixed = fix_entity_boundaries(correct_entity, original_doc3)
    
    print(f"\nTexto: '{original_doc3}'")
    print(f"Entidad: '{fixed['text']}' [{fixed['start']}:{fixed['end']}]")
    print(f"¿Necesitaba reparación? {fixed['boundary_fixed']}")
    print(f"Verificación: '{original_doc3[fixed['start']:fixed['end']]}'")
