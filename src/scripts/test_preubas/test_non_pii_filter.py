#!/usr/bin/env python3
"""
Script de prueba para verificar el filtro de etiquetas no-PII.
"""

# Simular el filtro sin importar todo el módulo
NON_PII_LABELS = {
    "FAMILIARES_SUJETO_ASISTENCIA",
    "PROFESION",
    "OTROS_SUJETO_ASISTENCIA",
}

def filter_non_pii_entities(entities):
    """Filtro de etiquetas no-PII"""
    if not entities:
        return []
    
    filtered_entities = []
    discarded_count = 0
    
    for entity in entities:
        label = entity.get('entity_group') or entity.get('label')
        
        if label and label in NON_PII_LABELS:
            discarded_count += 1
            continue
        
        filtered_entities.append(entity)
    
    return filtered_entities

# ============================================================================
# CASOS DE PRUEBA
# ============================================================================

def test_filter_basic():
    """Prueba básica: filtrar etiquetas no-PII"""
    entities = [
        {"entity_group": "NOMBRE_SUJETO_ASISTENCIA", "word": "Juan", "score": 0.99},
        {"entity_group": "FAMILIARES_SUJETO_ASISTENCIA", "word": "familia", "score": 0.99},
        {"entity_group": "PROFESION", "word": "médico", "score": 0.95},
        {"entity_group": "FECHAS", "word": "12/05/2023", "score": 0.98},
    ]
    
    filtered = filter_non_pii_entities(entities)
    
    print("TEST 1: Filtrado básico")
    print(f"  Entrada: {len(entities)} entidades")
    print(f"  Salida: {len(filtered)} entidades")
    print(f"  Esperado: 2 (NOMBRE y FECHAS)")
    
    assert len(filtered) == 2, f"Error: esperado 2, obtenido {len(filtered)}"
    assert filtered[0]["entity_group"] == "NOMBRE_SUJETO_ASISTENCIA"
    assert filtered[1]["entity_group"] == "FECHAS"
    print("  ✓ PASSED\n")

def test_filter_empty():
    """Prueba con lista vacía"""
    entities = []
    filtered = filter_non_pii_entities(entities)
    
    print("TEST 2: Lista vacía")
    print(f"  Entrada: {len(entities)} entidades")
    print(f"  Salida: {len(filtered)} entidades")
    
    assert len(filtered) == 0
    print("  ✓ PASSED\n")

def test_filter_all_non_pii():
    """Prueba donde todas son no-PII"""
    entities = [
        {"entity_group": "FAMILIARES_SUJETO_ASISTENCIA", "word": "padre", "score": 0.99},
        {"entity_group": "PROFESION", "word": "enfermera", "score": 0.95},
        {"entity_group": "OTROS_SUJETO_ASISTENCIA", "word": "otro", "score": 0.88},
    ]
    
    filtered = filter_non_pii_entities(entities)
    
    print("TEST 3: Todas no-PII (deben descartarse todas)")
    print(f"  Entrada: {len(entities)} entidades")
    print(f"  Salida: {len(filtered)} entidades")
    print(f"  Esperado: 0")
    
    assert len(filtered) == 0, f"Error: esperado 0, obtenido {len(filtered)}"
    print("  ✓ PASSED\n")

def test_filter_all_pii():
    """Prueba donde todas son PII válidas"""
    entities = [
        {"entity_group": "NOMBRE_SUJETO_ASISTENCIA", "word": "María García", "score": 0.99},
        {"entity_group": "ID_SUJETO_ASISTENCIA", "word": "12345678A", "score": 0.98},
        {"entity_group": "FECHAS", "word": "15/03/2024", "score": 0.97},
        {"entity_group": "TERRITORIO", "word": "Madrid", "score": 0.95},
    ]
    
    filtered = filter_non_pii_entities(entities)
    
    print("TEST 4: Todas PII válidas (deben mantenerse todas)")
    print(f"  Entrada: {len(entities)} entidades")
    print(f"  Salida: {len(filtered)} entidades")
    print(f"  Esperado: 4")
    
    assert len(filtered) == 4, f"Error: esperado 4, obtenido {len(filtered)}"
    print("  ✓ PASSED\n")

def test_filter_label_field():
    """Prueba con campo 'label' en lugar de 'entity_group'"""
    entities = [
        {"label": "NOMBRE_SUJETO_ASISTENCIA", "word": "Ana", "score": 0.99},
        {"label": "FAMILIARES_SUJETO_ASISTENCIA", "word": "hermano", "score": 0.99},
        {"label": "HOSPITAL", "word": "Hospital Central", "score": 0.96},
    ]
    
    filtered = filter_non_pii_entities(entities)
    
    print("TEST 5: Usando campo 'label' en lugar de 'entity_group'")
    print(f"  Entrada: {len(entities)} entidades")
    print(f"  Salida: {len(filtered)} entidades")
    print(f"  Esperado: 2 (NOMBRE y HOSPITAL)")
    
    assert len(filtered) == 2, f"Error: esperado 2, obtenido {len(filtered)}"
    assert filtered[0]["label"] == "NOMBRE_SUJETO_ASISTENCIA"
    assert filtered[1]["label"] == "HOSPITAL"
    print("  ✓ PASSED\n")

def test_real_scenario():
    """Escenario real simulado del paso 6"""
    entities = [
        {"entity_group": "NOMBRE_SUJETO_ASISTENCIA", "word": "Juan Pérez", "score": 0.99, "start": 10, "end": 20},
        {"entity_group": "FAMILIARES_SUJETO_ASISTENCIA", "word": "familia", "score": 1.00, "start": 150, "end": 157},
        {"entity_group": "FECHAS", "word": "05/11/2023", "score": 0.98, "start": 200, "end": 210},
        {"entity_group": "PROFESION", "word": "cardiólogo", "score": 0.96, "start": 300, "end": 310},
        {"entity_group": "FAMILIARES_SUJETO_ASISTENCIA", "word": "familiar", "score": 0.99, "start": 400, "end": 408},
        {"entity_group": "HOSPITAL", "word": "Hospital Universitario", "score": 0.97, "start": 500, "end": 522},
        {"entity_group": "OTROS_SUJETO_ASISTENCIA", "word": "paciente", "score": 0.92, "start": 600, "end": 608},
    ]
    
    filtered = filter_non_pii_entities(entities)
    
    print("TEST 6: Escenario real simulado")
    print(f"  Entrada: {len(entities)} entidades")
    print(f"    - 2 FAMILIARES_SUJETO_ASISTENCIA (filtradas)")
    print(f"    - 1 PROFESION (filtrada)")
    print(f"    - 1 OTROS_SUJETO_ASISTENCIA (filtrada)")
    print(f"    - 3 PII reales (mantenidas)")
    print(f"  Salida: {len(filtered)} entidades")
    print(f"  Esperado: 3")
    
    assert len(filtered) == 3, f"Error: esperado 3, obtenido {len(filtered)}"
    
    # Verificar que solo quedan las PII reales
    labels = [e.get("entity_group") for e in filtered]
    expected_labels = ["NOMBRE_SUJETO_ASISTENCIA", "FECHAS", "HOSPITAL"]
    assert labels == expected_labels, f"Error en etiquetas: {labels} vs {expected_labels}"
    
    print("  ✓ PASSED\n")

# ============================================================================
# EJECUTAR TODAS LAS PRUEBAS
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PRUEBAS DEL FILTRO POST-PROCESADO DE ETIQUETAS NO-PII")
    print("="*70)
    print()
    
    try:
        test_filter_basic()
        test_filter_empty()
        test_filter_all_non_pii()
        test_filter_all_pii()
        test_filter_label_field()
        test_real_scenario()
        
        print("="*70)
        print("✓ TODAS LAS PRUEBAS PASARON CORRECTAMENTE")
        print("="*70)
        print()
        print("El filtro está listo para integrarse en el pipeline.")
        print("Las etiquetas filtradas son:")
        for label in sorted(NON_PII_LABELS):
            print(f"  - {label}")
        
    except AssertionError as e:
        print(f"\n✗ ERROR EN PRUEBA: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
