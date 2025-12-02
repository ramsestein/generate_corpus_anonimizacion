#!/usr/bin/env python3
"""
Prueba rápida de evaluate_pipeline.py con subset de datos
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_pipeline import load_ground_truth, normalize_text
import json

# Test 1: Cargar subset de ground truth (solo 100 archivos)
print("=" * 70)
print("TEST 1: Cargando subset de ground truth (100 archivos)")
print("=" * 70)

gt_dir = Path('../../corpus/ANTIGUO/entidades')
print(f"\nDirectorio GT: {gt_dir}")
print(f"Total archivos disponibles: {len(list(gt_dir.glob('*.json')))}")

# Cargar solo los primeros 100 archivos para prueba
from collections import defaultdict

ground_truth = defaultdict(set)
for i, json_file in enumerate(list(gt_dir.glob('*.json'))[:100]):
    doc_id = json_file.stem
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Formato corpus/ANTIGUO: {"id": "...", "data": [...]}
    if isinstance(data, dict) and 'data' in data:
        entities = data['data']
    else:
        entities = []
    
    for ent in entities:
        text = ent.get('text', '')
        label = ent.get('entity', '')
        
        if text and label:
            text_norm = normalize_text(text)
            ground_truth[doc_id].add((text_norm, label))

print(f"\n✅ Cargados {len(ground_truth)} documentos")
print(f"✅ Total entidades: {sum(len(v) for v in ground_truth.values())}")

# Mostrar ejemplo
sample_doc = list(ground_truth.keys())[0]
print(f"\n📄 Documento ejemplo: {sample_doc}")
print(f"   Entidades: {len(ground_truth[sample_doc])}")
for i, (text, label) in enumerate(list(ground_truth[sample_doc])[:3]):
    print(f"   - [{label}] '{text}'")

# Test 2: Normalización
print("\n" + "=" * 70)
print("TEST 2: Normalización de texto")
print("=" * 70)

test_cases = [
    "España",
    "María José",
    "NÚMERO  123",
    "  espacios   múltiples  ",
    "Niño",
]

for test in test_cases:
    norm = normalize_text(test)
    print(f"'{test}' → '{norm}'")

# Test 3: Cargar archivo de resultados
print("\n" + "=" * 70)
print("TEST 3: Verificar archivo de resultados")
print("=" * 70)

results_file = Path('../../outputs/resultados_completo.json')
if results_file.exists():
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extraer decisiones
    decisions = results.get('decisions', results.get('results', []))
    print(f"\n✅ Archivo de resultados encontrado")
    print(f"✅ Total decisiones: {len(decisions)}")
    
    # Mostrar ejemplo
    if decisions:
        example = decisions[0]
        print(f"\n📄 Decisión ejemplo:")
        print(f"   doc_id: {example.get('document_id', 'N/A')}")
        print(f"   text: {example.get('entity_text', 'N/A')}")
        print(f"   label: {example.get('label', 'N/A')}")
        print(f"   decision: {example.get('final_decision', 'N/A')}")
else:
    print(f"\n❌ Archivo de resultados NO encontrado: {results_file}")

print("\n" + "=" * 70)
print("✅ TODOS LOS TESTS COMPLETADOS")
print("=" * 70)
