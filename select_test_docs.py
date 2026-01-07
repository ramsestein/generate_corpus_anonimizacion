#!/usr/bin/env python3
"""
Selecciona los 50 documentos con mayor diversidad de entidades para testing.
"""

import json
from pathlib import Path
from collections import defaultdict
import csv

# Cargar detecciones del CSV
doc_entities = defaultdict(lambda: {'labels': set(), 'count': 0, 'entities': []})

csv_path = 'step6_validation_results/aws2/detecciones_detalladas.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doc_id = row.get('doc_id', row.get('document_id', ''))
        label = row.get('etiqueta', row.get('label', ''))
        text = row.get('texto_detectado', row.get('entity_text', ''))
        
        if doc_id:
            doc_entities[doc_id]['labels'].add(label)
            doc_entities[doc_id]['count'] += 1
            if len(doc_entities[doc_id]['entities']) < 5:
                doc_entities[doc_id]['entities'].append(f"{label}: {text[:30]}")

# Ordenar por diversidad (numero de etiquetas diferentes) y cantidad
ranked = sorted(
    doc_entities.items(), 
    key=lambda x: (-len(x[1]['labels']), -x[1]['count'])
)

print("TOP 50 DOCUMENTOS CON MAYOR DIVERSIDAD DE ENTIDADES")
print("=" * 70)
print()

selected = []
for i, (doc_id, info) in enumerate(ranked[:50], 1):
    selected.append(doc_id)
    labels_str = ', '.join(sorted(info['labels']))
    print(f"{i:2}. {doc_id}")
    print(f"    Etiquetas: {len(info['labels'])} ({labels_str})")
    print(f"    Entidades: {info['count']}")
    print()

# Guardar lista
output_file = 'test_docs_50.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(selected))

print(f"\nLista guardada en: {output_file}")
print(f"Total documentos seleccionados: {len(selected)}")
