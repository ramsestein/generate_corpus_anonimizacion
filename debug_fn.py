#!/usr/bin/env python3
"""Verificar si los FN no fueron detectados por NER o fueron filtrados."""
import json
import unicodedata

def normalize_text(text):
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return ' '.join(text.split())

# Cargar datos
with open('outputs/resultados_completo.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

with open('entidades-procesadas-para-metricas.json', 'r', encoding='utf-8') as f:
    entrada = json.load(f)

doc_id = '00039e89-4b9a-4a81-bb08-334cf20b482c'

print('=== En entidades-procesadas-para-metricas.json ===')
for e in entrada.get('entities', []):
    if e.get('doc_id') == doc_id and '87654321' in e.get('text', ''):
        print(f'  text="{e.get("text")}" label={e.get("label")}')

print()
print('=== En resultados_completo.json ===')
for d in results['decisions']:
    if d.get('document_id') == doc_id and '87654321' in d.get('entity_text', ''):
        print(f'  entity_text="{d.get("entity_text")}" label={d.get("label")}')
