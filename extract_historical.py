import json

# Cargar archivo con metadata
with open('entidades-procesadas-anteriores.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extraer solo las entidades
entities = data.get('entities', [])

print(f"Total entidades: {len(entities)}")
print(f"Con manual_correction TRUE: {sum(1 for e in entities if e.get('manual_correction') == 'TRUE')}")
print(f"Con manual_correction FALSE: {sum(1 for e in entities if e.get('manual_correction') == 'FALSE')}")

# Adaptar formato para el pipeline (necesita keyword, label, document_id)
adapted_entities = []
for e in entities:
    adapted = {
        'document_id': e.get('doc_id', ''),
        'keyword': e.get('text', ''),
        'label': e.get('label', ''),
        'start': e.get('start', -1),
        'end': e.get('end', -1),
        'manual_correction': e.get('manual_correction', 'FALSE'),
        'confidence': e.get('confidence', 0.0),
        'model': e.get('model', '')
    }
    adapted_entities.append(adapted)

# Guardar adaptado
with open('outputs/historical_entities_adapted.json', 'w', encoding='utf-8') as f:
    json.dump(adapted_entities, f, indent=2, ensure_ascii=False)

print(f"\nAdaptadas {len(adapted_entities)} entidades")
print("Guardado en: outputs/historical_entities_adapted.json")
