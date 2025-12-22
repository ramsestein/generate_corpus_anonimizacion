#!/usr/bin/env python3
import json

f = open('outputs/documentos_corpus_antiguo_completo.json', 'r', encoding='utf-8')
data = json.load(f)
docs = data['documents']

# Encontrar documento con entidades
print("🔍 Buscando documentos con entidades extraídas...")
doc_with_entities = None
entity_count_distribution = {}

for doc_id, doc in docs.items():
    count = doc['entity_count']
    if count not in entity_count_distribution:
        entity_count_distribution[count] = 0
    entity_count_distribution[count] += 1
    
    if count > 0 and doc_with_entities is None:
        doc_with_entities = (doc_id, doc)

print("\n📊 DISTRIBUCIÓN DE ENTIDADES POR DOCUMENTO:")
for count in sorted(entity_count_distribution.keys())[:10]:
    docs_with_count = entity_count_distribution[count]
    print(f"  {count} entidades: {docs_with_count} documentos")

if doc_with_entities:
    doc_id, doc = doc_with_entities
    print(f"\n✅ EJEMPLO DE DOCUMENTO CON ENTIDADES:")
    print(f"  ID: {doc_id}")
    print(f"  Entidades extraídas: {doc['entity_count']}")
    print(f"  Primeras 10 entidades:")
    for ent in doc['entities'][:10]:
        print(f"    - {ent}")
else:
    print("\n⚠️  No se encontraron documentos con entidades [** **]")

# Estadísticas totales
total_entities = sum(d['entity_count'] for d in docs.values())
docs_with_entities = sum(1 for d in docs.values() if d['entity_count'] > 0)
print(f"\n📈 ESTADÍSTICAS TOTALES:")
print(f"  Documentos con entidades: {docs_with_entities} / {len(docs)}")
print(f"  Total de entidades extraídas: {total_entities}")
if docs_with_entities > 0:
    print(f"  Promedio entidades/doc (con entidades): {total_entities / docs_with_entities:.1f}")
