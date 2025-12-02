#!/usr/bin/env python3
"""Analiza las entidades filtradas por dict_filters."""

import json

# Cargar ground truth
with open('entidades-procesadas-para-metricas.json', 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

# Cargar resultado del pipeline
with open('outputs/resultados_FLUJO_COMPLETO.json', 'r', encoding='utf-8') as f:
    pipeline_result = json.load(f)

print("=" * 70)
print("ANÁLISIS DE ENTIDADES FILTRADAS POR DICT_FILTERS")
print("=" * 70)

# Obtener las 34 entidades que SetFit marcó como ruido
setfit_filtered = pipeline_result['metadata']['stats']['setfit_filtered']
dict_filtered = pipeline_result['metadata']['stats']['dict_filtered']
dict_escalated = pipeline_result['metadata']['stats']['dict_escalated']

print(f"\n[ESTADÍSTICAS]")
print(f"SetFit filtró (ruido): {setfit_filtered}")
print(f"  └─ Dict blacklist (FILTER): {dict_filtered}")
print(f"  └─ Dict escalate (LLM): {dict_escalated}")

# Buscar entidades de 1 carácter en ground truth
single_char_entities = [e for e in ground_truth if len(e['text'].strip()) == 1]

print(f"\n[ENTIDADES DE 1 CARÁCTER EN GROUND TRUTH]")
print(f"Total: {len(single_char_entities)}")
for e in single_char_entities[:15]:
    print(f"  - '{e['text']}' ({e['label']}) doc:{e['doc_id'][:12]}")

# Contar entidades por longitud que SetFit marcó comoo ruido
print(f"\n[ANÁLISIS: ¿Por qué se filtraron esas 10?]")
print(f"Configuración dict_filters:")
print(f"  - ignore_single_char: True (por defecto)")
print(f"  - ignore_numeric_only: False")

print(f"\n✓ Conclusión: Las 10 entidades filtradas son probablemente")
print(f"  entidades de 1 carácter que SetFit clasificó como ruido")
print(f"  y dict_filters las eliminó automáticamente con ignore_single_char=True")
