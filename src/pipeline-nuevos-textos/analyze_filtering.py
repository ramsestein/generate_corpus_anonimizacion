#!/usr/bin/env python3
"""
Análisis del filtrado del pipeline.
Compara entrada vs salida para ver qué fue filtrado.
"""
import json
import unicodedata
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent

def normalize_text(text):
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return ' '.join(text.split())

# Cargar entrada
with open(PROJECT_ROOT / 'entidades-procesadas-para-metricas.json', 'r', encoding='utf-8') as f:
    entrada = json.load(f)

# Cargar resultados
with open(PROJECT_ROOT / 'outputs' / 'resultados_completo.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Extraer doc_ids
doc_ids = set()
for d in results.get('decisions', []):
    if d.get('document_id'):
        doc_ids.add(d['document_id'])

# Cargar GT
gt_by_doc = {}
for doc_id in doc_ids:
    gt_file = PROJECT_ROOT / 'corpus' / 'ANTIGUO' / 'entidades' / f'{doc_id}.json'
    if gt_file.exists():
        with open(gt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        gt_by_doc[doc_id] = {normalize_text(e['text']) for e in data.get('data', [])}

# Construir set de entidades KEEP (normalizadas)
keep_set = set()
for d in results.get('decisions', []):
    doc_id = d.get('document_id', '')
    text_norm = normalize_text(d.get('entity_text', ''))
    label = d.get('label', '')
    keep_set.add((doc_id, text_norm, label))

# Analizar entidades de entrada
kept_in_gt = 0  # KEEP que está en GT
kept_not_in_gt = 0  # KEEP que NO está en GT
filtered_in_gt = 0  # FILTRADO que SÍ está en GT (malo!)
filtered_not_in_gt = 0  # FILTRADO que NO está en GT (bueno!)

filtered_in_gt_examples = []
filtered_not_in_gt_examples = []

for e in entrada.get('entities', []):
    doc_id = e.get('doc_id', '')
    text = e.get('text', '')
    text_norm = normalize_text(text)
    label = e.get('label', '')
    
    if doc_id not in doc_ids:
        continue
    
    gt_texts = gt_by_doc.get(doc_id, set())
    
    # Verificar si está en GT
    in_gt = False
    for gt_text in gt_texts:
        if text_norm == gt_text:
            in_gt = True
            break
        if len(text_norm) >= 3 and len(gt_text) >= 3:
            if text_norm in gt_text or gt_text in text_norm:
                in_gt = True
                break
    
    # Verificar si fue KEEP
    # Buscar con coincidencia por doc_id y texto (label puede variar)
    was_kept = False
    for keep_doc, keep_text, keep_label in keep_set:
        if keep_doc == doc_id:
            if text_norm == keep_text:
                was_kept = True
                break
            if len(text_norm) >= 3 and len(keep_text) >= 3:
                if text_norm in keep_text or keep_text in text_norm:
                    was_kept = True
                    break
    
    if was_kept:
        if in_gt:
            kept_in_gt += 1
        else:
            kept_not_in_gt += 1
    else:
        if in_gt:
            filtered_in_gt += 1
            if len(filtered_in_gt_examples) < 15:
                filtered_in_gt_examples.append({
                    'doc': doc_id[:8],
                    'text': text[:50],
                    'label': label
                })
        else:
            filtered_not_in_gt += 1
            if len(filtered_not_in_gt_examples) < 10:
                filtered_not_in_gt_examples.append({
                    'doc': doc_id[:8],
                    'text': text[:50],
                    'label': label
                })

print("=" * 80)
print("📊 ANÁLISIS COMPLETO DEL PIPELINE")
print("=" * 80)

total_input = len([e for e in entrada.get('entities', []) if e.get('doc_id', '') in doc_ids])
total_keep = kept_in_gt + kept_not_in_gt
total_filter = filtered_in_gt + filtered_not_in_gt

print(f"\n📥 ENTRADA: {total_input} entidades")
print(f"📤 KEEP: {total_keep} entidades")
print(f"🗑️  FILTRADAS: {total_filter} entidades")

print(f"\n🎯 ANÁLISIS DE KEEP ({total_keep}):")
print(f"  ✅ KEEP + en GT (TP): {kept_in_gt} ({kept_in_gt/total_keep*100:.1f}%)")
print(f"  ❌ KEEP + NO en GT (FP): {kept_not_in_gt} ({kept_not_in_gt/total_keep*100:.1f}%)")

if total_filter > 0:
    print(f"\n🎯 ANÁLISIS DE FILTRADAS ({total_filter}):")
    print(f"  ✅ FILTER + NO en GT (TN - correcto): {filtered_not_in_gt} ({filtered_not_in_gt/total_filter*100:.1f}%)")
    print(f"  ❌ FILTER + en GT (FN - perdimos PII): {filtered_in_gt} ({filtered_in_gt/total_filter*100:.1f}%)")

print(f"\n📈 MÉTRICAS:")
tp = kept_in_gt
fp = kept_not_in_gt
fn = filtered_in_gt
tn = filtered_not_in_gt

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

print(f"  Precision (KEEP correcto / total KEEP): {precision:.4f}")
print(f"  Recall (PII detectado / total PII en GT): {recall:.4f}")
print(f"  F1: {f1:.4f}")
print(f"  Accuracy: {accuracy:.4f}")

if filtered_in_gt_examples:
    print(f"\n⚠️  ENTIDADES FILTRADAS QUE DEBERÍAN SER KEEP (FN):")
    for ex in filtered_in_gt_examples:
        print(f"  [{ex['doc']}] {ex['label']:30s} | \"{ex['text']}\"")

if filtered_not_in_gt_examples:
    print(f"\n✅ ENTIDADES FILTRADAS CORRECTAMENTE (TN):")
    for ex in filtered_not_in_gt_examples:
        print(f"  [{ex['doc']}] {ex['label']:30s} | \"{ex['text']}\"")

print("=" * 80)
