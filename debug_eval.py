#!/usr/bin/env python3
"""Script temporal para debug de evaluación - análisis completo."""
import json
import unicodedata
from collections import defaultdict

def normalize_text(text):
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return ' '.join(text.split())

# Cargar resultados
with open('outputs/resultados_completo.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Extraer doc_ids procesados
processed_docs = set()
for d in results['decisions']:
    if d.get('document_id'):
        processed_docs.add(d['document_id'])

print(f"Documentos procesados: {len(processed_docs)}")

# Cargar GT solo para docs procesados
gt_by_doc = {}
for doc_id in processed_docs:
    try:
        with open(f'corpus/ANTIGUO/entidades/{doc_id}.json', 'r', encoding='utf-8') as f:
            gt = json.load(f)
        gt_by_doc[doc_id] = set()
        for e in gt.get('data', []):
            gt_by_doc[doc_id].add(normalize_text(e['text']))
    except:
        pass

# Cargar predicciones KEEP
pred_by_doc = defaultdict(set)
for d in results['decisions']:
    if d.get('decision') == 'KEEP':
        doc_id = d.get('document_id', '')
        text_norm = normalize_text(d.get('entity_text', ''))
        pred_by_doc[doc_id].add(text_norm)

# Análisis global
total_tp = 0
total_fp = 0  
total_fn = 0

sample_matches = []
sample_fps = []
sample_fns = []

for doc_id in processed_docs:
    gt_texts = gt_by_doc.get(doc_id, set())
    pred_texts = pred_by_doc.get(doc_id, set())
    
    tp = gt_texts & pred_texts
    fp = pred_texts - gt_texts
    fn = gt_texts - pred_texts
    
    total_tp += len(tp)
    total_fp += len(fp)
    total_fn += len(fn)
    
    # Guardar ejemplos
    for t in list(tp)[:2]:
        sample_matches.append((doc_id, t))
    for t in list(fp)[:2]:
        sample_fps.append((doc_id, t))
    for t in list(fn)[:2]:
        sample_fns.append((doc_id, t))

print(f"\n=== MÉTRICAS GLOBALES (solo texto) ===")
print(f"TP: {total_tp}")
print(f"FP: {total_fp}")
print(f"FN: {total_fn}")
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

print(f"\n=== EJEMPLOS DE MATCHES (TP) ===")
for doc, text in sample_matches[:5]:
    print(f"  [{doc[:8]}...] '{text}'")

print(f"\n=== EJEMPLOS DE FALSE POSITIVES (KEEP pero no en GT) ===")
for doc, text in sample_fps[:10]:
    print(f"  [{doc[:8]}...] '{text}'")

print(f"\n=== EJEMPLOS DE FALSE NEGATIVES (en GT pero no KEEP) ===")
for doc, text in sample_fns[:10]:
    # Buscar si el texto aparece en alguna predicción filtrada
    print(f"  [{doc[:8]}...] '{text}'")

# Análisis de un documento específico
print(f"\n=== ANÁLISIS DETALLADO DOC: 00039e89 ===")
doc_id = '00039e89-4b9a-4a81-bb08-334cf20b482c'
gt_texts = gt_by_doc.get(doc_id, set())
pred_texts = pred_by_doc.get(doc_id, set())

print(f"GT: {gt_texts}")
print(f"PRED: {pred_texts}")
print(f"Matches: {gt_texts & pred_texts}")
print(f"FN (en GT, no en PRED): {gt_texts - pred_texts}")
print(f"FP (en PRED, no en GT): {pred_texts - gt_texts}")

