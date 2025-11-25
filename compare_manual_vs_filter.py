#!/usr/bin/env python3
"""
Comparación de Métricas: Manual Correction vs Filtro Determinista
"""

import json
from collections import defaultdict

def calculate_confusion_matrix(predictions, ground_truth):
    """Calcula matriz de confusión."""
    tp = fp = fn = tn = 0
    
    for pred, gt in zip(predictions, ground_truth):
        if gt == True and pred == True:
            tp += 1
        elif gt == False and pred == True:
            fp += 1
        elif gt == True and pred == False:
            fn += 1
        elif gt == False and pred == False:
            tn += 1
    
    return tp, fp, fn, tn

def calculate_metrics(tp, fp, fn, tn):
    """Calcula precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# 1. Cargar entidades originales con manual_correction
with open('outputs/historical_entities_adapted.json', 'r', encoding='utf-8') as f:
    original_entities = json.load(f)

# 2. Cargar resultados del filtro
with open('outputs/historical_filtered.json', 'r', encoding='utf-8') as f:
    filtered_entities = json.load(f)

print("\n" + "="*80)
print("COMPARACIÓN: MANUAL_CORRECTION vs FILTRO DETERMINISTA")
print("="*80)

# 3. Preparar datos para comparación
# Ground truth: manual_correction (TRUE = anonimizar, FALSE = no anonimizar)
ground_truth = []
filter_decisions = []
filter_stats = defaultdict(int)

for orig, filt in zip(original_entities, filtered_entities):
    # Ground truth desde manual_correction
    gt = orig.get('manual_correction', 'FALSE') == 'TRUE'
    ground_truth.append(gt)
    
    # Decisión del filtro
    decision = filt.get('decision', 'ESCALATE_TO_LLM')
    filter_stats[decision] += 1
    
    # Convertir decisión del filtro a booleano
    if decision == 'FORCE_ANONYMIZE':
        filter_dec = True  # Anonimizar
    elif decision == 'FORCE_IGNORE':
        filter_dec = False  # No anonimizar
    else:  # ESCALATE_TO_LLM
        # Para esta comparación, asumimos que escalamos = no decidimos = FALSE
        # (en el pipeline completo iría al LLM)
        filter_dec = False
    
    filter_decisions.append(filter_dec)

# 4. Calcular métricas
tp, fp, fn, tn = calculate_confusion_matrix(filter_decisions, ground_truth)
precision, recall, f1 = calculate_metrics(tp, fp, fn, tn)

# 5. Mostrar resultados
print(f"\n📊 ESTADÍSTICAS DEL FILTRO:")
print(f"   Total entidades: {len(original_entities)}")
print(f"   - FORCE_ANONYMIZE: {filter_stats['FORCE_ANONYMIZE']:3d} ({filter_stats['FORCE_ANONYMIZE']/len(original_entities)*100:5.1f}%)")
print(f"   - FORCE_IGNORE:    {filter_stats['FORCE_IGNORE']:3d} ({filter_stats['FORCE_IGNORE']/len(original_entities)*100:5.1f}%)")
print(f"   - ESCALATE_TO_LLM: {filter_stats['ESCALATE_TO_LLM']:3d} ({filter_stats['ESCALATE_TO_LLM']/len(original_entities)*100:5.1f}%)")

reduced = filter_stats['FORCE_ANONYMIZE'] + filter_stats['FORCE_IGNORE']
print(f"\n   Llamadas LLM evitadas: {reduced} ({reduced/len(original_entities)*100:.1f}%)")

print(f"\n📊 GROUND TRUTH (MANUAL_CORRECTION):")
true_count = sum(ground_truth)
false_count = len(ground_truth) - true_count
print(f"   - TRUE (debe anonimizar):  {true_count:3d} ({true_count/len(ground_truth)*100:5.1f}%)")
print(f"   - FALSE (no anonimizar):   {false_count:3d} ({false_count/len(ground_truth)*100:5.1f}%)")

print(f"\n📈 MATRIZ DE CONFUSIÓN (Filtro vs Manual):")
print(f"   TP (Correcto + anonimizar):   {tp:3d}")
print(f"   TN (Correcto no anonimizar):  {tn:3d}")
print(f"   FP (Error: anonimizó mal):    {fp:3d}")
print(f"   FN (Error: no anonimizó):     {fn:3d}")

print(f"\n📈 MÉTRICAS DEL FILTRO:")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"   Accuracy:  {(tp+tn)/(tp+fp+fn+tn):.4f} ({(tp+tn)/(tp+fp+fn+tn)*100:.2f}%)")

# 6. Análisis por tipo de decisión
print(f"\n🔍 ANÁLISIS POR DECISIÓN DEL FILTRO:")
for decision_type in ['FORCE_ANONYMIZE', 'FORCE_IGNORE', 'ESCALATE_TO_LLM']:
    indices = [i for i, filt in enumerate(filtered_entities) if filt['decision'] == decision_type]
    if not indices:
        continue
    
    gt_true = sum(1 for i in indices if ground_truth[i])
    gt_false = len(indices) - gt_true
    
    print(f"\n   {decision_type}:")
    print(f"      Total: {len(indices)}")
    print(f"      - Ground truth TRUE:  {gt_true:3d} ({gt_true/len(indices)*100:5.1f}%)")
    print(f"      - Ground truth FALSE: {gt_false:3d} ({gt_false/len(indices)*100:5.1f}%)")
    
    if decision_type == 'FORCE_ANONYMIZE' and gt_false > 0:
        print(f"      ⚠️  {gt_false} falsos positivos (anonimizó incorrectamente)")
    elif decision_type == 'FORCE_IGNORE' and gt_true > 0:
        print(f"      ⚠️  {gt_true} falsos negativos (no anonimizó cuando debía)")

# 7. Mostrar ejemplos de errores
print(f"\n❌ EJEMPLOS DE ERRORES DEL FILTRO:")
errors_shown = 0
max_errors = 5

for i, (orig, filt, gt, pred) in enumerate(zip(original_entities, filtered_entities, ground_truth, filter_decisions)):
    if gt != pred and errors_shown < max_errors:
        error_type = "FP (anonimizó mal)" if pred and not gt else "FN (no anonimizó)"
        print(f"\n   {error_type}:")
        print(f"      Texto: '{orig['keyword']}'")
        print(f"      Label: {orig['label']}")
        print(f"      Decisión filtro: {filt['decision']}")
        print(f"      Manual correction: {orig['manual_correction']}")
        errors_shown += 1

print("\n" + "="*80 + "\n")

# 8. Guardar reporte
report = {
    "total_entities": len(original_entities),
    "filter_stats": dict(filter_stats),
    "ground_truth_stats": {
        "true": true_count,
        "false": false_count
    },
    "confusion_matrix": {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    },
    "metrics": {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp+tn)/(tp+fp+fn+tn)
    }
}

with open('outputs/historical_comparison_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("📁 Reporte guardado en: outputs/historical_comparison_report.json")
