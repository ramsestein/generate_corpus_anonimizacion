#!/usr/bin/env python3
"""
Compara métricas antes y después del filtrado.
"""

import json

def load_metrics(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Cargar métricas
metrics_original = load_metrics('outputs/llm_metrics.json')
metrics_filtered = load_metrics('outputs/final_metrics_with_filter.json')

orig = metrics_original['global_metrics']
filt = metrics_filtered['global_metrics']

print("\n" + "="*80)
print("COMPARACIÓN DE MÉTRICAS: ANTES vs DESPUÉS DEL FILTRADO")
print("="*80)

print("\n📊 SIN FILTRO (Solo LLM):")
print(f"   TP: {orig['tp']:3d} | FP: {orig['fp']:3d} | FN: {orig['fn']:3d}")
print(f"   Precision: {orig['precision']:.4f} ({orig['precision']*100:.2f}%)")
print(f"   Recall:    {orig['recall']:.4f} ({orig['recall']*100:.2f}%)")
print(f"   F1-Score:  {orig['f1']:.4f} ({orig['f1']*100:.2f}%)")

print("\n🔍 CON FILTRO (Filtro Determinista + LLM):")
print(f"   TP: {filt['tp']:3d} | FP: {filt['fp']:3d} | FN: {filt['fn']:3d}")
print(f"   Precision: {filt['precision']:.4f} ({filt['precision']*100:.2f}%)")
print(f"   Recall:    {filt['recall']:.4f} ({filt['recall']*100:.2f}%)")
print(f"   F1-Score:  {filt['f1']:.4f} ({filt['f1']*100:.2f}%)")

print("\n📈 DIFERENCIAS:")
delta_p = (filt['precision'] - orig['precision']) * 100
delta_r = (filt['recall'] - orig['recall']) * 100
delta_f1 = (filt['f1'] - orig['f1']) * 100

print(f"   Δ Precision: {delta_p:+.2f} puntos porcentuales")
print(f"   Δ Recall:    {delta_r:+.2f} puntos porcentuales")
print(f"   Δ F1-Score:  {delta_f1:+.2f} puntos porcentuales")

print("\n💡 EFICIENCIA:")
llm_calls_saved = 45  # De los resultados del filtro
print(f"   Entidades filtradas (FORCE_ANONYMIZE): 45")
print(f"   Reducción de llamadas LLM: 6.2%")

print("\n" + "="*80 + "\n")
