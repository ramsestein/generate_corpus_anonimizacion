#!/usr/bin/env python3
"""
Analyze dictionary filter results against ground truth.
Identify False Negatives (real entities filtered out) and False Positives (noise kept).
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

SCRIPT_DIR = Path(__file__).parent

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print("=" * 80)
    print("ANÁLISIS DE RESULTADOS - FILTROS DE DICCIONARIO")
    print("=" * 80)
    
    # Load results
    results_file = SCRIPT_DIR / "outputs" / "dict_filters_only_results.json"
    results = load_json(results_file)
    entities = results["entities"]
    
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"   Total entidades procesadas: {len(entities)}")
    print(f"   ✓ KEEP (whitelist):  {results['statistics']['kept']} ({results['statistics']['kept']/len(entities)*100:.1f}%)")
    print(f"   ✗ FILTER (blacklist): {results['statistics']['filtered']} ({results['statistics']['filtered']/len(entities)*100:.1f}%)")
    print(f"   ? ESCALATE (sin match): {results['statistics']['escalated']} ({results['statistics']['escalated']/len(entities)*100:.1f}%)")
    
    # Analyze filtered entities by label
    filtered = [e for e in entities if e.get("decision") == "FILTER"]
    kept = [e for e in entities if e.get("decision") == "KEEP"]
    escalated = [e for e in entities if e.get("decision") == "ESCALATE"]
    
    print(f"\n" + "=" * 80)
    print("❌ FALSOS NEGATIVOS - Entidades REALES filtradas incorrectamente")
    print("=" * 80)
    
    # Group filtered by reason
    filtered_by_reason = defaultdict(list)
    for e in filtered:
        reason = e.get('filter_reason', 'unknown')
        filtered_by_reason[reason].append(e)
    
    print(f"\n📋 Filtradas por razón:")
    for reason, items in sorted(filtered_by_reason.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"   • {reason}: {len(items)} entidades")
    
    # Analyze each reason
    print(f"\n" + "-" * 80)
    print("DETALLE POR RAZÓN DE FILTRADO:")
    print("-" * 80)
    
    for reason, items in sorted(filtered_by_reason.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n🔍 {reason} ({len(items)} entidades)")
        
        # Group by label
        by_label = defaultdict(list)
        for e in items:
            label = e.get('label', 'N/A')
            by_label[label].append(e)
        
        for label, label_items in sorted(by_label.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n   Label: {label} ({len(label_items)} entidades)")
            
            # Show top 5 examples
            for i, e in enumerate(label_items[:5], 1):
                text = e.get('entity_text', e.get('text', 'N/A'))
                matched = e.get('matched_term', 'N/A')
                print(f"      {i}. '{text}' (match: {matched})")
    
    print(f"\n" + "=" * 80)
    print("✅ ENTIDADES RESCATADAS - Por whitelist")
    print("=" * 80)
    
    # Analyze kept entities
    kept_by_label = defaultdict(list)
    for e in kept:
        label = e.get('label', 'N/A')
        kept_by_label[label].append(e)
    
    print(f"\n📋 Rescatadas por label:")
    for label, items in sorted(kept_by_label.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"   • {label}: {len(items)} entidades")
        
        # Show unique terms
        unique_terms = set(e.get('matched_term', 'N/A') for e in items)
        print(f"     Términos únicos en whitelist: {len(unique_terms)}")
        for term in sorted(unique_terms)[:5]:
            print(f"       - {term}")
    
    print(f"\n" + "=" * 80)
    print("❓ ENTIDADES SIN MATCH - Escaladas a LLM")
    print("=" * 80)
    
    # Analyze escalated entities
    escalated_by_label = defaultdict(list)
    for e in escalated:
        label = e.get('label', 'N/A')
        escalated_by_label[label].append(e)
    
    print(f"\n📋 Escaladas por label:")
    for label, items in sorted(escalated_by_label.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"   • {label}: {len(items)} entidades")
        
        # Show top 5 examples
        for i, e in enumerate(items[:5], 1):
            text = e.get('entity_text', e.get('text', 'N/A'))
            print(f"      {i}. '{text}'")
    
    # Calculate metrics
    print(f"\n" + "=" * 80)
    print("📈 MÉTRICAS DE FILTRADO")
    print("=" * 80)
    
    total = len(entities)
    true_positives = len(kept)  # Entities that should be kept and were kept
    false_negatives = len(filtered)  # Real entities that were filtered out
    uncertain = len(escalated)  # Entities that need LLM judgment
    
    # Assuming all entities in input are real PII (ground truth)
    # Then filtered entities are false negatives
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print(f"\n   Total entidades (ground truth): {total}")
    print(f"   ✓ Correctamente identificadas (KEEP): {true_positives}")
    print(f"   ✗ Incorrectamente filtradas (FILTER): {false_negatives}")
    print(f"   ? Sin decisión (ESCALATE): {uncertain}")
    print(f"\n   📊 RECALL (sensibilidad): {recall:.2%}")
    print(f"      - El filtro solo identificó correctamente {recall:.2%} de las entidades reales")
    print(f"      - Perdió {false_negatives} entidades reales ({false_negatives/total*100:.1f}%)")
    
    print(f"\n" + "=" * 80)
    print("💡 RECOMENDACIONES")
    print("=" * 80)
    
    print(f"\n1. REDUCIR FALSOS NEGATIVOS ({false_negatives} entidades):")
    print(f"   - Revisar reglas de longitud mínima (filtró muchas entidades cortas legítimas)")
    print(f"   - Revisar regla de un solo carácter (puede ser demasiado agresiva)")
    print(f"   - Considerar añadir excepciones por label")
    
    print(f"\n2. MEJORAR COBERTURA DE WHITELIST:")
    print(f"   - Solo {true_positives}/{total} ({true_positives/total*100:.1f}%) entidades tienen match")
    print(f"   - Añadir más términos a hospitales.json, lugares.json")
    print(f"   - Considerar agregar más categorías de whitelist")
    
    print(f"\n3. ENTIDADES SIN DECISIÓN ({uncertain}):")
    print(f"   - {uncertain/total*100:.1f}% de entidades necesitan LLM")
    print(f"   - Estas entidades pasan al siguiente filtro (LLM)")
    print(f"   - Considerar ampliar las listas para reducir carga en LLM")
    
    # Specific recommendations based on filtered reasons
    print(f"\n4. AJUSTES ESPECÍFICOS RECOMENDADOS:")
    
    if "Entidad de un solo carácter" in filtered_by_reason:
        count = len(filtered_by_reason["Entidad de un solo carácter"])
        print(f"   - ignore_single_char: Considera desactivar o hacer excepciones")
        print(f"     (filtró {count} entidades, algunas podrían ser válidas)")
    
    for reason, items in filtered_by_reason.items():
        if "Longitud" in reason and "mínimo" in reason:
            # Extract label from reason
            import re
            match = re.search(r'para (\w+)', reason)
            if match:
                label = match.group(1)
                print(f"   - min_length_per_label['{label}']: Reducir de {reason.split('mínimo ')[1].split()[0]} a valor menor")
    
    print(f"\n" + "=" * 80)

if __name__ == "__main__":
    main()
