#!/usr/bin/env python3
"""
Script para mostrar en detalle qué entidades pasaron por whitelist y blacklist.
"""

import json
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    results_file = SCRIPT_DIR / "outputs" / "dict_filters_only_results.json"
    results = load_json(results_file)
    entities = results["entities"]
    
    print("=" * 100)
    print("ENTIDADES RESCATADAS POR WHITELIST")
    print("=" * 100)
    
    kept = [e for e in entities if e.get("decision") == "KEEP"]
    
    # Agrupar por matched_term y label
    by_term = defaultdict(list)
    for e in kept:
        term = e.get('matched_term', 'N/A')
        by_term[term].append(e)
    
    print(f"\nTotal rescatadas: {len(kept)}\n")
    
    for term in sorted(by_term.keys()):
        items = by_term[term]
        print(f"\n{'─' * 100}")
        print(f"TÉRMINO WHITELIST: '{term}' ({len(items)} coincidencias)")
        print(f"{'─' * 100}")
        
        # Agrupar por label
        by_label = defaultdict(list)
        for e in items:
            label = e.get('label', 'N/A')
            by_label[label].append(e)
        
        for label in sorted(by_label.keys()):
            label_items = by_label[label]
            print(f"\n  Label: {label} ({len(label_items)} entidades)")
            
            for i, e in enumerate(label_items, 1):
                text = e.get('entity_text', e.get('text', 'N/A'))
                matched_list = e.get('matched_list', 'N/A')
                print(f"    {i}. '{text}' (lista: {matched_list})")
    
    # Estadísticas de whitelist
    print(f"\n{'=' * 100}")
    print("RESUMEN WHITELIST")
    print(f"{'=' * 100}")
    
    whitelist_by_label = defaultdict(int)
    for e in kept:
        label = e.get('label', 'N/A')
        whitelist_by_label[label] += 1
    
    print(f"\nPor categoría (label):")
    for label in sorted(whitelist_by_label.keys(), key=lambda x: whitelist_by_label[x], reverse=True):
        count = whitelist_by_label[label]
        print(f"  • {label}: {count} entidades")
    
    # Ahora mostrar blacklist
    print(f"\n\n{'=' * 100}")
    print("ENTIDADES FILTRADAS POR BLACKLIST")
    print(f"{'=' * 100}")
    
    filtered = [e for e in entities if e.get("decision") == "FILTER"]
    
    # Agrupar por razón de filtrado
    by_reason = defaultdict(list)
    for e in filtered:
        reason = e.get('filter_reason', 'N/A')
        by_reason[reason].append(e)
    
    print(f"\nTotal filtradas: {len(filtered)}\n")
    
    for reason in sorted(by_reason.keys()):
        items = by_reason[reason]
        print(f"\n{'─' * 100}")
        print(f"RAZÓN: {reason} ({len(items)} entidades)")
        print(f"{'─' * 100}")
        
        # Agrupar por label
        by_label = defaultdict(list)
        for e in items:
            label = e.get('label', 'N/A')
            by_label[label].append(e)
        
        for label in sorted(by_label.keys()):
            label_items = by_label[label]
            print(f"\n  Label: {label} ({len(label_items)} entidades)")
            
            # Mostrar solo 10 ejemplos por razón
            for i, e in enumerate(label_items[:10], 1):
                text = e.get('entity_text', e.get('text', 'N/A'))
                matched_term = e.get('matched_term', 'N/A')
                filter_decision = e.get('filter_decision', 'N/A')
                print(f"    {i}. '{text}'")
                if matched_term != 'N/A':
                    print(f"       Coincidencia: {matched_term}")
                if filter_decision != 'N/A':
                    print(f"       Decisión: {filter_decision}")
            
            if len(label_items) > 10:
                print(f"    ... y {len(label_items) - 10} más")
    
    # Estadísticas de blacklist
    print(f"\n{'=' * 100}")
    print("RESUMEN BLACKLIST")
    print(f"{'=' * 100}")
    
    print(f"\nPor categoría (label):")
    blacklist_by_label = defaultdict(int)
    for e in filtered:
        label = e.get('label', 'N/A')
        blacklist_by_label[label] += 1
    
    for label in sorted(blacklist_by_label.keys(), key=lambda x: blacklist_by_label[x], reverse=True):
        count = blacklist_by_label[label]
        print(f"  • {label}: {count} entidades")
    
    print(f"\n{'=' * 100}")
    print("COMPARATIVA")
    print(f"{'=' * 100}")
    
    print(f"\nRescatadas (KEEP):  {len(kept)} entidades (6.2%)")
    print(f"Filtradas (FILTER): {len(filtered)} entidades (16.2%)")
    print(f"Escaladas (ESCALATE): {len(entities) - len(kept) - len(filtered)} entidades (77.5%)")
    print(f"TOTAL: {len(entities)} entidades")

if __name__ == "__main__":
    main()
