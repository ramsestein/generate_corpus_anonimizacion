#!/usr/bin/env python3
"""
Combina resultados del filtro con decisiones del LLM y calcula métricas finales.

Este script:
1. Carga filtered_results.json (decisiones del filtro)
2. Carga test_results.json (decisiones del LLM)
3. Para FORCE_ANONYMIZE: fuerza TRUE
4. Para FORCE_IGNORE: fuerza FALSE  
5. Para ESCALATE_TO_LLM: usa decisión del LLM
6. Guarda resultado combinado
"""

import json
import sys

def combine_filter_and_llm_decisions(filter_path, llm_path, output_path):
    """Combina decisiones del filtro y del LLM."""
    
    # Cargar resultados del filtro
    with open(filter_path, 'r', encoding='utf-8') as f:
        filter_results = json.load(f)
    
    # Cargar resultados del LLM
    with open(llm_path, 'r', encoding='utf-8') as f:
        llm_results = json.load(f)
    
    # Crear índice de decisiones LLM por (document_id, entity_text, label)
    llm_index = {}
    for entry in llm_results:
        key = (
            entry.get('document_id', ''),
            entry.get('keyword', '').strip().lower(),
            entry.get('label', '')
        )
        llm_index[key] = entry.get('is_valid', False)
    
    # Combinar decisiones
    combined = []
    stats = {
        'force_anonymize': 0,
        'force_ignore': 0,
        'escalate_used_llm': 0,
        'escalate_no_llm_found': 0
    }
    
    for filter_entry in filter_results:
        doc_id = filter_entry.get('document_id', '')
        entity_text = filter_entry.get('entity_text', '').strip()
        label = filter_entry.get('ner_label', '')
        decision = filter_entry.get('decision', '')
        
        # Determinar decisión final
        final_decision = None
        
        if decision == 'FORCE_ANONYMIZE':
            final_decision = True  # Anonimizar
            stats['force_anonymize'] += 1
        elif decision == 'FORCE_IGNORE':
            final_decision = False  # No anonimizar
            stats['force_ignore'] += 1
        elif decision == 'ESCALATE_TO_LLM':
            # Buscar decisión del LLM
            key = (doc_id, entity_text.lower(), label)
            if key in llm_index:
                final_decision = llm_index[key]
                stats['escalate_used_llm'] += 1
            else:
                # No se encontró en LLM, marcar como FALSE por defecto
                final_decision = False
                stats['escalate_no_llm_found'] += 1
        
        # Crear entrada combinada
        combined_entry = {
            'document_id': doc_id,
            'entity_text': entity_text,
            'label': label,
            'filter_decision': decision,
            'llm_decision': llm_index.get((doc_id, entity_text.lower(), label), None),
            'final_decision': final_decision,
            'is_valid': final_decision,  # Para compatibilidad con compute_llm_metrics.py
            'start': filter_entry.get('start', -1),
            'end': filter_entry.get('end', -1)
        }
        
        if 'context' in filter_entry:
            combined_entry['context'] = filter_entry['context']
        
        combined.append(combined_entry)
    
    # Guardar resultados combinados
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    # Mostrar estadísticas
    print(f"\\n{'='*70}")
    print("COMBINACIÓN DE DECISIONES FILTRO + LLM")
    print(f"{'='*70}")
    print(f"\\nTotal entidades: {len(combined)}")
    print(f"  - FORCE_ANONYMIZE:     {stats['force_anonymize']:4d} (sin LLM)")
    print(f"  - FORCE_IGNORE:        {stats['force_ignore']:4d} (sin LLM)")
    print(f"  - ESCALATE + LLM:      {stats['escalate_used_llm']:4d}")
    print(f"  - ESCALATE sin LLM:    {stats['escalate_no_llm_found']:4d}")
    
    llm_calls = stats['escalate_used_llm'] + stats['escalate_no_llm_found']
    saved = stats['force_anonymize'] + stats['force_ignore']
    reduction = (saved / len(combined) * 100) if len(combined) > 0 else 0
    
    print(f"\\nLlamadas al LLM evitadas: {saved} ({reduction:.1f}%)")
    print(f"{'='*70}\\n")
    
    return combined

if __name__ == '__main__':
    filter_path = 'outputs/filtered_results.json'
    llm_path = 'outputs/test_results.json'
    output_path = 'outputs/combined_results.json'
    
    combined = combine_filter_and_llm_decisions(filter_path, llm_path, output_path)
    print(f"✅ Resultados combinados guardados en: {output_path}")
