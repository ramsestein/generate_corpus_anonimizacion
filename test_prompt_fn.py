#!/usr/bin/env python3
"""
Test del prompt mejorado sobre casos FN reales.
Comprueba si las mejoras en prompts.py reducen los FN.
"""

import json
import sys
from pathlib import Path

# Añadir paths
sys.path.insert(0, str(Path(__file__).parent / "src" / "pipeline-nuevos-textos"))

from llm_judge import LLMJudge

def main():
    # Cargar FN
    with open('fn_analysis/fn_detailed.json', 'r', encoding='utf-8') as f:
        fn_data = json.load(f)
    
    # Inicializar LLM Judge con el prompt mejorado
    judge = LLMJudge(
        model="qwen2.5:7b",
        template_name="strict_v3",  # El que acabamos de mejorar
        timeout=60
    )
    
    print("=" * 60)
    print("TEST DEL PROMPT MEJORADO EN CASOS FN")
    print("=" * 60)
    
    # Seleccionar muestras de cada categoria
    test_cases = []
    
    for category, samples in fn_data['samples_by_category'].items():
        # Tomar 8 muestras de cada categoria
        for s in samples[:8]:
            test_cases.append({
                'category': category,
                'text': s['gold_text'],
                'context': s['context'],
                'doc_id': s['doc_id']
            })
    
    print(f"\nTotal casos a probar: {len(test_cases)}")
    print("-" * 60)
    
    results = {'pass': 0, 'fail': 0, 'error': 0}
    failures = []
    
    for i, case in enumerate(test_cases):
        # Determinar etiqueta segun categoria
        label_map = {
            'FECHAS_CORTAS': 'FECHAS',
            'ANOS_AISLADOS': 'FECHAS',
            'FAMILIARES': 'FAMILIARES_SUJETO_ASISTENCIA',
            'NOMBRES_PROPIOS': 'NOMBRE_SUJETO_ASISTENCIA',
            'SEXO': 'SEXO_SUJETO_ASISTENCIA',
            'FRAGMENTOS_CORTOS': 'OTROS_SUJETO_ASISTENCIA',
            'NUMEROS_CORTOS': 'NUMERO_IDENTIF',
            'OTROS': 'OTROS_SUJETO_ASISTENCIA'
        }
        label = label_map.get(case['category'], 'OTROS_SUJETO_ASISTENCIA')
        
        try:
            result = judge.evaluate(
                entity_text=case['text'],
                entity_label=label,
                context=case['context'][:300],  # Limitar contexto
                debug=False
            )
            
            # Esperamos TRUE porque son FN (deberian ser PII)
            if result.is_valid:
                results['pass'] += 1
                status = "PASS"
            else:
                results['fail'] += 1
                status = "FAIL"
                failures.append({
                    'category': case['category'],
                    'text': case['text'],
                    'context': case['context'][:100]
                })
        except Exception as e:
            results['error'] += 1
            status = f"ERROR: {e}"
        
        print(f"[{i+1}/{len(test_cases)}] {case['category']}: \"{case['text'][:30]}\" -> {status}")
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    total = results['pass'] + results['fail'] + results['error']
    print(f"  PASS (correctamente TRUE): {results['pass']} ({results['pass']/total*100:.1f}%)")
    print(f"  FAIL (incorrectamente FALSE): {results['fail']} ({results['fail']/total*100:.1f}%)")
    print(f"  ERROR: {results['error']}")
    
    if failures:
        print(f"\n{'=' * 60}")
        print("CASOS QUE SIGUEN FALLANDO (primeros 10):")
        print("-" * 60)
        for f in failures[:10]:
            print(f"  [{f['category']}] \"{f['text']}\"")
            print(f"    Contexto: {f['context'][:80]}...")
    
    # Guardar resultados
    with open('fn_analysis/test_prompt_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_tested': total,
            'pass': results['pass'],
            'fail': results['fail'],
            'error': results['error'],
            'pass_rate': results['pass']/total*100 if total > 0 else 0,
            'failures': failures[:20]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nResultados guardados en fn_analysis/test_prompt_results.json")

if __name__ == '__main__':
    main()
