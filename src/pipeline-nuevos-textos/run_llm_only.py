#!/usr/bin/env python3
"""
Script para re-evaluar con LLM solo las entidades clasificadas como RUIDO por SetFit.

Uso:
    python run_llm_only.py --input outputs/aws2-results.json --output outputs/aws2-results-llm.json

Este script permite iterar rápidamente sobre el prompt del LLM sin tener que
ejecutar todo el pipeline completo.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Añadir paths del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import del módulo LLM (el directorio tiene guiones, necesitamos importar desde el path)
llm_judge_path = Path(__file__).parent / "llm_judge"
sys.path.insert(0, str(llm_judge_path.parent))

from llm_judge.judge_optimized import OptimizedLLMJudge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_setfit_results(input_path: Path) -> Dict[str, Any]:
    """Carga los resultados de SetFit desde un JSON."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_ruido_entities(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filtra entidades clasificadas como RUIDO o evaluadas previamente por LLM.
    
    Incluye:
    - Entidades con classification == 'RUIDO'
    - Entidades con is_pii == False
    - Entidades con classification_source que contenga 'llm' (para re-evaluar)
    """
    decisions = data.get('decisions', [])
    
    to_evaluate = []
    for d in decisions:
        # RUIDO directo
        if d.get('classification') == 'RUIDO' or not d.get('is_pii', True):
            to_evaluate.append(d)
        # Ya fue evaluado por LLM (para re-evaluar con nuevo prompt)
        elif 'llm' in d.get('classification_source', '').lower():
            to_evaluate.append(d)
    
    return to_evaluate


def evaluate_with_llm(
    entities: List[Dict[str, Any]],
    model: str = "qwen2.5:7b",
    rules_path: str = None,
    batch_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Evalúa las entidades con el LLM.
    
    Returns:
        Lista de entidades con la nueva clasificación del LLM.
    """
    judge = OptimizedLLMJudge(
        model=model,
        rules_path=rules_path,
        timeout=120,
        max_retries=2
    )
    
    results = []
    total = len(entities)
    rescued = 0
    still_ruido = 0
    errors = 0
    
    start_time = time.time()
    
    logger.info(f"Evaluando {total} entidades con LLM ({model})...")
    
    for i, entity in enumerate(entities):
        if (i + 1) % batch_size == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            logger.info(
                f"  Progreso: {i+1}/{total} ({(i+1)/total*100:.1f}%) - "
                f"Rescatados: {rescued}, Ruido: {still_ruido}, Errores: {errors} - "
                f"ETA: {remaining/60:.1f} min"
            )
        
        try:
            result = judge.evaluate(
                entity_text=entity.get('entity_text', ''),
                entity_label=entity.get('label', ''),
                context=entity.get('context', ''),
                debug=False
            )
            
            # Crear entidad con resultado del LLM
            new_entity = entity.copy()
            new_entity['llm_is_pii'] = result.is_pii
            new_entity['llm_confidence'] = result.confidence
            new_entity['llm_thought'] = result.thought
            new_entity['llm_status'] = result.status
            
            if result.is_pii:
                new_entity['is_pii'] = True
                new_entity['classification'] = 'PII'
                new_entity['classification_source'] = 'llm_rescue'
                rescued += 1
            else:
                new_entity['classification_source'] = 'llm_confirmed_ruido'
                still_ruido += 1
            
            results.append(new_entity)
            
        except Exception as e:
            logger.error(f"Error evaluando entidad: {e}")
            errors += 1
            # Mantener como está
            results.append(entity)
    
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"RESUMEN LLM:")
    logger.info(f"  Total evaluados: {total}")
    logger.info(f"  Rescatados (RUIDO -> PII): {rescued} ({rescued/total*100:.1f}%)")
    logger.info(f"  Confirmados como RUIDO: {still_ruido} ({still_ruido/total*100:.1f}%)")
    logger.info(f"  Errores: {errors}")
    logger.info(f"  Tiempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"{'='*60}")
    
    return results, {
        'total': total,
        'rescued': rescued,
        'still_ruido': still_ruido,
        'errors': errors,
        'elapsed_seconds': elapsed
    }


def merge_results(
    original_data: Dict[str, Any],
    llm_results: List[Dict[str, Any]],
    stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Combina los resultados originales con los del LLM."""
    # Crear índice de resultados LLM por (doc_id, entity_text, start)
    llm_index = {}
    for r in llm_results:
        key = (r.get('document_id'), r.get('entity_text'), r.get('start'))
        llm_index[key] = r
    
    # Actualizar decisions
    new_decisions = []
    for decision in original_data.get('decisions', []):
        key = (decision.get('document_id'), decision.get('entity_text'), decision.get('start'))
        if key in llm_index:
            # Usar resultado del LLM
            new_decisions.append(llm_index[key])
        else:
            # Mantener original
            new_decisions.append(decision)
    
    # Recalcular contadores
    pii_count = sum(1 for d in new_decisions if d.get('is_pii', False))
    ruido_count = len(new_decisions) - pii_count
    
    # Crear nuevo resultado
    result = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'input_file': str(original_data.get('metadata', {}).get('input_file', '')),
            'total_entities': len(new_decisions),
            'pii_entities': pii_count,
            'ruido_entities': ruido_count,
            'pipeline_version': 'llm_only_reeval',
            'llm_stats': stats,
            'original_stats': original_data.get('metadata', {}).get('stats', {})
        },
        'decisions': new_decisions
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Re-evaluar entidades RUIDO con LLM solamente'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Archivo JSON de resultados SetFit (ej: outputs/aws2-results.json)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Archivo JSON de salida (por defecto: input_llm.json)'
    )
    parser.add_argument(
        '--model', '-m',
        default='qwen2.5:7b',
        help='Modelo Ollama a usar (default: qwen2.5:7b)'
    )
    parser.add_argument(
        '--rules',
        default=None,
        help='Ruta al archivo de reglas de anotación (opcional)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limitar número de entidades a evaluar (para pruebas)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Archivo no encontrado: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + '_llm' + input_path.suffix
    )
    
    logger.info(f"{'='*60}")
    logger.info(f"RE-EVALUACIÓN LLM DE ENTIDADES RUIDO")
    logger.info(f"{'='*60}")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model:  {args.model}")
    logger.info(f"{'='*60}")
    
    # Cargar datos
    logger.info("Cargando resultados SetFit...")
    data = load_setfit_results(input_path)
    
    # Filtrar entidades a re-evaluar
    entities = filter_ruido_entities(data)
    
    # Contar tipos
    ruido_count = sum(1 for e in entities if not e.get('is_pii', True))
    llm_prev_count = sum(1 for e in entities if 'llm' in e.get('classification_source', '').lower())
    
    logger.info(f"Entidades a re-evaluar con LLM: {len(entities)}")
    logger.info(f"  - Clasificadas como RUIDO: {ruido_count}")
    logger.info(f"  - Ya evaluadas por LLM previamente: {llm_prev_count}")
    
    if args.limit:
        entities = entities[:args.limit]
        logger.info(f"Limitado a: {len(entities)} entidades")
    
    if not entities:
        logger.info("No hay entidades para evaluar.")
        sys.exit(0)
    
    # Evaluar con LLM
    llm_results, stats = evaluate_with_llm(
        entities,
        model=args.model,
        rules_path=args.rules
    )
    
    # Combinar resultados
    logger.info("Combinando resultados...")
    final_result = merge_results(data, llm_results, stats)
    
    # Guardar
    logger.info(f"Guardando resultados en {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETADO")
    logger.info(f"  PII final: {final_result['metadata']['pii_entities']}")
    logger.info(f"  RUIDO final: {final_result['metadata']['ruido_entities']}")
    logger.info(f"  Archivo guardado: {output_path}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
