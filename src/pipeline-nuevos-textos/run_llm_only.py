#!/usr/bin/env python3
"""
Script para re-evaluar con LLM entidades del pipeline.

Este script permite ejecutar SOLO la etapa LLM del pipeline, reutilizando JSONs
ya procesados por run_full_pipeline.py, sin necesidad de volver a ejecutar SetFit.

Modos de operación:
1. Re-evaluar solo entidades que pasaron por LLM previamente
2. Re-evaluar entidades filtradas como RUIDO
3. Evaluar todas las entidades del JSON

Uso básico:
    python run_llm_only.py --input outputs/aws2-results.json --output outputs/aws2-llm-rerun.json
    
Comparar con decisiones previas:
    python run_llm_only.py --input outputs/aws2-results.json --output outputs/comparison.json --compare-previous
    
Solo entidades RUIDO:
    python run_llm_only.py --input outputs/aws2-results.json --output outputs/ruido-reeval.json --filter-by ruido_only

Formato de salida:
    Por defecto, solo guarda TRUE/FALSE (is_pii). Con --verbose se guardan los detalles completos.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Añadir paths del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import del módulo LLM y IO
llm_judge_path = Path(__file__).parent / "llm_judge"
sys.path.insert(0, str(llm_judge_path.parent))

from llm_judge import run_llm_judge  # Usar API directa del módulo
from io_json import load_json, save_pipeline_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pipeline_json(input_path: Path) -> Dict[str, Any]:
    """
    Carga y valida un JSON generado por run_full_pipeline o formato compatible.
    
    Soporta múltiples formatos:
    - {'decisions': [...]}  # Formato del pipeline completo
    - {'entities': [...]}   # Formato alternativo
    - [...]                 # Lista directa de entidades
    
    Args:
        input_path: Ruta al archivo JSON
        
    Returns:
        Dict con 'metadata' y 'decisions'
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el formato no es válido
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")
    
    try:
        data = load_json(str(input_path))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido en {input_path}: {e}")
    
    # Si es una lista directa, convertir a formato estándar
    if isinstance(data, list):
        logger.info("JSON es una lista directa de entidades, convirtiendo a formato estándar")
        data = {
            'metadata': {},
            'decisions': data
        }
    elif isinstance(data, dict):
        # Buscar clave de entidades en orden de prioridad
        entities_key = None
        for key in ['decisions', 'entities', 'results', 'detecciones']:
            if key in data:
                entities_key = key
                break
        
        if entities_key is None:
            raise ValueError(
                "El JSON no contiene ninguna clave de entidades reconocida. "
                "Esperado: 'decisions', 'entities', 'results' o 'detecciones'"
            )
        
        # Normalizar a 'decisions' si es necesario
        if entities_key != 'decisions':
            logger.info(f"Convirtiendo clave '{entities_key}' a 'decisions'")
            data['decisions'] = data.pop(entities_key)
        
        if not isinstance(data['decisions'], list):
            raise ValueError(f"'{entities_key}' debe ser una lista de entidades")
        
        if 'metadata' not in data:
            logger.warning("El JSON no contiene 'metadata', se creará una vacía")
            data['metadata'] = {}
    else:
        raise ValueError(f"El JSON debe ser un diccionario o lista, encontrado: {type(data)}")
    
    return data


def get_llm_evaluated_entities(
    data: Dict[str, Any],
    filter_by: str = 'llm_only'
) -> List[Dict[str, Any]]:
    """
    Extrae entidades del JSON del pipeline según el criterio especificado.
    
    Args:
        data: Diccionario con 'decisions'
        filter_by: Criterio de filtrado:
            - 'all': Todas las entidades
            - 'llm_only': Solo las que fueron evaluadas por LLM previamente
            - 'ruido_only': Solo las clasificadas como RUIDO
            - 'pii_only': Solo las clasificadas como PII
    
    Returns:
        Lista de entidades que cumplen el criterio
    """
    decisions = data.get('decisions', [])
    
    if filter_by == 'all':
        return decisions
    
    to_evaluate = []
    
    for d in decisions:
        classification = d.get('classification', '').upper()
        classification_source = d.get('classification_source', '').lower()
        is_pii = d.get('is_pii', None)
        
        if filter_by == 'llm_only':
            # Solo entidades que pasaron por LLM (tienen llm en classification_source)
            if 'llm' in classification_source:
                to_evaluate.append(d)
        
        elif filter_by == 'ruido_only':
            # Solo RUIDO
            if classification == 'RUIDO' or is_pii is False:
                to_evaluate.append(d)
        
        elif filter_by == 'pii_only':
            # Solo PII
            if classification == 'PII' or is_pii is True:
                to_evaluate.append(d)
    
    return to_evaluate


def evaluate_with_llm(
    entities: List[Dict[str, Any]],
    model: str = "qwen2.5:7b",
    rules_path: str = None,
    verbose: bool = False,
    compare_previous: bool = False
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evalúa las entidades con el LLM usando la API del módulo llm_judge.
    
    Args:
        entities: Lista de entidades a evaluar
        model: Modelo LLM a usar
        rules_path: Ruta al archivo de reglas
        verbose: Si True, guarda detalles completos (thought, confidence, etc.)
        compare_previous: Si True, compara con decisiones previas
    
    Returns:
        Tupla (resultados, estadísticas)
    """
    if not entities:
        return [], {'total': 0, 'rescued': 0, 'still_ruido': 0, 'errors': 0, 'elapsed_seconds': 0}
    
    total = len(entities)
    start_time = time.time()
    
    logger.info(f"Evaluando {total} entidades con LLM ({model})...")
    
    # Configuración para run_llm_judge
    config = {
        "model": model,
        "rules_path": rules_path,
        "timeout": 120,
        "max_retries": 2,
        "debug": False,
    }
    
    # Ejecutar LLM usando la API del módulo
    try:
        llm_results = run_llm_judge(entities, None, config)
    except Exception as e:
        logger.error(f"Error en run_llm_judge: {e}")
        # Retornar entidades sin modificar en caso de error
        return entities, {
            'total': total,
            'rescued': 0,
            'still_ruido': 0,
            'errors': total,
            'elapsed_seconds': time.time() - start_time
        }
    
    # Procesar resultados
    results = []
    rescued = 0
    still_ruido = 0
    errors = 0
    changed_decisions = 0
    
    for i, result in enumerate(llm_results):
        entity = entities[i].copy()
        
        # Extraer decisión del LLM
        llm_decision = result.get('llm_decision')  # True/False/None
        decision = result.get('decision')  # 'KEEP'/'FILTER'
        
        # Guardar decisión previa si existe y se solicitó comparación
        if compare_previous and 'llm_is_pii' in entity:
            entity['llm_is_pii_prev'] = entity.get('llm_is_pii')
        
        # Actualizar con nueva decisión
        entity['llm_is_pii'] = llm_decision
        
        # Solo guardar detalles si verbose está activado
        if verbose:
            entity['llm_confidence'] = result.get('llm_confidence', 0.0)
            entity['llm_response'] = result.get('llm_response', '')
            entity['llm_status'] = result.get('llm_status', 'unknown')
            entity['llm_processing_time'] = result.get('llm_processing_time', 0.0)
        
        # Actualizar clasificación según decisión
        if decision == 'KEEP' and llm_decision is True:
            entity['is_pii'] = True
            entity['classification'] = 'PII'
            entity['classification_source'] = 'llm_rerun_rescue'
            rescued += 1
        elif decision == 'FILTER' or llm_decision is False:
            entity['is_pii'] = False
            entity['classification'] = 'RUIDO'
            entity['classification_source'] = 'llm_rerun_filtered'
            still_ruido += 1
        else:
            # Error o None
            errors += 1
            entity['classification_source'] = 'llm_rerun_error'
        
        # Detectar cambios
        if compare_previous and 'llm_is_pii_prev' in entity:
            if entity['llm_is_pii_prev'] != entity['llm_is_pii']:
                entity['decision_changed'] = True
                changed_decisions += 1
            else:
                entity['decision_changed'] = False
        
        results.append(entity)
        
        # Log periódico
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"  Progreso: {i+1}/{total} ({(i+1)/total*100:.1f}%) - "
                f"TRUE: {rescued}, FALSE: {still_ruido}, Errores: {errors} - "
                f"ETA: {remaining/60:.1f} min"
            )
    
    elapsed = time.time() - start_time
    
    # Log resumen
    logger.info(f"\n{'='*60}")
    logger.info(f"RESUMEN LLM:")
    logger.info(f"  Total evaluados: {total}")
    logger.info(f"  TRUE (PII detectado): {rescued} ({rescued/total*100:.1f}%)")
    logger.info(f"  FALSE (RUIDO confirmado): {still_ruido} ({still_ruido/total*100:.1f}%)")
    logger.info(f"  Errores: {errors}")
    if compare_previous:
        logger.info(f"  Decisiones que cambiaron: {changed_decisions} ({changed_decisions/total*100:.1f}%)")
    logger.info(f"  Tiempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"{'='*60}")
    
    stats = {
        'total': total,
        'rescued': rescued,
        'still_ruido': still_ruido,
        'errors': errors,
        'elapsed_seconds': elapsed
    }
    
    if compare_previous:
        stats['changed_decisions'] = changed_decisions
    
    return results, stats


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
        description='Re-ejecutar la etapa LLM del pipeline sobre entidades de un JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Re-evaluar entidades que pasaron por LLM previamente
  python run_llm_only.py --input outputs/aws2-results.json --output outputs/llm-rerun.json
  
  # Re-evaluar solo entidades RUIDO
  python run_llm_only.py --input outputs/aws2-results.json --output outputs/ruido-reeval.json --filter-by ruido_only
  
  # Comparar con decisiones previas
  python run_llm_only.py --input outputs/aws2-results.json --output outputs/comparison.json --compare-previous
  
  # Verbose con detalles completos
  python run_llm_only.py --input outputs/aws2-results.json --output outputs/detailed.json --verbose
        """
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Archivo JSON del pipeline completo (ej: outputs/aws2-results.json)'
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
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Guardar detalles completos del LLM (thought, confidence, etc.)'
    )
    parser.add_argument(
        '--filter-by',
        default='llm_only',
        choices=['all', 'llm_only', 'ruido_only', 'pii_only'],
        help='Filtro para seleccionar entidades: all (todas), llm_only (solo evaluadas por LLM), '
             'ruido_only (solo RUIDO), pii_only (solo PII). Default: llm_only'
    )
    parser.add_argument(
        '--compare-previous',
        action='store_true',
        help='Comparar nuevas decisiones con las previas y marcar cambios'
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
    logger.info(f"RE-EJECUCIÓN LLM DEL PIPELINE")
    logger.info(f"{'='*60}")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model:  {args.model}")
    logger.info(f"Filter: {args.filter_by}")
    if args.compare_previous:
        logger.info(f"Modo:   Comparación con decisiones previas")
    logger.info(f"{'='*60}")
    
    # Cargar datos del pipeline
    logger.info("Cargando JSON del pipeline...")
    try:
        data = load_pipeline_json(input_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error cargando archivo: {e}")
        sys.exit(1)
    
    logger.info(f"  Total de entidades en JSON: {len(data['decisions'])}")
    
    # Filtrar entidades según criterio
    entities = get_llm_evaluated_entities(data, filter_by=args.filter_by)
    
    logger.info(f"Entidades seleccionadas ({args.filter_by}): {len(entities)}")
    
    if not entities:
        logger.warning("No hay entidades que evaluar con el filtro especificado.")
        logger.info(f"Filtro usado: {args.filter_by}")
        logger.info("Opciones: all, llm_only, ruido_only, pii_only")
        sys.exit(0)
    
    if args.limit:
        entities = entities[:args.limit]
        logger.info(f"Limitado a: {len(entities)} entidades")
    
    # Evaluar con LLM
    llm_results, stats = evaluate_with_llm(
        entities,
        model=args.model,
        rules_path=args.rules,
        verbose=args.verbose,
        compare_previous=args.compare_previous
    )
    
    # Combinar resultados con el JSON original
    logger.info("Combinando resultados...")
    final_result = merge_results(data, llm_results, stats)
    
    # Añadir información sobre el modo de ejecución
    final_result['metadata']['mode'] = 'llm_only_rerun'
    final_result['metadata']['filter_by'] = args.filter_by
    final_result['metadata']['compare_previous'] = args.compare_previous
    final_result['metadata']['model'] = args.model
    final_result['metadata']['original_metadata'] = data.get('metadata', {})
    
    # Guardar
    logger.info(f"Guardando resultados en {output_path}...")
    save_pipeline_results(
        final_result['decisions'],
        str(output_path),
        metadata=final_result['metadata']
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETADO")
    logger.info(f"  PII final: {final_result['metadata']['pii_entities']}")
    logger.info(f"  RUIDO final: {final_result['metadata']['ruido_entities']}")
    if args.compare_previous and 'changed_decisions' in stats:
        logger.info(f"  Decisiones cambiadas: {stats['changed_decisions']}")
    logger.info(f"  Archivo guardado: {output_path}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
