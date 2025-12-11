#!/usr/bin/env python3
"""
evaluate_pipeline_filtering.py - Evaluación ROBUSTA del pipeline
=================================================================

Evalúa el pipeline de detección de PII comparando:
1. Predicciones (pipeline_results_full.json con field 'decisions')
2. Ground Truth (combined_entidades_ANTIGUO.json con múltiples formatos soportados)

FLUJO DE EVALUACIÓN:
- SetFit/DictFilters/LLM clasifica cada entidad como PII o descartada
- Se comparan contra GT de cada documento
- Se calculan TP/FP/FN por documento y se agregan globalmente

DEFINICIONES:
- TP: Entidad detectada que EXISTE en GT del mismo documento
- FP: Entidad detectada que NO existe en GT del mismo documento
- FN: Entidad en GT que NO fue detectada

VALIDACIÓN DE DEBUG:
- --debug: Imprime estadísticas por documento y ejemplos
- Detecta documentos sin GT o sin predicciones
- Verifica que document_id sea consistente entre archivos

RUTAS POR DEFECTO (si no se pasan argumentos):
- Resultados: outputs/pipeline_results_full.json
- Ground Truth: outputs/combined_entidades_ANTIGUO.json

USO:
    python evaluate_pipeline_filtering.py
    python evaluate_pipeline_filtering.py --debug
    python evaluate_pipeline_filtering.py \\
        --results custom_results.json \\
        --ground-truth custom_gt.json \\
        --debug
"""

import argparse
import json
import logging
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)-8s] %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# RUTAS POR DEFECTO - Se usan si no se pasan argumentos
DEFAULT_RESULTS = PROJECT_ROOT / "outputs" / "pipeline_results_full.json"
DEFAULT_GT = PROJECT_ROOT / "outputs" / "combined_entidades_ANTIGUO.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation_results_final.json"


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación."""
    text = text.lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    return ' '.join(text.split())


def load_predictions(results_path: str) -> tuple:
    """
    Carga predicciones del pipeline.
    Retorna: (results dict, predictions by doc, source mapping, all doc_ids)
    """
    logger.info(f"[Eval] Loading predictions from: {results_path}")
    
    if not Path(results_path).exists():
        logger.error(f"❌ Results file not found: {results_path}")
        sys.exit(1)
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extraer predicciones por documento
    pred_by_doc = defaultdict(set)
    source_by_text = {}  # (doc_id, text_norm) -> classification_source
    
    decisions = results.get('decisions', [])
    logger.info(f"   Total decisions in results: {len(decisions)}")
    
    for d in decisions:
        doc_id = d.get('document_id', '')
        text = d.get('entity_text', '')
        text_norm = normalize_text(text)
        classification_source = d.get('classification_source', 'unknown')
        
        if doc_id and text_norm:
            pred_by_doc[doc_id].add(text_norm)
            source_by_text[(doc_id, text_norm)] = classification_source
    
    doc_ids = set(pred_by_doc.keys())
    logger.info(f"   Unique documents with predictions: {len(doc_ids)}")
    
    return results, pred_by_doc, source_by_text, doc_ids


def load_ground_truth(gt_path: str) -> dict:
    """
    Carga Ground Truth desde archivo combinado (JSON).
    Soporta múltiples formatos:
      1. {'combined': [list of {doc_id, entities}]}  <- actual format
      2. {doc_id -> list[entities]}
      3. [list of {doc_id, entities}]
    
    Retorna: dict {doc_id -> set of normalized entity texts}
    """
    logger.info(f"[Eval] Loading ground truth from: {gt_path}")
    
    if not Path(gt_path).exists():
        logger.error(f"❌ Ground truth file not found: {gt_path}")
        sys.exit(1)
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_by_doc = {}
    
    # CASO 1: dict con clave 'combined' -> [list de docs]
    if isinstance(gt_data, dict) and 'combined' in gt_data:
        logger.info(f"   GT structure: dict with 'combined' key")
        combined_list = gt_data['combined']
        logger.info(f"   Processing {len(combined_list)} combined entries")
        
        for doc_entry in combined_list:
            if not isinstance(doc_entry, dict):
                continue
            
            doc_id = doc_entry.get('doc_id', '')
            if not doc_id:
                continue
            
            # Extraer entidades del doc_entry
            ents = []
            entities_obj = doc_entry.get('entities', {})
            
            # Si 'entities' es un dict con 'data', usar eso
            if isinstance(entities_obj, dict):
                ents = entities_obj.get('data', [])
            elif isinstance(entities_obj, list):
                # Si 'entities' es una lista directa
                ents = entities_obj
            
            # Normalizar entidades a conjunto de textos
            texts = set()
            for e in ents:
                if isinstance(e, dict):
                    txt = e.get('text') or e.get('entity_text') or ''
                    if txt:
                        texts.add(normalize_text(txt))
                elif isinstance(e, str):
                    texts.add(normalize_text(e))
            
            if texts:
                gt_by_doc[doc_id] = texts
    
    # CASO 2: dict mapping doc_id -> entities (simple)
    elif isinstance(gt_data, dict):
        logger.info(f"   GT structure: dict mapping doc_id -> entities ({len(gt_data)} entries)")
        
        for doc_id, doc_obj in gt_data.items():
            # Saltar campos de metadatos
            if doc_id in ('summary', 'errors', 'metadata', 'combined'):
                continue
            
            ents = []
            
            if isinstance(doc_obj, list):
                # Caso simple: lista directa
                ents = doc_obj
            elif isinstance(doc_obj, dict):
                # Caso compuesto: buscar campos comunes
                if 'data' in doc_obj:
                    ents = doc_obj['data']
                elif 'entities' in doc_obj:
                    ents = doc_obj['entities']
                elif 'text' in doc_obj:
                    ents = [doc_obj['text']]
            
            # Normalizar entidades a conjunto de textos
            texts = set()
            for e in ents:
                if isinstance(e, dict):
                    txt = e.get('text') or e.get('entity_text') or ''
                    if txt:
                        texts.add(normalize_text(txt))
                elif isinstance(e, str):
                    texts.add(normalize_text(e))
            
            if texts:
                gt_by_doc[doc_id] = texts
    
    # CASO 3: lista plana de documentos
    elif isinstance(gt_data, list):
        logger.info(f"   GT structure: list with {len(gt_data)} entries")
        
        for doc_entry in gt_data:
            if not isinstance(doc_entry, dict):
                continue
            
            doc_id = doc_entry.get('doc_id') or doc_entry.get('document_id')
            if not doc_id:
                continue
            
            ents = []
            entities_obj = doc_entry.get('entities', {})
            
            if isinstance(entities_obj, dict):
                ents = entities_obj.get('data', [])
            elif isinstance(entities_obj, list):
                ents = entities_obj
            
            texts = set()
            for e in ents:
                if isinstance(e, dict):
                    txt = e.get('text') or e.get('entity_text') or ''
                    if txt:
                        texts.add(normalize_text(txt))
                elif isinstance(e, str):
                    texts.add(normalize_text(e))
            
            if texts:
                gt_by_doc[doc_id] = texts
    
    logger.info(f"   Unique documents in GT: {len(gt_by_doc)}")
    
    return gt_by_doc


def evaluate(pred_by_doc: dict, source_by_text: dict, gt_by_doc: dict, debug: bool = False):
    """
    Evalúa el pipeline comparando predicciones contra GT.
    
    Definiciones:
    - TP: predicción que EXISTE en GT del mismo documento
    - FP: predicción que NO existe en GT del mismo documento
    - FN: entidad en GT que NO fue predicha
    
    Retorna: dict con métricas globales y por documento
    """
    tp = fp = fn = 0
    examples = {'tp': [], 'fp': [], 'fn': []}
    
    # Métricas por fuente de clasificación
    source_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'total': 0})
    
    # Validación de documentos
    all_doc_ids = set(pred_by_doc.keys()) | set(gt_by_doc.keys())
    docs_only_pred = set(pred_by_doc.keys()) - set(gt_by_doc.keys())
    docs_only_gt = set(gt_by_doc.keys()) - set(pred_by_doc.keys())
    docs_common = set(pred_by_doc.keys()) & set(gt_by_doc.keys())
    
    if debug:
        logger.info(f"[Debug] Document overlap analysis:")
        logger.info(f"        Predictions only: {len(docs_only_pred)}")
        logger.info(f"        GT only: {len(docs_only_gt)}")
        logger.info(f"        Both: {len(docs_common)}")
        if docs_only_pred and len(docs_only_pred) <= 5:
            logger.info(f"        Pred-only doc_ids: {docs_only_pred}")
        if docs_only_gt and len(docs_only_gt) <= 5:
            logger.info(f"        GT-only doc_ids: {docs_only_gt}")
    
    # Procesar cada documento
    doc_debug_count = 0
    for doc_id in all_doc_ids:
        gt_texts = gt_by_doc.get(doc_id, set())
        pred_texts = pred_by_doc.get(doc_id, set())
        
        if debug and doc_debug_count < 3:
            logger.info(f"[Debug] Doc: {doc_id[:30]}")
            logger.info(f"        GT entities: {len(gt_texts)}")
            logger.info(f"        Pred entities: {len(pred_texts)}")
            doc_debug_count += 1
        
        # TP: predicciones que están en GT
        tp_in_doc = pred_texts & gt_texts
        tp += len(tp_in_doc)
        
        # FP: predicciones que NO están en GT
        fp_in_doc = pred_texts - gt_texts
        fp += len(fp_in_doc)
        
        # FN: entidades en GT que NO fueron predichas
        fn_in_doc = gt_texts - pred_texts
        fn += len(fn_in_doc)
        
        if debug and (tp_in_doc or fp_in_doc or fn_in_doc):
            logger.info(f"        TP: {len(tp_in_doc)} | FP: {len(fp_in_doc)} | FN: {len(fn_in_doc)}")
        
        # Recopilar ejemplos
        for text in list(tp_in_doc)[:2]:
            source = source_by_text.get((doc_id, text), 'unknown')
            if len(examples['tp']) < 5:
                examples['tp'].append({'doc': doc_id[:8], 'text': text[:40], 'source': source})
            source_metrics[source]['tp'] += 1
            source_metrics[source]['total'] += 1
        
        for text in list(fp_in_doc)[:3]:
            source = source_by_text.get((doc_id, text), 'unknown')
            if len(examples['fp']) < 10:
                examples['fp'].append({'doc': doc_id[:8], 'text': text[:40], 'source': source})
            source_metrics[source]['fp'] += 1
            source_metrics[source]['total'] += 1
        
        for text in list(fn_in_doc)[:3]:
            if len(examples['fn']) < 10:
                examples['fn'].append({'doc': doc_id[:8], 'text': text[:40]})
    
    # Calcular métricas globales
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Precisión por fuente
    for source, m in source_metrics.items():
        if m['total'] > 0:
            m['precision'] = m['tp'] / m['total']
        else:
            m['precision'] = 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'examples': examples,
        'by_source': dict(source_metrics),
        'doc_stats': {
            'total': len(all_doc_ids),
            'with_predictions': len(pred_by_doc),
            'with_gt': len(gt_by_doc),
            'common': len(docs_common)
        }
    }


def print_results(metrics: dict):
    """Imprime resultados de forma legible."""
    print("\n" + "=" * 80)
    print("📊 EVALUACIÓN DEL PIPELINE DE DETECCIÓN DE PII")
    print("=" * 80)
    
    doc_stats = metrics.get('doc_stats', {})
    print(f"\n📁 ESTADÍSTICAS DE DOCUMENTOS:")
    print(f"   Total documentos: {doc_stats.get('total', 0)}")
    print(f"   Con predicciones: {doc_stats.get('with_predictions', 0)}")
    print(f"   Con GT: {doc_stats.get('with_gt', 0)}")
    print(f"   En ambos: {doc_stats.get('common', 0)}")
    
    print(f"\n🎯 MATRIZ DE CONFUSIÓN:")
    print(f"   TP (predicción ✓ en GT): {metrics['tp']}")
    print(f"   FP (predicción ✗ no en GT): {metrics['fp']}")
    print(f"   FN (GT ✗ no predicho): {metrics['fn']}")
    
    print(f"\n📈 MÉTRICAS GLOBALES:")
    print(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    print(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
    print(f"   F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.1f}%)")
    
    # Métricas por fuente
    if 'by_source' in metrics and metrics['by_source']:
        print(f"\n🔍 PRECISIÓN POR FUENTE DE CLASIFICACIÓN:")
        print(f"   {'Fuente':<20} {'Total':>8} {'TP':>8} {'FP':>8} {'Precisión':>10}")
        print(f"   {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        
        for source, m in sorted(metrics['by_source'].items(), key=lambda x: x[1]['total'], reverse=True):
            if m['total'] > 0:
                prec_pct = m['precision'] * 100
                print(f"   {source:<20} {m['total']:>8} {m['tp']:>8} {m['fp']:>8} {prec_pct:>9.1f}%")
    
    # Ejemplos
    if metrics['examples']['tp']:
        print(f"\n✅ EJEMPLOS DE TP (detección correcta):")
        for ex in metrics['examples']['tp']:
            print(f"   [{ex['doc']}] [{ex.get('source', '?'):15s}] \"{ex['text']}\"")
    
    if metrics['examples']['fp']:
        print(f"\n❌ EJEMPLOS DE FP (falsa alarma):")
        for ex in metrics['examples']['fp']:
            print(f"   [{ex['doc']}] [{ex.get('source', '?'):15s}] \"{ex['text']}\"")
    
    if metrics['examples']['fn']:
        print(f"\n⚠️  EJEMPLOS DE FN (falso negativo - PII perdido):")
        for ex in metrics['examples']['fn']:
            print(f"   [{ex['doc']}] \"{ex['text']}\"")
    
    print("=" * 80 + "\n")


def save_results(metrics: dict, output_path: str):
    """Guarda resultados en JSON."""
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'evaluation_type': 'pii_detection',
            'definitions': {
                'TP': 'Predicción que existe en GT (correcto)',
                'FP': 'Predicción que NO existe en GT (falsa alarma)',
                'FN': 'Entidad en GT que NO fue predicha (perdido)'
            }
        },
        'metrics': {
            'global': {
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'precision': round(metrics['precision'], 4),
                'recall': round(metrics['recall'], 4),
                'f1': round(metrics['f1'], 4)
            },
            'by_source': metrics.get('by_source', {}),
            'document_stats': metrics.get('doc_stats', {})
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[Eval] Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PII detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (auto-detect from repo structure)
  python evaluate_pipeline_filtering.py
  
  # Use default paths with debug output
  python evaluate_pipeline_filtering.py --debug
  
  # Use custom paths
  python evaluate_pipeline_filtering.py \\
    --results custom_results.json \\
    --ground-truth custom_gt.json \\
    --debug
        """
    )
    
    parser.add_argument(
        '--results', '-r',
        default=str(DEFAULT_RESULTS),
        help=f'Path to pipeline results JSON (default: {DEFAULT_RESULTS})'
    )
    parser.add_argument(
        '--ground-truth', '-g',
        default=str(DEFAULT_GT),
        help=f'Path to ground truth JSON (default: {DEFAULT_GT})'
    )
    parser.add_argument(
        '--output', '-o',
        default=str(DEFAULT_OUTPUT),
        help=f'Path to save evaluation results (default: {DEFAULT_OUTPUT})'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug output (document-level stats, missing docs, etc.)'
    )
    
    args = parser.parse_args()
    
    # Log inicio y configuración
    print("\n" + "=" * 80)
    print("🔍 PII DETECTION PIPELINE EVALUATION")
    print("=" * 80)
    logger.info(f"[Eval] Loading predictions from: {args.results}")
    logger.info(f"[Eval] Loading ground truth from: {args.ground_truth}")
    
    # Cargar datos
    try:
        results, pred_by_doc, source_by_text, doc_ids = load_predictions(args.results)
        gt_by_doc = load_ground_truth(args.ground_truth)
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluar
    logger.info(f"[Eval] Evaluating...")
    metrics = evaluate(pred_by_doc, source_by_text, gt_by_doc, debug=args.debug)
    
    # Mostrar resultados
    print_results(metrics)
    
    # Guardar
    save_results(metrics, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
