#!/usr/bin/env python3
"""
reevaluar_metricas.py - Re-evaluación detallada de métricas
===========================================================

Script para re-evaluar las métricas de detección PII comparando:
1. Predicciones del pipeline
2. Ground truth de los documentos

Calcula TP/FP/FN/TN correctamente considerando solo documentos procesados.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
from datetime import datetime
import unicodedata
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación."""
    text = text.lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = ' '.join(text.split())
    return text


def load_predictions(json_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Carga predicciones agrupadas por documento."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extraer entidades del formato del pipeline
    entities = data
    if isinstance(data, dict):
        entities = data.get('decisions', data.get('entities', data.get('detecciones', data.get('results', []))))
    
    predictions_by_doc = defaultdict(list)
    for ent in entities:
        doc_id = ent.get('document_id', ent.get('doc_id', 'UNKNOWN'))
        is_pii = ent.get('classification') == 'PII' or ent.get('is_pii', False)
        
        if is_pii:
            predictions_by_doc[doc_id].append({
                'text': ent.get('entity_text', ent.get('text', '')),
                'label': ent.get('label', ent.get('entity_type', '')),
                'source': ent.get('classification_source', 'unknown'),
                'confidence': ent.get('confidence', 0),
            })
    
    return dict(predictions_by_doc)


def load_ground_truth(json_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Carga ground truth de archivo JSON consolidado."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extraer entidades
    entities = data
    if isinstance(data, dict):
        entities = data.get('combined', data.get('entities', data.get('detecciones', data.get('results', []))))
    
    gt_by_doc = defaultdict(list)
    for ent in entities:
        doc_id = ent.get('document_id', ent.get('doc_id', 'UNKNOWN'))
        gt_by_doc[doc_id].append({
            'text': ent.get('texto_detectado', ent.get('entity_text', ent.get('text', ''))),
            'label': ent.get('etiqueta', ent.get('entity_type', ent.get('entity', ''))),
        })
    
    return dict(gt_by_doc)


def compute_metrics(predictions: Dict[str, List[Dict[str, Any]]], 
                   ground_truth: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Computa TP/FP/FN para cada documento y totales."""
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    document_metrics = {}
    fp_examples = []
    fn_examples = []
    
    processed_docs = set(predictions.keys())
    
    for doc_id in processed_docs:
        pred_list = predictions.get(doc_id, [])
        gt_list = ground_truth.get(doc_id, [])
        
        # Normalizar para comparación
        pred_normalized = {
            (normalize_text(p['text']), p['label']) 
            for p in pred_list if p['text'] and p['label']
        }
        gt_normalized = {
            (normalize_text(g['text']), g['label'])
            for g in gt_list if g['text'] and g['label']
        }
        
        # Calcular métricas
        tp = len(pred_normalized & gt_normalized)
        fp = len(pred_normalized - gt_normalized)
        fn = len(gt_normalized - pred_normalized)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        document_metrics[doc_id] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'predictions_count': len(pred_list),
            'ground_truth_count': len(gt_list),
        }
        
        # Guardar ejemplos de FP y FN
        if fp > 0:
            fp_set = pred_normalized - gt_normalized
            for text, label in list(fp_set)[:2]:
                fp_examples.append({
                    'doc_id': doc_id,
                    'text': text,
                    'label': label,
                    'type': 'False Positive'
                })
        
        if fn > 0:
            fn_set = gt_normalized - pred_normalized
            for text, label in list(fn_set)[:2]:
                fn_examples.append({
                    'doc_id': doc_id,
                    'text': text,
                    'label': label,
                    'type': 'False Negative'
                })
    
    # Calcular métricas globales
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'global_metrics': {
            'tp': tp_total,
            'fp': fp_total,
            'fn': fn_total,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'processed_documents': len(processed_docs),
            'gt_documents': len(ground_truth),
        },
        'document_metrics': document_metrics,
        'error_examples': {
            'false_positives': fp_examples[:10],
            'false_negatives': fn_examples[:10],
        }
    }


def generate_report(metrics: Dict[str, Any], output_path: str):
    """Genera reporte de re-evaluación."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Reporte guardado en {output_path}")
    
    # Mostrar resumen
    gm = metrics['global_metrics']
    print("\n" + "=" * 70)
    print("REEVALUACION DE METRICAS")
    print("=" * 70)
    print(f"\nDocumentos procesados: {gm['processed_documents']}")
    print(f"Documentos en GT: {gm['gt_documents']}")
    print(f"\nTP: {gm['tp']} | FP: {gm['fp']} | FN: {gm['fn']}")
    print(f"Precision: {gm['precision']:.4f} | Recall: {gm['recall']:.4f} | F1: {gm['f1']:.4f}")
    print("\n" + "=" * 70)
    
    if gm['fp'] > gm['tp'] * 0.5:
        print("\nADVERTENCIA: Muchos falsos positivos (FP > TP*0.5)")
        print("  -> Considera aumentar el threshold de confianza")
    
    if gm['recall'] < 0.7:
        print("\nADVERTENCIA: Recall muy bajo (< 0.7)")
        print("  -> El modelo pierde demasiados PII reales")


def main():
    parser = argparse.ArgumentParser(
        description="Re-evalua metricas de deteccion PII"
    )
    parser.add_argument('--predictions', '-p', required=True, 
                       help='Archivo JSON con predicciones del pipeline')
    parser.add_argument('--ground-truth', '-g', default='combined_entidades_ANTIGUO.json',
                       help='Archivo JSON con ground truth')
    parser.add_argument('--output', '-o', default='outputs/reevaluation_report.json',
                       help='Archivo de salida para reporte')
    
    args = parser.parse_args()
    
    # Validar archivos
    if not Path(args.predictions).exists():
        logger.error(f"Predicciones no encontradas: {args.predictions}")
        return 1
    
    if not Path(args.ground_truth).exists():
        logger.error(f"Ground truth no encontrado: {args.ground_truth}")
        return 1
    
    # Cargar datos
    logger.info(f"Cargando predicciones desde {args.predictions}")
    predictions = load_predictions(str(args.predictions))
    logger.info(f"  -> {len(predictions)} documentos con predicciones")
    
    logger.info(f"Cargando ground truth desde {args.ground_truth}")
    ground_truth = load_ground_truth(str(args.ground_truth))
    logger.info(f"  -> {len(ground_truth)} documentos en ground truth")
    
    # Computar métricas
    logger.info("Computando metricas...")
    metrics = compute_metrics(predictions, ground_truth)
    
    # Generar reporte
    generate_report(metrics, str(args.output))
    
    return 0


if __name__ == "__main__":
    exit(main())
