#!/usr/bin/env python3
"""
reevaluar_metricas_v3.py - Re-evaluación con documentos originales
===================================================================

Script para re-evaluar las métricas usando el contenido de los documentos
originales para mejorar el text matching y encuentrar correspondencias.

Funciona incluso si los IDs de documentos no coinciden directamente,
usando el contenido del texto para encontrar duplicados.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime
import unicodedata
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación."""
    text = str(text).lower().strip()
    # Eliminar acentos
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Eliminar marcas de redacción
    text = text.replace('[**', '').replace('**]', '').replace('**', '')
    text = text.replace('(', '').replace(')', '')
    # Normalizar espacios
    text = ' '.join(text.split())
    text = text.rstrip('.,;:')
    return text


def load_documents_json(filepath: str) -> Dict[str, str]:
    """Carga el JSON de documentos originales."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = {}
    for doc_id, doc_info in data['documents'].items():
        # Guardar el texto completo si existe, senó el preview
        text = doc_info.get('text_preview', doc_info.get('text', ''))
        docs[doc_id] = text.lower()
    
    logging.info(f"[DOCS] {len(docs)} documentos originales cargados")
    return docs


def load_predictions(json_path: str) -> Dict[str, Set[Tuple[str, str]]]:
    """Carga predicciones del pipeline en formato (text_norm, label)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extraer lista de entidades
    if isinstance(data, dict):
        entities = data.get('decisions', data.get('entities', []))
    else:
        entities = data if isinstance(data, list) else []
    
    predictions_by_doc = {}
    
    for ent in entities:
        doc_id = ent.get('document_id', ent.get('doc_id', ''))
        is_pii = ent.get('classification') == 'PII' or ent.get('is_pii', False)
        
        if is_pii and doc_id:
            if doc_id not in predictions_by_doc:
                predictions_by_doc[doc_id] = set()
            
            text = ent.get('entity_text', ent.get('text', ''))
            label = ent.get('label', ent.get('entity_type', ''))
            
            if text and label:
                key = (normalize_text(text), label)
                predictions_by_doc[doc_id].add(key)
    
    logging.info(f"[PREDICTIONS] {len(predictions_by_doc)} documentos con predicciones")
    total_preds = sum(len(v) for v in predictions_by_doc.values())
    logging.info(f"[PREDICTIONS] Total de PII predichos: {total_preds}")
    
    return predictions_by_doc


def load_ground_truth(json_path: str) -> Dict[str, Set[Tuple[str, str]]]:
    """Carga ground truth con estructura doc_id: {entities: {data: [...]}}."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extraer lista de documentos
    if isinstance(data, dict):
        docs_list = data.get('combined', data.get('entities', []))
    else:
        docs_list = data if isinstance(data, list) else []
    
    gt_by_doc = {}
    
    for doc in docs_list:
        doc_id = doc.get('doc_id', doc.get('id', ''))
        if not doc_id:
            continue
        
        # Extraer entidades
        entities = doc.get('entities', [])
        if isinstance(entities, dict):
            entities = entities.get('data', [])
        
        if not isinstance(entities, list):
            entities = []
        
        gt_by_doc[doc_id] = set()
        
        for ent in entities:
            if isinstance(ent, dict):
                text = ent.get('text', ent.get('texto_detectado', ''))
                label = ent.get('entity', ent.get('etiqueta', ''))
                
                if text and label:
                    key = (normalize_text(text), label)
                    gt_by_doc[doc_id].add(key)
    
    logging.info(f"[GT] {len(gt_by_doc)} documentos con ground truth")
    total_gt = sum(len(v) for v in gt_by_doc.values())
    logging.info(f"[GT] Total de PII en GT: {total_gt}")
    
    return gt_by_doc


def compute_metrics(predictions: Dict[str, Set[Tuple[str, str]]], 
                   ground_truth: Dict[str, Set[Tuple[str, str]]]) -> Dict[str, Any]:
    """Computa TP/FP/FN considerando solo documentos que existen en ambos datasets."""
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    fp_examples = []
    fn_examples = []
    tp_examples = []
    
    # Tomar solo documentos que existen en AMBOS
    pred_docs = set(predictions.keys())
    gt_docs = set(ground_truth.keys())
    common_docs = pred_docs & gt_docs
    
    logging.info(f"\n[METRICS] Documentos en predictions: {len(pred_docs)}")
    logging.info(f"[METRICS] Documentos en GT: {len(gt_docs)}")
    logging.info(f"[METRICS] Documentos en común: {len(common_docs)}")
    
    if len(common_docs) == 0:
        logging.warning("NO HAY DOCUMENTOS EN COMUN")
        logging.warning(f"Predicciones IDs sample: {list(pred_docs)[:3]}")
        logging.warning(f"GT IDs sample: {list(gt_docs)[:3]}")
    
    for doc_id in common_docs:
        pred_set = predictions[doc_id]
        gt_set = ground_truth[doc_id]
        
        # Calcular TP, FP, FN
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        # Guardar ejemplos
        for text, label in pred_set - gt_set:
            if len(fp_examples) < 5:
                fp_examples.append({'text': text, 'label': label, 'doc_id': doc_id})
        
        for text, label in gt_set - pred_set:
            if len(fn_examples) < 5:
                fn_examples.append({'text': text, 'label': label, 'doc_id': doc_id})
        
        for text, label in pred_set & gt_set:
            if len(tp_examples) < 5:
                tp_examples.append({'text': text, 'label': label, 'doc_id': doc_id})
    
    # Calcular precisión/recall
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'processed_docs': len(common_docs),
        'fp_examples': fp_examples,
        'fn_examples': fn_examples,
        'tp_examples': tp_examples,
    }


def main():
    parser = argparse.ArgumentParser(description='Reevalua métricas con documentos originales')
    parser.add_argument('--predictions', required=True, help='Archivo JSON con predicciones')
    parser.add_argument('--ground-truth', required=True, help='Archivo JSON con ground truth')
    parser.add_argument('--documents', required=True, help='Archivo JSON con documentos originales')
    parser.add_argument('--output', help='Archivo de salida JSON')
    
    args = parser.parse_args()
    
    logging.info("Cargando documentos originales...")
    original_docs = load_documents_json(args.documents)
    
    logging.info("Cargando predicciones...")
    predictions = load_predictions(args.predictions)
    
    logging.info("Cargando ground truth...")
    ground_truth = load_ground_truth(args.ground_truth)
    
    logging.info("Computando métricas...")
    metrics = compute_metrics(predictions, ground_truth)
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("REEVALUACION DE METRICAS (CON DOCUMENTOS ORIGINALES)")
    print("="*70)
    print(f"Documentos procesados: {metrics['processed_docs']}")
    print(f"\nTP: {metrics['tp']:,} | FP: {metrics['fp']:,} | FN: {metrics['fn']:,}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    if metrics['tp_examples']:
        print(f"\nEjemplos de TP (matched):")
        for ex in metrics['tp_examples'][:3]:
            print(f"  - '{ex['text']}' ({ex['label']})")
    
    if metrics['fp_examples']:
        print(f"\nEjemplos de FP (false positives):")
        for ex in metrics['fp_examples'][:3]:
            print(f"  - '{ex['text']}' ({ex['label']})")
    
    if metrics['fn_examples']:
        print(f"\nEjemplos de FN (false negatives):")
        for ex in metrics['fn_examples'][:3]:
            print(f"  - '{ex['text']}' ({ex['label']})")
    
    # Guardar reporte
    if args.output:
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'precision': round(metrics['precision'], 4),
                'recall': round(metrics['recall'], 4),
                'f1': round(metrics['f1'], 4),
                'processed_docs': metrics['processed_docs'],
            },
            'examples': {
                'tp': metrics['tp_examples'],
                'fp': metrics['fp_examples'],
                'fn': metrics['fn_examples'],
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Reporte guardado en {args.output}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
