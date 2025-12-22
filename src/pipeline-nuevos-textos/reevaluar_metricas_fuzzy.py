#!/usr/bin/env python3
"""
reevaluar_metricas_fuzzy.py - Matching de documentos por similitud de contenido
===============================================================================

Como los IDs de documentos no coinciden directamente, este script:
1. Toma los primeros N caracteres de cada documento (como firma única)
2. Busca documentos con similitud de contenido entre predictions y GT
3. Computa métricas basadas en matches de contenido
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime
import unicodedata
import logging
from difflib import SequenceMatcher
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación."""
    text = str(text).lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = text.replace('[**', '').replace('**]', '').replace('**', '')
    text = text.replace('(', '').replace(')', '')
    text = ' '.join(text.split())
    text = text.rstrip('.,;:')
    return text


def get_text_signature(text: str, length: int = 500) -> str:
    """Obtiene una firma única de los primeros N caracteres del texto."""
    return normalize_text(text)[:length]


def load_documents_json(filepath: str) -> Dict[str, str]:
    """Carga el JSON de documentos originales para comparación de contenido."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = {}
    for doc_id, doc_info in data.get('documents', {}).items():
        text = doc_info.get('text', doc_info.get('text_preview', ''))
        docs[doc_id] = text.lower()
    
    logger.info(f"[DOCS] {len(docs)} documentos originales cargados")
    return docs


def load_predictions(json_path: str) -> Dict[str, List[Dict]]:
    """Carga predicciones del pipeline."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        entities = data.get('decisions', data.get('entities', []))
    else:
        entities = data if isinstance(data, list) else []
    
    predictions_by_doc = defaultdict(list)
    
    for ent in entities:
        doc_id = ent.get('document_id', ent.get('doc_id', ''))
        is_pii = ent.get('classification') == 'PII' or ent.get('is_pii', False)
        
        if is_pii and doc_id:
            text = ent.get('entity_text', ent.get('text', ''))
            label = ent.get('label', ent.get('entity_type', ''))
            
            if text and label:
                predictions_by_doc[doc_id].append({
                    'text': text,
                    'label': label,
                    'text_norm': normalize_text(text)
                })
    
    total_preds = sum(len(v) for v in predictions_by_doc.values())
    logger.info(f"[PREDICTIONS] {len(predictions_by_doc)} documentos con predicciones")
    logger.info(f"[PREDICTIONS] Total de PII predichos: {total_preds}")
    
    return dict(predictions_by_doc)


def load_ground_truth(json_path: str) -> Dict[str, List[Dict]]:
    """Carga ground truth con estructura doc_id: {entities: {data: [...]}}."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        docs_list = data.get('combined', data.get('entities', []))
    else:
        docs_list = data if isinstance(data, list) else []
    
    gt_by_doc = defaultdict(list)
    
    for doc in docs_list:
        doc_id = doc.get('doc_id', doc.get('id', ''))
        if not doc_id:
            continue
        
        entities = doc.get('entities', [])
        if isinstance(entities, dict):
            entities = entities.get('data', [])
        
        if not isinstance(entities, list):
            entities = []
        
        for ent in entities:
            if isinstance(ent, dict):
                text = ent.get('text', ent.get('texto_detectado', ''))
                label = ent.get('entity', ent.get('etiqueta', ''))
                
                if text and label:
                    gt_by_doc[doc_id].append({
                        'text': text,
                        'label': label,
                        'text_norm': normalize_text(text)
                    })
    
    total_gt = sum(len(v) for v in gt_by_doc.values())
    logger.info(f"[GT] {len(gt_by_doc)} documentos con ground truth")
    logger.info(f"[GT] Total de PII en GT: {total_gt}")
    
    return dict(gt_by_doc)


def match_documents(documents: Dict[str, str], 
                   predictions_keys: List[str],
                   gt_keys: List[str]) -> Dict[str, str]:
    """
    Intenta hacer matching entre documentos de predicciones y GT basado en similitud.
    
    Returns: Dict[pred_doc_id] -> gt_doc_id
    """
    matches = {}
    
    # Crear signaturas para documentos
    doc_signatures = {}
    for doc_id, text in documents.items():
        sig = get_text_signature(text, 1000)
        doc_signatures[doc_id] = sig
    
    # Para cada prediction document, buscar mejor match en GT documents
    for pred_id in predictions_keys:
        best_match = None
        best_score = 0.0
        
        for gt_id in gt_keys:
            if gt_id in doc_signatures and pred_id not in doc_signatures:
                # No podemos comparar documentos que no existen
                continue
            
            if pred_id in doc_signatures and gt_id in doc_signatures:
                # Calcular similitud usando difflib
                similarity = SequenceMatcher(None, 
                                           doc_signatures[pred_id],
                                           doc_signatures[gt_id]).ratio()
                
                if similarity > best_score and similarity > 0.7:  # Threshold mínimo
                    best_score = similarity
                    best_match = gt_id
        
        if best_match:
            matches[pred_id] = best_match
            logger.debug(f"Matched {pred_id} -> {best_match} (score: {best_score:.3f})")
    
    logger.info(f"[MATCHING] {len(matches)} prediction documents matched a GT")
    return matches


def compute_metrics_with_matching(predictions: Dict[str, List[Dict]],
                                 ground_truth: Dict[str, List[Dict]],
                                 documents: Dict[str, str]) -> Dict[str, Any]:
    """Computa métricas usando matching de contenido entre documentos."""
    
    logger.info("\n[MATCHING] Intentando hacer matching por similitud de contenido...")
    document_matches = match_documents(documents, 
                                       list(predictions.keys()), 
                                       list(ground_truth.keys()))
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    fp_examples = []
    fn_examples = []
    tp_examples = []
    
    # Procesarmatches encontrados
    for pred_doc_id, gt_doc_id in document_matches.items():
        pred_entities = predictions[pred_doc_id]
        gt_entities = ground_truth[gt_doc_id]
        
        # Crear sets para comparación
        pred_set = {(ent['text_norm'], ent['label']) for ent in pred_entities}
        gt_set = {(ent['text_norm'], ent['label']) for ent in gt_entities}
        
        # Calcular TP, FP, FN
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        # Guardar ejemplos
        for text, label in (pred_set - gt_set)[:5]:
            if len(fp_examples) < 10:
                fp_examples.append({'text': text, 'label': label})
        
        for text, label in (gt_set - pred_set)[:5]:
            if len(fn_examples) < 10:
                fn_examples.append({'text': text, 'label': label})
        
        for text, label in (pred_set & gt_set)[:5]:
            if len(tp_examples) < 10:
                tp_examples.append({'text': text, 'label': label})
    
    # Procesar predicciones sin match (todas FP)
    unmatched_pred_count = len(predictions) - len(document_matches)
    if unmatched_pred_count > 0:
        logger.warning(f"[WARNING] {unmatched_pred_count} prediction documents sin matching")
        for pred_doc_id in predictions:
            if pred_doc_id not in document_matches:
                for ent in predictions[pred_doc_id]:
                    fp_total += 1
                    if len(fp_examples) < 10:
                        fp_examples.append({'text': ent['text_norm'], 'label': ent['label']})
    
    # Procesar GT sin match (todas FN)
    unmatched_gt_count = len(ground_truth) - len(set(document_matches.values()))
    if unmatched_gt_count > 0:
        logger.warning(f"[WARNING] {unmatched_gt_count} GT documents sin matching")
        gt_matched = set(document_matches.values())
        for gt_doc_id in ground_truth:
            if gt_doc_id not in gt_matched:
                for ent in ground_truth[gt_doc_id]:
                    fn_total += 1
                    if len(fn_examples) < 10:
                        fn_examples.append({'text': ent['text_norm'], 'label': ent['label']})
    
    # Calcular precisión/recall
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'matched_documents': len(document_matches),
        'unmatched_predictions': unmatched_pred_count,
        'unmatched_gt': unmatched_gt_count,
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fp_examples': fp_examples,
        'fn_examples': fn_examples,
        'tp_examples': tp_examples,
    }


def main():
    parser = argparse.ArgumentParser(description='Reevalúa métricas con matching fuzzy de contenido')
    parser.add_argument('--predictions', required=True, help='Archivo JSON con predicciones')
    parser.add_argument('--ground-truth', required=True, help='Archivo JSON con ground truth')
    parser.add_argument('--documents', required=True, help='Archivo JSON con documentos originales')
    parser.add_argument('--output', help='Archivo de salida JSON')
    
    args = parser.parse_args()
    
    logger.info("Cargando documentos originales...")
    original_docs = load_documents_json(args.documents)
    
    logger.info("Cargando predicciones...")
    predictions = load_predictions(args.predictions)
    
    logger.info("Cargando ground truth...")
    ground_truth = load_ground_truth(args.ground_truth)
    
    logger.info("Computando métricas con matching...")
    metrics = compute_metrics_with_matching(predictions, ground_truth, original_docs)
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("REEVALUACION DE METRICAS (FUZZY MATCHING DE CONTENIDO)")
    print("="*70)
    print(f"Documentos matched: {metrics['matched_documents']}")
    print(f"Predicciones sin matching: {metrics['unmatched_predictions']}")
    print(f"GT sin matching: {metrics['unmatched_gt']}")
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
            'method': 'fuzzy_matching_by_content',
            'metrics': {
                'matched_documents': metrics['matched_documents'],
                'unmatched_predictions': metrics['unmatched_predictions'],
                'unmatched_gt': metrics['unmatched_gt'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'precision': round(metrics['precision'], 4),
                'recall': round(metrics['recall'], 4),
                'f1': round(metrics['f1'], 4),
            },
            'examples': {
                'tp': metrics['tp_examples'][:5],
                'fp': metrics['fp_examples'][:5],
                'fn': metrics['fn_examples'][:5],
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reporte guardado en {args.output}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
