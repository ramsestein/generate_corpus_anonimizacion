#!/usr/bin/env python3
from __future__ import annotations

"""
evaluate_pipeline_filtering.py - Evaluación ROBUSTA del pipeline
=================================================================

Evalúa el pipeline de detección de PII comparando:
1. Predicciones (pipeline_results_full.json con field 'decisions')
2. Ground Truth (combined_entidades_ANTIGUO.json con mapping doc_id -> entities)

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
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)-8s] %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# RUTAS POR DEFECTO
DEFAULT_RESULTS = PROJECT_ROOT / "outputs" / "pipeline_results_full.json"
DEFAULT_GT = PROJECT_ROOT / "outputs" / "combined_entidades_ANTIGUO.json"
DEFAULT_INPUT = PROJECT_ROOT / "entidades-procesadas-para-metricas.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation_results_final.json"


def normalize_text(text: Optional[str]) -> str:
    """Normaliza texto: minúsculas, sin tildes, limpia puntuación decorativa y colapsa espacios."""
    if not text:
        return ''

    # Lower + strip tildes
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    # Quitar puntuación/asteriscos/comillas/relleno común en spans del pipeline
    text = re.sub(r"[\*\"'`´’”¿¡!?,.;:()\[\]{}<>]", " ", text)
    # Quitar dobles espacios
    text = ' '.join(text.split())
    return text


def token_overlap_score(text_a: str, text_b: str) -> float:
    """Calcula solapamiento de tokens para permitir matches relajados controlados."""
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))


def shared_numbers(text_a: str, text_b: str) -> bool:
    """Devuelve True si ambos textos comparten algún número, útil para direcciones o fechas."""
    nums_a = set(re.findall(r"\d+", text_a))
    nums_b = set(re.findall(r"\d+", text_b))
    if not nums_a or not nums_b:
        return False
    return bool(nums_a & nums_b)


def to_int(value):
    """Convierte a int si es posible; de lo contrario devuelve None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_entity_fields(entity_obj: dict | str) -> dict:
    """Extrae campos normalizados de una entidad (label, texto, spans)."""
    if isinstance(entity_obj, str):
        return {
            'label': '',
            'raw_text': entity_obj,
            'text': normalize_text(entity_obj),
            'start': None,
            'end': None,
        }

    label = (
        entity_obj.get('label')
        or entity_obj.get('entity')
        or entity_obj.get('entity_label')
        or entity_obj.get('entity_type')
        or entity_obj.get('type')
        or entity_obj.get('category')
        or ''
    )

    raw_text = (
        entity_obj.get('text')
        or entity_obj.get('entity_text')
        or entity_obj.get('value')
        or ''
    )

    start = to_int(entity_obj.get('start') or entity_obj.get('begin') or entity_obj.get('span_start'))
    end = to_int(entity_obj.get('end') or entity_obj.get('span_end'))

    return {
        'label': str(label),
        'raw_text': raw_text,
        'text': normalize_text(raw_text),
        'start': start,
        'end': end,
    }


def text_matches_gt(text_norm: str, gt_texts: set) -> bool:
    """Compatibility helper (no longer used)."""
    return text_norm in gt_texts


def load_predictions(results_path: str) -> tuple:
    """
    Carga predicciones del pipeline.
    Retorna: (results dict, predicciones por doc como lista de entidades, doc_ids)
    - Deduplica predicciones iguales (mismo doc, label, span/text) para no inflar FP.
    """
    logger.info(f"[Eval] Loading predictions from: {results_path}")

    if not Path(results_path).exists():
        logger.error(f"❌ Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    pred_by_doc: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    decisions = results.get('decisions', [])
    logger.info(f"   Total decisions in results: {len(decisions)}")

    for d in decisions:
        doc_id = d.get('document_id', '')
        if not doc_id:
            continue

        ent = extract_entity_fields(d)
        ent['source'] = d.get('classification_source', 'unknown')
        ent['doc_id'] = doc_id
        key = (doc_id, ent['label'], ent['start'], ent['end'], ent['text'])
        if key in seen:
            continue
        seen.add(key)
        pred_by_doc[doc_id].append(ent)

    doc_ids = set(pred_by_doc.keys())
    logger.info(f"   Unique documents with predictions: {len(doc_ids)}")

    return results, pred_by_doc, doc_ids


def load_ground_truth(gt_path: str) -> dict:
    """
    Carga Ground Truth desde JSON.
    Soporta:
    - Dict con clave 'combined' -> lista de docs (formato usado en combined_entidades_ANTIGUO)
    - Dict mapping doc_id -> lista/dict con entidades
    - Lista plana con doc_id en cada entidad
    Retorna: dict {doc_id -> list[entity dicts]}
    """
    logger.info(f"[Eval] Loading ground truth from: {gt_path}")

    if not Path(gt_path).exists():
        logger.error(f"❌ Ground truth file not found: {gt_path}")
        sys.exit(1)

    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    grouped: dict[str, list[dict]] = defaultdict(list)

    # Caso 1: dict con 'combined'
    if isinstance(gt_data, dict) and 'combined' in gt_data:
        combined_list = gt_data.get('combined', [])
        logger.info(f"   GT structure: dict['combined'] with {len(combined_list)} docs")
        for doc_entry in combined_list:
            doc_id = doc_entry.get('doc_id') or doc_entry.get('document_id') or ''
            if not doc_id:
                continue
            ents_obj = doc_entry.get('entities') or {}
            ents = []
            if isinstance(ents_obj, dict):
                ents = ents_obj.get('data', []) or ents_obj.get('entities', []) or []
            elif isinstance(ents_obj, list):
                ents = ents_obj
            for ent in ents:
                grouped[doc_id].append(extract_entity_fields(ent))

    # Caso 2: dict mapping doc_id -> entidades
    elif isinstance(gt_data, dict):
        logger.info(f"   GT structure: dict with {len(gt_data)} entries")
        for doc_id, doc_obj in gt_data.items():
            ents = []
            if isinstance(doc_obj, list):
                ents = doc_obj
            elif isinstance(doc_obj, dict):
                if 'data' in doc_obj:
                    ents = doc_obj['data']
                elif 'entities' in doc_obj:
                    ents = doc_obj['entities']
                elif 'text' in doc_obj:
                    ents = [doc_obj['text']]
            for ent in ents:
                grouped[doc_id].append(extract_entity_fields(ent))

    # Caso 3: lista plana
    elif isinstance(gt_data, list):
        logger.info(f"   GT structure: list with {len(gt_data)} entries")
        for ent in gt_data:
            if not isinstance(ent, dict):
                continue
            doc_id = ent.get('doc_id') or ent.get('document_id')
            if not doc_id:
                continue
            grouped[doc_id].append(extract_entity_fields(ent))

    # Deduplicar entidades por doc para evitar duplicados en GT
    deduped = {}
    for doc_id, ents in grouped.items():
        seen = set()
        uniq = []
        for ent in ents:
            key = (ent.get('label'), ent.get('start'), ent.get('end'), ent.get('text'))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(ent)
        deduped[doc_id] = uniq

    logger.info(f"   Unique documents in GT: {len(deduped)}")
    return deduped


def entities_match(pred: dict, gt: dict) -> bool:
    """Match estricto: mismo label y mismo span si existe; si no, mismo texto normalizado."""
    if pred.get('label') != gt.get('label'):
        return False

    p_start, p_end = pred.get('start'), pred.get('end')
    g_start, g_end = gt.get('start'), gt.get('end')
    if p_start is not None and p_end is not None and g_start is not None and g_end is not None:
        return p_start == g_start and p_end == g_end

    # Fallback a texto: exacto o contención para cubrir signos/puntuación removidos
    if pred.get('text') and gt.get('text'):
        if pred['text'] == gt['text']:
            return True
        # Contención mínima 3 chars para evitar falsos matches triviales
        if len(pred['text']) >= 3 and len(gt['text']) >= 3:
            if pred['text'] in gt['text'] or gt['text'] in pred['text']:
                return True

    return False


def evaluate(pred_by_doc: dict, gt_by_doc: dict, debug: bool = False):
    """
    Evalúa el pipeline comparando predicciones contra GT con criterios consistentes:
    - Mismo document_id
    - Mismo label
    - Match por span (start/end) si ambos lo tienen; si no, match por texto normalizado
    """
    tp = fp = fn = 0
    tp_relaxed = 0
    examples = {'tp': [], 'fp': [], 'fn': []}

    source_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'total': 0})

    all_doc_ids = set(pred_by_doc.keys()) | set(gt_by_doc.keys())
    docs_only_pred = set(pred_by_doc.keys()) - set(gt_by_doc.keys())
    docs_only_gt = set(gt_by_doc.keys()) - set(pred_by_doc.keys())
    docs_common = set(pred_by_doc.keys()) & set(gt_by_doc.keys())

    if debug:
        logger.info("[Debug] Document overlap analysis:")
        logger.info(f"        Predictions only: {len(docs_only_pred)}")
        logger.info(f"        GT only: {len(docs_only_gt)}")
        logger.info(f"        Both: {len(docs_common)}")
        for sample in list(docs_common)[:3]:
            logger.info(
                f"        Doc {sample}: pred={len(pred_by_doc.get(sample, []))} | GT={len(gt_by_doc.get(sample, []))}"
            )

    doc_debug_limit = 5
    for doc_id in all_doc_ids:
        preds = pred_by_doc.get(doc_id, [])
        gts = gt_by_doc.get(doc_id, [])
        matched_gt = [False] * len(gts)

        tp_doc = fp_doc = fn_doc = 0

        # Evaluar predicciones
        for pred_ent in preds:
            match_idx = None
            for idx, gt_ent in enumerate(gts):
                if matched_gt[idx]:
                    continue
                if entities_match(pred_ent, gt_ent):
                    match_idx = idx
                    break

            if match_idx is not None:
                matched_gt[match_idx] = True
                tp += 1
                tp_doc += 1

                if len(examples['tp']) < 5:
                    examples['tp'].append({
                        'doc': doc_id,
                        'label': pred_ent.get('label', ''),
                        'text': pred_ent.get('raw_text', '')[:80],
                        'match': 'span' if (pred_ent.get('start') is not None and pred_ent.get('end') is not None) else 'text',
                        'source': pred_ent.get('source', 'unknown'),
                        'span': (pred_ent.get('start'), pred_ent.get('end')),
                    })

                src = pred_ent.get('source', 'unknown')
                source_metrics[src]['tp'] += 1
                source_metrics[src]['total'] += 1
            else:
                # Segundo intento: emparejar solo por texto si no hay match estricto
                relaxed_idx = None
                for idx, gt_ent in enumerate(gts):
                    if matched_gt[idx]:
                        continue
                    if not pred_ent.get('text') or not gt_ent.get('text'):
                        continue

                    pred_text = pred_ent['text']
                    gt_text = gt_ent['text']

                    # Igual exacto o contención larga
                    if pred_text == gt_text:
                        relaxed_idx = idx
                        break
                    if len(pred_text) >= 5 and len(gt_text) >= 5:
                        if pred_text in gt_text or gt_text in pred_text:
                            relaxed_idx = idx
                            break

                    # Solapamiento de tokens alto evita sobrecontar, pero rescata variaciones
                    overlap = token_overlap_score(pred_text, gt_text)
                    if overlap >= 0.8 and len(pred_text.split()) >= 2 and len(gt_text.split()) >= 2:
                        relaxed_idx = idx
                        break

                    # Si comparten números (direcciones/fechas) y tokens solapan de forma moderada
                    if shared_numbers(pred_text, gt_text) and overlap >= 0.5:
                        relaxed_idx = idx
                        break

                if relaxed_idx is not None:
                    matched_gt[relaxed_idx] = True
                    tp += 1
                    tp_relaxed += 1
                    tp_doc += 1
                    if len(examples['tp']) < 5:
                        examples['tp'].append({
                            'doc': doc_id,
                            'label': pred_ent.get('label', ''),
                            'text': pred_ent.get('raw_text', '')[:80],
                            'match': 'text_relaxed',
                            'source': pred_ent.get('source', 'unknown'),
                            'span': (pred_ent.get('start'), pred_ent.get('end')),
                        })
                    src = pred_ent.get('source', 'unknown')
                    source_metrics[src]['tp'] += 1
                    source_metrics[src]['total'] += 1
                else:
                    fp += 1
                    fp_doc += 1
                    if len(examples['fp']) < 10:
                        examples['fp'].append({
                            'doc': doc_id,
                            'label': pred_ent.get('label', ''),
                            'text': pred_ent.get('raw_text', '')[:80],
                            'span': (pred_ent.get('start'), pred_ent.get('end')),
                            'source': pred_ent.get('source', 'unknown'),
                        })
                    src = pred_ent.get('source', 'unknown')
                    source_metrics[src]['fp'] += 1
                    source_metrics[src]['total'] += 1

        # FNs: GT no emparejados
        for idx, gt_ent in enumerate(gts):
            if not matched_gt[idx]:
                fn += 1
                fn_doc += 1
                if len(examples['fn']) < 10:
                    examples['fn'].append({
                        'doc': doc_id,
                        'label': gt_ent.get('label', ''),
                        'text': gt_ent.get('raw_text', '')[:80],
                        'span': (gt_ent.get('start'), gt_ent.get('end')),
                    })

        if debug and doc_debug_limit > 0:
            logger.info(
                f"[Debug] Doc {doc_id}: GT={len(gts)} Pred={len(preds)} | TP={tp_doc} FP={fp_doc} FN={fn_doc}"
            )

            # Mostrar casos donde antes se contaba FP pero coincide por label+texto/spans
            sample_fixes = []
            for pred_ent in preds[:5]:
                for gt_ent in gts[:5]:
                    if pred_ent.get('label') == gt_ent.get('label'):
                        # Coincidencia de texto aunque span falte
                        if pred_ent.get('text') and pred_ent.get('text') == gt_ent.get('text'):
                            sample_fixes.append((pred_ent, gt_ent))
                            break
                        # Coincidencia de span
                        if (
                            pred_ent.get('start') is not None
                            and pred_ent.get('end') is not None
                            and pred_ent.get('start') == gt_ent.get('start')
                            and pred_ent.get('end') == gt_ent.get('end')
                        ):
                            sample_fixes.append((pred_ent, gt_ent))
                            break
            for pred_ent, gt_ent in sample_fixes[:2]:
                logger.info(
                    "        ↳ Matched by label+text/span: "
                    f"pred='" + pred_ent.get('raw_text', '')[:60] + "' "
                    f"gt='" + gt_ent.get('raw_text', '')[:60] + "' "
                    f"label={pred_ent.get('label','')} "
                    f"span_pred={(pred_ent.get('start'), pred_ent.get('end'))} "
                    f"span_gt={(gt_ent.get('start'), gt_ent.get('end'))}"
                )
            doc_debug_limit -= 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    for source, m in source_metrics.items():
        m['precision'] = m['tp'] / m['total'] if m['total'] else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tp_relaxed': tp_relaxed,
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
    if metrics.get('tp_relaxed', 0):
        print(f"   TP por fallback texto (relajado): {metrics['tp_relaxed']}")
    
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
            span_txt = f" span={ex['span']}" if ex.get('span') else ''
            print(f"   [{ex['doc']}] [{ex.get('source', '?'):15s}] label={ex.get('label','')} {ex.get('match','')} {span_txt} \"{ex['text']}\"")
    
    if metrics['examples']['fp']:
        print(f"\n❌ EJEMPLOS DE FP (falsa alarma):")
        for ex in metrics['examples']['fp']:
            print(
                f"   [{ex['doc']}] [{ex.get('source', '?'):15s}] label={ex.get('label','')} span={ex.get('span')} \"{ex['text']}\""
            )
    
    if metrics['examples']['fn']:
        print(f"\n⚠️  EJEMPLOS DE FN (falso negativo - PII perdido):")
        for ex in metrics['examples']['fn']:
            print(f"   [{ex['doc']}] label={ex.get('label','')} span={ex.get('span')} \"{ex['text']}\"")
    
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
        results, pred_by_doc, doc_ids = load_predictions(args.results)
        gt_by_doc = load_ground_truth(args.ground_truth)
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        sys.exit(1)

    # Evaluar
    logger.info(f"[Eval] Evaluating...")
    metrics = evaluate(pred_by_doc, gt_by_doc, debug=args.debug)
    
    # Mostrar resultados
    print_results(metrics)
    
    # Guardar
    save_results(metrics, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
