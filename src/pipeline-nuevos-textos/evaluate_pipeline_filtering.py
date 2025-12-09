#!/usr/bin/env python3
"""
evaluate_pipeline_filtering.py - Evaluación CORRECTA del pipeline
==================================================================

Evalúa el pipeline comparando:
1. Entidades de ENTRADA (entidades-procesadas-para-metricas.json)
2. Entidades de SALIDA (resultados_completo.json -> entidades con classification=PII)
3. Ground Truth (corpus/ANTIGUO/entidades/)

NUEVO FLUJO (invertido):
- SetFit clasifica: PII (va directo) o RUIDO (pasa a rescate)
- Dict Filters rescata: whitelist → PII, blacklist → descartado
- LLM rescata casos dudosos

Métricas calculadas:
- TP: Entidades PII que están en GT (conservamos PII real)
- FP: Entidades PII que NO están en GT (conservamos ruido)
- TN: Entidades descartadas que NO están en GT (filtrado correcto)
- FN: Entidades descartadas que SÍ están en GT (perdimos PII real)

TRAZABILIDAD:
- Distingue PII directo de SetFit vs rescatados por listas/LLM
- Muestra efectividad de cada etapa de rescate

IMPORTANTE: Usa coincidencia parcial porque los modelos NER fragmentan textos.

USO:
    python evaluate_pipeline_filtering.py
    python evaluate_pipeline_filtering.py --results outputs/resultados_completo.json
"""

import argparse
import json
import logging
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)-8s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "outputs" / "resultados_completo.json"
DEFAULT_INPUT = PROJECT_ROOT / "entidades-procesadas-para-metricas.json"
DEFAULT_GT_DIR = PROJECT_ROOT / "corpus" / "ANTIGUO" / "entidades"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "pipeline_evaluation.json"


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación."""
    text = text.lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    return ' '.join(text.split())


def text_matches_gt(text_norm: str, gt_texts: set) -> bool:
    """
    Verifica si un texto coincide con algún texto del GT.
    Usa coincidencia exacta o parcial (contención).
    """
    if text_norm in gt_texts:
        return True
    
    for gt_text in gt_texts:
        if len(text_norm) >= 3 and len(gt_text) >= 3:
            if text_norm in gt_text or gt_text in text_norm:
                return True
    
    return False


def load_data(results_path: str, input_path: str, gt_dir: str):
    """Carga todos los datos necesarios."""
    # Cargar resultados del pipeline
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Cargar entrada
    with open(input_path, 'r', encoding='utf-8') as f:
        entrada = json.load(f)
    
    # Extraer doc_ids procesados
    doc_ids = set()
    for d in results.get('decisions', []):
        if d.get('document_id'):
            doc_ids.add(d['document_id'])
    
    # Cargar GT para docs procesados
    gt_by_doc = {}
    gt_path = Path(gt_dir)

    # Si 'gt_dir' apunta a un archivo combinado (ej. dataset_unificado.json), cargarlo
    if gt_path.is_file():
        with open(gt_path, 'r', encoding='utf-8') as f:
            combined = json.load(f)

        # Si el archivo es un dict mapping doc_id -> list[entities]
        if isinstance(combined, dict):
            for doc_id in doc_ids:
                doc_obj = combined.get(doc_id) or combined.get(str(doc_id))
                
                # Normalizar estructura: obtener lista de entidades
                ents = []
                if isinstance(doc_obj, list):
                    ents = doc_obj
                elif isinstance(doc_obj, dict):
                    # Caso dataset_unificado.json: {"id": "...", "data": [...]}
                    if 'data' in doc_obj:
                        ents = doc_obj['data']
                    elif 'entities' in doc_obj:
                        ents = doc_obj['entities']
                    else:
                        ents = []
                
                texts = set()
                # soportar formatos: lista de dicts o lista de strings
                for e in ents:
                    if isinstance(e, dict):
                        txt = e.get('text') or e.get('entity_text') or ''
                        if txt:
                            texts.add(normalize_text(txt))
                    elif isinstance(e, str):
                        texts.add(normalize_text(e))
                if texts:
                    gt_by_doc[doc_id] = texts
        else:
            # Si es una lista plana de entidades con campo 'doc_id', agruparlas
            if isinstance(combined, list):
                grouped = defaultdict(set)
                for e in combined:
                    doc = e.get('doc_id') or e.get('document_id')
                    if not doc:
                        continue
                    txt = e.get('text') or e.get('entity_text') or ''
                    if txt:
                        grouped[doc].add(normalize_text(txt))
                for doc_id in doc_ids:
                    if grouped.get(doc_id):
                        gt_by_doc[doc_id] = grouped[doc_id]
    else:
        # gt_dir es un directorio con archivos por documento
        for doc_id in doc_ids:
            gt_file = gt_path / f"{doc_id}.json"
            if gt_file.exists():
                with open(gt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # soportar estructuras comunes: {'data': [...]}, lista directa, {'entities': [...]}
                ents = []
                if isinstance(data, dict) and 'data' in data:
                    ents = data.get('data', [])
                elif isinstance(data, dict) and 'entities' in data:
                    ents = data.get('entities', [])
                elif isinstance(data, list):
                    ents = data

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
    
    return results, entrada, gt_by_doc, doc_ids


def evaluate(results: dict, entrada: dict, gt_by_doc: dict, doc_ids: set):
    """
    Evalúa el pipeline comparando entrada, salida y GT.
    Incluye trazabilidad de classification_source para saber qué etapa clasificó cada entidad.
    """
    # Construir set de entidades PII (ahora usamos classification en vez de decision)
    # y rastrear la fuente de clasificación
    keep_by_doc = defaultdict(set)
    source_by_text = {}  # Mapeo texto -> classification_source
    
    for d in results.get('decisions', []):
        doc_id = d.get('document_id', '')
        text_norm = normalize_text(d.get('entity_text', ''))
        classification_source = d.get('classification_source', 'unknown')
        
        if doc_id and text_norm:
            keep_by_doc[doc_id].add(text_norm)
            source_by_text[(doc_id, text_norm)] = classification_source
    
    # Métricas
    tp = fp = tn = fn = 0
    
    # Ejemplos para diagnóstico
    examples = {'tp': [], 'fp': [], 'tn': [], 'fn': []}
    
    # Métricas por etiqueta
    label_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0, 'kept': 0})
    
    # Trazabilidad: métricas por fuente de clasificación
    source_metrics = {
        'setfit': {'tp': 0, 'fp': 0, 'total': 0},
        'dict_whitelist': {'tp': 0, 'fp': 0, 'total': 0},
        'llm_rescue': {'tp': 0, 'fp': 0, 'total': 0},
        'unknown': {'tp': 0, 'fp': 0, 'total': 0}
    }
    
    # Analizar cada entidad de entrada
    for e in entrada.get('entities', []):
        doc_id = e.get('doc_id', '')
        text = e.get('text', '')
        text_norm = normalize_text(text)
        label = e.get('label', 'UNKNOWN')
        
        if doc_id not in doc_ids:
            continue
        
        gt_texts = gt_by_doc.get(doc_id, set())
        keep_texts = keep_by_doc.get(doc_id, set())
        
        # ¿Está en GT?
        in_gt = text_matches_gt(text_norm, gt_texts)
        
        # ¿Fue clasificado como PII (kept)?
        was_kept = text_matches_gt(text_norm, keep_texts)
        
        # Obtener fuente de clasificación
        source = source_by_text.get((doc_id, text_norm), 'unknown')
        if source not in source_metrics:
            source = 'unknown'
        
        label_metrics[label]['total'] += 1
        
        if was_kept:
            label_metrics[label]['kept'] += 1
            source_metrics[source]['total'] += 1
            
            if in_gt:
                tp += 1
                label_metrics[label]['tp'] += 1
                source_metrics[source]['tp'] += 1
                if len(examples['tp']) < 5:
                    examples['tp'].append({'doc': doc_id[:8], 'text': text[:40], 'label': label, 'source': source})
            else:
                fp += 1
                label_metrics[label]['fp'] += 1
                source_metrics[source]['fp'] += 1
                if len(examples['fp']) < 10:
                    examples['fp'].append({'doc': doc_id[:8], 'text': text[:40], 'label': label, 'source': source})
        else:
            if in_gt:
                fn += 1
                label_metrics[label]['fn'] += 1
                if len(examples['fn']) < 15:
                    examples['fn'].append({'doc': doc_id[:8], 'text': text[:40], 'label': label})
            else:
                tn += 1
                if len(examples['tn']) < 10:
                    examples['tn'].append({'doc': doc_id[:8], 'text': text[:40], 'label': label})
    
    # Calcular métricas globales
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    
    # Calcular precisión por fuente
    for source, m in source_metrics.items():
        if m['total'] > 0:
            m['precision'] = m['tp'] / m['total']
        else:
            m['precision'] = 0.0
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
        'examples': examples,
        'by_label': dict(label_metrics),
        'by_source': source_metrics  # Nueva métrica de trazabilidad
    }


def print_results(metrics: dict, results_metadata: dict):
    """Imprime resultados de forma legible con trazabilidad."""
    stats = results_metadata.get('stats', {})
    
    print("\n" + "=" * 80)
    print("📊 EVALUACIÓN DEL PIPELINE DE ANONIMIZACIÓN (FLUJO INVERTIDO)")
    print("=" * 80)
    
    print(f"\n🔢 ESTADÍSTICAS DEL PIPELINE:")
    print(f"  Entrada: {stats.get('total_input', 0)} entidades")
    print(f"  SetFit:  PII={stats.get('setfit_pii', stats.get('setfit_kept', 0))} (directo), RUIDO={stats.get('setfit_ruido', stats.get('setfit_filtered', 0))} (a rescate)")
    print(f"  Rescate: Listas={stats.get('dict_rescued', stats.get('dict_kept', 0))}, LLM={stats.get('llm_rescued', stats.get('llm_kept', 0))}")
    print(f"  Descartados: Listas={stats.get('dict_filtered', 0)}, LLM={stats.get('llm_filtered', 0)}")
    print(f"  Salida final: {stats.get('final_output', 0)} entidades PII")
    
    print(f"\n🎯 MÉTRICAS VS GROUND TRUTH:")
    print(f"  TP (PII correcto - información sensible conservada): {metrics['tp']}")
    print(f"  FP (PII incorrecto - ruido conservado): {metrics['fp']}")
    print(f"  TN (Descartado correcto - ruido eliminado): {metrics['tn']}")
    print(f"  FN (Descartado incorrecto - PII perdido): {metrics['fn']}")
    
    print(f"\n📈 MÉTRICAS GLOBALES:")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
    print(f"  F1:        {metrics['f1']:.4f} ({metrics['f1']*100:.1f}%)")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    
    # TRAZABILIDAD: Métricas por fuente de clasificación
    if 'by_source' in metrics:
        print(f"\n🔍 TRAZABILIDAD - MÉTRICAS POR FUENTE:")
        print(f"  {'Fuente':<20} {'Total':>8} {'TP':>8} {'FP':>8} {'Precisión':>10}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        
        source_labels = {
            'setfit': 'SetFit (directo)',
            'dict_whitelist': 'Listas (rescate)',
            'llm_rescue': 'LLM (rescate)',
            'unknown': 'Desconocido'
        }
        
        for source, label in source_labels.items():
            m = metrics['by_source'].get(source, {'total': 0, 'tp': 0, 'fp': 0, 'precision': 0})
            if m['total'] > 0:
                print(f"  {label:<20} {m['total']:>8} {m['tp']:>8} {m['fp']:>8} {m['precision']*100:>9.1f}%")
    
    # Por etiqueta
    print(f"\n📋 MÉTRICAS POR ETIQUETA:")
    print(f"  {'Etiqueta':<35} {'Total':>6} {'Keep':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>6} {'R':>6}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*6}")
    
    sorted_labels = sorted(metrics['by_label'].items(), key=lambda x: x[1]['total'], reverse=True)
    for label, m in sorted_labels:
        p = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
        r = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
        print(f"  {label:<35} {m['total']:>6} {m['kept']:>6} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {p:>6.2f} {r:>6.2f}")
    
    # Ejemplos con fuente
    if metrics['examples']['tp']:
        print(f"\n✅ EJEMPLOS DE TP (PII correcto):")
        for ex in metrics['examples']['tp']:
            source = ex.get('source', '?')
            print(f"  [{ex['doc']}] [{source:15s}] {ex['label']:25s} | \"{ex['text']}\"")
    
    if metrics['examples']['fp']:
        print(f"\n❌ EJEMPLOS DE FP (ruido conservado):")
        for ex in metrics['examples']['fp']:
            source = ex.get('source', '?')
            print(f"  [{ex['doc']}] [{source:15s}] {ex['label']:25s} | \"{ex['text']}\"")
    
    if metrics['examples']['fn']:
        print(f"\n⚠️  EJEMPLOS DE FN (PII perdido):")
        for ex in metrics['examples']['fn']:
            print(f"  [{ex['doc']}] {ex['label']:30s} | \"{ex['text']}\"")
    
    print("=" * 80)


def save_results(metrics: dict, output_path: str):
    """Guarda resultados en JSON con trazabilidad."""
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'evaluation_type': 'pipeline_filtering_inverted',
            'flow': 'SetFit(PII/RUIDO) -> RUIDO: DictFilters(rescue) -> LLM(rescue)'
        },
        'metrics': {
            'global': {
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'tn': metrics['tn'],
                'fn': metrics['fn'],
                'precision': round(metrics['precision'], 4),
                'recall': round(metrics['recall'], 4),
                'f1': round(metrics['f1'], 4),
                'accuracy': round(metrics['accuracy'], 4)
            },
            'by_label': metrics['by_label'],
            'by_source': metrics.get('by_source', {})  # Trazabilidad
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resultados guardados en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el pipeline de filtrado")
    parser.add_argument('--results', '-r', default=str(DEFAULT_RESULTS))
    parser.add_argument('--input', '-i', default=str(DEFAULT_INPUT))
    parser.add_argument('--ground-truth', '-g', default=str(DEFAULT_GT_DIR))
    parser.add_argument('--output', '-o', default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    
    logger.info("Cargando datos...")
    results, entrada, gt_by_doc, doc_ids = load_data(
        args.results, args.input, args.ground_truth
    )
    logger.info(f"Documentos procesados: {len(doc_ids)}, con GT: {len(gt_by_doc)}")
    
    logger.info("Evaluando...")
    metrics = evaluate(results, entrada, gt_by_doc, doc_ids)
    
    print_results(metrics, results.get('metadata', {}))
    save_results(metrics, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
