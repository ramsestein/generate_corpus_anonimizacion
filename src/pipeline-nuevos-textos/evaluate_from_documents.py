#!/usr/bin/env python3
"""
evaluate_from_documents.py - Evaluación basada en marcadores de documentos
==========================================================================

Este script evalúa las detecciones comparándolas con el ground truth
que está marcado en los documentos con el formato [** texto **].

Métricas:
- TP (True Positive): Detectado Y presente en ground truth [** **]
- FN (False Negative): En ground truth [** **] pero NO detectado
- FP (False Positive): Detectado pero NO está en ground truth [** **]

USO:
    python evaluate_from_documents.py --docs-dir corpus/output/aws3 --detections step6_validation_results/aws3/detecciones_detalladas.csv --output evaluation_results.json
"""

import argparse
import json
import re
import csv
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# REGEX PARA EXTRAER GROUND TRUTH
# ============================================================================

# Patrón para encontrar texto entre [** y **]
GROUND_TRUTH_PATTERN = re.compile(r'\[\*\*(.+?)\*\*\]')


@dataclass
class EvaluationResult:
    """Resultado de la evaluación por documento."""
    doc_id: str
    total_ground_truth: int = 0
    total_detections: int = 0
    tp: int = 0
    fn: int = 0
    fp: int = 0
    ground_truth_entities: List[str] = field(default_factory=list)
    detected_entities: List[str] = field(default_factory=list)
    tp_entities: List[str] = field(default_factory=list)
    fn_entities: List[str] = field(default_factory=list)
    fp_entities: List[str] = field(default_factory=list)


def extract_doc_id(filepath: Path) -> str:
    """Extrae el doc_id del nombre del archivo, manejando .txt.txt"""
    name = filepath.name
    if name.endswith('.txt.txt'):
        return name[:-8]
    elif name.endswith('.txt'):
        return name[:-4]
    return filepath.stem


def extract_ground_truth(text: str) -> List[Tuple[str, int, int]]:
    """
    Extrae todas las entidades ground truth marcadas con [** **].
    
    Returns:
        Lista de tuplas (texto_entidad, posicion_inicio, posicion_fin)
    """
    entities = []
    for match in GROUND_TRUTH_PATTERN.finditer(text):
        entity_text = match.group(1).strip()
        start = match.start()
        end = match.end()
        entities.append((entity_text, start, end))
    return entities


def load_documents(docs_dir: Path) -> Dict[str, Tuple[str, List[Tuple[str, int, int]]]]:
    """
    Carga todos los documentos y extrae el ground truth.
    
    Returns:
        Dict {doc_id: (texto_completo, lista_ground_truth)}
    """
    documents = {}
    
    for txt_file in docs_dir.rglob("*.txt"):
        doc_id = extract_doc_id(txt_file)
        try:
            text = txt_file.read_text(encoding="utf-8")
            ground_truth = extract_ground_truth(text)
            documents[doc_id] = (text, ground_truth)
        except Exception as e:
            logger.warning(f"Error leyendo {txt_file}: {e}")
    
    logger.info(f"Cargados {len(documents)} documentos desde {docs_dir}")
    return documents


def load_detections_from_csv(csv_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Carga las detecciones desde el CSV.
    
    Returns:
        Dict {doc_id: lista_de_detecciones}
    """
    detections = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get('doc_id', row.get('documento', row.get('document_id', '')))
            # Limpiar doc_id (quitar extensión si la tiene)
            if doc_id.endswith('.txt.txt'):
                doc_id = doc_id[:-8]
            elif doc_id.endswith('.txt'):
                doc_id = doc_id[:-4]
            
            detection = {
                'text': row.get('texto_detectado', row.get('entidad', row.get('text', row.get('entity_text', '')))),
                'label': row.get('etiqueta', row.get('label', '')),
                'start': int(row.get('posicion_inicio', row.get('inicio', row.get('start', -1)))),
                'end': int(row.get('posicion_fin', row.get('fin', row.get('end', -1)))),
                'confidence': float(row.get('confianza', row.get('confidence', 0.0))),
            }
            detections[doc_id].append(detection)
    
    total_detections = sum(len(v) for v in detections.values())
    logger.info(f"Cargadas {total_detections} detecciones de {len(detections)} documentos")
    return dict(detections)



def load_detections_from_json(json_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Carga las detecciones desde un archivo JSON.
    
    Returns:
        Dict {doc_id: lista_de_detecciones}
    """
    detections = defaultdict(list)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Manejar diferentes formatos de JSON
    if isinstance(data, list):
        entities = data
    elif 'decisions' in data:
        entities = data['decisions']
    elif 'entities' in data:
        entities = data['entities']
    elif 'results' in data:
        entities = data['results']
    else:
        entities = []

    
    for entity in entities:
        doc_id = entity.get('document_id', entity.get('doc_id', ''))
        if doc_id.endswith('.txt.txt'):
            doc_id = doc_id[:-8]
        elif doc_id.endswith('.txt'):
            doc_id = doc_id[:-4]
        
        detection = {
            'text': entity.get('entity_text', entity.get('text', '')),
            'label': entity.get('label', ''),
            'start': entity.get('start', -1),
            'end': entity.get('end', -1),
            'confidence': entity.get('confidence', 0.0),
            'classification': entity.get('classification', 'PII'),
        }
        detections[doc_id].append(detection)
    
    total_detections = sum(len(v) for v in detections.values())
    logger.info(f"Cargadas {total_detections} detecciones de {len(detections)} documentos")
    return dict(detections)


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación."""
    import re
    
    # Extraer contenido de entre [** **] si existe
    match = re.search(r'\[\*\*(.+?)\*\*\]', text)
    if match:
        text = match.group(1)
    else:
        # Limpiar marcadores parciales
        text = text.replace('[**', '').replace('**]', '')
        # Limpiar caracteres comunes pegados
        text = text.strip('.,;:()[]{}"\'-/')
    
    # Quitar espacios extra, convertir a minúsculas
    return ' '.join(text.lower().split())


def evaluate_document(
    doc_id: str,
    ground_truth: List[Tuple[str, int, int]],
    detections: List[Dict[str, Any]],
    only_pii: bool = True
) -> EvaluationResult:
    """
    Evalúa un documento comparando ground truth con detecciones.
    
    Lógica corregida:
    - Para cada entidad en ground truth: si fue detectada = TP, si no = FN
    - Para cada detección: si está en ground truth = TP (ya contado), si no = FP
    
    Args:
        doc_id: ID del documento
        ground_truth: Lista de (texto, inicio, fin) del ground truth
        detections: Lista de detecciones
        only_pii: Si True, solo considera detecciones clasificadas como PII
    
    Returns:
        EvaluationResult con métricas
    """
    result = EvaluationResult(doc_id=doc_id)
    
    # Filtrar detecciones si es necesario
    if only_pii:
        detections = [d for d in detections if d.get('classification', 'PII') == 'PII']
    
    result.ground_truth_entities = [gt[0] for gt in ground_truth]
    result.detected_entities = [d['text'] for d in detections]
    result.total_ground_truth = len(ground_truth)
    result.total_detections = len(detections)
    
    # Crear lista de textos detectados normalizados (con conteo de uso)
    det_texts_normalized = [normalize_text(d['text']) for d in detections]
    det_matched = [False] * len(detections)  # Para rastrear qué detecciones ya fueron emparejadas
    
    tp_count = 0
    fn_count = 0
    tp_entities = []
    fn_entities = []
    
    # Para cada entidad en ground truth, buscar si fue detectada
    for gt_text, gt_start, gt_end in ground_truth:
        gt_normalized = normalize_text(gt_text)
        found = False
        
        # Buscar una detección que coincida (y no haya sido usada)
        for i, det_norm in enumerate(det_texts_normalized):
            if not det_matched[i]:
                # Match exacto
                if det_norm == gt_normalized:
                    found = True
                    det_matched[i] = True
                    tp_count += 1
                    tp_entities.append(gt_text)
                    break
                # Match parcial: GT contenido en detección o viceversa
                elif len(gt_normalized) >= 3 and len(det_norm) >= 3:
                    if gt_normalized in det_norm or det_norm in gt_normalized:
                        found = True
                        det_matched[i] = True
                        tp_count += 1
                        tp_entities.append(gt_text)
                        break
        
        if not found:
            fn_count += 1
            fn_entities.append(gt_text)
    
    # Las detecciones no emparejadas son FP
    fp_count = 0
    fp_entities = []
    for i, matched in enumerate(det_matched):
        if not matched:
            fp_count += 1
            fp_entities.append(detections[i]['text'])
    
    result.tp = tp_count
    result.fn = fn_count
    result.fp = fp_count
    result.tp_entities = tp_entities
    result.fn_entities = fn_entities
    result.fp_entities = fp_entities
    
    return result



def calculate_metrics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Calcula métricas globales a partir de los resultados por documento."""
    total_tp = sum(r.tp for r in results)
    total_fn = sum(r.fn for r in results)
    total_fp = sum(r.fp for r in results)
    total_gt = sum(r.total_ground_truth for r in results)
    total_det = sum(r.total_detections for r in results)
    
    # Calcular precisión, recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'total_documents': len(results),
        'total_ground_truth': total_gt,
        'total_detections': total_det,
        'true_positives': total_tp,
        'false_negatives': total_fn,
        'false_positives': total_fp,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_pct': precision * 100,
        'recall_pct': recall * 100,
        'f1_pct': f1 * 100,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa detecciones contra ground truth marcado con [** **] en documentos"
    )
    parser.add_argument(
        '--docs-dir', '-d',
        required=True,
        help='Directorio con documentos que contienen ground truth [** **]'
    )
    parser.add_argument(
        '--detections', '-i',
        required=True,
        help='Archivo CSV o JSON con las detecciones'
    )
    parser.add_argument(
        '--output', '-o',
        default='evaluation_results.json',
        help='Archivo de salida con resultados'
    )
    parser.add_argument(
        '--only-pii',
        action='store_true',
        default=True,
        help='Solo considerar detecciones clasificadas como PII (por defecto: True)'
    )
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Incluir todas las detecciones, no solo PII'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    docs_dir = Path(args.docs_dir)
    detections_path = Path(args.detections)
    
    if not docs_dir.exists():
        logger.error(f"Directorio no encontrado: {docs_dir}")
        return 1
    
    if not detections_path.exists():
        logger.error(f"Archivo de detecciones no encontrado: {detections_path}")
        return 1
    
    # Cargar documentos con ground truth
    logger.info("Cargando documentos con ground truth...")
    documents = load_documents(docs_dir)
    
    # Cargar detecciones
    logger.info("Cargando detecciones...")
    if detections_path.suffix.lower() == '.csv':
        detections = load_detections_from_csv(detections_path)
    else:
        detections = load_detections_from_json(detections_path)
    
    # Evaluar cada documento
    logger.info("Evaluando documentos...")
    results = []
    only_pii = not args.include_all
    
    for doc_id, (text, ground_truth) in documents.items():
        doc_detections = detections.get(doc_id, [])
        result = evaluate_document(doc_id, ground_truth, doc_detections, only_pii=only_pii)
        results.append(result)
        
        if args.verbose and (result.fn > 0 or result.fp > 0):
            logger.debug(f"{doc_id}: TP={result.tp}, FN={result.fn}, FP={result.fp}")
    
    # Calcular métricas globales
    metrics = calculate_metrics(results)
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("RESULTADOS DE EVALUACIÓN")
    print("=" * 70)
    print(f"Documentos evaluados:     {metrics['total_documents']}")
    print(f"Total ground truth:       {metrics['total_ground_truth']}")
    print(f"Total detecciones:        {metrics['total_detections']}")
    print("-" * 70)
    print(f"True Positives (TP):      {metrics['true_positives']}")
    print(f"False Negatives (FN):     {metrics['false_negatives']}")
    print(f"False Positives (FP):     {metrics['false_positives']}")
    print("-" * 70)
    print(f"Precision:                {metrics['precision_pct']:.2f}%")
    print(f"Recall:                   {metrics['recall_pct']:.2f}%")
    print(f"F1-Score:                 {metrics['f1_pct']:.2f}%")
    print("=" * 70)
    
    # Guardar resultados
    output_data = {
        'metrics': metrics,
        'config': {
            'docs_dir': str(docs_dir),
            'detections_file': str(detections_path),
            'only_pii': only_pii,
        },
        'per_document': [
            {
                'doc_id': r.doc_id,
                'ground_truth': r.total_ground_truth,
                'detections': r.total_detections,
                'tp': r.tp,
                'fn': r.fn,
                'fp': r.fp,
                'fn_entities': r.fn_entities[:10],  # Limitar para no hacer archivo enorme
                'fp_entities': r.fp_entities[:10],
            }
            for r in results
            if r.fn > 0 or r.fp > 0  # Solo guardar docs con errores
        ]
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resultados guardados en: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
