#!/usr/bin/env python3
"""
setfit_module/evaluate.py - Evaluación y métricas del módulo SetFit
====================================================================

Script para evaluar el rendimiento del clasificador SetFit de forma aislada.
Calcula métricas comparando las predicciones SetFit contra ground truth.

MÉTRICAS CALCULADAS:
- TP: Entidades correctamente clasificadas como PII
- FP: Entidades clasificadas como PII que son ruido (falsos positivos)
- FN: Entidades PII no detectadas (falsos negativos)  
- TN: Entidades ruido correctamente filtradas
- Precision, Recall, F1, Accuracy

USO:
    # Desde el directorio pipeline-nuevos-textos:
    python -m setfit_module.evaluate --input entidades.json --ground-truth gt.json
    
    # Con verbose:
    python -m setfit_module.evaluate --input entidades.json --ground-truth gt.json -v

AUTOR: Pipeline Anonimización Clínica
VERSION: 1.0.0
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Añadir path del módulo
SCRIPT_DIR = Path(__file__).parent
MODULE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MODULE_DIR))

from setfit_module import run_setfit_filter, SetFitGatekeeper

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "entidades-procesadas-para-metricas.json"
DEFAULT_GT_DIR = PROJECT_ROOT / "corpus" / "ANTIGUO" / "entidades"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "setfit_evaluation.json"


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class SetFitMetrics:
    """Métricas de evaluación del SetFit."""
    total_entities: int = 0
    pii_predicted: int = 0
    noise_predicted: int = 0
    
    # Por método de clasificación
    obvious_noise_filtered: int = 0
    obvious_pii_detected: int = 0
    fragment_filtered: int = 0
    low_confidence_filtered: int = 0
    setfit_pii: int = 0
    setfit_noise: int = 0
    
    # Contra ground truth (si disponible)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    
    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0
    
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else 0.0
    
    @property
    def fp_reduction(self) -> float:
        """Porcentaje de falsos positivos filtrados."""
        total_filtered = self.obvious_noise_filtered + self.fragment_filtered + self.low_confidence_filtered + self.setfit_noise
        return total_filtered / self.total_entities * 100 if self.total_entities > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entities": self.total_entities,
            "pii_predicted": self.pii_predicted,
            "noise_predicted": self.noise_predicted,
            "classification_breakdown": {
                "obvious_noise_filtered": self.obvious_noise_filtered,
                "obvious_pii_detected": self.obvious_pii_detected,
                "fragment_filtered": self.fragment_filtered,
                "low_confidence_filtered": self.low_confidence_filtered,
                "setfit_pii": self.setfit_pii,
                "setfit_noise": self.setfit_noise,
            },
            "vs_ground_truth": {
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
                "tn": self.tn,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
                "accuracy": round(self.accuracy, 4),
            },
            "fp_reduction_percent": round(self.fp_reduction, 2),
        }


# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def load_entities(file_path: str) -> List[Dict[str, Any]]:
    """Carga entidades desde JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        for key in ['entities', 'detecciones', 'decisions', 'results']:
            if key in data:
                return data[key]
    
    raise ValueError(f"Formato no reconocido en {file_path}")


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación: minúsculas, sin tildes, espacios normalizados."""
    import unicodedata
    # Minúsculas y strip
    text = text.lower().strip()
    # Quitar tildes/acentos
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Normalizar espacios múltiples
    text = ' '.join(text.split())
    return text


def load_ground_truth(gt_path: str) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Carga ground truth desde directorio o archivo consolidado.
    
    Returns:
        Dict {doc_id: set((texto_norm, label), ...)}
    """
    gt_path = Path(gt_path)
    ground_truth = defaultdict(set)
    
    if gt_path.is_file():
        # Archivo consolidado
        with open(gt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        entities = data if isinstance(data, list) else data.get('detecciones', data.get('entities', []))
        
        for ent in entities:
            doc_id = ent.get('doc_id', ent.get('document_id', ''))
            text = ent.get('texto_detectado', ent.get('entity_text', ent.get('text', '')))
            label = ent.get('etiqueta', ent.get('label', ent.get('entity', '')))
            
            if text and label and doc_id:
                text_norm = normalize_text(text)
                ground_truth[doc_id].add((text_norm, label))
    
    elif gt_path.is_dir():
        # Directorio con archivos individuales
        # Formato corpus ANTIGUO: {"id": "...", "data": [{"entity": "...", "text": "..."}]}
        for json_file in gt_path.glob("*.json"):
            doc_id = json_file.stem
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Soportar múltiples formatos
            if isinstance(data, dict) and 'data' in data:
                entities = data['data']
            elif isinstance(data, list):
                entities = data
            else:
                entities = data.get('entities', data.get('entidades', []))
            
            for ent in entities:
                # Campos del formato ANTIGUO: "entity" y "text"
                text = ent.get('text', ent.get('texto_detectado', ''))
                label = ent.get('entity', ent.get('label', ent.get('etiqueta', '')))
                
                if text and label:
                    text_norm = normalize_text(text)
                    ground_truth[doc_id].add((text_norm, label))
    
    return dict(ground_truth)


# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluate_setfit(
    entities: List[Dict[str, Any]],
    ground_truth: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], SetFitMetrics]:
    """
    Evalúa el módulo SetFit sobre un conjunto de entidades.
    
    Args:
        entities: Lista de entidades a clasificar
        ground_truth: Ground truth opcional para calcular métricas
        config: Configuración del SetFit
        
    Returns:
        Tuple de (resultados, métricas)
    """
    metrics = SetFitMetrics()
    metrics.total_entities = len(entities)
    
    # Ejecutar SetFit
    logger.info(f"Procesando {len(entities)} entidades con SetFit...")
    results = run_setfit_filter(entities, config=config)
    
    # Contar por método de clasificación
    for result in results:
        method = result.get('classification_method', '')
        is_pii = result.get('is_pii', result.get('decision') == 'KEEP')
        
        if is_pii:
            metrics.pii_predicted += 1
        else:
            metrics.noise_predicted += 1
        
        if method == 'obvious_noise':
            metrics.obvious_noise_filtered += 1
        elif method == 'obvious_pii':
            metrics.obvious_pii_detected += 1
        elif method == 'fragment':
            metrics.fragment_filtered += 1
        elif method == 'low_confidence':
            metrics.low_confidence_filtered += 1
        elif method == 'setfit':
            if is_pii:
                metrics.setfit_pii += 1
            else:
                metrics.setfit_noise += 1
    
    # Evaluar contra ground truth si está disponible
    if ground_truth:
        logger.info("Evaluando contra ground truth...")
        
        # Agrupar predicciones por documento
        predictions_by_doc = defaultdict(set)
        processed_doc_ids = set()
        
        for result in results:
            doc_id = result.get('document_id', result.get('doc_id', ''))
            text = result.get('entity_text', result.get('text', ''))
            label = result.get('label', '')
            is_pii = result.get('is_pii', result.get('decision') == 'KEEP')
            
            if doc_id:
                processed_doc_ids.add(doc_id)
            
            if text and label and doc_id and is_pii:
                text_norm = normalize_text(text)
                predictions_by_doc[doc_id].add((text_norm, label))
        
        # CRÍTICO: Solo evaluar documentos procesados, no todos los del GT
        if processed_doc_ids:
            eval_docs = processed_doc_ids
            logger.info(f"Evaluando {len(eval_docs)} documentos procesados contra GT")
        else:
            eval_docs = set(predictions_by_doc.keys())
        
        # Calcular métricas por documento (solo docs procesados)
        for doc_id in eval_docs:
            gt_set = ground_truth.get(doc_id, set())
            pred_set = predictions_by_doc.get(doc_id, set())
            
            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            
            metrics.tp += tp
            metrics.fp += fp
            metrics.fn += fn
        
        # TN = entidades filtradas que no estaban en GT (difícil de calcular exactamente)
        metrics.tn = metrics.noise_predicted  # Aproximación
    
    return results, metrics


def print_metrics(metrics: SetFitMetrics):
    """Imprime métricas de forma legible."""
    print("\n" + "=" * 70)
    print("📊 MÉTRICAS DEL MÓDULO SETFIT")
    print("=" * 70)
    
    print(f"\n🔢 TOTALES:")
    print(f"  Entidades procesadas: {metrics.total_entities}")
    print(f"  Clasificadas como PII: {metrics.pii_predicted}")
    print(f"  Clasificadas como Ruido: {metrics.noise_predicted}")
    
    print(f"\n🔧 DESGLOSE POR MÉTODO:")
    print(f"  Ruido obvio filtrado: {metrics.obvious_noise_filtered}")
    print(f"  PII obvio detectado: {metrics.obvious_pii_detected}")
    print(f"  Fragmentos filtrados: {metrics.fragment_filtered}")
    print(f"  Baja confianza filtrado: {metrics.low_confidence_filtered}")
    print(f"  SetFit → PII: {metrics.setfit_pii}")
    print(f"  SetFit → Ruido: {metrics.setfit_noise}")
    
    print(f"\n📈 REDUCCIÓN DE FP: {metrics.fp_reduction:.1f}%")
    
    if metrics.tp > 0 or metrics.fp > 0 or metrics.fn > 0:
        print(f"\n🎯 VS GROUND TRUTH:")
        print(f"  TP: {metrics.tp}")
        print(f"  FP: {metrics.fp}")
        print(f"  FN: {metrics.fn}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall: {metrics.recall:.4f}")
        print(f"  F1: {metrics.f1:.4f}")
    
    print("=" * 70)


def save_results(
    results: List[Dict[str, Any]],
    metrics: SetFitMetrics,
    output_path: str
):
    """Guarda resultados y métricas en JSON."""
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "module": "setfit_module",
            "total_entities": len(results),
        },
        "metrics": metrics.to_dict(),
        "results": results,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resultados guardados en {output_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa el módulo SetFit y calcula métricas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--input', '-i',
        default=str(DEFAULT_INPUT),
        help='Archivo JSON de entrada con entidades'
    )
    
    parser.add_argument(
        '--ground-truth', '-g',
        default=str(DEFAULT_GT_DIR),
        help='Directorio o archivo con ground truth'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=str(DEFAULT_OUTPUT),
        help='Archivo de salida para resultados'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.75,
        help='Umbral de confianza SetFit (default: 0.75)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configurar logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(levelname)-8s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Cargar entidades
    logger.info(f"Cargando entidades desde {args.input}")
    entities = load_entities(args.input)
    logger.info(f"Cargadas {len(entities)} entidades")
    
    # Cargar ground truth
    ground_truth = None
    if Path(args.ground_truth).exists():
        logger.info(f"Cargando ground truth desde {args.ground_truth}")
        ground_truth = load_ground_truth(args.ground_truth)
        logger.info(f"Ground truth: {len(ground_truth)} documentos")
    
    # Configuración
    config = {
        "confidence_threshold": args.threshold,
    }
    
    # Evaluar
    results, metrics = evaluate_setfit(entities, ground_truth, config)
    
    # Mostrar métricas
    print_metrics(metrics)
    
    # Guardar resultados
    save_results(results, metrics, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
