#!/usr/bin/env python3
"""
evaluate_pipeline.py - Evaluación completa del pipeline de anonimización
=========================================================================

Script principal para evaluar el pipeline completo y calcular métricas
globales y por etapa. Puede ejecutarse después de run_full_pipeline.py
o de forma independiente.

MÉTRICAS CALCULADAS:
- Métricas globales (TP, FP, FN, Precision, Recall, F1)
- Métricas por etapa (SetFit, DictFilters, LLM)
- Análisis por tipo de etiqueta
- Comparación con ground truth

USO:
    # Evaluar resultados del pipeline:
    python evaluate_pipeline.py --results outputs/pipeline_results.json --ground-truth corpus/ANTIGUO/entidades
    
    # Ejecutar pipeline completo y evaluar:
    python evaluate_pipeline.py --input entidades.json --ground-truth corpus/ANTIGUO/entidades --run-pipeline
    
    # Solo métricas por etapa:
    python evaluate_pipeline.py --results outputs/pipeline_results.json -v

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
sys.path.insert(0, str(SCRIPT_DIR))

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "outputs" / "resultados_completo.json"
DEFAULT_GT_DIR = PROJECT_ROOT / "corpus" / "ANTIGUO" / "entidades"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "pipeline_evaluation.json"


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class StageMetrics:
    """Métricas de una etapa del pipeline."""
    name: str
    input_count: int = 0
    output_count: int = 0
    kept: int = 0
    filtered: int = 0
    escalated: int = 0
    
    # Contra GT
    tp: int = 0
    fp: int = 0
    fn: int = 0
    
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
    def filter_rate(self) -> float:
        return self.filtered / self.input_count * 100 if self.input_count > 0 else 0.0


@dataclass
class LabelMetrics:
    """Métricas por etiqueta."""
    label: str
    total: int = 0
    kept: int = 0
    filtered: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    
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


@dataclass
class PipelineMetrics:
    """Métricas globales del pipeline."""
    # Contadores globales
    total_input: int = 0
    total_output: int = 0
    total_kept: int = 0
    total_filtered: int = 0
    
    # Por etapa
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    
    # Por etiqueta
    labels: Dict[str, LabelMetrics] = field(default_factory=dict)
    
    # Contra GT
    tp: int = 0
    fp: int = 0
    fn: int = 0
    
    # Tiempos
    total_time_seconds: float = 0.0
    
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
    def reduction_rate(self) -> float:
        return (self.total_input - self.total_output) / self.total_input * 100 if self.total_input > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "global": {
                "total_input": self.total_input,
                "total_output": self.total_output,
                "total_kept": self.total_kept,
                "total_filtered": self.total_filtered,
                "reduction_rate_percent": round(self.reduction_rate, 2),
                "total_time_seconds": round(self.total_time_seconds, 2),
            },
            "vs_ground_truth": {
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
            },
            "by_stage": {
                name: {
                    "input": s.input_count,
                    "output": s.output_count,
                    "kept": s.kept,
                    "filtered": s.filtered,
                    "escalated": s.escalated,
                    "filter_rate_percent": round(s.filter_rate, 2),
                    "precision": round(s.precision, 4),
                    "recall": round(s.recall, 4),
                    "f1": round(s.f1, 4),
                }
                for name, s in self.stages.items()
            },
            "by_label": {
                label: {
                    "total": m.total,
                    "kept": m.kept,
                    "filtered": m.filtered,
                    "tp": m.tp,
                    "fp": m.fp,
                    "fn": m.fn,
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1": round(m.f1, 4),
                }
                for label, m in sorted(self.labels.items())
            },
        }


# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def load_pipeline_results(file_path: str) -> Dict[str, Any]:
    """Carga resultados del pipeline."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación: minúsculas, sin tildes, espacios normalizados."""
    import unicodedata
    # Minúsculas
    text = text.lower().strip()
    # Quitar tildes
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Normalizar espacios múltiples
    text = ' '.join(text.split())
    return text


def load_ground_truth(gt_path: str) -> Dict[str, Set[Tuple[str, str]]]:
    """Carga ground truth desde directorio o archivo."""
    gt_path = Path(gt_path)
    ground_truth = defaultdict(set)
    
    if gt_path.is_file():
        # Archivo consolidado
        with open(gt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Diferentes formatos posibles
        if isinstance(data, dict) and 'detecciones' in data:
            # Formato: {"detecciones": [...]}
            entities = data['detecciones']
        elif isinstance(data, list):
            # Formato: [...]
            entities = data
        else:
            entities = data.get('entities', [])
        
        for ent in entities:
            doc_id = ent.get('doc_id', ent.get('document_id', ent.get('id', '')))
            text = ent.get('texto_detectado', ent.get('entity_text', ent.get('text', '')))
            label = ent.get('etiqueta', ent.get('entity', ent.get('label', '')))
            
            if text and label and doc_id:
                text_norm = normalize_text(text)
                ground_truth[doc_id].add((text_norm, label))
    
    elif gt_path.is_dir():
        # Directorio con archivos individuales por documento
        for json_file in gt_path.glob("*.json"):
            doc_id = json_file.stem
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Formato del corpus ANTIGUO: {"id": "...", "data": [{"entity": "...", "text": "..."}]}
            if isinstance(data, dict) and 'data' in data:
                entities = data['data']
            elif isinstance(data, list):
                entities = data
            else:
                entities = data.get('entities', [])
            
            for ent in entities:
                text = ent.get('text', ent.get('texto_detectado', ''))
                label = ent.get('entity', ent.get('label', ent.get('etiqueta', '')))
                
                if text and label:
                    text_norm = normalize_text(text)
                    ground_truth[doc_id].add((text_norm, label))
    
    return dict(ground_truth)


# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluate_pipeline(
    results: Dict[str, Any],
    ground_truth: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
) -> PipelineMetrics:
    """
    Evalúa los resultados del pipeline.
    
    Args:
        results: Resultados del pipeline (de run_full_pipeline.py)
        ground_truth: Ground truth opcional
        
    Returns:
        PipelineMetrics con todas las métricas
    """
    metrics = PipelineMetrics()
    
    # Extraer metadata
    metadata = results.get('metadata', {})
    pipeline_stats = results.get('pipeline_stats', metadata.get('pipeline_stats', {}))
    
    metrics.total_input = pipeline_stats.get('input_entities', 0)
    metrics.total_output = pipeline_stats.get('output_entities', 0)
    metrics.total_time_seconds = pipeline_stats.get('total_time_seconds', 0)
    
    # Extraer decisiones
    decisions = results.get('decisions', results.get('results', []))
    
    # Métricas por etapa
    setfit_stats = pipeline_stats.get('setfit', {})
    if setfit_stats:
        metrics.stages['setfit'] = StageMetrics(
            name='setfit',
            input_count=setfit_stats.get('input', 0),
            output_count=setfit_stats.get('output', 0),
            kept=setfit_stats.get('kept', 0),
            filtered=setfit_stats.get('filtered', 0),
        )
    
    dict_stats = pipeline_stats.get('dict_filters', {})
    if dict_stats:
        metrics.stages['dict_filters'] = StageMetrics(
            name='dict_filters',
            input_count=dict_stats.get('input', 0),
            output_count=dict_stats.get('output', 0),
            kept=dict_stats.get('kept', 0),
            filtered=dict_stats.get('filtered', 0),
            escalated=dict_stats.get('escalated', 0),
        )
    
    llm_stats = pipeline_stats.get('llm', {})
    if llm_stats:
        metrics.stages['llm'] = StageMetrics(
            name='llm',
            input_count=llm_stats.get('input', 0),
            output_count=llm_stats.get('output', 0),
            kept=llm_stats.get('kept', 0),
            filtered=llm_stats.get('filtered', 0),
        )
    
    # Métricas por etiqueta
    for decision in decisions:
        label = decision.get('label', 'UNKNOWN')
        final_decision = decision.get('final_decision', decision.get('decision', ''))
        is_kept = final_decision == 'KEEP' or final_decision is True
        
        if label not in metrics.labels:
            metrics.labels[label] = LabelMetrics(label=label)
        
        metrics.labels[label].total += 1
        if is_kept:
            metrics.labels[label].kept += 1
            metrics.total_kept += 1
        else:
            metrics.labels[label].filtered += 1
            metrics.total_filtered += 1
    
    # Evaluar contra ground truth
    if ground_truth:
        logger.info("Evaluando contra ground truth...")
        
        # Agrupar predicciones por documento
        predictions_by_doc = defaultdict(set)
        for decision in decisions:
            doc_id = decision.get('document_id', decision.get('doc_id', ''))
            text = decision.get('entity_text', decision.get('text', ''))
            label = decision.get('label', '')
            final = decision.get('final_decision', decision.get('decision', ''))
            is_kept = final == 'KEEP' or final is True
            
            if text and label and doc_id and is_kept:
                text_norm = normalize_text(text)
                predictions_by_doc[doc_id].add((text_norm, label))
        
        # Calcular métricas globales
        all_docs = set(ground_truth.keys()) | set(predictions_by_doc.keys())
        
        for doc_id in all_docs:
            gt_set = ground_truth.get(doc_id, set())
            pred_set = predictions_by_doc.get(doc_id, set())
            
            metrics.tp += len(gt_set & pred_set)
            metrics.fp += len(pred_set - gt_set)
            metrics.fn += len(gt_set - pred_set)
        
        # Calcular métricas por etiqueta
        for doc_id in all_docs:
            gt_set = ground_truth.get(doc_id, set())
            pred_set = predictions_by_doc.get(doc_id, set())
            
            for text, label in gt_set | pred_set:
                if label not in metrics.labels:
                    metrics.labels[label] = LabelMetrics(label=label)
                
                in_gt = (text, label) in gt_set
                in_pred = (text, label) in pred_set
                
                if in_gt and in_pred:
                    metrics.labels[label].tp += 1
                elif in_pred and not in_gt:
                    metrics.labels[label].fp += 1
                elif in_gt and not in_pred:
                    metrics.labels[label].fn += 1
    
    return metrics


def print_metrics(metrics: PipelineMetrics):
    """Imprime métricas de forma legible."""
    print("\n" + "=" * 80)
    print("📊 EVALUACIÓN DEL PIPELINE DE ANONIMIZACIÓN")
    print("=" * 80)
    
    print(f"\n🔢 MÉTRICAS GLOBALES:")
    print(f"  Entidades entrada: {metrics.total_input}")
    print(f"  Entidades salida: {metrics.total_output}")
    print(f"  Mantenidas (KEEP): {metrics.total_kept}")
    print(f"  Filtradas (FILTER): {metrics.total_filtered}")
    print(f"  Reducción: {metrics.reduction_rate:.1f}%")
    print(f"  Tiempo total: {metrics.total_time_seconds:.2f}s")
    
    if metrics.stages:
        print(f"\n📈 MÉTRICAS POR ETAPA:")
        for name, stage in metrics.stages.items():
            print(f"\n  [{name.upper()}]")
            print(f"    Input: {stage.input_count} → Output: {stage.output_count}")
            print(f"    Kept: {stage.kept} | Filtered: {stage.filtered} | Escalated: {stage.escalated}")
            print(f"    Filter rate: {stage.filter_rate:.1f}%")
            if stage.tp > 0 or stage.fp > 0:
                print(f"    P: {stage.precision:.4f} | R: {stage.recall:.4f} | F1: {stage.f1:.4f}")
    
    if metrics.tp > 0 or metrics.fp > 0 or metrics.fn > 0:
        print(f"\n🎯 VS GROUND TRUTH:")
        print(f"  TP: {metrics.tp}")
        print(f"  FP: {metrics.fp}")
        print(f"  FN: {metrics.fn}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall: {metrics.recall:.4f}")
        print(f"  F1: {metrics.f1:.4f}")
    
    if metrics.labels:
        print(f"\n📋 MÉTRICAS POR ETIQUETA:")
        print(f"  {'Etiqueta':<35} {'Total':>6} {'Keep':>6} {'P':>6} {'R':>6} {'F1':>6}")
        print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
        
        for label, m in sorted(metrics.labels.items(), key=lambda x: x[1].total, reverse=True):
            print(f"  {label:<35} {m.total:>6} {m.kept:>6} {m.precision:>6.2f} {m.recall:>6.2f} {m.f1:>6.2f}")
    
    print("=" * 80)


def save_evaluation(metrics: PipelineMetrics, output_path: str):
    """Guarda evaluación en JSON."""
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "evaluation_type": "pipeline",
        },
        "metrics": metrics.to_dict(),
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluación guardada en {output_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa el pipeline de anonimización completo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Evaluar resultados existentes:
  python evaluate_pipeline.py --results outputs/pipeline_results.json
  
  # Evaluar contra ground truth:
  python evaluate_pipeline.py --results outputs/pipeline_results.json --ground-truth corpus/ANTIGUO/entidades
  
  # Ejecutar pipeline y evaluar:
  python evaluate_pipeline.py --input entidades.json --run-pipeline
"""
    )
    
    parser.add_argument(
        '--results', '-r',
        default=str(DEFAULT_RESULTS),
        help='Archivo JSON con resultados del pipeline'
    )
    
    parser.add_argument(
        '--ground-truth', '-g',
        default=str(DEFAULT_GT_DIR),
        help='Directorio o archivo con ground truth'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=str(DEFAULT_OUTPUT),
        help='Archivo de salida para la evaluación'
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Archivo de entrada para ejecutar pipeline (requiere --run-pipeline)'
    )
    
    parser.add_argument(
        '--run-pipeline',
        action='store_true',
        help='Ejecutar pipeline antes de evaluar'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(levelname)-8s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Si se pidió ejecutar pipeline primero
    if args.run_pipeline:
        if not args.input:
            logger.error("Se requiere --input cuando se usa --run-pipeline")
            return 1
        
        logger.info("Ejecutando pipeline...")
        from run_full_pipeline import FullPipeline, DEFAULT_CONFIG
        from io_json import load_entities, save_pipeline_results
        
        # Cargar entidades
        entities = load_entities(args.input)
        
        # Ejecutar pipeline
        config = DEFAULT_CONFIG.copy()
        config["pipeline"]["skip_llm"] = True  # Por defecto skip LLM en evaluación
        
        pipeline = FullPipeline(config)
        results = pipeline.run(entities)
        results_path = args.results
        
        # Guardar resultados para análisis
        save_pipeline_results(
            results,
            results_path,
            metadata={"generated_at": datetime.now().isoformat()},
            stats=pipeline.get_stats()
        )
        
        # Recargar para evaluar
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results_path = args.results
    
    # Cargar resultados
    logger.info(f"Cargando resultados desde {results_path}")
    results = load_pipeline_results(results_path)
    
    # Cargar ground truth
    ground_truth = None
    if Path(args.ground_truth).exists():
        logger.info(f"Cargando ground truth desde {args.ground_truth}")
        ground_truth = load_ground_truth(args.ground_truth)
        logger.info(f"Ground truth: {len(ground_truth)} documentos")
    
    # Evaluar
    metrics = evaluate_pipeline(results, ground_truth)
    
    # Mostrar métricas
    print_metrics(metrics)
    
    # Guardar evaluación
    save_evaluation(metrics, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
