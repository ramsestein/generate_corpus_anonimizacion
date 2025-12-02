#!/usr/bin/env python3
"""
llm_judge/evaluate.py - Evaluación y métricas del módulo LLM Judge
===================================================================

Script para evaluar el rendimiento del LLM Judge de forma aislada.
Calcula métricas comparando las decisiones del LLM contra ground truth.

MÉTRICAS CALCULADAS:
- Accuracy del LLM en clasificación binaria
- Precision/Recall/F1 para decisiones PII vs Ruido
- Tiempo promedio de respuesta
- Tasa de errores/timeouts

USO:
    # Desde el directorio pipeline-nuevos-textos:
    python -m llm_judge.evaluate --input entidades.json --ground-truth gt.json
    
    # Sin ejecutar LLM (solo analizar resultados previos):
    python -m llm_judge.evaluate --results llm_results.json --ground-truth gt.json

AUTOR: Pipeline Anonimización Clínica
VERSION: 1.0.0
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Añadir path del módulo
SCRIPT_DIR = Path(__file__).parent
MODULE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MODULE_DIR))

from llm_judge import run_llm_judge, LLMJudge

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "entidades-procesadas-para-metricas.json"
DEFAULT_GT_DIR = PROJECT_ROOT / "corpus" / "ANTIGUO" / "entidades"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "llm_judge_evaluation.json"


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class LLMJudgeMetrics:
    """Métricas de evaluación del LLM Judge."""
    total_entities: int = 0
    
    # Decisiones
    pii_decisions: int = 0
    noise_decisions: int = 0
    errors: int = 0
    timeouts: int = 0
    
    # Tiempos
    total_time_seconds: float = 0.0
    avg_time_per_entity: float = 0.0
    
    # Contra ground truth
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
    def success_rate(self) -> float:
        successful = self.pii_decisions + self.noise_decisions
        return successful / self.total_entities * 100 if self.total_entities > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entities": self.total_entities,
            "decisions": {
                "pii": self.pii_decisions,
                "noise": self.noise_decisions,
                "errors": self.errors,
                "timeouts": self.timeouts,
            },
            "timing": {
                "total_seconds": round(self.total_time_seconds, 2),
                "avg_per_entity": round(self.avg_time_per_entity, 3),
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
            "success_rate_percent": round(self.success_rate, 2),
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
    """Carga ground truth desde directorio o archivo."""
    gt_path = Path(gt_path)
    ground_truth = defaultdict(set)
    
    if gt_path.is_file():
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
                entities = data.get('entities', [])
            
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

def evaluate_llm_judge(
    entities: List[Dict[str, Any]],
    ground_truth: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
    config: Optional[Dict[str, Any]] = None,
    max_entities: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], LLMJudgeMetrics]:
    """
    Evalúa el LLM Judge sobre un conjunto de entidades.
    
    Args:
        entities: Lista de entidades a evaluar
        ground_truth: Ground truth opcional
        config: Configuración del LLM
        max_entities: Límite de entidades a procesar (para pruebas)
        
    Returns:
        Tuple de (resultados, métricas)
    """
    if max_entities:
        entities = entities[:max_entities]
    
    metrics = LLMJudgeMetrics()
    metrics.total_entities = len(entities)
    
    # Ejecutar LLM Judge
    logger.info(f"Procesando {len(entities)} entidades con LLM Judge...")
    start_time = time.time()
    
    try:
        results = run_llm_judge(entities, config=config)
    except Exception as e:
        logger.error(f"Error ejecutando LLM: {e}")
        metrics.errors = len(entities)
        return [], metrics
    
    metrics.total_time_seconds = time.time() - start_time
    metrics.avg_time_per_entity = metrics.total_time_seconds / len(entities) if entities else 0
    
    # Contar decisiones
    for result in results:
        decision = result.get('llm_decision', result.get('decision', ''))
        
        if decision == 'KEEP' or decision is True:
            metrics.pii_decisions += 1
        elif decision == 'FILTER' or decision is False:
            metrics.noise_decisions += 1
        elif 'error' in str(decision).lower():
            metrics.errors += 1
        elif 'timeout' in str(decision).lower():
            metrics.timeouts += 1
    
    # Evaluar contra ground truth
    if ground_truth:
        logger.info("Evaluando contra ground truth...")
        
        # Recopilar doc_ids procesados
        processed_doc_ids = set()
        for result in results:
            doc_id = result.get('document_id', result.get('doc_id', ''))
            if doc_id:
                processed_doc_ids.add(doc_id)
        
        for result in results:
            doc_id = result.get('document_id', result.get('doc_id', ''))
            text = result.get('entity_text', result.get('text', ''))
            label = result.get('label', '')
            decision = result.get('llm_decision', result.get('decision', ''))
            is_pii = decision == 'KEEP' or decision is True
            
            # Normalizar texto para comparación
            text_norm = normalize_text(text)
            
            gt_set = ground_truth.get(doc_id, set())
            is_in_gt = (text_norm, label) in gt_set
            
            if is_pii and is_in_gt:
                metrics.tp += 1
            elif is_pii and not is_in_gt:
                metrics.fp += 1
            elif not is_pii and is_in_gt:
                metrics.fn += 1
            else:
                metrics.tn += 1
    
    return results, metrics


def print_metrics(metrics: LLMJudgeMetrics):
    """Imprime métricas de forma legible."""
    print("\n" + "=" * 70)
    print("📊 MÉTRICAS DEL MÓDULO LLM JUDGE")
    print("=" * 70)
    
    print(f"\n🔢 TOTALES:")
    print(f"  Entidades procesadas: {metrics.total_entities}")
    print(f"  Decisiones PII: {metrics.pii_decisions}")
    print(f"  Decisiones Ruido: {metrics.noise_decisions}")
    print(f"  Errores: {metrics.errors}")
    print(f"  Timeouts: {metrics.timeouts}")
    
    print(f"\n⏱️  TIEMPOS:")
    print(f"  Tiempo total: {metrics.total_time_seconds:.2f}s")
    print(f"  Promedio por entidad: {metrics.avg_time_per_entity:.3f}s")
    
    print(f"\n📈 TASA DE ÉXITO: {metrics.success_rate:.1f}%")
    
    if metrics.tp > 0 or metrics.fp > 0 or metrics.fn > 0:
        print(f"\n🎯 VS GROUND TRUTH:")
        print(f"  TP: {metrics.tp}")
        print(f"  FP: {metrics.fp}")
        print(f"  FN: {metrics.fn}")
        print(f"  TN: {metrics.tn}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall: {metrics.recall:.4f}")
        print(f"  F1: {metrics.f1:.4f}")
        print(f"  Accuracy: {metrics.accuracy:.4f}")
    
    print("=" * 70)


def save_results(
    results: List[Dict[str, Any]],
    metrics: LLMJudgeMetrics,
    output_path: str
):
    """Guarda resultados y métricas en JSON."""
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "module": "llm_judge",
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
        description="Evalúa el módulo LLM Judge y calcula métricas",
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
        '--model', '-m',
        default='gemma3:270m',
        help='Modelo LLM a usar (default: gemma3:270m)'
    )
    
    parser.add_argument(
        '--max-entities', '-n',
        type=int,
        default=None,
        help='Límite de entidades a procesar (para pruebas)'
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
    
    logger.info(f"Cargando entidades desde {args.input}")
    entities = load_entities(args.input)
    logger.info(f"Cargadas {len(entities)} entidades")
    
    ground_truth = None
    if Path(args.ground_truth).exists():
        logger.info(f"Cargando ground truth desde {args.ground_truth}")
        ground_truth = load_ground_truth(args.ground_truth)
        logger.info(f"Ground truth: {len(ground_truth)} documentos")
    
    config = {
        "model": args.model,
    }
    
    results, metrics = evaluate_llm_judge(
        entities, 
        ground_truth, 
        config,
        max_entities=args.max_entities
    )
    
    print_metrics(metrics)
    save_results(results, metrics, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
