#!/usr/bin/env python3
"""
dict_filters/evaluate.py - Evaluación y métricas del módulo de filtros
=======================================================================

Script para evaluar el rendimiento de los filtros de diccionario de forma aislada.
Calcula métricas comparando las decisiones de filtro contra ground truth.

MÉTRICAS CALCULADAS:
- Entidades resueltas por whitelist (FORCE_ANONYMIZE)
- Entidades resueltas por blacklist (FORCE_IGNORE)
- Entidades resueltas por CIE10 (FORCE_IGNORE)
- Entidades escaladas al LLM (ESCALATE)
- Precisión de cada tipo de decisión vs ground truth

USO:
    # Desde el directorio pipeline-nuevos-textos:
    python -m dict_filters.evaluate --input entidades.json --ground-truth gt.json
    
    # Con verbose:
    python -m dict_filters.evaluate --input entidades.json -v

AUTOR: Pipeline Anonimización Clínica
VERSION: 1.0.0
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Añadir path del módulo
SCRIPT_DIR = Path(__file__).parent
MODULE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MODULE_DIR))

from dict_filters import apply_dict_filters, ListLoader, DictFilter, FilterDecision

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "entidades-procesadas-para-metricas.json"
DEFAULT_GT_DIR = PROJECT_ROOT / "corpus" / "ANTIGUO" / "entidades"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "dict_filters_evaluation.json"


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class DictFilterMetrics:
    """Métricas de evaluación de los filtros de diccionario."""
    total_entities: int = 0
    
    # Por decisión
    force_anonymize: int = 0  # Whitelist match
    force_ignore: int = 0     # Blacklist/CIE10 match
    escalate: int = 0         # Sin match -> LLM
    
    # Por tipo de lista
    whitelist_matches: int = 0
    blacklist_matches: int = 0
    cie10_matches: int = 0
    nomenclator_matches: int = 0
    
    # Contra ground truth
    whitelist_correct: int = 0
    whitelist_incorrect: int = 0
    blacklist_correct: int = 0
    blacklist_incorrect: int = 0
    
    @property
    def resolution_rate(self) -> float:
        """Porcentaje de entidades resueltas sin LLM."""
        resolved = self.force_anonymize + self.force_ignore
        return resolved / self.total_entities * 100 if self.total_entities > 0 else 0.0
    
    @property
    def whitelist_precision(self) -> float:
        total = self.whitelist_correct + self.whitelist_incorrect
        return self.whitelist_correct / total if total > 0 else 0.0
    
    @property
    def blacklist_precision(self) -> float:
        total = self.blacklist_correct + self.blacklist_incorrect
        return self.blacklist_correct / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entities": self.total_entities,
            "decisions": {
                "force_anonymize": self.force_anonymize,
                "force_ignore": self.force_ignore,
                "escalate_to_llm": self.escalate,
            },
            "list_matches": {
                "whitelist": self.whitelist_matches,
                "blacklist": self.blacklist_matches,
                "cie10": self.cie10_matches,
                "nomenclator": self.nomenclator_matches,
            },
            "accuracy": {
                "whitelist_correct": self.whitelist_correct,
                "whitelist_incorrect": self.whitelist_incorrect,
                "whitelist_precision": round(self.whitelist_precision, 4),
                "blacklist_correct": self.blacklist_correct,
                "blacklist_incorrect": self.blacklist_incorrect,
                "blacklist_precision": round(self.blacklist_precision, 4),
            },
            "resolution_rate_percent": round(self.resolution_rate, 2),
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
            text = ent.get('texto_detectado', ent.get('entity_text', ent.get('text', ''))).strip().lower()
            label = ent.get('etiqueta', ent.get('label', ''))
            
            if text and label and doc_id:
                ground_truth[doc_id].add((text, label))
    
    elif gt_path.is_dir():
        for json_file in gt_path.glob("*.json"):
            doc_id = json_file.stem
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entities = data if isinstance(data, list) else data.get('entities', [])
            
            for ent in entities:
                text = ent.get('text', ent.get('texto_detectado', '')).strip().lower()
                label = ent.get('label', ent.get('etiqueta', ''))
                
                if text and label:
                    ground_truth[doc_id].add((text, label))
    
    return dict(ground_truth)


# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluate_dict_filters(
    entities: List[Dict[str, Any]],
    ground_truth: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], DictFilterMetrics]:
    """
    Evalúa los filtros de diccionario sobre un conjunto de entidades.
    
    Args:
        entities: Lista de entidades a filtrar
        ground_truth: Ground truth opcional
        config: Configuración de filtros
        
    Returns:
        Tuple de (resultados, métricas)
    """
    metrics = DictFilterMetrics()
    metrics.total_entities = len(entities)
    
    # Ejecutar filtros
    logger.info(f"Procesando {len(entities)} entidades con filtros de diccionario...")
    results = apply_dict_filters(entities, config=config)
    
    # Contar por tipo de decisión
    for result in results:
        decision = result.get('filter_decision', result.get('decision', ''))
        matched_list = result.get('matched_list', '')
        
        if decision == 'FORCE_ANONYMIZE' or decision == 'KEEP':
            metrics.force_anonymize += 1
        elif decision == 'FORCE_IGNORE' or decision == 'FILTER':
            metrics.force_ignore += 1
        else:
            metrics.escalate += 1
        
        # Contar por tipo de lista
        if 'whitelist' in matched_list.lower():
            metrics.whitelist_matches += 1
        elif 'blacklist' in matched_list.lower():
            metrics.blacklist_matches += 1
        elif 'cie10' in matched_list.lower():
            metrics.cie10_matches += 1
        elif 'nomenclator' in matched_list.lower():
            metrics.nomenclator_matches += 1
    
    # Evaluar contra ground truth si está disponible
    if ground_truth:
        logger.info("Evaluando contra ground truth...")
        
        for result in results:
            doc_id = result.get('document_id', result.get('doc_id', ''))
            text = result.get('entity_text', result.get('text', '')).strip().lower()
            label = result.get('label', '')
            decision = result.get('filter_decision', result.get('decision', ''))
            matched_list = result.get('matched_list', '')
            
            gt_set = ground_truth.get(doc_id, set())
            is_in_gt = (text, label) in gt_set
            
            # Whitelist debería dar KEEP para entidades en GT
            if 'whitelist' in matched_list.lower():
                if is_in_gt:
                    metrics.whitelist_correct += 1
                else:
                    metrics.whitelist_incorrect += 1
            
            # Blacklist debería dar FILTER para entidades NO en GT
            elif 'blacklist' in matched_list.lower() or 'cie10' in matched_list.lower():
                if not is_in_gt:
                    metrics.blacklist_correct += 1
                else:
                    metrics.blacklist_incorrect += 1
    
    return results, metrics


def print_metrics(metrics: DictFilterMetrics):
    """Imprime métricas de forma legible."""
    print("\n" + "=" * 70)
    print("📊 MÉTRICAS DEL MÓDULO DICT_FILTERS")
    print("=" * 70)
    
    print(f"\n🔢 TOTALES:")
    print(f"  Entidades procesadas: {metrics.total_entities}")
    print(f"  FORCE_ANONYMIZE (whitelist): {metrics.force_anonymize}")
    print(f"  FORCE_IGNORE (blacklist/CIE10): {metrics.force_ignore}")
    print(f"  ESCALATE_TO_LLM: {metrics.escalate}")
    
    print(f"\n📋 MATCHES POR TIPO DE LISTA:")
    print(f"  Whitelist: {metrics.whitelist_matches}")
    print(f"  Blacklist: {metrics.blacklist_matches}")
    print(f"  CIE10: {metrics.cie10_matches}")
    print(f"  Nomenclátor: {metrics.nomenclator_matches}")
    
    print(f"\n📈 TASA DE RESOLUCIÓN: {metrics.resolution_rate:.1f}%")
    print(f"   (Entidades resueltas sin necesidad de LLM)")
    
    if metrics.whitelist_correct > 0 or metrics.blacklist_correct > 0:
        print(f"\n🎯 PRECISIÓN VS GROUND TRUTH:")
        print(f"  Whitelist: {metrics.whitelist_precision:.4f} ({metrics.whitelist_correct}/{metrics.whitelist_correct + metrics.whitelist_incorrect})")
        print(f"  Blacklist: {metrics.blacklist_precision:.4f} ({metrics.blacklist_correct}/{metrics.blacklist_correct + metrics.blacklist_incorrect})")
    
    print("=" * 70)


def save_results(
    results: List[Dict[str, Any]],
    metrics: DictFilterMetrics,
    output_path: str
):
    """Guarda resultados y métricas en JSON."""
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "module": "dict_filters",
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
        description="Evalúa el módulo de filtros de diccionario y calcula métricas",
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
    
    results, metrics = evaluate_dict_filters(entities, ground_truth)
    
    print_metrics(metrics)
    save_results(results, metrics, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
