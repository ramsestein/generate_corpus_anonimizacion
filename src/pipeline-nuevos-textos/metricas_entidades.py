#!/usr/bin/env python3
"""
SCRIPT DE MÉTRICAS PARA EVALUACIÓN DE ENTIDADES NER - v3.0
===========================================================

CORREGIDO: Lógica de matching limpia basada en conjuntos (sets) por documento.

PROBLEMA ANTERIOR:
- Se contaban FN de documentos SIN predicciones (13,985 docs de GT vs 50 docs predichos).
- Esto generaba >100,000 FN cuando el GT relevante es ~700 entidades.

SOLUCIÓN:
- Por defecto, evaluar SOLO documentos que tienen AMBOS: predicciones Y ground truth.
- Opción --include-missing para incluir documentos solo en GT (genera FN masivos).

LÓGICA DE MATCHING (por documento, usando SETS):
------------------------------------------------
Para cada documento con predicciones:
    1. Construir SET de entidades únicas de GT: {(texto_norm, label), ...}
    2. Construir SET de entidades únicas de predicciones: {(texto_norm, label), ...}
    3. TP = |intersección de ambos sets|
    4. FP = |predicciones - GT|
    5. FN = |GT - predicciones|

INVARIANTE MATEMÁTICO:
- TP + FN = |GT del documento| (siempre)
- TP + FP = |Predicciones del documento| (siempre)
- FN total <= Total entidades GT evaluadas (siempre)

AUTOR: Pipeline Anonimización Clínica
VERSION: 3.0.0 - Lógica corregida con sets
"""

from __future__ import annotations

import json
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Literal

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "entidades-procesadas-para-metricas.json"
GROUND_TRUTH_DIR = PROJECT_ROOT / "corpus" / "ANTIGUO" / "entidades"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class DocumentMetrics:
    """Métricas calculadas para un documento específico."""
    doc_id: str
    gt_entities: Set[Tuple[str, str]]  # {(texto_norm, label), ...}
    pred_entities: Set[Tuple[str, str]]
    tp: int = 0
    fp: int = 0
    fn: int = 0
    
    def __post_init__(self):
        # Calcular métricas usando operaciones de conjuntos
        self.tp = len(self.gt_entities & self.pred_entities)
        self.fp = len(self.pred_entities - self.gt_entities)
        self.fn = len(self.gt_entities - self.pred_entities)
        
        # INVARIANTE: TP + FN debe ser igual al tamaño del GT
        assert self.tp + self.fn == len(self.gt_entities), \
            f"Bug en {self.doc_id}: TP({self.tp}) + FN({self.fn}) != GT({len(self.gt_entities)})"
        # INVARIANTE: TP + FP debe ser igual al tamaño de predicciones
        assert self.tp + self.fp == len(self.pred_entities), \
            f"Bug en {self.doc_id}: TP({self.tp}) + FP({self.fp}) != Pred({len(self.pred_entities)})"


@dataclass
class GlobalMetrics:
    """Métricas globales agregadas."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    total_gt_entities: int = 0
    total_pred_entities: int = 0
    docs_evaluated: int = 0
    docs_only_gt: int = 0
    docs_only_pred: int = 0
    
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
    
    def validate(self):
        """Valida que las métricas son matemáticamente coherentes."""
        # FN nunca puede ser mayor que el total de GT evaluado
        if self.fn > self.total_gt_entities:
            raise ValueError(
                f"BUG DETECTADO: FN ({self.fn}) > Total GT ({self.total_gt_entities}). "
                "Esto es matemáticamente imposible."
            )
        # TP + FN debe ser <= total GT
        if self.tp + self.fn > self.total_gt_entities:
            raise ValueError(
                f"BUG DETECTADO: TP+FN ({self.tp + self.fn}) > Total GT ({self.total_gt_entities})"
            )
        # TP + FP debe ser <= total predicciones
        if self.tp + self.fp > self.total_pred_entities:
            raise ValueError(
                f"BUG DETECTADO: TP+FP ({self.tp + self.fp}) > Total Pred ({self.total_pred_entities})"
            )


# ============================================================================
# FUNCIONES DE NORMALIZACIÓN
# ============================================================================

def normalize_entity(text: str, label: str) -> Tuple[str, str]:
    """
    Normaliza una entidad para comparación.
    
    Returns:
        Tupla (texto_normalizado, label_normalizado)
    """
    return (text.strip().lower(), label.strip().upper())


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_predictions(predictions_path: Path) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Carga las predicciones y las agrupa por documento como SETS de entidades únicas.
    
    Returns:
        Dict[doc_id] = {(texto_norm, label), ...}
    """
    if not predictions_path.exists():
        raise FileNotFoundError(f"Archivo de predicciones no encontrado: {predictions_path}")
    
    print(f"[INFO] Cargando predicciones desde: {predictions_path.name}")
    
    with open(predictions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'entities' not in data:
        raise KeyError("El archivo de predicciones debe tener un campo 'entities'")
    
    # Agrupar por documento como SETS (elimina duplicados automáticamente)
    predictions: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    total_mentions = 0
    
    for item in data['entities']:
        doc_id = item.get('doc_id', '')
        text = item.get('text', '')
        label = item.get('label', '')
        
        if not doc_id or not text:
            continue
        
        entity_key = normalize_entity(text, label)
        predictions[doc_id].add(entity_key)
        total_mentions += 1
    
    total_unique = sum(len(s) for s in predictions.values())
    print(f"  → {total_mentions} menciones → {total_unique} entidades únicas en {len(predictions)} documentos")
    
    return dict(predictions)


def load_ground_truth(gt_dir: Path, filter_docs: Optional[Set[str]] = None) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Carga el ground truth y lo agrupa por documento como SETS de entidades únicas.
    
    Args:
        gt_dir: Directorio con archivos JSON de ground truth.
        filter_docs: Si se especifica, solo carga GT para estos doc_ids.
    
    Returns:
        Dict[doc_id] = {(texto_norm, label), ...}
    """
    if not gt_dir.exists():
        raise FileNotFoundError(f"Directorio de ground truth no encontrado: {gt_dir}")
    
    print(f"[INFO] Cargando ground truth desde: {gt_dir.name}/")
    if filter_docs:
        print(f"  → Filtrando a {len(filter_docs)} documentos específicos")
    
    ground_truth: Dict[str, Set[Tuple[str, str]]] = {}
    total_mentions = 0
    files_processed = 0
    
    for json_file in gt_dir.glob("*.json"):
        doc_id = json_file.stem
        
        # Si hay filtro, saltar documentos no incluidos
        if filter_docs and doc_id not in filter_docs:
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # El doc_id puede estar en el contenido o ser el nombre del archivo
            doc_id_from_content = data.get('id', doc_id)
            
            entities_set: Set[Tuple[str, str]] = set()
            for item in data.get('data', []):
                text = item.get('text', '')
                label = item.get('entity', '')
                
                if not text:
                    continue
                
                entity_key = normalize_entity(text, label)
                entities_set.add(entity_key)
                total_mentions += 1
            
            ground_truth[doc_id] = entities_set
            files_processed += 1
            
        except Exception as e:
            print(f"  [WARN] Error procesando {json_file.name}: {e}")
            continue
    
    total_unique = sum(len(s) for s in ground_truth.values())
    print(f"  → {total_mentions} menciones → {total_unique} entidades únicas en {files_processed} documentos")
    
    return ground_truth


# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluate(
    predictions: Dict[str, Set[Tuple[str, str]]],
    ground_truth: Dict[str, Set[Tuple[str, str]]],
    include_missing_predictions: bool = False
) -> Tuple[GlobalMetrics, List[DocumentMetrics]]:
    """
    Evalúa predicciones contra ground truth usando operaciones de conjuntos.
    
    LÓGICA LIMPIA:
    - Por defecto, solo evalúa documentos con AMBOS: predicciones Y GT.
    - Si include_missing_predictions=True, también cuenta FN de docs sin predicciones.
    
    Args:
        predictions: Dict[doc_id] = set de entidades predichas
        ground_truth: Dict[doc_id] = set de entidades GT
        include_missing_predictions: Si incluir docs sin predicciones (genera muchos FN)
    
    Returns:
        (GlobalMetrics, lista de DocumentMetrics)
    """
    print(f"\n[INFO] Evaluando métricas...")
    print(f"  → include_missing_predictions: {include_missing_predictions}")
    
    global_metrics = GlobalMetrics()
    doc_metrics: List[DocumentMetrics] = []
    
    # Determinar qué documentos evaluar
    pred_docs = set(predictions.keys())
    gt_docs = set(ground_truth.keys())
    
    if include_missing_predictions:
        # Evaluar todos los documentos que tengan GT
        docs_to_evaluate = gt_docs
    else:
        # Solo documentos con AMBOS (predicciones Y GT)
        docs_to_evaluate = pred_docs & gt_docs
    
    print(f"  → Documentos a evaluar: {len(docs_to_evaluate)}")
    
    for doc_id in docs_to_evaluate:
        gt_set = ground_truth.get(doc_id, set())
        pred_set = predictions.get(doc_id, set())
        
        # Crear métricas del documento (calcula TP, FP, FN automáticamente)
        dm = DocumentMetrics(
            doc_id=doc_id,
            gt_entities=gt_set,
            pred_entities=pred_set
        )
        doc_metrics.append(dm)
        
        # Agregar a métricas globales
        global_metrics.tp += dm.tp
        global_metrics.fp += dm.fp
        global_metrics.fn += dm.fn
        global_metrics.total_gt_entities += len(gt_set)
        global_metrics.total_pred_entities += len(pred_set)
        
        if pred_set and gt_set:
            global_metrics.docs_evaluated += 1
        elif gt_set and not pred_set:
            global_metrics.docs_only_gt += 1
        elif pred_set and not gt_set:
            global_metrics.docs_only_pred += 1
    
    # Validar que las métricas son coherentes
    global_metrics.validate()
    
    # Ordenar por FN descendente para análisis
    doc_metrics.sort(key=lambda x: x.fn, reverse=True)
    
    return global_metrics, doc_metrics


# ============================================================================
# SALIDA Y REPORTES
# ============================================================================

def print_summary(metrics: GlobalMetrics, doc_metrics: List[DocumentMetrics]) -> None:
    """Imprime resumen detallado de métricas."""
    
    print("\n" + "="*70)
    print("RESUMEN DE MÉTRICAS DE EVALUACIÓN")
    print("="*70)
    
    print(f"\n📊 Cobertura de documentos:")
    print(f"   - Documentos evaluados (con ambos): {metrics.docs_evaluated}")
    print(f"   - Documentos solo con GT (sin pred): {metrics.docs_only_gt}")
    print(f"   - Documentos solo con pred (sin GT): {metrics.docs_only_pred}")
    
    print(f"\n📊 Entidades:")
    print(f"   - Total entidades GT evaluadas: {metrics.total_gt_entities}")
    print(f"   - Total entidades predichas: {metrics.total_pred_entities}")
    
    print(f"\n📈 Métricas Globales:")
    print(f"   - True Positives (TP):  {metrics.tp}")
    print(f"   - False Positives (FP): {metrics.fp}")
    print(f"   - False Negatives (FN): {metrics.fn}")
    print(f"   - Precision: {metrics.precision:.4f} ({metrics.precision*100:.2f}%)")
    print(f"   - Recall:    {metrics.recall:.4f} ({metrics.recall*100:.2f}%)")
    print(f"   - F1 Score:  {metrics.f1:.4f} ({metrics.f1*100:.2f}%)")
    
    # Verificación matemática
    print(f"\n✅ Verificación matemática:")
    print(f"   - TP + FN = {metrics.tp + metrics.fn} (debe ser == {metrics.total_gt_entities} GT)")
    print(f"   - TP + FP = {metrics.tp + metrics.fp} (debe ser == {metrics.total_pred_entities} Pred)")
    print(f"   - FN <= Total GT: {metrics.fn} <= {metrics.total_gt_entities} → {'✓ OK' if metrics.fn <= metrics.total_gt_entities else '✗ ERROR'}")
    
    # Top 5 documentos con más FN
    if doc_metrics:
        print(f"\n📋 Top 5 documentos con más FN:")
        for dm in doc_metrics[:5]:
            print(f"   - {dm.doc_id[:30]}...: GT={len(dm.gt_entities)}, Pred={len(dm.pred_entities)}, TP={dm.tp}, FP={dm.fp}, FN={dm.fn}")
    
    print("\n" + "="*70)


def print_debug_document(doc_metrics: List[DocumentMetrics]) -> None:
    """Imprime análisis detallado del documento con más FN para depuración."""
    
    if not doc_metrics:
        return
    
    dm = doc_metrics[0]  # El de más FN (ya está ordenado)
    
    print(f"\n🔍 DEBUG: Análisis detallado del documento con más FN")
    print(f"   Doc ID: {dm.doc_id}")
    print(f"   GT entities: {len(dm.gt_entities)}")
    print(f"   Pred entities: {len(dm.pred_entities)}")
    print(f"   TP: {dm.tp}, FP: {dm.fp}, FN: {dm.fn}")
    
    # Mostrar algunas entidades
    print(f"\n   Entidades GT (primeras 5):")
    for i, (text, label) in enumerate(list(dm.gt_entities)[:5]):
        print(f"      {i+1}. [{label}] '{text}'")
    
    print(f"\n   Entidades Pred (primeras 5):")
    for i, (text, label) in enumerate(list(dm.pred_entities)[:5]):
        print(f"      {i+1}. [{label}] '{text}'")
    
    # True Positives (intersección)
    tp_set = dm.gt_entities & dm.pred_entities
    print(f"\n   True Positives (primeras 5 de {len(tp_set)}):")
    for i, (text, label) in enumerate(list(tp_set)[:5]):
        print(f"      {i+1}. [{label}] '{text}'")
    
    # False Negatives (en GT pero no en Pred)
    fn_set = dm.gt_entities - dm.pred_entities
    print(f"\n   False Negatives (primeras 5 de {len(fn_set)}):")
    for i, (text, label) in enumerate(list(fn_set)[:5]):
        print(f"      {i+1}. [{label}] '{text}'")
    
    # False Positives (en Pred pero no en GT)
    fp_set = dm.pred_entities - dm.gt_entities
    print(f"\n   False Positives (primeras 5 de {len(fp_set)}):")
    for i, (text, label) in enumerate(list(fp_set)[:5]):
        print(f"      {i+1}. [{label}] '{text}'")
    
    # Verificar invariante
    print(f"\n   ✅ Verificación: TP({dm.tp}) + FN({dm.fn}) = {dm.tp + dm.fn} == GT({len(dm.gt_entities)}) → {'✓' if dm.tp + dm.fn == len(dm.gt_entities) else '✗'}")


def save_results_json(
    metrics: GlobalMetrics,
    doc_metrics: List[DocumentMetrics],
    output_path: Path
) -> None:
    """Guarda resultados en JSON."""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "3.0.0",
            "evaluation_mode": "set-based (unique entities per document)"
        },
        "global_metrics": {
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "total_gt_entities": metrics.total_gt_entities,
            "total_pred_entities": metrics.total_pred_entities,
            "docs_evaluated": metrics.docs_evaluated,
        },
        "validation": {
            "fn_lte_gt": metrics.fn <= metrics.total_gt_entities,
            "tp_plus_fn_eq_gt": metrics.tp + metrics.fn == metrics.total_gt_entities,
        },
        "top_fn_documents": [
            {
                "doc_id": dm.doc_id,
                "gt_count": len(dm.gt_entities),
                "pred_count": len(dm.pred_entities),
                "tp": dm.tp,
                "fp": dm.fp,
                "fn": dm.fn
            }
            for dm in doc_metrics[:20]
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Resultados guardados en: {output_path}")


def save_results_csv(doc_metrics: List[DocumentMetrics], output_path: Path) -> None:
    """Guarda métricas por documento en CSV."""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'doc_id', 'gt_count', 'pred_count', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'
        ], delimiter=';')
        
        writer.writeheader()
        for dm in doc_metrics:
            prec = dm.tp / (dm.tp + dm.fp) if (dm.tp + dm.fp) > 0 else 0
            rec = dm.tp / (dm.tp + dm.fn) if (dm.tp + dm.fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            writer.writerow({
                'doc_id': dm.doc_id,
                'gt_count': len(dm.gt_entities),
                'pred_count': len(dm.pred_entities),
                'tp': dm.tp,
                'fp': dm.fp,
                'fn': dm.fn,
                'precision': f"{prec:.4f}",
                'recall': f"{rec:.4f}",
                'f1': f"{f1:.4f}"
            })
    
    print(f"[INFO] CSV por documento guardado en: {output_path}")


# ============================================================================
# TESTS DE VERIFICACIÓN
# ============================================================================

def run_tests() -> int:
    """Ejecuta tests para verificar la lógica de métricas."""
    
    print("\n" + "="*70)
    print("EJECUTANDO TESTS DE VERIFICACIÓN")
    print("="*70 + "\n")
    
    all_passed = True
    
    # TEST 1: Caso básico
    print("TEST 1: Caso básico con entidades que coinciden y no coinciden")
    print("-" * 50)
    
    pred = {
        "doc1": {("juan pérez", "PERSON"), ("hospital central", "LOCATION"), ("extra", "OTHER")}
    }
    gt = {
        "doc1": {("juan pérez", "PERSON"), ("hospital central", "LOCATION"), ("missing", "PERSON")}
    }
    
    metrics, doc_metrics = evaluate(pred, gt)
    
    # Esperado: TP=2, FP=1 (extra), FN=1 (missing)
    expected = (2, 1, 1)
    actual = (metrics.tp, metrics.fp, metrics.fn)
    
    if actual == expected:
        print(f"  ✅ TP={actual[0]}, FP={actual[1]}, FN={actual[2]} (CORRECTO)")
    else:
        print(f"  ❌ TP={actual[0]}, FP={actual[1]}, FN={actual[2]}")
        print(f"     Esperado: TP={expected[0]}, FP={expected[1]}, FN={expected[2]}")
        all_passed = False
    
    # Verificar invariante
    if metrics.fn <= metrics.total_gt_entities:
        print(f"  ✅ FN ({metrics.fn}) <= Total GT ({metrics.total_gt_entities})")
    else:
        print(f"  ❌ FN ({metrics.fn}) > Total GT ({metrics.total_gt_entities}) - BUG!")
        all_passed = False
    
    # TEST 2: Documento sin predicciones (no debe generar FN si include_missing=False)
    print("\nTEST 2: Documento solo en GT (sin predicciones) - include_missing=False")
    print("-" * 50)
    
    pred2 = {"doc1": {("found", "PERSON")}}
    gt2 = {
        "doc1": {("found", "PERSON")},
        "doc2": {("not_predicted_1", "PERSON"), ("not_predicted_2", "LOCATION")}
    }
    
    metrics2, _ = evaluate(pred2, gt2, include_missing_predictions=False)
    
    # Solo debe evaluar doc1, no doc2
    # Esperado: TP=1, FP=0, FN=0 (doc2 ignorado)
    expected2 = (1, 0, 0)
    actual2 = (metrics2.tp, metrics2.fp, metrics2.fn)
    
    if actual2 == expected2:
        print(f"  ✅ TP={actual2[0]}, FP={actual2[1]}, FN={actual2[2]} (CORRECTO)")
        print(f"     doc2 correctamente ignorado (no tiene predicciones)")
    else:
        print(f"  ❌ TP={actual2[0]}, FP={actual2[1]}, FN={actual2[2]}")
        print(f"     Esperado: TP={expected2[0]}, FP={expected2[1]}, FN={expected2[2]}")
        all_passed = False
    
    # TEST 3: Mismo test pero con include_missing=True
    print("\nTEST 3: Documento solo en GT - include_missing=True")
    print("-" * 50)
    
    metrics3, _ = evaluate(pred2, gt2, include_missing_predictions=True)
    
    # Ahora doc2 debe generar 2 FN
    # Esperado: TP=1, FP=0, FN=2
    expected3 = (1, 0, 2)
    actual3 = (metrics3.tp, metrics3.fp, metrics3.fn)
    
    if actual3 == expected3:
        print(f"  ✅ TP={actual3[0]}, FP={actual3[1]}, FN={actual3[2]} (CORRECTO)")
        print(f"     doc2 incluido, genera 2 FN")
    else:
        print(f"  ❌ TP={actual3[0]}, FP={actual3[1]}, FN={actual3[2]}")
        print(f"     Esperado: TP={expected3[0]}, FP={expected3[1]}, FN={expected3[2]}")
        all_passed = False
    
    # TEST 4: Invariante FN <= GT
    print("\nTEST 4: Verificar invariante FN <= Total GT siempre")
    print("-" * 50)
    
    # Crear caso con muchas entidades
    large_gt = {f"doc{i}": {(f"entity_{j}", "LABEL") for j in range(10)} for i in range(5)}
    large_pred = {f"doc{i}": {(f"entity_{j}", "LABEL") for j in range(3)} for i in range(5)}
    
    metrics4, _ = evaluate(large_pred, large_gt)
    
    total_gt = sum(len(s) for s in large_gt.values())
    
    if metrics4.fn <= total_gt:
        print(f"  ✅ FN ({metrics4.fn}) <= Total GT ({total_gt})")
    else:
        print(f"  ❌ FN ({metrics4.fn}) > Total GT ({total_gt}) - BUG!")
        all_passed = False
    
    # Verificar que TP + FN = Total GT evaluado
    if metrics4.tp + metrics4.fn == metrics4.total_gt_entities:
        print(f"  ✅ TP + FN = {metrics4.tp + metrics4.fn} == Total GT evaluado ({metrics4.total_gt_entities})")
    else:
        print(f"  ❌ TP + FN = {metrics4.tp + metrics4.fn} != Total GT evaluado ({metrics4.total_gt_entities})")
        all_passed = False
    
    # RESUMEN
    print("\n" + "="*70)
    if all_passed:
        print("✅ TODOS LOS TESTS PASARON")
        return 0
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        return 1


# ============================================================================
# MAIN
# ============================================================================

def main(
    predictions_file: Optional[Path] = None,
    ground_truth_dir: Optional[Path] = None,
    include_missing: bool = False,
    run_verification_tests: bool = False,
    debug: bool = False
) -> int:
    """
    Función principal.
    
    Args:
        predictions_file: Ruta al archivo de predicciones.
        ground_truth_dir: Directorio de ground truth.
        include_missing: Si incluir documentos sin predicciones (genera muchos FN).
        run_verification_tests: Si ejecutar tests de verificación.
        debug: Si mostrar análisis detallado del documento con más FN.
    
    Returns:
        0 si éxito, 1 si error.
    """
    
    if run_verification_tests:
        return run_tests()
    
    print("\n" + "="*70)
    print("EVALUACIÓN DE ENTIDADES NER v3.0")
    print("="*70 + "\n")
    
    pred_path = predictions_file or PREDICTIONS_FILE
    gt_path = ground_truth_dir or GROUND_TRUTH_DIR
    
    print(f"📋 Configuración:")
    print(f"   - Incluir docs sin predicciones: {'Sí' if include_missing else 'No (recomendado)'}")
    
    try:
        # 1. Cargar predicciones
        print("\n[PASO 1] Cargando predicciones...")
        predictions = load_predictions(pred_path)
        
        # 2. Cargar ground truth (filtrado a docs con predicciones si include_missing=False)
        print("\n[PASO 2] Cargando ground truth...")
        filter_docs = None if include_missing else set(predictions.keys())
        ground_truth = load_ground_truth(gt_path, filter_docs=filter_docs)
        
        # 3. Evaluar
        print("\n[PASO 3] Evaluando...")
        metrics, doc_metrics = evaluate(predictions, ground_truth, include_missing)
        
        # 4. Mostrar resultados
        print_summary(metrics, doc_metrics)
        
        if debug:
            print_debug_document(doc_metrics)
        
        # 5. Guardar resultados
        print("\n[PASO 4] Guardando resultados...")
        json_path = OUTPUTS_DIR / "metricas_entidades.json"
        csv_path = OUTPUTS_DIR / "metricas_entidades_por_doc.csv"
        
        save_results_json(metrics, doc_metrics, json_path)
        save_results_csv(doc_metrics, csv_path)
        
        print("\n" + "="*70)
        print("✅ Evaluación completada exitosamente")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calcula métricas de evaluación para entidades NER (v3.0 - corregido)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANTE: Por defecto, solo evalúa documentos que tienen AMBOS predicciones Y ground truth.
Esto evita inflar los FN con documentos que no fueron procesados.

Ejemplos:
  # Evaluación estándar (solo docs con predicciones)
  python metricas_entidades.py
  
  # Incluir docs sin predicciones (genera muchos FN)
  python metricas_entidades.py --include-missing
  
  # Mostrar análisis detallado del doc con más FN
  python metricas_entidades.py --debug
  
  # Ejecutar tests de verificación
  python metricas_entidades.py --test
"""
    )
    parser.add_argument(
        "--predictions", "-p",
        type=Path,
        default=None,
        help=f"Ruta al archivo de predicciones (default: {PREDICTIONS_FILE.name})"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=Path,
        default=None,
        help=f"Directorio de ground truth (default: {GROUND_TRUTH_DIR.name})"
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Incluir documentos sin predicciones (genera muchos FN)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mostrar análisis detallado del documento con más FN"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Ejecutar tests de verificación de la lógica"
    )
    
    args = parser.parse_args()
    
    sys.exit(main(
        predictions_file=args.predictions,
        ground_truth_dir=args.ground_truth,
        include_missing=args.include_missing,
        run_verification_tests=args.test,
        debug=args.debug
    ))
