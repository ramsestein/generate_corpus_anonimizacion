#!/usr/bin/env python3
"""
ANÁLISIS COMPLETO DEL PIPELINE DE VALIDACIÓN DE ENTIDADES
==========================================================

Este script ejecuta el pipeline completo de filtrado + LLM Judge y genera
un análisis exhaustivo de errores por etiqueta con propuestas de mejora.

FLUJO:
------
1. Carga detecciones del NER entidades-procesadas-para-metricas.json
2. Aplica EntityFastFilter con todas las listas (whitelists, blacklists, CIE10)
3. Simula/carga decisiones LLM para entidades ESCALATE_TO_LLM
4. Compara contra ground truth (correcciones manuales)
5. Calcula métricas globales y por etiqueta
6. Analiza patrones de error
7. Genera informe markdown con recomendaciones

AUTOR: Pipeline Anonimización Clínica
"""

import json
import os
import sys
import re
import csv
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime

# Configurar paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LISTAS_DIR = PROJECT_ROOT / "LISTAS"
CORPUS_DIR = PROJECT_ROOT / "corpus"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Asegurar que podemos importar módulos locales
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from entity_fast_filter import EntityFastFilter, EnumDecision
except ImportError as e:
    print(f"Error importando EntityFastFilter: {e}")
    sys.exit(1)


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class EntityRecord:
    """Registro completo de una entidad con todas sus decisiones."""
    document_id: str
    entity_text: str
    label: str
    start: int
    end: int
    confidence: float
    model: str
    # Decisiones del pipeline
    filter_decision: str = ""  # FORCE_ANONYMIZE, FORCE_IGNORE, ESCALATE_TO_LLM
    llm_decision: Optional[bool] = None  # True/False si fue al LLM
    system_decision: Optional[bool] = None  # Decisión final del sistema
    # Ground truth
    ground_truth: Optional[bool] = None  # Corrección manual TRUE/FALSE
    # Contexto adicional
    context: str = ""
    matched_list: str = ""  # whitelist, blacklist, cie10, none


@dataclass  
class LabelMetrics:
    """Métricas calculadas para una etiqueta específica."""
    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    # Desglose por origen de decisión
    filter_correct: int = 0
    filter_incorrect: int = 0
    llm_correct: int = 0
    llm_incorrect: int = 0
    
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
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_detections() -> List[Dict]:
    """Carga las detecciones del NER ya hecho."""
    detections_path = PROJECT_ROOT / "entidades-procesadas-para-metricas.json"
    
    if not detections_path.exists():
        raise FileNotFoundError(f"No se encontró: {detections_path}")
    
    with open(detections_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Soportar ambos formatos de archivo
    if 'entities' in data:
        # Formato nuevo: {"metadata": {...}, "entities": [...]}
        entities = data['entities']
        # Adaptar nombres de campos al formato esperado
        detections = []
        for e in entities:
            detections.append({
                'doc_id': e.get('doc_id', ''),
                'etiqueta': e.get('label', ''),
                'texto_detectado': e.get('text', ''),
                'posicion_inicio': e.get('start', -1),
                'posicion_fin': e.get('end', -1),
                'confianza': e.get('confidence', 0.0),
                'modelo_detector': e.get('model', '')
            })
        print(f"[INFO] Cargadas {len(detections)} detecciones NER")
        return detections
    elif 'detecciones' in data:
        # Formato antiguo: {"total_detecciones": N, "detecciones": [...]}
        print(f"[INFO] Cargadas {data['total_detecciones']} detecciones NER")
        return data['detecciones']
    else:
        raise KeyError("Formato de archivo no reconocido (esperado 'entities' o 'detecciones')")


def load_ground_truth_aws2() -> Dict[str, Dict]:
    """
    Carga el ground truth desde el CSV con correcciones manuales.
    
    Returns:
        Dict[key] = {"ground_truth": bool, "label": str, ...}
        donde key = f"{doc_id}|{label}|{start}|{end}"
    """
    gt_path = CORPUS_DIR / "step6_validation" / "aws2-validation" / "detecciones_detalladas-resueltas.csv"
    
    if not gt_path.exists():
        raise FileNotFoundError(f"No se encontró ground truth: {gt_path}")
    
    ground_truth = {}
    
    with open(gt_path, 'r', encoding='latin-1') as f:
        # El CSV usa punto y coma como delimitador
        reader = csv.DictReader(f, delimiter=';')
        
        for row in reader:
            try:
                doc_id = row.get('doc_id', '').strip()
                label = row.get('etiqueta', '').strip()
                start = row.get('posicion_inicio', '').strip()
                end = row.get('posicion_fin', '').strip()
                correction = row.get('Correcci\ufffdn manual', row.get('Corrección manual', '')).strip()
                
                if not doc_id or not label:
                    continue
                
                # La corrección manual indica si ES dato sensible (TRUE) o NO (FALSE)
                is_sensitive = correction.upper() == 'TRUE'
                
                # Crear clave única
                key = f"{doc_id}|{label}|{start}|{end}"
                
                ground_truth[key] = {
                    "doc_id": doc_id,
                    "label": label,
                    "start": start,
                    "end": end,
                    "text": row.get('texto_detectado', ''),
                    "ground_truth": is_sensitive,
                    "model": row.get('modelo_detector', '')
                }
                
            except Exception as e:
                print(f"[WARN] Error procesando fila GT: {e}")
                continue
    
    print(f"[INFO] Cargadas {len(ground_truth)} anotaciones de ground truth")
    return ground_truth


def load_llm_decisions() -> Dict[str, Dict]:
    """
    Carga las decisiones del LLM desde los archivos de verificación.
    
    Returns:
        Dict[doc_id] = {(start, end, text): llm_decision_bool}
    """
    llm_decisions = defaultdict(dict)
    validation_dir = CORPUS_DIR / "step6_validation" / "aws2-validation"
    
    # Buscar archivos de verificación
    for json_file in validation_dir.glob("*_verification_result.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            doc_id = data.get('document_id', json_file.stem.replace('_verification_result', ''))
            
            # Extraer decisiones de entidades combinadas
            combined = data.get('combined_analysis', {})
            suspicious = combined.get('suspicious_entities', [])
            
            for entity in suspicious:
                start = entity.get('start', -1)
                end = entity.get('end', -1)
                text = entity.get('actual_text', entity.get('word', '')).strip()
                
                # La verificación indica si fue detectado como "problemático"
                # Necesitamos buscar en test_results.json para la decisión real del LLM
                key = (start, end, text)
                # Por ahora marcamos como detectado
                llm_decisions[doc_id][key] = True
                
        except Exception as e:
            print(f"[WARN] Error cargando {json_file.name}: {e}")
            continue
    
    print(f"[INFO] Cargadas decisiones LLM para {len(llm_decisions)} documentos")
    return dict(llm_decisions)


def load_test_results() -> Dict[str, List[Dict]]:
    """
    Carga los resultados completos del test (NER + LLM) desde test_results.json.
    
    Returns:
        Dict[doc_id] = [list of entity results with llm_response]
    """
    test_results_path = OUTPUTS_DIR / "test_results.json"
    
    if not test_results_path.exists():
        print(f"[WARN] No se encontró test_results.json, usando solo filtro")
        return {}
    
    with open(test_results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Organizar por documento
    by_doc = defaultdict(list)
    for entity in results:
        doc_id = entity.get('document_id', '')
        by_doc[doc_id].append(entity)
    
    print(f"[INFO] Cargados {len(results)} resultados de test_results.json")
    return dict(by_doc)


# ============================================================================
# INICIALIZACIÓN DEL FILTRO
# ============================================================================

def initialize_filter(
    whitelist_paths_arg: Optional[List[str]] = None,
    blacklist_paths_arg: Optional[List[str]] = None,
    cie10_path_arg: Optional[str] = None
) -> EntityFastFilter:
    """Inicializa EntityFastFilter con todas las listas disponibles.

    Args:
        whitelist_paths_arg: Lista opcional de rutas a ficheros whitelist (JSON).
        blacklist_paths_arg: Lista opcional de rutas a ficheros blacklist (JSON).
        cie10_path_arg: Ruta opcional al fichero CIE10 (xls/tsv).
    """

    # Paths por defecto si no se han pasado por args
    default_whitelist = [
        str(DATA_DIR / "hospitales.json"),
        str(DATA_DIR / "lugares.json"),
    ]
    default_blacklist = [
        str(DATA_DIR / "medicamentos.json"),
        str(DATA_DIR / "patologias.json"),
    ]
    default_cie10 = str(LISTAS_DIR / "cie10.xls")

    whitelist_paths = whitelist_paths_arg if whitelist_paths_arg is not None else default_whitelist
    blacklist_paths = blacklist_paths_arg if blacklist_paths_arg is not None else default_blacklist
    cie10_path = cie10_path_arg if cie10_path_arg is not None else default_cie10

    # Filtrar solo los que existen
    whitelist_paths = [p for p in whitelist_paths if Path(p).exists()]
    blacklist_paths = [p for p in blacklist_paths if Path(p).exists()]
    if cie10_path and not Path(cie10_path).exists():
        cie10_path = None

    print(f"[INFO] Inicializando EntityFastFilter...")
    print(f"  → Whitelists: {[Path(p).name for p in whitelist_paths]}")
    print(f"  → Blacklists: {[Path(p).name for p in blacklist_paths]}")
    print(f"  → CIE10: {Path(cie10_path).name if cie10_path else 'No disponible'}")

    entity_filter = EntityFastFilter(
        whitelist_paths=whitelist_paths if whitelist_paths else None,
        blacklist_paths=blacklist_paths if blacklist_paths else None,
        cie10_path=cie10_path
    )

    stats = entity_filter.get_stats()
    print(f"  → WhiteList terms: {stats.get('whitelist_terms', 0)}")
    print(f"  → BlackList terms: {stats.get('blacklist_terms', 0)}")
    if stats.get('cie10_loaded'):
        print(f"  → CIE10 terms: {stats.get('cie10_terms', 0)}")

    return entity_filter


# ============================================================================
# PROCESAMIENTO DEL PIPELINE
# ============================================================================

def process_pipeline(
    detections: List[Dict],
    ground_truth: Dict[str, Dict],
    entity_filter: EntityFastFilter,
    test_results: Dict[str, List[Dict]]
) -> List[EntityRecord]:
    """
    Ejecuta el pipeline completo sobre las detecciones.
    
    Args:
        detections: Lista de detecciones NER
        ground_truth: Dict con correcciones manuales
        entity_filter: Instancia de EntityFastFilter
        test_results: Resultados previos del LLM (si existen)
    
    Returns:
        Lista de EntityRecord con todas las decisiones y ground truth
    """
    print(f"\n[INFO] Procesando {len(detections)} detecciones...")
    
    records = []
    
    # Indexar test_results para búsqueda rápida
    test_results_index = {}
    for doc_id, entities in test_results.items():
        for e in entities:
            key = (e.get('start', -1), e.get('end', -1), e.get('keyword', '').strip())
            test_results_index[(doc_id, key)] = e
    
    for det in detections:
        doc_id = det.get('doc_id', '')
        label = det.get('etiqueta', '')
        text = det.get('texto_detectado', '').strip()
        start = det.get('posicion_inicio', -1)
        end = det.get('posicion_fin', -1)
        confidence = float(det.get('confianza', 0.0))
        model = det.get('modelo_detector', '')
        
        # Crear registro
        record = EntityRecord(
            document_id=doc_id,
            entity_text=text,
            label=label,
            start=int(start) if start else -1,
            end=int(end) if end else -1,
            confidence=confidence,
            model=model
        )
        
        # 1. Aplicar filtro determinista
        try:
            decision = entity_filter.evaluate_candidate(text, label)
            record.filter_decision = decision.name
            
            # Identificar qué lista matcheó
            if decision == EnumDecision.FORCE_ANONYMIZE:
                record.matched_list = "whitelist"
            elif decision == EnumDecision.FORCE_IGNORE:
                # Verificar si es CIE10 o blacklist normal
                if entity_filter.is_cie10_match(text):
                    record.matched_list = "cie10"
                else:
                    record.matched_list = "blacklist"
            else:
                record.matched_list = "none"
                
        except Exception as e:
            print(f"[WARN] Error en filtro para '{text}': {e}")
            record.filter_decision = "ESCALATE_TO_LLM"
            record.matched_list = "error"
        
        # 2. Buscar decisión LLM si fue escalado
        if record.filter_decision == "ESCALATE_TO_LLM":
            # Buscar en test_results
            lookup_key = (doc_id, (record.start, record.end, text))
            if lookup_key in test_results_index:
                llm_result = test_results_index[lookup_key]
                llm_response = llm_result.get('llm_response', '').strip().upper()
                record.llm_decision = llm_response == 'TRUE'
            else:
                # Buscar por aproximación (solo texto y label)
                for (d_id, key), result in test_results_index.items():
                    if d_id == doc_id and result.get('keyword', '').strip() == text:
                        llm_response = result.get('llm_response', '').strip().upper()
                        record.llm_decision = llm_response == 'TRUE'
                        break
        
        # 3. Calcular decisión final del sistema
        if record.filter_decision == "FORCE_ANONYMIZE":
            record.system_decision = True
        elif record.filter_decision == "FORCE_IGNORE":
            record.system_decision = False
        elif record.llm_decision is not None:
            record.system_decision = record.llm_decision
        else:
            # Sin decisión LLM disponible, asumimos que debería anonimizarse
            record.system_decision = True
        
        # 4. Buscar ground truth
        gt_key = f"{doc_id}|{label}|{start}|{end}"
        if gt_key in ground_truth:
            record.ground_truth = ground_truth[gt_key]['ground_truth']
        else:
            # Intentar buscar sin posición exacta (por texto)
            for k, v in ground_truth.items():
                parts = k.split('|')
                if len(parts) >= 2 and parts[0] == doc_id and parts[1] == label:
                    if v['text'].strip() == text:
                        record.ground_truth = v['ground_truth']
                        break
        
        records.append(record)
    
    # Estadísticas de procesamiento
    with_gt = sum(1 for r in records if r.ground_truth is not None)
    escalated = sum(1 for r in records if r.filter_decision == "ESCALATE_TO_LLM")
    with_llm = sum(1 for r in records if r.llm_decision is not None)
    
    print(f"[INFO] Procesamiento completado:")
    print(f"  → Total registros: {len(records)}")
    print(f"  → Con ground truth: {with_gt}")
    print(f"  → Escalados a LLM: {escalated}")
    print(f"  → Con decisión LLM: {with_llm}")
    
    return records


# ============================================================================
# CÁLCULO DE MÉTRICAS
# ============================================================================

def calculate_metrics(records: List[EntityRecord]) -> Tuple[LabelMetrics, Dict[str, LabelMetrics]]:
    """
    Calcula métricas globales y por etiqueta.
    
    Returns:
        (global_metrics, {label: metrics})
    """
    print(f"\n[INFO] Calculando métricas...")
    
    # Solo considerar registros con ground truth
    valid_records = [r for r in records if r.ground_truth is not None]
    print(f"  → Registros con ground truth: {len(valid_records)}/{len(records)}")
    
    # Métricas globales
    global_metrics = LabelMetrics(label="GLOBAL")
    
    # Métricas por etiqueta
    by_label: Dict[str, LabelMetrics] = {}
    
    for record in valid_records:
        # Obtener o crear métricas para la etiqueta
        if record.label not in by_label:
            by_label[record.label] = LabelMetrics(label=record.label)
        
        label_metrics = by_label[record.label]
        
        # Clasificar resultado
        system_positive = record.system_decision  # Sistema dice "anonimizar"
        gt_positive = record.ground_truth  # Ground truth dice "es sensible"
        
        if system_positive and gt_positive:
            # True Positive: Sistema dice anonimizar y es correcto
            global_metrics.tp += 1
            label_metrics.tp += 1
        elif system_positive and not gt_positive:
            # False Positive: Sistema dice anonimizar pero no debería
            global_metrics.fp += 1
            label_metrics.fp += 1
        elif not system_positive and gt_positive:
            # False Negative: Sistema no anonimiza pero debería
            global_metrics.fn += 1
            label_metrics.fn += 1
        else:
            # True Negative: Sistema no anonimiza y es correcto
            global_metrics.tn += 1
            label_metrics.tn += 1
        
        # Contabilizar origen de la decisión
        filter_decided = record.filter_decision in ["FORCE_ANONYMIZE", "FORCE_IGNORE"]
        is_correct = (system_positive == gt_positive)
        
        if filter_decided:
            if is_correct:
                global_metrics.filter_correct += 1
                label_metrics.filter_correct += 1
            else:
                global_metrics.filter_incorrect += 1
                label_metrics.filter_incorrect += 1
        else:
            if is_correct:
                global_metrics.llm_correct += 1
                label_metrics.llm_correct += 1
            else:
                global_metrics.llm_incorrect += 1
                label_metrics.llm_incorrect += 1
    
    return global_metrics, by_label


# ============================================================================
# ANÁLISIS DE ERRORES
# ============================================================================

@dataclass
class ErrorExample:
    """Ejemplo de error para análisis."""
    document_id: str
    entity_text: str
    label: str
    error_type: str  # "FP" o "FN"
    decision_source: str  # "filter_only", "llm_only", "combined"
    filter_decision: str
    llm_decision: Optional[bool]
    system_decision: bool
    ground_truth: bool
    matched_list: str
    context: str = ""
    confidence: float = 0.0


def analyze_errors(records: List[EntityRecord]) -> Dict[str, List[ErrorExample]]:
    """
    Analiza los errores y extrae ejemplos por etiqueta.
    
    Returns:
        Dict[label] = [lista de ErrorExample]
    """
    print(f"\n[INFO] Analizando errores...")
    
    errors_by_label: Dict[str, List[ErrorExample]] = defaultdict(list)
    
    for record in records:
        if record.ground_truth is None:
            continue
        
        system_positive = record.system_decision
        gt_positive = record.ground_truth
        
        # Solo nos interesan los errores (FP y FN)
        if system_positive == gt_positive:
            continue
        
        error_type = "FP" if system_positive and not gt_positive else "FN"
        
        # Determinar origen de la decisión
        if record.filter_decision in ["FORCE_ANONYMIZE", "FORCE_IGNORE"]:
            decision_source = "filter_only"
        elif record.llm_decision is not None:
            decision_source = "llm_only"
        else:
            decision_source = "unknown"
        
        error = ErrorExample(
            document_id=record.document_id,
            entity_text=record.entity_text,
            label=record.label,
            error_type=error_type,
            decision_source=decision_source,
            filter_decision=record.filter_decision,
            llm_decision=record.llm_decision,
            system_decision=record.system_decision,
            ground_truth=record.ground_truth,
            matched_list=record.matched_list,
            context=record.context,
            confidence=record.confidence
        )
        
        errors_by_label[record.label].append(error)
    
    total_errors = sum(len(errors) for errors in errors_by_label.values())
    print(f"  → Total errores detectados: {total_errors}")
    print(f"  → Etiquetas con errores: {len(errors_by_label)}")
    
    return dict(errors_by_label)


def detect_error_patterns(records: List[EntityRecord], errors_by_label: Dict[str, List[ErrorExample]]) -> Dict[str, Any]:
    """
    Detecta patrones comunes en los errores.
    
    Returns:
        Dict con patrones detectados
    """
    print(f"\n[INFO] Detectando patrones de error...")
    
    patterns = {
        "short_terms": [],  # Términos muy cortos (<=3 chars)
        "single_char": [],  # Términos de un solo carácter
        "numeric_only": [],  # Términos solo numéricos
        "mixed_case_issues": [],  # Problemas de mayúsculas/minúsculas
        "common_words": [],  # Palabras comunes que generan conflicto
        "abbreviations": [],  # Abreviaturas problemáticas
        "filter_errors": defaultdict(list),  # Errores del filtro por lista
        "llm_errors": defaultdict(list),  # Errores del LLM por tipo
        "label_concentration": {},  # Concentración de errores por etiqueta
    }
    
    all_errors = []
    for label_errors in errors_by_label.values():
        all_errors.extend(label_errors)
    
    # Analizar cada error
    for error in all_errors:
        text = error.entity_text.strip()
        
        # Términos cortos
        if len(text) <= 3:
            patterns["short_terms"].append({
                "text": text,
                "label": error.label,
                "error_type": error.error_type,
                "doc": error.document_id
            })
        
        # Un solo carácter
        if len(text) == 1:
            patterns["single_char"].append({
                "text": text,
                "label": error.label,
                "error_type": error.error_type
            })
        
        # Solo numérico
        if text.replace('.', '').replace(',', '').replace('-', '').replace('/', '').isdigit():
            patterns["numeric_only"].append({
                "text": text,
                "label": error.label,
                "error_type": error.error_type
            })
        
        # Abreviaturas (2-4 letras mayúsculas)
        if re.match(r'^[A-Z]{2,4}$', text):
            patterns["abbreviations"].append({
                "text": text,
                "label": error.label,
                "error_type": error.error_type
            })
        
        # Errores del filtro
        if error.decision_source == "filter_only":
            patterns["filter_errors"][error.matched_list].append({
                "text": text,
                "label": error.label,
                "error_type": error.error_type
            })
        
        # Errores del LLM
        if error.decision_source == "llm_only":
            patterns["llm_errors"][error.label].append({
                "text": text,
                "error_type": error.error_type
            })
    
    # Calcular concentración por etiqueta
    for label, errors in errors_by_label.items():
        fp_count = sum(1 for e in errors if e.error_type == "FP")
        fn_count = sum(1 for e in errors if e.error_type == "FN")
        patterns["label_concentration"][label] = {
            "total_errors": len(errors),
            "fp": fp_count,
            "fn": fn_count,
            "severity": len(errors)  # Para ordenar
        }
    
    # Ordenar concentración por severidad
    patterns["label_concentration"] = dict(
        sorted(
            patterns["label_concentration"].items(),
            key=lambda x: x[1]["severity"],
            reverse=True
        )
    )
    
    return patterns


# ============================================================================
# GENERACIÓN DE PROPUESTAS DE MEJORA
# ============================================================================

def generate_recommendations(
    global_metrics: LabelMetrics,
    by_label: Dict[str, LabelMetrics],
    patterns: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Genera propuestas de mejora basadas en el análisis.
    
    Returns:
        Lista de recomendaciones ordenadas por impacto esperado
    """
    print(f"\n[INFO] Generando recomendaciones...")
    
    recommendations = []
    
    # 1. Recomendación para términos de un solo carácter
    single_chars = patterns.get("single_char", [])
    if single_chars:
        unique_chars = set(e["text"] for e in single_chars)
        recommendations.append({
            "id": 1,
            "priority": "ALTA",
            "category": "Filtro Determinista",
            "title": "Filtrar entidades de un solo carácter",
            "problem": f"Se detectaron {len(single_chars)} errores con términos de un solo carácter: {unique_chars}",
            "solution": "Añadir regla en EntityFastFilter para ignorar entidades de longitud 1 (excepto si son parte de un identificador válido)",
            "impact": "Alto - Reduce FP significativamente sin impacto en recall",
            "implementation": """
```python
# En EntityFastFilter.evaluate_candidate():
if len(entity_text.strip()) <= 1:
    return EnumDecision.FORCE_IGNORE
```
""",
            "affected_labels": list(set(e["label"] for e in single_chars))
        })
    
    # 2. Recomendación para términos muy cortos
    short_terms = patterns.get("short_terms", [])
    short_non_single = [t for t in short_terms if len(t["text"]) > 1]
    if short_non_single:
        recommendations.append({
            "id": 2,
            "priority": "ALTA",
            "category": "Filtro Determinista",
            "title": "Revisar umbral de longitud mínima",
            "problem": f"Se detectaron {len(short_non_single)} errores con términos de 2-3 caracteres",
            "solution": "Implementar longitud mínima configurable por etiqueta",
            "impact": "Medio-Alto - Reduce FP en entidades ambiguas",
            "implementation": """
```python
# Umbrales por etiqueta
MIN_LENGTH = {
    "NUMERO_IDENTIF": 3,
    "FECHAS": 4,
    "NUMERO_TELEFONO": 6,
    # ...
}
```
""",
            "affected_labels": list(set(e["label"] for e in short_non_single))
        })
    
    # 3. Recomendación para términos numéricos aislados
    numeric_only = patterns.get("numeric_only", [])
    if numeric_only:
        recommendations.append({
            "id": 3,
            "priority": "MEDIA",
            "category": "Filtro Determinista",
            "title": "Validar formato de identificadores numéricos",
            "problem": f"Se detectaron {len(numeric_only)} errores con términos puramente numéricos",
            "solution": "Añadir validación de formato para identificadores (regex patterns)",
            "impact": "Medio - Mejora precisión en NUMERO_IDENTIF",
            "implementation": """
```python
# Patrones válidos de identificadores
VALID_ID_PATTERNS = [
    r'^[0-9]{8}[A-Z]$',  # DNI
    r'^[A-Z][0-9]{7}[A-Z0-9]$',  # NIE
    r'^[0-9]{9,}$',  # Teléfono
]
```
""",
            "affected_labels": list(set(e["label"] for e in numeric_only))
        })
    
    # 4. Recomendaciones específicas por etiqueta problemática
    for label, stats in patterns.get("label_concentration", {}).items():
        if stats["total_errors"] >= 3:  # Umbral mínimo de errores para recomendar
            label_metrics = by_label.get(label)
            
            rec = {
                "id": len(recommendations) + 1,
                "priority": "MEDIA" if stats["total_errors"] < 10 else "ALTA",
                "category": f"Etiqueta: {label}",
                "title": f"Mejorar detección de {label}",
                "problem": f"FP: {stats['fp']}, FN: {stats['fn']} (Total: {stats['total_errors']})",
                "affected_labels": [label]
            }
            
            # Recomendación según tipo de error predominante
            if stats["fp"] > stats["fn"]:
                rec["solution"] = "Reducir FP: Añadir términos a blacklist o endurecer criterios del LLM"
                rec["impact"] = "Alto en precisión"
            else:
                rec["solution"] = "Reducir FN: Revisar whitelist o suavizar criterios del LLM"
                rec["impact"] = "Alto en recall"
            
            recommendations.append(rec)
    
    # 5. Recomendación para errores del filtro
    filter_errors = patterns.get("filter_errors", {})
    for list_type, errors in filter_errors.items():
        if errors:
            recommendations.append({
                "id": len(recommendations) + 1,
                "priority": "MEDIA",
                "category": "Listas",
                "title": f"Revisar lista: {list_type}",
                "problem": f"{len(errors)} errores originados por la lista {list_type}",
                "solution": f"Revisar y depurar términos en {list_type}",
                "impact": "Medio",
                "examples": errors[:5],  # Primeros 5 ejemplos
                "affected_labels": list(set(e["label"] for e in errors))
            })
    
    # Ordenar por prioridad
    priority_order = {"ALTA": 0, "MEDIA": 1, "BAJA": 2}
    recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "BAJA"), 3))
    
    return recommendations


# ============================================================================
# GENERACIÓN DEL INFORME
# ============================================================================

def generate_markdown_report(
    records: List[EntityRecord],
    global_metrics: LabelMetrics,
    by_label: Dict[str, LabelMetrics],
    errors_by_label: Dict[str, List[ErrorExample]],
    patterns: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    output_path: Path
):
    """Genera el informe completo en formato Markdown."""
    
    print(f"\n[INFO] Generando informe Markdown...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Estadísticas básicas
    total_records = len(records)
    with_gt = sum(1 for r in records if r.ground_truth is not None)
    filter_decided = sum(1 for r in records if r.filter_decision in ["FORCE_ANONYMIZE", "FORCE_IGNORE"])
    escalated = sum(1 for r in records if r.filter_decision == "ESCALATE_TO_LLM")
    
    report = f"""# Análisis Completo del Pipeline de Validación de Entidades

**Fecha de generación:** {timestamp}

---

## 1. Resumen Ejecutivo

Este informe presenta un análisis exhaustivo del pipeline de validación de entidades,
incluyendo métricas globales, análisis por etiqueta, patrones de error y recomendaciones de mejora.

### 1.1 Estadísticas del Dataset

| Métrica | Valor |
|---------|-------|
| Total entidades procesadas | {total_records} |
| Entidades con ground truth | {with_gt} |
| Resueltas por filtro | {filter_decided} ({filter_decided/total_records*100:.1f}%) |
| Escaladas a LLM | {escalated} ({escalated/total_records*100:.1f}%) |

---

## 2. Métricas Globales del Pipeline

| Métrica | Valor |
|---------|-------|
| **True Positives (TP)** | {global_metrics.tp} |
| **False Positives (FP)** | {global_metrics.fp} |
| **False Negatives (FN)** | {global_metrics.fn} |
| **True Negatives (TN)** | {global_metrics.tn} |
| **Precision** | {global_metrics.precision:.4f} ({global_metrics.precision*100:.2f}%) |
| **Recall** | {global_metrics.recall:.4f} ({global_metrics.recall*100:.2f}%) |
| **F1 Score** | {global_metrics.f1:.4f} ({global_metrics.f1*100:.2f}%) |

### 2.1 Desglose por Origen de Decisión

| Origen | Correctas | Incorrectas | Accuracy |
|--------|-----------|-------------|----------|
| Filtro Determinista | {global_metrics.filter_correct} | {global_metrics.filter_incorrect} | {(global_metrics.filter_correct/(global_metrics.filter_correct+global_metrics.filter_incorrect)*100) if (global_metrics.filter_correct+global_metrics.filter_incorrect) > 0 else 0:.1f}% |
| LLM Judge | {global_metrics.llm_correct} | {global_metrics.llm_incorrect} | {(global_metrics.llm_correct/(global_metrics.llm_correct+global_metrics.llm_incorrect)*100) if (global_metrics.llm_correct+global_metrics.llm_incorrect) > 0 else 0:.1f}% |

---

## 3. Métricas por Etiqueta

"""

    # Tabla de métricas por etiqueta ordenada por F1
    sorted_labels = sorted(by_label.items(), key=lambda x: x[1].f1, reverse=True)
    
    report += """| Etiqueta | TP | FP | FN | TN | Precision | Recall | F1 |
|----------|----|----|----|----|-----------|--------|-----|
"""
    
    for label, metrics in sorted_labels:
        report += f"| {label} | {metrics.tp} | {metrics.fp} | {metrics.fn} | {metrics.tn} | {metrics.precision:.3f} | {metrics.recall:.3f} | {metrics.f1:.3f} |\n"
    
    report += """
---

## 4. Etiquetas Problemáticas (Ordenadas por Gravedad)

"""
    
    # Lista de etiquetas con más errores
    label_errors = patterns.get("label_concentration", {})
    if label_errors:
        report += """| Rank | Etiqueta | Total Errores | FP | FN | Ratio FP/FN |
|------|----------|---------------|----|----|-------------|
"""
        for rank, (label, stats) in enumerate(label_errors.items(), 1):
            if stats["total_errors"] > 0:
                ratio = stats["fp"] / stats["fn"] if stats["fn"] > 0 else float('inf')
                ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
                report += f"| {rank} | {label} | {stats['total_errors']} | {stats['fp']} | {stats['fn']} | {ratio_str} |\n"
    else:
        report += "*No se detectaron errores significativos.*\n"
    
    report += """
---

## 5. Ejemplos Representativos de Errores

"""
    
    # Ejemplos por etiqueta problemática
    for label, errors in sorted(errors_by_label.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        if not errors:
            continue
        
        report += f"""### 5.{list(errors_by_label.keys()).index(label)+1}. {label}

"""
        
        # Separar FP y FN
        fp_examples = [e for e in errors if e.error_type == "FP"][:5]
        fn_examples = [e for e in errors if e.error_type == "FN"][:5]
        
        if fp_examples:
            report += """#### Falsos Positivos (Sistema dice anonimizar, pero NO debería)

| Texto | Doc ID | Decisión | Lista | Confianza |
|-------|--------|----------|-------|-----------|
"""
            for ex in fp_examples:
                report += f"| `{ex.entity_text}` | {ex.document_id[:20]}... | {ex.decision_source} | {ex.matched_list} | {ex.confidence:.2f} |\n"
        
        if fn_examples:
            report += """
#### Falsos Negativos (Sistema NO anonimiza, pero debería)

| Texto | Doc ID | Decisión | Lista | Confianza |
|-------|--------|----------|-------|-----------|
"""
            for ex in fn_examples:
                report += f"| `{ex.entity_text}` | {ex.document_id[:20]}... | {ex.decision_source} | {ex.matched_list} | {ex.confidence:.2f} |\n"
        
        report += "\n"
    
    report += """---

## 6. Patrones Comunes Detectados en los Errores

"""
    
    # Patrón: términos de un solo carácter
    single_chars = patterns.get("single_char", [])
    if single_chars:
        unique_chars = list(set(e["text"] for e in single_chars))
        report += f"""### 6.1 Términos de un solo carácter
- **Cantidad:** {len(single_chars)} errores
- **Caracteres problemáticos:** `{', '.join(unique_chars)}`
- **Impacto:** Alto en precisión
- **Causa probable:** El NER detecta fragmentos de tokens más largos

"""
    
    # Patrón: términos cortos
    short_terms = patterns.get("short_terms", [])
    if short_terms:
        report += f"""### 6.2 Términos muy cortos (2-3 caracteres)
- **Cantidad:** {len(short_terms)} errores
- **Ejemplos:** `{', '.join(set(e['text'] for e in short_terms[:10]))}`
- **Etiquetas afectadas:** {', '.join(set(e['label'] for e in short_terms))}

"""
    
    # Patrón: numéricos
    numeric_only = patterns.get("numeric_only", [])
    if numeric_only:
        report += f"""### 6.3 Términos puramente numéricos
- **Cantidad:** {len(numeric_only)} errores
- **Ejemplos:** `{', '.join(set(e['text'] for e in numeric_only[:10]))}`
- **Problema:** Números aislados sin contexto de identificador

"""
    
    # Patrón: abreviaturas
    abbreviations = patterns.get("abbreviations", [])
    if abbreviations:
        report += f"""### 6.4 Abreviaturas (2-4 mayúsculas)
- **Cantidad:** {len(abbreviations)} errores
- **Ejemplos:** `{', '.join(set(e['text'] for e in abbreviations[:10]))}`
- **Problema:** Códigos médicos confundidos con identificadores

"""
    
    report += """---

## 7. Propuestas de Mejora (Ordenadas por Impacto)

"""
    
    for rec in recommendations:
        report += f"""### 7.{rec['id']}. [{rec['priority']}] {rec['title']}

**Categoría:** {rec['category']}

**Problema detectado:**
{rec['problem']}

**Solución propuesta:**
{rec['solution']}

**Impacto esperado:** {rec.get('impact', 'No especificado')}

**Etiquetas afectadas:** {', '.join(rec.get('affected_labels', []))}

"""
        if 'implementation' in rec:
            report += f"""**Implementación sugerida:**
{rec['implementation']}
"""
        
        report += "\n---\n\n"
    
    report += """## 8. Conclusiones y Próximos Pasos

### 8.1 Hallazgos Principales

"""
    
    # Generar conclusiones basadas en los datos
    if global_metrics.precision < 0.9:
        report += f"- ⚠️ **Precisión por debajo del 90%** ({global_metrics.precision*100:.1f}%): Hay demasiados falsos positivos.\n"
    else:
        report += f"- ✅ **Precisión aceptable** ({global_metrics.precision*100:.1f}%)\n"
    
    if global_metrics.recall < 0.95:
        report += f"- ⚠️ **Recall por debajo del 95%** ({global_metrics.recall*100:.1f}%): Se están escapando entidades sensibles.\n"
    else:
        report += f"- ✅ **Recall alto** ({global_metrics.recall*100:.1f}%): Buena cobertura de entidades sensibles.\n"
    
    total_errors = sum(len(e) for e in errors_by_label.values())
    report += f"- **Total de errores detectados:** {total_errors}\n"
    
    # Top 3 etiquetas problemáticas
    top_problems = list(label_errors.items())[:3]
    if top_problems:
        report += f"- **Etiquetas más problemáticas:** {', '.join(l for l, _ in top_problems)}\n"
    
    report += """
### 8.2 Acciones Recomendadas (Prioridad)

1. **Inmediato:** Implementar filtro de longitud mínima (1-2 caracteres)
2. **Corto plazo:** Revisar y depurar listas de whitelist/blacklist
3. **Medio plazo:** Ajustar prompts del LLM Judge para etiquetas problemáticas
4. **Continuo:** Monitorizar métricas y actualizar reglas según nuevos patrones

---

## 9. Anexo: Datos Técnicos

### 9.1 Configuración del Pipeline

- **Filtro determinista:** EntityFastFilter con Aho-Corasick (flashtext)
- **Listas cargadas:** hospitales.json, lugares.json, medicamentos.json, patologias.json, cie10.xls
- **LLM Judge:** Modelo de validación semántica

### 9.2 Reproducción del Análisis

```bash
cd src/pipeline-nuevos-textos
python full_pipeline_analysis.py
```

---

*Informe generado automáticamente por el pipeline de análisis.*
"""
    
    # Guardar informe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"[INFO] Informe guardado en: {output_path}")


def save_json_results(
    records: List[EntityRecord],
    global_metrics: LabelMetrics,
    by_label: Dict[str, LabelMetrics],
    errors_by_label: Dict[str, List[ErrorExample]],
    patterns: Dict[str, Any],
    output_path: Path
):
    """Guarda los resultados en formato JSON para análisis programático."""
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_records": len(records),
            "with_ground_truth": sum(1 for r in records if r.ground_truth is not None),
            "filter_decided": sum(1 for r in records if r.filter_decision in ["FORCE_ANONYMIZE", "FORCE_IGNORE"]),
            "escalated_to_llm": sum(1 for r in records if r.filter_decision == "ESCALATE_TO_LLM"),
        },
        "global_metrics": {
            "tp": global_metrics.tp,
            "fp": global_metrics.fp,
            "fn": global_metrics.fn,
            "tn": global_metrics.tn,
            "precision": global_metrics.precision,
            "recall": global_metrics.recall,
            "f1": global_metrics.f1,
            "filter_correct": global_metrics.filter_correct,
            "filter_incorrect": global_metrics.filter_incorrect,
            "llm_correct": global_metrics.llm_correct,
            "llm_incorrect": global_metrics.llm_incorrect,
        },
        "metrics_by_label": {
            label: {
                "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
                "precision": m.precision, "recall": m.recall, "f1": m.f1
            }
            for label, m in by_label.items()
        },
        "error_counts_by_label": {
            label: {
                "total": len(errors),
                "fp": sum(1 for e in errors if e.error_type == "FP"),
                "fn": sum(1 for e in errors if e.error_type == "FN"),
            }
            for label, errors in errors_by_label.items()
        },
        "patterns_summary": {
            "single_char_errors": len(patterns.get("single_char", [])),
            "short_term_errors": len(patterns.get("short_terms", [])),
            "numeric_only_errors": len(patterns.get("numeric_only", [])),
            "abbreviation_errors": len(patterns.get("abbreviations", [])),
        },
        "detailed_records": [
            {
                "document_id": r.document_id,
                "entity_text": r.entity_text,
                "label": r.label,
                "filter_decision": r.filter_decision,
                "llm_decision": r.llm_decision,
                "system_decision": r.system_decision,
                "ground_truth": r.ground_truth,
                "matched_list": r.matched_list,
                "confidence": r.confidence,
            }
            for r in records if r.ground_truth is not None
        ]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Resultados JSON guardados en: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Función principal del análisis."""
    # Parsear argumentos CLI
    parser = argparse.ArgumentParser(description="Análisis completo del pipeline de validación de entidades.")
    parser.add_argument('--whitelist', '-w', nargs='+', help='Rutas a ficheros JSON de whitelist (exact match).')
    parser.add_argument('--blacklist', '-b', nargs='+', help='Rutas a ficheros JSON de blacklist (exact match).')
    parser.add_argument('--cie10', '-c', help='Ruta al fichero CIE10 (xls/tsv)')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("ANÁLISIS COMPLETO DEL PIPELINE DE VALIDACIÓN")
    print("="*70 + "\n")
    
    try:
        # 1. Cargar datos
        print("[PASO 1] Cargando datos...")
        detections = load_detections()
        ground_truth = load_ground_truth_aws2()
        test_results = load_test_results()
        
        # 2. Inicializar filtro
        print("\n[PASO 2] Inicializando filtro...")
        entity_filter = initialize_filter(
            whitelist_paths_arg=args.whitelist,
            blacklist_paths_arg=args.blacklist,
            cie10_path_arg=args.cie10
        )
        
        # 3. Procesar pipeline
        print("\n[PASO 3] Procesando pipeline...")
        records = process_pipeline(detections, ground_truth, entity_filter, test_results)
        
        # 4. Calcular métricas
        print("\n[PASO 4] Calculando métricas...")
        global_metrics, by_label = calculate_metrics(records)
        
        # 5. Analizar errores
        print("\n[PASO 5] Analizando errores...")
        errors_by_label = analyze_errors(records)
        
        # 6. Detectar patrones
        print("\n[PASO 6] Detectando patrones...")
        patterns = detect_error_patterns(records, errors_by_label)
        
        # 7. Generar recomendaciones
        print("\n[PASO 7] Generando recomendaciones...")
        recommendations = generate_recommendations(global_metrics, by_label, patterns)
        
        # 8. Generar informe
        print("\n[PASO 8] Generando informe...")
        markdown_path = REPORTS_DIR / "full_pipeline_analysis.md"
        json_path = OUTPUTS_DIR / "full_pipeline_analysis.json"
        
        generate_markdown_report(
            records, global_metrics, by_label,
            errors_by_label, patterns, recommendations,
            markdown_path
        )
        
        save_json_results(
            records, global_metrics, by_label,
            errors_by_label, patterns,
            json_path
        )
        
        # Resumen final
        print("\n" + "="*70)
        print("RESUMEN FINAL")
        print("="*70)
        print(f"\n Métricas Globales:")
        print(f"   - Precision: {global_metrics.precision:.4f} ({global_metrics.precision*100:.2f}%)")
        print(f"   - Recall:    {global_metrics.recall:.4f} ({global_metrics.recall*100:.2f}%)")
        print(f"   - F1 Score:  {global_metrics.f1:.4f} ({global_metrics.f1*100:.2f}%)")
        print(f"\n📁 Archivos generados:")
        print(f"   - Informe Markdown: {markdown_path}")
        print(f"   - Datos JSON: {json_path}")
        print("\n✅ Análisis completado exitosamente")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
