#!/usr/bin/env python3
"""
model_comparison_study.py - Comparativa Head-to-Head de Modelos SetFit
=======================================================================

LÓGICA DE NEGOCIO (CORRECTA):
1. Base (Ensemble): Fusiona Meddocan + Carmen, deduplica por coordenadas
2. Filtrado SetFit:
   - label=1 (PII) -> La entidad SOBREVIVE
   - label=0 (RUIDO) -> La entidad MUERE
3. Métricas: Se calculan sobre las entidades que SOBREVIVIERON

COMPARATIVA:
- Base (Sin Filtro): Ensemble puro
- Pipeline A (Base + SetFit A): Después de filtrar con modelo A
- Pipeline B (Base + SetFit B): Después de filtrar con modelo B

USO:
  python model_comparison_study.py \
    --gold corpus/output/aws3 \
    --meddocan step6_validation_results/aws3/detecciones_detalladas.csv \
    --carmen outputs/carmen.json \
    --setfit-a outputs/resultados_aws3.json \
    --setfit-b salida.json \
    --output-dir comparison_results
"""

import argparse
import json
import re
import sys
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field, asdict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None  # Para evitar errores en type hints
    HAS_PANDAS = False
    print("[WARN] pandas no disponible. Instalar con: pip install pandas")


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class Entity:
    """Entidad candidata del ensemble."""
    doc_id: str
    text: str
    start: int
    end: int
    source: str  # 'meddocan', 'carmen', 'both'
    
    def key(self) -> Tuple[str, int, int]:
        """Clave única para deduplicación (doc, start, end)."""
        return (self.doc_id, self.start, self.end)
    
    def norm_key(self) -> Tuple[str, str]:
        """Clave normalizada para matching (doc, texto_normalizado)."""
        return (self.doc_id, normalize(self.text))


@dataclass
class ModelMetrics:
    """Métricas para una configuración."""
    name: str
    total_entities: int = 0  # Entidades en predicción final
    
    # Métricas básicas
    tp: int = 0  # True Positives (correcto, es PII)
    fp: int = 0  # False Positives (basura que pasó)
    fn: int = 0  # False Negatives (PII que no llegó)
    
    # Métricas derivadas
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # Desglose específico
    fp_basura_restante: int = 0  # Ruido que NO fue filtrado
    fn_fugas_inducidas: int = 0  # PII real que SetFit MATÓ
    fn_no_detectado: int = 0     # PII que el Ensemble no vio
    
    # Estadísticas de filtrado
    entidades_filtradas: int = 0  # Cuántas eliminó SetFit
    tasa_filtrado_pct: float = 0.0
    
    def calculate(self):
        self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0.0


@dataclass
class DeltaAnalysis:
    """Análisis comparativo entre dos modelos."""
    model_a_name: str
    model_b_name: str
    
    # Noise Leakage: Basura que A filtró pero B no (o viceversa)
    noise_a_filtered_b_kept: List[Dict] = field(default_factory=list)
    noise_b_filtered_a_kept: List[Dict] = field(default_factory=list)
    
    # Over-Cleaning: PII real que fue matado
    pii_a_killed: List[Dict] = field(default_factory=list)
    pii_b_killed: List[Dict] = field(default_factory=list)
    
    # Resumen numérico
    winner_precision: str = ""
    winner_recall: str = ""
    winner_f1: str = ""


# =============================================================================
# UTILIDADES
# =============================================================================

GOLD_PATTERN = re.compile(r"\[\*\*(.+?)\*\*\]")


def normalize(text: str) -> str:
    """Normaliza texto para comparación."""
    if not text:
        return ''
    return ' '.join(str(text).lower().strip().split())


def clean_entity(text: str) -> str:
    """Limpia marcadores de una entidad."""
    if not text:
        return ''
    t = str(text)
    t = re.sub(r"\[\*\*(.+?)\*\*\]", r"\1", t)
    t = t.replace('**', '').replace('[', '').replace(']', '').replace('*', '')
    return ' '.join(t.split()).strip()


def norm_doc_id(name: str) -> str:
    """Normaliza ID de documento."""
    base = str(name)
    if base.endswith('.txt.txt'):
        base = base[:-8]
    elif base.endswith('.txt'):
        base = base[:-4]
    return base


# =============================================================================
# FASE 1: CARGA DE DATOS
# =============================================================================

def load_gold_standard(path: Path) -> Dict[str, Set[str]]:
    """
    Carga Gold Standard.
    Soporta carpeta con .txt o archivo JSON.
    """
    result: Dict[str, Set[str]] = defaultdict(set)
    
    # Si es carpeta
    if path.is_dir():
        txt_files = list(path.rglob("*.txt"))
        print(f"  [INFO] Extrayendo Gold Standard de {len(txt_files)} archivos .txt")
        for txt_file in txt_files:
            try:
                text = txt_file.read_text(encoding='utf-8')
            except:
                text = txt_file.read_text(errors='ignore')
            
            doc_id = norm_doc_id(txt_file.name)
            for match in GOLD_PATTERN.finditer(text):
                entity = clean_entity(match.group(1).strip())
                if entity:
                    result[doc_id].add(normalize(entity))
        return dict(result)
    
    # Si es archivo JSON
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        docs = data.get('documents', data)
        for doc_id, entities in docs.items():
            if doc_id in ('generated_at', 'gold_dir', 'metadata'):
                continue
            base = norm_doc_id(doc_id)
            if isinstance(entities, list):
                for e in entities:
                    cleaned = clean_entity(e)
                    if cleaned:
                        result[base].add(normalize(cleaned))
    
    return dict(result)


def load_detector_output(path: Path, source_name: str) -> List[Entity]:
    """
    Carga output de un detector (Meddocan o Carmen).
    Soporta CSV y JSON.
    """
    entities = []
    
    if not path or not Path(path).exists():
        print(f"[WARN] Archivo no encontrado: {path}")
        return entities
    
    path = Path(path)
    
    # CSV
    if path.suffix.lower() == '.csv':
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = norm_doc_id(row.get('doc_id', row.get('document_id', '')))
                text = clean_entity(row.get('texto_detectado', row.get('entity_text', row.get('text', ''))))
                start = int(row.get('posicion_inicio', row.get('start', -1)))
                end = int(row.get('posicion_fin', row.get('end', -1)))
                
                if doc_id and text and start >= 0:
                    entities.append(Entity(doc_id, text, start, end, source_name))
    
    # JSON
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        if isinstance(data, dict) and 'decisions' in data:
            items = data['decisions']
        elif isinstance(data, list):
            items = data
        
        for item in items:
            doc_id = norm_doc_id(item.get('document_id', item.get('doc_id', '')))
            text = clean_entity(item.get('entity_text', item.get('text', '')))
            start = item.get('start', item.get('posicion_inicio', -1))
            end = item.get('end', item.get('posicion_fin', -1))
            
            if doc_id and text:
                entities.append(Entity(doc_id, text, start, end, source_name))
    
    return entities


def create_ensemble(meddocan: List[Entity], carmen: List[Entity]) -> List[Entity]:
    """
    Fusiona Meddocan + Carmen y deduplica por coordenadas (doc, start, end).
    Si ambos detectan la misma entidad, cuenta como UNA sola.
    """
    seen = {}  # key -> Entity
    
    for entity in meddocan + carmen:
        key = entity.key()
        if key in seen:
            # Ya existe, marcar como detectado por ambos
            seen[key].source = 'both'
        else:
            seen[key] = entity
    
    return list(seen.values())


def load_setfit_predictions(path: Path, candidatos_ensemble: List[Entity]) -> Dict[Tuple[str, str], int]:
    """
    Carga predicciones SetFit.
    
    LÓGICA ESPECIAL: Los archivos de pipeline ya tienen filtradas las entidades RUIDO.
    Por tanto:
    - Si una entidad aparece en decisions -> label=1 (PII, sobrevivió)
    - Si una entidad del ensemble NO aparece en decisions -> label=0 (RUIDO, fue filtrada)
    
    Returns:
        {(doc_id, normalized_text): label}
        donde label = 1 (PII, mantener) o 0 (RUIDO, eliminar)
    """
    if not path.exists():
        print(f"[ERROR] Archivo SetFit no encontrado: {path}")
        return {}
    
    result = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Primero, marcar TODAS las entidades del ensemble como RUIDO (label=0)
    for entity in candidatos_ensemble:
        key = entity.norm_key()
        result[key] = 0  # Asumimos que fue filtrado
    
    # Luego, las que aparecen en decisions son PII (label=1)
    items = []
    if isinstance(data, dict) and 'decisions' in data:
        items = data['decisions']
    elif isinstance(data, list):
        items = data
    
    sobrevivientes_count = 0
    for item in items:
        doc_id = norm_doc_id(item.get('document_id', item.get('doc_id', '')))
        text = clean_entity(item.get('entity_text', item.get('text', '')))
        
        if doc_id and text:
            norm_text = normalize(text)
            key = (doc_id, norm_text)
            result[key] = 1  # Sobrevivió = PII
            sobrevivientes_count += 1
    
    filtrados_count = sum(1 for v in result.values() if v == 0)
    print(f"    → SetFit clasificó: {sobrevivientes_count} PII, {filtrados_count} RUIDO")
    
    return result


# =============================================================================
# FASE 2: APLICAR FILTRO SETFIT
# =============================================================================

def apply_setfit_filter(
    candidatos: List[Entity],
    setfit_preds: Dict[Tuple[str, str], int]
) -> Tuple[List[Entity], List[Entity]]:
    """
    Aplica filtro SetFit sobre candidatos del Ensemble.
    
    REGLA DE SUPERVIVENCIA:
    - label=1 (PII) -> SOBREVIVE
    - label=0 (RUIDO) -> MUERE
    
    Returns:
        (sobrevivientes, filtrados)
    """
    sobrevivientes = []
    filtrados = []
    
    for entity in candidatos:
        key = entity.norm_key()
        label = setfit_preds.get(key, 1)  # Default: conservador (mantener)
        
        if label == 1:
            sobrevivientes.append(entity)
        else:
            filtrados.append(entity)
    
    return sobrevivientes, filtrados


# =============================================================================
# FASE 3: CÁLCULO DE MÉTRICAS
# =============================================================================

def calculate_metrics(
    predicciones: List[Entity],
    candidatos_ensemble: List[Entity],
    filtrados: List[Entity],
    gold: Dict[str, Set[str]],
    name: str
) -> Tuple[ModelMetrics, Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    Calcula métricas para una configuración.
    
    Args:
        predicciones: Entidades en la salida final (sobrevivientes)
        candidatos_ensemble: Todos los candidatos pre-filtro
        filtrados: Entidades que SetFit eliminó
        gold: Gold standard
        name: Nombre de la configuración
    
    Returns:
        (metrics, tp_set, fp_set)
    """
    metrics = ModelMetrics(name=name)
    metrics.total_entities = len(predicciones)
    metrics.entidades_filtradas = len(filtrados)
    
    if len(candidatos_ensemble) > 0:
        metrics.tasa_filtrado_pct = len(filtrados) / len(candidatos_ensemble) * 100
    
    # Construir sets
    pred_por_doc: Dict[str, Set[str]] = defaultdict(set)
    for ent in predicciones:
        pred_por_doc[ent.doc_id].add(normalize(ent.text))
    
    candidatos_por_doc: Dict[str, Set[str]] = defaultdict(set)
    for ent in candidatos_ensemble:
        candidatos_por_doc[ent.doc_id].add(normalize(ent.text))
    
    filtrados_por_doc: Dict[str, Set[str]] = defaultdict(set)
    for ent in filtrados:
        filtrados_por_doc[ent.doc_id].add(normalize(ent.text))
    
    tp_set: Set[Tuple[str, str]] = set()
    fp_set: Set[Tuple[str, str]] = set()
    
    # Evaluar por documento
    all_docs = set(gold.keys()) | set(pred_por_doc.keys())
    
    for doc_id in all_docs:
        gold_ents = gold.get(doc_id, set())
        pred_ents = pred_por_doc.get(doc_id, set())
        candidatos_ents = candidatos_por_doc.get(doc_id, set())
        filtrados_ents = filtrados_por_doc.get(doc_id, set())
        
        # TP: en predicción Y en gold
        for ent in pred_ents:
            if ent in gold_ents:
                metrics.tp += 1
                tp_set.add((doc_id, ent))
        
        # FP: en predicción pero NO en gold (basura que pasó)
        for ent in pred_ents:
            if ent not in gold_ents:
                metrics.fp += 1
                metrics.fp_basura_restante += 1
                fp_set.add((doc_id, ent))
        
        # FN: en gold pero NO en predicción
        for ent in gold_ents:
            if ent not in pred_ents:
                metrics.fn += 1
                
                # ¿Por qué falló?
                if ent not in candidatos_ents:
                    # Nunca fue detectado por el Ensemble
                    metrics.fn_no_detectado += 1
                elif ent in filtrados_ents:
                    # Fue detectado pero SetFit lo mató (FUGA INDUCIDA)
                    metrics.fn_fugas_inducidas += 1
    
    metrics.calculate()
    return metrics, tp_set, fp_set


# =============================================================================
# FASE 4: ANÁLISIS DELTA
# =============================================================================

def analyze_delta(
    metrics_a: ModelMetrics,
    metrics_b: ModelMetrics,
    candidatos: List[Entity],
    sobrevivientes_a: List[Entity],
    sobrevivientes_b: List[Entity],
    filtrados_a: List[Entity],
    filtrados_b: List[Entity],
    gold: Dict[str, Set[str]],
    tp_set_base: Set[Tuple[str, str]],
    fp_set_base: Set[Tuple[str, str]]
) -> DeltaAnalysis:
    """
    Análisis comparativo entre dos modelos SetFit.
    """
    delta = DeltaAnalysis(
        model_a_name=metrics_a.name,
        model_b_name=metrics_b.name
    )
    
    # Construir sets de sobrevivientes
    sobrev_a_set = {(e.doc_id, normalize(e.text)) for e in sobrevivientes_a}
    sobrev_b_set = {(e.doc_id, normalize(e.text)) for e in sobrevivientes_b}
    
    filtrados_a_set = {(e.doc_id, normalize(e.text)) for e in filtrados_a}
    filtrados_b_set = {(e.doc_id, normalize(e.text)) for e in filtrados_b}
    
    # Noise Leakage: Basura que un modelo filtró pero el otro no
    for doc_id, ent in fp_set_base:
        # Basura que A filtró pero B dejó pasar
        if (doc_id, ent) in filtrados_a_set and (doc_id, ent) in sobrev_b_set:
            if len(delta.noise_a_filtered_b_kept) < 20:
                delta.noise_a_filtered_b_kept.append({
                    'doc_id': doc_id,
                    'entity': ent,
                    'nota': f'{metrics_a.name} filtró correctamente, {metrics_b.name} dejó pasar'
                })
        
        # Basura que B filtró pero A dejó pasar
        if (doc_id, ent) in filtrados_b_set and (doc_id, ent) in sobrev_a_set:
            if len(delta.noise_b_filtered_a_kept) < 20:
                delta.noise_b_filtered_a_kept.append({
                    'doc_id': doc_id,
                    'entity': ent,
                    'nota': f'{metrics_b.name} filtró correctamente, {metrics_a.name} dejó pasar'
                })
    
    # Over-Cleaning: PII real que el Ensemble detectó pero SetFit mató
    for doc_id, ent in tp_set_base:
        # PII que A mató
        if (doc_id, ent) in filtrados_a_set:
            if len(delta.pii_a_killed) < 20:
                delta.pii_a_killed.append({
                    'doc_id': doc_id,
                    'entity': ent,
                    'nota': f'{metrics_a.name} mató PII real (CRÍTICO)'
                })
        
        # PII que B mató
        if (doc_id, ent) in filtrados_b_set:
            if len(delta.pii_b_killed) < 20:
                delta.pii_b_killed.append({
                    'doc_id': doc_id,
                    'entity': ent,
                    'nota': f'{metrics_b.name} mató PII real (CRÍTICO)'
                })
    
    # Determinar ganadores
    delta.winner_precision = metrics_a.name if metrics_a.precision > metrics_b.precision else metrics_b.name
    delta.winner_recall = metrics_a.name if metrics_a.recall > metrics_b.recall else metrics_b.name
    delta.winner_f1 = metrics_a.name if metrics_a.f1 > metrics_b.f1 else metrics_b.name
    
    return delta


# =============================================================================
# REPORTES
# =============================================================================

def generate_comparison_table(
    base_metrics: ModelMetrics,
    metrics_a: ModelMetrics,
    metrics_b: ModelMetrics
) -> str:
    """Genera tabla comparativa en formato texto."""
    
    lines = []
    lines.append("\n" + "=" * 120)
    lines.append("TABLA COMPARATIVA HEAD-TO-HEAD")
    lines.append("=" * 120)
    lines.append("")
    
    header = (
        f"{'Configuración':<35} "
        f"{'Entidades':>10} "
        f"{'TP':>8} "
        f"{'FP':>8} "
        f"{'FN':>8} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>8} "
        f"{'FP Basura':>10} "
        f"{'FN Fugas':>10}"
    )
    lines.append(header)
    lines.append("-" * 120)
    
    for m in [base_metrics, metrics_a, metrics_b]:
        row = (
            f"{m.name:<35} "
            f"{m.total_entities:>10} "
            f"{m.tp:>8} "
            f"{m.fp:>8} "
            f"{m.fn:>8} "
            f"{m.precision:>9.2%} "
            f"{m.recall:>9.2%} "
            f"{m.f1:>8.4f} "
            f"{m.fp_basura_restante:>10} "
            f"{m.fn_fugas_inducidas:>10}"
        )
        lines.append(row)
    
    lines.append("=" * 120)
    lines.append("")
    lines.append("LEYENDA:")
    lines.append("  • FP Basura: Ruido que NO fue filtrado (queremos BAJO)")
    lines.append("  • FN Fugas: PII real que SetFit MATÓ por error (CRÍTICO - queremos 0)")
    lines.append("  • Precision: De lo que sobrevivió, ¿cuánto es PII real? (queremos ALTO)")
    lines.append("  • Recall: Del PII real total, ¿cuánto sobrevivió? (queremos que NO baje)")
    lines.append("")
    
    return '\n'.join(lines)


def generate_pandas_table(
    base_metrics: ModelMetrics,
    metrics_a: ModelMetrics,
    metrics_b: ModelMetrics
) -> Optional[Any]:
    """Genera DataFrame de pandas con la comparativa."""
    if not HAS_PANDAS:
        return None
    
    rows = []
    for m in [base_metrics, metrics_a, metrics_b]:
        rows.append({
            'Configuración': m.name,
            'Entidades Totales': m.total_entities,
            'TP': m.tp,
            'FP': m.fp,
            'FN': m.fn,
            'Precision': f"{m.precision:.2%}",
            'Recall': f"{m.recall:.2%}",
            'F1-Score': f"{m.f1:.4f}",
            'FP Basura Restante': m.fp_basura_restante,
            'FN Fugas Inducidas': m.fn_fugas_inducidas,
            'FN No Detectado': m.fn_no_detectado,
            'Entidades Filtradas': m.entidades_filtradas,
            'Tasa Filtrado %': f"{m.tasa_filtrado_pct:.1f}%"
        })
    
    return pd.DataFrame(rows)


def generate_delta_report(delta: DeltaAnalysis, output_dir: Path):
    """Genera reporte de análisis delta."""
    
    md = []
    md.append("# Análisis Delta - Comparativa de Modelos SetFit\n\n")
    md.append(f"**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Ganadores por métrica
    md.append("## 🏆 Ganadores por Métrica\n\n")
    md.append(f"- **Precision (Limpieza):** {delta.winner_precision}\n")
    md.append(f"- **Recall (Seguridad):** {delta.winner_recall}\n")
    md.append(f"- **F1-Score (Balance):** {delta.winner_f1}\n\n")
    
    # Noise Leakage
    md.append("## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)\n\n")
    
    md.append(f"### {delta.model_a_name} filtró, {delta.model_b_name} dejó pasar\n")
    md.append(f"**Total:** {len(delta.noise_a_filtered_b_kept)} ejemplos\n\n")
    if delta.noise_a_filtered_b_kept:
        md.append("| # | Entidad | Documento |\n")
        md.append("|---|---------|----------|\n")
        for i, ex in enumerate(delta.noise_a_filtered_b_kept[:10], 1):
            md.append(f"| {i} | `{ex['entity']}` | {ex['doc_id']} |\n")
    else:
        md.append("*No hay ejemplos*\n")
    md.append("\n")
    
    md.append(f"### {delta.model_b_name} filtró, {delta.model_a_name} dejó pasar\n")
    md.append(f"**Total:** {len(delta.noise_b_filtered_a_kept)} ejemplos\n\n")
    if delta.noise_b_filtered_a_kept:
        md.append("| # | Entidad | Documento |\n")
        md.append("|---|---------|----------|\n")
        for i, ex in enumerate(delta.noise_b_filtered_a_kept[:10], 1):
            md.append(f"| {i} | `{ex['entity']}` | {ex['doc_id']} |\n")
    else:
        md.append("*No hay ejemplos*\n")
    md.append("\n")
    
    # Over-Cleaning
    md.append("## ❌ Over-Cleaning (PII real que fue matado por error)\n\n")
    
    md.append(f"### {delta.model_a_name} - PII Real Eliminado (CRÍTICO)\n")
    md.append(f"**Total Fugas:** {len(delta.pii_a_killed)}\n\n")
    if delta.pii_a_killed:
        md.append("| # | Entidad | Documento |\n")
        md.append("|---|---------|----------|\n")
        for i, ex in enumerate(delta.pii_a_killed[:10], 1):
            md.append(f"| {i} | `{ex['entity']}` | {ex['doc_id']} |\n")
    else:
        md.append("*No hay fugas* ✅\n")
    md.append("\n")
    
    md.append(f"### {delta.model_b_name} - PII Real Eliminado (CRÍTICO)\n")
    md.append(f"**Total Fugas:** {len(delta.pii_b_killed)}\n\n")
    if delta.pii_b_killed:
        md.append("| # | Entidad | Documento |\n")
        md.append("|---|---------|----------|\n")
        for i, ex in enumerate(delta.pii_b_killed[:10], 1):
            md.append(f"| {i} | `{ex['entity']}` | {ex['doc_id']} |\n")
    else:
        md.append("*No hay fugas* ✅\n")
    md.append("\n")
    
    # Recomendación
    md.append("## 📊 Recomendación\n\n")
    md.append("**Criterios de Decisión:**\n")
    md.append("1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)\n")
    md.append("2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible\n")
    md.append("3. **Balance (F1):** Equilibrio entre ambos\n\n")
    
    # Guardar
    report_path = output_dir / 'delta_analysis.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(''.join(md))
    
    print(f"  ✓ Análisis delta: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comparativa Head-to-Head de modelos SetFit',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--gold', '-g', required=True,
                        help='Gold Standard (carpeta .txt o JSON)')
    parser.add_argument('--meddocan', '-m', required=True,
                        help='Output de Meddocan (CSV o JSON)')
    parser.add_argument('--carmen', '-c',
                        help='Output de Carmen (CSV o JSON, opcional)')
    parser.add_argument('--setfit-a', '-a', required=True,
                        help='Predicciones SetFit modelo A')
    parser.add_argument('--setfit-b', '-b', required=True,
                        help='Predicciones SetFit modelo B')
    parser.add_argument('--output-dir', '-o', default='comparison_results',
                        help='Directorio de salida')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MODEL COMPARISON STUDY - SetFit A vs SetFit B")
    print("=" * 80)
    
    # FASE 1: Cargar datos
    print("\n[1/5] Cargando datos...")
    gold = load_gold_standard(Path(args.gold))
    print(f"  ✓ Gold Standard: {len(gold)} docs, {sum(len(v) for v in gold.values())} entidades")
    
    meddocan_entities = load_detector_output(Path(args.meddocan), 'meddocan')
    print(f"  ✓ Meddocan: {len(meddocan_entities)} detecciones")
    
    carmen_entities = []
    if args.carmen:
        carmen_entities = load_detector_output(Path(args.carmen), 'carmen')
        print(f"  ✓ Carmen: {len(carmen_entities)} detecciones")
    
    # Crear ensemble
    candidatos = create_ensemble(meddocan_entities, carmen_entities)
    print(f"  ✓ Ensemble: {len(candidatos)} candidatos únicos")
    
    # Cargar predicciones SetFit (necesitan el ensemble para deducir qué fue filtrado)
    setfit_a_preds = load_setfit_predictions(Path(args.setfit_a), candidatos)
    setfit_b_preds = load_setfit_predictions(Path(args.setfit_b), candidatos)
    
    # FASE 2: Calcular métricas BASE (sin filtro)
    print("\n[2/5] Calculando métricas BASE (Ensemble sin filtrar)...")
    base_metrics, tp_set_base, fp_set_base = calculate_metrics(
        predicciones=candidatos,
        candidatos_ensemble=candidatos,
        filtrados=[],
        gold=gold,
        name="Base (Sin Filtro)"
    )
    print(f"  ✓ Base: P={base_metrics.precision:.2%}, R={base_metrics.recall:.2%}, F1={base_metrics.f1:.4f}")
    
    # FASE 3: Aplicar filtros SetFit
    print("\n[3/5] Aplicando filtros SetFit...")
    sobrevivientes_a, filtrados_a = apply_setfit_filter(candidatos, setfit_a_preds)
    sobrevivientes_b, filtrados_b = apply_setfit_filter(candidatos, setfit_b_preds)
    
    print(f"  ✓ SetFit A: {len(sobrevivientes_a)} sobreviven, {len(filtrados_a)} filtrados ({len(filtrados_a)/len(candidatos)*100:.1f}%)")
    print(f"  ✓ SetFit B: {len(sobrevivientes_b)} sobreviven, {len(filtrados_b)} filtrados ({len(filtrados_b)/len(candidatos)*100:.1f}%)")
    
    # Calcular métricas filtradas
    metrics_a, _, _ = calculate_metrics(
        predicciones=sobrevivientes_a,
        candidatos_ensemble=candidatos,
        filtrados=filtrados_a,
        gold=gold,
        name="Pipeline A (Base + SetFit A)"
    )
    print(f"  ✓ Pipeline A: P={metrics_a.precision:.2%}, R={metrics_a.recall:.2%}, F1={metrics_a.f1:.4f}, Fugas={metrics_a.fn_fugas_inducidas}")
    
    metrics_b, _, _ = calculate_metrics(
        predicciones=sobrevivientes_b,
        candidatos_ensemble=candidatos,
        filtrados=filtrados_b,
        gold=gold,
        name="Pipeline B (Base + SetFit B)"
    )
    print(f"  ✓ Pipeline B: P={metrics_b.precision:.2%}, R={metrics_b.recall:.2%}, F1={metrics_b.f1:.4f}, Fugas={metrics_b.fn_fugas_inducidas}")
    
    # FASE 4: Análisis delta
    print("\n[4/5] Análisis comparativo (delta)...")
    delta = analyze_delta(
        metrics_a, metrics_b,
        candidatos,
        sobrevivientes_a, sobrevivientes_b,
        filtrados_a, filtrados_b,
        gold, tp_set_base, fp_set_base
    )
    print(f"  ✓ Noise Leakage (A filtró/B no): {len(delta.noise_a_filtered_b_kept)}")
    print(f"  ✓ Noise Leakage (B filtró/A no): {len(delta.noise_b_filtered_a_kept)}")
    print(f"  ✓ Over-Cleaning A: {len(delta.pii_a_killed)}")
    print(f"  ✓ Over-Cleaning B: {len(delta.pii_b_killed)}")
    
    # FASE 5: Generar reportes
    print("\n[5/5] Generando reportes...")
    
    # Tabla en consola
    table = generate_comparison_table(base_metrics, metrics_a, metrics_b)
    print(table)
    
    # Pandas DataFrame
    if HAS_PANDAS:
        df = generate_pandas_table(base_metrics, metrics_a, metrics_b)
        if df is not None:
            csv_path = output_dir / 'comparison_table.csv'
            df.to_csv(csv_path, index=False)
            print(f"  ✓ CSV guardado: {csv_path}")
    
    # Análisis delta
    generate_delta_report(delta, output_dir)
    
    # Guardar métricas JSON
    metrics_json = {
        'generated_at': datetime.now().isoformat(),
        'base': asdict(base_metrics),
        'pipeline_a': asdict(metrics_a),
        'pipeline_b': asdict(metrics_b),
        'delta': {
            'winner_precision': delta.winner_precision,
            'winner_recall': delta.winner_recall,
            'winner_f1': delta.winner_f1,
            'noise_leakage': {
                'a_filtered_b_kept': len(delta.noise_a_filtered_b_kept),
                'b_filtered_a_kept': len(delta.noise_b_filtered_a_kept)
            },
            'over_cleaning': {
                'a_killed_pii': len(delta.pii_a_killed),
                'b_killed_pii': len(delta.pii_b_killed)
            }
        }
    }
    
    json_path = output_dir / 'comparison_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Métricas JSON: {json_path}")
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO")
    print(f"📁 Resultados en: {output_dir}")
    print("=" * 80)
    
    # Recomendación final
    print("\n🎯 RECOMENDACIÓN:")
    if metrics_a.fn_fugas_inducidas == 0 and metrics_b.fn_fugas_inducidas > 0:
        print(f"  → Usar {metrics_a.name} (NO mata PII real)")
    elif metrics_b.fn_fugas_inducidas == 0 and metrics_a.fn_fugas_inducidas > 0:
        print(f"  → Usar {metrics_b.name} (NO mata PII real)")
    elif metrics_a.fn_fugas_inducidas == 0 and metrics_b.fn_fugas_inducidas == 0:
        if metrics_a.precision > metrics_b.precision:
            print(f"  → Usar {metrics_a.name} (Mayor precision, sin fugas)")
        else:
            print(f"  → Usar {metrics_b.name} (Mayor precision, sin fugas)")
    else:
        print(f"  ⚠️ AMBOS modelos matan PII real - requiere re-entrenamiento")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
