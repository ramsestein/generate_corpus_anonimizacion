#!/usr/bin/env python3
"""
CÁLCULO DE MÉTRICAS PARA EVALUACIÓN DE LLM JUDGE
=================================================

Script independiente para calcular métricas de evaluación comparando
las predicciones del LLM contra la verdad terreno (ground truth).

NO LLAMA A NINGÚN LLM. Solo lee JSONs y calcula métricas.

ENTRADA:
--------
1. JSON con resultados de predicciones del LLM (--predictions):
   - Lista de objetos con: document_id, keyword/entity, label, is_valid/llm_decision
   - Ejemplo: outputs/test_results.json

2. Directorio con JSON de entidades reales (--corpus-dir):
   - Un JSON por documento nombrado <document_id>.json
   - Cada JSON tiene: { "id": "...", "data": [ {"entity": "...", "text": "..."}, ... ] }
   - Ejemplo: corpus/ANTIGUO/entidades/

LÓGICA DE MÉTRICAS:
-------------------
Para cada documento:
  - TP: Entidad REAL que aparece en predicciones con TRUE
  - FN: Entidad REAL que NO aparece o aparece como FALSE
  - FP: Entidad marcada como TRUE que NO está en entidades reales

Alineación por texto normalizado (lowercase + strip).

SALIDA:
-------
JSON con métricas globales y por documento:
{
  "global_metrics": { "tp": ..., "fp": ..., "fn": ..., "precision": ..., "recall": ..., "f1": ... },
  "documents": [ { "document_id": ..., "tp": ..., ... }, ... ]
}

USO:
----
python compute_llm_metrics.py --predictions outputs/test_results.json --corpus-dir corpus/ANTIGUO/entidades --output outputs/llm_metrics.json
"""

import os
import sys
import json
import argparse
import datetime
import unicodedata
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ============================================================================
# FUNCIONES DE NORMALIZACIÓN
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparación robusta.
    
    - Convierte a minúsculas
    - Elimina espacios al inicio/final
    - Normaliza espacios múltiples
    - Normaliza caracteres Unicode (acentos)
    
    Args:
        text: Texto a normalizar
        
    Returns:
        Texto normalizado
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Strip y lowercase
    normalized = text.strip().lower()
    
    # Normalizar espacios múltiples a uno solo
    normalized = " ".join(normalized.split())
    
    # Normalización Unicode (NFD -> NFC para consistencia)
    normalized = unicodedata.normalize("NFC", normalized)
    
    return normalized


def normalize_label(label: str) -> str:
    """
    Normaliza etiquetas/labels para comparación.
    
    Args:
        label: Etiqueta a normalizar
        
    Returns:
        Etiqueta normalizada
    """
    if not label or not isinstance(label, str):
        return ""
    
    return label.strip().upper()


# ============================================================================
# FUNCIONES DE LOGGING
# ============================================================================

def log_message(message: str, level: str = "INFO"):
    """
    Imprime mensajes con timestamp y nivel de logging.
    
    Args:
        message: Mensaje a imprimir
        level: Nivel de log (DEBUG, INFO, WARN, ERROR)
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def log_debug(message: str):
    """Log de nivel DEBUG."""
    log_message(message, "DEBUG")


def log_info(message: str):
    """Log de nivel INFO."""
    log_message(message, "INFO")


def log_warn(message: str):
    """Log de nivel WARN."""
    log_message(message, "WARN")


def log_error(message: str):
    """Log de nivel ERROR."""
    log_message(message, "ERROR")


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class DocumentMetrics:
    """Métricas de evaluación para un documento."""
    document_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    total_real_entities: int = 0
    total_predictions_true: int = 0
    
    def calculate_derived_metrics(self):
        """Calcula precision, recall y F1 a partir de TP, FP, FN."""
        # Precision = TP / (TP + FP)
        if (self.tp + self.fp) > 0:
            self.precision = self.tp / (self.tp + self.fp)
        else:
            self.precision = 0.0
        
        # Recall = TP / (TP + FN)
        if (self.tp + self.fn) > 0:
            self.recall = self.tp / (self.tp + self.fn)
        else:
            self.recall = 0.0
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        if (self.precision + self.recall) > 0:
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1 = 0.0
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario para exportación."""
        return {
            "document_id": self.document_id,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "total_real_entities": self.total_real_entities,
            "total_predictions_true": self.total_predictions_true
        }


@dataclass
class GlobalMetrics:
    """Métricas globales de evaluación."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    total_documents: int = 0
    total_real_entities: int = 0
    total_predictions_true: int = 0
    total_predictions: int = 0
    
    def calculate_derived_metrics(self):
        """Calcula precision, recall y F1 a partir de TP, FP, FN."""
        # Precision = TP / (TP + FP)
        if (self.tp + self.fp) > 0:
            self.precision = self.tp / (self.tp + self.fp)
        else:
            self.precision = 0.0
        
        # Recall = TP / (TP + FN)
        if (self.tp + self.fn) > 0:
            self.recall = self.tp / (self.tp + self.fn)
        else:
            self.recall = 0.0
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        if (self.precision + self.recall) > 0:
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1 = 0.0
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario para exportación."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "total_documents": self.total_documents,
            "total_real_entities": self.total_real_entities,
            "total_predictions_true": self.total_predictions_true,
            "total_predictions": self.total_predictions
        }


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_predictions(predictions_path: str) -> List[Dict]:
    """
    Carga el archivo JSON de predicciones del LLM.
    
    Args:
        predictions_path: Ruta al archivo JSON de predicciones
        
    Returns:
        Lista de predicciones
    """
    log_info(f"Cargando predicciones desde: {predictions_path}")
    
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"No se encontró el archivo de predicciones: {predictions_path}")
    
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    if not isinstance(predictions, list):
        raise ValueError("El archivo de predicciones debe ser una lista JSON")
    
    log_info(f"  → Cargadas {len(predictions)} predicciones")
    return predictions


def load_corpus_entities(corpus_dir: str) -> Dict[str, Set[str]]:
    """
    Carga todas las entidades reales del corpus.
    
    Cada archivo JSON en el directorio representa un documento con sus entidades.
    
    Args:
        corpus_dir: Directorio con los JSON de entidades
        
    Returns:
        Diccionario: { document_id: set(texto_normalizado, ...) }
    """
    log_info(f"Cargando entidades del corpus desde: {corpus_dir}")
    
    if not os.path.exists(corpus_dir):
        raise FileNotFoundError(f"No se encontró el directorio del corpus: {corpus_dir}")
    
    corpus_entities: Dict[str, Set[str]] = {}
    total_entities = 0
    files_loaded = 0
    
    # Buscar todos los archivos JSON
    json_files = list(Path(corpus_dir).glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No se encontraron archivos JSON en: {corpus_dir}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Obtener document_id del archivo
            # Puede estar en "id", "document_id" o en el nombre del archivo
            doc_id = None
            if "id" in doc_data:
                doc_id = doc_data["id"]
            elif "document_id" in doc_data:
                doc_id = doc_data["document_id"]
            else:
                # Usar el nombre del archivo sin extensión
                doc_id = json_file.stem
            
            # Extraer entidades
            entities_set: Set[str] = set()
            
            # Las entidades pueden estar en "data" o "entities"
            entities_list = doc_data.get("data", doc_data.get("entities", []))
            
            for entity in entities_list:
                # El texto puede estar en "text", "entity_text", "palabra", "keyword"
                text = entity.get("text", entity.get("entity_text", entity.get("palabra", entity.get("keyword", ""))))
                
                if text:
                    normalized = normalize_text(text)
                    if normalized:
                        entities_set.add(normalized)
            
            if doc_id:
                corpus_entities[doc_id] = entities_set
                total_entities += len(entities_set)
                files_loaded += 1
                
        except Exception as e:
            log_warn(f"Error al cargar {json_file}: {e}")
            continue
    
    log_info(f"  → Cargados {files_loaded} documentos con {total_entities} entidades únicas")
    return corpus_entities


# ============================================================================
# PROCESAMIENTO Y CÁLCULO DE MÉTRICAS
# ============================================================================

def extract_prediction_info(prediction: Dict) -> Tuple[str, str, bool]:
    """
    Extrae document_id, texto y decisión del LLM de una predicción.
    
    Args:
        prediction: Diccionario con los datos de una predicción
        
    Returns:
        Tupla (document_id, texto_normalizado, llm_decision_bool)
    """
    # Document ID
    doc_id = prediction.get("document_id", prediction.get("doc_id", ""))
    
    # Texto de la entidad (puede tener varios nombres)
    text = prediction.get("keyword", 
           prediction.get("entity", 
           prediction.get("entity_text", 
           prediction.get("text", 
           prediction.get("texto_detectado", "")))))
    
    # Decisión del LLM (puede ser bool o string)
    llm_decision = prediction.get("is_valid",
                   prediction.get("llm_decision",
                   prediction.get("llm_bool", False)))
    
    # Convertir a bool si es string
    if isinstance(llm_decision, str):
        llm_decision = llm_decision.strip().upper() in ["TRUE", "1", "YES", "SI", "VÁLIDO", "CORRECTO"]
    elif not isinstance(llm_decision, bool):
        llm_decision = bool(llm_decision)
    
    return doc_id, normalize_text(text), llm_decision


def calculate_document_metrics(
    doc_id: str,
    real_entities: Set[str],
    predictions_true: Set[str],
    predictions_false: Set[str]
) -> DocumentMetrics:
    """
    Calcula métricas para un documento individual.
    
    Definición de métricas:
    - TP: Entidad REAL que está en predicciones TRUE
    - FN: Entidad REAL que NO está en predicciones o está como FALSE  
    - FP: Entidad en predicciones TRUE que NO está en entidades reales
    
    Args:
        doc_id: ID del documento
        real_entities: Set de textos normalizados de entidades reales
        predictions_true: Set de textos normalizados marcados como TRUE por LLM
        predictions_false: Set de textos normalizados marcados como FALSE por LLM
        
    Returns:
        DocumentMetrics con los valores calculados
    """
    metrics = DocumentMetrics(document_id=doc_id)
    metrics.total_real_entities = len(real_entities)
    metrics.total_predictions_true = len(predictions_true)
    
    # TP: Entidades reales que están en predicciones TRUE
    tp_entities = real_entities & predictions_true
    metrics.tp = len(tp_entities)
    
    # FN: Entidades reales que NO están en predicciones TRUE
    # (ya sea porque no aparecen o porque están como FALSE)
    fn_entities = real_entities - predictions_true
    metrics.fn = len(fn_entities)
    
    # FP: Entidades en predicciones TRUE que NO son reales
    fp_entities = predictions_true - real_entities
    metrics.fp = len(fp_entities)
    
    # Calcular métricas derivadas
    metrics.calculate_derived_metrics()
    
    return metrics


def process_metrics(
    predictions: List[Dict],
    corpus_entities: Dict[str, Set[str]],
    verbose: bool = False
) -> Tuple[GlobalMetrics, List[DocumentMetrics]]:
    """
    Procesa todas las predicciones y calcula métricas.
    
    Args:
        predictions: Lista de predicciones del LLM
        corpus_entities: Diccionario de entidades reales por documento
        verbose: Si True, muestra información detallada
        
    Returns:
        Tupla (GlobalMetrics, lista de DocumentMetrics)
    """
    log_info("Procesando predicciones y calculando métricas...")
    
    # Agrupar predicciones por documento
    predictions_by_doc: Dict[str, Dict[str, Set[str]]] = {}
    
    for pred in predictions:
        doc_id, text, llm_decision = extract_prediction_info(pred)
        
        if not doc_id or not text:
            continue
        
        if doc_id not in predictions_by_doc:
            predictions_by_doc[doc_id] = {"true": set(), "false": set()}
        
        if llm_decision:
            predictions_by_doc[doc_id]["true"].add(text)
        else:
            predictions_by_doc[doc_id]["false"].add(text)
    
    log_info(f"  → Predicciones agrupadas en {len(predictions_by_doc)} documentos")
    
    # Calcular métricas por documento
    document_metrics_list: List[DocumentMetrics] = []
    global_metrics = GlobalMetrics()
    
    # Documentos que tienen predicciones
    docs_with_predictions = set(predictions_by_doc.keys())
    # Documentos que tienen entidades reales
    docs_with_entities = set(corpus_entities.keys())
    
    # Procesar documentos que tienen predicciones
    for doc_id in docs_with_predictions:
        preds_true = predictions_by_doc[doc_id]["true"]
        preds_false = predictions_by_doc[doc_id]["false"]
        
        # Obtener entidades reales (si no hay, set vacío)
        real_entities = corpus_entities.get(doc_id, set())
        
        if not real_entities and verbose:
            log_warn(f"  → Documento {doc_id}: sin entidades reales en corpus")
        
        # Calcular métricas del documento
        doc_metrics = calculate_document_metrics(
            doc_id, real_entities, preds_true, preds_false
        )
        document_metrics_list.append(doc_metrics)
        
        # Acumular en métricas globales
        global_metrics.tp += doc_metrics.tp
        global_metrics.fp += doc_metrics.fp
        global_metrics.fn += doc_metrics.fn
        global_metrics.total_real_entities += doc_metrics.total_real_entities
        global_metrics.total_predictions_true += doc_metrics.total_predictions_true
        
        if verbose:
            log_debug(f"  → Doc {doc_id}: TP={doc_metrics.tp}, FP={doc_metrics.fp}, FN={doc_metrics.fn}")
    
    # Contar documentos y predicciones
    global_metrics.total_documents = len(document_metrics_list)
    global_metrics.total_predictions = len(predictions)
    
    # Calcular métricas derivadas globales
    global_metrics.calculate_derived_metrics()
    
    log_info(f"  → Procesados {global_metrics.total_documents} documentos")
    log_info(f"  → Métricas globales: TP={global_metrics.tp}, FP={global_metrics.fp}, FN={global_metrics.fn}")
    log_info(f"  → Precision={global_metrics.precision:.4f}, Recall={global_metrics.recall:.4f}, F1={global_metrics.f1:.4f}")
    
    return global_metrics, document_metrics_list


# ============================================================================
# EXPORTACIÓN DE RESULTADOS
# ============================================================================

def export_results(
    global_metrics: GlobalMetrics,
    document_metrics: List[DocumentMetrics],
    output_path: str
):
    """
    Exporta los resultados a un archivo JSON.
    
    Args:
        global_metrics: Métricas globales
        document_metrics: Lista de métricas por documento
        output_path: Ruta del archivo de salida
    """
    log_info(f"Exportando resultados a: {output_path}")
    
    # Crear directorio si no existe
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Construir estructura de salida
    results = {
        "metadata": {
            "generated_at": datetime.datetime.now().isoformat(),
            "script": "compute_llm_metrics.py",
            "description": "Métricas de evaluación LLM vs Ground Truth"
        },
        "global_metrics": global_metrics.to_dict(),
        "documents": [doc.to_dict() for doc in document_metrics]
    }
    
    # Guardar JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    log_info(f"  → Resultados guardados exitosamente")


def print_summary(global_metrics: GlobalMetrics, document_metrics: List[DocumentMetrics]):
    """
    Imprime un resumen de las métricas en consola.
    
    Args:
        global_metrics: Métricas globales
        document_metrics: Lista de métricas por documento
    """
    print("\n" + "="*70)
    print("RESUMEN DE MÉTRICAS")
    print("="*70)
    
    print(f"\n📊 MÉTRICAS GLOBALES:")
    print(f"   Total documentos evaluados: {global_metrics.total_documents}")
    print(f"   Total predicciones:         {global_metrics.total_predictions}")
    print(f"   Total entidades reales:     {global_metrics.total_real_entities}")
    print(f"   Predicciones TRUE:          {global_metrics.total_predictions_true}")
    
    print(f"\n📈 MATRIZ DE CONFUSIÓN:")
    print(f"   TP (True Positives):  {global_metrics.tp}")
    print(f"   FP (False Positives): {global_metrics.fp}")
    print(f"   FN (False Negatives): {global_metrics.fn}")
    
    print(f"\n📉 MÉTRICAS DE RENDIMIENTO:")
    print(f"   Precision: {global_metrics.precision:.4f} ({global_metrics.precision*100:.2f}%)")
    print(f"   Recall:    {global_metrics.recall:.4f} ({global_metrics.recall*100:.2f}%)")
    print(f"   F1-Score:  {global_metrics.f1:.4f} ({global_metrics.f1*100:.2f}%)")
    
    # Top 5 documentos con peor rendimiento (F1 más bajo)
    if document_metrics:
        sorted_docs = sorted(document_metrics, key=lambda x: x.f1)
        print(f"\n⚠️  TOP 5 DOCUMENTOS CON PEOR F1:")
        for i, doc in enumerate(sorted_docs[:5], 1):
            print(f"   {i}. {doc.document_id}: F1={doc.f1:.4f} (TP={doc.tp}, FP={doc.fp}, FN={doc.fn})")
        
        # Top 5 documentos con mejor rendimiento
        sorted_docs_best = sorted(document_metrics, key=lambda x: x.f1, reverse=True)
        print(f"\n✅ TOP 5 DOCUMENTOS CON MEJOR F1:")
        for i, doc in enumerate(sorted_docs_best[:5], 1):
            print(f"   {i}. {doc.document_id}: F1={doc.f1:.4f} (TP={doc.tp}, FP={doc.fp}, FN={doc.fn})")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# CLI Y MAIN
# ============================================================================

def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Calcula métricas de evaluación LLM comparando predicciones con entidades reales.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python compute_llm_metrics.py --predictions outputs/test_results.json --corpus-dir corpus/ANTIGUO/entidades --output outputs/metrics.json
  python compute_llm_metrics.py -p outputs/test_results.json -c corpus/ANTIGUO/entidades -o outputs/metrics.json -v
        """
    )
    
    parser.add_argument(
        "--predictions", "-p",
        required=True,
        help="Ruta al archivo JSON con las predicciones del LLM"
    )
    
    parser.add_argument(
        "--corpus-dir", "-c",
        required=True,
        help="Directorio con los JSON de entidades reales del corpus"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Ruta del archivo JSON de salida con las métricas"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Muestra información detallada de depuración"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="No mostrar el resumen en consola"
    )
    
    return parser.parse_args()


def main():
    """Función principal del script."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("COMPUTE LLM METRICS - Evaluación de predicciones LLM")
    print("="*70 + "\n")
    
    try:
        # 1. Cargar predicciones
        predictions = load_predictions(args.predictions)
        
        # 2. Cargar entidades del corpus
        corpus_entities = load_corpus_entities(args.corpus_dir)
        
        # 3. Procesar y calcular métricas
        global_metrics, document_metrics = process_metrics(
            predictions, 
            corpus_entities,
            verbose=args.verbose
        )
        
        # 4. Exportar resultados
        export_results(global_metrics, document_metrics, args.output)
        
        # 5. Mostrar resumen (opcional)
        if not args.no_summary:
            print_summary(global_metrics, document_metrics)
        
        log_info("✅ Proceso completado exitosamente")
        return 0
        
    except FileNotFoundError as e:
        log_error(f"Archivo no encontrado: {e}")
        return 1
    except ValueError as e:
        log_error(f"Error de validación: {e}")
        return 1
    except Exception as e:
        log_error(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
