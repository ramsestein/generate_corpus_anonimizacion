#!/usr/bin/env python3
"""
Evaluate SetFit Gatekeeper Model WITH CONTEXT.

Este script evalúa un modelo SetFit entrenado como "Gatekeeper Semántico"
sobre un dataset de entidades NER. El modelo clasifica cada entidad como:
- Clase 1 (PII): Información sensible que debe anonimizarse
- Clase 0 (Ruido): Contexto médico seguro, no anonimizar

IMPORTANTE: Esta versión usa CONTEXTO del documento original.
Para cada entidad, construye un input del estilo:
    "ENTITY: {texto_entidad}\nSENTENCE: {frase_del_documento}"

Funcionalidades:
- Carga modelo SetFit desde disco
- Carga documentos originales para extraer contexto
- Procesa dataset de entidades en formato JSON
- Extrae la frase/segmento donde aparece cada entidad
- Genera predicciones binarias usando entidad + contexto
- Calcula métricas si hay ground truth disponible
- Guarda resultados detallados y métricas resumidas

Author: Pipeline Anonimización Clínica
Version: 2.0.0 (con contexto)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas por defecto (relativas a la raíz del proyecto)
DEFAULT_MODEL_PATHS = [
    "models/gatekeeper_setfit",
    "models/setfit_entidades",
    "models/setfit_gatekeeper",
]

DEFAULT_DATASET_PATHS = [
    "entidades-procesadas-para-metricas.json",
    "entidades-para-metricas.json",
    "entidades-para-metricas.csv",
]

# Rutas donde buscar los documentos originales
DEFAULT_DOCUMENTS_PATHS = [
    "corpus/ANTIGUO/documents",
    "corpus/documents",
    "documents",
    "corpus/output",
]

DEFAULT_OUTPUT_DIR = "outputs"

# Configuración de extracción de contexto
CONTEXT_WINDOW_CHARS = 150  # Caracteres de contexto a cada lado si no hay frase
SENTENCE_DELIMITERS = r'[.!?;:\n]'  # Delimitadores de frase

# Mapeo de columnas conocidas
TEXT_COLUMN_CANDIDATES = ["text", "entity_text", "span_text", "texto", "entidad"]
LABEL_COLUMN_CANDIDATES = ["label", "true_label", "ground_truth", "etiqueta", "ner_label"]
DOC_ID_COLUMN_CANDIDATES = ["doc_id", "document_id", "id", "doc"]


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def find_project_root() -> Path:
    """
    Encuentra la raíz del proyecto buscando archivos clave.
    
    Returns:
        Path a la raíz del proyecto.
    """
    current = Path(__file__).resolve()
    
    # Subir hasta encontrar la raíz (donde está requirements.txt o .git)
    for parent in [current] + list(current.parents):
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent
    
    # Fallback: directorio actual
    return Path.cwd()


def find_file(candidates: List[str], base_path: Path) -> Optional[Path]:
    """
    Busca un archivo en una lista de candidatos.
    
    Args:
        candidates: Lista de rutas relativas posibles.
        base_path: Ruta base desde donde buscar.
    
    Returns:
        Path al primer archivo encontrado, o None.
    """
    for candidate in candidates:
        path = base_path / candidate
        if path.exists():
            return path
    return None


def detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Detecta una columna en un DataFrame buscando entre candidatos.
    
    Args:
        df: DataFrame donde buscar.
        candidates: Lista de nombres de columna candidatos.
    
    Returns:
        Nombre de la columna encontrada, o None.
    """
    df_columns_lower = {col.lower(): col for col in df.columns}
    
    for candidate in candidates:
        if candidate.lower() in df_columns_lower:
            return df_columns_lower[candidate.lower()]
    
    return None


def get_device() -> str:
    """
    Detecta el dispositivo disponible (GPU o CPU).
    
    Returns:
        String del dispositivo ('cuda', 'mps', o 'cpu').
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ============================================================================
# CARGA Y GESTIÓN DE DOCUMENTOS
# ============================================================================

class DocumentLoader:
    """
    Carga y gestiona los documentos originales para extraer contexto.
    
    Mantiene un cache de documentos cargados para evitar lecturas repetidas.
    """
    
    def __init__(self, documents_dir: Path):
        """
        Inicializa el DocumentLoader.
        
        Args:
            documents_dir: Directorio donde están los documentos originales.
        """
        self.documents_dir = Path(documents_dir)
        self._cache: Dict[str, str] = {}
        self._missing_docs: set = set()
        
        if not self.documents_dir.exists():
            logger.warning(f"Directorio de documentos no existe: {self.documents_dir}")
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """
        Obtiene el texto de un documento por su ID.
        
        Args:
            doc_id: ID del documento.
        
        Returns:
            Texto del documento, o None si no se encuentra.
        """
        # Verificar cache
        if doc_id in self._cache:
            return self._cache[doc_id]
        
        # Verificar si ya sabemos que no existe
        if doc_id in self._missing_docs:
            return None
        
        # Buscar archivo
        possible_paths = [
            self.documents_dir / f"{doc_id}.txt",
            self.documents_dir / f"{doc_id}",
            self.documents_dir / f"{doc_id}.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    self._cache[doc_id] = text
                    return text
                except Exception as e:
                    logger.debug(f"Error leyendo {path}: {e}")
        
        # No encontrado
        self._missing_docs.add(doc_id)
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """
        Retorna estadísticas de documentos cargados.
        
        Returns:
            Dict con estadísticas.
        """
        return {
            "loaded": len(self._cache),
            "missing": len(self._missing_docs)
        }


def find_documents_directory(base_path: Path) -> Optional[Path]:
    """
    Busca el directorio de documentos originales.
    
    Args:
        base_path: Ruta base del proyecto.
    
    Returns:
        Path al directorio de documentos, o None.
    """
    for candidate in DEFAULT_DOCUMENTS_PATHS:
        path = base_path / candidate
        if path.exists() and path.is_dir():
            # Verificar que tiene archivos .txt
            txt_files = list(path.glob("*.txt"))
            if txt_files:
                logger.info(f"Directorio de documentos encontrado: {path} ({len(txt_files)} archivos)")
                return path
    
    return None


def extract_sentence_context(
    text: str,
    start: int,
    end: int,
    entity_text: str,
    window_chars: int = CONTEXT_WINDOW_CHARS
) -> str:
    """
    Extrae la frase o segmento del documento donde aparece la entidad.
    
    Intenta encontrar la frase completa. Si no es posible, usa una
    ventana de caracteres alrededor de la entidad.
    
    Args:
        text: Texto completo del documento.
        start: Posición de inicio de la entidad.
        end: Posición de fin de la entidad.
        entity_text: Texto de la entidad (para validación).
        window_chars: Caracteres de contexto si no hay frase.
    
    Returns:
        Frase o segmento de contexto.
    """
    if not text or start < 0 or end > len(text):
        return ""
    
    # Verificar que las posiciones son correctas
    extracted = text[start:end]
    if extracted.strip() != entity_text.strip():
        # Las posiciones no coinciden exactamente, usar búsqueda
        pos = text.find(entity_text)
        if pos != -1:
            start = pos
            end = pos + len(entity_text)
        else:
            # No se encontró la entidad, retornar ventana centrada en la posición original
            context_start = max(0, start - window_chars)
            context_end = min(len(text), end + window_chars)
            return text[context_start:context_end].strip()
    
    # Buscar límites de frase
    # Encontrar inicio de frase (buscar hacia atrás)
    sentence_start = start
    for i in range(start - 1, max(0, start - 500), -1):
        if re.match(SENTENCE_DELIMITERS, text[i]):
            sentence_start = i + 1
            break
    else:
        # No encontró delimitador, usar ventana
        sentence_start = max(0, start - window_chars)
    
    # Encontrar fin de frase (buscar hacia adelante)
    sentence_end = end
    for i in range(end, min(len(text), end + 500)):
        if re.match(SENTENCE_DELIMITERS, text[i]):
            sentence_end = i + 1
            break
    else:
        # No encontró delimitador, usar ventana
        sentence_end = min(len(text), end + window_chars)
    
    # Extraer y limpiar
    sentence = text[sentence_start:sentence_end].strip()
    
    # Limitar longitud máxima
    max_length = 500
    if len(sentence) > max_length:
        # Centrar en la entidad
        entity_pos = start - sentence_start
        half_window = max_length // 2
        new_start = max(0, entity_pos - half_window)
        new_end = min(len(sentence), entity_pos + len(entity_text) + half_window)
        sentence = sentence[new_start:new_end].strip()
    
    return sentence


def build_contextualized_input(entity_text: str, sentence: str) -> str:
    """
    Construye el input contextualizado para el modelo SetFit.
    
    Args:
        entity_text: Texto de la entidad.
        sentence: Frase o contexto donde aparece.
    
    Returns:
        String formateado para el modelo.
    """
    if not sentence:
        # Sin contexto, usar solo la entidad
        return f"ENTITY: {entity_text}"
    
    return f"ENTITY: {entity_text}\nSENTENCE: {sentence}"


# ============================================================================
# CARGA DE MODELO
# ============================================================================

def load_setfit_model(model_path: Union[str, Path]) -> Any:
    """
    Carga un modelo SetFit desde disco.
    
    Args:
        model_path: Ruta al directorio del modelo.
    
    Returns:
        Modelo SetFit cargado.
    
    Raises:
        FileNotFoundError: Si el modelo no existe.
        ImportError: Si setfit no está instalado.
    """
    try:
        from setfit import SetFitModel
    except ImportError:
        logger.error("La librería 'setfit' no está instalada.")
        logger.error("Instálala con: pip install setfit")
        raise ImportError("setfit no está instalado")
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    
    device = get_device()
    logger.info(f"Cargando modelo desde: {model_path}")
    logger.info(f"Dispositivo detectado: {device}")
    
    # Cargar modelo
    model = SetFitModel.from_pretrained(str(model_path))
    
    # Mover a GPU si está disponible
    if device != "cpu":
        try:
            model = model.to(device)
            logger.info(f"Modelo movido a {device}")
        except Exception as e:
            logger.warning(f"No se pudo mover a {device}: {e}. Usando CPU.")
    
    return model


def find_model(base_path: Path) -> Optional[Path]:
    """
    Busca el modelo SetFit en ubicaciones conocidas.
    
    Args:
        base_path: Ruta base del proyecto.
    
    Returns:
        Path al modelo, o None si no se encuentra.
    """
    # Buscar en rutas por defecto
    model_path = find_file(DEFAULT_MODEL_PATHS, base_path)
    if model_path:
        return model_path
    
    # Buscar recursivamente en models/
    models_dir = base_path / "models"
    if models_dir.exists():
        for path in models_dir.rglob("*"):
            if path.is_dir() and (path / "config.json").exists():
                # Verificar que sea un modelo SetFit
                try:
                    config = json.load(open(path / "config.json"))
                    if "setfit" in str(config).lower():
                        return path
                except:
                    pass
    
    return None


# ============================================================================
# CARGA DE DATASET
# ============================================================================

def load_dataset(dataset_path: Union[str, Path]) -> pd.DataFrame:
    """
    Carga un dataset de entidades desde CSV o JSON.
    
    Args:
        dataset_path: Ruta al archivo del dataset.
    
    Returns:
        DataFrame con las entidades.
    
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato no es soportado.
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
    
    logger.info(f"Cargando dataset desde: {dataset_path}")
    
    suffix = dataset_path.suffix.lower()
    
    if suffix == ".csv":
        df = pd.read_csv(dataset_path, encoding="utf-8")
    elif suffix == ".json":
        # Intentar cargar como JSON estructurado (con metadata + entities)
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Verificar si tiene estructura con 'entities'
        if isinstance(data, dict) and "entities" in data:
            df = pd.DataFrame(data["entities"])
            logger.info(f"Dataset JSON con metadata. Total entidades: {len(df)}")
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Formato JSON no reconocido en: {dataset_path}")
    else:
        raise ValueError(f"Formato no soportado: {suffix}. Use .csv o .json")
    
    logger.info(f"Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")
    logger.info(f"Columnas: {list(df.columns)}")
    
    return df


def prepare_data_with_context(
    df: pd.DataFrame,
    doc_loader: DocumentLoader
) -> Tuple[List[str], List[str], List[str], Optional[List[int]], Optional[str]]:
    """
    Prepara los datos para evaluación con SetFit USANDO CONTEXTO.
    
    Para cada entidad:
    1. Obtiene el documento original
    2. Extrae la frase/segmento donde aparece la entidad
    3. Construye el input contextualizado: "ENTITY: {entity}\nSENTENCE: {context}"
    
    Args:
        df: DataFrame con las entidades.
        doc_loader: DocumentLoader para obtener documentos originales.
    
    Returns:
        Tuple con:
        - Lista de inputs contextualizados para el modelo
        - Lista de textos originales de entidades (sin contexto)
        - Lista de document IDs
        - Lista de ground truth labels (si existe)
        - Nombre de la columna de etiqueta NER
    
    Raises:
        ValueError: Si no se encuentra la columna de texto.
    """
    # Detectar columna de texto
    text_col = detect_column(df, TEXT_COLUMN_CANDIDATES)
    if not text_col:
        raise ValueError(
            f"No se encontró columna de texto. "
            f"Columnas disponibles: {list(df.columns)}. "
            f"Esperadas: {TEXT_COLUMN_CANDIDATES}"
        )
    
    logger.info(f"Columna de texto detectada: '{text_col}'")
    
    # Detectar columna de document ID
    doc_id_col = detect_column(df, DOC_ID_COLUMN_CANDIDATES)
    if doc_id_col:
        doc_ids = df[doc_id_col].astype(str).tolist()
        logger.info(f"Columna de document ID detectada: '{doc_id_col}'")
    else:
        doc_ids = [str(i) for i in range(len(df))]
        logger.warning("No se encontró columna de document ID, usando índices.")
    
    # Detectar columna de etiqueta NER
    label_col = detect_column(df, LABEL_COLUMN_CANDIDATES)
    if label_col:
        logger.info(f"Columna de etiqueta NER detectada: '{label_col}'")
    
    # Verificar columnas de posición
    has_positions = "start" in df.columns and "end" in df.columns
    if has_positions:
        logger.info("Columnas de posición (start, end) detectadas.")
    else:
        logger.warning("No se encontraron columnas de posición (start, end).")
    
    # Construir inputs contextualizados
    contextualized_inputs = []
    original_texts = []
    sentences = []
    docs_found = 0
    docs_missing = 0
    
    logger.info("Construyendo inputs contextualizados...")
    
    for idx, row in df.iterrows():
        entity_text = str(row[text_col])
        original_texts.append(entity_text)
        
        doc_id = str(row[doc_id_col]) if doc_id_col else str(idx)
        
        # Obtener documento
        doc_text = doc_loader.get_document(doc_id)
        
        if doc_text and has_positions:
            # Extraer contexto usando posiciones
            start = int(row["start"])
            end = int(row["end"])
            sentence = extract_sentence_context(doc_text, start, end, entity_text)
            sentences.append(sentence)
            
            if sentence:
                docs_found += 1
            else:
                docs_missing += 1
        elif doc_text:
            # Documento encontrado pero sin posiciones, buscar la entidad
            pos = doc_text.find(entity_text)
            if pos != -1:
                sentence = extract_sentence_context(
                    doc_text, pos, pos + len(entity_text), entity_text
                )
                sentences.append(sentence)
                docs_found += 1
            else:
                sentences.append("")
                docs_missing += 1
        else:
            sentences.append("")
            docs_missing += 1
        
        # Construir input contextualizado
        contextualized_input = build_contextualized_input(entity_text, sentences[-1])
        contextualized_inputs.append(contextualized_input)
    
    logger.info(f"Contexto extraído: {docs_found} entidades con contexto, {docs_missing} sin contexto")
    
    # Verificar si hay ground truth binario (0/1)
    ground_truth = None
    for col in ["ground_truth", "true_label", "is_pii", "pii", "binary_label"]:
        if col in df.columns:
            try:
                gt = df[col].astype(int).tolist()
                if all(v in [0, 1] for v in gt):
                    ground_truth = gt
                    logger.info(f"Ground truth binario encontrado en columna: '{col}'")
                    break
            except:
                pass
    
    if ground_truth is None:
        logger.info("No se encontró ground truth binario. Solo se generarán predicciones.")
    
    return contextualized_inputs, original_texts, doc_ids, ground_truth, label_col


# ============================================================================
# EVALUACIÓN
# ============================================================================

def run_predictions(model: Any, texts: List[str]) -> Tuple[List[int], List[float]]:
    """
    Ejecuta predicciones con el modelo SetFit.
    
    Args:
        model: Modelo SetFit cargado.
        texts: Lista de textos a clasificar.
    
    Returns:
        Tuple con:
        - Lista de predicciones (0 o 1)
        - Lista de probabilidades (si el modelo las proporciona)
    """
    logger.info(f"Ejecutando predicciones sobre {len(texts)} textos...")
    
    # Hacer predicciones
    predictions = model.predict(texts)
    
    # Convertir a lista de enteros
    if hasattr(predictions, 'tolist'):
        pred_list = predictions.tolist()
    else:
        pred_list = list(predictions)
    
    # Intentar obtener probabilidades
    probabilities = []
    try:
        probs = model.predict_proba(texts)
        if hasattr(probs, 'tolist'):
            probs = probs.tolist()
        # Tomar la probabilidad de la clase predicha
        for i, pred in enumerate(pred_list):
            if isinstance(probs[i], (list, np.ndarray)):
                probabilities.append(float(probs[i][pred]))
            else:
                probabilities.append(float(probs[i]))
    except Exception as e:
        logger.debug(f"No se pudieron obtener probabilidades: {e}")
        probabilities = [1.0] * len(pred_list)
    
    # Asegurar que son enteros
    pred_list = [int(p) for p in pred_list]
    
    logger.info(f"Predicciones completadas.")
    logger.info(f"  - Clase 0 (Ruido): {pred_list.count(0)}")
    logger.info(f"  - Clase 1 (PII): {pred_list.count(1)}")
    
    return pred_list, probabilities


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calcula métricas de clasificación.
    
    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.
        labels: Nombres de las clases (opcional).
    
    Returns:
        Dict con todas las métricas calculadas.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report
    )
    
    if labels is None:
        labels = ["Ruido (0)", "PII (1)"]
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=labels, output_dict=True),
        "classification_report_str": classification_report(y_true, y_pred, target_names=labels),
        "total_samples": len(y_true),
        "class_distribution_true": {
            "class_0": y_true.count(0),
            "class_1": y_true.count(1)
        },
        "class_distribution_pred": {
            "class_0": y_pred.count(0),
            "class_1": y_pred.count(1)
        }
    }
    
    return metrics


def analyze_by_ner_label(
    df: pd.DataFrame,
    predictions: List[int],
    label_col: str
) -> Dict[str, Dict[str, int]]:
    """
    Analiza las predicciones agrupadas por etiqueta NER.
    
    Args:
        df: DataFrame original.
        predictions: Lista de predicciones.
        label_col: Nombre de la columna de etiqueta NER.
    
    Returns:
        Dict con estadísticas por etiqueta NER.
    """
    df_analysis = df.copy()
    df_analysis["prediction"] = predictions
    
    analysis = {}
    for label in df_analysis[label_col].unique():
        mask = df_analysis[label_col] == label
        preds = df_analysis.loc[mask, "prediction"]
        analysis[label] = {
            "total": int(len(preds)),
            "predicted_pii": int((preds == 1).sum()),
            "predicted_ruido": int((preds == 0).sum()),
            "pii_rate": float((preds == 1).mean()) if len(preds) > 0 else 0.0
        }
    
    return analysis


# ============================================================================
# GUARDADO DE RESULTADOS
# ============================================================================

def save_results(
    df: pd.DataFrame,
    predictions: List[int],
    probabilities: List[float],
    doc_ids: List[str],
    original_texts: List[str],
    contextualized_inputs: List[str],
    text_col: str,
    label_col: Optional[str],
    output_dir: Path,
    metrics: Optional[Dict[str, Any]] = None,
    ner_analysis: Optional[Dict[str, Dict[str, int]]] = None
) -> Tuple[Path, Path]:
    """
    Guarda los resultados de la evaluación con contexto.
    
    Args:
        df: DataFrame original.
        predictions: Lista de predicciones.
        probabilities: Lista de probabilidades.
        doc_ids: Lista de document IDs.
        original_texts: Lista de textos originales de entidades.
        contextualized_inputs: Lista de inputs contextualizados usados.
        text_col: Nombre de la columna de texto.
        label_col: Nombre de la columna de etiqueta NER.
        output_dir: Directorio de salida.
        metrics: Métricas calculadas (opcional).
        ner_analysis: Análisis por etiqueta NER (opcional).
    
    Returns:
        Tuple con paths a los archivos guardados.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # =========================================================================
    # Guardar resultados detallados (CSV)
    # =========================================================================
    results_df = pd.DataFrame({
        "document_id": doc_ids,
        "entity_text": original_texts,
        "ner_label": df[label_col].tolist() if label_col else ["N/A"] * len(df),
        "setfit_prediction": predictions,
        "setfit_prediction_label": ["PII" if p == 1 else "Ruido" for p in predictions],
        "confidence": probabilities,
        "contextualized_input": contextualized_inputs,
    })
    
    # Añadir columnas adicionales si existen
    for col in ["model", "start", "end", "unified"]:
        if col in df.columns:
            results_df[f"original_{col}"] = df[col].tolist()
    
    results_path = output_dir / f"setfit_context_resultados_{timestamp}.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8")
    logger.info(f"Resultados guardados en: {results_path}")
    
    # =========================================================================
    # Guardar métricas (JSON)
    # =========================================================================
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_mode": "WITH_CONTEXT",
        "total_entities": len(df),
        "predictions_summary": {
            "total_pii": predictions.count(1),
            "total_ruido": predictions.count(0),
            "pii_rate": predictions.count(1) / len(predictions) if predictions else 0
        }
    }
    
    if metrics:
        metrics_data["classification_metrics"] = {
            k: v for k, v in metrics.items() 
            if k != "classification_report_str"
        }
    
    if ner_analysis:
        metrics_data["analysis_by_ner_label"] = ner_analysis
    
    metrics_path = output_dir / f"setfit_context_metricas_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas guardadas en: {metrics_path}")
    
    return results_path, metrics_path


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def evaluate_setfit_gatekeeper(
    model_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    documents_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ejecuta la evaluación completa del modelo SetFit Gatekeeper CON CONTEXTO.
    
    Para cada entidad:
    1. Obtiene el documento original usando el document_id
    2. Extrae la frase/segmento donde aparece la entidad
    3. Construye input: "ENTITY: {texto}\nSENTENCE: {contexto}"
    4. Clasifica con SetFit usando este input contextualizado
    
    Args:
        model_path: Ruta al modelo (opcional, se buscará automáticamente).
        dataset_path: Ruta al dataset (opcional, se buscará automáticamente).
        documents_path: Ruta al directorio de documentos originales.
        output_dir: Directorio de salida (opcional).
    
    Returns:
        Dict con resultados de la evaluación.
    """
    logger.info("=" * 60)
    logger.info("🔍 Evaluación SetFit Gatekeeper CON CONTEXTO")
    logger.info("=" * 60)
    
    # =========================================================================
    # 1. Encontrar raíz del proyecto
    # =========================================================================
    project_root = find_project_root()
    logger.info(f"Raíz del proyecto: {project_root}")
    
    # =========================================================================
    # 2. Encontrar/cargar modelo
    # =========================================================================
    if model_path:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = project_root / model_path
    else:
        model_path = find_model(project_root)
        if not model_path:
            raise FileNotFoundError(
                f"No se encontró modelo SetFit. "
                f"Buscado en: {DEFAULT_MODEL_PATHS}"
            )
    
    model = load_setfit_model(model_path)
    
    # =========================================================================
    # 3. Encontrar/cargar dataset
    # =========================================================================
    if dataset_path:
        dataset_path = Path(dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path
    else:
        dataset_path = find_file(DEFAULT_DATASET_PATHS, project_root)
        if not dataset_path:
            raise FileNotFoundError(
                f"No se encontró dataset. "
                f"Buscado en: {DEFAULT_DATASET_PATHS}"
            )
    
    df = load_dataset(dataset_path)
    
    # =========================================================================
    # 4. Encontrar directorio de documentos
    # =========================================================================
    if documents_path:
        docs_dir = Path(documents_path)
        if not docs_dir.is_absolute():
            docs_dir = project_root / docs_dir
    else:
        docs_dir = find_documents_directory(project_root)
        if not docs_dir:
            logger.warning(
                f"No se encontró directorio de documentos. "
                f"Buscado en: {DEFAULT_DOCUMENTS_PATHS}. "
                f"Se usarán solo los textos de entidades sin contexto."
            )
            docs_dir = project_root / "corpus/ANTIGUO/documents"  # Default fallback
    
    doc_loader = DocumentLoader(docs_dir)
    logger.info(f"Directorio de documentos: {docs_dir}")
    
    # =========================================================================
    # 5. Preparar datos CON CONTEXTO
    # =========================================================================
    (
        contextualized_inputs,
        original_texts,
        doc_ids,
        ground_truth,
        label_col
    ) = prepare_data_with_context(df, doc_loader)
    
    # Mostrar estadísticas de documentos
    doc_stats = doc_loader.get_stats()
    logger.info(f"Documentos cargados: {doc_stats['loaded']}")
    logger.info(f"Documentos no encontrados: {doc_stats['missing']}")
    
    # Mostrar ejemplos de inputs contextualizados
    print("\n" + "=" * 60)
    print("📝 EJEMPLOS DE INPUTS CONTEXTUALIZADOS")
    print("=" * 60)
    for i in range(min(3, len(contextualized_inputs))):
        print(f"\n[Ejemplo {i+1}]")
        print(contextualized_inputs[i][:300])
        if len(contextualized_inputs[i]) > 300:
            print("...")
        print("-" * 40)
    
    # =========================================================================
    # 6. Ejecutar predicciones con inputs contextualizados
    # =========================================================================
    predictions, probabilities = run_predictions(model, contextualized_inputs)
    
    # =========================================================================
    # 7. Calcular métricas (si hay ground truth)
    # =========================================================================
    metrics = None
    if ground_truth:
        logger.info("Calculando métricas con ground truth...")
        metrics = calculate_metrics(ground_truth, predictions)
        
        print("\n" + "=" * 60)
        print("📊 MÉTRICAS DE EVALUACIÓN (CON CONTEXTO)")
        print("=" * 60)
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"F1 (macro):         {metrics['f1_macro']:.4f}")
        print(f"F1 (weighted):      {metrics['f1_weighted']:.4f}")
        print("\nMatriz de Confusión:")
        print(f"  [[TN={metrics['confusion_matrix'][0][0]}, FP={metrics['confusion_matrix'][0][1]}],")
        print(f"   [FN={metrics['confusion_matrix'][1][0]}, TP={metrics['confusion_matrix'][1][1]}]]")
        print("\nReporte de Clasificación:")
        print(metrics['classification_report_str'])
    
    # =========================================================================
    # 8. Análisis por etiqueta NER
    # =========================================================================
    ner_analysis = None
    if label_col:
        logger.info("Analizando predicciones por etiqueta NER...")
        ner_analysis = analyze_by_ner_label(df, predictions, label_col)
        
        print("\n" + "=" * 60)
        print("📋 ANÁLISIS POR ETIQUETA NER (CON CONTEXTO)")
        print("=" * 60)
        print(f"{'Etiqueta':<35} {'Total':>8} {'PII':>8} {'Ruido':>8} {'%PII':>8}")
        print("-" * 67)
        
        # Ordenar por tasa de PII descendente
        sorted_labels = sorted(
            ner_analysis.items(),
            key=lambda x: x[1]['pii_rate'],
            reverse=True
        )
        
        for label, stats in sorted_labels:
            print(
                f"{label:<35} {stats['total']:>8} "
                f"{stats['predicted_pii']:>8} {stats['predicted_ruido']:>8} "
                f"{stats['pii_rate']*100:>7.1f}%"
            )
    
    # =========================================================================
    # 9. Guardar resultados
    # =========================================================================
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = project_root / DEFAULT_OUTPUT_DIR
    
    text_col = detect_column(df, TEXT_COLUMN_CANDIDATES)
    results_path, metrics_path = save_results(
        df=df,
        predictions=predictions,
        probabilities=probabilities,
        doc_ids=doc_ids,
        original_texts=original_texts,
        contextualized_inputs=contextualized_inputs,
        text_col=text_col,
        label_col=label_col,
        output_dir=output_path,
        metrics=metrics,
        ner_analysis=ner_analysis
    )
    
    # =========================================================================
    # Resumen final
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ EVALUACIÓN CON CONTEXTO COMPLETADA")
    print("=" * 60)
    print(f"Modo: CONTEXTUALIZADO (ENTITY + SENTENCE)")
    print(f"Total entidades evaluadas: {len(df)}")
    print(f"Documentos procesados: {doc_stats['loaded']}")
    print(f"Predicciones PII (Clase 1): {predictions.count(1)} ({predictions.count(1)/len(predictions)*100:.1f}%)")
    print(f"Predicciones Ruido (Clase 0): {predictions.count(0)} ({predictions.count(0)/len(predictions)*100:.1f}%)")
    print(f"\nArchivos generados:")
    print(f"  📄 Resultados: {results_path}")
    print(f"  📊 Métricas: {metrics_path}")
    
    return {
        "total_entities": len(df),
        "documents_loaded": doc_stats['loaded'],
        "documents_missing": doc_stats['missing'],
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
        "ner_analysis": ner_analysis,
        "results_path": str(results_path),
        "metrics_path": str(metrics_path)
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evalúa un modelo SetFit Gatekeeper sobre un dataset de entidades CON CONTEXTO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python evaluate_setfit_gatekeeper.py
  python evaluate_setfit_gatekeeper.py --model models/gatekeeper_setfit
  python evaluate_setfit_gatekeeper.py --dataset entidades.json --documents corpus/ANTIGUO/documents
  python evaluate_setfit_gatekeeper.py --dataset entidades.json --output results/

MODO CONTEXTUALIZADO:
  Este script usa el contexto del documento para clasificar cada entidad.
  Para cada entidad, construye un input del estilo:
    "ENTITY: {texto_entidad}
     SENTENCE: {frase_del_documento_donde_aparece}"
  
  Esto permite al modelo entender mejor si la entidad es PII o ruido
  basándose en el contexto donde aparece.
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Ruta al modelo SetFit (default: busca automáticamente)"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Ruta al dataset de entidades (default: busca automáticamente)"
    )
    
    parser.add_argument(
        "--documents", "-D",
        type=str,
        default=None,
        help="Ruta al directorio con documentos originales (default: corpus/ANTIGUO/documents)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    try:
        results = evaluate_setfit_gatekeeper(
            model_path=args.model,
            dataset_path=args.dataset,
            documents_path=args.documents,
            output_dir=args.output
        )
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Error de importación: {e}")
        logger.error("Instala las dependencias: pip install setfit scikit-learn pandas")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        raise
