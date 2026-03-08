"""
Script para evaluar modelos NER sobre el conjunto de validación.
Calcula métricas detalladas y guarda resultados para comparación.

VERSIÓN REFACTORIZADA con:
- Corrección de mapeo de tokens usando word_ids()
- Validación de formato IOB2
- Batching para eficiencia
- Manejo robusto de errores GPU
- Métricas adicionales por documento
"""
import json
import argparse
import logging
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from datetime import datetime
from collections import defaultdict
import time

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import (
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score
)
from seqeval.scheme import IOB2
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convertir_a_nativo(obj):
    """Convierte valores numpy a tipos nativos de Python para JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convertir_a_nativo(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convertir_a_nativo(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convertir_a_nativo(item) for item in obj)
    else:
        return obj


def validate_iob2_sequence(labels: List[str]) -> Tuple[bool, List[str]]:
    """
    Valida que una secuencia de etiquetas siga el formato IOB2.
    
    Args:
        labels: Lista de etiquetas IOB2
        
    Returns:
        Tupla de (es_válido, lista_de_errores)
    """
    errors = []
    
    for i, label in enumerate(labels):
        # Validar formato básico
        if label != 'O' and '-' not in label:
            errors.append(f"Posición {i}: Etiqueta '{label}' no sigue formato IOB2")
            continue
            
        # Validar que I- solo aparezca después de B- o I- del mismo tipo
        if label.startswith('I-'):
            entity_type = label[2:]
            if i == 0:
                errors.append(f"Posición {i}: I-{entity_type} al inicio de secuencia")
            else:
                prev_label = labels[i-1]
                if prev_label == 'O':
                    errors.append(f"Posición {i}: I-{entity_type} después de O")
                elif prev_label.startswith('B-') or prev_label.startswith('I-'):
                    prev_type = prev_label[2:]
                    if prev_type != entity_type:
                        errors.append(f"Posición {i}: I-{entity_type} después de {prev_label}")
    
    return len(errors) == 0, errors


def extract_entities_from_iob2(labels: List[str]) -> List[Tuple[int, int, str]]:
    """
    Extrae entidades de una secuencia IOB2.
    
    Args:
        labels: Lista de etiquetas IOB2
        
    Returns:
        Lista de tuplas (inicio, fin, tipo_entidad)
    """
    entities = []
    current_entity = None
    
    for i, label in enumerate(labels):
        if label.startswith('B-'):
            # Guardar entidad anterior si existe
            if current_entity is not None:
                entities.append(current_entity)
            # Iniciar nueva entidad
            entity_type = label[2:]
            current_entity = (i, i + 1, entity_type)
        elif label.startswith('I-'):
            # Continuar entidad actual
            if current_entity is not None:
                entity_type = label[2:]
                if current_entity[2] == entity_type:
                    current_entity = (current_entity[0], i + 1, entity_type)
                else:
                    # Tipo inconsistente, guardar anterior e iniciar nueva
                    entities.append(current_entity)
                    current_entity = (i, i + 1, entity_type)
            else:
                # I- sin B- previo, tratar como nueva entidad
                entity_type = label[2:]
                current_entity = (i, i + 1, entity_type)
        else:  # 'O' o cualquier otra cosa
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
    
    # Guardar última entidad si existe
    if current_entity is not None:
        entities.append(current_entity)
    
    return entities


class SimpleNERDataset(TorchDataset):
    """Dataset simple para batching de documentos NER."""
    
    def __init__(self, documents: List[Dict]):
        self.documents = documents
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return self.documents[idx]


class NERModelEvaluator:
    """Evaluador de modelos NER con cálculo de métricas detalladas."""
    
    def __init__(self, model_path: str, validation_set_path: str, batch_size: int = 8, chunk_size: int = 400, chunk_overlap: int = 50):
        """
        Inicializa el evaluador.
        
        Args:
            model_path: Path al modelo a evaluar
            validation_set_path: Path al conjunto de validación
            batch_size: Tamaño del batch para procesamiento
            chunk_size: Tamaño máximo de chunk en palabras (default: 400)
            chunk_overlap: Solapamiento entre chunks en palabras (default: 50)
        """
        self.model_path = Path(model_path)
        self.validation_set_path = Path(validation_set_path)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Usando dispositivo: {self.device}")
        logger.info(f"Modelo: {self.model_path}")
        logger.info(f"Conjunto de validación: {self.validation_set_path}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Chunk size: {self.chunk_size} palabras (overlap: {self.chunk_overlap})")
        
        # Cargar modelo y tokenizer
        self._load_model()
        
        # Cargar conjunto de validación (simple - solo IDs)
        self._load_validation_set()
        
    def _load_model(self):
        """Carga el modelo y tokenizer."""
        logger.info("Cargando modelo y tokenizer...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                use_fast=True,
                trust_remote_code=True,
                add_prefix_space=True  # Importante para RoBERTa
            )
            logger.info("✅ Tokenizer cargado")
        except Exception as e:
            logger.warning(f"Error cargando tokenizer rápido: {e}")
            # Fallback a tokenizer sin fast
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                use_fast=False,
                trust_remote_code=True
            )
            logger.info("✅ Tokenizer cargado (modo lento)")
        
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Modelo cargado")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
        
        # Guardar mapeo del modelo (se usa para convertir predicciones)
        self.model_id2label = self.model.config.id2label
        self.model_label2id = self.model.config.label2id
        
        logger.info(f"✅ Mapeo del modelo guardado: {len(self.model_id2label)} etiquetas")
        logger.info(f"   Etiquetas del modelo: {list(self.model_id2label.values())[:10]}...")
        
        # Detectar automáticamente el ID de "no-entidad" (O)
        self._detect_non_entity_id()
    
    def _detect_non_entity_id(self):
        """
        Detecta automáticamente qué ID usa el modelo para 'no-entidad' (O).
        Envía una frase sin entidades y ve qué ID predice más frecuentemente.
        """
        test_sentence = "la camisa es roja"
        
        logger.info(f"\n🔍 Detectando ID de 'no-entidad' con frase de prueba: '{test_sentence}'")
        
        # Tokenizar
        words = test_sentence.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        )
        
        # Mover a dispositivo y predecir
        inputs = {k: v.to(self.device) for k, v in encoding.items() if k in ['input_ids', 'attention_mask']}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        # Obtener word_ids para mapear tokens a palabras
        word_ids = encoding.word_ids()
        
        # Recoger predicciones por palabra
        word_predictions_ids = []
        current_word_idx = None
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:  # Token especial
                continue
            if word_idx != current_word_idx:
                word_predictions_ids.append(predictions[token_idx])
                current_word_idx = word_idx
        
        # El ID más común es el de "no-entidad"
        from collections import Counter
        id_counts = Counter(word_predictions_ids)
        self.non_entity_id = id_counts.most_common(1)[0][0]
        self.non_entity_label = self.model_id2label.get(self.non_entity_id, 'UNKNOWN')
        
        logger.info(f"✅ ID de 'no-entidad' detectado: {self.non_entity_id} -> '{self.non_entity_label}'")
        logger.info(f"   Distribución de IDs en frase de prueba: {dict(id_counts)}")
        logger.info(f"   Palabras: {words}")
        logger.info(f"   IDs predichos: {word_predictions_ids}")
        logger.info(f"\n📊 MODO BINARIO:")
        logger.info(f"   ID {self.non_entity_id} = 'O' (no-entidad)")
        logger.info(f"   Cualquier otro ID = 'ENTITY' (entidad)")
    
    def _split_into_chunks(self, words: List[str], labels: List) -> List[Tuple[List[str], List, int, int]]:
        """
        Divide un documento largo en chunks con solapamiento.
        
        Args:
            words: Lista de palabras del documento
            labels: Lista de etiquetas (IDs o strings)
        
        Returns:
            Lista de (chunk_words, chunk_labels, start_idx, end_idx)
        """
        if len(words) <= self.chunk_size:
            # Documento corto, no dividir
            return [(words, labels, 0, len(words))]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            
            chunk_words = words[start:end]
            chunk_labels = labels[start:end]
            
            chunks.append((chunk_words, chunk_labels, start, end))
            
            # Si ya llegamos al final, salir
            if end >= len(words):
                break
            
            # Avanzar con solapamiento
            start = end - self.chunk_overlap
        
        return chunks
    
    def _merge_chunk_predictions(self, chunks_results: List[Tuple[List[str], int, int]], total_words: int) -> List[str]:
        """
        Combina predicciones de chunks solapados usando votación.
        """
        # Crear diccionario para contar votos por posición
        votes = {}  # {word_idx: {'O': count, 'B-ENTITY': count, 'I-ENTITY': count}}
        
        for preds, start_idx, end_idx in chunks_results:
            for i, pred in enumerate(preds):
                word_idx = start_idx + i
                if word_idx not in votes:
                    votes[word_idx] = {'O': 0, 'B-ENTITY': 0, 'I-ENTITY': 0}
                votes[word_idx][pred] += 1
        
        # Seleccionar predicción con más votos para cada palabra
        merged = []
        for i in range(total_words):
            if i in votes:
                # Elegir la predicción con más votos
                best_pred = max(votes[i].items(), key=lambda x: x[1])[0]
                # En caso de empate técnico o para evitar O falsos en zonas críticas
                if votes[i]['B-ENTITY'] > 0 and votes[i]['B-ENTITY'] == votes[i]['O']:
                    best_pred = 'B-ENTITY'
                merged.append(best_pred)
            else:
                merged.append('O')
        
        # Post-procesamiento: Asegurar que I-ENTITY no sea el primer token de una entidad
        final_merged = []
        for i, label in enumerate(merged):
            if label == 'I-ENTITY':
                if i == 0 or merged[i-1] == 'O':
                    final_merged.append('B-ENTITY')
                else:
                    final_merged.append('I-ENTITY')
            else:
                final_merged.append(label)
                
        return final_merged
    
    def _convert_to_binary(self, labels: List) -> List[str]:
        """
        Convierte etiquetas a formato binario (B-ENTITY, I-ENTITY vs O).
        Usa self.validation_id2label para decodificar IDs.
        """
        binary_labels = []
        for label in labels:
            label_str = 'O'
            if isinstance(label, (int, np.integer)):
                label_str = self.validation_id2label.get(int(label), 'O')
            elif isinstance(label, str):
                label_str = label
            
            if label_str.startswith('B-'):
                binary_labels.append('B-ENTITY')
            elif label_str.startswith('I-'):
                binary_labels.append('I-ENTITY')
            else:
                binary_labels.append('O')
        return binary_labels
        
    def _load_validation_set(self):
        """Carga y valida el conjunto de validación."""
        logger.info(f"Cargando conjunto de validación desde {self.validation_set_path}...")
        
        with open(self.validation_set_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validar estructura
        if not isinstance(data, list):
            raise ValueError("El archivo debe contener una lista de documentos")
        
        # CRÍTICO: Verificar formato de los datos
        if data:
            sample_doc = data[0]
            logger.info(f"Formato de documento de ejemplo:")
            logger.info(f"  - Claves: {list(sample_doc.keys())}")
            
            if 'labels' in sample_doc:
                logger.info(f"  - Tipo de labels: {type(sample_doc['labels'])}")
                logger.info(f"  - Número de labels: {len(sample_doc['labels'])}")
                logger.info(f"  - Primeras 10 labels: {sample_doc['labels'][:10]}")
                
                # Verificar alineación palabras-labels
                words = sample_doc['text'].split()
                logger.info(f"  - Número de palabras (split): {len(words)}")
                
                if len(words) != len(sample_doc['labels']):
                    logger.error(f"⚠️ CRÍTICO: Desalineación detectada!")
                    logger.error(f"    Palabras: {len(words)}, Labels: {len(sample_doc['labels'])}")
                else:
                    logger.info(f"  ✅ Alineación correcta: {len(words)} palabras = {len(sample_doc['labels'])} labels")
                
                # Si las etiquetas son numéricas, mostrar información
                if isinstance(sample_doc['labels'][0], int):
                    max_label = max(sample_doc['labels'])
                    logger.info(f"  - Rango de IDs en validation_set: 0 a {max_label}")
                    logger.info(f"  - Conversión binaria: ID 0 = 'O', otros IDs = 'B-ENTITY'")
        
        # Validar cada documento
        self.validation_data = []
        for i, doc in enumerate(data):
            required_fields = ['text', 'labels', 'id']
            if not all(field in doc for field in required_fields):
                logger.warning(f"Documento {i} sin campos requeridos {required_fields}, saltando...")
                continue
            
            # Validar que text no esté vacío
            if not doc['text'] or not doc['text'].strip():
                logger.warning(f"Documento {doc['id']}: texto vacío, saltando...")
                continue
            
            # Validar que labels sea lista
            if not isinstance(doc['labels'], list):
                logger.warning(f"Documento {doc['id']}: labels no es lista, saltando...")
                continue
            
            # CRÍTICO: Verificar alineación palabras-labels para cada documento
            words = doc['text'].split()
            if len(words) != len(doc['labels']):
                logger.warning(f"Documento {doc['id']}: {len(words)} palabras != {len(doc['labels'])} labels, saltando...")
                continue
            
            self.validation_data.append(doc)
        
        logger.info(f"✅ {len(self.validation_data)} documentos válidos cargados")
        
    def _predict_document(self, text: str, expected_num_words: Optional[int] = None) -> Tuple[List[str], Dict]:
        """
        Predice etiquetas alineadas con palabras usando is_split_into_words=True.
        
        CRÍTICO: Las etiquetas en el JSON están alineadas con text.split(),
        por lo que debemos usar la misma tokenización.
        
        Args:
            text: Texto del documento
            expected_num_words: Número esperado de palabras (para validación)
            
        Returns:
            Tupla de (etiquetas_predichas_por_palabra, metadata)
        """
        # CRÍTICO: Usar la misma tokenización que se usó para crear las etiquetas
        words = text.split()
        
        # Validar si se esperaba un número específico de palabras
        if expected_num_words is not None and len(words) != expected_num_words:
            logger.warning(f"Advertencia: Se esperaban {expected_num_words} palabras pero se obtuvieron {len(words)}")
        
        # Tokenizar usando is_split_into_words=True para mantener alineación
        try:
            encoding = self.tokenizer(
                words,  # Pasar lista de palabras
                is_split_into_words=True,  # CRÍTICO: mantener alineación con palabras
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
                return_special_tokens_mask=True
            )
        except Exception as e:
            logger.error(f"Error tokenizando texto: {e}")
            return [], {'error': str(e)}
        
        # Obtener word_ids - mapeo de cada token a su palabra original
        word_ids = encoding.word_ids()
        
        # Mover a dispositivo
        inputs = {k: v.to(self.device) for k, v in encoding.items() if k != 'special_tokens_mask'}
        special_tokens_mask = encoding['special_tokens_mask'][0].tolist()
        
        # Predicción con manejo de errores GPU
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM detectado, limpiando caché y reintentando en CPU...")
                torch.cuda.empty_cache()
                # Mover modelo temporalmente a CPU
                self.model.cpu()
                inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs_cpu)
                    predictions = torch.argmax(outputs.logits, dim=-1)[0].numpy()
                # Devolver modelo a GPU
                self.model.to(self.device)
            else:
                raise
        
        # CRÍTICO: Alinear predicciones con palabras originales usando word_ids
        # Creamos un mapeo de word_idx -> list of predictions
        predictions_by_word = defaultdict(list)
        
        for token_idx, word_idx in enumerate(word_ids):
            # Saltar tokens especiales ([CLS], [SEP], [PAD])
            if word_idx is None:
                continue
            
            # Obtener predicción del token (ID numérico)
            pred_label_id = predictions[token_idx]
            
            # CONVERTIR A BINARIO: Preservar B-/I-
            label_str = self.id2label.get(pred_label_id, 'O')
            if label_str.startswith('B-'):
                pred_label = 'B-ENTITY'
            elif label_str.startswith('I-'):
                pred_label = 'I-ENTITY'
            else:
                pred_label = 'O'
            
            predictions_by_word[word_idx].append(pred_label)
        
        # Construir lista final asegurando que hay una entrada por cada palabra original
        word_predictions = []
        for i in range(len(words)):
            preds = predictions_by_word.get(i, [])
            if preds:
                # Prioridad: B-ENTITY > I-ENTITY > O
                if 'B-ENTITY' in preds:
                    word_predictions.append('B-ENTITY')
                elif 'I-ENTITY' in preds:
                    word_predictions.append('I-ENTITY')
                else:
                    word_predictions.append('O')
            else:
                # Si no hay tokens para esta palabra (palabra ignorada por el tokenizer)
                # O si estamos fuera del rango por truncamiento
                word_predictions.append('O')
        
        # VERIFICACIÓN: debe haber exactamente una predicción por palabra
        if len(word_predictions) != len(words):
            logger.error(f"⚠️ CRÍTICO: Desalineación persistente!")
            logger.error(f"   Palabras: {len(words)}, Predicciones: {len(word_predictions)}")
        
        metadata = {
            'num_tokens': len(predictions),
            'num_words': len(words),
            'num_predictions': len(word_predictions),
            'aligned': len(word_predictions) == len(words),
            'truncated': len(words) > 510  # Aproximado, dejando espacio para [CLS] y [SEP]
        }
        
        return word_predictions, metadata
    
    def _calculate_entity_level_metrics(self, true_labels: List[str], pred_labels: List[str]) -> Dict:
        """
        Calcula métricas a nivel de documento individual usando extracción de entidades IOB2.
        
        Args:
            true_labels: Etiquetas verdaderas
            pred_labels: Etiquetas predichas
            
        Returns:
            Dict con métricas del documento
        """
        # Validar formato IOB2
        is_valid_true, errors_true = validate_iob2_sequence(true_labels)
        is_valid_pred, errors_pred = validate_iob2_sequence(pred_labels)
        
        if not is_valid_true:
            logger.warning(f"Etiquetas verdaderas con errores IOB2: {errors_true[:3]}")
        if not is_valid_pred:
            logger.debug(f"Etiquetas predichas con errores IOB2: {errors_pred[:3]}")
        
        # Extraer entidades
        true_entities = extract_entities_from_iob2(true_labels)
        pred_entities = extract_entities_from_iob2(pred_labels)
        
        # Convertir a conjuntos para comparación
        true_entities_set = set(true_entities)
        pred_entities_set = set(pred_entities)
        
        # Calcular métricas a nivel de entidad
        true_positives = len(true_entities_set & pred_entities_set)
        false_positives = len(pred_entities_set - true_entities_set)
        false_negatives = len(true_entities_set - pred_entities_set)
        
        # Métricas por tipo de entidad
        entity_types = set([e[2] for e in true_entities] + [e[2] for e in pred_entities])
        metrics_by_type = {}
        
        for entity_type in entity_types:
            true_type = set([e for e in true_entities if e[2] == entity_type])
            pred_type = set([e for e in pred_entities if e[2] == entity_type])
            
            tp = len(true_type & pred_type)
            fp = len(pred_type - true_type)
            fn = len(true_type - pred_type)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics_by_type[entity_type] = {
                'true_count': len(true_type),
                'pred_count': len(pred_type),
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        # Calcular métricas a nivel de token
        total_tokens = len(true_labels)
        correct_tokens = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Métricas generales de entidad
        entity_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        entity_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0.0
        
        return {
            'token_accuracy': token_accuracy,
            'total_tokens': total_tokens,
            'correct_tokens': correct_tokens,
            'total_true_entities': len(true_entities),
            'total_pred_entities': len(pred_entities),
            'total_correct_entities': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'entity_precision': entity_precision,
            'entity_recall': entity_recall,
            'entity_f1': entity_f1,
            'metrics_by_type': metrics_by_type,
            'iob2_valid_true': is_valid_true,
            'iob2_valid_pred': is_valid_pred
        }
    
    def evaluate(self) -> Dict:
        """
        Evalúa el modelo sobre el conjunto de validación con batching opcional.
        
        Returns:
            Dict con métricas de evaluación
        """
        logger.info("Iniciando evaluación...")
        
        all_true_labels = []
        all_pred_labels = []
        document_results = []
        
        total_inference_time = 0.0
        
        # Evaluar cada documento
        for doc_idx, doc in enumerate(tqdm(self.validation_data, desc="Evaluando documentos")):
            doc_id = doc['id']
            text = doc['text']
            true_labels_ids = doc['labels']  # Mantener como IDs numéricos
            
            # DEBUGGING: Mostrar detalles de los primeros 3 documentos
            if doc_idx < 3:
                words = text.split()
                logger.info(f"\n=== Documento {doc_idx} ({doc_id}) ===")
                logger.info(f"Texto (primeras 100 chars): {text[:100]}...")
                logger.info(f"Número de palabras (split): {len(words)}")
                logger.info(f"Número de etiquetas en JSON: {len(true_labels_ids)}")
                logger.info(f"Primeras 5 palabras: {words[:5]}")
                logger.info(f"Primeras 5 etiquetas (IDs): {true_labels_ids[:5]}")
            
            # Dividir en chunks si es necesario
            words = text.split()
            chunks = self._split_into_chunks(words, true_labels_ids)
            
            # DEBUGGING: Mostrar división en chunks
            if doc_idx < 3 and len(chunks) > 1:
                logger.info(f"   📄 Documento largo: {len(words)} palabras → {len(chunks)} chunks")
            
            # Predecir cada chunk
            start_time = time.time()
            chunk_predictions = []
            
            for chunk_words, chunk_labels, start_idx, end_idx in chunks:
                # Reconstruir texto del chunk
                chunk_text = ' '.join(chunk_words)
                chunk_preds, chunk_metadata = self._predict_document(chunk_text, expected_num_words=len(chunk_words))
                chunk_predictions.append((chunk_preds, start_idx, end_idx))
            
            # Combinar predicciones de chunks si hay múltiples
            if len(chunks) > 1:
                pred_labels = self._merge_chunk_predictions(chunk_predictions, len(words))
                metadata = {'chunked': True, 'num_chunks': len(chunks), 'chunk_size': self.chunk_size}
            else:
                pred_labels = chunk_predictions[0][0]
                metadata = chunk_metadata
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # DEBUGGING: Mostrar predicciones de los primeros 3 documentos
            if doc_idx < 3:
                logger.info(f"Número de predicciones: {len(pred_labels)}")
                logger.info(f"Primeras 5 predicciones (binarias): {pred_labels[:5]}")
                logger.info(f"Metadata: {metadata}")
            
            # Validar que ambas listas tengan elementos
            if not pred_labels or not true_labels_ids:
                logger.warning(f"Documento {doc_id}: etiquetas vacías, saltando...")
                continue
            
            # Alinear longitudes (truncar o rellenar)
            min_len = min(len(true_labels_ids), len(pred_labels))
            max_len = max(len(true_labels_ids), len(pred_labels))
            
            # Si hay diferencia de longitud, registrarlo
            length_mismatch = max_len - min_len
            
            true_labels_ids_aligned = true_labels_ids[:min_len]
            pred_labels_aligned = pred_labels[:min_len]  # Ya son binarias
            
            # Convertir IDs verdaderos a formato BINARIO (ID 0 = O, otros = B-ENTITY)
            true_labels_binary = self._convert_to_binary(true_labels_ids_aligned)
            pred_labels_binary = pred_labels_aligned  # Ya son binarias desde _predict_document
            
            # DEBUGGING: Mostrar conversión binaria para primeros documentos
            if doc_idx < 3:
                logger.info(f"\n--- Conversión a formato BINARIO ---")
                logger.info(f"IDs originales (primeras 10): {true_labels_ids_aligned[:10]}")
                logger.info(f"Binario (primeras 10):        {true_labels_binary[:10]}")
                # Mostrar estadísticas
                num_entities_orig = sum(1 for l in true_labels_ids_aligned if l != 0)
                num_entities_bin = sum(1 for l in true_labels_binary if l != 'O')
                logger.info(f"Tokens de entidad: {num_entities_orig} -> {num_entities_bin}")
            
            # Calcular métricas por documento (usando etiquetas binarias)
            doc_metrics = self._calculate_entity_level_metrics(true_labels_binary, pred_labels_binary)
            
            # Guardar resultado del documento
            document_result = {
                'document_id': doc_id,
                'text_length': len(text),
                'num_words_in_text': len(text.split()),
                'num_tokens_true': len(true_labels_ids),
                'num_tokens_pred': len(pred_labels),
                'length_mismatch': length_mismatch,
                'inference_time_seconds': inference_time,
                'truncated': metadata.get('truncated', False),
                'true_labels_original_ids': true_labels_ids_aligned,  # IDs originales
                'pred_labels_original': pred_labels_aligned,  # Ya binarias
                'true_labels': true_labels_binary,  # Binarias para métricas
                'pred_labels': pred_labels_binary,
                'metrics': doc_metrics
            }
            document_results.append(document_result)
            
            # Añadir a listas globales (BINARIAS)
            all_true_labels.append(true_labels_binary)
            all_pred_labels.append(pred_labels_binary)
        
        # Calcular métricas globales usando seqeval
        logger.info("Calculando métricas globales...")
        
        # Métricas con modo 'strict' (default - requiere match exacto de entidad completa)
        precision_strict = precision_score(all_true_labels, all_pred_labels, mode='strict', scheme=IOB2, zero_division=0)
        recall_strict = recall_score(all_true_labels, all_pred_labels, mode='strict', scheme=IOB2, zero_division=0)
        f1_strict = f1_score(all_true_labels, all_pred_labels, mode='strict', scheme=IOB2, zero_division=0)
        
        # Accuracy a nivel de token
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        
        # Reporte detallado por entidad (strict mode)
        report_strict = classification_report(
            all_true_labels,
            all_pred_labels,
            mode='strict',
            scheme=IOB2,
            output_dict=True,
            zero_division=0
        )
        
        # Aplanar etiquetas para calcular Cohen's Kappa
        flat_true = [label for doc_labels in all_true_labels for label in doc_labels]
        flat_pred = [label for doc_labels in all_pred_labels for label in doc_labels]
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(flat_true, flat_pred)
        
        # Matriz de confusión para tipos de entidades (simplificada)
        # Convertir a tipos de entidad (sin B-/I-)
        entity_types_true = [label.split('-')[-1] if label != 'O' else 'O' for label in flat_true]
        entity_types_pred = [label.split('-')[-1] if label != 'O' else 'O' for label in flat_pred]
        
        # Calcular estadísticas agregadas
        total_documents_with_entities = sum(1 for doc in document_results 
                                           if doc['metrics']['total_true_entities'] > 0)
        total_perfect_documents = sum(1 for doc in document_results 
                                     if doc['metrics']['token_accuracy'] == 1.0)
        
        # Estadísticas de tiempo
        avg_inference_time = total_inference_time / len(document_results) if document_results else 0
        
        # Compilar resultados
        results = {
            'model_name': str(self.model_path),
            'validation_set': str(self.validation_set_path),
            'num_documents': len(self.validation_data),
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'batch_size': self.batch_size,
            'metrics': {
                # Métricas principales (strict mode - match exacto de entidad)
                'precision': float(precision_strict),
                'recall': float(recall_strict),
                'f1_score': float(f1_strict),
                'accuracy': float(accuracy),
                
                # Métricas adicionales
                'cohen_kappa': float(kappa),
                
                # Estadísticas agregadas
                'total_documents': len(self.validation_data),
                'documents_evaluated': len(document_results),
                'documents_with_entities': total_documents_with_entities,
                'perfect_documents': total_perfect_documents,
                'perfect_documents_ratio': total_perfect_documents / len(document_results) if document_results else 0,
                
                # Totales de entidades
                'total_true_entities': sum(doc['metrics']['total_true_entities'] for doc in document_results),
                'total_pred_entities': sum(doc['metrics']['total_pred_entities'] for doc in document_results),
                'total_correct_entities': sum(doc['metrics']['total_correct_entities'] for doc in document_results),
                
                # Estadísticas de tiempo
                'total_inference_time_seconds': total_inference_time,
                'avg_inference_time_seconds': avg_inference_time,
                'documents_per_second': 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            },
            'detailed_report': report_strict,
            'document_results': document_results
        }
        
        return results
    
    def save_results(self, results: Dict, output_path: Path):
        """
        Guarda los resultados de la evaluación.
        
        Args:
            results: Resultados de evaluación
            output_path: Path donde guardar los resultados
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir valores numpy a tipos nativos
        results_clean = convertir_a_nativo(results)
        
        # Guardar resultados principales (sin documentos individuales)
        results_summary = {k: v for k, v in results_clean.items() if k != 'document_results'}
        
        # Agregar estadísticas de documentos
        results_summary['document_statistics'] = {
            'total': len(results_clean['document_results']),
            'with_errors': sum(1 for doc in results_clean['document_results'] 
                             if doc['metrics']['total_correct_entities'] < doc['metrics']['total_true_entities']),
            'perfect': sum(1 for doc in results_clean['document_results'] 
                          if doc['metrics']['token_accuracy'] == 1.0),
            'truncated': sum(1 for doc in results_clean['document_results'] 
                           if doc.get('truncated', False))
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Resultados principales guardados en: {output_path}")
        
        # Guardar resultados por documento en carpeta separada
        model_name = Path(results['model_name']).name
        docs_dir = output_path.parent / f"{output_path.stem}_documents"
        docs_dir.mkdir(exist_ok=True)
        
        logger.info(f"Guardando resultados por documento en: {docs_dir}")
        
        for doc_result in tqdm(results_clean['document_results'], desc="Guardando documentos"):
            doc_id = doc_result['document_id']
            doc_path = docs_dir / f"{doc_id}.json"
            
            with open(doc_path, 'w', encoding='utf-8') as f:
                json.dump(doc_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ {len(results_clean['document_results'])} resultados de documentos guardados en: {docs_dir}")
        
    def print_summary(self, results: Dict):
        """
        Imprime un resumen de los resultados.
        
        Args:
            results: Resultados de evaluación
        """
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DE EVALUACIÓN - MODO BINARIO")
        logger.info("="*60)
        logger.info(f"Modelo: {results['model_name']}")
        logger.info(f"Documentos evaluados: {results['metrics']['documents_evaluated']}/{results['num_documents']}")
        logger.info(f"Dispositivo: {results['device']}")
        logger.info(f"Batch size: {results['batch_size']}")
        logger.info("")
        logger.info("⚠️ EVALUACIÓN BINARIA: Solo detecta ENTITY (cualquier tipo) vs O")
        logger.info(f"   - Predicciones: Detectado automáticamente ID no-entidad = {self.non_entity_id} ('{self.non_entity_label}')")
        logger.info(f"     Cualquier otro ID del modelo -> ENTITY")
        logger.info("   - Etiquetas verdaderas: B-*/I-* -> B-ENTITY/I-ENTITY, O -> O")
        logger.info("   - Solo cuenta entidades MEDDOCAN (filtrado desde CSV)")
        logger.info("-"*60)
        
        # Obtener métricas agregadas
        report = results['detailed_report']
        micro_avg = report.get('micro avg', {})
        macro_avg = report.get('macro avg', {})
        weighted_avg = report.get('weighted avg', {})
        
        logger.info("MÉTRICAS GENERALES (MICRO-AVERAGE - entidad completa exacta):")
        logger.info(f"  Precision: {results['metrics']['precision']:.4f}")
        logger.info(f"  Recall:    {results['metrics']['recall']:.4f}")
        logger.info(f"  F1-Score:  {results['metrics']['f1_score']:.4f}")
        logger.info(f"  Accuracy (token-level): {results['metrics']['accuracy']:.4f}")
        logger.info(f"  Cohen's Kappa: {results['metrics']['cohen_kappa']:.4f}")
        logger.info("")
        logger.info("ESTADÍSTICAS DE ENTIDADES:")
        logger.info(f"  Entidades verdaderas (ground truth): {results['metrics']['total_true_entities']}")
        logger.info(f"  Entidades predichas: {results['metrics']['total_pred_entities']}")
        logger.info(f"  Entidades correctas: {results['metrics']['total_correct_entities']}")
        logger.info(f"  Documentos evaluados: {results['metrics']['documents_evaluated']}")
        logger.info(f"  Documentos con entidades: {results['metrics']['documents_with_entities']}")
        logger.info(f"  Documentos perfectos (100% accuracy): {results['metrics']['perfect_documents']} ({results['metrics']['perfect_documents_ratio']:.2%})")
        logger.info("")
        logger.info("ESTADÍSTICAS DE TIEMPO:")
        logger.info(f"  Tiempo total de inferencia: {results['metrics']['total_inference_time_seconds']:.2f}s")
        logger.info(f"  Tiempo promedio por documento: {results['metrics']['avg_inference_time_seconds']:.4f}s")
        logger.info(f"  Documentos por segundo: {results['metrics']['documents_per_second']:.2f}")
        logger.info("")
        logger.info("MÉTRICAS MACRO-AVERAGE (promedio simple por tipo de entidad):")
        logger.info(f"  Precision: {macro_avg.get('precision', 0):.4f}")
        logger.info(f"  Recall:    {macro_avg.get('recall', 0):.4f}")
        logger.info(f"  F1-Score:  {macro_avg.get('f1-score', 0):.4f}")
        logger.info("")
        logger.info("MÉTRICAS WEIGHTED-AVERAGE (ponderado por frecuencia):")
        logger.info(f"  Precision: {weighted_avg.get('precision', 0):.4f}")
        logger.info(f"  Recall:    {weighted_avg.get('recall', 0):.4f}")
        logger.info(f"  F1-Score:  {weighted_avg.get('f1-score', 0):.4f}")
        logger.info("-"*60)
        
        # Métricas por entidad
        logger.info("MÉTRICAS POR TIPO DE ENTIDAD:")
        
        # Filtrar solo las entidades (ignorar 'micro avg', 'macro avg', etc.)
        entity_types = [k for k in report.keys() if not k.endswith('avg') and k != 'accuracy']
        
        for entity_type in sorted(entity_types):
            metrics = report[entity_type]
            logger.info(f"\n  {entity_type}:")
            logger.info(f"    Precision: {metrics.get('precision', 0):.4f}")
            logger.info(f"    Recall:    {metrics.get('recall', 0):.4f}")
            logger.info(f"    F1-Score:  {metrics.get('f1-score', 0):.4f}")
            logger.info(f"    Support:   {metrics.get('support', 0)}")
        
        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar modelo NER sobre conjunto de validación'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path al modelo a evaluar (ej: models/bsc-bio-ehr-es-meddocan)'
    )
    parser.add_argument(
        '--validation_set',
        type=str,
        default='corpus/validation_set.json',
        help='Path al conjunto de validación (default: corpus/validation_set.json)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directorio donde guardar resultados (default: evaluation_results)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Tamaño del batch para procesamiento (default: 8)'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=400,
        help='Tamaño máximo de chunk en palabras para documentos largos (default: 400)'
    )
    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=50,
        help='Solapamiento entre chunks en palabras (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Crear nombre de archivo de salida basado en el modelo
    model_name = Path(args.model).name
    output_filename = f"evaluation_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = Path(args.output_dir) / output_filename
    
    try:
        # Inicializar evaluador
        evaluator = NERModelEvaluator(
            model_path=args.model,
            validation_set_path=args.validation_set,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Evaluar
        results = evaluator.evaluate()
        
        # Guardar resultados
        evaluator.save_results(results, output_path)
        
        # Imprimir resumen
        evaluator.print_summary(results)
        
        logger.info(f"\n✅ Evaluación completada exitosamente!")
        logger.info(f"Resultados guardados en: {output_path}")
        
    except Exception as e:
        logger.error(f"Error durante la evaluación: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return


if __name__ == "__main__":
    main()
