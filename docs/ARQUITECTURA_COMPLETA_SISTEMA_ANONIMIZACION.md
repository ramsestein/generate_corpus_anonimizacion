# Arquitectura del Sistema de Anonimización y Verificación de Textos Clínicos

## Índice

1. [Visión General del Sistema](#1-visión-general-del-sistema)
2. [Arquitectura del Pipeline](#2-arquitectura-del-pipeline)
3. [Componentes Principales](#3-componentes-principales)
4. [Entrenamiento del Modelo SetFit](#4-entrenamiento-del-modelo-setfit)
5. [Ejecución del Pipeline Completo (runFullPipeline)](#5-ejecución-del-pipeline-completo-runfullpipeline)
6. [Validación y Métricas del Sistema](#6-validación-y-métricas-del-sistema)
7. [Configuración y Parámetros](#7-configuración-y-parámetros)
8. [Anexos Técnicos](#8-anexos-técnicos)

---

## 1. Visión General del Sistema

### 1.1 Propósito y Contexto

Este sistema implementa un pipeline de anonimización automática de textos clínicos en español, diseñado para cumplir con las normativas de protección de datos (RGPD/LOPD) en el ámbito sanitario. El sistema procesa informes médicos identificando y ocultando Información Personal Identificable (PII) como nombres de pacientes, profesionales sanitarios, fechas, direcciones, identificadores y otra información sensible.

El problema fundamental que resuelve este sistema es el **balance entre recall (sensibilidad) y precision**:

- **Alto recall**: Detectar TODAS las entidades PII reales (evitar fugas de privacidad)
- **Alta precision**: No anonimizar términos médicos legítimos que degradarían la utilidad clínica del documento

### 1.2 Arquitectura de Alto Nivel

El sistema emplea una arquitectura de **dos etapas**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ETAPA 1: DETECCIÓN (NER)                              │
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐                                       │
│   │  MEDDOCAN   │     │   CARMEN-I  │      Modelos Transformer-based        │
│   │  (bsc-bio   │     │  (bsc-bio   │      fine-tuned para NER en           │
│   │  -ehr-es)   │     │  -ehr-es    │      textos clínicos en español       │
│   └──────┬──────┘     └──────┬──────┘                                       │
│          │                   │                                              │
│          └─────────┬─────────┘                                              │
│                    │                                                        │
│          ┌────────▼─────────┐                                               │
│          │     ENSEMBLE     │  Unión de detecciones + deduplicación         │
│          │   (Fusión NER)   │  por coordenadas (doc_id, start, end)         │
│          └────────┬─────────┘                                               │
└───────────────────┼─────────────────────────────────────────────────────────┘
                    │
                    │  Candidatos (entidades brutas con ruido)
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ETAPA 2: FILTRADO INTELIGENTE                          │
│                                                                              │
│          ┌─────────────────────┐                                            │
│          │    SetFit Gatekeeper │   Clasificador binario few-shot           │
│          │    (PII vs RUIDO)    │   que filtra falsos positivos del NER     │
│          └──────────┬──────────┘                                            │
│                     │                                                       │
│         ┌──────────┴──────────┐                                            │
│         │                     │                                             │
│    ┌────▼────┐          ┌────▼────┐                                        │
│    │   PII   │          │  RUIDO  │                                         │
│    │ (KEEP)  │          │ (DROP)  │                                         │
│    └────┬────┘          └────┬────┘                                         │
│         │                    │                                              │
│         │            ┌───────▼──────────┐                                   │
│         │            │   LLM Judge      │  Rescate opcional de              │
│         │            │   (qwen2.5:7b)   │  entidades ambiguas               │
│         │            └───────┬──────────┘                                   │
│         │                    │                                              │
│         │         ┌──────────┴──────────┐                                   │
│         │         │                     │                                   │
│    ┌────▼────┐ ┌──▼────┐          ┌────▼────┐                              │
│    │  FINAL  │ │RESCUED│          │FILTERED │                               │
│    │   PII   │ │ (PII) │          │ (DROP)  │                               │
│    └────┬────┘ └───┬───┘          └─────────┘                               │
│         └──────────┴────────────┐                                           │
│                                 │                                           │
│                       ┌─────────▼─────────┐                                 │
│                       │  Entidades Finales │                                │
│                       │   para Anonimizar  │                                │
│                       └───────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Justificación de la Arquitectura

#### ¿Por qué un sistema de dos etapas?

Los modelos NER pre-entrenados (MEDDOCAN, CARMEN) logran **alto recall** (detectan la mayoría de entidades PII) pero presentan **precision limitada** debido a:

1. **Ambigüedad léxica**: Palabras que pueden ser nombres propios o términos médicos (ej: "Rosa" puede ser nombre o adjetivo de color)
2. **Contexto clínico especializado**: Abreviaturas, acrónimos y jerga médica confunden al NER
3. **Sobre-detección deliberada**: Los modelos NER priorizan recall por seguridad

La etapa de filtrado SetFit actúa como **gatekeeper** que discrimina entre PII real y ruido contextual.

#### ¿Por qué SetFit y no fine-tuning de BERT?

SetFit (Sentence-BERT Fine-Tuning) se eligió por las siguientes razones técnicas:

1. **Few-shot learning**: Requiere órdenes de magnitud menos ejemplos de entrenamiento (~100-1000 vs ~10000)
2. **Velocidad de entrenamiento**: Minutos vs horas de un BERT completo
3. **Eficiencia en inferencia**: Modelo compacto basado en sentence embeddings
4. **Flexibilidad para iterar**: Permite ajustar rápidamente con nuevos ejemplos de falsos negativos

El formato de entrada del SetFit está diseñado para capturar **contexto semántico**:

```
ENTITY: <texto_entidad>
SENTENCE: <oración_contexto>
```

Esto permite al modelo aprender patrones como:
- "María García" en contexto de "La Dra. María García prescribe..." → PII
- "María" en contexto de "Presenta síndrome de santa María..." → RUIDO

---

## 2. Arquitectura del Pipeline

### 2.1 Pipeline de 6 Pasos (Generación de Corpus)

El sistema incluye un pipeline de 6 pasos para **generar corpus de entrenamiento y evaluación** a partir de documentos clínicos anotados:

| Paso | Script | Función |
|------|--------|---------|
| 1 | `step1_generate_annotations.py` | Extrae entidades de documentos anotados |
| 2 | `step2_clean_jsonl.py` | Limpia y normaliza formato JSONL |
| 2.5 | `step2_5_semantic_cleaning.py` | Limpieza semántica avanzada |
| 3 | `step3_generate_documents.py` | Genera documentos estructurados |
| 4 | `step4_correct_docs.py` | Corrección automática de errores |
| 4.5 | `step4_5_clean_entity_names_enhanced.py` | Limpieza mejorada de nombres de entidades |
| 5 | `step5_ocult_and_localization.py` | Ocultación y localización de entidades |
| 6 | `step6_validation.py` | Validación del corpus generado |

### 2.2 Pipeline de Nuevos Textos (Inferencia)

Para procesar **nuevos documentos** no anotados, se utiliza el pipeline de inferencia:

```
src/pipeline-nuevos-textos/
├── run_full_pipeline.py          # Orquestador principal
├── train_gatekeeper_recall.py    # Entrenamiento de SetFit
├── setfit_module/
│   ├── gatekeeper.py             # Clasificador SetFit
│   ├── api.py                    # API de clasificación
│   ├── filters.py                # Filtros adicionales
│   └── evaluate.py               # Evaluación del modelo
├── llm_judge/
│   ├── judge.py                  # Juez LLM principal
│   ├── judge_optimized.py        # Versión optimizada
│   └── prompts.py                # Templates de prompts
├── io_json/                      # Utilidades I/O JSON
└── utils/                        # Utilidades generales
```

---

## 3. Componentes Principales

### 3.1 Modelos NER Base

#### MEDDOCAN (bsc-bio-ehr-es-meddocan)

- **Base**: BERT pre-entrenado en español biomédico (BSC)
- **Fine-tuning**: Corpus MEDDOCAN de textos clínicos
- **Etiquetas detectadas**: 29 tipos de entidades PII
- **Fortaleza**: Alta cobertura de tipos estándar (nombres, fechas, identificadores)

#### CARMEN-I (bsc-bio-ehr-es-carmen-anon)

- **Base**: BERT biomédico español
- **Fine-tuning**: Corpus CARMEN de anonimización
- **Etiquetas detectadas**: Similar a MEDDOCAN con variantes
- **Fortaleza**: Mejor en contextos clínicos específicos

#### Ensemble (Fusión)

El ensemble combina ambos modelos mediante **unión de detecciones**:

```python
def deduplicate_entities(entities: List[Entity]) -> List[Entity]:
    """
    Deduplica entidades por coordenadas (doc, start, end).
    Si se detecta la misma entidad múltiples veces, cuenta como UNA sola.
    """
    seen = {}  # key -> Entity
    for entity in entities:
        key = (entity.doc_id, entity.start, entity.end)
        if key not in seen:
            seen[key] = entity
    return list(seen.values())
```

Esta estrategia maximiza recall al incluir todas las detecciones de ambos modelos.

### 3.2 SetFit Gatekeeper

El módulo `setfit_module/gatekeeper.py` implementa la clasificación binaria PII/RUIDO:

```python
@dataclass
class ClassificationResult:
    """Resultado de clasificación SetFit."""
    is_pii: bool                     # True = PII real, False = Ruido
    confidence: float                 # Probabilidad [0,1]
    classification_method: str        # 'setfit', 'obvious_pii', 'low_confidence'
    original_label: str               # Etiqueta NER original
    entity_text: str                  # Texto de la entidad
    details: Dict[str, Any]           # Metadatos adicionales
```

#### Flujo de Clasificación

```python
def classify(self, entity_text, entity_label, sentence_context, ...):
    """
    FLUJO:
    1. (Opcional) Detectar PII obvio por regex (DNI, emails, teléfonos)
    2. Clasificar con SetFit usando formato: "ENTITY: X\nSENTENCE: Y"
    3. (Opcional) Filtrar por umbral de confianza
    """
    
    # PASO 1: Detector de PII obvio (patrones regex)
    if self.enable_pii_detector:
        is_pii, pattern = self._is_obvious_pii(entity_text, entity_label)
        if is_pii:
            return ClassificationResult(is_pii=True, confidence=1.0, 
                                        classification_method="obvious_pii", ...)
    
    # PASO 2: Clasificación SetFit
    input_text = f"ENTITY: {entity_text}\nSENTENCE: {sentence_context}"
    prediction, confidence = self._classify_with_setfit(input_text)
    
    # PASO 3: Filtro de baja confianza
    if self.enable_low_confidence_filter:
        if prediction == 1 and confidence < self.confidence_threshold:
            return ClassificationResult(is_pii=False, confidence=confidence,
                                        classification_method="low_confidence", ...)
    
    return ClassificationResult(is_pii=(prediction == 1), confidence=confidence,
                                classification_method="setfit", ...)
```

#### Configuración del Gatekeeper

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `model_path` | `models/gatekeeper_setfit` | Ruta al modelo entrenado |
| `confidence_threshold` | 0.85 | Umbral mínimo de confianza |
| `enable_pii_detector` | False | Activar detección regex previa |
| `enable_low_confidence_filter` | True | Filtrar predicciones de baja confianza |

### 3.3 LLM Judge (Rescate)

El módulo `llm_judge/judge.py` implementa un mecanismo de **rescate** para entidades ambiguas:

```python
class LLMJudge:
    """
    Evaluador de entidades usando LLM local (Ollama).
    
    Valida si una entidad clasificada como RUIDO por SetFit
    es realmente un falso positivo o debe rescatarse como PII.
    """
    
    # Tokens que indican TRUE (es PII válido)
    TRUE_TOKENS = {'TRUE', '1', 'SI', 'SÍ', 'YES', 'CORRECTO', 'VALIDO', 'VÁLIDO'}
    
    # Tokens que indican FALSE (es ruido)
    FALSE_TOKENS = {'FALSE', '0', 'NO', 'INCORRECTO', 'INVALIDO', 'INVÁLIDO', 'FALSO'}
```

#### Flujo de Evaluación LLM

1. Construir prompt con contexto y reglas de anotación
2. Llamar a Ollama con el modelo configurado (ej: `qwen2.5:7b`)
3. Parsear respuesta booleana (TRUE/FALSE)
4. Devolver `JudgeResult` con veredicto

El LLM actúa como **red de seguridad** para casos donde SetFit tiene baja confianza o el contexto es ambiguo.

---

## 4. Entrenamiento del Modelo SetFit

### 4.1 Arquitectura de SetFit

SetFit (Sentence-BERT Fine-Tuning) es un framework de few-shot learning que combina:

1. **Sentence Transformers**: Modelo pre-entrenado para embeddings de oraciones
2. **Contrastive Learning**: Entrenamiento con pares (anchor, positive, negative)
3. **Classification Head**: Capa final para clasificación binaria

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA SETFIT                           │
│                                                                  │
│   Input: "ENTITY: Juan Pérez\nSENTENCE: El paciente Juan..."    │
│                            │                                     │
│                            ▼                                     │
│   ┌────────────────────────────────────────────┐                │
│   │     Sentence Transformer (MiniLM-L12)      │                │
│   │     paraphrase-multilingual-MiniLM-L12-v2  │                │
│   └────────────────────────────────────────────┘                │
│                            │                                     │
│                            ▼                                     │
│              [Embedding 384-dimensional]                         │
│                            │                                     │
│                            ▼                                     │
│   ┌────────────────────────────────────────────┐                │
│   │      Classification Head (LogisticRegr)    │                │
│   └────────────────────────────────────────────┘                │
│                            │                                     │
│                            ▼                                     │
│               Output: P(PII) ∈ [0, 1]                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Script de Entrenamiento

El script `train_gatekeeper_recall.py` implementa el entrenamiento optimizado para recall:

```python
# Formato de entrada alineado con inferencia
def _format_input(entity_text: str, sentence_context: str) -> str:
    return f"ENTITY: {entity_text}\nSENTENCE: {sentence_context}"
```

#### Fuentes de Datos de Entrenamiento

1. **Dataset Base** (`audit/training_dataset.csv`):
   - Ejemplos sintéticos y curados manualmente
   - Balance aproximado 50/50 entre PII y RUIDO
   
2. **Hard Positives** (opcional):
   - Ejemplos de falsos negativos minados de producción
   - Se repiten N veces para dar más peso

```python
def build_training_rows(base_df, fn_examples, max_fn_examples, repeat_fn, seed):
    """
    Construye dataset de entrenamiento combinando:
    - Dataset base CSV (positivos y negativos)
    - Hard positives de análisis de FN (para mejorar recall)
    """
    rows = []
    
    # Dataset base
    for _, r in base_df.iterrows():
        entity, sentence = _row_to_entity_and_sentence(str(r["text"]))
        rows.append({
            "text": _format_input(entity, sentence),
            "label": int(r["label"]),
            "source": "base_csv"
        })
    
    # Hard positives (FN del sistema anterior)
    if fn_examples:
        for ex in fn_examples[:max_fn_examples]:
            for _ in range(repeat_fn):  # Repetir para dar más peso
                rows.append({
                    "text": _format_input(ex["entity_text"], ex["context"]),
                    "label": 1,  # Siempre PII
                    "source": "hard_positive_fn"
                })
    
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed)
```

### 4.3 Hiperparámetros de Entrenamiento

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `--base-model` | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo sentence transformer base |
| `--num-iterations` | 20 | Iteraciones de contrastive learning |
| `--num-epochs` | 1 | Épocas por iteración |
| `--learning-rate` | 3e-5 | Tasa de aprendizaje |
| `--batch-size` | 16 | Tamaño de batch |
| `--target-precision` | 0.75 | Precision mínima objetivo |

### 4.4 Calibración de Umbral

El script implementa búsqueda de umbral óptimo que maximiza recall sujeto a precision mínima:

```python
def choose_threshold(y_true, p_pos, target_precision):
    """
    Búsqueda grid de umbral óptimo.
    
    Objetivo: Maximizar RECALL manteniendo PRECISION >= target_precision
    """
    best = None
    
    for t in range(1, 100):
        threshold = t / 100.0
        tp = fp = fn = 0
        
        for yt, pp in zip(y_true, p_pos):
            yp = 1 if pp >= threshold else 0
            if yt == 1 and yp == 1: tp += 1
            elif yt == 0 and yp == 1: fp += 1
            elif yt == 1 and yp == 0: fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        
        # Preferir umbrales que cumplan precision objetivo
        if precision >= target_precision:
            if best is None or recall > best.recall:
                best = ThresholdSearchResult(threshold, precision, recall, ...)
    
    return best
```

### 4.5 Outputs del Entrenamiento

El entrenamiento genera:

```
models/gatekeeper_setfit_recall_v1/
├── config.json                 # Configuración del modelo
├── model.safetensors          # Pesos del modelo
├── special_tokens_map.json    # Tokens especiales
├── tokenizer_config.json      # Configuración tokenizer
├── tokenizer.json             # Vocabulario
└── training_metadata.json     # Metadatos y umbral recomendado
```

Ejemplo de `training_metadata.json`:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "dataset_size": 2400,
  "dataset_label_counts": {"pos": 1200, "neg": 1200},
  "hyperparameters": {
    "base_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "num_iterations": 20,
    "num_epochs": 1,
    "learning_rate": 3e-5
  },
  "recommended_threshold": {
    "target_precision": 0.75,
    "threshold": 0.72,
    "precision": 0.78,
    "recall": 0.91,
    "f1": 0.84
  }
}
```

---

## 5. Ejecución del Pipeline Completo (runFullPipeline)

### 5.1 Flujo de Ejecución

El script `run_full_pipeline.py` orquesta todo el proceso de inferencia:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         run_full_pipeline.py                             │
│                                                                          │
│   INPUT: entidades.json o detecciones_detalladas.csv                    │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │ PASO 0: Preparación                                               │  │
│   │   - Cargar documentos .txt                                        │  │
│   │   - Extraer contexto (oraciones) para cada entidad               │  │
│   │   - Token healing: reparar fronteras de entidades                │  │
│   └───────────────────────────────┬──────────────────────────────────┘  │
│                                   │                                      │
│   ┌───────────────────────────────▼──────────────────────────────────┐  │
│   │ PASO 1: Dict Filters (ELIMINADO en versión actual)               │  │
│   │   - Originalmente filtraba por listas de términos                │  │
│   │   - Ahora todo pasa directamente a SetFit                        │  │
│   └───────────────────────────────┬──────────────────────────────────┘  │
│                                   │                                      │
│   ┌───────────────────────────────▼──────────────────────────────────┐  │
│   │ PASO 2: SetFit Classification                                    │  │
│   │   - Clasificación binaria PII/RUIDO                              │  │
│   │   - PII → final_kept                                             │  │
│   │   - RUIDO → candidatos para LLM (si está habilitado)             │  │
│   └───────────────────────────────┬──────────────────────────────────┘  │
│                                   │                                      │
│   ┌───────────────────────────────▼──────────────────────────────────┐  │
│   │ PASO 3: LLM Judge (Rescate)                                      │  │
│   │   - Evalúa entidades clasificadas como RUIDO                     │  │
│   │   - KEEP → rescue a PII                                          │  │
│   │   - FILTER → descarte definitivo                                 │  │
│   └───────────────────────────────┬──────────────────────────────────┘  │
│                                   │                                      │
│   OUTPUT: resultados.json con clasificaciones finales                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Configuración por Defecto

```python
DEFAULT_CONFIG = {
    "setfit": {
        "model_path": "models/gatekeeper_setfit",
        "threshold": 0.75,
        "batch_size": 32
    },
    "llm": {
        "model": "qwen2.5:7b",
        "timeout": 120,
        "max_retries": 2
    },
    "pipeline": {
        "skip_setfit": False,
        "skip_llm": False,
        "save_intermediate": False,
        "docs_dir": None
    }
}
```

### 5.3 Token Healing

Mecanismo para reparar fronteras incorrectas de entidades (ej: tokens truncados):

```python
def apply_token_healing_to_entities(entities, documents):
    """
    Repara fronteras de entidades que quedaron mal tokenizadas.
    
    Ejemplo:
      Input:  "##ez" en "González" → mal tokenizado
      Output: "González" completo con offsets corregidos
    """
    for entity in entities:
        doc_text = documents.get(entity['doc_id'], '')
        if not doc_text:
            continue
        
        # Expandir hacia atrás si empieza con ## (subword)
        if entity['text'].startswith('##'):
            new_start = find_word_boundary(doc_text, entity['start'], direction='back')
            entity['start'] = new_start
            entity['text'] = doc_text[new_start:entity['end']]
            entity['boundary_fixed'] = True
    
    return entities
```

### 5.4 Extracción de Contexto

El contexto de cada entidad se extrae como la oración que la contiene:

```python
def add_context_to_entities(entities, documents):
    """
    Añade contexto (oración) a cada entidad para SetFit.
    """
    for entity in entities:
        doc_text = documents.get(entity['doc_id'], '')
        if doc_text:
            start = entity['start']
            end = entity['end']
            
            # Encontrar límites de oración
            sent_start = doc_text.rfind('.', 0, start) + 1
            sent_end = doc_text.find('.', end)
            if sent_end == -1:
                sent_end = len(doc_text)
            
            entity['context'] = doc_text[sent_start:sent_end].strip()
    
    return entities
```

### 5.5 Argumentos CLI

```bash
python run_full_pipeline.py \
    --input entidades.json \        # JSON de entrada con entidades
    --output resultados.json \      # JSON de salida
    --config custom_config.json \   # Configuración personalizada
    --skip-setfit \                 # Omitir clasificación SetFit
    --skip-llm \                    # Omitir rescate LLM
    --docs-dir /ruta/docs \         # Directorio de documentos .txt
    --verbose                       # Logs detallados
```

### 5.6 Estructura de Salida

```json
{
  "metadata": {
    "generated_at": "2024-01-15T10:30:00",
    "input_file": "entidades.json",
    "total_entities": 5000,
    "pii_entities": 3500,
    "ruido_entities": 1500,
    "pipeline_version": "2.0-optimized"
  },
  "stats": {
    "total_input": 5000,
    "setfit_pii": 3200,
    "setfit_ruido": 1800,
    "llm_rescued": 300,
    "llm_filtered": 1500,
    "final_output": 3500,
    "execution_time": 45.2
  },
  "decisions": [
    {
      "doc_id": "doc001",
      "text": "Juan Pérez",
      "start": 145,
      "end": 155,
      "label": "NOMBRE_SUJETO_ASISTENCIA",
      "classification": "PII",
      "classification_source": "setfit",
      "setfit_confidence": 0.92,
      "context": "El paciente Juan Pérez acude a consulta por dolor lumbar."
    }
  ]
}
```

---

## 6. Validación y Métricas del Sistema

### 6.1 Script de Comparación de Modelos

El script `model_comparision.py` implementa evaluación head-to-head de diferentes configuraciones:

```bash
python model_comparision.py \
    --gold corpus/output/aws3 \                        # Gold Standard
    --detections detecciones_detalladas.csv \         # Detecciones NER
    --setfit-a outputs/modelo_a.json \                # Predicciones modelo A
    --setfit-b outputs/modelo_b.json \                # Predicciones modelo B
    --output-dir comparison_results \                  # Directorio de salida
    --debug                                            # Modo debug
```

### 6.2 Métricas Implementadas

#### Métricas Básicas

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **Precision** | TP / (TP + FP) | % de predicciones PII que son correctas |
| **Recall** | TP / (TP + FN) | % de PII reales que fueron detectados |
| **F1** | 2·P·R / (P + R) | Media armónica de P y R |

#### Métricas de Análisis de Errores

| Métrica | Descripción |
|---------|-------------|
| `fp_basura_restante` | Ruido que el filtro NO eliminó (falsos positivos) |
| `fn_fugas_inducidas` | PII real que SetFit eliminó incorrectamente |
| `fn_no_detectado` | PII que el NER no detectó (error del ensemble) |
| `tasa_filtrado_pct` | % de entidades eliminadas por SetFit |

### 6.3 Algoritmo de Matching

El sistema usa **matching 1:1 greedy basado en overlap_ratio**:

```python
OVERLAP_THRESHOLD = 0.5  # Mínimo 50% de solapamiento

def compute_overlap_ratio(det_start, det_end, gold_start, gold_end):
    """
    Calcula ratio de solapamiento entre detección y gold.
    
    overlap_ratio = overlap_chars / min(det_len, gold_len)
    """
    overlap_chars = max(0, min(det_end, gold_end) - max(det_start, gold_start))
    det_len = det_end - det_start
    gold_len = gold_end - gold_start
    
    if min(det_len, gold_len) <= 0:
        return 0.0
    
    return overlap_chars / min(det_len, gold_len)
```

#### Proceso de Matching

```python
def calculate_metrics(predicciones, gold, ...):
    """
    Matching 1:1 greedy por documento.
    
    1. Para cada par (gold, predicción):
       - Calcular overlap_ratio
       - Si >= OVERLAP_THRESHOLD: candidato a match
    
    2. Ordenar candidatos por overlap_ratio (mayor primero)
    
    3. Asignar matches greedy (cada gold/pred solo puede matchear una vez)
    
    4. Calcular métricas:
       - TP: golds que matchearon
       - FN: golds sin match
       - FP: predicciones sin match
    """
    for doc_id in all_docs:
        doc_gold = gold[doc_id]
        doc_pred = predicciones_por_doc[doc_id]
        
        # Construir matriz de overlap
        match_candidates = []
        for g_idx, g in enumerate(doc_gold):
            for p_idx, p in enumerate(doc_pred):
                ratio = compute_overlap_ratio(p.start, p.end, g.start, g.end)
                if ratio >= OVERLAP_THRESHOLD:
                    match_candidates.append((ratio, g_idx, p_idx))
        
        # Greedy assignment
        match_candidates.sort(key=lambda x: x[0], reverse=True)
        matched_gold, matched_pred = set(), set()
        
        for ratio, g_idx, p_idx in match_candidates:
            if g_idx not in matched_gold and p_idx not in matched_pred:
                matched_gold.add(g_idx)
                matched_pred.add(p_idx)
                metrics.tp += 1
        
        # FN: golds no matcheados
        for g_idx, g in enumerate(doc_gold):
            if g_idx not in matched_gold:
                metrics.fn += 1
                # Determinar causa: ¿SetFit lo mató o NER no lo detectó?
        
        # FP: predicciones no matcheadas
        for p_idx, p in enumerate(doc_pred):
            if p_idx not in matched_pred:
                metrics.fp += 1
```

### 6.4 Análisis Delta

El sistema genera análisis comparativo entre modelos:

```markdown
## Delta Analysis: Model A vs Model B

### Precision
- Winner: Model B (+5.3pp)
- A: 72.45%, B: 77.75%

### Recall
- Winner: Model A (+2.1pp)
- A: 79.99%, B: 77.89%

### F1 Score
- Winner: Model B (+1.2pp)
- A: 0.7045, B: 0.7165

### Noise Leakage Analysis
- Noise A filtered but B kept: 234 entities
- Noise B filtered but A kept: 156 entities

### Over-Cleaning Analysis
- PII killed by A (FN induced): 1,245 entities
- PII killed by B (FN induced): 987 entities
```

### 6.5 Gold Standard

El Gold Standard se extrae de documentos con marcadores `[**...**]`:

```python
GOLD_PATTERN = re.compile(r"\[\*\*(.+?)\*\*\]")

def load_gold_standard(path: Path):
    """
    Extrae entidades gold de archivos .txt con marcadores.
    
    Los offsets se calculan en texto RAW (incluyendo marcadores):
    - inner_start = match.start() + 3  (después de '[**')
    - inner_end = match.end() - 3      (antes de '**]')
    """
    for txt_file in path.rglob("*.txt"):
        text_raw = txt_file.read_text(encoding='utf-8')
        
        for match in GOLD_PATTERN.finditer(text_raw):
            inner_text = match.group(1)
            inner_start = match.start() + 3
            inner_end = match.end() - 3
            
            yield Entity(doc_id, inner_text, inner_start, inner_end, 'gold')
```

---

## 7. Configuración y Parámetros

### 7.1 Archivos de Configuración

| Archivo | Propósito |
|---------|-----------|
| `config/setfit_original.json` | Configuración SetFit base |
| `config/setfit_recall_v1.json` | Configuración optimizada para recall |
| `audit/gatekeeper_config_recall_familiares.json` | Config especializada para familiares |

### 7.2 Variables de Entorno

```bash
# Modelo Ollama para LLM Judge
export OLLAMA_MODEL=qwen2.5:7b

# Timeouts
export LLM_TIMEOUT=120
export LLM_MAX_RETRIES=2
```

### 7.3 Umbrales Críticos

| Umbral | Valor | Impacto |
|--------|-------|---------|
| `setfit.threshold` | 0.75 | P(PII) >= threshold → clasificar como PII |
| `gatekeeper.confidence_threshold` | 0.85 | Filtrar predicciones con confianza < umbral |
| `OVERLAP_THRESHOLD` | 0.5 | Mínimo overlap para considerar match en evaluación |
| `MIN_CHAR_MATCH` | 3 | Mínimo caracteres para fallback textual |

---

## 8. Anexos Técnicos

### 8.1 Estructura del Corpus

```
corpus/output/aws3/
├── doc001.txt          # Documento con marcadores [**PII**]
├── doc002.txt
├── ...
└── metadata.json       # Metadatos del corpus
```

### 8.2 Formato de Detecciones CSV

El CSV de detecciones debe contener las columnas:
- `doc_id`: Identificador del documento
- `texto_detectado`: Texto de la entidad
- `posicion_inicio`, `posicion_fin`: Coordenadas en el texto
- `etiqueta`: Tipo de entidad PII
- `fuente`: Modelo detector (meddocan, carmen, etc.)

### 8.3 Formato de Predicciones SetFit JSON

```json
{
  "decisions": [
    {
      "doc_id": "doc001",
      "text": "Juan Pérez",
      "start": 145,
      "end": 155,
      "label": "NOMBRE_SUJETO_ASISTENCIA",
      "classification": "PII",
      "setfit_confidence": 0.92
    }
  ]
}
```

### 8.4 Tipos de Entidades PII

| Etiqueta | Descripción |
|----------|-------------|
| `NOMBRE_SUJETO_ASISTENCIA` | Nombre del paciente |
| `NOMBRE_PERSONAL_SANITARIO` | Nombre de profesional sanitario |
| `FECHAS` | Fechas (nacimiento, consulta, etc.) |
| `TERRITORIO` | Direcciones, localidades |
| `ID_SUJETO_ASISTENCIA` | Identificadores del paciente (NHC, DNI) |
| `CORREO_ELECTRONICO` | Direcciones de email |
| `NUMERO_TELEFONO` | Números de teléfono |
| `PROFESION` | Profesión del paciente |
| `SEXO_SUJETO_ASISTENCIA` | Sexo/género |
| `FAMILIARES_SUJETO_ASISTENCIA` | Relaciones familiares |

### 8.5 Dependencias del Sistema

```
# requirements.txt
transformers>=4.30.0
setfit>=1.0.0
sentence-transformers>=2.2.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
datasets>=2.14.0
```

### 8.6 Referencias

1. SetFit: Efficient Few-Shot Learning Without Prompts (Tunstall et al., 2022)
2. MEDDOCAN: Medical Document Anonymization Task
3. CARMEN: Corpus for Anonymization of Medical Records in Spanish
4. BSC Bio-EHR: Spanish Biomedical Language Models (Barcelona Supercomputing Center)

---

*Documentación generada para el Sistema de Anonimización de Textos Clínicos v2.0*
*Última actualización: Diciembre 2025*
