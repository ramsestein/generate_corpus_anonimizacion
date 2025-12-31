# Estructura y Funcionamiento de la Carpeta src

**Sistema de Anonimización de Textos Clínicos**  
**Documentación técnica de la estructura de código fuente**

---

## Tabla de Contenidos

1. [Visión General](#1-visión-general)
2. [Descripción de Archivos por Directorio](#2-descripción-de-archivos-por-directorio)
   - 2.1 [pipeline/ - Generación de Corpus Sintético](#21-pipeline---generación-de-corpus-sintético)
   - 2.2 [pipeline-nuevos-textos/ - Pipeline de Inferencia](#22-pipeline-nuevos-textos---pipeline-de-inferencia)
   - 2.3 [pipeline-auxiliar/ - Scripts de Soporte](#23-pipeline-auxiliar---scripts-de-soporte)
   - 2.4 [pipeline-antiguo/ - Implementaciones Legacy](#24-pipeline-antiguo---implementaciones-legacy)
   - 2.5 [scripts/ - Utilidades y Benchmarks](#25-scripts---utilidades-y-benchmarks)
3. [Flujo de Datos entre Archivos](#3-flujo-de-datos-entre-archivos)
4. [Clasificación Funcional](#4-clasificación-funcional)
5. [Resumen de Dependencias](#5-resumen-de-dependencias)
6. [Guía de Mantenimiento](#6-guía-de-mantenimiento)

---

## 1. Visión General

La carpeta `src/` contiene todo el código fuente del sistema de anonimización de textos clínicos. Su estructura responde a dos necesidades fundamentales:

1. **Generación de corpus de entrenamiento**: Pipeline de 6 pasos que genera documentos clínicos sintéticos con anotaciones de entidades PII (Personally Identifiable Information).

2. **Inferencia sobre nuevos textos**: Pipeline modular que procesa documentos reales aplicando modelos NER, clasificación SetFit y rescate mediante LLM.

### Estructura de Directorios

```
src/
├── entidades-pipeline.json          # Archivo de configuración de entidades
├── pipeline/                        # Pipeline de generación (6 pasos)
├── pipeline-nuevos-textos/          # Pipeline de inferencia en producción
│   ├── setfit_module/              # Módulo de clasificación SetFit
│   ├── llm_judge/                  # Módulo de rescate LLM
│   ├── io_json/                    # Entrada/salida de datos
│   └── utils/                      # Utilidades compartidas
├── pipeline-auxiliar/              # Scripts de entrenamiento y procesamiento
├── pipeline-antiguo/               # Versiones legacy de los steps
└── scripts/                        # Herramientas de benchmark y métricas
```

### Responsabilidades por Directorio

| Directorio | Responsabilidad Principal | Uso |
|------------|--------------------------|-----|
| `pipeline/` | Generación de corpus sintético | Entrenamiento de modelos |
| `pipeline-nuevos-textos/` | Procesamiento de documentos reales | Producción |
| `pipeline-auxiliar/` | Entrenamiento de SetFit y utilidades | Desarrollo |
| `pipeline-antiguo/` | Versiones anteriores | Referencia histórica |
| `scripts/` | Evaluación y benchmarks | Testing y optimización |

---

## 2. Descripción de Archivos por Directorio

### 2.1 pipeline/ - Generación de Corpus Sintético

Este directorio implementa el pipeline de 6 pasos para generar documentos clínicos sintéticos con entidades PII anotadas.

#### step1_generate_annotations.py
**Líneas**: 525  
**Propósito**: Genera anotaciones médicas en formato JSONL utilizando la API de DeepSeek.

**Funcionalidad**:
- Carga guías de anotación desde `guias-anotacion.json`
- Genera anotaciones sintéticas para cada categoría de entidad PII
- Utiliza técnicas de few-shot prompting para mejorar la calidad
- Implementa rate limiting y reintentos automáticos

**Entrada**: 
- `guias-anotacion.json` (reglas de anotación)
- Configuración de API (modelo, temperatura, tokens)

**Salida**: 
- Archivos JSONL en `corpus/output/step1_annotations/`

**Dependencias externas**: `openai`, `httpx`, `tqdm`

---

#### step2_clean_jsonl.py
**Líneas**: 365  
**Propósito**: Limpia los archivos JSONL eliminando duplicados y textos inválidos.

**Funcionalidad**:
- Detecta y elimina entradas duplicadas exactas
- Filtra anotaciones con formato incorrecto
- Normaliza estructura de campos JSON
- Genera estadísticas de limpieza

**Entrada**: 
- Archivos JSONL de step1

**Salida**: 
- Archivos JSONL limpios en `corpus/output/step2_cleaned/`

**Dependencias externas**: `tqdm`

---

#### step2_5_semantic_cleaning.py
**Líneas**: 509  
**Propósito**: Realiza limpieza semántica utilizando embeddings para detectar duplicados semánticos.

**Funcionalidad**:
- Genera embeddings de texto usando modelos LabSE o MiniLM
- Calcula similitud coseno entre anotaciones
- Elimina duplicados semánticos (alta similitud)
- Balancea la distribución de categorías

**Entrada**: 
- Archivos JSONL de step2

**Salida**: 
- Archivos JSONL con diversidad semántica mejorada

**Dependencias externas**: `sentence-transformers`, `numpy`, `sklearn`

---

#### step3_generate_documents.py
**Líneas**: 768  
**Propósito**: Genera documentos clínicos completos a partir de las anotaciones limpias.

**Funcionalidad**:
- Construye prompts contextualizados con anotaciones
- Genera textos clínicos coherentes mediante DeepSeek
- Inserta marcadores de entidad `[** ... **]` en posiciones correctas
- Valida coherencia entre texto generado y anotaciones

**Entrada**: 
- Anotaciones JSONL de step2/step2.5

**Salida**: 
- Documentos `.txt` en `corpus/output/step3_documents/`
- JSON con metadatos de cada documento

**Dependencias externas**: `openai`, `tiktoken`

---

#### step4_correct_docs.py
**Líneas**: 1011  
**Propósito**: Verifica y corrige iterativamente documentos con entidades faltantes.

**Funcionalidad**:
- Detecta entidades que no aparecen correctamente en el texto
- Utiliza LLM para regenerar secciones problemáticas
- Implementa hasta N iteraciones de corrección
- Registra estadísticas de correcciones aplicadas

**Entrada**: 
- Documentos de step3
- Lista de entidades esperadas por documento

**Salida**: 
- Documentos corregidos en `corpus/output/step4_corrected/`

**Dependencias externas**: `openai`

---

#### step4_5_clean_entity_names_enhanced.py
**Líneas**: 591  
**Propósito**: Limpieza avanzada de nombres de entidades con normalización contextual.

**Funcionalidad**:
- Normaliza formatos de nombres (mayúsculas, espaciado)
- Detecta y corrige errores tipográficos frecuentes
- Estandariza formatos de fechas, teléfonos y direcciones
- Valida consistencia interna de cada entidad

**Entrada**: 
- Documentos de step4

**Salida**: 
- Documentos con entidades normalizadas en `corpus/output/step4_5_cleaned/`

**Dependencias externas**: `re`, `unicodedata`

---

#### step5_ocult_and_localization.py
**Líneas**: 637  
**Propósito**: Realiza la anonimización final y registra coordenadas de entidades.

**Funcionalidad**:
- Reemplaza contenido dentro de `[** ... **]` por token `XXX`
- Calcula posiciones exactas (start, end) de cada entidad
- Genera archivo de localización con metadatos
- Preserva estructura original del documento

**Entrada**: 
- Documentos de step4.5

**Salida**: 
- Documentos anonimizados en `corpus/step5_anonymized_documents/`
- JSON con coordenadas en `corpus/step5_anonymized_documents/localizaciones/`

**Dependencias externas**: Ninguna (stdlib)

---

#### step6_validation.py
**Líneas**: 854  
**Propósito**: Valida la calidad de anonimización usando modelos NER del BSC.

**Funcionalidad**:
- Carga modelos MEDDOCAN y CARMEN
- Procesa documentos anonimizados buscando fugas
- Compara entidades detectadas contra ground truth
- Calcula métricas: Precision, Recall, F1, tasa de fuga

**Entrada**: 
- Documentos anonimizados de step5
- Ground truth de entidades originales

**Salida**: 
- Informe de validación en `corpus/step6_validation/`
- Métricas detalladas por documento y categoría

**Dependencias externas**: `transformers`, `torch`, `pandas`

---

### 2.2 pipeline-nuevos-textos/ - Pipeline de Inferencia

Este directorio contiene el pipeline de producción para procesar documentos clínicos reales.

#### run_full_pipeline.py
**Líneas**: 856  
**Propósito**: Orquestador principal que ejecuta el flujo completo de filtrado y anonimización.

**Flujo de ejecución**:
```
Input -> [SetFit] --(PII)--> Aceptar
              |               
          (Ruido)
              ↓
       [LLM Judge] --(Rescatar)--> Aceptar
              |
          (Filtrar)
              ↓
          Descartar
```

**Funcionalidad**:
- Carga entidades desde JSON o CSV
- Ejecuta clasificación SetFit (PII vs RUIDO)
- Aplica rescate LLM para entidades dudosas
- Añade trazabilidad completa a cada entidad

**Configuración por defecto**:
```python
{
    "setfit": {
        "model_path": "models/gatekeeper_setfit",
        "confidence_threshold": 0.75,
    },
    "llm": {
        "model": "qwen2.5:7b",
        "rules_path": "guias-anotacion.json",
    }
}
```

**Entrada**: 
- JSON con entidades candidatas
- (Opcional) CSV de detecciones

**Salida**: 
- JSON con entidades filtradas y clasificadas
- Estadísticas de procesamiento

**Dependencias internas**: `io_json`, `setfit_module`, `llm_judge`, `utils`

---

#### train_gatekeeper_recall.py
**Líneas**: 378  
**Propósito**: Entrena modelos SetFit optimizados para maximizar recall.

**Funcionalidad**:
- Carga dataset base de entrenamiento
- Inyecta ejemplos de falsos negativos (hard positives)
- Configura SetFit con hiperparámetros personalizados
- Calcula threshold óptimo para precisión objetivo

**Formato de entrada esperado**:
```
ENTITY: <texto_entidad>
SENTENCE: <contexto>
```

**Entrada**: 
- CSV base con columnas `text`, `label`
- JSON con análisis de falsos negativos

**Salida**: 
- Modelo SetFit en `models/gatekeeper_setfit_recall_vX/`
- `training_metadata.json` con métricas

**Dependencias externas**: `setfit`, `sentence-transformers`, `pandas`

---

#### Submódulo: setfit_module/

##### gatekeeper.py
**Líneas**: 297  
**Propósito**: Clase principal del clasificador SetFit.

**Clase `SetFitGatekeeper`**:
- Carga modelo SetFit entrenado
- Clasifica entidades como PII (1) o RUIDO (0)
- Aplica threshold configurable
- Devuelve predicción con confianza

**Métodos principales**:
```python
def predict(self, entity_text: str, context: str) -> Tuple[int, float]
def batch_predict(self, inputs: List[Dict]) -> List[Dict]
```

---

##### api.py
**Líneas**: 197  
**Propósito**: API de alto nivel para filtrado SetFit.

**Función principal**:
```python
def run_setfit_filter(entities: List[Dict], config: Dict) -> Tuple[List[Dict], List[Dict], Dict]
```

**Retorna**: (entidades_pii, entidades_ruido, estadísticas)

---

##### filters.py
**Líneas**: 261  
**Propósito**: Filtros auxiliares para clasificación temprana.

**Funcionalidad**:
- Detecta ruido obvio (números aislados, caracteres especiales)
- Identifica PII obvio (patrones de DNI, teléfonos)
- Filtra fragmentos de palabras incompletas
- Maneja entidades con baja confianza del modelo NER

---

##### evaluate.py
**Líneas**: 465  
**Propósito**: Evaluación del rendimiento del módulo SetFit.

**Métricas calculadas**:
- TP, FP, FN, TN contra ground truth
- Precision, Recall, F1, Accuracy
- Desglose por método de clasificación

---

#### Submódulo: llm_judge/

##### judge.py
**Líneas**: 328  
**Propósito**: Evaluador de entidades mediante LLM.

**Clase `LLMJudge`**:
- Conecta con Ollama para inferencia local
- Evalúa si una entidad descartada debe rescatarse
- Parsea respuestas TRUE/FALSE del modelo
- Implementa reintentos y timeouts

**Dataclass `JudgeResult`**:
```python
@dataclass
class JudgeResult:
    is_valid: Optional[bool]
    confidence: float
    raw_response: str
    status: str
    processing_time: float
```

---

##### prompts.py
**Líneas**: 429  
**Propósito**: Plantillas de prompts para el juez LLM.

**Plantillas disponibles**:
- `default`: Clasificador estándar por palabra y contexto
- `paranoid`: Auditor estricto para verificación de anonimización
- `simple`: Clasificador minimalista

**Estructura de plantilla**:
```python
{
    "name": "...",
    "version": "...",
    "system": "...",  # Prompt de sistema con {rules}, {keyword}, {context}, {label}
    "user": "..."     # Prompt de usuario
}
```

---

##### api.py
**Propósito**: API de alto nivel para el juez LLM (análogo a setfit_module/api.py).

---

#### Submódulo: io_json/

##### loaders.py
**Líneas**: 244  
**Propósito**: Funciones para cargar datos desde diversos formatos.

**Funciones principales**:
```python
def load_json(file_path: str) -> Any
def load_csv(file_path: str, delimiter: str = ',') -> List[Dict]
def load_excel(file_path: str, sheet_name: str = None) -> List[Dict]
def load_entities(file_path: str) -> List[Dict]  # Auto-detecta formato
```

---

##### savers.py
**Líneas**: 225  
**Propósito**: Funciones para guardar resultados en formato JSON estándar.

**Formato de salida**:
```json
{
    "metadata": {
        "generated_at": "2024-01-01T12:00:00",
        "total_entities": 100,
        "pipeline_version": "2.0"
    },
    "entities": [...]
}
```

---

##### converters.py
**Líneas**: 214  
**Propósito**: Conversión entre formatos de entidades.

**Formatos soportados**:
- `STANDARD`: Formato interno del pipeline
- `MEDDOCAN`: Formato de salida del modelo MEDDOCAN
- `NER_JSON`: Formato genérico NER
- `SPACY`: Formato spaCy
- `BRAT`: Formato BRAT

---

##### text_utils.py
**Propósito**: Utilidades de normalización de texto.

**Función principal**:
```python
def normalize_text(text: str) -> str:
    """Minúsculas, sin tildes, espacios normalizados."""
```

---

#### Submódulo: utils/

##### token_healing.py
**Líneas**: 316  
**Propósito**: Reparación de fronteras de entidades truncadas.

**Problema que resuelve**:
Cuando el modelo NER devuelve coordenadas que cortan palabras a la mitad:
```
Input:  start=5, end=8, text="ola" (debería ser "Hola")
Output: start=4, end=9, text="Hola"
```

**Función principal**:
```python
def fix_entity_boundaries(entity: Dict, original_text: str) -> Dict
```

---

##### csv_converter.py
**Líneas**: 312  
**Propósito**: Convierte CSV de detecciones a formato JSON del pipeline.

**Funcionalidad**:
- Lee CSV con detecciones de entidades
- Fusiona entidades contiguas del mismo documento
- Normaliza nombres de columnas heterogéneos
- Genera JSON compatible con `run_full_pipeline.py`

---

### 2.3 pipeline-auxiliar/ - Scripts de Soporte

#### train_gatekeeper_audit_optimized.py
**Líneas**: 893  
**Propósito**: Entrenamiento de SetFit con generación de datos sintéticos optimizada.

**Optimizaciones implementadas**:
- Caché global de datos Faker
- Compilación previa de patrones regex
- Batch processing de generación
- Hiperparámetros reducidos (20→15 iteraciones)

**Clase `FakeDataCache`**:
```python
class FakeDataCache:
    """Caché de datos sintéticos para evitar regeneraciones."""
    def generate_batch(self, count: int) -> List[Dict]
```

**Entrada**: 
- `guias-anotacion.json`

**Salida**: 
- Modelo SetFit entrenado
- Archivos de auditoría con métricas

---

#### unify_json_entities.py
**Líneas**: 152  
**Propósito**: Unifica múltiples archivos JSON de entidades en un diccionario maestro.

**Formato de salida**:
```json
{
    "doc_001": [...entidades...],
    "doc_002": [...entidades...],
    ...
}
```

**Entrada**: 
- Directorio con archivos JSON (por defecto: `corpus/ANTIGUO/entidades/`)

**Salida**: 
- `dataset_unificado.json`

---

#### reprocess_all_antiguo_documents.py
**Líneas**: 191  
**Propósito**: Reprocesa documentos del corpus ANTIGUO con modelos NER actualizados.

**Funcionalidad**:
- Carga modelos MEDDOCAN y CARMEN
- Procesa todos los documentos en `corpus/ANTIGUO/documents/`
- Genera CSV con todas las detecciones
- Útil para regenerar ground truth

---

### 2.4 pipeline-antiguo/ - Implementaciones Legacy

#### step5.1.py
**Líneas**: 118  
**Propósito**: Versión simplificada del paso de anonimización.

**Diferencia con step5 actual**:
- Solo reemplaza marcas `[** ... **]` por un token fijo
- No calcula ni registra coordenadas
- Diseño minimalista sin dependencias externas

---

#### step6.1.py
**Líneas**: 1043  
**Propósito**: Versión alternativa de validación con chunking configurable.

**Características**:
- Implementa chunking con overlap para textos largos
- Constantes configurables: `CHUNK_TOKEN_SIZE`, `CHUNK_TOKEN_OVERLAP`
- Incluye nota histórica sobre eliminación de filtrado de etiquetas

**Nota importante del código**:
```python
# NOTA IMPORTANTE: FILTRADO DE ETIQUETAS ELIMINADO
# ANTERIORMENTE, este script filtraba ciertas etiquetas...
# AHORA, TODAS las entidades detectadas pasan al pipeline
# sin ningún filtrado previo por tipo de etiqueta.
```

---

### 2.5 scripts/ - Utilidades y Benchmarks

#### threshold_metrics.py
**Líneas**: 436  
**Propósito**: Análisis de métricas por threshold para optimización de clasificador.

**Funcionalidad**:
- Carga CSV con métricas por threshold
- Recalcula todas las métricas desde valores básicos (TP, FP, FN)
- Compara métricas originales vs recalculadas
- Encuentra threshold óptimo según criterio (F1, Precision, Recall)

---

#### bench_step6_chunks.py
**Líneas**: 646  
**Propósito**: Benchmark de configuraciones de chunking para evaluación NER.

**Funcionalidad**:
- Genera grid de configuraciones (chunk_size × overlap)
- Evalúa cada configuración con métricas IoU
- Identifica configuración óptima para el corpus

---

#### merge_entidades.py
**Propósito**: Combina archivos JSON de entidades de `corpus/ANTIGUO/entidades/` en un único archivo.

**Salida**:
```json
{
    "summary": {"files_processed": N, "files_failed": M},
    "errors": [...],
    "combined": [...]
}
```

---

## 3. Flujo de Datos entre Archivos

### 3.1 Pipeline de Generación de Corpus

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE GENERACIÓN                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  guias-anotacion.json                                           │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ step1_generate_      │ → anotaciones.jsonl                   │
│  │ annotations.py       │   (DeepSeek API)                      │
│  └──────────────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ step2_clean_jsonl.py │ → anotaciones_limpias.jsonl           │
│  └──────────────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ step2_5_semantic_    │ → anotaciones_diversas.jsonl          │
│  │ cleaning.py          │   (embeddings)                        │
│  └──────────────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ step3_generate_      │ → documentos.txt                      │
│  │ documents.py         │   (DeepSeek API)                      │
│  └──────────────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ step4_correct_       │ → documentos_corregidos.txt           │
│  │ docs.py              │                                        │
│  └──────────────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ step4_5_clean_       │ → documentos_normalizados.txt         │
│  │ entity_names.py      │                                        │
│  └──────────────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ step5_ocult_and_     │ → documentos_anonimizados.txt         │
│  │ localization.py      │ + localizaciones.json                 │
│  └──────────────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ step6_validation.py  │ → informe_validacion.json             │
│  │ (MEDDOCAN + CARMEN)  │   + métricas                          │
│  └──────────────────────┘                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Pipeline de Inferencia (Producción)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE INFERENCIA                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  entidades.json (candidatas)                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ io_json/loaders.py   │ → entidades normalizadas              │
│  │ io_json/converters.py│                                        │
│  └──────────────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ utils/token_healing  │ → entidades con fronteras reparadas   │
│  └──────────────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐     ┌──────────────────────┐         │
│  │ setfit_module/       │     │ Ruido obvio          │         │
│  │ filters.py           │ ──► │ (descartar directo)  │         │
│  └──────────────────────┘     └──────────────────────┘         │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ setfit_module/       │                                       │
│  │ gatekeeper.py        │                                       │
│  └──────────────────────┘                                       │
│         │                                                        │
│    ┌────┴────┐                                                  │
│    │         │                                                   │
│    ▼         ▼                                                   │
│   PII      RUIDO                                                 │
│    │         │                                                   │
│    │         ▼                                                   │
│    │  ┌──────────────────────┐                                  │
│    │  │ llm_judge/judge.py   │                                  │
│    │  │ (rescate con LLM)    │                                  │
│    │  └──────────────────────┘                                  │
│    │         │                                                   │
│    │    ┌────┴────┐                                             │
│    │    │         │                                              │
│    │    ▼         ▼                                              │
│    │  Rescate   Descartar                                        │
│    │    │                                                        │
│    ▼    ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ Entidades PII        │                                       │
│  │ (clasificadas)       │                                       │
│  └──────────────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ io_json/savers.py    │ → resultados.json                     │
│  └──────────────────────┘                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Clasificación Funcional

### 4.1 Por Tipo de Operación

| Categoría | Archivos |
|-----------|----------|
| **Generación de datos** | step1, step3 (usan LLM para generar contenido) |
| **Limpieza y normalización** | step2, step2_5, step4_5, text_utils, token_healing |
| **Verificación y corrección** | step4, step6, evaluate.py |
| **Clasificación ML** | gatekeeper.py, filters.py, api.py (setfit_module) |
| **Inferencia LLM** | judge.py, prompts.py (llm_judge) |
| **I/O de datos** | loaders.py, savers.py, converters.py, csv_converter.py |
| **Entrenamiento** | train_gatekeeper_recall.py, train_gatekeeper_audit_optimized.py |
| **Evaluación y métricas** | threshold_metrics.py, bench_step6_chunks.py, evaluate.py |

### 4.2 Por Dependencia de Servicios Externos

| Servicio | Archivos que lo utilizan |
|----------|-------------------------|
| **DeepSeek API** | step1, step3, step4 |
| **Ollama (local)** | judge.py |
| **Modelos BSC (local)** | step6, reprocess_all_antiguo_documents.py |
| **SetFit (local)** | gatekeeper.py, train_*.py |
| **Ninguno** | step2, step4_5, step5, io_json/*, utils/*, scripts/* |

### 4.3 Por Criticidad en Producción

| Nivel | Archivos | Justificación |
|-------|----------|---------------|
| **Crítico** | run_full_pipeline.py, gatekeeper.py, judge.py | Núcleo del pipeline de inferencia |
| **Alto** | loaders.py, savers.py, converters.py | Interoperabilidad de datos |
| **Medio** | filters.py, token_healing.py, prompts.py | Calidad de resultados |
| **Bajo** | scripts/*, pipeline-antiguo/* | Solo desarrollo/testing |

---

## 5. Resumen de Dependencias

### 5.1 Dependencias Externas por Módulo

```
pipeline/
├── openai (httpx)       → step1, step3, step4
├── sentence-transformers → step2_5
├── transformers + torch  → step6
├── tqdm                  → todos
└── pandas                → step6

pipeline-nuevos-textos/
├── setfit                → setfit_module/
├── sentence-transformers → setfit_module/
├── subprocess (ollama)   → llm_judge/
└── pandas                → io_json/, utils/

pipeline-auxiliar/
├── setfit                → train_*.py
├── faker                 → train_gatekeeper_audit_optimized.py
├── transformers + torch  → reprocess_all_antiguo_documents.py
└── pandas                → todos
```

### 5.2 Grafo de Importaciones Internas (pipeline-nuevos-textos)

```
run_full_pipeline.py
    ├── io_json/
    │   ├── loaders.py
    │   ├── savers.py
    │   └── converters.py
    ├── setfit_module/
    │   ├── gatekeeper.py
    │   ├── api.py
    │   └── filters.py
    ├── llm_judge/
    │   ├── judge.py
    │   └── prompts.py
    └── utils/
        ├── csv_converter.py
        └── token_healing.py
```

---

## 6. Guía de Mantenimiento

### 6.1 Añadir un Nuevo Paso al Pipeline de Generación

1. Crear archivo `stepN_nombre.py` en `src/pipeline/`
2. Seguir estructura estándar con docstring descriptivo
3. Definir argumentos CLI con `argparse`
4. Implementar función `main()` con logging
5. Actualizar documentación y README

### 6.2 Modificar el Clasificador SetFit

1. Entrenar nuevo modelo con `train_gatekeeper_recall.py`
2. Guardar en `models/gatekeeper_setfit_vX/`
3. Actualizar `model_path` en configuración de `run_full_pipeline.py`
4. Ejecutar evaluación con `setfit_module/evaluate.py`

### 6.3 Añadir Nueva Plantilla de Prompts LLM

1. Editar `llm_judge/prompts.py`
2. Añadir entrada al diccionario `PROMPT_TEMPLATES`
3. Definir campos `name`, `version`, `system`, `user`
4. Probar con `--template_name=nuevo_template`

### 6.4 Extensión de Formatos de Entrada

1. Añadir loader en `io_json/loaders.py`
2. Añadir mapping en `io_json/converters.py`
3. Actualizar detección automática en `load_entities()`

---

## Anexo: Tabla de Referencia Rápida

| Archivo | Líneas | Propósito Principal |
|---------|--------|---------------------|
| step1_generate_annotations.py | 525 | Generar anotaciones con LLM |
| step2_clean_jsonl.py | 365 | Limpiar duplicados |
| step2_5_semantic_cleaning.py | 509 | Deduplicación semántica |
| step3_generate_documents.py | 768 | Generar documentos con LLM |
| step4_correct_docs.py | 1011 | Corregir entidades faltantes |
| step4_5_clean_entity_names_enhanced.py | 591 | Normalizar nombres |
| step5_ocult_and_localization.py | 637 | Anonimizar y localizar |
| step6_validation.py | 854 | Validar con NER |
| run_full_pipeline.py | 856 | Orquestador inferencia |
| gatekeeper.py | 297 | Clasificador SetFit |
| judge.py | 328 | Evaluador LLM |
| prompts.py | 429 | Plantillas de prompts |
| loaders.py | 244 | Carga de datos |
| savers.py | 225 | Guardado de resultados |
| converters.py | 214 | Conversión de formatos |
| token_healing.py | 316 | Reparar fronteras |
| train_gatekeeper_audit_optimized.py | 893 | Entrenar SetFit optimizado |

---

*Documentación generada para el proyecto de anonimización de textos clínicos.*  
*Última actualización: Enero 2025*
