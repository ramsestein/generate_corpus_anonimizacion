# PIPELINE DE EVALUACIÓN CON JUEZ LLM

## 📋 OBJETIVO PRINCIPAL

Crear un sistema de evaluación de entidades detectadas donde un **LLM actúa como juez** para determinar si cada detección es correcta (TRUE) o incorrecta (FALSE).

### Prioridades del Sistema

1. **MAXIMIZAR RECALL** (minimizar falsos negativos - FN)
   - Es CRÍTICO no dejar entidades reales sin detectar
   - El juez debe favorecer TRUE en caso de duda
   - Prioridad: no perder ninguna detección válida

2. **Mantener precisión aceptable** (evitar exceso de falsos positivos - FP)
   - No queremos marcar todo como TRUE indiscriminadamente
   - Balance: recall alto + precisión suficiente

## 🔄 ARQUITECTURA DEL PIPELINE

```
┌─────────────────────────────────────────────────────────────────┐
│                    PASO 1: PREPROCESAMIENTO                      │
│                        ✅ COMPLETADO                             │
├─────────────────────────────────────────────────────────────────┤
│  CSV Input → Carga sin filtros → Unificación → JSON Output      │
│                                                                  │
│  • NO se filtran etiquetas (eliminado filtro del step 6.1)      │
│  • Unifica entidades fragmentadas (ej: "G" + "045" → "G045")    │
│  • Preserva TODAS las detecciones originales                    │
│  • Output: entidades_procesadas.json                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              PASO 2: CARGA DE ETIQUETAS GOLD                     │
│                      🔨 POR IMPLEMENTAR                          │
├─────────────────────────────────────────────────────────────────┤
│  CSV Etiquetas → Parse → Estructuración → Gold Standard         │
│                                                                  │
│  • Cargar etiquetas_anonimizacion_meddocan_carmenI.csv          │
│  • Crear índice por doc_id para lookup rápido                   │
│  • Validar formato y completitud                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│           PASO 3: CONFIGURACIÓN DEL JUEZ LLM                     │
│                      🔨 POR IMPLEMENTAR                          │
├─────────────────────────────────────────────────────────────────┤
│  Selección modelo → Prompt engineering → Testing                │
│                                                                  │
│  OPCIONES DE MODELO:                                            │
│  • OpenAI (GPT-4, GPT-3.5-turbo)                               │
│  • Claude (Opus, Sonnet, Haiku)                                │
│  • Llama local (3.1, 3.2)                                      │
│  • Otros modelos OSS                                           │
│                                                                  │
│  PROMPT DESIGN:                                                 │
│  • System: Definición del rol del juez                         │
│  • Context: TODAS las etiquetas del dataset gold               │
│  • Instructions: Maximizar recall, favorecer TRUE en duda      │
│  • Format: Output estricto TRUE/FALSE                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              PASO 4: EJECUCIÓN DEL JUEZ LLM                      │
│                      🔨 POR IMPLEMENTAR                          │
├─────────────────────────────────────────────────────────────────┤
│  For each entity → Build context → LLM call → Parse result      │
│                                                                  │
│  ESTRATEGIA DE CONTEXTO:                                        │
│  • Extraer chunk de texto alrededor de la entidad              │
│  • Incluir N caracteres antes/después (configurable)           │
│  • Pasar etiqueta detectada + texto + gold labels              │
│  • Obtener TRUE/FALSE del LLM                                  │
│                                                                  │
│  OPTIMIZACIONES:                                                │
│  • Batch processing cuando sea posible                          │
│  • Rate limiting para APIs                                      │
│  • Caching de resultados                                        │
│  • Reintentos con backoff exponencial                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│            PASO 5: CÁLCULO DE MÉTRICAS                          │
│                      🔨 POR IMPLEMENTAR                          │
├─────────────────────────────────────────────────────────────────┤
│  Results → Classification → Metrics → Report                    │
│                                                                  │
│  MÉTRICAS PRINCIPALES:                                          │
│  • True Positives (TP): Detecciones correctas marcadas TRUE    │
│  • False Positives (FP): Detecciones incorrectas marcadas TRUE │
│  • False Negatives (FN): Detecciones correctas marcadas FALSE  │
│  • True Negatives (TN): Detecciones incorrectas marcadas FALSE │
│                                                                  │
│  • Precision = TP / (TP + FP)                                  │
│  • Recall = TP / (TP + FN)    ← MÉTRICA CRÍTICA                │
│  • F1-Score = 2 * (Precision * Recall) / (Precision + Recall) │
│                                                                  │
│  ANÁLISIS ADICIONAL:                                            │
│  • Métricas por etiqueta                                       │
│  • Métricas por modelo (CARMEN vs MEDDOCAN)                    │
│  • Análisis de confianza vs accuracy                           │
│  • Identificar patrones de error                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│        PASO 6: EXPERIMENTACIÓN CON CHUNKING                      │
│                      🔨 POR IMPLEMENTAR                          │
├─────────────────────────────────────────────────────────────────┤
│  Iterate chunk_sizes → Compare metrics → Select optimal         │
│                                                                  │
│  PARÁMETROS A EXPERIMENTAR:                                     │
│  • Chunk size: [50, 100, 200, 500, 1000] caracteres           │
│  • Context window: simétrico vs asimétrico                      │
│  • Include document metadata: yes/no                            │
│                                                                  │
│  COMPARACIÓN:                                                   │
│  • Recall por configuración                                     │
│  • Precision por configuración                                  │
│  • F1-Score por configuración                                  │
│  • Costo (tokens) por configuración                            │
│  • Tiempo de ejecución                                          │
│                                                                  │
│  OUTPUT:                                                        │
│  • Configuración óptima recomendada                            │
│  • Gráficos comparativos                                        │
│  • Trade-offs documentados                                      │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 ESTADO ACTUAL DEL PROYECTO

### ✅ PASO 1: PREPROCESAMIENTO - COMPLETADO

**Archivo**: `src/pipeline-nuevos-textos/llm_judge_pipeline.py`

**Funcionalidades implementadas**:

1. **Carga del CSV sin filtros**
   ```python
   def load_csv_detections(csv_path: str) -> Tuple[List[Entity], ProcessingStats]
   ```
   - Lee TODAS las entidades del CSV
   - NO aplica filtros de etiquetas (eliminado el filtro de step 6.1)
   - Valida formato y rangos
   - Genera estadísticas de carga

2. **Unificación de entidades fragmentadas**
   ```python
   def unify_fragmented_entities(entities: List[Entity], 
                                 max_gap: int = 5,
                                 same_label_only: bool = True)
   ```
   - Detecta entidades consecutivas que deberían ser una sola
   - Ejemplos de unificación:
     - `"G" + "045"` → `"G045"` (NUMERO_IDENTIF)
     - `"Sol" + "ara" + "t"` → `"Solarat"` (NOMBRE)
     - `"26" + "/" + "7"` → `"26/7"` (FECHAS)
   - Preserva referencias a entidades originales
   - Calcula confianza promedio

3. **Análisis estadístico**
   ```python
   def analyze_entities(entities: List[Entity]) -> Dict
   ```
   - Distribución por etiqueta
   - Distribución por modelo
   - Distribución de confianza
   - Estadísticas de longitud de texto

4. **Exportación a JSON**
   ```python
   def save_processed_entities(entities, output_path, stats, analysis)
   ```
   - Formato estructurado para siguiente paso
   - Incluye metadata completa
   - Preserva toda la información original

**Resultados del test con aws2**:
```
Total entidades cargadas: 220
Total entidades unificadas: 186 (62 fusionadas)
Documentos procesados: 64

Distribución por etiqueta:
- NUMERO_IDENTIF: 78 (41.9%)
- FAMILIARES_SUJETO_ASISTENCIA: 76 (40.9%)
- FECHAS: 21 (11.3%)
- INSTITUCION: 5 (2.7%)
- HOSPITAL: 2 (1.1%)
- ID_SUJETO_ASISTENCIA: 2 (1.1%)
- NOMBRE_PERSONAL_SANITARIO: 2 (1.1%)

Confianza: 100% de las entidades tienen confianza ≥0.95
```

**Uso del script**:
```bash
python llm_judge_pipeline.py preprocess \
  --csv corpus/step6_validation/aws2-validation/detecciones_detalladas.csv \
  --output outputs/entidades_procesadas_aws2.json \
  --max-gap 5
```

### 🔨 PASO 2: CARGA DE ETIQUETAS GOLD - POR IMPLEMENTAR

**Objetivo**: Cargar y estructurar el gold standard para comparación

**Archivo**: `etiquetas_anonimizacion_meddocan_carmenI.csv`

**Tareas pendientes**:

1. Implementar función de carga
   ```python
   def load_gold_labels(csv_path: str) -> Dict[str, List[Entity]]
   ```
   - Parsear CSV de etiquetas gold
   - Crear índice por doc_id
   - Validar formato

2. Estructura de datos
   ```python
   @dataclass
   class GoldEntity:
       doc_id: str
       label: str
       text: str
       start: int
       end: int
   ```

3. Funciones de matching
   ```python
   def find_gold_match(detected: Entity, gold_entities: List[GoldEntity]) -> Optional[GoldEntity]
   ```
   - Matching exacto (start, end, label)
   - Matching con IoU (Intersection over Union)
   - Matching fuzzy para texto

### 🔨 PASO 3: CONFIGURACIÓN DEL JUEZ LLM - POR IMPLEMENTAR

**Objetivo**: Diseñar y configurar el sistema de evaluación LLM

**Componentes principales**:

1. **Selección de modelo**
   - Comparar opciones (OpenAI, Claude, Llama)
   - Evaluar costos vs calidad
   - Setup de credenciales/API

2. **Diseño de prompts**
   
   **System Prompt (ejemplo inicial)**:
   ```
   Eres un experto evaluador de sistemas de anonimización de documentos 
   clínicos en español. Tu tarea es determinar si una entidad detectada 
   por un sistema automático es correcta o incorrecta.
   
   ETIQUETAS VÁLIDAS DEL SISTEMA:
   - NOMBRE_SUJETO_ASISTENCIA
   - NOMBRE_PERSONAL_SANITARIO
   - NUMERO_IDENTIF
   - FECHAS
   - EDAD_SUJETO_ASISTENCIA
   - HOSPITAL
   - INSTITUCION
   - CALLE
   - TERRITORIO
   - PAIS
   - ID_TITULACION_PERSONAL_SANITARIO
   - ID_ASEGURAMIENTO
   - ID_CONTACTO_ASISTENCIAL
   - ID_SUJETO_ASISTENCIA
   - CORREO_ELECTRONICO
   - NUMERO_TELEFONO
   - NUMERO_FAX
   - FAMILIARES_SUJETO_ASISTENCIA
   - PROFESION
   - OTROS_SUJETO_ASISTENCIA
   
   INSTRUCCIONES CRÍTICAS:
   1. MAXIMIZA RECALL: En caso de duda, favorece TRUE
   2. Una entidad es TRUE si corresponde genuinamente a su etiqueta
   3. Una entidad es FALSE si es un error del detector
   4. Responde ÚNICAMENTE con "TRUE" o "FALSE"
   ```
   
   **User Prompt (template)**:
   ```
   Evalúa esta detección:
   
   Documento: {doc_id}
   Etiqueta detectada: {label}
   Texto detectado: "{text}"
   Confianza del modelo: {confidence}
   
   Contexto (±{chunk_size} caracteres):
   "{context_text}"
   
   ¿Es esta una detección correcta de {label}?
   Responde solo TRUE o FALSE.
   ```

3. **Sistema de llamadas a API**
   ```python
   class LLMJudge:
       def __init__(self, model_name: str, api_key: str):
           pass
       
       def evaluate_entity(self, entity: Entity, context: str) -> bool:
           pass
       
       def batch_evaluate(self, entities: List[Entity], contexts: List[str]) -> List[bool]:
           pass
   ```

### 🔨 PASO 4: EJECUCIÓN DEL JUEZ - POR IMPLEMENTAR

**Tareas**:

1. Extracción de contexto
   ```python
   def extract_context(doc_text: str, entity: Entity, 
                      chunk_size: int = 200) -> str
   ```

2. Pipeline de evaluación
   ```python
   def evaluate_all_entities(entities: List[Entity], 
                            gold_entities: Dict,
                            llm_judge: LLMJudge,
                            chunk_size: int = 200) -> List[EvaluationResult]
   ```

3. Manejo de errores y reintentos
4. Progress tracking
5. Guardado incremental de resultados

### 🔨 PASO 5: MÉTRICAS - POR IMPLEMENTAR

**Implementar**:

1. Clasificación de resultados
   ```python
   @dataclass
   class EvaluationResult:
       entity: Entity
       llm_judgment: bool  # TRUE/FALSE del juez
       is_correct: bool    # Comparación con gold
       classification: str  # TP, FP, FN, TN
   ```

2. Cálculo de métricas
   ```python
   def calculate_metrics(results: List[EvaluationResult]) -> Metrics
   ```

3. Reportes
   - Global metrics
   - Per-label metrics
   - Per-model metrics
   - Confusion matrix
   - Error analysis

### 🔨 PASO 6: EXPERIMENTACIÓN - POR IMPLEMENTAR

**Experimentos a realizar**:

1. Variar chunk_size: [50, 100, 200, 500, 1000]
2. Comparar modelos LLM
3. A/B testing de prompts
4. Threshold tuning

**Métricas de comparación**:
- Recall (prioridad #1)
- Precision
- F1-Score
- Costo (tokens/llamadas)
- Tiempo de ejecución

## 🎯 PRÓXIMOS PASOS INMEDIATOS

### 1. Implementar carga de gold labels
- Leer `etiquetas_anonimizacion_meddocan_carmenI.csv`
- Crear estructura de datos gold
- Implementar matching functions

### 2. Elegir y configurar LLM
- Decisión: ¿OpenAI, Claude o Llama local?
- Setup de credenciales
- Testing básico de conectividad

### 3. Diseñar y probar prompts
- Crear versión inicial del system prompt
- Crear template del user prompt
- Hacer pruebas manuales con 5-10 entidades

### 4. Implementar pipeline básico
- Integrar todos los componentes
- Ejecutar en subset pequeño (10-20 documentos)
- Validar métricas

### 5. Optimizar y experimentar
- Ajustar prompts basado en resultados
- Variar chunk sizes
- Maximizar recall manteniendo precisión

## 📝 DECISIONES DE DISEÑO IMPORTANTES

### ¿Por qué NO filtrar entidades en el preprocesamiento?

**Razón**: El filtrado previo (como el del step 6.1) elimina información que el juez LLM podría evaluar correctamente.

**Ejemplo**:
- Filtro previo: "familia" → FALSE (descartado por etiqueta)
- Juez LLM: Puede analizar contexto y determinar si es PII real o no

**Beneficio**: Mayor flexibilidad y potencial para recall más alto.

### ¿Por qué priorizar RECALL sobre PRECISION?

**Contexto**: Sistema de anonimización médica donde:
- **FN (False Negative)**: Datos sensibles sin anonimizar → RIESGO ALTO
- **FP (False Positive)**: Dato no sensible anonimizado → RIESGO BAJO

**Prioridad**: Mejor anonimizar de más que dejar datos sin anonimizar.

### ¿Por qué unificar entidades fragmentadas?

**Problema**: Los modelos detectan a veces entidades en fragmentos:
- `"G"` + `"045"` en lugar de `"G045"`
- `"26"` + `"/"` + `"7"` en lugar de `"26/7"`

**Solución**: Unificación automática mejora:
- Calidad de contexto para el juez LLM
- Matching con gold standard
- Métricas finales

## 🔧 CONFIGURACIÓN Y PARÁMETROS

### Parámetros actuales (Paso 1)

```python
# Unificación de entidades
MAX_GAP = 5  # Máximo gap de caracteres entre entidades para unificar
SAME_LABEL_ONLY = True  # Solo unificar entidades con misma etiqueta
```

### Parámetros futuros (próximos pasos)

```python
# Contexto para el juez LLM
CHUNK_SIZE = 200  # Caracteres de contexto alrededor de la entidad
SYMMETRIC_CONTEXT = True  # Misma cantidad antes y después

# Configuración del juez
LLM_MODEL = "gpt-4"  # o "claude-3-opus", "llama-3.1-70b", etc.
TEMPERATURE = 0.0  # Determinístico para consistencia
MAX_RETRIES = 3  # Reintentos en caso de error

# Batch processing
BATCH_SIZE = 10  # Entidades a evaluar por llamada
MAX_WORKERS = 4  # Workers paralelos
```

## 📚 REFERENCIAS Y RECURSOS

### Datasets
- Detecciones: `corpus/step6_validation/aws2-validation/detecciones_detalladas.csv`
- Gold labels: `etiquetas_anonimizacion_meddocan_carmenI.csv`
- Documentos originales: `corpus/output/aws2/`

### Scripts relacionados
- Preprocesamiento: `src/pipeline-nuevos-textos/llm_judge_pipeline.py`
- Step 6.1 (referencia): `src/pipeline-nuevos-textos/step6.1.py`

### Papers relevantes
- MEDDOCAN: Medical Document Anonymization
- CARMEN: Clinical Assertion and Relation Extraction

---

**Última actualización**: 2025-11-18
**Estado**: Paso 1 completado, Pasos 2-6 en diseño
