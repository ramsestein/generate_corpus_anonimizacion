# RESUMEN EJECUTIVO: PIPELINE JUEZ LLM - PASO 1 COMPLETADO ✅

## 🎯 OBJETIVO DEL PROYECTO

Desarrollar un sistema completo de evaluación de entidades detectadas donde un **LLM actúa como juez** para determinar la validez de cada detección, con el objetivo principal de:

1. **MAXIMIZAR RECALL** (minimizar falsos negativos)
2. Mantener precisión aceptable (evitar exceso de falsos positivos)
3. Eliminar cualquier filtrado previo de etiquetas

## ✅ LO QUE SE HA COMPLETADO (PASO 1)

### Script Implementado

**Archivo**: `src/pipeline-nuevos-textos/llm_judge_pipeline.py`

**Funcionalidades Core**:

1. **Carga completa del CSV sin filtros**
   - NO se aplican filtros de etiquetas (eliminado completamente el filtro del step 6.1)
   - Se cargan TODAS las entidades detectadas
   - Validación robusta de formato y rangos

2. **Unificación inteligente de entidades fragmentadas**
   - Detecta y fusiona entidades que fueron detectadas en múltiples fragmentos
   - Ejemplos reales unificados:
     - `"G" + "045"` → `"G045"`
     - `"Sol" + "ara" + "t" + "P" + "are" + "des"` → `"SolaratParedes"`
     - `"3/4" + "3/4" + "3/4"` → `"3/43/43/4"`
     - `"I" + "064"` → `"I064"`
   - Preserva metadata de entidades originales
   - Calcula confianza promedio

3. **Análisis estadístico completo**
   - Distribución por etiqueta
   - Distribución por modelo (CARMEN vs MEDDOCAN)
   - Distribución de confianza
   - Estadísticas de longitud de texto
   - Documentos procesados

4. **Exportación estructurada a JSON**
   - Formato limpio y consistente
   - Incluye metadata completa
   - Listo para consumo por el juez LLM

### Resultados del Test (aws2)

```
📊 DATOS PROCESADOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entidades cargadas (raw):     220
Entidades unificadas:         186
Entidades fusionadas:         62
Documentos procesados:        64

📈 DISTRIBUCIÓN POR ETIQUETA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUMERO_IDENTIF                  78 (41.9%)
FAMILIARES_SUJETO_ASISTENCIA    76 (40.9%)
FECHAS                          21 (11.3%)
INSTITUCION                      5 (2.7%)
HOSPITAL                         2 (1.1%)
ID_SUJETO_ASISTENCIA             2 (1.1%)
NOMBRE_PERSONAL_SANITARIO        2 (1.1%)

🤖 MODELOS DETECTORES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CARMEN                         107 (57.5%)
MEDDOCAN                        79 (42.5%)

⚡ NIVEL DE CONFIANZA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Muy alta (≥0.95):              186 (100%)
```

### Cómo Usar el Script

```bash
# Uso básico
python llm_judge_pipeline.py preprocess \
  --csv corpus/step6_validation/aws2-validation/detecciones_detalladas.csv \
  --output outputs/entidades_procesadas_aws2.json

# Con parámetros personalizados
python llm_judge_pipeline.py preprocess \
  --csv detecciones.csv \
  --output salida.json \
  --max-gap 10 \
  --no-same-label
```

**Parámetros**:
- `--csv`: Ruta al CSV de detecciones (requerido)
- `--output`: Ruta de salida JSON (requerido)
- `--max-gap`: Gap máximo para unificar entidades (default: 5)
- `--no-same-label`: Permitir unificación cross-label (default: solo misma etiqueta)

### Output Generado

**Archivo**: `outputs/entidades_procesadas_aws2.json`

**Estructura**:
```json
{
  "metadata": {
    "generated_at": "2025-11-18T11:42:44",
    "total_entities": 186,
    "processing_stats": { ... },
    "analysis": { ... }
  },
  "entities": [
    {
      "doc_id": "NHC102219_episodio1008744732",
      "label": "NUMERO_IDENTIF",
      "model": "CARMEN",
      "text": "G045",
      "confidence": 0.9951,
      "start": 7652,
      "end": 7656,
      "unified": true,
      "original_entities": [
        { "text": "G", "start": 7652, "end": 7653, ... },
        { "text": "045", "start": 7653, "end": 7656, ... }
      ]
    },
    ...
  ]
}
```

## 📋 LO QUE FALTA (PASOS 2-6)

### Paso 2: Carga de Etiquetas Gold
- Leer `etiquetas_anonimizacion_meddocan_carmenI.csv`
- Crear índice por documento
- Implementar funciones de matching

### Paso 3: Configuración del Juez LLM
- Seleccionar modelo (OpenAI, Claude, Llama)
- Diseñar system prompt optimizado para recall
- Crear template de user prompt
- Setup de credenciales y API

### Paso 4: Ejecución del Juez
- Extracción de contexto por entidad
- Llamadas al LLM con rate limiting
- Parsing de respuestas TRUE/FALSE
- Manejo de errores y reintentos

### Paso 5: Cálculo de Métricas
- Clasificación: TP, FP, FN, TN
- Precision, Recall, F1-Score
- Análisis por etiqueta y modelo
- Generación de reportes

### Paso 6: Experimentación con Chunking
- Variar tamaño de contexto
- Comparar configuraciones
- Identificar configuración óptima

## 🎨 DECISIONES DE DISEÑO CLAVE

### 1. Sin Filtrado Previo ❌ → ✅ Evaluación LLM

**Antes (step 6.1)**:
```python
# Filtro hardcoded que eliminaba entidades
NON_PII_LABELS = {
    "FAMILIARES_SUJETO_ASISTENCIA",
    "PROFESION",
    "OTROS_SUJETO_ASISTENCIA"
}
# Estas entidades se descartaban ANTES de cualquier evaluación
```

**Ahora (juez LLM)**:
```python
# TODAS las entidades pasan al juez LLM
# El LLM decide basándose en contexto si es PII o no
# Mayor flexibilidad y potencial para recall más alto
```

**Beneficio**: El juez puede considerar contexto que el filtro hardcoded ignora.

### 2. Unificación de Fragmentos

**Problema**: Modelos detectan entidades fragmentadas
- `"G"` detectado en posición 7652-7653
- `"045"` detectado en posición 7653-7656
- En realidad es un solo código: `"G045"`

**Solución**: Unificación automática con criterios:
- Mismo documento
- Mismo modelo
- Misma etiqueta (configurable)
- Gap pequeño entre fragmentos (≤5 caracteres por defecto)

**Resultado**: 220 entidades → 186 (62 fusionadas)

### 3. Prioridad en Recall

**Filosofía**:
- Sistema de anonimización médica
- Peor caso: Dejar datos sensibles sin anonimizar (FN) → RIESGO ALTO
- Caso aceptable: Anonimizar datos no sensibles (FP) → RIESGO BAJO

**Implementación**:
- Prompts diseñados para favorecer TRUE en caso de duda
- Sin filtros agresivos previos
- Métricas centradas en recall

## 📊 ESTRUCTURA DE DATOS

### Clase `Entity`
```python
@dataclass
class Entity:
    doc_id: str                    # ID del documento
    label: str                     # Etiqueta (NOMBRE_SUJETO_ASISTENCIA, etc.)
    model: str                     # Modelo detector (CARMEN, MEDDOCAN)
    text: str                      # Texto detectado
    confidence: float              # Confianza [0-1]
    start: int                     # Posición inicial
    end: int                       # Posición final
    unified: bool = False          # Si fue unificada
    original_entities: List[Dict]  # Fragmentos originales si unified=True
```

### Clase `ProcessingStats`
```python
@dataclass
class ProcessingStats:
    total_raw_entities: int           # Entidades cargadas del CSV
    total_unified_entities: int       # Entidades después de unificación
    entities_merged: int              # Número de entidades fusionadas
    entities_by_label: Dict[str, int] # Conteo por etiqueta
    entities_by_model: Dict[str, int] # Conteo por modelo
    documents_processed: int          # Documentos únicos
```

## 📝 PRÓXIMOS PASOS INMEDIATOS

### 1. Revisar archivo de gold labels
```bash
# Explorar estructura del CSV de etiquetas gold
head -n 20 etiquetas_anonimizacion_meddocan_carmenI.csv
```

### 2. Decidir modelo LLM
**Opciones**:
- ✅ **OpenAI GPT-4**: Mayor calidad, API estable, costo medio
- ✅ **Claude 3 Opus/Sonnet**: Excelente con instrucciones, API estable
- ⚠️ **Llama 3.1 local**: Sin costo API, requiere hardware, setup complejo

**Recomendación inicial**: Empezar con GPT-3.5-turbo o Claude Haiku para prototipado rápido y barato, luego evaluar con GPT-4 o Opus para producción.

### 3. Diseñar el prompt inicial
**Elementos críticos**:
- System: Rol del juez + lista completa de etiquetas válidas
- Instructions: Maximizar recall, favorecer TRUE en duda
- Format: Output estricto TRUE/FALSE
- Examples: Few-shot con casos claros

### 4. Implementar Paso 2 (gold labels)
```python
# Funciones a implementar
def load_gold_labels(csv_path: str) -> Dict[str, List[GoldEntity]]
def find_gold_match(detected: Entity, gold_entities: List[GoldEntity]) -> Optional[GoldEntity]
def calculate_iou(detected: Entity, gold: GoldEntity) -> float
```

### 5. Probar pipeline end-to-end
- Ejecutar con subset pequeño (10 documentos)
- Validar todos los componentes
- Medir métricas iniciales
- Iterar sobre prompts y configuración

## 📚 DOCUMENTACIÓN GENERADA

1. **`llm_judge_pipeline.py`**: Script completo del Paso 1
2. **`LLM_JUDGE_PIPELINE_DESIGN.md`**: Diseño detallado del pipeline completo
3. **`entidades_procesadas_aws2.json`**: Output de ejemplo

## 🔧 MEJORAS FUTURAS CONSIDERADAS

### Para el preprocesamiento (Paso 1)
- [ ] Carga incremental de CSVs grandes
- [ ] Paralelización de unificación por documento
- [ ] Normalización de textos (minúsculas, acentos)
- [ ] Detección de duplicados exactos

### Para el pipeline completo
- [ ] Cache de resultados del LLM (evitar llamadas duplicadas)
- [ ] Rate limiting inteligente con backoff exponencial
- [ ] Streaming de resultados (no esperar batch completo)
- [ ] Dashboard interactivo de métricas
- [ ] A/B testing automatizado de prompts
- [ ] Fine-tuning del LLM en datos del proyecto

## 🎓 LECCIONES APRENDIDAS

### 1. Importancia de la unificación
El 28% de las entidades finales (62/220) son resultado de unificación de fragmentos. Sin este paso, las métricas serían artificialmente peores por mismatch con gold standard.

### 2. Diversidad de etiquetas
Solo 7 etiquetas diferentes en las 186 entidades, con dos categorías dominantes:
- NUMERO_IDENTIF: 41.9%
- FAMILIARES_SUJETO_ASISTENCIA: 40.9%

Esto sugiere que el juez LLM deberá prestar especial atención a estas categorías.

### 3. Alta confianza de modelos
100% de entidades tienen confianza ≥0.95, lo que indica que:
- Los modelos están muy seguros de sus detecciones
- Los errores probables son falsos positivos (alta confianza pero incorrectos)
- El juez LLM deberá ser crítico incluso con detecciones de alta confianza

## 🚀 TIMELINE ESTIMADO

```
✅ Paso 1: Preprocesamiento             [COMPLETADO]
🔨 Paso 2: Gold labels                  [2-3 horas]
🔨 Paso 3: Configuración LLM            [1-2 horas]
🔨 Paso 4: Ejecución juez               [3-4 horas]
🔨 Paso 5: Métricas                     [2-3 horas]
🔨 Paso 6: Experimentación              [Variable - días/semanas]

Total estimado (sin experimentos): 10-15 horas
```

---

**Preparado por**: Claude Sonnet (GitHub Copilot)  
**Fecha**: 2025-11-18  
**Estado del proyecto**: ✅ Paso 1 completado exitosamente
