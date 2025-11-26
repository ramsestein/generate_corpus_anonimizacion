# 📦 Análisis Arquitectónico: `src/pipeline-nuevos-textos`

> **Fecha de generación:** 26 de noviembre de 2025  
> **Autor:** Análisis automatizado MLOps  
> **Versión del pipeline:** 3.0.0

---

## Índice

1. [Estructura del Directorio](#1-estructura-del-directorio)
2. [Flujo del Pipeline](#2-flujo-del-pipeline-paso-a-paso)
3. [Interfaces y Formatos de Datos](#3-interfaces-y-formatos-de-datos)
4. [Relación con Otros Módulos](#4-relación-con-otros-módulos)
5. [Parámetros, CLI y Configuración](#5-parámetros-cli-y-configuración)
6. [Resumen Final](#6-resumen-final)

---

## 1. Estructura del Directorio

| Archivo | Tipo | Función Principal |
|---------|------|-------------------|
| `llm_judge_pipeline.py` | **Script principal (Preproceso)** | Carga CSV de detecciones NER, unifica fragmentos consecutivos, exporta JSON. Entrypoint para PASO 1. |
| `llm_entity_judge.py` | **Script principal (Evaluación LLM)** | Ejecuta evaluación de entidades llamando a Ollama con `gemma3:270m`. PASO 3 del pipeline. |
| `entity_fast_filter.py` | **Módulo core** | Filtro determinista: whitelists/blacklists exactas → decisiones FORCE_ANONYMIZE / FORCE_IGNORE / ESCALATE_TO_LLM. |
| `apply_first_filter.py` | **Script ejecutable** | Aplica `EntityFastFilter` a entidades NER y genera JSON con decisiones del filtro. |
| `combine_filter_llm.py` | **Script ejecutable** | Combina decisiones del filtro + LLM → decisión final por entidad. |
| `full_pipeline_analysis.py` | **Script ejecutable** | Ejecuta pipeline completo (filtro + LLM), calcula métricas, genera informe Markdown/JSON. |
| `metricas_entidades.py` | **Script ejecutable** | Calcula TP/FP/FN comparando predicciones vs ground truth (`corpus/ANTIGUO/entidades/`). |
| `conceptual_analysis.py` | **Script análisis** | Análisis estructural sin overfitting léxico: patrones de longitud, segmentación, confusiones, etc. |
| `llm_prompts.py` | **Módulo configuración** | Plantilla de prompts para el LLM Judge + carga de reglas desde `guias-anotacion.json`. |
| `csv_list_manager.py` | **Módulo auxiliar** | Carga listas blancas/negras desde CSV/Excel (flashtext Aho-Corasick). |

---

## 2. Flujo del Pipeline (Paso a Paso)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE NUEVOS TEXTOS                              │
│                     (Validación de Entidades NER)                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              CSV de Detecciones NER
                       (doc_id, etiqueta, texto, posición)
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 1: PREPROCESAMIENTO                                                   │
│  Script: llm_judge_pipeline.py preprocess                                   │
│                                                                             │
│  • Carga CSV sin filtrar ninguna entidad                                    │
│  • Unifica fragmentos consecutivos (ej: "G" + "045" → "G045")               │
│  • Preserva columna manual_correction (nunca va al LLM)                     │
│  • Exporta JSON estructurado (entidades-procesadas.json)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         JSON con entidades unificadas
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 2: FILTRADO DETERMINISTA                                              │
│  Script: apply_first_filter.py                                              │
│  Módulo: entity_fast_filter.py                                              │
│                                                                             │
│  Para cada entidad:                                                         │
│    1. Whitelist exacta (case-sensitive) → FORCE_ANONYMIZE                   │
│    2. Blacklist exacta (case-insensitive) → FORCE_IGNORE                    │
│    3. CIE10 exacta → FORCE_IGNORE                                           │
│    4. Default → ESCALATE_TO_LLM                                             │
│                                                                             │
│  Output: filtered_results.json con decisión por entidad                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
           FORCE_ANONYMIZE     FORCE_IGNORE      ESCALATE_TO_LLM
           (→ Anonimizar)      (→ Ignorar)       (→ Validar con LLM)
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 3: EVALUACIÓN CON LLM JUDGE                                           │
│  Script: llm_entity_judge.py                                                │
│  Módulo: llm_prompts.py (plantillas y reglas)                               │
│                                                                             │
│  Solo para entidades ESCALATE_TO_LLM:                                       │
│    1. Cargar documento original                                             │
│    2. Extraer contexto (ventana ±80 chars alrededor de la entidad)          │
│    3. Cargar reglas de guias-anotacion.json                                 │
│    4. Construir prompt (system: reglas, user: palabra+contexto+etiqueta)    │
│    5. Llamar a Ollama gemma3:270m                                           │
│    6. Parsear respuesta: TRUE (anonimizar) / FALSE (ignorar)                │
│                                                                             │
│  Output: llm_entity_judgments.json                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 4: COMBINACIÓN DE DECISIONES                                          │
│  Script: combine_filter_llm.py                                              │
│                                                                             │
│  Fusiona:                                                                   │
│    • FORCE_ANONYMIZE → final_decision = TRUE                                │
│    • FORCE_IGNORE → final_decision = FALSE                                  │
│    • ESCALATE_TO_LLM → final_decision = llm_response                        │
│                                                                             │
│  Output: combined_results.json                                              │
│  Métricas: % llamadas LLM evitadas                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 5: CÁLCULO DE MÉTRICAS                                                │
│  Script: metricas_entidades.py                                              │
│                                                                             │
│  Compara predicciones vs Ground Truth (corpus/ANTIGUO/entidades/):          │
│    • Matching set-based por documento (entidades únicas)                    │
│    • TP = predicciones ∩ GT                                                 │
│    • FP = predicciones - GT                                                 │
│    • FN = GT - predicciones                                                 │
│    • Precision, Recall, F1                                                  │
│                                                                             │
│  Output: metricas_entidades.json, metricas_entidades_por_doc.csv            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 6 (OPCIONAL): ANÁLISIS DE ERRORES                                     │
│  Script: full_pipeline_analysis.py                                          │
│  Script: conceptual_analysis.py                                             │
│                                                                             │
│  • Detecta patrones de error (FP/FN por etiqueta)                           │
│  • Analiza problemas de segmentación                                        │
│  • Genera recomendaciones de mejora                                         │
│  • Informe Markdown detallado                                               │
│                                                                             │
│  Output: reports/full_pipeline_analysis.md, conceptual_analysis.md          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Diagrama de Dependencias entre Módulos

```
                    ┌──────────────────────┐
                    │   guias-anotacion.json│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    llm_prompts.py    │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│csv_list_manager│    │ llm_entity_judge│    │ conceptual_analysis │
└───────┬───────┘    └────────┬────────┘    └─────────────────────┘
        │                     │
        ▼                     │
┌───────────────────┐         │
│entity_fast_filter │◄────────┘
└───────┬───────────┘
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
┌───────────────────┐               ┌─────────────────────┐
│apply_first_filter │               │full_pipeline_analysis│
└───────┬───────────┘               └──────────┬──────────┘
        │                                      │
        ▼                                      │
┌───────────────────┐                          │
│combine_filter_llm │◄─────────────────────────┘
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│metricas_entidades │
└───────────────────┘
```

---

## 3. Interfaces y Formatos de Datos

### 3.1 Entrada del Pipeline

**CSV de detecciones NER** (`detecciones_detalladas.csv`):

```csv
doc_id;etiqueta;modelo_detector;texto_detectado;confianza;posicion_inicio;posicion_fin;Corrección manual
NHC102219;NUMERO_IDENTIF;CARMEN;G;0.95;7652;7653;
NHC102219;NUMERO_IDENTIF;CARMEN;045;0.98;7653;7656;
```

| Campo | Descripción |
|-------|-------------|
| `doc_id` | ID del documento (nombre de archivo) |
| `etiqueta` | Tipo de entidad NER (NOMBRE_SUJETO_ASISTENCIA, FECHAS, etc.) |
| `modelo_detector` | Modelo NER (CARMEN, MEDDOCAN) |
| `texto_detectado` | Texto de la entidad detectada |
| `confianza` | Score del modelo (0-1) |
| `posicion_inicio/fin` | Offsets en el documento |
| `Corrección manual` | Ground truth humano (NO se envía al LLM) |

### 3.2 Salidas Intermedias

#### JSON de entidades unificadas (`entidades-procesadas.json`)

```json
{
  "metadata": {
    "generated_at": "2025-01-01T00:00:00",
    "total_entities": 729,
    "processing_stats": {
      "total_raw_entities": 800,
      "entities_merged": 71
    }
  },
  "entities": [
    {
      "doc_id": "abc123",
      "label": "NUMERO_IDENTIF",
      "text": "G045",
      "start": 7652,
      "end": 7656,
      "confidence": 0.965,
      "model": "CARMEN",
      "unified": true
    }
  ]
}
```

#### JSON de filtrado (`filtered_results.json`)

```json
[
  {
    "document_id": "abc123",
    "entity_text": "Hospital Clínic",
    "ner_label": "HOSPITAL",
    "decision": "FORCE_ANONYMIZE",
    "start": 100,
    "end": 115
  },
  {
    "document_id": "abc123",
    "entity_text": "ibuprofeno",
    "ner_label": "MEDICATION",
    "decision": "FORCE_IGNORE",
    "start": 200,
    "end": 210
  },
  {
    "document_id": "abc123",
    "entity_text": "García",
    "ner_label": "PERSON",
    "decision": "ESCALATE_TO_LLM",
    "start": 300,
    "end": 306
  }
]
```

#### JSON de decisiones LLM (`llm_entity_judgments.json`)

```json
[
  {
    "document_id": "abc123",
    "document_path": "/path/to/docs/abc123.txt",
    "keyword": "García",
    "label": "PERSON",
    "context": "...paciente Sr. García ingresó...",
    "llm_response": "TRUE",
    "is_valid": true,
    "status": "success"
  }
]
```

### 3.3 Salidas Finales

#### JSON combinado (`combined_results.json`)

```json
[
  {
    "document_id": "abc123",
    "entity_text": "García",
    "label": "PERSON",
    "filter_decision": "ESCALATE_TO_LLM",
    "llm_decision": true,
    "final_decision": true,
    "is_valid": true
  }
]
```

#### Métricas (`metricas_entidades.json`)

```json
{
  "metadata": {
    "generated_at": "2025-11-26T10:00:00",
    "version": "3.0.0",
    "evaluation_mode": "set-based (unique entities per document)"
  },
  "global_metrics": {
    "tp": 114,
    "fp": 483,
    "fn": 296,
    "precision": 0.191,
    "recall": 0.278,
    "f1": 0.226,
    "total_gt_entities": 410,
    "total_pred_entities": 597,
    "docs_evaluated": 50
  },
  "validation": {
    "fn_lte_gt": true,
    "tp_plus_fn_eq_gt": true
  }
}
```

---

## 4. Relación con Otros Módulos

### 4.1 Listas de Filtrado

| Fuente | Tipo | Ruta | Uso |
|--------|------|------|-----|
| `hospitales.json` | Whitelist | `data/` | FORCE_ANONYMIZE si match exacto |
| `lugares.json` | Whitelist | `data/` | FORCE_ANONYMIZE si match exacto |
| `medicamentos.json` | Blacklist | `data/` | FORCE_IGNORE si match exacto |
| `patologias.json` | Blacklist | `data/` | FORCE_IGNORE si match exacto |
| `cie10.xls` | Blacklist | `LISTAS/` | Códigos médicos → FORCE_IGNORE |
| CSV/Excel en `LISTAS/` | Auto | `LISTAS/` | Cargados por `CsvListManager` |

### 4.2 Ground Truth

| Script | Fuente de GT | Descripción |
|--------|--------------|-------------|
| `metricas_entidades.py` | `corpus/ANTIGUO/entidades/*.json` | Un JSON por documento con lista de entidades anotadas |
| `full_pipeline_analysis.py` | `corpus/step6_validation/aws2-validation/detecciones_detalladas-resueltas.csv` | CSV con correcciones manuales |

#### Formato de GT (`corpus/ANTIGUO/entidades/{doc_id}.json`)

```json
{
  "id": "abc123",
  "data": [
    {"entity": "NOMBRE_SUJETO_ASISTENCIA", "text": "Juan García"},
    {"entity": "FECHAS", "text": "15/03/2023"}
  ]
}
```

### 4.3 Reglas de Anotación

El archivo `guias-anotacion.json` en la raíz del proyecto contiene las reglas por etiqueta:

```json
{
  "NOMBRE_SUJETO_ASISTENCIA": [
    "Anotar nombres propios de pacientes",
    "Incluir apellidos completos",
    "No anotar títulos como Dr., Dra., etc."
  ],
  "FECHAS": [
    "Anotar fechas completas (dd/mm/aaaa)",
    "Incluir fechas parciales si identifican al paciente"
  ],
  "NUMERO_IDENTIF": [
    "Anotar DNI, NIE, pasaporte",
    "Incluir números de historia clínica"
  ]
}
```

Estas reglas se inyectan en el prompt del LLM Judge vía `llm_prompts.py`.

---

## 5. Parámetros, CLI y Configuración

### 5.1 Scripts Ejecutables con CLI

#### `llm_judge_pipeline.py`

```bash
# Preprocesar CSV de detecciones NER
python llm_judge_pipeline.py preprocess \
    --csv detecciones.csv \
    --output entidades_procesadas.json \
    --max-gap 5
```

| Flag | Descripción | Default |
|------|-------------|---------|
| `--csv` | Ruta al CSV de detecciones | **Requerido** |
| `--output` | Ruta JSON de salida | **Requerido** |
| `--max-gap` | Gap máximo para unificar fragmentos | 5 |
| `--no-same-label` | Permitir unificación entre etiquetas diferentes | False |

---

#### `apply_first_filter.py`

```bash
# Aplicar filtro determinista
python apply_first_filter.py \
    --input outputs/test_results.json \
    --output outputs/first_filter_results.json \
    --whitelist data/hospitales.json data/lugares.json \
    --blacklist data/medicamentos.json data/patologias.json \
    --cie10 LISTAS/cie10.xls \
    --verbose
```

| Flag | Descripción | Default |
|------|-------------|---------|
| `-i, --input` | JSON de entidades NER | **Requerido** |
| `-o, --output` | JSON de salida con decisiones | **Requerido** |
| `-w, --whitelist` | Rutas a JSONs de whitelist | None |
| `-b, --blacklist` | Rutas a JSONs de blacklist | None |
| `-c, --cie10` | Archivo Excel CIE10 | None |
| `-v, --verbose` | Modo verboso | False |

---

#### `llm_entity_judge.py`

```bash
# Evaluar entidades con LLM (Ollama gemma3:270m)
python llm_entity_judge.py \
    --entities entidades.json \
    --docs corpus/output/aws2 \
    --rules-file guias-anotacion.json \
    --output results.json \
    --left-window 80 \
    --right-window 80 \
    --debug
```

| Flag | Descripción | Default |
|------|-------------|---------|
| `--entities` | JSON de entidades | **Requerido** |
| `--docs` | Carpeta base de documentos | **Requerido** |
| `--rules-file` | Archivo JSON con reglas | `guias-anotacion.json` |
| `--output` | JSON de salida | `llm_entity_judgments.json` |
| `--left-window` | Contexto izquierdo (chars) | 80 |
| `--right-window` | Contexto derecho (chars) | 80 |
| `--debug` | Imprimir prompts completos | False |

---

#### `metricas_entidades.py`

```bash
# Calcular métricas vs ground truth
python metricas_entidades.py \
    --predictions entidades-procesadas.json \
    --ground-truth corpus/ANTIGUO/entidades \
    --include-missing \
    --debug \
    --test
```

| Flag | Descripción | Default |
|------|-------------|---------|
| `-p, --predictions` | Archivo de predicciones | `entidades-procesadas-para-metricas.json` |
| `-g, --ground-truth` | Directorio de ground truth | `corpus/ANTIGUO/entidades` |
| `--include-missing` | Incluir docs sin predicciones (genera muchos FN) | False |
| `--debug` | Mostrar análisis del doc con más FN | False |
| `--test` | Ejecutar tests de verificación | False |

---

#### `full_pipeline_analysis.py`

```bash
# Ejecutar pipeline completo con análisis
python full_pipeline_analysis.py \
    --whitelist data/hospitales.json data/lugares.json \
    --blacklist data/medicamentos.json data/patologias.json \
    --cie10 LISTAS/cie10.xls
```

| Flag | Descripción | Default |
|------|-------------|---------|
| `-w, --whitelist` | Rutas a JSONs de whitelist | `data/hospitales.json`, `data/lugares.json` |
| `-b, --blacklist` | Rutas a JSONs de blacklist | `data/medicamentos.json`, `data/patologias.json` |
| `-c, --cie10` | Archivo Excel CIE10 | `LISTAS/cie10.xls` |

---

### 5.2 Ficheros de Configuración

| Archivo | Ubicación | Contenido |
|---------|-----------|-----------|
| `guias-anotacion.json` | Raíz del proyecto | Reglas de anotación por etiqueta (inyectadas en prompts LLM) |
| `data/*.json` | `data/` | Listas whitelist/blacklist en formato JSON array |
| `LISTAS/*.xls` | `LISTAS/` | CIE10 y otros catálogos médicos en Excel |

---

## 6. Resumen Final

### ¿Para qué sirve `pipeline-nuevos-textos`?

Es un **sistema de validación post-NER** que decide si cada entidad detectada debe ser anonimizada o ignorada, usando una arquitectura de **dos capas**:

1. **Filtro determinista rápido**: Decisiones instantáneas basadas en listas exactas (whitelists → anonimizar, blacklists/CIE10 → ignorar).
2. **LLM Judge**: Validación semántica con contexto para casos ambiguos que el filtro no resuelve.

### ¿En qué se diferencia del pipeline clásico?

| Aspecto | Pipeline Clásico | Pipeline Nuevos Textos |
|---------|------------------|------------------------|
| **Validación** | Solo NER | NER + Filtro + LLM Judge |
| **Escalabilidad** | Llamadas LLM masivas | ~70% resuelto sin LLM |
| **Reglas** | Hardcoded | Guías JSON externas |
| **Métricas** | Básicas | Set-based por documento |
| **Análisis** | Manual | Automático con reportes |

### Componentes Críticos vs Auxiliares

| Crítico (Core) | Auxiliar |
|----------------|----------|
| `entity_fast_filter.py` | `conceptual_analysis.py` |
| `llm_entity_judge.py` | `csv_list_manager.py` |
| `llm_prompts.py` | `combine_filter_llm.py` |
| `metricas_entidades.py` | |

### Puntos Mejorables Detectados

1. **Duplicación de lógica de carga de GT**: `metricas_entidades.py` usa `corpus/ANTIGUO/entidades/`, mientras `full_pipeline_analysis.py` usa un CSV diferente.

2. **Acoplamiento de rutas**: Muchos scripts asumen rutas hardcodeadas (PROJECT_ROOT).

3. **Falta de configuración centralizada**: No hay un `.env` o `config.yaml` que unifique parámetros.

4. **Dos formatos de entidades**: El JSON de preproceso y el de predicciones tienen estructuras ligeramente diferentes.

---

## Apéndice: Ejemplos de Ejecución Completa

### Ejecución Paso a Paso

```bash
# 1. Preprocesar CSV
python src/pipeline-nuevos-textos/llm_judge_pipeline.py preprocess \
    --csv corpus/step6_validation/aws2-validation/detecciones_detalladas.csv \
    --output entidades-procesadas.json

# 2. Aplicar filtro determinista
python src/pipeline-nuevos-textos/apply_first_filter.py \
    -i entidades-procesadas.json \
    -o outputs/filtered_results.json \
    -w data/hospitales.json data/lugares.json \
    -b data/medicamentos.json data/patologias.json \
    -c LISTAS/cie10.xls

# 3. Evaluar con LLM (solo ESCALATE_TO_LLM)
python src/pipeline-nuevos-textos/llm_entity_judge.py \
    --entities outputs/filtered_results.json \
    --docs corpus/output/aws2 \
    --output outputs/llm_judgments.json

# 4. Combinar decisiones
python src/pipeline-nuevos-textos/combine_filter_llm.py

# 5. Calcular métricas
python src/pipeline-nuevos-textos/metricas_entidades.py
```

### Ejecución con Script Único (Pipeline Completo)

```bash
python src/pipeline-nuevos-textos/full_pipeline_analysis.py \
    -w data/hospitales.json data/lugares.json \
    -b data/medicamentos.json data/patologias.json \
    -c LISTAS/cie10.xls
```

---

*Documento generado automáticamente - Pipeline de Anonimización Clínica*
