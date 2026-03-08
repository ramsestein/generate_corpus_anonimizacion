# Sistema de Generación de Corpus Sintético para Entrenamiento de Modelos NER

Sistema completo para generar corpus sintético de documentos médicos en español con anotaciones de información de salud protegida (PHI) siguiendo estándares MEDDOCAN y CARMEN-I.

## 📋 Descripción

**Corpus generado**: 14,035 documentos médicos sintéticos (~21M caracteres)  
**Calidad validada**: 93.06% de anotaciones correctas (validación humana con 6 expertos)  
**Propósito**: Entrenamiento de modelos NER para detección y anonimización de PHI

## 🗂️ Estructura y Localización de Componentes

### Corpus Principal (`corpus/`)

```
corpus_v1/
├── documents/                    # 14,035 documentos originales generados
├── anonymized_documents/         # 14,035 documentos con entidades → XXX
├── entidades/                    # 14,035 JSON con metadata de entidades
├── validation_results/           # 1,300 resultados de validación automática
├── ner_dataset.json             # Dataset completo formato IOB para entrenamiento
├── train_set.json               # Dataset de entrenamiento
├── validation_set.json          # 335 documentos sintéticos para validación
└── real_validation_set.json     # 500 documentos reales Hospital Clínic
```

### Scripts de Pipeline (`src/pipeline/`)

```
pipeline/
├── step1_generate_annotations.py        # Generación de anotaciones médicas
├── step2_clean_jsonl.py                # Limpieza de datos
├── step2_5_semantic_cleaning.py        # Limpieza semántica
├── step3_generate_documents.py         # Generación de documentos
├── step4_correct_docs.py               # Corrección iterativa
├── step4_5_clean_entity_names_enhanced.py
├── step5_ocult_and_localization.py     # Anonimización (entidades → XXX)
├── step6_1_validation_text.py          # Validación automática con modelos BSC
└── step6_2_validation_entities.py      # Purificación de sobreexpresión (DeepSeek)
```

### Scripts de Análisis y Entrenamiento (`src/`)

```
src/
├── analisis_validacion.py          # Inter-rater reliability (Fleiss' Kappa, etc.)
├── analizar_deteccion_ia.py        # Detección contenido generado por IA
├── corpus_bias_analysis.py         # Análisis diversidad léxica (MATTR, MTLD)
├── weat_gender_analysis.py         # Análisis sesgo de género (WEAT)
├── train_ner_simple.py             # Fine-tuning modelos NER (BIO)
├── train_deid_bert.py              # Fine-tuning especializado deid_bert (i2b2)
└── evaluate_model.py               # Evaluador binario original
```

### Modelos (`models/`)

```
models/
├── bsc-bio-ehr-es-meddocan/        # Modelo base BSC (MEDDOCAN)
├── bsc-bio-ehr-es-carmen-anon/     # Modelo base BSC (CARMEN-I)
└── ner-meddocan-retrained/         # RECOMENDADO: Fine-tuning en corpus purificado
    └── final/
        ├── pytorch_model.bin       # Pesos del modelo
        ├── config.json             # Configuración y etiquetas
        └── tokenizer_config.json
```

### Validación Humana (`evaluation_results/results_human/`)

```
evaluation_results/results_human/
├── validacion_[evaluador].csv     # Resultados individuales
└── tabla_consolidada_validaciones.csv  # Resultados fusionados
```

### Resultados de Evaluación (`evaluation_results/`)

```
evaluation_results/
├── models_results/                # Evaluaciones de modelos NER
├── analysis_results/              # Análisis de calidad del corpus
│   ├── corpus_analysis_results.json      # Diversidad léxica
│   ├── gender_bias_weat.json             # Análisis sesgo género
│   └── visualizations/                   # Gráficos
└── results_human/                 # Resultados validación humana
```

### Configuración

```
.
├── requirements.txt               # Dependencias Python
├── api_keys                      # Claves API (no versionado)
├── etiquetas_anonimizacion_meddocan_carmenI.csv  # Mapeo etiquetas
└── venv/                         # Entorno virtual Python
```

## ⚙️ Funcionamiento del Sistema

### 1. Pipeline de Generación de Corpus (6 Pasos)

**Step 1-3**: Generación de Documentos
- Genera documentos médicos sintéticos con entidades PHI usando IA
- Limpia y normaliza el contenido
- Output: `corpus/documents/` (14,035 archivos .txt)

**Step 4-4.5**: Corrección Iterativa
- Corrige entidades faltantes mediante IA (DeepSeek)
- Valida que todas las entidades esperadas existan en el texto
- Tasa de éxito: 74.17% (3,323/4,480 documentos procesados)

**Step 5**: Anonimización y Localización
- Reemplaza entidades sensibles con `XXX`
- Genera archivos JSON con localización exacta de entidades
- Output: `corpus/anonymized_documents/` + `corpus/entidades/`

**Step 6.1**: Validación Automática Textual
- Valida anonimización con modelos BSC (MEDDOCAN + CARMEN-I)
- Elimina documentos con entidades residuales expuestas

**Step 6.2**: Purificación de Entidades (DeepSeek)
- Análisis estadístico de longitud para detectar entidades sobreexpresadas
- Limpieza in-situ en caliente de miles de strings mediante IA generativa
- Garantiza variabilidad mediante Jaccard constraint y control de recurrencias global
- Output extra: `corpus/overexpression_correction_metrics.json`

### 2. Validación de Calidad

**Validación Humana** (360 documentos, 6 evaluadores):
- Tasa de éxito: **93.06%** (335/360 documentos correctos)
- Fleiss' Kappa: 0.819 (acuerdo casi perfecto)
- Krippendorff's Alpha: 0.961 (conclusiones confiables)

**Análisis de IA** (14,035 documentos):
- 92.4% clasificados como HUMANO
- Métricas en rango humano: TTR (0.653), MATTR (0.464), Perplejidad (113.67)

**Diversidad Léxica**:
- MATTR: 0.4644 (alta diversidad léxica)
- MTLD: 116.32 (vocabulario muy rico)
- Distinct-4: 31.58% (alta variación sintáctica)

**Sesgo de Género (WEAT)**:
- Effect size: 0.207 (p=0.751, **NO significativo**)
- Ratio términos F/M: 96.2% / 3.8%

### 3. Fine-tuning de Modelos NER

**Preparación de Datos**:
```bash
# Dataset ya preparado en formato IOB
corpus/ner_dataset.json       # Corpus completo
corpus/train_set.json         # Entrenamiento
corpus/validation_set.json    # Validación sintética (335 docs)
corpus/real_validation_set.json  # Validación real (500 docs)
```

**Entrenamiento** (GPU RTX 5080):
```bash
# Modelo MEDDOCAN / CARMEN-I (Fine-tuning)
python src/train_ner_simple.py \
  --model PlanTL-GOB-ES/bsc-bio-ehr-es-meddocan \
  --output_dir models/ner-meddocan-retrained

**Evaluación**:
```bash
# Evaluación binaria (ENTITY vs O)
python src/evaluate_model.py \
  --model models/ner-meddocan-retrained/final \
  --validation_set corpus/real_validation_set.json \
  --chunk_size 300
```

### 4. Resultados de Modelos Entrenados

| Métrica | Estrategia | F1 Sintético | F1 Real | Recomendación |
|--------|-----------|--------------|---------|---------------|
| `bsc-bio-ehr-es-meddocan` | Original BSC | 76.05% | 79.20% | Baseline |
| `bsc-bio-ehr-es-carmen-anon`| Original BSC | 77.62% | 77.20% | Baseline |
| `ner-meddocan-retrained` | **Fine-tuning sobre meddocan** | 94.93% | **85.85%** | **Producción ⭐** |

### 5. Análisis de Calidad

**Scripts de análisis** (`src/`):
```bash
# Inter-rater reliability
python src/analisis_validacion.py

# Detección de contenido generado por IA
python src/analizar_deteccion_ia.py

# Diversidad léxica
python src/corpus_bias_analysis.py

# Sesgo de género
python src/weat_gender_analysis.py
```

## 🚀 Instalación y Uso

```bash
# Activar entorno virtual
.\venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar API keys
echo "tu_api_key" > api_keys

# Ejecutar pipeline completo (pasos 1-6)
python src/pipeline/step1_generate_annotations.py
# ... (ejecutar steps 2-6 en orden)

# Entrenar modelo NER descargando desde HuggingFace el oficial del BSC
python src/train_ner_simple.py --model PlanTL-GOB-ES/bsc-bio-ehr-es --output_dir models/ner-meddocan-retrained --ignore_mismatched_sizes

# Evaluar modelo
python src/evaluate_model.py --model models/ner-meddocan-retrained/final --validation_set corpus/real_validation_set.json --chunk_size 300
```

## 📊 Métricas Resumen

| Aspecto | Métrica | Valor |
|---------|---------|-------|
| **Corpus** | Documentos | 14,035 |
| | Caracteres | 20,980,851 |
| | Calidad (validación humana) | 93.06% |
| **Validación Humana** | Fleiss' Kappa | 0.819 |
| | Krippendorff's Alpha | 0.961 |
| **Detección IA** | Clasificados HUMANO | 92.4% |
| **Diversidad Léxica** | MATTR | 0.4644 |
| | MTLD | 116.32 |
| **Sesgo Género** | WEAT p-value | 0.751 (NO significativo) |
| **Mejor Modelo NER** | F1 Real (con chunks) | 85.85% |

## 📚 Referencias

- **MEDDOCAN**: Corpus de anonimización médica en español
- **CARMEN-I**: Corpus de informes clínicos
- **BSC (Barcelona Supercomputing Center)**: Modelos base
- **DeepSeek**: Modelos de generación y corrección

## 📄 Licencia

Proyecto desarrollado para investigación en generación de corpus sintético para entrenamiento de modelos NER médicos en español.

## 📞 Contacto

- **GitHub**: [@ramsestein](https://github.com/ramsestein)
