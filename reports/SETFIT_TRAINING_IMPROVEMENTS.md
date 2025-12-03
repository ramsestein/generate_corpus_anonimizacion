<!--
Archivo: reports/SETFIT_TRAINING_IMPROVEMENTS.md
Propósito: Resumir diagnóstico y plan de mejora para reentrenar SetFit
Fecha: 2025-12-03
-->

# Mejora del Entrenamiento de SetFit (PII vs RUIDO) — Actualización

Fecha: 2025-12-03

Resumen: este documento recoge el diagnóstico y un plan de acción concreto para mejorar la precisión de SetFit en el pipeline de anonimización clínica (idioma: español/catalán). El objetivo es aumentar la precisión en la detección de PII sin sacrificar demasiado el recall.

## 1. Diagnóstico

- Señales observadas:
  - Precisión global muy baja (ej.: ~7%). Muchas predicciones son FPs.
  - El NER upstream (MEDDOCAN/CARMEN) produce detecciones ruidosas y a veces concatenadas o truncadas.
  - Dataset con inconsistencias: mismos textos etiquetados distinto, contextos cortos o corruptos.

- Qué investigar inmediatamente:
  1. Distribución PII vs RUIDO en los datos de entrenamiento.
  2. Conteo de ejemplos por tipo de entidad (NOMBRE, HOSPITAL, DNI, DIRECCIÓN, etc.).
  3. Presencia de entradas mal formadas (concatenaciones duplicadas, truncados).

## 2. Mejoras del dataset y features

### 2.1 Reequilibrio

- Objetivo: ratio PII:RUIDO cercano a 1:1.5 (ligeramente más RUIDO) para favorecer precisión.
- Si hay exceso de PII, añadir RUIDO difícil; si hay exceso de RUIDO, submuestrear manteniendo diversidad.

### 2.2 Hard negatives (RUIDO difícil)

- Extraer: predicciones del NER que fueron marcadas como FP por humanos y con alta confianza del NER (>0.8).
- Categorías prioritarias: epónimos, topónimos ambiguos, códigos truncados (HC, B12), nombres genéricos de centros, fragmentos de URLs.

### 2.3 Enriquecimiento de la entrada (features)

- Usar entidad + contexto marcado y tipo NER. Template recomendado:

```text
[NER_LABEL] <ENT>entidad_texto</ENT> contexto_ventana
```

- Incluir, si es posible, metadatos sencillos: `ner_label`, `doc_type` (si disponible), `position` (start/middle/end).

## 3. Mejoras del entrenamiento SetFit (orientadas a precisión)

### 3.1 Hiperparámetros recomendados

- `model_base`: `intfloat/multilingual-e5-base` (recomendado) o `paraphrase-multilingual-MiniLM-L12-v2` como baseline.
- `num_epochs`: 3–5
- `batch_size`: 8–12
- `samples_per_class`: 16–32 (sobre-muestrear RUIDO difícil)
- `num_iterations` (SetFit contrastive pairs): 40–60

### 3.2 Estrategia de sampling

- Sobre-muestrear ejemplos marcados `is_hard_negative` con peso >1.
- Asegurar diversidad semántica en cada batch (mezclar tipos de NER).

### 3.3 Loss / Regularización

- Usar `MultipleNegativesRankingLoss` o `CosineSimilarityLoss` según configuración. Considerar aumento de ejemplos negativos por muestra.

### 3.4 Modelos base alternativos y razones

- `intfloat/multilingual-e5-base`: excelente para embeddings multilingües y clasificación fina.
- `paraphrase-multilingual-mpnet-base-v2`: equilibrio entre velocidad y calidad.
- `PlanTL-GOB-ES/roberta-base-biomedical-es` (adaptado a sentence-transformers) si se quiere orientación biomédica en español.

## 4. Umbral de decisión (threshold)

- Entrenar como clasificador normal; luego calibrar el threshold de la probabilidad de `PII`.
- Protocolo:
  1. Predecir probabilidades en validation set.
  2. Calcular `precision-recall curve`.
  3. Elegir threshold que maximice precisión con un recall mínimo aceptable (ej. recall ≥ 0.70).

Pseudocódigo de selección de threshold:

```python
precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
valid = np.where(recalls[:-1] >= MIN_RECALL)[0]
best = valid[np.argmax(precisions[:-1][valid])]
optimal_threshold = thresholds[best]
```

## 5. Protocolo de evaluación (antes/después)

- Split por documento (evitar leakage): Train 70%, Val 15%, Test 15% por `doc_id`.
- Métricas a reportar:
  - Precision, Recall, F1 para clase `PII`.
  - Precision global y matriz de confusión.
  - Soporte (número de ejemplos) por categoría NER.
- Análisis de errores por `ner_label`: contar FP y FN por categoría.

## 6. Plan de Acción (concreto)

1. **Auditoría dataset (2–3 días)**
   - Contar distribución, identificar inconsistencias, corregir concatenaciones.
2. **Enriquecimiento y re-equilibrio (3–5 días)**
   - Añadir 200–300 hard negatives; añadir 100–200 ejemplos de fragmentos truncados.
   - Formatear entradas con `[NER_LABEL] <ENT>...`.
3. **Reentrenamiento SetFit (1–2 días)**
   - Base: `intfloat/multilingual-e5-base`.
   - `num_epochs=4`, `batch_size=8`, `samples_per_class=24`, `num_iterations=50`.
4. **Calibración de threshold (1 día)**
   - Generar PR curve y elegir threshold según trade-off negocio (precision objetivo ≥ 0.75 con recall ≥ 0.70 si posible).
5. **Validación final e integración (2 días)**
   - Ejecutar evaluación en test set, documentar mejoras (antes/después) y desplegar en `run_full_pipeline.py` con nuevo modelo y threshold.

## 7. Resumen ejecutivo (1 línea)

Mejorar SetFit requiere primero arreglar calidad y balance del dataset (hard negatives, contextos marcados), luego reentrenar con un embedding multilingüe de mayor calidad (`multilingual-e5-base`) y finalmente calibrar un threshold para priorizar precisión; esperado: reducción sustancial de FPs con pérdida moderada de recall.

---

Si quieres, puedo:

- generar el script de preprocesado para reformatear los ejemplos (`[NER_LABEL] <ENT>...`).
- proponer un script de entrenamiento SetFit con los hiperparámetros sugeridos.
- preparar el experimento de threshold y la visualización PR.
