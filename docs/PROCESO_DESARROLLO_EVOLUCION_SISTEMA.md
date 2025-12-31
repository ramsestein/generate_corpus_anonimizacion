# Proceso de Desarrollo y Evolución del Sistema

## Índice

1. [Punto de Partida: Hipótesis Inicial](#1-punto-de-partida-hipótesis-inicial)
2. [Primeros Problemas Detectados](#2-primeros-problemas-detectados)
3. [Iteraciones y Experimentación](#3-iteraciones-y-experimentación)
4. [Introducción de SetFit](#4-introducción-de-setfit)
5. [Refinamiento del Pipeline](#5-refinamiento-del-pipeline)
6. [Estado Actual del Sistema](#6-estado-actual-del-sistema)
7. [Reflexión Metodológica](#7-reflexión-metodológica)

---

## 1. Punto de Partida: Hipótesis Inicial

### 1.1 Objetivo del Proyecto

El proyecto nació con el objetivo de desarrollar un sistema automatizado de anonimización de textos clínicos en español que cumpliera con las normativas de protección de datos (RGPD/LOPD) vigentes en el ámbito sanitario. La necesidad surgía de un problema práctico: los hospitales y centros de investigación disponen de grandes volúmenes de informes médicos que contienen información clínica valiosa para investigación, pero cuyo uso está restringido por la presencia de datos personales identificables.

### 1.2 Expectativas sobre los Modelos NER

La hipótesis inicial se basaba en la disponibilidad de modelos NER pre-entrenados específicamente para el dominio biomédico en español, desarrollados por el Barcelona Supercomputing Center (BSC):

- **MEDDOCAN** (bsc-bio-ehr-es-meddocan): Fine-tuned sobre el corpus MEDDOCAN de anonimización médica
- **CARMEN-I** (bsc-bio-ehr-es-carmen-anon): Fine-tuned sobre el corpus CARMEN de informes clínicos

La expectativa era que estos modelos, al estar entrenados específicamente en textos clínicos en español, proporcionarían detecciones de alta calidad que podrían utilizarse directamente para la anonimización. Se asumía que el problema principal sería de ingeniería (integración, procesamiento de documentos, generación de corpus) más que de calidad de las predicciones.

### 1.3 Arquitectura Inicial Planteada

El diseño inicial contemplaba un pipeline lineal simple:

```
Documento → NER (MEDDOCAN/CARMEN) → Sustitución por XXX → Documento Anonimizado
```

Se estimaba que los modelos NER alcanzarían métricas similares a las reportadas en sus publicaciones originales (F1 > 0.85) y que el esfuerzo principal se centraría en la gestión de documentos y la validación humana final.

---

## 2. Primeros Problemas Detectados

### 2.1 Precision Significativamente Inferior a lo Esperado

Las primeras evaluaciones sistemáticas revelaron una realidad muy distinta a las expectativas. El análisis sobre un conjunto de validación mostró que, aunque el sistema detectaba la mayoría de las entidades PII reales (alto recall), presentaba una **tasa muy elevada de falsos positivos** que degradaba severamente la precision. Esto significaba que por cada entidad PII real detectada, se marcaban múltiples fragmentos de texto que no debían anonimizarse.

### 2.2 Taxonomía de Errores Identificados

El análisis detallado de los falsos positivos reveló patrones sistemáticos de error:

**Fragmentación de entidades**: El tokenizador de los modelos NER dividía identificadores alfanuméricos en tokens individuales. Un DNI como "12345678A" podía detectarse como tres entidades separadas: "1234", "5678" y "A". Este problema afectaba especialmente a:
- Números de identificación
- Números de teléfono
- Identificadores de paciente

**Detección de vocabulario clínico común**: El NER clasificaba palabras del vocabulario clínico estándar como entidades PII:
- "familia", "familiar", "familiares" → detectados como FAMILIARES_SUJETO_ASISTENCIA
- "madre", "padre" → detectados como relaciones familiares cuando aparecían en contextos como "leche madre" o "efecto padre"
- Códigos CIE-10 fragmentados interpretados como identificadores

**Entidades de longitud insuficiente**: Una proporción significativa de los falsos positivos correspondía a caracteres únicos (/, G, I, V) detectados como entidades independientes, así como fragmentos de 2-3 caracteres que carecían de significado identificativo.

**Fechas fragmentadas**: El sistema detectaba componentes aislados de fechas (barras separadoras, días sin mes, años truncados) en lugar de fechas completas.

### 2.3 Solapamiento entre Modelos

Al combinar MEDDOCAN y CARMEN en un ensemble, se detectaron múltiples casos de solapamiento donde ambos modelos detectaban la misma región textual con diferentes límites o etiquetas. Por ejemplo, era frecuente encontrar situaciones donde CARMEN detectaba un fragmento parcial de un identificador mientras MEDDOCAN detectaba el identificador completo, generando conflictos en las coordenadas de las entidades.

Este problema evidenciaba que la simple unión de detecciones no era una estrategia viable sin un mecanismo de resolución de conflictos.

---

## 3. Iteraciones y Experimentación

### 3.1 Introducción de Métricas de Validación

El primer paso metodológico fue establecer un sistema riguroso de medición. Se implementó un framework de evaluación basado en:

- **TP (True Positives)**: Entidades PII correctamente detectadas y anonimizadas
- **FP (False Positives)**: Texto no-PII incorrectamente marcado como sensible
- **FN (False Negatives)**: Entidades PII no detectadas (fugas de privacidad)

Esta distinción era crítica porque el coste de los errores no es simétrico:
- Un FP degrada la utilidad del documento (sobreanonimización)
- Un FN constituye una fuga de privacidad (violación normativa potencial)

### 3.2 El Problema del Gold Standard

Un desafío metodológico significativo fue la ausencia de un gold standard clásico para evaluación. Los corpus MEDDOCAN y CARMEN originales no estaban disponibles para uso directo, y crear anotaciones manuales exhaustivas requería recursos que excedían el alcance del proyecto.

La solución adoptada fue utilizar los marcadores `[**...**]` presentes en los documentos ya anonimizados como **silver standard**. Estos marcadores, insertados por procesos previos de anonimización manual o semi-automática, indicaban las posiciones donde existían entidades PII reales. Si bien no constituían un gold standard perfecto, proporcionaban una aproximación razonable para evaluar el rendimiento del sistema.

### 3.3 Matching de Entidades: Del Exacto al Fuzzy

Las primeras implementaciones de evaluación usaban matching exacto por posición (start, end), lo cual resultó inadecuado por dos razones:

1. Los offsets de las detecciones NER no siempre coincidían exactamente con los marcadores gold debido a diferencias en tokenización
2. Detecciones parcialmente correctas (que cubrían parte de la entidad) se contabilizaban como errores completos

Se implementó entonces un sistema de **overlap ratio** para el matching:

```python
def compute_overlap_ratio(det_start, det_end, gold_start, gold_end):
    overlap_chars = max(0, min(det_end, gold_end) - max(det_start, gold_start))
    return overlap_chars / min(det_end - det_start, gold_end - gold_start)
```

Con un umbral de 0.5 (50% de solapamiento), el sistema podía reconocer detecciones parcialmente correctas, proporcionando una evaluación más realista del rendimiento.

### 3.4 Matching 1:1 Greedy

Un problema adicional era que múltiples detecciones podían matchear con el mismo gold, inflando artificialmente las métricas. Se implementó un algoritmo de matching 1:1 greedy que:

1. Calcula overlap_ratio para todos los pares (detección, gold)
2. Ordena los candidatos por overlap descendente
3. Asigna matches de forma exclusiva (cada gold/detección solo puede emparejarse una vez)
4. Clasifica como TP los matches, FN los golds sin emparejar, FP las detecciones sin emparejar

---

## 4. Introducción de SetFit

### 4.1 Diagnóstico del Problema

Tras las iteraciones de evaluación, quedó claro que el problema no era de recall (los modelos NER detectaban las entidades PII) sino de **precision** (también detectaban mucho ruido). El análisis por etiquetas mostró que ciertas categorías como identificadores numéricos y relaciones familiares presentaban tasas de precision muy bajas, mientras que otras como nombres de personal sanitario mostraban mejor rendimiento.

Se necesitaba un mecanismo de filtrado que discriminara entre detecciones correctas (PII real) y ruido (falsos positivos del NER).

### 4.2 Alternativas Consideradas

Se evaluaron varias opciones:

**Reentrenar MEDDOCAN/CARMEN**: Descartado por varias razones:
- Requería acceso a los corpus originales de entrenamiento
- El fine-tuning de modelos transformer completos demanda recursos computacionales significativos
- Los tiempos de iteración serían largos (horas por experimento)
- Riesgo de degradar el recall mientras se mejoraba precision

**Reglas heurísticas**: Implementadas parcialmente (filtros de longitud mínima, blacklists de vocabulario clínico), pero insuficientes para capturar la complejidad semántica del problema.

**Modelo de clasificación binaria**: La opción elegida. Entrenar un clasificador ligero que, dada una detección del NER y su contexto, decidiera si era PII real o ruido.

### 4.3 Por Qué SetFit

SetFit (Sentence-BERT Fine-Tuning) se seleccionó por sus características técnicas:

1. **Few-shot learning**: Permite entrenar con pocos ejemplos (cientos vs miles), crucial dado el limitado gold standard disponible
2. **Velocidad de entrenamiento**: Minutos vs horas de un BERT completo, permitiendo iteraciones rápidas
3. **Eficiencia en inferencia**: Modelo compacto basado en embeddings, integrable en el pipeline sin penalización significativa de latencia
4. **Capacidad de capturar contexto**: Al recibir entidad + oración como entrada, puede aprender patrones contextuales (ej: "la Dra. María" vs "la enfermedad de María")

### 4.4 Diseño del Formato de Entrada

Se diseñó un formato de entrada que proporcionara contexto semántico al clasificador:

```
ENTITY: <texto_entidad>
SENTENCE: <oración_que_contiene_la_entidad>
```

Este formato permite al modelo aprender patrones como:
- "María" precedido de "Dra." → probablemente PII
- "familia" en "informar a la familia" → probablemente ruido
- "12345678A" en contexto de "DNI:" → probablemente PII

### 4.5 Errores que SetFit Corrige

Tras el entrenamiento inicial, SetFit demostró capacidad para filtrar:

- Palabras comunes del vocabulario clínico (familia, familiar, madre)
- Fragmentos de códigos CIE-10 detectados erróneamente
- Caracteres aislados y puntuación
- Fechas fragmentadas sin contexto identificativo

Las métricas mejoraron significativamente, observándose un aumento notable en precision con una reducción controlada de recall.

### 4.6 Limitaciones Persistentes

SetFit no resuelve todos los problemas:

- **Trade-off precision/recall**: Aumentar el filtrado mejora precision pero reduce recall (fugas inducidas)
- **Entidades ambiguas**: Casos donde el contexto no es suficiente para determinar si es PII (ej: "Valencia" como apellido vs topónimo)
- **Dependencia del dataset de entrenamiento**: La calidad del clasificador depende de los ejemplos disponibles

---

## 5. Refinamiento del Pipeline

### 5.1 Token Healing

Se detectó que algunas entidades llegaban con fronteras incorrectas debido a la tokenización subword de los modelos BERT. Entidades como "##ez" (fragmento de "González") aparecían como detecciones independientes.

Se implementó un mecanismo de **token healing** que:
1. Detecta entidades que comienzan con prefijos de subword (##)
2. Expande hacia atrás hasta encontrar el inicio de palabra
3. Corrige los offsets y el texto de la entidad

### 5.2 Extracción de Contexto

Para alimentar SetFit con contexto de calidad, se implementó extracción automática de la oración que contiene cada entidad:

```python
def extract_context(doc_text, start, end):
    sent_start = doc_text.rfind('.', 0, start) + 1
    sent_end = doc_text.find('.', end)
    return doc_text[sent_start:sent_end].strip()
```

### 5.3 Separación de Etapas

El pipeline evolucionó hacia una arquitectura modular con separación clara de responsabilidades:

1. **Detección** (NER): Maximiza recall, genera candidatos
2. **Filtrado** (SetFit): Optimiza precision, elimina ruido
3. **Rescate** (LLM Judge, opcional): Recupera entidades ambiguas
4. **Validación**: Evalúa métricas sobre gold standard

Esta separación permite optimizar cada componente independientemente y facilita el diagnóstico de errores.

### 5.4 LLM Judge como Red de Seguridad

Se introdujo un mecanismo opcional de rescate mediante LLM (qwen2.5:7b a través de Ollama) para evaluar entidades que SetFit clasifica como ruido:

- Si el LLM determina que es PII real → se rescata
- Si el LLM confirma que es ruido → se descarta definitivamente

Este componente actúa como red de seguridad para casos ambiguos, aunque añade latencia al pipeline.

---

## 6. Estado Actual del Sistema

### 6.1 Arquitectura Final

El sistema actual implementa una arquitectura de dos etapas con rescate opcional:

```
Input → NER Ensemble → SetFit Gatekeeper → [LLM Judge] → Output
         (recall)        (precision)        (rescue)
```

- **NER Ensemble**: Combina MEDDOCAN y CARMEN, deduplica por coordenadas
- **SetFit Gatekeeper**: Clasifica cada detección como PII o RUIDO
- **LLM Judge** (opcional): Evalúa entidades clasificadas como RUIDO

### 6.2 Métricas Actuales

El sistema permite evaluar diferentes configuraciones sobre los corpus de evaluación disponibles. Las métricas varían según la configuración de SetFit y LLM Judge utilizada, y pueden consultarse ejecutando el script `model_comparision.py` con los parámetros correspondientes.

### 6.3 Garantías del Sistema

El sistema ofrece las siguientes garantías:

1. **Trazabilidad completa**: Cada entidad incluye metadatos sobre su origen (NER), clasificación (SetFit) y decisión final
2. **Configurabilidad**: Umbrales ajustables para balancear precision/recall según requisitos
3. **Evaluación sistemática**: Framework de métricas para comparar configuraciones
4. **Modularidad**: Componentes independientes que pueden actualizarse sin afectar al resto

### 6.4 Problemas Abiertos

Persisten desafíos no resueltos completamente:

- **Entidades genuinamente ambiguas**: Casos donde incluso un humano tendría dificultad para decidir sin contexto adicional
- **Calibración de confianza del NER**: Los modelos reportan alta confianza incluso en fragmentos claramente erróneos
- **Fugas inducidas por SetFit**: Al filtrar agresivamente, algunas entidades PII reales son eliminadas incorrectamente
- **Variabilidad por tipo de documento**: El rendimiento varía según el tipo de informe clínico

### 6.5 Mejoras Futuras Razonables

Direcciones de trabajo futuro que podrían mejorar el sistema:

1. **Post-procesamiento de segmentación**: Implementar fusión de entidades fragmentadas antes del filtrado
2. **Calibración de umbral adaptativa**: Ajustar umbrales según el tipo de etiqueta y contexto
3. **Dataset de entrenamiento enriquecido**: Incorporar hard negatives minados de errores en producción
4. **Validación humana sistemática**: Establecer proceso de revisión continua para identificar patrones de error nuevos

---

## 7. Reflexión Metodológica

### 7.1 La Necesidad del Enfoque Iterativo

El desarrollo de este sistema ilustra por qué los proyectos de NLP aplicado raramente siguen un camino lineal. Las hipótesis iniciales sobre el rendimiento de los modelos pre-entrenados resultaron incorrectas no por deficiencias de los modelos, sino por una comprensión insuficiente del dominio de aplicación.

Los textos clínicos presentan características que los corpus de entrenamiento estándar no capturan completamente:
- Vocabulario especializado que comparte patrones léxicos con entidades PII
- Formatos de identificadores específicos del sistema sanitario español
- Variabilidad en la estructura de los documentos según el tipo de informe

Solo mediante iteraciones de evaluación-análisis-mejora fue posible identificar estos problemas y diseñar soluciones adecuadas.

### 7.2 Lecciones sobre NER Clínico en Práctica

El trabajo con modelos NER en contexto clínico real reveló varias lecciones:

1. **Las métricas de benchmark no predicen el rendimiento en producción**: Los modelos MEDDOCAN y CARMEN reportan F1 > 0.85 en sus benchmarks, pero en nuestro contexto específico mostraron comportamientos muy diferentes.

2. **El recall es más fácil que la precision**: Es relativamente sencillo diseñar un sistema que detecte todas las entidades PII (alto recall), pero mucho más difícil evitar falsos positivos sin perder entidades reales.

3. **El contexto determina la clasificación**: La misma cadena de caracteres puede ser PII o no dependiendo del contexto. "María" es PII en "la paciente María García" pero no en "Ave María Purísima" (fórmula religiosa en documentos históricos).

4. **Los errores de tokenización se propagan**: Problemas en la segmentación a nivel de tokens generan cascadas de errores en la detección de entidades.

### 7.3 Métricas Útiles vs Métricas Bonitas

Una reflexión importante del proyecto es la diferencia entre métricas que se ven bien en reportes y métricas que reflejan el riesgo real de privacidad:

- **Precision alta con recall bajo**: Puede parecer bueno (pocas detecciones erróneas) pero implica fugas de datos sensibles no detectados
- **Recall 100% con precision baja**: Parece seguro (todo detectado) pero degrada severamente la utilidad del documento
- **F1 equilibrado**: Oculta el trade-off real entre seguridad (recall) y utilidad (precision)

Para un sistema de anonimización, la métrica más relevante es el **recall sobre entidades de alto riesgo** (nombres completos, identificadores únicos, direcciones completas). Un FN en "Juan Pérez García" es mucho más grave que un FN en "26/07" (fecha que podría referirse a cualquier año).

El sistema actual intenta capturar esta distinción mediante el desglose de métricas por etiqueta y el análisis de fugas inducidas, pero queda pendiente una formalización más rigurosa del concepto de "riesgo de privacidad" como métrica de evaluación.

---

*Documentación del proceso de desarrollo del Sistema de Anonimización de Textos Clínicos*
*Última actualización: Diciembre 2025*
