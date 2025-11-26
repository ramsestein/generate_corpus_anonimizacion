# Análisis Conceptual del Pipeline de Anonimización

**Fecha:** 2025-11-26 09:40

---

## 1. Resumen Ejecutivo

Este análisis identifica patrones de error **sin crear reglas basadas en palabras específicas**.
Se centra exclusivamente en:

- Problemas estructurales (longitud, formato, segmentación)
- Confusiones entre categorías
- Calibración de confianza
- Insuficiencia de contexto

### 1.1 Dataset Analizado

| Métrica | Valor |
|---------|-------|
| Total entidades | 729 |
| Documentos | 50 |
| Resultados validación | 50 |

---

## 2. Patrones Estructurales

### 2.1 Distribución por Longitud

- **13+_chars**: 297 entidades
  - NOMBRE_PERSONAL_SANITARIO: 61
  - CORREO_ELECTRONICO: 47
  - NUMERO_TELEFONO: 37
- **1_char**: 80 entidades
  - NUMERO_TELEFONO: 61
  - NUMERO_IDENTIF: 12
  - ID_SUJETO_ASISTENCIA: 3
- **2-3_chars**: 116 entidades
  - NUMERO_TELEFONO: 69
  - NUMERO_IDENTIF: 15
  - ID_SUJETO_ASISTENCIA: 14
- **4-6_chars**: 64 entidades
  - NOMBRE_PERSONAL_SANITARIO: 18
  - PAIS: 15
  - FAMILIARES_SUJETO_ASISTENCIA: 11
- **7-12_chars**: 172 entidades
  - NUMERO_TELEFONO: 40
  - SEXO_SUJETO_ASISTENCIA: 26
  - ID_SUJETO_ASISTENCIA: 24

### 2.2 Distribución por Tipo de Caracteres

- **alphanumeric_code**: 10 entidades
- **mixed_case**: 74 entidades
- **mixed_content**: 349 entidades
- **numeric_with_separator**: 35 entidades
- **only_digits**: 63 entidades
- **only_lower**: 95 entidades
- **only_punctuation**: 32 entidades
- **only_upper**: 71 entidades

---

## 3. Problemas de Segmentación

**Total problemas detectados:** 596

- **overlap**: 269 casos
- **contiguous_same_label**: 215 casos
- **single_char_fragment**: 80 casos
- **punctuation_only**: 32 casos

---

## 4. Confusiones entre Categorías

**Patrones con múltiples categorías:** 25

### Patrón: `a+`
Etiquetas: NOMBRE_PERSONAL_SANITARIO, SEXO_SUJETO_ASISTENCIA, ID_SUJETO_ASISTENCIA, INSTITUCION, PROFESION, OTROS_SUJETO_ASISTENCIA, FAMILIARES_SUJETO_ASISTENCIA, NUMERO_TELEFONO

### Patrón: `Aa+`
Etiquetas: NOMBRE_PERSONAL_SANITARIO, SEXO_SUJETO_ASISTENCIA, PAIS, TERRITORIO, NOMBRE_SUJETO_ASISTENCIA, FAMILIARES_SUJETO_ASISTENCIA

### Patrón: `aaa`
Etiquetas: NOMBRE_PERSONAL_SANITARIO, INSTITUCION, OTROS_SUJETO_ASISTENCIA, FAMILIARES_SUJETO_ASISTENCIA, NUMERO_TELEFONO, CORREO_ELECTRONICO

### Patrón: `Aa+ Aa+`
Etiquetas: NOMBRE_SUJETO_ASISTENCIA, NOMBRE_PERSONAL_SANITARIO, HOSPITAL, PAIS

### Patrón: `a+ a+`
Etiquetas: PROFESION, OTROS_SUJETO_ASISTENCIA, FAMILIARES_SUJETO_ASISTENCIA


---

## 5. Análisis de Confianza

| Etiqueta | Conf. Promedio | Tasa Problemas |
|----------|----------------|----------------|
| NUMERO_IDENTIF | 0.807 | 76.7% |
| NUMERO_TELEFONO | 0.898 | 47.2% |
| ID_SUJETO_ASISTENCIA | 0.809 | 37.8% |
| INSTITUCION | 0.782 | 12.5% |
| CALLE | 0.994 | 7.7% |
| HOSPITAL | 0.993 | 6.2% |
| OTROS_SUJETO_ASISTENCIA | 0.794 | 5.9% |
| FAMILIARES_SUJETO_ASISTENCIA | 0.936 | 4.2% |
| URL_WEB | 0.768 | 3.2% |
| PAIS | 0.959 | 2.6% |
| CORREO_ELECTRONICO | 0.933 | 1.8% |
| FECHAS | 0.999 | 0.0% |
| SEXO_SUJETO_ASISTENCIA | 0.939 | 0.0% |
| NOMBRE_PERSONAL_SANITARIO | 0.953 | 0.0% |
| PROFESION | 0.963 | 0.0% |
| CENTRO_SALUD | 0.993 | 0.0% |
| EDAD_SUJETO_ASISTENCIA | 0.998 | 0.0% |
| NOMBRE_SUJETO_ASISTENCIA | 0.873 | 0.0% |
| TERRITORIO | 0.757 | 0.0% |

---

## 6. Resultados del Filtrado

| Decisión | Cantidad | Porcentaje |
|----------|----------|------------|
| ESCALATE_TO_LLM | 452 | 62.0% |
| FORCE_ANONYMIZE | 45 | 6.2% |
| FORCE_IGNORE | 232 | 31.8% |

---

## 7. Recomendaciones (Sin Overfitting Léxico)


### 7.1 [ALTA] SEGMENTACIÓN

**Problema:** Alta tasa de entidades de un solo carácter (80 casos)

**Causa raíz:** El modelo NER fragmenta identificadores y códigos en tokens individuales

**Soluciones estructurales:**
- Implementar post-procesamiento que fusione entidades contiguas del mismo tipo
- Ajustar el chunking para evitar cortes en medio de identificadores
- Considerar usar modelos NER con tokenización a nivel de palabra completa

**Mejoras en prompts:**
- Instruir al LLM Judge que ignore entidades < 2 caracteres
- Añadir contexto de ventana más amplio (±50 chars) para entidades cortas

### 7.2 [ALTA] CONFUSIÓN DE CATEGORÍAS

**Problema:** Patrones estructurales asignados a múltiples categorías (25 casos)

**Causa raíz:** Entidades con formato similar reciben etiquetas diferentes según contexto

**Soluciones estructurales:**
- Implementar reglas de desambiguación basadas en contexto sintáctico
- Usar el contexto circundante (preposiciones, verbos) para distinguir
- Crear validadores de formato específicos por etiqueta

**Mejoras en prompts:**
- Incluir en el prompt ejemplos de confusiones frecuentes
- Pedir al LLM que justifique por qué NO es otra categoría
- Añadir contexto de oración completa, no solo fragmento

### 7.3 [ALTA] CONTEXTO

**Problema:** Alta tasa de escalado a LLM (62.0%)

**Causa raíz:** El filtro determinista no tiene suficiente información para decidir

**Soluciones estructurales:**
- Ampliar las listas de términos médicos conocidos (sin memorizar palabras del test)
- Implementar detección de patrones de sección (Diagnóstico, Datos personales, etc.)
- Usar contexto de ventana más amplio en el filtro

**Mejoras en prompts:**
- Proporcionar al LLM el contexto de oración completa
- Incluir información sobre la sección del documento
- Pedir razonamiento explícito antes de la decisión

---

## 8. Conclusiones

Este análisis ha identificado patrones de error basándose únicamente en:

1. **Características estructurales** (longitud, formato, tipo de caracteres)
2. **Problemas de segmentación** (fragmentación, solapamientos)
3. **Confusiones de categoría** (mismos patrones con etiquetas diferentes)
4. **Calibración de confianza** (alta confianza en entidades problemáticas)

**NO se han generado reglas basadas en:**
- Palabras específicas del dataset
- Vocabulario memorizado
- Tokens concretos

Las recomendaciones se centran en mejoras estructurales del pipeline, prompts, y estrategias de razonamiento.

---

*Informe generado automáticamente - Análisis conceptual sin overfitting léxico*
