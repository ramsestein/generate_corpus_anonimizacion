# Análisis Profundo del Pipeline de Validación de Entidades

**Fecha de generación:** 2025-11-26
**Dataset analizado:** aws2-validation (220 detecciones, 217 con ground truth)

---

## 📋 Resumen Ejecutivo

### Problema Principal Identificado

El análisis revela que **el 90.8% de las entidades detectadas son Falsos Positivos**. El NER está detectando fragmentos de texto que:
1. Son demasiado cortos (1-3 caracteres)
2. Son palabras comunes del lenguaje clínico
3. Son códigos médicos (CIE-10) confundidos con identificadores
4. Son fragmentos de fechas mal delimitados

### Métricas Globales Actuales

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Precision** | 9.22% | ⚠️ MUY BAJA - Demasiados FP |
| **Recall** | 100.00% | ✅ Todos los positivos detectados |
| **F1 Score** | 16.88% | ⚠️ Balance muy desequilibrado |
| **Total TP** | 20 | Entidades correctamente anonimizadas |
| **Total FP** | 197 | Entidades incorrectamente marcadas |
| **Total FN** | 0 | Sin fugas de datos sensibles |

> **Conclusión clave:** El sistema es muy conservador (cero fugas), pero a costa de anonimizar mucho texto que no debería.

---

## 📊 Análisis por Etiqueta

### Ranking de Etiquetas por Gravedad de Error

| # | Etiqueta | Total | FP | FN | Precision | Problema Principal |
|---|----------|-------|----|----|-----------|-------------------|
| 1 | **NUMERO_IDENTIF** | 103 | 101 | 0 | 1.9% | Códigos CIE-10 y letras sueltas |
| 2 | **FAMILIARES_SUJETO_ASISTENCIA** | 76 | 71 | 0 | 6.6% | Palabras "familia/familiar" |
| 3 | **FECHAS** | 22 | 20 | 0 | 9.1% | Fragmentos de fechas (/, 26/7) |
| 4 | **HOSPITAL** | 2 | 2 | 0 | 0.0% | Siglas médicas (H.C, HEL2) |
| 5 | **ID_SUJETO_ASISTENCIA** | 2 | 2 | 0 | 0.0% | Fragmentos ("Ges") |
| 6 | **NOMBRE_PERSONAL_SANITARIO** | 7 | 1 | 0 | 85.7% | Preposiciones ("DE") |

### Etiquetas con Buen Rendimiento

| Etiqueta | TP | FP | Precision | F1 |
|----------|----|----|-----------|-----|
| **INSTITUCION** | 5 | 0 | 100% | 1.0 |
| **NOMBRE_PERSONAL_SANITARIO** | 6 | 1 | 85.7% | 0.92 |

---

## 🔍 Análisis Detallado de Errores

### 1. NUMERO_IDENTIF (101 FP)

**Causa raíz:** El NER confunde códigos CIE-10 y caracteres sueltos con identificadores personales.

**Ejemplos de FP:**
| Texto | Documento | Probable Significado |
|-------|-----------|---------------------|
| `G` | NHC102219 | Inicio de código CIE-10 (G = Enfermedades del sistema nervioso) |
| `045` | NHC102219 | Fragmento de código numérico |
| `06.2` | NHC103087 | Código CIE-10 parcial |
| `5.7` | NHC103087 | Valor numérico médico |
| `OB` | NHC103087 | Abreviatura obstétrica |
| `I` | NHC115410 | Inicio de código CIE-10 (I = Enfermedades cardiovasculares) |
| `PUR` | NHC107102 | Abreviatura médica |

**Patrón detectado:**
- 55 errores son caracteres de 1 sola letra
- 44 errores son fragmentos de 2-3 caracteres
- La mayoría corresponden a códigos CIE-10 fragmentados

### 2. FAMILIARES_SUJETO_ASISTENCIA (71 FP)

**Causa raíz:** El NER detecta las palabras comunes "familia", "familiar", "familiares" como nombres de personas relacionadas con el paciente.

**Ejemplos de FP:**
| Texto | Documento | Contexto probable |
|-------|-----------|-------------------|
| `familia` | NHC103087 | "...informar a la familia del paciente..." |
| `familiar` | NHC104109 | "...antecedente familiar de..." |
| `familiares` | NHC104109 | "...cuidados familiares..." |
| `madre` | NHC108175 | "...leche madre..." o "...madre refiere..." |
| `Familia` | NHC103087 | "Familia informa que..." |

**Patrón detectado:**
- Todas son palabras del vocabulario común que NO son nombres propios
- El NER no distingue entre "familia" (sustantivo) y "Familia" como apellido
- Solo 5 de 76 detecciones son verdaderos datos de familiares

### 3. FECHAS (20 FP)

**Causa raíz:** El NER detecta fragmentos de fechas o caracteres de separación.

**Ejemplos de FP:**
| Texto | Documento | Problema |
|-------|-----------|----------|
| `/` | NHC102219 | Carácter separador aislado |
| `26/7` | NHC102219 | Fecha fragmentada sin año |
| `201` | NHC104109 | Fragmento de año (2019?) |
| `9/10` | NHC109003 | Fecha o fracción médica |
| `05-oct` | NHC125128 | Formato de fecha incompleto |

**Patrón detectado:**
- Fechas parciales sin contexto suficiente
- El filtro debería validar formato completo de fecha

### 4. HOSPITAL / ID_SUJETO_ASISTENCIA (4 FP)

**Ejemplos:**
| Texto | Etiqueta | Problema |
|-------|----------|----------|
| `H.C` | HOSPITAL | Abreviatura de "Historia Clínica" |
| `HEL2` | HOSPITAL | Código interno de sala/helipuerto |
| `Ges` | ID_SUJETO | Fragmento de palabra (gestación?) |

---

## 🔬 Patrones de Error Identificados

### Patrón 1: Entidades de longitud mínima insuficiente

```
DISTRIBUCIÓN DE ERRORES POR LONGITUD:
- 1 carácter:  55 errores (28.0%)  → /, G, I, V, H, E, f, i, .
- 2 caracteres: 19 errores (9.6%)  → OB, DE, HC
- 3 caracteres: 25 errores (12.7%) → 045, PUR, Ges, HEL
- 4+ caracteres: 98 errores (49.7%) → familia, familiar, etc.
```

**Recomendación:** Implementar longitud mínima de 4 caracteres para la mayoría de etiquetas.

### Patrón 2: Palabras comunes del vocabulario clínico

| Palabra | Frecuencia | Etiqueta asignada | Realidad |
|---------|------------|-------------------|----------|
| familia | 35 | FAMILIARES | Sustantivo común |
| familiar | 25 | FAMILIARES | Sustantivo/adjetivo |
| madre | 5 | FAMILIARES | Sustantivo (no nombre) |
| Familia | 6 | FAMILIARES | Sustantivo (mayúscula de inicio) |

**Recomendación:** Crear blacklist explícita con estas palabras comunes.

### Patrón 3: Códigos CIE-10 fragmentados

El NER está detectando partes de códigos CIE-10 como identificadores:
- `G06.2` → detecta `G` y `06.2` por separado
- `I64` → detecta `I` y `64` por separado

**Recomendación:** Ya tenemos CIE-10 en blacklist, pero hay que:
1. Incluir también los prefijos de capítulos (A-Z)
2. Filtrar patrones numéricos que sigan formato CIE-10 (.XX)

### Patrón 4: Caracteres de puntuación

El NER detecta signos de puntuación como entidades:
- `/` (barra de fecha)
- `.` (punto decimal)
- `-` (guión de fecha)

**Recomendación:** Filtrar cualquier entidad que sea solo puntuación.

---

## ✅ Propuestas de Mejora Concretas

### PRIORIDAD ALTA (Impacto inmediato)

#### 1. Implementar filtro de longitud mínima

```python
# Añadir en entity_fast_filter.py

MIN_LENGTH_BY_LABEL = {
    "NUMERO_IDENTIF": 4,
    "NUMERO_TELEFONO": 6,
    "FECHAS": 5,  # Mínimo "dd/mm" o "01-ene"
    "FAMILIARES_SUJETO_ASISTENCIA": 4,
    "HOSPITAL": 4,
    "ID_SUJETO_ASISTENCIA": 4,
    "CORREO_ELECTRONICO": 6,
    "URL_WEB": 6,
    "DEFAULT": 3,
}

def evaluate_candidate(self, entity_text: str, ner_label: str) -> EnumDecision:
    text = entity_text.strip()
    
    # NUEVA REGLA: Longitud mínima
    min_len = MIN_LENGTH_BY_LABEL.get(ner_label, MIN_LENGTH_BY_LABEL["DEFAULT"])
    if len(text) < min_len:
        return EnumDecision.FORCE_IGNORE
    
    # ... resto de la lógica
```

**Impacto esperado:** Elimina 99 FP (50% de los errores)

#### 2. Añadir blacklist de palabras clínicas comunes

```python
# Añadir en entity_fast_filter.py

CLINICAL_COMMON_WORDS = {
    # Palabras de familiares (NO son nombres)
    "familia", "familiar", "familiares", "familias",
    "madre", "padre", "hijo", "hija", "hijos", "hijas",
    "hermano", "hermana", "hermanos", "hermanas",
    "abuelo", "abuela", "abuelos", "abuelas",
    "esposo", "esposa", "cónyuge", "pareja",
    "paciente", "enfermo", "enferma",
    
    # Palabras genéricas
    "centro", "hospital", "clínica", "servicio",
    "historia", "clínica", "informe", "nota",
    
    # Preposiciones y artículos
    "de", "del", "la", "el", "los", "las", "un", "una",
}

def evaluate_candidate(self, entity_text: str, ner_label: str) -> EnumDecision:
    text_lower = entity_text.strip().lower()
    
    # NUEVA REGLA: Blacklist de palabras comunes
    if text_lower in CLINICAL_COMMON_WORDS:
        return EnumDecision.FORCE_IGNORE
    
    # ... resto de la lógica
```

**Impacto esperado:** Elimina 71 FP de FAMILIARES (36% de los errores)

#### 3. Filtrar entidades solo puntuación/números aislados

```python
# Añadir en entity_fast_filter.py

import re

def evaluate_candidate(self, entity_text: str, ner_label: str) -> EnumDecision:
    text = entity_text.strip()
    
    # NUEVA REGLA: Solo puntuación
    if re.match(r'^[\.\,\;\:\-\/\(\)\[\]]+$', text):
        return EnumDecision.FORCE_IGNORE
    
    # NUEVA REGLA: Solo números sueltos (sin formato de ID)
    if re.match(r'^[\d\.\/\-]+$', text) and len(text) < 6:
        if ner_label not in ["NUMERO_TELEFONO", "NUMERO_IDENTIF"]:
            return EnumDecision.FORCE_IGNORE
    
    # ... resto de la lógica
```

**Impacto esperado:** Elimina 50+ FP de FECHAS y NUMERO_IDENTIF

### PRIORIDAD MEDIA (Mejora de precisión)

#### 4. Validar formato de identificadores

```python
# Patrones válidos de identificadores españoles
VALID_IDENTIFIER_PATTERNS = [
    r'^[0-9]{8}[A-Z]$',           # DNI: 12345678A
    r'^[XYZ][0-9]{7}[A-Z]$',      # NIE: X1234567A
    r'^[A-Z][0-9]{7}[A-Z0-9]$',   # CIF: A12345678
    r'^[0-9]{9,12}$',             # Teléfono: 912345678
    r'^[A-Z]{3}[0-9]{6}$',        # Pasaporte: ABC123456
    r'^HC[\-]?[0-9]{4,}$',        # Historia clínica: HC-123456
    r'^NHC[\-]?[0-9]{4,}$',       # NHC: NHC-123456
]

def is_valid_identifier(text: str) -> bool:
    return any(re.match(pattern, text) for pattern in VALID_IDENTIFIER_PATTERNS)
```

#### 5. Mejorar detección de fechas

```python
# Patrones válidos de fechas
VALID_DATE_PATTERNS = [
    r'^\d{1,2}[\-\/]\d{1,2}[\-\/]\d{2,4}$',  # dd/mm/yyyy
    r'^\d{1,2}[\-\/](ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)[\-\/]?\d{0,4}$',  # dd-mes-yyyy
    r'^\d{4}[\-\/]\d{1,2}[\-\/]\d{1,2}$',    # yyyy-mm-dd
]
```

### PRIORIDAD BAJA (Refinamiento)

#### 6. Añadir prefijos CIE-10 a blacklist

```python
# Prefijos de capítulos CIE-10
CIE10_CHAPTER_PREFIXES = set("ABCDEFGHIJKLMNOPQRSTUV")

# Patrón de código CIE-10
CIE10_PATTERN = re.compile(r'^[A-Z]\d{2}(\.\d{1,2})?$')
```

#### 7. Ajustar prompts del LLM Judge

Para las etiquetas problemáticas, añadir instrucciones específicas:

```
INSTRUCCIONES ADICIONALES PARA FAMILIARES_SUJETO_ASISTENCIA:
- Las palabras "familia", "familiar", "familiares" por sí solas NO son datos personales
- Solo anonimizar si el texto es un NOMBRE PROPIO de un familiar (ej: "María García")
- "Informar a la familia" → NO anonimizar
- "Hija: María García" → SÍ anonimizar "María García"
```

---

## 📈 Impacto Estimado de las Mejoras

| Mejora | FP Eliminados | Nuevo FP | Nueva Precision |
|--------|---------------|----------|-----------------|
| Estado actual | - | 197 | 9.2% |
| + Longitud mínima | ~99 | ~98 | ~17% |
| + Blacklist clínica | ~71 | ~27 | ~43% |
| + Solo puntuación | ~10 | ~17 | ~54% |
| **Total estimado** | **~180** | **~17** | **~54%** |

> Con todas las mejoras implementadas, la precisión estimada subiría de **9.2%** a aproximadamente **54%** manteniendo el recall al 100%.

---

## 📝 Archivos a Modificar

1. **`src/pipeline-nuevos-textos/entity_fast_filter.py`**
   - Añadir MIN_LENGTH_BY_LABEL
   - Añadir CLINICAL_COMMON_WORDS
   - Añadir validación de patrones

2. **`data/blacklist_clinical_common.json`** (crear nuevo)
   - Lista de palabras clínicas comunes a ignorar

3. **`src/pipeline-nuevos-textos/llm_prompts.py`** (si existe)
   - Actualizar prompts con instrucciones específicas

---

## 🔄 Plan de Implementación

### Fase 1 (Inmediato - 1 día)
1. ✅ Análisis completado
2. ⏳ Implementar filtro de longitud mínima
3. ⏳ Añadir blacklist de palabras clínicas

### Fase 2 (Corto plazo - 1 semana)
4. ⏳ Implementar validación de patrones de ID
5. ⏳ Ajustar detección de fechas
6. ⏳ Re-ejecutar evaluación con mejoras

### Fase 3 (Medio plazo - 2 semanas)
7. ⏳ Ajustar prompts del LLM Judge
8. ⏳ Monitorizar métricas en producción
9. ⏳ Iterar según nuevos patrones

---

## 📚 Apéndice: Datos Completos

### Distribución de errores por texto

| Texto | Frecuencia | Etiqueta |
|-------|------------|----------|
| familia | 35 | FAMILIARES |
| familiar | 25 | FAMILIARES |
| G | 15 | NUMERO_IDENTIF |
| I | 8 | NUMERO_IDENTIF |
| familiares | 6 | FAMILIARES |
| Familia | 5 | FAMILIARES |
| madre | 5 | FAMILIARES |
| 5.7 | 4 | NUMERO_IDENTIF |
| ... | ... | ... |

### Comando para reproducir análisis

```bash
cd C:\Users\joanv\Desktop\VILA\TRABAJO\generate_corpus_anonimizacion\src\pipeline-nuevos-textos
python full_pipeline_analysis.py
```

---

*Informe generado automáticamente - Pipeline de Anonimización Clínica*
