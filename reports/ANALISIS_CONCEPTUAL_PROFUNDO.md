# ANÁLISIS PROFUNDO DEL PIPELINE DE ANONIMIZACIÓN
## Informe Conceptual Sin Overfitting Léxico

**Fecha:** 26 de Noviembre de 2025  
**Dataset:** `entidades-procesadas-para-metricas.json`  
**Entidades analizadas:** 729  
**Documentos:** 50  

---

## 1. RESUMEN EJECUTIVO

Este informe presenta un análisis **puramente conceptual** de los patrones de error del pipeline de anonimización. Se han identificado los siguientes problemas fundamentales:

| Categoría | Impacto | Entidades Afectadas |
|-----------|---------|---------------------|
| **Fragmentación del NER** | CRÍTICO | 80 entidades de 1 char + 269 solapamientos |
| **Confusión entre categorías** | ALTO | 25 patrones estructurales ambiguos |
| **Calibración de confianza** | MEDIO | 3 etiquetas con >37% de problemas estructurales |
| **Sobrecarga del LLM** | MEDIO | 62% de entidades escaladas |

### Hallazgo Principal

> **El problema fundamental NO es léxico, sino de SEGMENTACIÓN y CONTEXTO.**  
> El NER fragmenta identificadores en tokens individuales, y el pipeline carece de mecanismos para reconstruirlos.

---

## 2. ANÁLISIS DE PROBLEMAS DE SEGMENTACIÓN

### 2.1 Fragmentación de Identificadores

**Problema detectado:** El modelo NER divide códigos y números de identificación en múltiples entidades.

| Tipo de Problema | Casos | % del Total |
|------------------|-------|-------------|
| Entidades de 1 carácter | 80 | 11.0% |
| Entidades contiguas mismo tipo | 215 | 29.5% |
| Solapamientos entre modelos | 269 | 36.9% |
| Solo puntuación | 32 | 4.4% |

**Etiquetas más afectadas por fragmentación:**

| Etiqueta | % Entidades ≤3 chars | Tasa de Problemas |
|----------|----------------------|-------------------|
| NUMERO_IDENTIF | 90% | 76.7% |
| NUMERO_TELEFONO | 60.7% | 47.2% |
| ID_SUJETO_ASISTENCIA | 37.8% | 37.8% |

### 2.2 Patrón Conceptual de la Fragmentación

```
PATRÓN TÍPICO DE FRAGMENTACIÓN:

Texto original:     "DNI: 12345678A"
                          ↓
NER detecta:        ["1", "2345678", "A"]
                          ↓
Etiquetas:          [NUMERO_IDENTIF, NUMERO_TELEFONO, NUMERO_IDENTIF]
                          ↓
Resultado:          3 entidades fragmentadas en lugar de 1
```

**Causa raíz:** El tokenizador del modelo NER no está optimizado para identificadores alfanuméricos del dominio clínico español.

### 2.3 Solapamiento entre Modelos

Se detectaron **269 solapamientos** donde CARMEN y MEDDOCAN detectan la misma región con diferentes límites o etiquetas.

```
EJEMPLO DE SOLAPAMIENTO:

CARMEN:    [pos 784-785] "8" → NUMERO_TELEFONO (conf: 0.988)
MEDDOCAN:  [pos 784-793] "87654321B" → NUMERO_TELEFONO (conf: 0.846)

Diagnóstico: CARMEN fragmenta, MEDDOCAN detecta completo
```

**Implicación:** El pipeline necesita una estrategia de fusión de detecciones que priorice spans más largos y coherentes.

---

## 3. CONFUSIONES SEMÁNTICAS ENTRE CATEGORÍAS

### 3.1 Patrones Estructurales Ambiguos

Se identificaron **25 patrones estructurales** que el NER asigna a múltiples categorías. Esto NO es un problema léxico, sino de **ambigüedad estructural inherente**.

| Patrón | Categorías Confundidas | Interpretación |
|--------|------------------------|----------------|
| `a+` (solo minúsculas) | 8 categorías | Texto genérico sin mayúsculas |
| `Aa+` (mayúscula inicial) | 6 categorías | Nombre propio vs. topónimo vs. término clínico |
| `0+` (solo dígitos) | 3 categorías | Teléfono vs. ID vs. código |
| `A` (1 mayúscula) | 3 categorías | Inicial fragmentada de algo mayor |
| `Aa+ Aa+` (dos palabras) | 4 categorías | Nombre completo vs. lugar vs. institución |

### 3.2 Matriz de Confusión Conceptual

Las confusiones más frecuentes siguen patrones predecibles:

```
                    ┌─────────────────────────────────────┐
                    │  CONFUSIONES ESTRUCTURALES          │
                    ├─────────────────────────────────────┤
   IDENTIFICADORES ←→ TELÉFONOS                           │
   (patrón: dígitos)  Ambos son secuencias numéricas      │
                    ├─────────────────────────────────────┤
   NOMBRES PROPIOS ←→ TOPÓNIMOS                           │
   (patrón: Aa+)      Ambos empiezan con mayúscula        │
                    ├─────────────────────────────────────┤
   FAMILIARES ←→ PROFESIONES                              │
   (patrón: a+ a+)    Ambos son frases descriptivas       │
                    └─────────────────────────────────────┘
```

### 3.3 Análisis de las Confusiones Clave

#### Confusión 1: NUMERO_IDENTIF ↔ NUMERO_TELEFONO

**Patrón común:** Secuencias de dígitos de 6-9 caracteres

**Por qué falla:**
- Un número de 8-9 dígitos puede ser teléfono o DNI
- El NER decide basándose en tokens vecinos, pero en texto clínico el contexto es ambiguo
- "identificador 87654321" vs "teléfono 87654321" → mismo patrón, distinta categoría

**Solución conceptual:**
- NO añadir palabras a listas
- SÍ implementar validación de formato (DNI termina en letra, teléfono no)
- SÍ usar contexto sintáctico ("tfno:", "DNI:", "NIF:")

#### Confusión 2: NOMBRE_PERSONAL_SANITARIO ↔ FAMILIARES ↔ PAIS

**Patrón común:** Palabra con mayúscula inicial (`Aa+`)

**Por qué falla:**
- "Elena" puede ser nombre de médico, familiar, o pueblo
- "Valencia" puede ser lugar, apellido, o parte de nombre de hospital
- Sin contexto de oración, es imposible distinguir

**Solución conceptual:**
- Analizar la estructura sintáctica circundante
- Detectar patrones como "Dr./Dra." antes → PERSONAL_SANITARIO
- Detectar patrones como "su madre/padre/hijo" → FAMILIARES
- Detectar patrones como "natural de" → PAIS/TERRITORIO

#### Confusión 3: ID_SUJETO_ASISTENCIA ↔ OTROS_SUJETO_ASISTENCIA

**Patrón común:** Códigos alfanuméricos cortos

**Por qué falla:**
- "HC" puede ser "Historia Clínica" (código) o abreviatura de hospital
- Códigos de 2-3 caracteres son inherentemente ambiguos
- El NER no distingue entre tipos de identificadores administrativos

**Solución conceptual:**
- Definir reglas de formato esperado por tipo de ID
- Usar posición en el documento (inicio = más probable datos administrativos)
- Implementar validación de patrones conocidos de cada comunidad autónoma

---

## 4. ANÁLISIS DE CALIBRACIÓN DE CONFIANZA

### 4.1 Problema: Alta Confianza ≠ Alta Calidad

| Etiqueta | Confianza Media | Tasa de Problemas | Diagnóstico |
|----------|-----------------|-------------------|-------------|
| NUMERO_IDENTIF | 0.807 | **76.7%** | MAL CALIBRADO |
| NUMERO_TELEFONO | 0.898 | **47.2%** | MAL CALIBRADO |
| ID_SUJETO_ASISTENCIA | 0.809 | **37.8%** | MAL CALIBRADO |
| FECHAS | 0.999 | 0.0% | BIEN CALIBRADO |
| NOMBRE_PERSONAL_SANITARIO | 0.953 | 0.0% | BIEN CALIBRADO |

### 4.2 Patrón Conceptual

```
CALIBRACIÓN DEFICIENTE:

                    Confianza Alta
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
    Entidad bien segmentada    Fragmento (1-2 chars)
    → CORRECTO                 → INCORRECTO pero
                                 alta confianza
```

**Causa raíz:** El modelo NER asigna alta confianza a tokens individuales que reconoce como parte de un patrón, sin considerar si el span completo tiene sentido.

### 4.3 Implicaciones para el Pipeline

1. **No usar confianza como único criterio de filtrado**
2. **Implementar validación de coherencia del span** (longitud mínima, formato)
3. **Penalizar confianza de entidades muy cortas**

---

## 5. ANÁLISIS DEL FLUJO DE DECISIONES

### 5.1 Distribución de Decisiones del Filtro

| Decisión | Cantidad | % |
|----------|----------|---|
| ESCALATE_TO_LLM | 452 | 62.0% |
| FORCE_IGNORE | 232 | 31.8% |
| FORCE_ANONYMIZE | 45 | 6.2% |

### 5.2 Problema: Sobrecarga del LLM

Con **62% de entidades escaladas al LLM**, el pipeline tiene dos problemas:

1. **Coste computacional:** Cada llamada al LLM tiene latencia y coste
2. **El filtro no está haciendo su trabajo:** Debería resolver más casos determinísticamente

### 5.3 Análisis por Etiqueta

**Etiquetas que siempre escalan (100%):**
- CENTRO_SALUD
- NOMBRE_PERSONAL_SANITARIO
- NOMBRE_SUJETO_ASISTENCIA
- EDAD_SUJETO_ASISTENCIA
- FECHAS
- PROFESION
- TERRITORIO

**Etiquetas bien filtradas:**
- NUMERO_IDENTIF: 90% FORCE_IGNORE (correctamente filtradas como fragmentos)
- SEXO_SUJETO_ASISTENCIA: 92.9% FORCE_IGNORE
- HOSPITAL: 81.2% FORCE_ANONYMIZE (correctamente identificados en whitelist)

### 5.4 Diagnóstico

El filtro está funcionando correctamente para:
- **Entidades estructuralmente inválidas** (fragmentos, puntuación)
- **Entidades en listas conocidas** (hospitales, lugares)

El filtro está fallando para:
- **Nombres propios** (no hay regla estructural, siempre escala)
- **Fechas** (aunque bien formateadas, siempre escala)
- **Instituciones** (requieren contexto semántico)

---

## 6. RECOMENDACIONES ESTRUCTURALES

### 6.1 [CRÍTICO] Post-procesamiento de Segmentación

**Objetivo:** Reconstruir entidades fragmentadas antes del filtrado.

**Estrategia:**
```
ALGORITMO DE FUSIÓN:

1. Agrupar entidades por documento
2. Ordenar por posición (start)
3. Para cada par de entidades contiguas (gap ≤ 1 char):
   - Si misma etiqueta → FUSIONAR
   - Si etiquetas compatibles (ej: ambas numéricas) → FUSIONAR
   - Si diferentes → mantener separadas
4. Recalcular confianza del span fusionado
```

**Implementación sugerida:**
- Crear clase `EntityMerger` en el pipeline
- Aplicar ANTES del filtro determinista
- Priorizar detecciones de MEDDOCAN sobre CARMEN cuando hay solapamiento

### 6.2 [ALTA] Validadores de Formato por Categoría

**Objetivo:** Implementar reglas de formato SIN memorizar palabras.

| Etiqueta | Validación de Formato |
|----------|-----------------------|
| NUMERO_IDENTIF | Longitud 8-9, termina en letra |
| NUMERO_TELEFONO | Longitud 9, solo dígitos |
| FECHAS | Patrón dd/mm/yyyy o variantes |
| CORREO_ELECTRONICO | Contiene @ y dominio |
| URL_WEB | Empieza con http o contiene .es/.com |

**Importante:** Estas reglas son sobre ESTRUCTURA, no sobre CONTENIDO.

### 6.3 [ALTA] Mejora del Contexto para el LLM

**Objetivo:** Proporcionar más información contextual sin escalar todo.

**Estrategia:**
```
CONTEXTO ENRIQUECIDO:

Actual:
  "Entidad: Elena | Etiqueta: NOMBRE_PERSONAL_SANITARIO"

Mejorado:
  "Entidad: Elena
   Contexto: '...la Dra. [Elena] realizó la exploración...'
   Posición: principio del documento (sección datos)
   Patrón sintáctico: precedido por título profesional
   Etiqueta NER: NOMBRE_PERSONAL_SANITARIO
   Confianza: 0.96"
```

### 6.4 [MEDIA] Detección de Secciones del Documento

**Objetivo:** Usar la posición y estructura del documento para desambiguar.

**Secciones típicas en documentos clínicos:**
1. **Cabecera:** Datos administrativos (NHC, fecha, doctor responsable)
2. **Anamnesis:** Historia clínica (familiares, antecedentes)
3. **Exploración:** Resultados de pruebas (códigos CIE, valores)
4. **Diagnóstico:** Conclusiones (patologías, tratamientos)
5. **Pie:** Firmas, contactos

**Implementación:**
- Detectar marcadores de sección ("ANTECEDENTES", "EXPLORACIÓN", etc.)
- Asignar probabilidades a priori por tipo de entidad según sección
- Usar como contexto adicional para el LLM

### 6.5 [MEDIA] Reestructuración del Prompt del LLM Judge

**Prompt actual (inferido):**
```
"¿Esta entidad es un dato sensible que debe anonimizarse?"
```

**Prompt mejorado:**
```
TAREA: Determinar si la siguiente entidad contiene información 
personal identificable que requiere anonimización.

ENTIDAD: "{texto}"
ETIQUETA NER: {label}
CONTEXTO: "...{contexto_antes} [{texto}] {contexto_despues}..."
SECCIÓN DEL DOCUMENTO: {seccion}
LONGITUD: {longitud} caracteres
FORMATO: {patron_estructural}

CONSIDERACIONES:
1. Entidades de 1-2 caracteres suelen ser fragmentos → NO anonimizar
2. Códigos médicos (CIE-10, ATC) NO son datos personales
3. Nombres genéricos de roles (médico, enfermera) NO son identificables
4. Nombres propios SÍ requieren anonimización

RAZONA paso a paso antes de decidir.
```

---

## 7. PLAN DE ACCIÓN PRIORIZADO

### Fase 1: Corrección Crítica (1-2 semanas)
- [ ] Implementar `EntityMerger` para fusionar fragmentos
- [ ] Añadir validación de longitud mínima por etiqueta (ya parcialmente hecho)
- [ ] Filtrar entidades de solo puntuación

### Fase 2: Mejora del Filtro (2-4 semanas)
- [ ] Implementar validadores de formato por categoría
- [ ] Añadir detección de patrones sintácticos (Dr./Dra., tfno:, etc.)
- [ ] Reducir tasa de escalado a LLM al <40%

### Fase 3: Mejora del LLM (4-6 semanas)
- [ ] Enriquecer contexto pasado al LLM
- [ ] Implementar detección de secciones
- [ ] Reestructurar prompt con razonamiento explícito

### Fase 4: Evaluación y Ajuste (continuo)
- [ ] Crear dataset de evaluación sin solapamiento con entrenamiento
- [ ] Medir precision/recall por etiqueta
- [ ] Iterar basándose en patrones de error, NO en palabras específicas

---

## 8. CONCLUSIONES

### Lo que SÍ está funcionando:
1. ✅ Detección de hospitales y lugares conocidos
2. ✅ Filtrado de entidades muy cortas (parcial)
3. ✅ Identificación de URLs y emails por formato

### Lo que NO está funcionando:
1. ❌ Segmentación de identificadores (se fragmentan)
2. ❌ Calibración de confianza (alta confianza en fragmentos)
3. ❌ Desambiguación de categorías similares
4. ❌ Uso de contexto del documento

### Principio guía para mejoras futuras:

> **No memorices palabras. Entiende estructuras.**
> 
> Cada regla debe ser generalizable a texto no visto.
> Si una regla solo funciona porque "vimos esa palabra en los datos",
> entonces es overfitting y fallará en producción.

---

*Informe generado el 26/11/2025 - Análisis conceptual sin overfitting léxico*
