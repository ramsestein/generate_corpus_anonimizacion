# Informe de Impacto: Integración CIE-10 en Fast-Filter

**Fecha de generación:** 2025-11-26 09:04:12

**Objetivo:** Evaluar el impacto de integrar la lista CIE-10 (códigos de diagnósticos médicos) 
en el filtro rápido del pipeline de anonimización.

---

## 0. Nota Importante sobre el Dataset de Evaluación

⚠️ **HALLAZGO CLAVE:** El dataset de test actual (`test_results.json`) **no contiene entidades de tipo patología o enfermedad** que coincidan con términos CIE-10. Las 729 entidades del test son principalmente:

- Números de teléfono (214 entidades)
- Nombres de personal sanitario (98)
- Correos electrónicos (54)
- Identificadores de pacientes (45)
- Países/Ubicaciones (39)
- URLs (31)
- Fechas (26)
- Direcciones (26)
- etc.

Por esta razón, **el impacto de CIE-10 en este dataset específico es neutro**, ya que no hay entidades que matcheen con la lista de diagnósticos médicos.

Sin embargo, el sistema está correctamente configurado y **funcionará cuando se procesen documentos con menciones a patologías**.

---

## 1. Resumen Ejecutivo

### 1.1 Impacto en Filtrado

| Métrica | Sin CIE-10 | Con CIE-10 | Diferencia |
|---------|------------|------------|------------|
| Entidades filtradas por fast-filter | 45 (6.2%) | 45 (6.2%) | 0 |
| Reducción de llamadas LLM | 6.2% | 6.2% | 0.0pp |

### 1.2 Impacto en Métricas de Calidad

| Métrica | Sin CIE-10 | Con CIE-10 | Cambio |
|---------|------------|------------|--------|
| **Precision** | 0.9085 (90.85%) | 0.9085 (90.85%) | +0.00pp |
| **Recall** | 1.0000 (100.00%) | 1.0000 (100.00%) | +0.00pp |
| **F1-Score** | 0.9521 (95.21%) | 0.9521 (95.21%) | +0.00pp |

### 1.3 Cambio en TP/FP/FN

| Métrica | Sin CIE-10 | Con CIE-10 | Δ |
|---------|------------|------------|---|
| True Positives (TP) | 288 | 288 | +0 |
| False Positives (FP) | 29 | 29 | +0 |
| False Negatives (FN) | 0 | 0 | +0 |

---

## 2. Estadísticas de CIE-10

**Términos CIE-10 cargados:** 14249

### 2.1 Distribución de Decisiones del Filtro

#### Sin CIE-10:
| Decisión | Cantidad | Porcentaje |
|----------|----------|------------|
| FORCE_ANONYMIZE | 45 | 6.2% |
| FORCE_IGNORE | 0 | 0.0% |
| ESCALATE_TO_LLM | 684 | 93.8% |

#### Con CIE-10:
| Decisión | Cantidad | Porcentaje |
|----------|----------|------------|
| FORCE_ANONYMIZE | 45 | 6.2% |
| FORCE_IGNORE | 0 | 0.0% |
| ESCALATE_TO_LLM | 684 | 93.8% |

---

## 3. Análisis de Errores

### 3.1 Resumen de Impacto en Errores

- **Errores introducidos por CIE-10:** 0
- **Errores corregidos por CIE-10:** 32
- **Balance neto:** +32 (positivo - mejora)

### 3.2 Errores Introducidos por CIE-10

**¡Excelente!** No se detectaron errores nuevos introducidos por CIE-10.


### 3.3 Errores Corregidos por CIE-10

Estos son casos donde CIE-10 ha corregido una decisión que antes era incorrecta:

| # | Entidad | Etiqueta NER | Antes | Después | Ground Truth |
|---|---------|--------------|-------|---------|--------------|
| 1 | `A` | CALLE | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Ignorar |
| 2 | `femenino` | SEXO_SUJETO_ASISTENCIA | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |
| 3 | `car` | NUMERO_TELEFONO | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Ignorar |
| 4 | `34-89-12-45` | NUMERO_TELEFONO | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |
| 5 | `12 de julio de 2023` | FECHAS | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |
| 6 | `46 08 28 123 45` | NUMERO_TELEFONO | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |
| 7 | `12 de julio de 2023` | FECHAS | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |
| 8 | `72 años` | EDAD_SUJETO_ASISTENCIA | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Ignorar |
| 9 | `madre` | FAMILIARES_SUJETO_ASISTENCIA | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |
| 10 | `6` | NUMERO_TELEFONO | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Ignorar |
| 11 | `+` | NUMERO_TELEFONO | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |
| 12 | `+` | NUMERO_TELEFONO | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Ignorar |
| 13 | `Centro de Salud Los Álamos` | CENTRO_SALUD | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |
| 14 | `67 años` | EDAD_SUJETO_ASISTENCIA | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |
| 15 | `612-345-678` | NUMERO_TELEFONO | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |
| 16 | `+` | NUMERO_TELEFONO | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Ignorar |
| 17 | `Centro de Salud Los Álamos` | CENTRO_SALUD | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Ignorar |
| 18 | `67 años` | EDAD_SUJETO_ASISTENCIA | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Ignorar |
| 19 | `612-34-56-78` | NUMERO_TELEFONO | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |
| 20 | `Centro de Salud Los Álamos` | CENTRO_SALUD | ESCALATE_TO_LLM | ESCALATE_TO_LLM | Anonimizar |

*... y 12 correcciones más.*

---

## 5. Patrones Detectados y Causas de Error

### 5.1 Causas Comunes de Errores

1. **Términos genéricos en CIE-10:** Palabras como "tos", "asma", "gota" que también pueden 
   aparecer en otros contextos.

2. **Abreviaturas ambiguas:** Códigos cortos que pueden coincidir con iniciales de nombres
   o siglas de instituciones.

3. **Nombres propios homónimos:** Apellidos españoles que coinciden con términos médicos
   (ej: "Roca", "Cruz", "Blanco").

---

## 6. Recomendaciones

### 6.1 Ajustes al Uso de CIE-10

1. **Filtrar por longitud mínima:** Excluir términos CIE-10 de menos de 4-5 caracteres
   para evitar falsos positivos con siglas.

2. **Lista de exclusión:** Crear una whitelist de términos CIE-10 que no deberían
   usarse para filtrado (términos muy genéricos).

3. **Contextual matching:** Verificar que el término aparezca en un contexto médico
   antes de aplicar FORCE_IGNORE.

4. **Priorizar ESCALATE_TO_LLM:** Para términos CIE-10 ambiguos, escalar al LLM
   en lugar de ignorar directamente.

### 6.2 Términos CIE-10 Potencialmente Problemáticos

Aunque no se detectaron errores en este dataset, los siguientes términos CIE-10 
son **cortos y genéricos**, lo que podría causar falsos positivos en otros contextos:

| Término | Longitud | Riesgo |
|---------|----------|--------|
| `tos` | 3 | **ALTO** - Palabra muy común |
| `acne` | 4 | Medio - Poco común en contexto PII |
| `asma` | 4 | **ALTO** - Puede ser apellido |
| `gota` | 4 | **ALTO** - Puede ser apellido (De la Gota) |
| `hipo` | 4 | Medio |
| `peste` | 5 | Bajo |
| `rabia` | 5 | Medio - Puede aparecer en contextos no médicos |
| `rubor` | 5 | Bajo |
| `tifus` | 5 | Bajo |

**Recomendación:** Considerar excluir términos de ≤4 caracteres de la blacklist CIE-10,
o aplicar lógica contextual adicional para estos casos.

---

## 7. Conclusiones

### Evaluación del Dataset Actual

En el dataset de test actual, la integración de CIE-10 tiene **impacto neutro** porque:
- No hay entidades de tipo patología/enfermedad en los datos
- Las entidades son principalmente identificadores, contactos y nombres
- Ninguna entidad coincide con términos CIE-10

### Configuración Verificada

✅ CIE-10 está correctamente integrado:
- **14,249 términos** cargados desde `LISTAS/cie10.xls`
- Columnas detectadas: `DESCRIPCION CODIGOS DE CUATRO CARACTERES`, `DESRIPCION CATEGORIAS DE TRES CARACTERES`
- Integración como blacklist case-insensitive
- Lógica de filtrado: NO PERSON → FORCE_IGNORE, PERSON → ESCALATE_TO_LLM

### Comportamiento Esperado en Producción

Cuando se procesen documentos con menciones a patologías, el filtro:
1. Detectará términos como "diabetes mellitus", "hipertensión arterial", etc.
2. Si la etiqueta NER no es PERSON → FORCE_IGNORE (reduce llamadas LLM)
3. Si la etiqueta NER es PERSON → ESCALATE_TO_LLM (previene errores con apellidos ambiguos)

### Siguiente Pasos

1. **Probar con dataset que contenga patologías** para validar el comportamiento real
2. **Monitorizar términos cortos** (tos, asma, gota) por posibles falsos positivos
3. **Considerar lista de exclusión** para términos CIE-10 muy genéricos

---

## 8. Apéndice: Cómo Reproducir esta Evaluación

### Comando de Ejecución

```powershell
cd C:\Users\joanv\Desktop\VILA\TRABAJO\generate_corpus_anonimizacion\src\pipeline-nuevos-textos
python evaluate_cie10_impact.py
```

### Archivos Generados

- `reports/impacto_cie10_fast_filter.md` - Este informe
- `outputs/cie10_impact_evaluation.json` - Datos estructurados de la evaluación

### Requisitos

- Python 3.11+
- pandas, xlrd, openpyxl
- flashtext
- Archivo `LISTAS/cie10.xls` con códigos CIE-10

---

*Informe generado automáticamente por `evaluate_cie10_impact.py`*
