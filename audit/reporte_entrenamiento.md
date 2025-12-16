# 🔍 Reporte de Entrenamiento - Gatekeeper SetFit

## 📅 Información General

| Campo | Valor |
|-------|-------|
| **Fecha/Hora** | 2025-12-15 16:23:41 |
| **Modelo Base** | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| **Ruta del Modelo** | `models\gatekeeper_setfit_v2` |

---

## 📊 Estadísticas del Dataset

| Métrica | Valor |
|---------|-------|
| **Total de frases** | 1800 |
| **Clase 1 (PII)** | 900 (50.0%) |
| **Clase 0 (Ruido)** | 900 (50.0%) |
| **Categorías procesadas** | 30 |

---

## 📈 Métricas de Evaluación

### 🎯 Métricas Críticas (Clase 1 - PII)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Precision** | 0.0000 | Reducción de Falsos Positivos |
| **Recall** | 0.0000 | ⚠️ CRÍTICO: No perder datos sensibles |
| **F1-Score** | 0.0000 | Balance óptimo Precision/Recall |

### 📊 Métricas Generales

| Métrica | Valor |
|---------|-------|
| **error** | int() argument must be a string, a bytes-like object or a real number, not 'NoneType' |

---

## 🔬 Muestras de Verificación

### ✅ Ejemplos Clase 1 (PII - Anonimizar)

| # | Categoría | Texto |
|---|-----------|-------|
| 1 | `OTROS_SUJETO_ASISTENCIA` | Conocido como 'Chato'. |
| 2 | `FAMILIARES_SUJETO_ASISTENCIA` | Hijo/a [Julie] de [10] años. |
| 3 | `ID_SUJETO_ASISTENCIA` | CIPA: [nhc-907498]. |
| 4 | `OTRO_NUMERO_IDENTIF` | Nº Socio: [9766137210]. |
| 5 | `PROFESION` | Trabaja como [electricista]. |

### ❌ Ejemplos Clase 0 (Ruido - NO Anonimizar)

| # | Categoría | Texto |
|---|-----------|-------|
| 1 | `SEXO_SUJETO_ASISTENCIA` | Cromosoma [X] normal. |
| 2 | `CALLE` | Administrar [pauta 3x1]. |
| 3 | `EDAD_SUJETO_ASISTENCIA` | Glucemia [110] mg/dl. |
| 4 | `PROFESION` | Paciente [trabajador] respiratorio. |
| 5 | `URL_WEB` | Término médico no sensible para URL_WEB. |

---

## 📋 Categorías Procesadas

- `CALLE` (60 ejemplos)
- `CENTRO_SALUD` (60 ejemplos)
- `CORREO_ELECTRONICO` (60 ejemplos)
- `DIREC_PROT_INTERNET` (60 ejemplos)
- `EDAD_SUJETO_ASISTENCIA` (60 ejemplos)
- `FAMILIARES_SUJETO_ASISTENCIA` (60 ejemplos)
- `FECHAS` (60 ejemplos)
- `HOSPITAL` (60 ejemplos)
- `IDENTIF_BIOMETRICOS` (60 ejemplos)
- `IDENTIF_DISPOSITIVOS_NRSERIE` (60 ejemplos)
- `IDENTIF_VEHICULOS_NRSERIE_PLACAS` (60 ejemplos)
- `ID_ASEGURAMIENTO` (60 ejemplos)
- `ID_CONTACTO_ASISTENCIAL` (60 ejemplos)
- `ID_EMPLEO_PERSONAL_SANITARIO` (60 ejemplos)
- `ID_SUJETO_ASISTENCIA` (60 ejemplos)
- `ID_TITULACION_PERSONAL_SANITARIO` (60 ejemplos)
- `INSTITUCION` (60 ejemplos)
- `NOMBRE_PERSONAL_SANITARIO` (60 ejemplos)
- `NOMBRE_SUJETO_ASISTENCIA` (60 ejemplos)
- `NUMERO_BENEF_PLAN_SALUD` (60 ejemplos)
- `NUMERO_FAX` (60 ejemplos)
- `NUMERO_IDENTIF` (60 ejemplos)
- `NUMERO_TELEFONO` (60 ejemplos)
- `OTROS_SUJETO_ASISTENCIA` (60 ejemplos)
- `OTRO_NUMERO_IDENTIF` (60 ejemplos)
- `PAIS` (60 ejemplos)
- `PROFESION` (60 ejemplos)
- `SEXO_SUJETO_ASISTENCIA` (60 ejemplos)
- `TERRITORIO` (60 ejemplos)
- `URL_WEB` (60 ejemplos)

---

## 🔧 Configuración de Entrenamiento (v2 - High Precision)

### Hiperparámetros Aplicados

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **num_iterations** | 20 | Más pares contrastivos → mejor boundary |
| **learning_rate** | 3e-05 | Learning rate conservador para estabilidad |
| **batch_size** | 8 | Balance entre velocidad y precisión |
| **metric** | F1-Score | Optimización del balance Precision/Recall |

### ⚠️ Análisis de Recall

🚨 **Recall BAJO** (<95%): RIESGO de pérdida de datos sensibles. Reentrenar.

---

## 💡 Notas de Interpretación

### Estrategia de Generación de Trampas (Clase 0)

El generador crea ejemplos negativos siguiendo estas estrategias:

1. **TERRITORIO → Anatomía**: Términos como "región lumbar", "zona temporal" suenan a lugares pero son anatómicos.
2. **FECHAS → Referencias genéricas**: "Varios días", "últimas horas" no son fechas específicas.
3. **EDAD → Constantes vitales**: Números como "95%" (saturación), "120/80" (tensión) no son edades.
4. **NOMBRES → Epónimos médicos**: "Síndrome de Cushing", "Maniobra de Heimlich" no son pacientes reales.
5. **HOSPITAL → Servicios**: "UCI", "Urgencias", "Quirófano" son servicios, no hospitales identificables.
6. **ID → Códigos médicos**: "CIE-10: E11.9", "Cama 305" no son identificadores de paciente.

### Cómo Usar Este Reporte

1. **Verificar calidad**: Revisa las muestras para confirmar que los ejemplos son coherentes.
2. **Detectar problemas**: Si ves ejemplos mal generados, ajusta las plantillas en el código.
3. **Trazabilidad**: Este archivo documenta exactamente qué datos se usaron para entrenar.

---

*Generado automáticamente por `train_gatekeeper_audit.py`*
