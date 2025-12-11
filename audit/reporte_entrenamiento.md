# 🔍 Reporte de Entrenamiento - Gatekeeper SetFit

## 📅 Información General

| Campo | Valor |
|-------|-------|
| **Fecha/Hora** | 2025-12-11 13:04:33 |
| **Modelo Base** | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| **Ruta del Modelo** | `models\setfit_high_precision_v2` |

---

## 📊 Estadísticas del Dataset

| Métrica | Valor |
|---------|-------|
| **Total de frases** | 900 |
| **Clase 1 (PII)** | 450 (50.0%) |
| **Clase 0 (Ruido)** | 450 (50.0%) |
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
| 1 | `FECHAS` | Alta médica el día [23/11/1986]. |
| 2 | `NOMBRE_PERSONAL_SANITARIO` | Médico responsable: [Georgina Monreal-Piñeiro]. |
| 3 | `ID_TITULACION_PERSONAL_SANITARIO` | Médico colegiado [39-11777]. |
| 4 | `IDENTIF_VEHICULOS_NRSERIE_PLACAS` | Vehículo [4440TCJ]. |
| 5 | `CENTRO_SALUD` | Pertenece al Centro de Salud [Puerta del Ángel]. |

### ❌ Ejemplos Clase 0 (Ruido - NO Anonimizar)

| # | Categoría | Texto |
|---|-----------|-------|
| 1 | `NUMERO_FAX` | Término médico no sensible para NUMERO_FAX. |
| 2 | `EDAD_SUJETO_ASISTENCIA` | IMC [25]. |
| 3 | `CALLE` | Grado [III/IV]. |
| 4 | `ID_ASEGURAMIENTO` | Término médico no sensible para ID_ASEGURAMIENTO. |
| 5 | `EDAD_SUJETO_ASISTENCIA` | Temperatura [36.5]ºC. |

---

## 📋 Categorías Procesadas

- `CALLE` (30 ejemplos)
- `CENTRO_SALUD` (30 ejemplos)
- `CORREO_ELECTRONICO` (30 ejemplos)
- `DIREC_PROT_INTERNET` (30 ejemplos)
- `EDAD_SUJETO_ASISTENCIA` (30 ejemplos)
- `FAMILIARES_SUJETO_ASISTENCIA` (30 ejemplos)
- `FECHAS` (30 ejemplos)
- `HOSPITAL` (30 ejemplos)
- `IDENTIF_BIOMETRICOS` (30 ejemplos)
- `IDENTIF_DISPOSITIVOS_NRSERIE` (30 ejemplos)
- `IDENTIF_VEHICULOS_NRSERIE_PLACAS` (30 ejemplos)
- `ID_ASEGURAMIENTO` (30 ejemplos)
- `ID_CONTACTO_ASISTENCIAL` (30 ejemplos)
- `ID_EMPLEO_PERSONAL_SANITARIO` (30 ejemplos)
- `ID_SUJETO_ASISTENCIA` (30 ejemplos)
- `ID_TITULACION_PERSONAL_SANITARIO` (30 ejemplos)
- `INSTITUCION` (30 ejemplos)
- `NOMBRE_PERSONAL_SANITARIO` (30 ejemplos)
- `NOMBRE_SUJETO_ASISTENCIA` (30 ejemplos)
- `NUMERO_BENEF_PLAN_SALUD` (30 ejemplos)
- `NUMERO_FAX` (30 ejemplos)
- `NUMERO_IDENTIF` (30 ejemplos)
- `NUMERO_TELEFONO` (30 ejemplos)
- `OTROS_SUJETO_ASISTENCIA` (30 ejemplos)
- `OTRO_NUMERO_IDENTIF` (30 ejemplos)
- `PAIS` (30 ejemplos)
- `PROFESION` (30 ejemplos)
- `SEXO_SUJETO_ASISTENCIA` (30 ejemplos)
- `TERRITORIO` (30 ejemplos)
- `URL_WEB` (30 ejemplos)

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
