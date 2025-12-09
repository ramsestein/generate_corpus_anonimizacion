# 🔍 Reporte de Entrenamiento - Gatekeeper SetFit

## 📅 Información General

| Campo | Valor |
|-------|-------|
| **Fecha/Hora** | 2025-12-09 13:01:08 |
| **Modelo Base** | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| **Ruta del Modelo** | `models\setfit_high_precision_v2` |

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
| **error** | (MaxRetryError("HTTPSConnectionPool(host='cas-bridge.xethub.hf.co', port=443): Max retries exceeded with url: /xet-bridge-us/621ffdc136468d709f1802ed/c253da53de897bed72b0c450f220f159fd512827b02704b12d98b363eb0274a8?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20251209%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251209T115309Z&X-Amz-Expires=3600&X-Amz-Signature=f6290751353cb965938abd7d6b29c07349e1b087a65ac2861ed59c0770c34593&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=public&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27model.safetensors%3B+filename%3D%22model.safetensors%22%3B&x-id=GetObject&Expires=1765284789&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NTI4NDc4OX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82MjFmZmRjMTM2NDY4ZDcwOWYxODAyZWQvYzI1M2RhNTNkZTg5N2JlZDcyYjBjNDUwZjIyMGYxNTlmZDUxMjgyN2IwMjcwNGIxMmQ5OGIzNjNlYjAyNzRhOCoifV19&Signature=KI9H3R6NuExxu8HrTBY9uobttszXV9vyKoNTavhffPsxpmi0POrCt~WdmswU7iO6d1v7EvBoCLZlP~XRQ7MYs~RWwnEdv8qKOvtoQZy6S3S1vAX1yGAUwP5dfaDpz3RDxOOuf~~foUJDGNnXquq9RFpmRQ9Gie-vBGJXJIYZC-qaOdOliRabNtfObjeDX2oyQvacLaIXXc6X6lBD3wAI4RJLrYxOzBYIXP7FKD4oEQoqtTooksipa~fFcDXKGn2rOYVHy77k-G-zz3Cbodm3mqoSj6Cu1HmdpOmk7i9mFXaaoOKspwpFL9QOexGUJ5JyiB7lQikmACUl2wBEdYrihA__&Key-Pair-Id=K2L8F4GPSG1IFC (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1032)')))"), '(Request ID: 407f7d76-de94-4d8b-814d-c597734d9f1a)') |

---

## 🔬 Muestras de Verificación

### ✅ Ejemplos Clase 1 (PII - Anonimizar)

| # | Categoría | Texto |
|---|-----------|-------|
| 1 | `OTROS_SUJETO_ASISTENCIA` | Conocido como 'Chato'. |
| 2 | `FAMILIARES_SUJETO_ASISTENCIA` | Padre: [Jose Manuel de Cerezo]. |
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
| **num_iterations** | 40 | Más pares contrastivos → mejor boundary |
| **learning_rate** | 2e-05 | Learning rate conservador para estabilidad |
| **batch_size** | 16 | Balance entre velocidad y precisión |
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
