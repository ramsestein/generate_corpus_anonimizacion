# 🔍 Reporte de Entrenamiento - Gatekeeper SetFit

## 📅 Información General

| Campo | Valor |
|-------|-------|
| **Fecha/Hora** | 2025-12-01 11:13:52 |
| **Modelo Base** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Ruta del Modelo** | `models\gatekeeper_setfit` |

---

## 📊 Estadísticas del Dataset

| Métrica | Valor |
|---------|-------|
| **Total de frases** | 1160 |
| **Clase 1 (PII)** | 560 (48.3%) |
| **Clase 0 (Ruido)** | 600 (51.7%) |
| **Categorías procesadas** | 30 |

---

## 📈 Métricas de Evaluación

| Métrica | Valor |
|---------|-------|
| **accuracy** | 0.9914 |

---

## 🔬 Muestras de Verificación

### ✅ Ejemplos Clase 1 (PII - Anonimizar)

| # | Categoría | Texto |
|---|-----------|-------|
| 1 | `ID_TITULACION_PERSONAL_SANITARIO` | Nº Colegiado: [50-71148]. |
| 2 | `SEXO_SUJETO_ASISTENCIA` | Enfermo [mujer] que acude. |
| 3 | `CORREO_ELECTRONICO` | Email: [mauricio01@example.org]. |
| 4 | `ID_SUJETO_ASISTENCIA` | Identificador paciente: [P29212]. |
| 5 | `NUMERO_FAX` | Fax: [982 698 901]. |

### ❌ Ejemplos Clase 0 (Ruido - NO Anonimizar)

| # | Categoría | Texto |
|---|-----------|-------|
| 1 | `IDENTIF_VEHICULOS_NRSERIE_PLACAS` | Término médico no sensible para IDENTIF_VEHICULOS_NRSERIE_PLACAS. |
| 2 | `ID_EMPLEO_PERSONAL_SANITARIO` | Término médico no sensible para ID_EMPLEO_PERSONAL_SANITARIO. |
| 3 | `OTROS_SUJETO_ASISTENCIA` | Contexto clínico seguro, no PII (OTROS_SUJETO_ASISTENCIA). |
| 4 | `CALLE` | T[3]N[1]M[0]. |
| 5 | `ID_TITULACION_PERSONAL_SANITARIO` | Contexto clínico seguro, no PII (ID_TITULACION_PERSONAL_SANITARIO). |

---

## 📋 Categorías Procesadas

- `CALLE` (40 ejemplos)
- `CENTRO_SALUD` (40 ejemplos)
- `CORREO_ELECTRONICO` (40 ejemplos)
- `DIREC_PROT_INTERNET` (40 ejemplos)
- `EDAD_SUJETO_ASISTENCIA` (40 ejemplos)
- `FAMILIARES_SUJETO_ASISTENCIA` (40 ejemplos)
- `FECHAS` (40 ejemplos)
- `HOSPITAL` (40 ejemplos)
- `IDENTIF_BIOMETRICOS` (20 ejemplos)
- `IDENTIF_DISPOSITIVOS_NRSERIE` (40 ejemplos)
- `IDENTIF_VEHICULOS_NRSERIE_PLACAS` (40 ejemplos)
- `ID_ASEGURAMIENTO` (40 ejemplos)
- `ID_CONTACTO_ASISTENCIAL` (40 ejemplos)
- `ID_EMPLEO_PERSONAL_SANITARIO` (40 ejemplos)
- `ID_SUJETO_ASISTENCIA` (40 ejemplos)
- `ID_TITULACION_PERSONAL_SANITARIO` (40 ejemplos)
- `INSTITUCION` (40 ejemplos)
- `NOMBRE_PERSONAL_SANITARIO` (40 ejemplos)
- `NOMBRE_SUJETO_ASISTENCIA` (40 ejemplos)
- `NUMERO_BENEF_PLAN_SALUD` (40 ejemplos)
- `NUMERO_FAX` (40 ejemplos)
- `NUMERO_IDENTIF` (20 ejemplos)
- `NUMERO_TELEFONO` (40 ejemplos)
- `OTROS_SUJETO_ASISTENCIA` (40 ejemplos)
- `OTRO_NUMERO_IDENTIF` (40 ejemplos)
- `PAIS` (40 ejemplos)
- `PROFESION` (40 ejemplos)
- `SEXO_SUJETO_ASISTENCIA` (40 ejemplos)
- `TERRITORIO` (40 ejemplos)
- `URL_WEB` (40 ejemplos)

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
