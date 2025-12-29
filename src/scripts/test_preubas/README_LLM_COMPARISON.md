# Test de Comparación LLM: Qwen vs Hermes

## 📋 Descripción

Este script compara el rendimiento de **Qwen 2.5:7b** vs **Hermes 3:8b** en casos donde **SetFit falla**.

El objetivo es determinar cuál LLM es mejor como "rescate" después de que SetFit comete errores (falsos positivos o falsos negativos).

## 🎯 Casos de Prueba

El script incluye **15 casos específicos** donde SetFit suele fallar:

### Falsos Positivos de SetFit (11 casos)
- "dolores" clasificado como NOMBRE (es síntoma)
- "Mercedes" clasificado como NOMBRE (es marca de vehículo)
- "Madrid" clasificado como PII (es ubicación genérica)
- "72" clasificado como EDAD (es presión arterial)
- "familia" clasificado como PII (es referencia genérica)
- "h" clasificado como NOMBRE (es letra de apartado)
- ".08.2024" clasificado como FECHA (es fragmento malformado)
- "cuidadores" clasificado como PII (es plural genérico)
- "arcía" clasificado como NOMBRE (es fragmento incompleto)
- "65" clasificado como EDAD (es criterio de protocolo)

### Falsos Negativos de SetFit (4 casos)
- "Dolores García" rechazado por SetFit (es nombre real)
- "Hospital San Carlos de Madrid" rechazado (es hospital específico)
- "15/03/2024" rechazado (es fecha con contexto médico)
- "Dr. López" rechazado (es personal sanitario)
- "su hija María" rechazado (es familiar específico)

## 🚀 Uso

### Ejecutar comparación completa

```bash
# Comparación Qwen vs Hermes (todos los casos)
python test_llm_comparison_qwen_hermes.py

# Con modo debug (ver prompts y respuestas)
python test_llm_comparison_qwen_hermes.py --debug

# Con archivo de reglas personalizado
python test_llm_comparison_qwen_hermes.py --rules "../../guias-anotacion.json"
```

### Ejecutar caso específico

```bash
# Probar solo el caso #5
python test_llm_comparison_qwen_hermes.py --test-id 5
```

### Comparar otros modelos

```bash
# Comparar Qwen con Llama
python test_llm_comparison_qwen_hermes.py --models qwen2.5:7b llama3.2:latest
```

## 📊 Métricas Reportadas

El script genera un informe comparativo con:

- **Precisión general**: % de casos correctamente clasificados
- **Falsos Positivos corregidos**: % de FP de SetFit que el LLM corrige
- **Falsos Negativos corregidos**: % de FN de SetFit que el LLM rescata
- **Tiempo promedio**: Tiempo de procesamiento por caso
- **Ganador**: Modelo con mejor precisión general

## 📁 Salida

Los resultados se guardan en:
- **Consola**: Informe detallado con comparativa
- **JSON**: `llm_comparison_results.json` con datos completos

## ⚙️ Requisitos

- Ollama instalado con los modelos:
  - `qwen2.5:7b`
  - `hermes3:8b`
- Módulo `llm_judge` configurado
- Archivo `guias-anotacion.json` (opcional)

## 🔍 Ejemplo de Salida

```
🔬 TEST DE COMPARACIÓN: QWEN vs HERMES (Casos donde SetFit FALLA)
================================================================================
Modelos a comparar: qwen2.5:7b, hermes3:8b
Total casos: 15
================================================================================

CASO #1 (1/15): SetFit detecta 'dolores' como NOMBRE pero es un síntoma
================================================================================
Entidad: 'dolores'
Etiqueta: NOMBRE_SUJETO_ASISTENCIA
SetFit clasificó como: PII
Decisión esperada del LLM: FILTER

  🤖 Probando con qwen2.5:7b...
     Decisión: FILTER - ✅ CORRECTO
     Confianza: 0.95
     Tiempo: 1.23s

  🤖 Probando con hermes3:8b...
     Decisión: FILTER - ✅ CORRECTO
     Confianza: 0.98
     Tiempo: 1.45s

...

📊 RESUMEN COMPARATIVO
================================================================================

🤖 QWEN2.5:7B:
   Precisión general: 93.3% (14/15)
   Falsos Positivos corregidos: 90.9% (10/11)
   Falsos Negativos corregidos: 100.0% (4/4)
   Tiempo promedio: 1.35s

🤖 HERMES3:8B:
   Precisión general: 86.7% (13/15)
   Falsos Positivos corregidos: 81.8% (9/11)
   Falsos Negativos corregidos: 100.0% (4/4)
   Tiempo promedio: 1.52s

🏆 GANADOR
================================================================================
🥇 QWEN2.5:7B gana por 6.6% de precisión
```

## 📝 Notas

- El script solo prueba el **rescate LLM**, no SetFit
- SetFit ya falló en estos casos, estamos midiendo quién lo arregla mejor
- Todos los casos son realistas basados en errores observados en producción
