# Pipeline LLM Judge con Ollama + gemma3:270m

## 🎯 Descripción

Este pipeline evalúa automáticamente entidades detectadas en documentos clínicos usando **Ollama** con el modelo local **gemma3:270m**.

### ✅ Ventajas de usar Ollama

- **100% Local**: No requiere conexión a internet
- **Sin Costos**: No hay cargos por tokens o API calls
- **Sin Límites**: Sin restricciones de rate limiting
- **Privacidad Total**: Los datos nunca salen de tu máquina
- **Rápido**: gemma3:270m es un modelo ligero optimizado para velocidad

## 📋 Requisitos

### 1. Instalar Ollama

**Windows/macOS/Linux:**
```bash
# Descarga desde: https://ollama.ai
# O usa el instalador oficial para tu sistema operativo
```

**Verificar instalación:**
```bash
ollama --version
```

### 2. Descargar el modelo gemma3:270m

```bash
ollama pull gemma3:270m
```

Este comando descargará el modelo (~270MB). La primera descarga puede tomar unos minutos.

**Verificar que el modelo está disponible:**
```bash
ollama list
```

Deberías ver `gemma3:270m` en la lista.

## 🚀 Uso Rápido

### Paso 1: Verificar Setup

Ejecuta el test de verificación:

```bash
python tests/test_ollama_setup.py
```

Deberías ver:
```
✓ PASS: Instalación de Ollama
✓ PASS: Modelo gemma3:270m
✓ PASS: Llamada a Ollama
✓ PASS: Módulos del Pipeline
```

### Paso 2: Ejecutar con Datos de Ejemplo

```bash
python src/pipeline-nuevos-textos/llm_entity_judge.py \
  --entities examples/entities_example.json \
  --docs examples/documents \
  --output results_ollama.csv
```

### Paso 3: Ejecutar con Tus Datos

```bash
python src/pipeline-nuevos-textos/llm_entity_judge.py \
  --entities corpus/output/aws2/entities.json \
  --docs corpus/output/aws2/documents \
  --output outputs/aws2_evaluation_ollama.csv
```

## 📊 Parámetros del Pipeline

| Parámetro | Descripción | Default | Ejemplo |
|-----------|-------------|---------|---------|
| `--entities` | JSON con entidades detectadas | - | `entities.json` |
| `--docs` | Directorio con documentos originales | - | `corpus/documents` |
| `--rules-file` | JSON con reglas de anotación | `guias-anotacion.json` | `custom_rules.json` |
| `--output` | Archivo CSV de salida | `llm_entity_judgments.csv` | `results.csv` |
| `--left-window` | Caracteres de contexto izquierdo | 80 | 120 |
| `--right-window` | Caracteres de contexto derecho | 80 | 120 |

## 🔧 Cómo Funciona

### Flujo de Procesamiento

```
┌─────────────────────────────────────────────────────────────┐
│                    1. Cargar JSON de Entidades               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              2. Para cada entidad en el JSON:                │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ a) Cargar documento original (.txt)                │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ b) Extraer chunk de contexto                       │    │
│  │    - left_window caracteres antes                  │    │
│  │    - right_window caracteres después               │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ c) Obtener reglas de la etiqueta                   │    │
│  │    desde guias-anotacion.json                      │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ d) Construir prompts:                              │    │
│  │    • System: reglas de la etiqueta                 │    │
│  │    • User: palabra + contexto + consulta           │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ e) Llamar a Ollama:                                │    │
│  │    ollama run gemma3:270m [prompt]                 │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ f) Parsear respuesta: TRUE/FALSE                   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              3. Guardar resultados en CSV                    │
└─────────────────────────────────────────────────────────────┘
```

### Construcción de Prompts

#### System Prompt (Reglas de la Etiqueta)
```
ERES UN DETECTOR PROFESIONAL DE ENTIDADES MÉDICAS.

TU TAREA: Determinar si una palabra es una entidad válida del tipo 
{ETIQUETA} según las reglas oficiales.

OBJETIVO: MAXIMIZAR RECALL. En caso de duda, prefiere TRUE.
MANTÉN PRECISIÓN RAZONABLE para evitar falsos positivos.

REGLAS OFICIALES DE LA ETIQUETA {ETIQUETA}:
1. [Regla 1 de la etiqueta]
2. [Regla 2 de la etiqueta]
...

FORMATO DE RESPUESTA OBLIGATORIO:
- "TRUE" si la palabra ES una entidad válida
- "FALSE" si la palabra NO ES una entidad válida
```

#### User Prompt (Palabra + Contexto)
```
PALABRA CANDIDATA: "{keyword}"

TEXTO DONDE APARECE LA PALABRA (del documento original):
"{context_text}"

ETIQUETA A VERIFICAR: {label}
UBICACIÓN EN DOCUMENTO: offset {start}-{end}

¿Es "{keyword}" una entidad válida del tipo {label} según las reglas?

Responde SOLO: "TRUE" o "FALSE"
```

## 📤 Formato de Salida

El CSV de salida contiene:

| Campo | Descripción |
|-------|-------------|
| `document_id` | ID del documento |
| `entity_id` | ID de la entidad |
| `keyword` | Palabra evaluada |
| `label` | Tipo de entidad |
| `start` | Posición inicio |
| `end` | Posición fin |
| `context` | Chunk de contexto extraído |
| `llm_response` | Respuesta cruda de Ollama |
| `is_valid` | TRUE/FALSE parseado |
| `status` | Estado del procesamiento |

### Ejemplo de CSV
```csv
document_id,entity_id,keyword,label,start,end,context,llm_response,is_valid,status
doc_001,ent_001,García,NOMBRE_SUJETO_ASISTENCIA,12,18,"El paciente García tiene...",TRUE,True,success
doc_001,ent_002,25/12/2020,FECHAS,50,60,"Fecha de ingreso: 25/12/2020...",TRUE,True,success
```

## ⚡ Rendimiento

### Tiempos Estimados (gemma3:270m)

- **Primera llamada**: ~5-10 segundos (carga del modelo)
- **Llamadas subsecuentes**: ~2-3 segundos por entidad
- **100 entidades**: ~5-10 minutos
- **1000 entidades**: ~50-100 minutos

### Optimización de Velocidad

gemma3:270m permanece en memoria después de la primera llamada, por lo que las evaluaciones subsecuentes son mucho más rápidas.

**Tip**: Mantén Ollama ejecutándose para evitar tiempos de carga:
```bash
# En una terminal separada
ollama serve
```

## 🐛 Troubleshooting

### Problema: "Ollama no está instalado"

**Solución**: Instala Ollama desde https://ollama.ai

**Windows:**
```powershell
# Descarga el instalador .exe desde ollama.ai
```

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

---

### Problema: "Modelo gemma3:270m no encontrado"

**Solución**: Descarga el modelo:
```bash
ollama pull gemma3:270m
```

Para verificar:
```bash
ollama list
```

---

### Problema: "Timeout al ejecutar Ollama"

**Causa**: El modelo está descargándose o el sistema está lento.

**Solución**:
1. Espera a que la descarga del modelo termine completamente
2. Verifica recursos del sistema (RAM, CPU)
3. Aumenta el timeout si es necesario (edita el código: `timeout=60` → `timeout=120`)

---

### Problema: "Respuesta no contiene TRUE/FALSE"

**Causa**: El modelo puede dar respuestas no estándar.

**Solución**: El pipeline tiene parsing robusto que busca TRUE/FALSE en cualquier parte de la respuesta. Si persiste, verifica:
1. El prompt está bien formateado
2. El modelo gemma3:270m está actualizado
3. Considera ajustar el prompt para ser más explícito

---

## 📈 Comparación: Ollama vs API Cloud

| Aspecto | Ollama (gemma3:270m) | API Cloud (GPT-4) |
|---------|---------------------|-------------------|
| **Costo** | 🟢 Gratis | 🔴 ~$0.05/entidad |
| **Velocidad** | 🟡 ~2-3s/entidad | 🟢 ~1-2s/entidad |
| **Privacidad** | 🟢 100% local | 🔴 Envía datos externos |
| **Rate Limits** | 🟢 Sin límites | 🔴 3500 req/min |
| **Setup** | 🟡 Requiere Ollama | 🟢 Solo API key |
| **Calidad** | 🟡 Buena | 🟢 Excelente |

## 🔄 Migración desde API Cloud

Si vienes de la versión anterior con OpenAI/Claude:

### Cambios Principales

1. **No necesitas `.env`**: Ollama no requiere API keys
2. **Instalación local**: Debes instalar Ollama + modelo
3. **Sin costos**: Evaluaciones ilimitadas sin cargo
4. **Mismo formato**: El CSV de salida es idéntico

### Comandos Equivalentes

**Antes (OpenAI):**
```bash
# Configurar .env
LLM_API_KEY=sk-...
LLM_MODEL_NAME=gpt-4

python llm_entity_judge.py --entities data.json --docs docs/
```

**Ahora (Ollama):**
```bash
# Instalar una vez
ollama pull gemma3:270m

# Usar directamente
python llm_entity_judge.py --entities data.json --docs docs/
```

## 📚 Recursos Adicionales

- **Ollama Docs**: https://github.com/ollama/ollama/blob/main/docs/README.md
- **Gemma Models**: https://ai.google.dev/gemma
- **Pipeline Técnico**: Ver `TECHNICAL_DOCUMENTATION.md`

## 💡 Tips y Mejores Prácticas

### 1. Contexto Óptimo

```bash
# Contexto corto (más rápido)
--left-window 50 --right-window 50

# Contexto estándar (recomendado)
--left-window 80 --right-window 80

# Contexto amplio (más preciso)
--left-window 150 --right-window 150
```

### 2. Procesar por Lotes

Si tienes muchos documentos, procesa en lotes:

```bash
# Lote 1
python llm_entity_judge.py \
  --entities batch1.json \
  --output results_batch1.csv

# Lote 2
python llm_entity_judge.py \
  --entities batch2.json \
  --output results_batch2.csv

# Combinar después
cat results_batch*.csv > results_final.csv
```

### 3. Monitorear Progreso

El pipeline muestra progreso en tiempo real:

```
[1/6] Verificando Ollama...
  ✓ Ollama está instalado
  ✓ Modelo: gemma3:270m

[2/6] Cargando reglas de anotación...
  ✓ 29 tipos de entidades cargados

[3/6] Cargando entidades detectadas...
  ✓ 150 entidades cargadas

[4/6] Agrupando entidades por documento...
  ✓ 10 documentos con entidades

[5/6] Procesando entidades con LLM...
  Documento 1/10: doc_001
    ✓ Documento cargado (2500 caracteres)
    [1/15] NOMBRE_SUJETO_ASISTENCIA: 'García' ✓ TRUE
    [2/15] FECHAS: '25/12/2020' ✓ TRUE
    ...
```

## 🎓 Próximos Pasos

1. **Ejecuta tests**: `python tests/test_ollama_setup.py`
2. **Prueba con ejemplos**: Usa `examples/entities_example.json`
3. **Evalúa tus datos**: Apunta al JSON de tus entidades detectadas
4. **Analiza resultados**: Abre el CSV en Excel/pandas para análisis

## 📧 Soporte

Para problemas o preguntas específicas de Ollama:
- Ollama GitHub: https://github.com/ollama/ollama/issues
- Ollama Discord: https://discord.gg/ollama

Para problemas del pipeline, revisa `TECHNICAL_DOCUMENTATION.md`.

---

**Versión**: 2.0 (Ollama)  
**Última actualización**: Noviembre 2024  
**Modelo**: gemma3:270m
