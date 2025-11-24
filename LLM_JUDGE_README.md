# Pipeline de Evaluación con Juez LLM

Este módulo implementa la evaluación automática de entidades detectadas usando un juez LLM real a través de su API oficial.

## 📋 Configuración Inicial

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará:
- `python-dotenv`: Para cargar variables de entorno
- `requests`: Para llamadas HTTP a la API del LLM

### 2. Configurar Credenciales del LLM

1. Copia el archivo de ejemplo:
   ```bash
   cp .env.example .env
   ```

2. Edita `.env` con tus credenciales reales:
   ```env
   LLM_API_KEY=tu_api_key_real_aqui
   LLM_API_URL=https://api.openai.com/v1/chat/completions
   LLM_MODEL_NAME=gpt-4o
   LLM_TIMEOUT=60
   LLM_TEMPERATURE=0.0
   ```

### 3. Proveedores Soportados

El módulo funciona con cualquier API compatible con el formato ChatCompletion:

#### OpenAI
```env
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL_NAME=gpt-4o
```

#### Claude (Anthropic)
```env
LLM_API_URL=https://api.anthropic.com/v1/messages
LLM_MODEL_NAME=claude-3-opus-20240229
```

#### Mistral AI
```env
LLM_API_URL=https://api.mistral.ai/v1/chat/completions
LLM_MODEL_NAME=mistral-large-latest
```

#### Azure OpenAI
```env
LLM_API_URL=https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT/chat/completions?api-version=2023-05-15
LLM_MODEL_NAME=gpt-4
```

## 🚀 Uso

### Comando Básico

```bash
python src/pipeline-nuevos-textos/llm_entity_judge.py \
  --entities path/to/entities.json \
  --docs path/to/documents/ \
  --output results.csv
```

### Parámetros

- `--entities` (requerido): Ruta al JSON con entidades detectadas
- `--docs` (requerido): Directorio base con documentos originales (.txt)
- `--rules-file` (opcional): Archivo JSON con reglas de anotación (default: `guias-anotacion.json`)
- `--output` (opcional): Archivo CSV de salida (default: `llm_entity_judgments.csv`)
- `--left-window` (opcional): Caracteres de contexto a la izquierda (default: 80)
- `--right-window` (opcional): Caracteres de contexto a la derecha (default: 80)

### Ejemplos

#### Ejemplo 1: Evaluación Básica
```bash
python src/pipeline-nuevos-textos/llm_entity_judge.py \
  --entities corpus/output/aws2/entities.json \
  --docs corpus/output/aws2/documents \
  --output results/aws2_evaluation.csv
```

#### Ejemplo 2: Con Ventanas de Contexto Amplias
```bash
python src/pipeline-nuevos-textos/llm_entity_judge.py \
  --entities corpus/output/aws2/entities.json \
  --docs corpus/output/aws2/documents \
  --output results/aws2_evaluation_wide_context.csv \
  --left-window 150 \
  --right-window 150
```

#### Ejemplo 3: Con Reglas Personalizadas
```bash
python src/pipeline-nuevos-textos/llm_entity_judge.py \
  --entities corpus/output/aws2/entities.json \
  --docs corpus/output/aws2/documents \
  --rules-file custom_rules.json \
  --output results/aws2_custom_rules.csv
```

## 📊 Formato de Entrada

### JSON de Entidades

El archivo JSON de entrada debe contener las entidades detectadas. Formato esperado:

```json
[
  {
    "document_id": "doc_001",
    "id": "ent_001",
    "text": "García",
    "label": "NOMBRE_SUJETO_ASISTENCIA",
    "start": 15,
    "end": 21
  },
  {
    "document_id": "doc_001",
    "id": "ent_002",
    "text": "25/12/2020",
    "label": "FECHAS",
    "start": 50,
    "end": 60
  }
]
```

O con metadata:

```json
{
  "metadata": {
    "source": "aws2",
    "date": "2024-11-18"
  },
  "entities": [
    {
      "document_id": "doc_001",
      "id": "ent_001",
      "text": "García",
      "label": "NOMBRE_SUJETO_ASISTENCIA",
      "start": 15,
      "end": 21
    }
  ]
}
```

### Documentos

Los documentos deben estar en formato texto plano (.txt) con el nombre: `{document_id}.txt`

Estructura de directorio:
```
corpus/output/aws2/documents/
├── doc_001.txt
├── doc_002.txt
└── doc_003.txt
```

## 📄 Formato de Salida

El CSV de salida contiene los resultados de la evaluación:

| Campo | Descripción |
|-------|-------------|
| `document_id` | ID del documento |
| `entity_id` | ID de la entidad |
| `keyword` | Palabra evaluada |
| `label` | Tipo de entidad |
| `start` | Posición de inicio (caracteres) |
| `end` | Posición de fin (caracteres) |
| `context` | Texto contextual extraído |
| `llm_response` | Respuesta cruda del LLM |
| `is_valid` | TRUE/FALSE/None (resultado parseado) |
| `status` | Estado del procesamiento |

### Ejemplo de Salida

```csv
document_id,entity_id,keyword,label,start,end,context,llm_response,is_valid,status
doc_001,ent_001,García,NOMBRE_SUJETO_ASISTENCIA,15,21,"El paciente García tiene 45 años...",TRUE,True,success
doc_001,ent_002,25/12/2020,FECHAS,50,60,"Fecha de ingreso: 25/12/2020...",TRUE,True,success
doc_002,ent_003,familia,FAMILIARES_SUJETO_ASISTENCIA,100,107,"La familia del paciente...",FALSE,False,success
```

## 🔧 Flujo del Pipeline

1. **Carga de Credenciales**: Lee API key, URL y modelo desde `.env`
2. **Carga de Reglas**: Lee las reglas de anotación desde `guias-anotacion.json`
3. **Carga de Entidades**: Lee el JSON con entidades detectadas
4. **Agrupación**: Agrupa entidades por documento
5. **Procesamiento**: Para cada entidad:
   - Carga el documento original
   - Extrae ventana de contexto alrededor de la entidad
   - Obtiene reglas específicas de la etiqueta
   - Construye prompts (system + user)
   - Llama al LLM
   - Parsea respuesta (TRUE/FALSE)
6. **Guardado**: Exporta resultados a CSV

## 🎯 Características

- ✅ **Soporte Multi-Proveedor**: OpenAI, Claude, Mistral, Azure OpenAI, etc.
- ✅ **Contexto Dinámico**: Extrae ventanas de texto alrededor de cada entidad
- ✅ **Reglas Personalizadas**: Carga reglas específicas por etiqueta
- ✅ **Manejo de Errores**: Captura y reporta errores por entidad
- ✅ **Progreso en Tiempo Real**: Muestra estado de procesamiento
- ✅ **Resumen Estadístico**: Genera resumen de evaluación al final
- ✅ **Formato CSV**: Salida compatible con Excel, pandas, etc.

## ⚠️ Consideraciones

### Costos
Cada entidad genera una llamada al LLM. Estima costos según tu proveedor:
- **OpenAI GPT-4**: ~$0.03 por 1K tokens
- **Claude 3**: ~$0.015 por 1K tokens
- **Mistral Large**: ~$0.008 por 1K tokens

Para 1000 entidades con prompts de ~500 tokens cada una:
- Total tokens: ~500K tokens
- Costo estimado (GPT-4): ~$15 USD

### Límites de Rate
Considera los límites de tu proveedor:
- OpenAI: 3500 requests/min (tier 1)
- Claude: 4000 requests/min
- Mistral: 5000 requests/min

Para grandes volúmenes, implementa:
- Batching de requests
- Rate limiting (sleep entre llamadas)
- Retry logic con backoff exponencial

### Timeout
Ajusta `LLM_TIMEOUT` según la complejidad de tus prompts:
- Prompts simples: 30s
- Prompts con muchas reglas: 60s
- Prompts muy largos: 120s

## 🐛 Troubleshooting

### Error: "No se encontró el archivo .env"
**Solución**: Copia `.env.example` a `.env` y configura tus credenciales.

### Error: "LLM_API_KEY no está configurada"
**Solución**: Asegúrate de que `.env` contenga tu API key real (no el placeholder).

### Error: "Timeout al llamar al LLM"
**Solución**: Aumenta `LLM_TIMEOUT` en `.env` o reduce el tamaño del contexto.

### Error: "Respuesta del LLM no tiene el formato esperado"
**Solución**: Verifica que la URL de la API sea correcta para tu proveedor.

### Error: "Documento no encontrado"
**Solución**: Verifica que los documentos existan en `--docs` con el nombre `{document_id}.txt`.

## 📚 API Programática

También puedes usar el módulo directamente en Python:

```python
from llm_entity_judge import run_llm_entity_judgment

run_llm_entity_judgment(
    entities_json_path="corpus/output/aws2/entities.json",
    docs_base_path="corpus/output/aws2/documents",
    rules_file="guias-anotacion.json",
    output_csv="results/evaluation.csv",
    left_window=100,
    right_window=100
)
```

O funciones individuales:

```python
from llm_entity_judge import call_llm, load_llm_credentials

# Cargar credenciales
api_key, api_url, model, timeout, temp = load_llm_credentials()

# Llamar al LLM
response = call_llm(
    system_prompt="Eres un experto en...",
    user_prompt="¿Es 'García' un nombre válido?"
)

print(response)  # "TRUE" o "FALSE"
```

## 📝 Notas

- El módulo usa `temperature=0.0` por defecto para respuestas determinísticas
- El `max_tokens` está limitado a 10 (solo necesitamos "TRUE" o "FALSE")
- Los prompts se construyen usando `PROMPT_CONFIG` de `llm_prompts.py`
- El contexto extraído incluye caracteres antes y después de la entidad

## 🔒 Seguridad

- **NUNCA** commitees el archivo `.env` al repositorio
- El archivo `.env` está en `.gitignore`
- Las API keys son sensibles: guárdalas de forma segura
- Considera usar servicios de gestión de secretos en producción

## 📧 Soporte

Para problemas o preguntas, contacta al equipo de desarrollo.
