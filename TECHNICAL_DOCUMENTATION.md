# Pipeline LLM Judge - Documentación Técnica Completa

## 📋 Índice

1. [Visión General](#visión-general)
2. [Arquitectura](#arquitectura)
3. [Módulos Implementados](#módulos-implementados)
4. [Flujo de Datos](#flujo-de-datos)
5. [API Reference](#api-reference)
6. [Configuración](#configuración)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## Visión General

El **Pipeline LLM Judge** es un sistema completo para evaluar automáticamente entidades detectadas en documentos clínicos usando un juez LLM (Large Language Model) real a través de su API oficial.

### Características Principales

- ✅ **Soporte Multi-Proveedor**: OpenAI, Claude, Mistral, Azure OpenAI, etc.
- ✅ **Contexto Dinámico**: Extrae ventanas de texto alrededor de cada entidad
- ✅ **Reglas Personalizadas**: Usa reglas oficiales de anotación por etiqueta
- ✅ **Prompts Optimizados**: Maximiza RECALL mientras mantiene precisión
- ✅ **Manejo de Errores**: Captura y reporta errores por entidad
- ✅ **Procesamiento Batch**: Procesa múltiples documentos y entidades
- ✅ **Exportación CSV**: Resultados compatibles con Excel, pandas, etc.

---

## Arquitectura

### Componentes del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                        PIPELINE LLM JUDGE                    │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │llm_prompts.py│  │llm_entity_  │  │   .env      │
    │             │  │  judge.py   │  │             │
    │- PROMPT_    │  │             │  │- API_KEY    │
    │  CONFIG     │  │- load_llm_  │  │- API_URL    │
    │- load_      │  │  credentials│  │- MODEL_NAME │
    │  entity_    │  │- call_llm() │  │             │
    │  rules()    │  │- extract_   │  │             │
    │- get_entity_│  │  context()  │  │             │
    │  rules_for_ │  │- process_   │  │             │
    │  label()    │  │  entity()   │  │             │
    └─────────────┘  └─────────────┘  └─────────────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │   LLM API Server    │
                  │  (OpenAI/Claude/   │
                  │   Mistral/etc.)     │
                  └────────────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │   Output CSV        │
                  │  - document_id      │
                  │  - entity_id        │
                  │  - keyword          │
                  │  - label            │
                  │  - is_valid         │
                  │  - llm_response     │
                  └────────────────────┘
```

### Flujo de Procesamiento

1. **Inicialización**
   - Carga credenciales desde `.env`
   - Carga reglas de anotación desde `guias-anotacion.json`
   - Valida configuración

2. **Carga de Datos**
   - Lee JSON con entidades detectadas
   - Agrupa entidades por documento

3. **Procesamiento por Documento**
   - Carga documento original (.txt)
   - Para cada entidad:
     - Extrae contexto (ventana alrededor de la entidad)
     - Obtiene reglas de la etiqueta
     - Construye prompts (system + user)
     - Llama al LLM
     - Parsea respuesta (TRUE/FALSE)

4. **Exportación**
   - Guarda resultados en CSV
   - Genera resumen estadístico

---

## Módulos Implementados

### 1. `llm_prompts.py` (120 líneas)

**Propósito**: Define la plantilla de prompt para el juez LLM.

**Contenido Principal**:
```python
PROMPT_CONFIG = {
    "name": "Clasificador de entidad por palabra y contexto",
    "version": "1.0",
    "template": "...",
    "input_variables": ["keyword", "context_text", "label", 
                        "location", "reglas_etiqueta"],
    "output_format": "text",
    "expected_fields": ["TRUE or FALSE"]
}
```

**Funciones Clave**:
- `load_entity_rules()`: Carga reglas desde JSON
- `get_entity_rules_for_label()`: Obtiene reglas de una etiqueta específica
- `get_prompt_config()`: Devuelve configuración del prompt

### 2. `llm_entity_judge.py` (450+ líneas)

**Propósito**: Implementa el pipeline completo de evaluación con LLM.

**Secciones**:

#### A. Carga de Credenciales
```python
def load_llm_credentials() -> Tuple[str, str, str, int, float]:
    """Carga credenciales desde .env"""
    # Retorna: api_key, api_url, model_name, timeout, temperature
```

#### B. Llamada al LLM
```python
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Realiza llamada HTTP POST al LLM"""
    # Payload estándar ChatCompletion
    # Headers con Authorization Bearer
    # Timeout configurable
    # Extrae content de la respuesta
```

#### C. Extracción de Contexto
```python
def extract_context_window(
    document_text: str,
    entity_start: int,
    entity_end: int,
    left_window: int = 80,
    right_window: int = 80
) -> str:
    """Extrae ventana de contexto alrededor de la entidad"""
```

#### D. Procesamiento de Entidades
```python
def process_entity(
    entity: Dict,
    document_text: str,
    rules_text: str,
    left_window: int = 80,
    right_window: int = 80
) -> Dict:
    """
    Procesa una entidad completa:
    1. Extrae contexto
    2. Construye prompts
    3. Llama al LLM
    4. Parsea respuesta
    5. Maneja errores
    """
```

#### E. Pipeline Principal
```python
def run_llm_entity_judgment(
    entities_json_path: str,
    docs_base_path: str,
    rules_file: str = "guias-anotacion.json",
    output_csv: str = "llm_entity_judgments.csv",
    left_window: int = 80,
    right_window: int = 80
) -> None:
    """Ejecuta el pipeline completo end-to-end"""
```

### 3. `.env.example`

Plantilla de configuración con variables requeridas:
```env
LLM_API_KEY=your_api_key_here
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL_NAME=gpt-4o
LLM_TIMEOUT=60
LLM_TEMPERATURE=0.0
```

### 4. `test_llm_pipeline.py` (300+ líneas)

Suite de tests automatizados:
- Test de imports
- Test de PROMPT_CONFIG
- Test de carga de reglas
- Test de extracción de contexto
- Test de parsing de respuestas
- Test de construcción de prompts
- Test de verificación de .env

---

## Flujo de Datos

### Entrada: JSON de Entidades

```json
{
  "entities": [
    {
      "document_id": "doc_001",
      "id": "ent_001",
      "text": "García",
      "label": "NOMBRE_SUJETO_ASISTENCIA",
      "start": 12,
      "end": 18
    }
  ]
}
```

### Procesamiento Interno

1. **Carga de documento**: `doc_001.txt`
   ```
   El paciente García tiene 45 años y acude por dolor abdominal...
   ```

2. **Extracción de contexto** (ventana: 80-80):
   ```
   El paciente García tiene 45 años y acude por dolor abdominal
   ```

3. **Obtención de reglas** para `NOMBRE_SUJETO_ASISTENCIA`:
   ```
   1. Se anota cualquier nombre, apellidos o apodos del paciente
   2. Se anotan nombres completos o parciales
   3. Se anotan iniciales si se refieren al paciente
   ...
   ```

4. **Construcción de prompts**:
   - **System Prompt**: Instrucciones + reglas completas
   - **User Prompt**: Palabra + contexto + etiqueta

5. **Llamada al LLM**:
   ```http
   POST https://api.openai.com/v1/chat/completions
   {
     "model": "gpt-4o",
     "messages": [
       {"role": "system", "content": "..."},
       {"role": "user", "content": "..."}
     ],
     "temperature": 0.0,
     "max_tokens": 10
   }
   ```

6. **Respuesta del LLM**:
   ```json
   {
     "choices": [{
       "message": {
         "content": "TRUE"
       }
     }]
   }
   ```

7. **Parsing**: `"TRUE"` → `is_valid = True`

### Salida: CSV de Resultados

```csv
document_id,entity_id,keyword,label,start,end,context,llm_response,is_valid,status
doc_001,ent_001,García,NOMBRE_SUJETO_ASISTENCIA,12,18,"El paciente García tiene...",TRUE,True,success
```

---

## API Reference

### `llm_prompts.py`

#### `load_entity_rules(rules_file: str = "guias-anotacion.json") -> Dict[str, List[str]]`

Carga las reglas de anotación desde archivo JSON.

**Args**:
- `rules_file`: Ruta al archivo JSON con reglas

**Returns**:
- Dict con reglas por tipo de entidad

**Raises**:
- `FileNotFoundError`: Si el archivo no existe
- `json.JSONDecodeError`: Si el JSON es inválido

**Ejemplo**:
```python
rules = load_entity_rules()
# {'NOMBRE_SUJETO_ASISTENCIA': [...], 'FECHAS': [...], ...}
```

---

#### `get_entity_rules_for_label(label: str, rules_file: str = "guias-anotacion.json") -> str`

Obtiene las reglas formateadas para una etiqueta específica.

**Args**:
- `label`: Etiqueta de la entidad (ej: "NOMBRE_SUJETO_ASISTENCIA")
- `rules_file`: Archivo JSON con reglas

**Returns**:
- String con reglas formateadas numeradas

**Raises**:
- `KeyError`: Si la etiqueta no existe

**Ejemplo**:
```python
rules = get_entity_rules_for_label("FECHAS")
# "1. Se anotan todas las fechas...\n2. ..."
```

---

#### `get_prompt_config() -> Dict`

Devuelve la configuración de la plantilla del prompt.

**Returns**:
- Dict con configuración completa

**Ejemplo**:
```python
config = get_prompt_config()
template = config["template"]
variables = config["input_variables"]
```

---

### `llm_entity_judge.py`

#### `load_llm_credentials() -> Tuple[str, str, str, int, float]`

Carga las credenciales del LLM desde `.env`.

**Returns**:
- Tuple: (api_key, api_url, model_name, timeout, temperature)

**Raises**:
- `FileNotFoundError`: Si no existe `.env`
- `ValueError`: Si falta alguna variable requerida

**Ejemplo**:
```python
api_key, api_url, model, timeout, temp = load_llm_credentials()
```

---

#### `call_llm(system_prompt: str, user_prompt: str) -> str`

Realiza una llamada al LLM usando la API configurada.

**Args**:
- `system_prompt`: Prompt de sistema con instrucciones
- `user_prompt`: Prompt de usuario con consulta específica

**Returns**:
- Respuesta del LLM como string (ej: "TRUE" o "FALSE")

**Raises**:
- `requests.RequestException`: Si hay error HTTP
- `ValueError`: Si la respuesta no tiene formato esperado

**Ejemplo**:
```python
response = call_llm(
    system_prompt="Eres un experto en...",
    user_prompt="¿Es 'García' un nombre?"
)
# "TRUE"
```

---

#### `extract_context_window(...) -> str`

Extrae ventana de contexto alrededor de la entidad.

**Args**:
- `document_text`: Texto completo del documento
- `entity_start`: Posición inicio (caracteres)
- `entity_end`: Posición fin (caracteres)
- `left_window`: Caracteres antes (default: 80)
- `right_window`: Caracteres después (default: 80)

**Returns**:
- String con contexto extraído

**Ejemplo**:
```python
context = extract_context_window(
    document_text="El paciente García tiene 45 años...",
    entity_start=12,
    entity_end=18,
    left_window=20,
    right_window=20
)
# "El paciente García tiene 45 años"
```

---

#### `run_llm_entity_judgment(...) -> None`

Ejecuta el pipeline completo de evaluación.

**Args**:
- `entities_json_path`: Ruta al JSON con entidades
- `docs_base_path`: Directorio con documentos
- `rules_file`: Archivo JSON con reglas (default: "guias-anotacion.json")
- `output_csv`: Archivo CSV de salida (default: "llm_entity_judgments.csv")
- `left_window`: Ventana izquierda (default: 80)
- `right_window`: Ventana derecha (default: 80)

**Ejemplo**:
```python
run_llm_entity_judgment(
    entities_json_path="entities.json",
    docs_base_path="documents/",
    output_csv="results.csv"
)
```

---

## Configuración

### Variables de Entorno (`.env`)

| Variable | Requerida | Default | Descripción |
|----------|-----------|---------|-------------|
| `LLM_API_KEY` | ✅ Sí | - | API key del proveedor |
| `LLM_API_URL` | ✅ Sí | - | URL del endpoint API |
| `LLM_MODEL_NAME` | ✅ Sí | - | Nombre del modelo |
| `LLM_TIMEOUT` | ❌ No | 60 | Timeout en segundos |
| `LLM_TEMPERATURE` | ❌ No | 0.0 | Temperatura del modelo |

### Ejemplo de Configuración por Proveedor

#### OpenAI
```env
LLM_API_KEY=sk-proj-...
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL_NAME=gpt-4o
```

#### Claude (Anthropic)
```env
LLM_API_KEY=sk-ant-...
LLM_API_URL=https://api.anthropic.com/v1/messages
LLM_MODEL_NAME=claude-3-opus-20240229
```

#### Mistral AI
```env
LLM_API_KEY=...
LLM_API_URL=https://api.mistral.ai/v1/chat/completions
LLM_MODEL_NAME=mistral-large-latest
```

---

## Testing

### Ejecutar Suite de Tests

```bash
python tests/test_llm_pipeline.py
```

### Tests Incluidos

1. **Test de Imports**: Verifica que todos los módulos se importen correctamente
2. **Test de PROMPT_CONFIG**: Valida estructura y placeholders
3. **Test de Reglas**: Verifica carga de reglas desde JSON
4. **Test de Contexto**: Valida extracción de ventanas
5. **Test de Parsing**: Verifica interpretación de respuestas
6. **Test de Prompts**: Valida construcción de prompts
7. **Test de .env**: Verifica existencia del archivo

### Resultado Esperado

```
Total: 7 tests
  ✓ Pasados: 6-7
  ✗ Fallidos: 0-1 (solo .env si no está configurado)
  Porcentaje: 85.7-100%
```

---

## Troubleshooting

### Problema: "No se encontró el archivo .env"

**Causa**: El archivo `.env` no existe.

**Solución**:
```bash
cp .env.example .env
# Editar .env con tus credenciales
```

---

### Problema: "LLM_API_KEY no está configurada"

**Causa**: La API key tiene el valor por defecto o está vacía.

**Solución**:
Edita `.env` y configura tu API key real:
```env
LLM_API_KEY=sk-real-api-key-here
```

---

### Problema: "Timeout al llamar al LLM"

**Causa**: El LLM tarda más que el timeout configurado.

**Solución**:
Aumenta `LLM_TIMEOUT` en `.env`:
```env
LLM_TIMEOUT=120
```

---

### Problema: "Respuesta del LLM no tiene el formato esperado"

**Causa**: La URL de la API no es correcta para tu proveedor.

**Solución**:
Verifica la URL correcta para tu proveedor:
- OpenAI: `.../v1/chat/completions`
- Claude: `.../v1/messages`
- Mistral: `.../v1/chat/completions`

---

### Problema: "Documento no encontrado"

**Causa**: El archivo del documento no existe en `--docs`.

**Solución**:
- Verifica que los documentos existan
- Nombres deben ser: `{document_id}.txt`
- Estructura: `docs_dir/doc_id.txt`

---

### Problema: "Etiqueta desconocida"

**Causa**: La etiqueta no existe en `guias-anotacion.json`.

**Solución**:
- Verifica que la etiqueta esté en el JSON de reglas
- Revisa el nombre exacto (case-sensitive)

---

## Métricas y Estimaciones

### Tokens por Entidad

- System Prompt: ~1500 tokens
- User Prompt: ~150 tokens
- **Total por entidad**: ~1650 tokens

### Costos Estimados (GPT-4)

- Precio: $0.03 por 1K tokens de input
- 1 entidad: ~$0.05
- 100 entidades: ~$5
- 1000 entidades: ~$50

### Tiempo de Procesamiento

- Llamada LLM: ~2-5 segundos
- Extracción contexto: <0.1 segundos
- **Total por entidad**: ~2-5 segundos
- **1000 entidades**: ~30-80 minutos

---

## Mejoras Futuras

1. **Rate Limiting**: Implementar control de tasa de requests
2. **Retry Logic**: Reintentos con backoff exponencial
3. **Caching**: Cache de respuestas para entidades duplicadas
4. **Batch Processing**: Agrupar múltiples entidades en una llamada
5. **Async I/O**: Procesamiento asíncrono para mayor velocidad
6. **Métricas en Tiempo Real**: Dashboard de progreso y estadísticas
7. **Soporte Multi-Thread**: Paralelización del procesamiento

---

## Contacto y Soporte

Para preguntas, bugs o sugerencias, contacta al equipo de desarrollo.

**Versión**: 1.0  
**Última actualización**: Noviembre 2024  
**Autor**: Pipeline Team
