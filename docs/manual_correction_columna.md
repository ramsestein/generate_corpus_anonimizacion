# Columna `manual_correction` en el Pipeline

## Resumen de Cambios

Se ha añadido soporte para la columna `manual_correction` en el PASO 1 del pipeline de evaluación con LLM (`llm_judge_pipeline.py`).

## ¿Qué es `manual_correction`?

Es una columna opcional que almacena correcciones o etiquetas manuales realizadas por humanos sobre las entidades detectadas. Esta columna:

- ✅ Se carga automáticamente si existe en el CSV de entrada
- ✅ Se crea automáticamente con valor vacío (`""`) si no existe
- ✅ Se preserva durante todo el pipeline (carga, unificación, exportación)
- ✅ Se incluye en las exportaciones finales (JSON) para cálculo de métricas
- ⚠️ **NO se envía NUNCA al LLM** (no contamina prompts ni contexto)

## Modificaciones Realizadas

### 1. Clase `Entity` (línea ~100)
```python
@dataclass
class Entity:
    # ... campos existentes ...
    manual_correction: str = ""  # NUEVO: valor por defecto vacío
```

### 2. Función `load_csv_detections` (línea ~195)
- Detecta automáticamente si el CSV incluye la columna `manual_correction`
- Si existe: carga el valor de cada fila
- Si no existe: asigna `""` por defecto
- Imprime mensaje informativo cuando detecta la columna

### 3. Función `merge_entities` (línea ~430)
- Cuando se unifican fragmentos consecutivos, se preserva el `manual_correction` del **primer fragmento**
- Esto asegura que la corrección manual no se pierde durante la unificación

### 4. Documentación (línea ~55)
- Se añadió sección explicativa en el docstring principal
- Advierte explícitamente que esta columna NO debe enviarse al LLM

## Formato del CSV

### CSV sin `manual_correction` (retrocompatible)
```csv
doc_id,etiqueta,modelo_detector,texto_detectado,confianza,posicion_inicio,posicion_fin
DOC001,NOMBRE_SUJETO_ASISTENCIA,CARMEN,Juan,0.95,100,104
```
→ Se crea automáticamente con valor `""`

### CSV con `manual_correction`
```csv
doc_id,etiqueta,modelo_detector,texto_detectado,confianza,posicion_inicio,posicion_fin,manual_correction
DOC001,NOMBRE_SUJETO_ASISTENCIA,CARMEN,Juan,0.95,100,104,CORRECTO
DOC001,FECHAS,CARMEN,26/7,0.88,200,204,
DOC002,NUMERO_IDENTIF,MEDDOCAN,G045,0.92,50,54,REVISAR
```

## Valores Recomendados

Puedes usar cualquier string, pero se recomienda:

- `""` (vacío): Sin corrección manual aún
- `"CORRECTO"` o `"TRUE"`: La detección es correcta
- `"INCORRECTO"` o `"FALSE"`: La detección es incorrecta
- `"REVISAR"`: Necesita revisión humana adicional
- `"PARCIAL"`: Correcto pero incompleto
- O cualquier otra etiqueta específica de tu flujo

## Ejemplo de Uso

```bash
# Preprocesar CSV sin manual_correction (retrocompatible)
python llm_judge_pipeline.py preprocess \
  --csv detecciones.csv \
  --output entidades_procesadas.json

# Preprocesar CSV con manual_correction
python llm_judge_pipeline.py preprocess \
  --csv detecciones_revisadas.csv \
  --output entidades_procesadas.json
```

## JSON de Salida

El JSON exportado incluirá `manual_correction` en cada entidad:

```json
{
  "entities": [
    {
      "doc_id": "DOC001",
      "label": "NUMERO_IDENTIF",
      "text": "G045",
      "start": 100,
      "end": 104,
      "unified": true,
      "manual_correction": "TRUE"
    }
  ]
}
```

## Unificación de Fragmentos

Cuando se unifican fragmentos consecutivos:

**Entrada (CSV):**
```csv
doc_id,etiqueta,texto_detectado,posicion_inicio,posicion_fin,manual_correction
DOC001,NUMERO_IDENTIF,G,100,101,TRUE
DOC001,NUMERO_IDENTIF,045,101,104,
```

**Salida (JSON):**
```json
{
  "text": "G045",
  "start": 100,
  "end": 104,
  "unified": true,
  "manual_correction": "TRUE"  // Se preserva del primer fragmento
}
```

## Pasos Siguientes del Pipeline

⚠️ **IMPORTANTE**: Los pasos posteriores (2-6) que invoquen al LLM deben:

1. **Excluir explícitamente** `manual_correction` al construir prompts
2. **No incluir** este campo en el contexto enviado al modelo
3. **Preservar** el campo en dataframes/estructuras intermedias
4. **Incluir** el campo en métricas finales (Precision/Recall/F1)

## Tests Incluidos

Se han creado 3 archivos de test en `examples/`:

1. `test_manual_correction_sin_columna.csv` - Sin la columna (retrocompatibilidad)
2. `test_manual_correction_con_columna.csv` - Con la columna y valores variados
3. `test_fragmentos_con_manual.csv` - Fragmentos que se unifican con correcciones

Para ejecutar los tests:
```bash
# Test sin columna
python llm_judge_pipeline.py preprocess \
  --csv examples/test_manual_correction_sin_columna.csv \
  --output outputs/test_sin_columna.json

# Test con columna
python llm_judge_pipeline.py preprocess \
  --csv examples/test_manual_correction_con_columna.csv \
  --output outputs/test_con_columna.json

# Test de unificación
python llm_judge_pipeline.py preprocess \
  --csv examples/test_fragmentos_con_manual.csv \
  --output outputs/test_fragmentos_unificados.json
```

## Retrocompatibilidad

✅ **100% retrocompatible**:
- Los CSV existentes sin `manual_correction` funcionan sin modificaciones
- El comportamiento del pipeline no cambia para datasets antiguos
- Solo se añade funcionalidad, no se elimina ni cambia nada existente
