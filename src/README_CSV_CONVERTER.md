# Convertidor CSV a JSON para Pipeline de Anonimización

## Descripción

Script que convierte archivos CSV de detecciones (formato `detecciones_detalladas.csv`) al formato JSON requerido por el pipeline de anonimización.

## Funcionalidades

- ✅ Lee CSV con detecciones de entidades PII
- ✅ **Fusiona entidades continuas** del mismo documento
- ✅ Genera JSON compatible con `run_full_pipeline.py`
- ✅ Calcula estadísticas y metadatos automáticamente

## Uso

```bash
python src/convert_csv_to_pipeline_input.py <csv_input> <json_output> [opciones]
```

### Ejemplos

```bash
# Convertir detecciones con fusión automática
python src/convert_csv_to_pipeline_input.py \
  corpus/step6_validation/aws1-validation/detecciones_detalladas.csv \
  entidades-pipeline.json

# Sin fusionar entidades (mantener todas separadas)
python src/convert_csv_to_pipeline_input.py \
  corpus/step6_validation/aws1-validation/detecciones_detalladas.csv \
  entidades-pipeline.json \
  --no-merge

# Con información detallada
python src/convert_csv_to_pipeline_input.py \
  corpus/step6_validation/aws1-validation/detecciones_detalladas.csv \
  entidades-pipeline.json \
  -v
```

## Formato de Entrada (CSV)

```csv
doc_id,etiqueta,modelo_detector,texto_detectado,confianza,posicion_inicio,posicion_fin
NHC123,NUMERO_IDENTIF,CARMEN,G,0.9997,1803,1804
NHC123,NUMERO_IDENTIF,CARMEN,064,0.9913,1804,1807
```

## Formato de Salida (JSON)

```json
{
  "metadata": {
    "generated_at": "2025-12-02T10:35:18.281909",
    "total_entities": 161,
    "processing_stats": {
      "total_raw_entities": 202,
      "total_unified_entities": 161,
      "entities_merged": 41
    }
  },
  "entities": [
    {
      "doc_id": "NHC123",
      "label": "NUMERO_IDENTIF",
      "model": "CARMEN",
      "text": "G064",
      "confidence": 0.9955,
      "start": 1803,
      "end": 1807,
      "unified": true
    }
  ]
}
```

## Lógica de Fusión

Dos entidades se fusionan si:

- ✅ Pertenecen al **mismo documento**
- ✅ Tienen la **misma etiqueta NER**
- ✅ Sus posiciones son **consecutivas** o tienen un gap mínimo (≤ 2 caracteres)

### Ejemplos de Fusión

| CSV Original | Entidad Fusionada |
|--------------|-------------------|
| `G` (1803-1804) + `064` (1804-1807) | `G064` (1803-1807) |
| `I` (856-857) + `037` (857-860) | `I037` (856-860) |
| `fami` (4591-4596) + `lia` (4596-4599) | `familia` (4591-4599) |

## Opciones

| Opción | Descripción |
|--------|-------------|
| `--no-merge` | Desactiva la fusión de entidades continuas |
| `-v, --verbose` | Muestra información detallada del procesamiento |

## Integración con el Pipeline

Una vez generado el JSON, úsalo como entrada del pipeline:

```bash
# Pipeline completo: SetFit → Dict Filters → LLM
python src/pipeline-nuevos-textos/run_full_pipeline.py \
  --input entidades-pipeline.json \
  --output resultados.json \
  -v

# Evaluación contra ground truth
python src/pipeline-nuevos-textos/evaluate_pipeline.py \
  --results resultados.json \
  --ground-truth entidades-ground-truth.json
```

## Estadísticas Generadas

El script genera automáticamente:

- **Total de entidades** (antes y después de fusionar)
- **Número de fusiones realizadas**
- **Distribución por etiqueta NER**
- **Distribución por modelo detector** (CARMEN/MEDDOCAN)
- **Número de documentos procesados**

## Notas

- El script detecta automáticamente el delimitador del CSV (`,` o `;`)
- Normaliza nombres de columnas automáticamente
- Filtra filas vacías
- Promedia la confianza al fusionar entidades
- Prioriza el modelo CARMEN sobre MEDDOCAN en fusiones
