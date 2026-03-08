# Reporte de Detección de Contenido Generado por IA

## Resumen General

- **Total de documentos analizados**: 14035
- **Clasificados como HUMANO**: 12964 (92.4%)
- **Clasificados como IA**: 1071 (7.6%)
- **Probabilidad promedio de ser humano**: 78.0%

## Estadísticas por Métrica

### Perplejidad
- **Media**: 113.67
- **Rango esperado humano**: 50-150
- **Rango esperado IA**: 20-50

### Burstiness
- **Media**: 0.379
- **Rango esperado humano**: >0.4
- **Rango esperado IA**: <0.3

### Type-Token Ratio (TTR)
- **Media**: 0.653
- **Rango esperado humano**: 0.6-0.8
- **Rango esperado IA**: 0.4-0.6

### Hapax Legomena Ratio
- **Media**: 54.1%
- **Rango esperado humano**: 40-60%
- **Rango esperado IA**: 20-40%

## Tabla de Resultados

Ver archivo `tabla_resumen.csv` para resultados detallados por documento.

## Notas

- Las métricas están basadas en literatura científica estándar
- La clasificación usa un sistema de votación (voting ensemble)
- La probabilidad de ser humano se calcula como promedio ponderado de métricas
- Algunas métricas pueden requerir modelos adicionales para mayor precisión
