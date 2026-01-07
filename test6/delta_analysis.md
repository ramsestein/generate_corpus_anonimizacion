# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2026-01-07 15:40:05

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline (test50docs)
- **Recall (Seguridad):** Pipeline (test50docs)
- **F1-Score (Balance):** Pipeline (test50docs)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline (test50docs) filtró, Pipeline (test50docs) dejó pasar
**Total:** 0 ejemplos

*No hay ejemplos*

### Pipeline (test50docs) filtró, Pipeline (test50docs) dejó pasar
**Total:** 0 ejemplos

*No hay ejemplos*

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline (test50docs) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `22.07.2024` | NHC107102_episodio1008411916 |
| 2 | `62a` | NHC302538_episodio1008414306 |
| 3 | `09` | NHC5583668_episodio1008559582 |
| 4 | `18 MAR 2024` | NHC5585300_episodio1008687210 |
| 5 | `Vi` | NHC5580063_episodio1008372518 |
| 6 | `15/08` | NHC4034835_episodio1008530518 |
| 7 | `16/07/2024` | NHC5574823_episodio1008341183 |
| 8 | `zo/24` | NHC4034835_episodio1008530518 |
| 9 | `año 2017` | NHC5569798_episodio1008326489 |
| 10 | `25/7` | NHC107102_episodio1008411916 |

### Pipeline (test50docs) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `22.07.2024` | NHC107102_episodio1008411916 |
| 2 | `62a` | NHC302538_episodio1008414306 |
| 3 | `09` | NHC5583668_episodio1008559582 |
| 4 | `18 MAR 2024` | NHC5585300_episodio1008687210 |
| 5 | `Vi` | NHC5580063_episodio1008372518 |
| 6 | `15/08` | NHC4034835_episodio1008530518 |
| 7 | `16/07/2024` | NHC5574823_episodio1008341183 |
| 8 | `zo/24` | NHC4034835_episodio1008530518 |
| 9 | `año 2017` | NHC5569798_episodio1008326489 |
| 10 | `25/7` | NHC107102_episodio1008411916 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

