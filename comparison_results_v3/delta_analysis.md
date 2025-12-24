# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2025-12-24 16:55:14

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline A (Base + SetFit A)
- **Recall (Seguridad):** Pipeline B (Base + SetFit B)
- **F1-Score (Balance):** Pipeline B (Base + SetFit B)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline A (Base + SetFit A) filtró, Pipeline B (Base + SetFit B) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `5.7` | NHC103087_episodio1008807132 |
| 2 | `5.7` | NHC103087_episodio1008807132 |
| 3 | `pur3` | NHC4719468_episodio1008460753 |
| 4 | `pur 3` | NHC4722141_episodio1008643470 |
| 5 | `18/ 3/24` | NHC4722141_episodio1008643470 |
| 6 | `3-10-202 1` | NHC4723257_episodio1008504125 |
| 7 | `16/0 3/2024` | NHC4724180_episodio1008623907 |
| 8 | `familiar` | NHC4724180_episodio1008623907 |
| 9 | `familiar` | NHC4724180_episodio1008623907 |
| 10 | `madre` | NHC4731874_episodio1008794146 |

### Pipeline B (Base + SetFit B) filtró, Pipeline A (Base + SetFit A) dejó pasar
**Total:** 0 ejemplos

*No hay ejemplos*

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline A (Base + SetFit A) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `20.8.20` | NHC102219_episodio1008744732 |
| 2 | `65` | NHC102219_episodio1008744732 |
| 3 | `her` | NHC102219_episodio1008744732 |
| 4 | `her` | NHC102219_episodio1008744732 |
| 5 | `mano` | NHC102219_episodio1008744732 |
| 6 | `65` | NHC102219_episodio1008744732 |
| 7 | `045` | NHC102219_episodio1008744732 |
| 8 | `her` | NHC102219_episodio1008744732 |
| 9 | `20.8.20` | NHC102219_episodio1008744732 |
| 10 | `se` | NHC102219_episodio1008744732 |

### Pipeline B (Base + SetFit B) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `65` | NHC102219_episodio1008744732 |
| 2 | `her` | NHC102219_episodio1008744732 |
| 3 | `her` | NHC102219_episodio1008744732 |
| 4 | `mano` | NHC102219_episodio1008744732 |
| 5 | `65` | NHC102219_episodio1008744732 |
| 6 | `045` | NHC102219_episodio1008744732 |
| 7 | `her` | NHC102219_episodio1008744732 |
| 8 | `se` | NHC102219_episodio1008744732 |
| 9 | `ptiembre 2020` | NHC102219_episodio1008744732 |
| 10 | `mano` | NHC102219_episodio1008744732 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

