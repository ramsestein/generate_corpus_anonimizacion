# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2025-12-29 13:55:53

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline A (Base + SetFit A)
- **Recall (Seguridad):** Pipeline B (Base + SetFit B)
- **F1-Score (Balance):** Pipeline B (Base + SetFit B)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline A (Base + SetFit A) filtró, Pipeline B (Base + SetFit B) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `hij o` | NHC5625947_episodio1008584588 |
| 2 | `9/10` | NHC5598922_episodio1008676554 |
| 3 | `tía` | NHC5592970_episodio1008441028 |
| 4 | `1/10` | NHC4902665_episodio1008537292 |
| 5 | `23.` | NHC5613578_episodio1008593753 |
| 6 | `H. Clínic o` | NHC4882263_episodio1008521208 |
| 7 | `12/6` | NHC5625264_episodio1008609956 |
| 8 | `16/12` | NHC5587511_episodio1008498342 |
| 9 | `familiar` | NHC5592969_episodio1008441824 |
| 10 | `madre` | NHC5612496_episodio1008791460 |

### Pipeline B (Base + SetFit B) filtró, Pipeline A (Base + SetFit A) dejó pasar
**Total:** 0 ejemplos

*No hay ejemplos*

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline A (Base + SetFit A) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `20.05.2024` | NHC4919653_episodio1008333252 |
| 2 | `05/2022` | NHC4911293_episodio1008509153 |
| 3 | `65` | NHC5597199_episodio1008625628 |
| 4 | `27.08.2024` | NHC4911293_episodio1008509153 |
| 5 | `15.07.2024` | NHC5044954_episodio1008368022 |
| 6 | `21.08.2024` | NHC5587640_episodio1008514321 |
| 7 | `ón` | NHC5612887_episodio1008610655 |
| 8 | `6` | NHC5613100_episodio1008534652 |
| 9 | `rera` | NHC5621392_episodio1008563041 |
| 10 | `15.08.2023` | NHC5618903_episodio1008538341 |

### Pipeline B (Base + SetFit B) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `65` | NHC5597199_episodio1008625628 |
| 2 | `ón` | NHC5612887_episodio1008610655 |
| 3 | `6` | NHC5613100_episodio1008534652 |
| 4 | `rera` | NHC5621392_episodio1008563041 |
| 5 | `hi` | NHC4876654_episodio1008764171 |
| 6 | `65` | NHC5597872_episodio1008512142 |
| 7 | `familia` | NHC5108048_episodio1008709623 |
| 8 | `papa` | NHC5592681_episodio1008443020 |
| 9 | `65` | NHC5592342_episodio1008725210 |
| 10 | `es` | NHC4793208_episodio1008642559 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

