# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2026-01-05 09:56:12

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline (resultados_recall_v2)
- **Recall (Seguridad):** Pipeline (aws2-results)
- **F1-Score (Balance):** Pipeline (aws2-results)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline (aws2-results) filtró, Pipeline (resultados_recall_v2) dejó pasar
**Total:** 5 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `5.7` | NHC103087_episodio1008807132 |
| 2 | `5.7` | NHC103087_episodio1008807132 |
| 3 | `03` | NHC103087_episodio1008807132 |
| 4 | `20.8.20` | NHC102219_episodio1008744732 |
| 5 | `654789123` | NHC103087_episodio1008807132 |

### Pipeline (resultados_recall_v2) filtró, Pipeline (aws2-results) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `19.09.2024` | NHC4030770_episodio1008537783 |
| 2 | `23.03.2023` | NHC32687_episodio1008245416 |
| 3 | `16.03.2024` | NHC5566556_episodio1008287085 |
| 4 | `dic 2022` | NHC4188050_episodio1008296891 |
| 5 | `05/07` | NHC5579411_episodio1008368061 |
| 6 | `18.03.2024` | NHC5566744_episodio1008293508 |
| 7 | `15/04/2 020` | NHC163212_episodio1008555010 |
| 8 | `26.04.2024` | NHC5566708_episodio1008322681 |
| 9 | `22.03.2024` | NHC154130_episodio1008745098 |
| 10 | `17.03.2024` | NHC5559132_episodio1008529231 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline (aws2-results) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `cu` | NHC4084564_episodio1008735773 |
| 2 | `6` | NHC5580066_episodio1008372727 |
| 3 | `20.05.20` | NHC5572668_episodio1008323781 |
| 4 | `9` | NHC5568076_episodio1008296876 |
| 5 | `brino` | NHC5579052_episodio1008646107 |
| 6 | `23` | NHC147284_episodio1008654873 |
| 7 | `25` | NHC5561592_episodio1008341537 |
| 8 | `J` | NHC305871_episodio1008252483 |
| 9 | `68` | NHC133628_episodio1008465941 |
| 10 | `es` | NHC5574741_episodio1008342090 |

### Pipeline (resultados_recall_v2) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `15/03/24` | NHC4112487_episodio1008769107 |
| 2 | `15.03.2024` | NHC4179748_episodio1008597861 |
| 3 | `23.03.2023` | NHC134933_episodio1008245813 |
| 4 | `cu` | NHC4084564_episodio1008735773 |
| 5 | `15.03.2024` | NHC4036782_episodio1008417096 |
| 6 | `22.03.2024` | NHC152574_episodio1008627449 |
| 7 | `López` | NHC4020663_episodio1008638435 |
| 8 | `6` | NHC5580066_episodio1008372727 |
| 9 | `18.03.2024` | NHC5560829_episodio1008246830 |
| 10 | `18.11.2024` | NHC5563647_episodio1008647806 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

