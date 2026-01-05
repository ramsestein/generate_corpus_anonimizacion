# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2026-01-05 10:00:26

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline (aws2-results)
- **Recall (Seguridad):** Pipeline (resultados_recall_v2-LLM)
- **F1-Score (Balance):** Pipeline (resultados_recall_v2-LLM)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline (aws2-results) filtró, Pipeline (resultados_recall_v2-LLM) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `22.08.2023` | NHC4880133_episodio1008515936 |
| 2 | `16.03.2024` | NHC5623942_episodio1008580566 |
| 3 | `03/11` | NHC5628518_episodio1008644728 |
| 4 | `21.03.2024` | NHC4856621_episodio1008790398 |
| 5 | `20.06.2024` | NHC4791402_episodio1008409674 |
| 6 | `20.03.2024` | NHC5592981_episodio1008464970 |
| 7 | `21.03.2024` | NHC4876654_episodio1008764308 |
| 8 | `16.03.2024` | NHC5590663_episodio1008692517 |
| 9 | `12.03.2024` | NHC5625839_episodio1008676574 |
| 10 | `17.03.2024` | NHC5587524_episodio1008740061 |

### Pipeline (resultados_recall_v2-LLM) filtró, Pipeline (aws2-results) dejó pasar
**Total:** 0 ejemplos

*No hay ejemplos*

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline (aws2-results) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `12/11/24` | NHC4916741_episodio1008788025 |
| 2 | `Hospital Quirón` | NHC5627192_episodio1008599235 |
| 3 | `15/03` | NHC4824475_episodio1008689863 |
| 4 | `15.03.2024` | NHC4956785_episodio1008641567 |
| 5 | `11/03/20` | NHC5618054_episodio1008722538 |
| 6 | `15/08/2024` | NHC4915575_episodio1008345314 |
| 7 | `17/03` | NHC5628947_episodio1008663807 |
| 8 | `15.03.2023` | NHC5128623_episodio1008237090 |
| 9 | `her` | NHC5592444_episodio1008684661 |
| 10 | `23.03.2024` | NHC4801586_episodio1008292880 |

### Pipeline (resultados_recall_v2-LLM) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `her` | NHC5592444_episodio1008684661 |
| 2 | `Lo` | NHC5590209_episodio1008510399 |
| 3 | `Mar` | NHC5625947_episodio1008584588 |
| 4 | `H` | NHC5610037_episodio1008490172 |
| 5 | `34` | NHC5093852_episodio1008411581 |
| 6 | `22.` | NHC5127548_episodio1008653269 |
| 7 | `her` | NHC5617494_episodio1008535296 |
| 8 | `H` | NHC4939534_episodio1008798436 |
| 9 | `12` | NHC5588014_episodio1008425095 |
| 10 | `27` | NHC5618615_episodio1008611166 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

