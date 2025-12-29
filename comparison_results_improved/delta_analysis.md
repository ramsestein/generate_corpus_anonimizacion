# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2025-12-29 14:00:03

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline B (Base + SetFit B)
- **Recall (Seguridad):** Pipeline A (Base + SetFit A)
- **F1-Score (Balance):** Pipeline A (Base + SetFit A)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline A (Base + SetFit A) filtró, Pipeline B (Base + SetFit B) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `78a` | NHC4838037_episodio1008753942 |
| 2 | `1/10` | NHC5610715_episodio1008492595 |
| 3 | `02` | NHC5620428_episodio1008552232 |
| 4 | `familiar` | NHC5124681_episodio1008662713 |
| 5 | `familiar` | NHC5050043_episodio1008609360 |
| 6 | `familiar` | NHC5042092_episodio1008601683 |
| 7 | `familiar` | NHC4794762_episodio1008536043 |
| 8 | `familiar` | NHC5050043_episodio1008609360 |
| 9 | `familiar` | NHC4949619_episodio1008620713 |
| 10 | `familiar` | NHC5063356_episodio1008706639 |

### Pipeline B (Base + SetFit B) filtró, Pipeline A (Base + SetFit A) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `E` | NHC5588230_episodio1008425705 |
| 2 | `/04` | NHC4786376_episodio1008327607 |
| 3 | `hospital de` | NHC5079696_episodio1008677642 |
| 4 | `Go` | NHC5621838_episodio1008733689 |
| 5 | `/04` | NHC5093852_episodio1008327206 |
| 6 | `bebé` | NHC5618867_episodio1008550924 |
| 7 | `10/0 3/24` | NHC5620081_episodio1008550177 |
| 8 | `G` | NHC4948708_episodio1008311771 |
| 9 | `RN` | NHC5617524_episodio1008548475 |
| 10 | `PUR` | NHC5594075_episodio1008448649 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline A (Base + SetFit A) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `her` | NHC5587367_episodio1008418800 |
| 2 | `65` | NHC5614352_episodio1008675314 |
| 3 | `69` | NHC5612448_episodio1008545948 |
| 4 | `famil` | NHC5613642_episodio1008511229 |
| 5 | `dez` | NHC4859377_episodio1008765046 |
| 6 | `15` | NHC5595706_episodio1008462787 |
| 7 | `72` | NHC4993016_episodio1008527785 |
| 8 | `familia` | NHC5586429_episodio1008411253 |
| 9 | `her` | NHC5591595_episodio1008435789 |
| 10 | `cuidador` | NHC4954210_episodio1008354280 |

### Pipeline B (Base + SetFit B) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `her` | NHC5587367_episodio1008418800 |
| 2 | `posa` | NHC5585358_episodio1008474135 |
| 3 | `famil` | NHC5622143_episodio1008564105 |
| 4 | `M` | NHC5594583_episodio1008789399 |
| 5 | `65` | NHC5614352_episodio1008675314 |
| 6 | `padres` | NHC5616338_episodio1008527603 |
| 7 | `69` | NHC5612448_episodio1008545948 |
| 8 | `famil` | NHC5613642_episodio1008511229 |
| 9 | `ón` | NHC4992212_episodio1008616029 |
| 10 | `posa` | NHC4811732_episodio1008275820 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

