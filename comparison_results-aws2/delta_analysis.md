# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2025-12-31 12:02:07

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline A (Base + SetFit A)
- **Recall (Seguridad):** Pipeline B (Base + SetFit B)
- **F1-Score (Balance):** Pipeline B (Base + SetFit B)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline A (Base + SetFit A) filtró, Pipeline B (Base + SetFit B) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `hermano` | NHC4144991_episodio1008398458 |
| 2 | `SS` | NHC4143222_episodio1008340270 |
| 3 | `padre` | NHC5559374_episodio1008238280 |
| 4 | `04` | NHC134173_episodio1008241264 |
| 5 | `18.03` | NHC184246_episodio1008456007 |
| 6 | `03` | NHC5566350_episodio1008286141 |
| 7 | `es` | NHC5561240_episodio1008252370 |
| 8 | `16/09` | NHC4127813_episodio1008541067 |
| 9 | `posa` | NHC5584618_episodio1008464891 |
| 10 | `go` | NHC175850_episodio1008786042 |

### Pipeline B (Base + SetFit B) filtró, Pipeline A (Base + SetFit A) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `familia` | NHC279725_episodio1008405211 |
| 2 | `familia` | NHC4113705_episodio1008649160 |
| 3 | `familia` | NHC106104_episodio1008741129 |
| 4 | `familia` | NHC138131_episodio1008335773 |
| 5 | `familia` | NHC4025039_episodio1008560068 |
| 6 | `16/11` | NHC153949_episodio1008631651 |
| 7 | `familia` | NHC5574670_episodio1008336736 |
| 8 | `22/11` | NHC176285_episodio1008701803 |
| 9 | `Caribbean` | NHC5576525_episodio1008354287 |
| 10 | `familia` | NHC5578037_episodio1008574440 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline A (Base + SetFit A) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `6` | NHC5585001_episodio1008406349 |
| 2 | `H052` | NHC5580364_episodio1008375004 |
| 3 | `posa` | NHC378007_episodio1008525424 |
| 4 | `H142` | NHC135242_episodio1008340505 |
| 5 | `2017` | NHC4067438_episodio1008262995 |
| 6 | `2001` | NHC5565023_episodio1008475010 |
| 7 | `2020` | NHC4191419_episodio1008573217 |
| 8 | `tin` | NHC5577377_episodio1008354324 |
| 9 | `03` | NHC139466_episodio1008280567 |
| 10 | `2021` | NHC292371_episodio1008464898 |

### Pipeline B (Base + SetFit B) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `6` | NHC5585001_episodio1008406349 |
| 2 | `H052` | NHC5580364_episodio1008375004 |
| 3 | `H142` | NHC135242_episodio1008340505 |
| 4 | `2020` | NHC4191419_episodio1008573217 |
| 5 | `tin` | NHC5577377_episodio1008354324 |
| 6 | `2021` | NHC292371_episodio1008464898 |
| 7 | `familiar` | NHC5578715_episodio1008372742 |
| 8 | `Vega` | NHC5566054_episodio1008285909 |
| 9 | `2010` | NHC5574205_episodio1008693878 |
| 10 | `21/11` | NHC176285_episodio1008701803 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

