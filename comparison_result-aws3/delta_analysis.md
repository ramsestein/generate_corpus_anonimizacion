# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2025-12-31 10:32:25

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline B (Base + SetFit B)
- **Recall (Seguridad):** Pipeline A (Base + SetFit A)
- **F1-Score (Balance):** Pipeline A (Base + SetFit A)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline A (Base + SetFit A) filtró, Pipeline B (Base + SetFit B) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `familia` | NHC5024534_episodio1008347950 |
| 2 | `familia` | NHC5124681_episodio1008540516 |
| 3 | `familia` | NHC5615183_episodio1008526385 |
| 4 | `23` | NHC4856621_episodio1008790398 |
| 5 | `9/10` | NHC5626949_episodio1008643003 |
| 6 | `familia` | NHC4827117_episodio1008350898 |
| 7 | `familia` | NHC4903759_episodio1008528322 |
| 8 | `9/10` | NHC5591180_episodio1008605089 |
| 9 | `familia` | NHC4722141_episodio1008643470 |
| 10 | `familia` | NHC5593550_episodio1008655411 |

### Pipeline B (Base + SetFit B) filtró, Pipeline A (Base + SetFit A) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `familiar` | NHC4928865_episodio1008570355 |
| 2 | `posa` | NHC4834950_episodio1008438784 |
| 3 | `familiar` | NHC5617494_episodio1008538306 |
| 4 | `E` | NHC5622286_episodio1008673640 |
| 5 | `posa` | NHC5595755_episodio1008463774 |
| 6 | `posa` | NHC4778738_episodio1008311678 |
| 7 | `hij o` | NHC4791402_episodio1008409465 |
| 8 | `L` | NHC5620428_episodio1008552232 |
| 9 | `eja` | NHC5618265_episodio1008759760 |
| 10 | `mana` | NHC5030067_episodio1008520626 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline A (Base + SetFit A) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `6` | NHC5611455_episodio1008737483 |
| 2 | `68` | NHC4926390_episodio1008414462 |
| 3 | `6` | NHC5591495_episodio1008739670 |
| 4 | `65` | NHC4853814_episodio1008249334 |
| 5 | `65` | NHC4773520_episodio1008441592 |
| 6 | `cuador` | NHC4904833_episodio1008279258 |
| 7 | `6` | NHC5618655_episodio1008793743 |
| 8 | `6` | NHC5594587_episodio1008541472 |
| 9 | `27` | NHC5593096_episodio1008493471 |
| 10 | `31a` | NHC4895808_episodio1008446430 |

### Pipeline B (Base + SetFit B) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `6` | NHC5611455_episodio1008737483 |
| 2 | `68` | NHC4926390_episodio1008414462 |
| 3 | `6` | NHC5591495_episodio1008739670 |
| 4 | `65` | NHC4853814_episodio1008249334 |
| 5 | `65` | NHC4773520_episodio1008441592 |
| 6 | `cuador` | NHC4904833_episodio1008279258 |
| 7 | `6` | NHC5618655_episodio1008793743 |
| 8 | `6` | NHC5594587_episodio1008541472 |
| 9 | `so` | NHC5047760_episodio1008665787 |
| 10 | `par` | NHC4714528_episodio1008282766 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

