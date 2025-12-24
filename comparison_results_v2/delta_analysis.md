# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2025-12-24 09:50:15

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline B (Base + SetFit B)
- **Recall (Seguridad):** Pipeline A (Base + SetFit A)
- **F1-Score (Balance):** Pipeline A (Base + SetFit A)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline A (Base + SetFit A) filtró, Pipeline B (Base + SetFit B) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `mana` | NHC5618865_episodio1008550904 |
| 2 | `mana` | NHC5622706_episodio1008565620 |
| 3 | `mana` | NHC5086393_episodio1008351043 |
| 4 | `04` | NHC4735763_episodio1008601338 |
| 5 | `/03` | NHC103087_episodio1008807132 |
| 6 | `04/10/20` | NHC5627600_episodio1008612227 |
| 7 | `mana` | NHC5626949_episodio1008643003 |
| 8 | `mana` | NHC5586443_episodio1008632566 |
| 9 | `/03` | NHC5624483_episodio1008700183 |
| 10 | `mana` | NHC5007959_episodio1008760345 |

### Pipeline B (Base + SetFit B) filtró, Pipeline A (Base + SetFit A) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `madre` | NHC4966984_episodio1008577332 |
| 2 | `ges` | NHC5066813_episodio1008319859 |
| 3 | `madre` | NHC505442_episodio1008566354 |
| 4 | `ama` | NHC5610244_episodio1008491303 |
| 5 | `16.` | NHC4817256_episodio1008252133 |
| 6 | `ges` | NHC4978236_episodio1008681939 |
| 7 | `fami lia` | NHC4949619_episodio1008620713 |
| 8 | `hermana` | NHC5623437_episodio1008675840 |
| 9 | `familia` | NHC5588602_episodio1008427223 |
| 10 | `pañeras` | NHC5041584_episodio1008788156 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline A (Base + SetFit A) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `g045` | NHC5612977_episodio1008595878 |
| 2 | `17/3` | NHC4767241_episodio1008221485 |
| 3 | `h024` | NHC5616443_episodio1008622978 |
| 4 | `i018` | NHC5043254_episodio1008588315 |
| 5 | `14/3` | NHC5627123_episodio1008632576 |
| 6 | `2021` | NHC4926390_episodio1008412633 |
| 7 | `15/9` | NHC5621068_episodio1008586372 |
| 8 | `20/3` | NHC5594075_episodio1008448809 |
| 9 | `h095` | NHC4801577_episodio1008419801 |
| 10 | `12/09` | NHC5624618_episodio1008578118 |

### Pipeline B (Base + SetFit B) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `madre` | NHC4869119_episodio1008262704 |
| 2 | `17/3` | NHC4767241_episodio1008221485 |
| 3 | `19.07.2024` | NHC5030195_episodio1008350084 |
| 4 | `h024` | NHC5616443_episodio1008622978 |
| 5 | `martínez` | NHC5030199_episodio1008683832 |
| 6 | `14/3` | NHC5627123_episodio1008632576 |
| 7 | `madre` | NHC5591229_episodio1008823652 |
| 8 | `16.04.2024` | NHC5097797_episodio1008307929 |
| 9 | `19.07.2024` | NHC5586594_episodio1008411582 |
| 10 | `12.08.2024` | NHC5587784_episodio1008420966 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

