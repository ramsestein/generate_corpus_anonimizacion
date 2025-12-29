# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2025-12-29 13:13:27

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline A (Base + SetFit A)
- **Recall (Seguridad):** Pipeline B (Base + SetFit B)
- **F1-Score (Balance):** Pipeline B (Base + SetFit B)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline A (Base + SetFit A) filtró, Pipeline B (Base + SetFit B) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `hij os` | NHC5626191_episodio1008718198 |
| 2 | `12/202` | NHC4982380_episodio1008665926 |
| 3 | `E014` | NHC5065660_episodio1008312068 |
| 4 | `16/9` | NHC4886918_episodio1008435372 |
| 5 | `madre` | NHC4880133_episodio1008515937 |
| 6 | `1/4` | NHC5128623_episodio1008237090 |
| 7 | `9/8` | NHC4735763_episodio1008601338 |
| 8 | `7.3` | NHC4856621_episodio1008790398 |
| 9 | `12/7` | NHC5588516_episodio1008426704 |
| 10 | `H. Clínic o` | NHC4882263_episodio1008521208 |

### Pipeline B (Base + SetFit B) filtró, Pipeline A (Base + SetFit A) dejó pasar
**Total:** 0 ejemplos

*No hay ejemplos*

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline A (Base + SetFit A) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `es` | NHC5628114_episodio1008616161 |
| 2 | `6` | NHC5619036_episodio1008555628 |
| 3 | `17.08.2024` | NHC4780857_episodio1008492280 |
| 4 | `Martín` | NHC4765232_episodio1008340955 |
| 5 | `El` | NHC5596928_episodio1008470849 |
| 6 | `18.07.2024` | NHC5044954_episodio1008368022 |
| 7 | `72 años` | NHC5036159_episodio1008714058 |
| 8 | `15.08.2024` | NHC5616015_episodio1008524486 |
| 9 | `.2024` | NHC5625512_episodio1008778844 |
| 10 | `15` | NHC5066813_episodio1008319859 |

### Pipeline B (Base + SetFit B) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `es` | NHC5628114_episodio1008616161 |
| 2 | `6` | NHC5619036_episodio1008555628 |
| 3 | `Martín` | NHC4765232_episodio1008340955 |
| 4 | `El` | NHC5596928_episodio1008470849 |
| 5 | `15` | NHC5066813_episodio1008319859 |
| 6 | `famil` | NHC5621071_episodio1008554794 |
| 7 | `IG` | NHC4786870_episodio1008324121 |
| 8 | `es` | NHC5614352_episodio1008675314 |
| 9 | `Silva` | NHC5594922_episodio1008641169 |
| 10 | `Mar` | NHC5628795_episodio1008608864 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

