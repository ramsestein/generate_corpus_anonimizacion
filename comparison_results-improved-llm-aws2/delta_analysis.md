# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2025-12-31 10:27:33

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline A (Base + SetFit A)
- **Recall (Seguridad):** Pipeline B (Base + SetFit B)
- **F1-Score (Balance):** Pipeline B (Base + SetFit B)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline A (Base + SetFit A) filtró, Pipeline B (Base + SetFit B) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `07/2023` | NHC321080_episodio1008653654 |
| 2 | `19.03.20` | NHC5558558_episodio1008281141 |
| 3 | `años` | NHC4166225_episodio1008543153 |
| 4 | `21.03.20` | NHC5577453_episodio1008523568 |
| 5 | `mana` | NHC5563932_episodio1008323478 |
| 6 | `mana` | NHC127309_episodio1008462187 |
| 7 | `03` | NHC4183664_episodio1008718300 |
| 8 | `24.05.20` | NHC5558885_episodio1008328557 |
| 9 | `15.03.2024` | NHC4084564_episodio1008711915 |
| 10 | `16/08` | NHC4145660_episodio1008506750 |

### Pipeline B (Base + SetFit B) filtró, Pipeline A (Base + SetFit A) dejó pasar
**Total:** 4 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `5.7` | NHC103087_episodio1008807132 |
| 2 | `5.7` | NHC103087_episodio1008807132 |
| 3 | `septiembre 2020` | NHC102219_episodio1008744732 |
| 4 | `03` | NHC103087_episodio1008807132 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline A (Base + SetFit A) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `24/08/24` | NHC5563493_episodio1008443167 |
| 2 | `15/02/24` | NHC4072273_episodio1008742093 |
| 3 | `15.07.2024` | NHC5574210_episodio1008350987 |
| 4 | `Hospital Clínic` | NHC5565575_episodio1008336744 |
| 5 | `15/03` | NHC5558544_episodio1008598736 |
| 6 | `15.03.20` | NHC313111_episodio1008251283 |
| 7 | `16/04` | NHC5568744_episodio1008312235 |
| 8 | `19.04.2024` | NHC5573276_episodio1008327582 |
| 9 | `24.03.2024` | NHC5583277_episodio1008390622 |
| 10 | `22.03.2024` | NHC4081159_episodio1008713076 |

### Pipeline B (Base + SetFit B) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `posa` | NHC5562205_episodio1008254480 |
| 2 | `05/10` | NHC5581920_episodio1008696337 |
| 3 | `6` | NHC5563647_episodio1008647806 |
| 4 | `16/3` | NHC323252_episodio1008378249 |
| 5 | `P` | NHC5559905_episodio1008242169 |
| 6 | `16/11` | NHC5584258_episodio1008699722 |
| 7 | `22/11` | NHC147284_episodio1008654877 |
| 8 | `HCL4` | NHC318981_episodio1008543774 |
| 9 | `2020` | NHC4186814_episodio1008679243 |
| 10 | `12.03.2024` | NHC102219_episodio1008744732 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

