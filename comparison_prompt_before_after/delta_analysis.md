# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2026-01-07 15:47:54

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts)
- **Recall (Seguridad):** Pipeline (aws2-results)
- **F1-Score (Balance):** Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) filtró, Pipeline (aws2-results) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `16` | NHC125128_episodio1008459847 |
| 2 | `enero 2024` | NHC309873_episodio1008408200 |
| 3 | `ptiembre de 2019` | NHC5565575_episodio1008368519 |
| 4 | `I 016` | NHC5563489_episodio1008373353 |
| 5 | `847392` | NHC306498_episodio1008622432 |
| 6 | `16 /3` | NHC4197365_episodio1008260806 |
| 7 | `054` | NHC182893_episodio1008774422 |
| 8 | `familia` | NHC295309_episodio1008536044 |
| 9 | `j` | NHC5583962_episodio1008612234 |
| 10 | `familia` | NHC5577613_episodio1008358619 |

### Pipeline (aws2-results) filtró, Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `47` | NHC4183400_episodio1008301720 |
| 2 | `/10` | NHC5563653_episodio1008302257 |
| 3 | `ALES` | NHC4125532_episodio1008251547 |
| 4 | `familia` | NHC5578265_episodio1008360809 |
| 5 | `20 ABRIL` | NHC5576168_episodio1008347043 |
| 6 | `ig` | NHC324116_episodio1008757038 |
| 7 | `5.LO` | NHC4144991_episodio1008398760 |
| 8 | `lune` | NHC5574670_episodio1008336736 |
| 9 | `án` | NHC5577659_episodio1008712090 |
| 10 | `01` | NHC5562205_episodio1008254480 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `20/03` | NHC5559361_episodio1008237065 |
| 2 | `18.03.2024` | NHC5559321_episodio1008266948 |
| 3 | `17/03/24` | NHC4136987_episodio1008532772 |
| 4 | `2019` | NHC5569402_episodio1008308163 |
| 5 | `22.07.2024` | NHC107102_episodio1008411916 |
| 6 | `16/03/24` | NHC133628_episodio1008732802 |
| 7 | `I087` | NHC413251_episodio1008514330 |
| 8 | `/10` | NHC5568189_episodio1008607094 |
| 9 | `15.03.2024` | NHC171137_episodio1008233563 |
| 10 | `febrero` | NHC388606_episodio1008648115 |

### Pipeline (aws2-results) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `Soltero` | NHC163212_episodio1008555010 |
| 2 | `a` | NHC277930_episodio1008328293 |
| 3 | `f` | NHC5560451_episodio1008251049 |
| 4 | `20` | NHC179468_episodio1008390837 |
| 5 | `34` | NHC5583043_episodio1008622618 |
| 6 | `C` | NHC4043534_episodio1008264830 |
| 7 | `clínica` | NHC107102_episodio1008411058 |
| 8 | `amiliar` | NHC5570876_episodio1008313771 |
| 9 | `22` | NHC5573682_episodio1008329424 |
| 10 | `.` | NHC237633_episodio1008272606 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

