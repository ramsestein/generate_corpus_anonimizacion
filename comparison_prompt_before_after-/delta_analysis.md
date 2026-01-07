# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2026-01-07 15:44:18

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts-reevaluado)
- **Recall (Seguridad):** Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts)
- **F1-Score (Balance):** Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts-reevaluado)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) filtró, Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts-reevaluado) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `hermanos` | NHC351744_episodio1008221307 |
| 2 | `09.07` | NHC5578701_episodio1008372666 |
| 3 | `12` | NHC5569164_episodio1008301098 |
| 4 | `14-15/11` | NHC4159364_episodio1008695041 |
| 5 | `02/2023` | NHC32687_episodio1008245416 |
| 6 | `17/09/` | NHC5568189_episodio1008607094 |
| 7 | `mana` | NHC4077590_episodio1008583870 |
| 8 | `15/11` | NHC4159364_episodio1008695041 |
| 9 | `15/09/2024` | NHC4132650_episodio1008545510 |
| 10 | `10/07` | NHC5569848_episodio1008318950 |

### Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts-reevaluado) filtró, Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `madre` | NHC5562760_episodio1008211848 |
| 2 | `15` | NHC4034835_episodio1008528558 |
| 3 | `18` | NHC5564129_episodio1008267756 |
| 4 | `2 hijos` | NHC4145660_episodio1008506750 |
| 5 | `E` | NHC257129_episodio1008767354 |
| 6 | `09` | NHC4110815_episodio1008588321 |
| 7 | `16` | NHC5562375_episodio1008258173 |
| 8 | `19/0` | NHC4080845_episodio1008418644 |
| 9 | `tiembre 202 2` | NHC4105854_episodio1008514113 |
| 10 | `08/24` | NHC5584915_episodio1008497774 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `24/4` | NHC5572299_episodio1008323615 |
| 2 | `15/3/24` | NHC5569164_episodio1008302960 |
| 3 | `C152` | NHC5569869_episodio1008310088 |
| 4 | `26/11` | NHC176285_episodio1008701803 |
| 5 | `25/08/24` | NHC5584421_episodio1008512426 |
| 6 | `20` | NHC299322_episodio1008329677 |
| 7 | `08:` | NHC5585011_episodio1008408513 |
| 8 | `Vega` | NHC5574670_episodio1008336736 |
| 9 | `24/08/24` | NHC5580792_episodio1008507448 |
| 10 | `2010` | NHC4019653_episodio1008296886 |

### Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts-reevaluado) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `24/4` | NHC5572299_episodio1008323615 |
| 2 | `2008` | NHC4003258_episodio1008638291 |
| 3 | `15/3/24` | NHC5569164_episodio1008302960 |
| 4 | `24/7` | NHC5578520_episodio1008366916 |
| 5 | `C152` | NHC5569869_episodio1008310088 |
| 6 | `26/11` | NHC176285_episodio1008701803 |
| 7 | `25/08/24` | NHC5584421_episodio1008512426 |
| 8 | `15/03/2024` | NHC5584902_episodio1008791451 |
| 9 | `20` | NHC299322_episodio1008329677 |
| 10 | `08:` | NHC5585011_episodio1008408513 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

