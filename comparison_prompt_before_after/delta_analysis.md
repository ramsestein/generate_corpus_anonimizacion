# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2026-01-05 15:22:08

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts)
- **Recall (Seguridad):** Pipeline (aws2-results)
- **F1-Score (Balance):** Pipeline (aws2-results)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline (aws2-results) filtró, Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `03` | NHC250724_episodio1008223893 |
| 2 | `19` | NHC5561592_episodio1008341537 |
| 3 | `03.2023` | NHC4162126_episodio1008375781 |
| 4 | `03` | NHC5561168_episodio1008250782 |
| 5 | `16/9` | NHC5578949_episodio1008584837 |
| 6 | `hijo` | NHC4041443_episodio1008253585 |
| 7 | `2015` | NHC5565605_episodio1008281918 |
| 8 | `16.03.` | NHC5569164_episodio1008302960 |
| 9 | `ab` | NHC5561353_episodio1008248653 |
| 10 | `03` | NHC5564303_episodio1008468132 |

### Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) filtró, Pipeline (aws2-results) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `21/07` | NHC4119863_episodio1008370215 |
| 2 | `16.03.2024` | NHC5568746_episodio1008303673 |
| 3 | `1995` | NHC386317_episodio1008771994 |
| 4 | `familia` | NHC371252_episodio1008640620 |
| 5 | `17/03/20` | NHC11308_episodio1008291758 |
| 6 | `18.03.20` | NHC5566744_episodio1008293508 |
| 7 | `18/03` | NHC386335_episodio1008794387 |
| 8 | `17/03/` | NHC189823_episodio1008736790 |
| 9 | `15.09.20` | NHC143452_episodio1008557860 |
| 10 | `26.09.20` | NHC5575218_episodio1008612225 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline (aws2-results) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `4` | NHC5580066_episodio1008371852 |
| 2 | `Es` | NHC154130_episodio1008745098 |
| 3 | `15/8/24` | NHC5577278_episodio1008356097 |
| 4 | `4` | NHC4084564_episodio1008735773 |
| 5 | `18/3` | NHC5579677_episodio1008427136 |
| 6 | `05` | NHC5561050_episodio1008327201 |
| 7 | `i` | NHC4021540_episodio1008240732 |
| 8 | `15/6/24` | NHC4055482_episodio1008411865 |
| 9 | `Hermana` | NHC5561300_episodio1008248601 |
| 10 | `her` | NHC331022_episodio1008368677 |

### Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `20.03.2024` | NHC5567557_episodio1008295167 |
| 2 | `2015` | NHC242992_episodio1008405953 |
| 3 | `2018` | NHC252742_episodio1008316525 |
| 4 | `22/07` | NHC311119_episodio1008376010 |
| 5 | `21.03.2024` | NHC5564794_episodio1008551546 |
| 6 | `17.07.2024` | NHC5578027_episodio1008380566 |
| 7 | `17.09.2024` | NHC4149674_episodio1008620731 |
| 8 | `2019` | NHC5577515_episodio1008362064 |
| 9 | `16/03` | NHC5570219_episodio1008310560 |
| 10 | `19.04.2024` | NHC5575304_episodio1008341118 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

