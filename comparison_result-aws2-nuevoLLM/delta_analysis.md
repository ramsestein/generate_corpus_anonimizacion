# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2026-01-05 15:12:32

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts)
- **Recall (Seguridad):** Pipeline (aws2-results)
- **F1-Score (Balance):** Pipeline (aws2-results)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline (aws2-results) filtró, Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `hijo` | NHC5568076_episodio1008296876 |
| 2 | `04:02` | NHC25761_episodio1008560659 |
| 3 | `HIJO` | NHC182072_episodio1008251773 |
| 4 | `15/02` | NHC5558959_episodio1008238966 |
| 5 | `65` | NHC4049566_episodio1008588955 |
| 6 | `03` | NHC5575055_episodio1008711159 |
| 7 | `2012` | NHC4016967_episodio1008605628 |
| 8 | `82a` | NHC4043534_episodio1008263417 |
| 9 | `62` | NHC4074096_episodio1008723818 |
| 10 | `03` | NHC5575055_episodio1008711159 |

### Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) filtró, Pipeline (aws2-results) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `jue ves` | NHC314973_episodio1008281005 |
| 2 | `23.03.2023` | NHC5576471_episodio1008387759 |
| 3 | `abuelo` | NHC5577516_episodio1008362092 |
| 4 | `18.03.20` | NHC5568365_episodio1008672422 |
| 5 | `26.03.2024` | NHC361029_episodio1008428452 |
| 6 | `19.08.20` | NHC109003_episodio1008474128 |
| 7 | `16.03.2024` | NHC148043_episodio1008397767 |
| 8 | `Ruiz` | NHC4135450_episodio1008588452 |
| 9 | `marzo/24` | NHC5559154_episodio1008459851 |
| 10 | `agosto` | NHC4054538_episodio1008694526 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline (aws2-results) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `26.03.` | NHC361029_episodio1008428452 |
| 2 | `IZ` | NHC4181263_episodio1008279606 |
| 3 | `76` | NHC4058437_episodio1008564572 |
| 4 | `C` | NHC5564659_episodio1008272010 |
| 5 | `25/02` | NHC5559856_episodio1008244115 |
| 6 | `cu` | NHC5567337_episodio1008460011 |
| 7 | `6` | NHC4018547_episodio1008573241 |
| 8 | `S` | NHC29354_episodio1008289327 |
| 9 | `6` | NHC258884_episodio1008706977 |
| 10 | `J` | NHC5567450_episodio1008761012 |

### Pipeline (aws2-gatekeeper_improved-llm-mejora-prompts) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `26.03.` | NHC361029_episodio1008428452 |
| 2 | `19.08.2024` | NHC358870_episodio1008500780 |
| 3 | `28/02` | NHC163212_episodio1008592596 |
| 4 | `IZ` | NHC4181263_episodio1008279606 |
| 5 | `20.04.2024` | NHC5568744_episodio1008312235 |
| 6 | `20.04.2024` | NHC4051163_episodio1008350964 |
| 7 | `18.03.2024` | NHC218631_episodio1008645330 |
| 8 | `19.03.2024` | NHC220201_episodio1008614732 |
| 9 | `76` | NHC4058437_episodio1008564572 |
| 10 | `18.08.2023` | NHC5578969_episodio1008490906 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

