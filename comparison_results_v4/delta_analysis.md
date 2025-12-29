# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2025-12-29 12:49:59

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline B (Base + SetFit B)
- **Recall (Seguridad):** Pipeline B (Base + SetFit B)
- **F1-Score (Balance):** Pipeline B (Base + SetFit B)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline A (Base + SetFit A) filtró, Pipeline B (Base + SetFit B) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `21.08.2024` | NHC5596918_episodio1008470784 |
| 2 | `15.08.2024` | NHC4852702_episodio1008527242 |
| 3 | `16.08.2024` | NHC4737379_episodio1008557970 |
| 4 | `23.08.2024` | NHC5597872_episodio1008512142 |
| 5 | `Ana` | NHC4742874_episodio1008667837 |
| 6 | `23.08.2024` | NHC5018905_episodio1008513875 |
| 7 | `16.07.2024` | NHC4878577_episodio1008392216 |
| 8 | `-Ruiz` | NHC560653_episodio1008654342 |
| 9 | `15.08.2023` | NHC5597235_episodio1008472361 |
| 10 | `16.08.2024` | NHC5610303_episodio1008491109 |

### Pipeline B (Base + SetFit B) filtró, Pipeline A (Base + SetFit A) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `22/11` | NHC5627628_episodio1008642331 |
| 2 | `mana` | NHC5127728_episodio1008601790 |
| 3 | `18/4` | NHC5592484_episodio1008506826 |
| 4 | `mana` | NHC5614796_episodio1008523550 |
| 5 | `/03` | NHC5624394_episodio1008582065 |
| 6 | `12/09` | NHC5624618_episodio1008578118 |
| 7 | `2020` | NHC4926390_episodio1008414462 |
| 8 | `18.03` | NHC5621071_episodio1008554794 |
| 9 | `22/11` | NHC506852_episodio1008708778 |
| 10 | `5/09` | NHC5624565_episodio1008579727 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline A (Base + SetFit A) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 0

*No hay fugas* ✅

### Pipeline B (Base + SetFit B) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 0

*No hay fugas* ✅

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

