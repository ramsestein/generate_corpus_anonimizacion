# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2025-12-30 13:22:25

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline A (Base + SetFit A)
- **Recall (Seguridad):** Pipeline B (Base + SetFit B)
- **F1-Score (Balance):** Pipeline B (Base + SetFit B)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline A (Base + SetFit A) filtró, Pipeline B (Base + SetFit B) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `familiar` | NHC5593132_episodio1008453457 |
| 2 | `familiar` | NHC4963678_episodio1008597823 |
| 3 | `familiar` | NHC5610548_episodio1008491803 |
| 4 | `familiar` | NHC5593550_episodio1008655411 |
| 5 | `1/10` | NHC4800746_episodio1008438786 |
| 6 | `familiar` | NHC5592970_episodio1008441028 |
| 7 | `6/10` | NHC5619546_episodio1008613416 |
| 8 | `familiar` | NHC5626191_episodio1008718198 |
| 9 | `familiar` | NHC4811983_episodio1008629242 |
| 10 | `familiar` | NHC5050043_episodio1008609360 |

### Pipeline B (Base + SetFit B) filtró, Pipeline A (Base + SetFit A) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `familiar` | NHC5626824_episodio1008748204 |
| 2 | `20` | NHC5593397_episodio1008650995 |
| 3 | `familiar` | NHC4742874_episodio1008667837 |
| 4 | `familiar` | NHC5626824_episodio1008748204 |
| 5 | `1/10` | NHC5617714_episodio1008789482 |
| 6 | `familiar de referencia` | NHC4767241_episodio1008221485 |
| 7 | `16/1 1` | NHC4805562_episodio1008643902 |
| 8 | `MS 9` | NHC5626391_episodio1008602648 |
| 9 | `familiar` | NHC4802955_episodio1008325084 |
| 10 | `familiar` | NHC5624555_episodio1008592237 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline A (Base + SetFit A) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `7` | NHC5613238_episodio1008599188 |
| 2 | `68` | NHC5032352_episodio1008375782 |
| 3 | `od` | NHC5616347_episodio1008527639 |
| 4 | `uela` | NHC5616177_episodio1008528237 |
| 5 | `6` | NHC5598922_episodio1008676554 |
| 6 | `her` | NHC5625566_episodio1008587783 |
| 7 | `65` | NHC5590663_episodio1008655564 |
| 8 | `45` | NHC5619220_episodio1008541555 |
| 9 | `RES` | NHC483936_episodio1008492589 |
| 10 | `es` | NHC5592438_episodio1008447326 |

### Pipeline B (Base + SetFit B) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `7` | NHC5613238_episodio1008599188 |
| 2 | `68` | NHC5032352_episodio1008375782 |
| 3 | `od` | NHC5616347_episodio1008527639 |
| 4 | `20` | NHC5623781_episodio1008641029 |
| 5 | `uela` | NHC5616177_episodio1008528237 |
| 6 | `6` | NHC5598922_episodio1008676554 |
| 7 | `65` | NHC5590663_episodio1008655564 |
| 8 | `45` | NHC5619220_episodio1008541555 |
| 9 | `RES` | NHC483936_episodio1008492589 |
| 10 | `es` | NHC5592438_episodio1008447326 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

