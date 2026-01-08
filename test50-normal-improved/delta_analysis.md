# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2026-01-08 10:27:52

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline (prueba-0.97-normal)
- **Recall (Seguridad):** Pipeline (prueba-0.97)
- **F1-Score (Balance):** Pipeline (prueba-0.97)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline (prueba-0.97-normal) filtró, Pipeline (prueba-0.97) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `familia` | NHC4067438_episodio1008262995 |
| 2 | `G` | NHC4067438_episodio1008262995 |
| 3 | `V` | NHC5581723_episodio1008436709 |
| 4 | `H Clínic de Barcelona` | NHC5580111_episodio1008374350 |
| 5 | `Hospital del Mar` | NHC179468_episodio1008393022 |
| 6 | `pers` | NHC5565961_episodio1008280467 |
| 7 | `Enero de 2003` | NHC365276_episodio1008350813 |
| 8 | `E` | NHC5568921_episodio1008350100 |
| 9 | `16.07.20` | NHC365276_episodio1008350813 |
| 10 | `men` | NHC5581074_episodio1008646905 |

### Pipeline (prueba-0.97) filtró, Pipeline (prueba-0.97-normal) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `09` | NHC5583668_episodio1008559582 |
| 2 | `03` | NHC5580111_episodio1008374350 |
| 3 | `Fernández` | NHC4072273_episodio1008742093 |
| 4 | `zo/24` | NHC4034835_episodio1008530518 |
| 5 | `familia` | NHC5580063_episodio1008372518 |
| 6 | `Centro SS Mediterráneo` | NHC37937_episodio1008254299 |
| 7 | `17/03` | NHC4072273_episodio1008742093 |
| 8 | `familia` | NHC107102_episodio1008411916 |
| 9 | `17.04.20` | NHC5569653_episodio1008312191 |
| 10 | `ERM` | NHC4034835_episodio1008530518 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline (prueba-0.97-normal) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `19 MAR 2024` | NHC5585300_episodio1008687210 |
| 2 | `junio 8` | NHC5582910_episodio1008396001 |
| 3 | `15/08` | NHC4034835_episodio1008530518 |
| 4 | `G028` | NHC5569798_episodio1008326489 |
| 5 | `18.03.2024` | NHC4072273_episodio1008742093 |
| 6 | `12.03.2024` | NHC250452_episodio1008719568 |
| 7 | `es` | NHC107102_episodio1008411916 |
| 8 | `men` | NHC107102_episodio1008411916 |
| 9 | `19/03/24` | NHC5566693_episodio1008774659 |
| 10 | `2010` | NHC37937_episodio1008254299 |

### Pipeline (prueba-0.97) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `junio 8` | NHC5582910_episodio1008396001 |
| 2 | `15/08` | NHC4034835_episodio1008530518 |
| 3 | `G028` | NHC5569798_episodio1008326489 |
| 4 | `limpieza en oficinas` | NHC37937_episodio1008254299 |
| 5 | `25/4/24` | NHC5569653_episodio1008312191 |
| 6 | `.` | NHC5582715_episodio1008397802 |
| 7 | `23/09` | NHC5583668_episodio1008559582 |
| 8 | `S` | NHC19816_episodio1008346144 |
| 9 | `padres` | NHC37937_episodio1008254299 |
| 10 | `dos fills` | NHC37937_episodio1008254299 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

