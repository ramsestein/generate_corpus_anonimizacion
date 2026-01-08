# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2026-01-08 10:11:43

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline (prueba-0.97)
- **Recall (Seguridad):** Pipeline (test50docs)
- **F1-Score (Balance):** Pipeline (prueba-0.97)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline (test50docs) filtró, Pipeline (prueba-0.97) dejó pasar
**Total:** 3 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `Abr il/2024` | NHC5582910_episodio1008394614 |
| 2 | `A` | NHC5582910_episodio1008394614 |
| 3 | `br il/2024` | NHC5582910_episodio1008394614 |

### Pipeline (prueba-0.97) filtró, Pipeline (test50docs) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `Mar` | NHC4034835_episodio1008530518 |
| 2 | `10/05/2` | NHC5581723_episodio1008436709 |
| 3 | `familia` | NHC4198704_episodio1008285899 |
| 4 | `ills` | NHC37937_episodio1008254299 |
| 5 | `mano` | NHC5585300_episodio1008687210 |
| 6 | `14` | NHC5575478_episodio1008341198 |
| 7 | `ERM` | NHC4034835_episodio1008530518 |
| 8 | `10/05/2 0` | NHC5581723_episodio1008436709 |
| 9 | `. En` | NHC5574823_episodio1008341183 |
| 10 | `zo/24` | NHC4034835_episodio1008530518 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline (test50docs) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `Martínez` | NHC5578853_episodio1008590097 |
| 2 | `21.08.2024` | NHC302538_episodio1008414306 |
| 3 | `G028` | NHC5569798_episodio1008326489 |
| 4 | `2019` | NHC5569653_episodio1008312191 |
| 5 | `03/2023` | NHC5565961_episodio1008280467 |
| 6 | `20` | NHC5569653_episodio1008312191 |
| 7 | `24 años` | NHC365276_episodio1008350813 |
| 8 | `Sevilla` | NHC5573424_episodio1008335354 |
| 9 | `17.` | NHC4198704_episodio1008285899 |
| 10 | `2015` | NHC250452_episodio1008719568 |

### Pipeline (prueba-0.97) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `Martínez` | NHC5578853_episodio1008590097 |
| 2 | `21.08.2024` | NHC302538_episodio1008414306 |
| 3 | `G028` | NHC5569798_episodio1008326489 |
| 4 | `2019` | NHC5569653_episodio1008312191 |
| 5 | `03/2023` | NHC5565961_episodio1008280467 |
| 6 | `20` | NHC5569653_episodio1008312191 |
| 7 | `24 años` | NHC365276_episodio1008350813 |
| 8 | `23/7` | NHC107102_episodio1008411916 |
| 9 | `22/11` | NHC5581074_episodio1008647757 |
| 10 | `Sevilla` | NHC5573424_episodio1008335354 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

