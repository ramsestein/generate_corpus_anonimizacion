# Análisis Delta - Comparativa de Modelos SetFit

**Generado:** 2025-12-29 14:55:36

## 🏆 Ganadores por Métrica

- **Precision (Limpieza):** Pipeline B (Base + SetFit B)
- **Recall (Seguridad):** Pipeline A (Base + SetFit A)
- **F1-Score (Balance):** Pipeline A (Base + SetFit A)

## 🚨 Noise Leakage (Basura que un modelo filtró pero el otro no)

### Pipeline A (Base + SetFit A) filtró, Pipeline B (Base + SetFit B) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `familia` | NHC5622312_episodio1008624363 |
| 2 | `familia` | NHC5628781_episodio1008608304 |
| 3 | `familiar` | NHC5622245_episodio1008637257 |
| 4 | `PUR3` | NHC5593132_episodio1008451534 |
| 5 | `familia` | NHC5611032_episodio1008494463 |
| 6 | `familiar` | NHC512287_episodio1008332398 |
| 7 | `familia` | NHC4992212_episodio1008617462 |
| 8 | `1/4` | NHC4719636_episodio1008256439 |
| 9 | `familia` | NHC4768718_episodio1008594735 |
| 10 | `familiar` | NHC5617177_episodio1008540502 |

### Pipeline B (Base + SetFit B) filtró, Pipeline A (Base + SetFit A) dejó pasar
**Total:** 20 ejemplos

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `E` | NHC5595718_episodio1008461331 |
| 2 | `G` | NHC5586939_episodio1008708692 |
| 3 | `GEL` | NHC5116271_episodio1008527339 |
| 4 | `bebé` | NHC5617207_episodio1008529071 |
| 5 | `ernes` | NHC4986451_episodio1008440273 |
| 6 | `G` | NHC4923757_episodio1008231942 |
| 7 | `BA` | NHC5097797_episodio1008305791 |
| 8 | `parella` | NHC4835235_episodio1008508947 |
| 9 | `h` | NHC5620403_episodio1008548369 |
| 10 | `pad res` | NHC5595566_episodio1008458040 |

## ❌ Over-Cleaning (PII real que fue matado por error)

### Pipeline A (Base + SetFit A) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `posa` | NHC5628591_episodio1008722042 |
| 2 | `MART` | NHC5616835_episodio1008545974 |
| 3 | `6` | NHC5622143_episodio1008564105 |
| 4 | `mano` | NHC5625151_episodio1008657039 |
| 5 | `6` | NHC4945710_episodio1008716369 |
| 6 | `mana` | NHC5627432_episodio1008718272 |
| 7 | `65` | NHC5627432_episodio1008718272 |
| 8 | `27` | NHC5593096_episodio1008493471 |
| 9 | `es` | NHC4953266_episodio1008678487 |
| 10 | `Z` | NHC5613138_episodio1008774094 |

### Pipeline B (Base + SetFit B) - PII Real Eliminado (CRÍTICO)
**Total Fugas:** 20

| # | Entidad | Documento |
|---|---------|----------|
| 1 | `cu` | NHC4788436_episodio1008262953 |
| 2 | `posa` | NHC5628591_episodio1008722042 |
| 3 | `MART` | NHC5616835_episodio1008545974 |
| 4 | `6` | NHC5622143_episodio1008564105 |
| 5 | `mano` | NHC5625151_episodio1008657039 |
| 6 | `mana` | NHC5618865_episodio1008550904 |
| 7 | `6` | NHC4945710_episodio1008716369 |
| 8 | `mana` | NHC5627432_episodio1008718272 |
| 9 | `posa` | NHC5622853_episodio1008573163 |
| 10 | `65` | NHC5627432_episodio1008718272 |

## 📊 Recomendación

**Criterios de Decisión:**
1. **Seguridad (Recall):** El modelo NO debe matar PII real (FN Fugas debe ser 0)
2. **Limpieza (Precision):** Debe filtrar el máximo ruido posible
3. **Balance (F1):** Equilibrio entre ambos

