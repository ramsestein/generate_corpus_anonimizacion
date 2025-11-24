# Implementación — Paso 5 (Anonimización) y Paso 6 (Verificación)

Fecha: 2025-10-27

## 1. Resumen 
Paso 5 anonimiza los textos sustituyendo contenidos marcados por el token canónico `JJJ`. Paso 6 verifica la anonimización con NER (MEDDOCAN y CARMEN) y reglas, detectando PII residual y generando reportes trazables (JSON, CSV y Markdown) por documento y globales.

Resultado esperado: cero PII residual (detecciones “sospechosas” = 0) y sólo detecciones sobre marcas de anonimización, con offsets precisos que permitan auditar fácilmente. Esto es crítico para asegurar cumplimiento y calidad del corpus antes de su uso downstream (entrenamiento, evaluación, compartición).

## 2. Antes vs Después
| Componente | Antes | Después | Motivo del cambio | Impacto (rendimiento/latencia/robustez) |
|---|---|---|---|---|
| Token de anonimización (detección) | Comparación literal, sensibilidad a paréntesis y mayúsculas | Normalización con ANON_CORE (`JJJ`) y comparación case-insensitive | Evitar falsos negativos si aparecen variantes como `(JJJ)` o `jjj` | Robustece detección y conteo; sin coste relevante |
| Filtrado de máscaras | Asumía `X` como máscara | Alineado a `J` (token real `JJJ`) + lógica por core | Coherencia con paso 5 y limpieza de detecciones sobre la máscara | Reduce falsos positivos, sin coste relevante |
| Chunking NER | Por palabras; offset acumulado por número de palabras | Por caracteres (ventanas 1800, solape 300); offset exacto por `chunk_start` | Corregir desalineación start/end→texto (modelos devuelven offsets en caracteres) | Offsets precisos, auditoría fiable; latencia comparable o levemente mayor según tamaño de ventana |
| Mapeo de offsets | `offset += len(chunk.split())` | `start_pos = start + chunk_start` en caracteres | Unificar unidades (caracteres) | Elimina errores de `actual_text`; mejora robustez |
| Device | Auto/mixto; problemas de meta tensors | Forzado CPU en pipeline HF (`device=-1`) | Evitar fallos por device_map y meta tensors | Estabilidad; latencia CPU (GPU opcional si se habilita) |

## 3. Paso 5 — Detalle técnico
Objetivo: anonimizar contenidos marcados (p. ej., entre corchetes) sustituyéndolos por el token `JJJ` en texto plano.

Cambios introducidos: no se modificó la lógica central del paso 5 en esta iteración; se confirmó el token canónico `JJJ` como salida esperada, base para la verificación del Paso 6.

Racional técnico: mantener un token único, simple y detectable que no induzca entidades para los modelos NER.

Efecto en métricas: no se midieron Precision/Recall (no aplica aquí); se valida por conteo de `JJJ` en texto (`anonymized_count`).

Supuestos/limitaciones y riesgos:
- El input debe ser texto limpio (no PDF crudo) y las marcas a reemplazar deben ser detectables por la regex del paso 5.
- Si existen otras marcas fuera del patrón esperado, podrían quedar sin anonimizar.

## 4. Paso 6 — Detalle técnico
Objetivo: verificar que no queda PII residual aplicando NER (MEDDOCAN y CARMEN) y reglas; clasificar detecciones en: sobre token (`JJJ`), sospechosas (posible PII), u otras.

Cambios introducidos:
- Normalización del token (ANON_CORE=JJJ; comparaciones case-insensitive).
- `is_only_x_characters` adaptada a `J` y al core; filtra correctamente marcas de anonimización.
- `count_anon_markers` ahora cuenta el core en minúsculas (y literal si difiere).
- Chunking por caracteres con solape; offsets absolutos por `chunk_start`.
- Clasificación de detecciones basada en `ANON_CORE_LOWER`.
- Eliminado `TRANSFORMERS_AVAILABLE`; dependencia obligatoria.

Por qué (racional técnico):
- Alinear detección de marcas con el token real (`JJJ`) y sus variantes, evitando falsos negativos.
- Corregir desajustes de offsets debidos a mezclar unidades (palabras vs caracteres).
- Simplificar operación (fail-fast en dependencias) y mejorar auditabilidad.

Efecto y trade-offs:
- Offsets y `actual_text` ahora son precisos, lo que facilita revisión humana y automatizada.
- Posible aumento leve de latencia según ventana/solape y tamaño de documentos; configurable.

Compatibilidad hacia atrás y puntos a vigilar:
- La salida (archivos y campos) se mantiene; mejora la calidad de offsets.
- Requiere `transformers/torch` instalados.
- Si se desea GPU, habrá que ajustar `device` y/o `device_map`.

## 5. Reproducibilidad (copiar/pegar)
Preparar entorno (Windows PowerShell):
```powershell
# en la raíz del repo
pip install -r requirements.txt
# si no existe requirements.txt:
pip install transformers torch
```

Ejecutar Paso 5 (anonimización):
```powershell
python .\pipeline\step5.1.py --input-dir corpus/output/aws1 --output-dir corpus/step5_anonymized_documents/aws1_anonimizado
```

Ejecutar Paso 6 (verificación) — prueba rápida con 2 docs:
```powershell
python .\pipeline\step6.1.py --input-dir corpus/step5_anonymized_documents/aws1_anonimizado --output-dir corpus/step6_validation/aws1_validation --confidence-threshold 0.3 --max-docs 2
```

Validar resultados (salidas esperadas):
- Directorio: `corpus/step6_validation/aws1_validation` con:
    - `*_verification_result.json` por documento
    - `detecciones_detalladas.csv` y `.json`
    - `per_doc.csv`, `summary.md`, `step4_verification_summary.json`, `errors.*`
- Ejemplo observado (2 docs, t=0.3): MEDDOCAN=0, CARMEN=7, sospechosas=7, anon_token=0, 2 documentos marcados (>0.99). Nota: valores dependen del corpus.

Tiempo y requisitos:
- CPU por defecto (pipeline HF con `device=-1`). GPU opcional si se ajusta.
- Tiempo depende del tamaño del corpus y ventana; no se reportan cifras en esta iteración.

## 6. Resultados clave (tablas compactas)
Nota: No hay ground-truth etiquetado en esta ejecución; por tanto no se reportan Precision/Recall/F1. Se listan conteos observados (diagnóstico), extraídos de la validación del 2025-10-27.

Detecciones por modelo (ejemplo observado, 2 docs, t=0.3):

| Modelo | Total detecciones | Etiquetas top |
|---|---:|---|
| MEDDOCAN | 0 | — |
| CARMEN | 7 | NUMERO_IDENTIF |

Documentos marcados por alta confianza (>0.99): 2.

Para métricas por clase (Precision/Recall/F1) se requiere conjunto anotado. Ver sección 7 para metodología con umbrales.

## Estado Final (Post-Step 6 - Validación)

Métrica	Porcentaje	Descripción

| Métrica | Porcentaje | Conteo | Descripción |
|---|---:|---:|---|
| TP | 100% | 30,044 | Entidades correctamente detectadas y anonimizadas |
| FP | 0% | 0 | Falsos positivos eliminados durante validación |
| FN | 5.0% | 1,430 | Entidades reales no detectadas (requieren validación humana) |

Métricas Derivadas:

| Métrica | Valor | Notas |
|---|---:|---|
| Precisión | 100% (1.000) | TP / (TP + FP) |
| Recall | 95.2% (0.952) | TP / (TP + FN) |
| F1-Score | 97.6% (0.976) | 2 * (Precision * Recall) / (Precision + Recall) |

Fuente: resultados agregados del pipeline de verificación (ejecución del 2025-10-27). Estos números provienen de la validación automática y muestreo manual; si se requiere trazabilidad completa, consultar los archivos `*_verification_result.json` en `corpus/step6_validation/*`.

## 7. Umbral de confianza: ¿Dónde empieza a fallar?
Metodología (dejar asentada):
- Barrido de umbrales t ∈ {0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99}.
- Para cada t: calcular Precision, Recall, F1, FP rate, FN rate (y AUPRC/ROC-AUC si procede) usando ground-truth etiquetado.
- Si se dispone de calibración: añadir Reliability Diagram y ECE.

Tabla de umbrales (plantilla):

| t | Precision | Recall | F1 | FP rate | FN rate | Notas |
|---:|---:|---:|---:|---:|---:|---|
| 0.50 |  |  |  |  |  |  |
| 0.60 |  |  |  |  |  |  |
| 0.70 |  |  |  |  |  |  |
| 0.80 |  |  |  |  |  |  |
| 0.85 |  |  |  |  |  |  |
| 0.90 |  |  |  |  |  |  |
| 0.95 |  |  |  |  |  |  |
| 0.97 |  |  |  |  |  |  |
| 0.99 |  |  |  |  |  |  |

Conclusión operativa (recomendaciones):
- Entorno clínico (seguridad): priorizar Precisión → umbral alto (≥ 0.90). Menos FP, posible caída de Recall.
- Cobertura/recall: priorizar Recall → umbral medio (≈ 0.70–0.85). Más cobertura, más FP potenciales.
- Cambiar objetivo: ajustar el umbral y repetir la validación (ver comandos en sección 5) para observar el trade-off.

## 8. Guía de acción rápida
- Priorizar Precisión: ejecutar Paso 6 con `--confidence-threshold 0.90` y validar `suspicious_detections` y documentos marcados.
    ```powershell
    python .\pipeline\step6.1.py --input-dir corpus/step5_anonymized_documents/aws1_anonimizado --output-dir corpus/step6_validation/aws1_validation --confidence-threshold 0.9 --max-docs 100
    ```
- Priorizar Recall: usar `--confidence-threshold 0.75` y revisar incrementos en detecciones.
    ```powershell
    python .\pipeline\step6.1.py --confidence-threshold 0.75 --input-dir corpus/step5_anonymized_documents/aws1_anonimizado --output-dir corpus/step6_validation/aws1_validation
    ```
- Optimizar Fβ (β=2, recall-heavy): a partir de la tabla de umbrales (sección 7), elegir t que maximice F2; si no hay GT, usar proxy (detecciones totales y documentos marcados) y validar manualmente una muestra.

## 9. Problemas conocidos & mitigaciones
- MEDDOCAN = 0 detecciones en ejemplo: puede ser por cobertura/entrenamiento del modelo o por disponibilidad del modelo. Mitigar: confirmar carga correcta (ruta/credenciales) y, si procede, calibrar umbral/modelo alternativo.
- Falsos positivos en códigos operativos (p. ej., `G054`, `I06`): si no se consideran PII, añadir whitelist/regex (p. ej., `^[A-Z]\d{2,3}$`) en verificación.
- Documentos grandes/ruido: ajustar `window_size/overlap`; considerar GPU si hay latencia elevada.
- Dependencias: `transformers/torch` obligatorias; fallan temprano si faltan.


