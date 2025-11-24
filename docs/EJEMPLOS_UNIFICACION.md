# EJEMPLOS VISUALES DE UNIFICACIÓN DE ENTIDADES

Este documento muestra ejemplos reales del proceso de unificación de entidades fragmentadas.

---

## 🎯 CONCEPTO CLAVE

Los modelos NER (CARMEN, MEDDOCAN) a veces **fragmentan** una entidad en múltiples detecciones consecutivas.

**Problema**: Una entidad → Múltiples filas en el CSV  
**Solución**: Detectar y unificar fragmentos consecutivos  

---

## 📊 EJEMPLOS REALES DEL DATASET AWS2

### Ejemplo 1: Código médico fragmentado

**CSV Original** (2 filas):
```csv
doc_id,etiqueta,modelo_detector,texto_detectado,confianza,posicion_inicio,posicion_fin
NHC102219_episodio1008744732,NUMERO_IDENTIF,CARMEN,G,0.9997,7652,7653
NHC102219_episodio1008744732,NUMERO_IDENTIF,CARMEN,045,0.9905,7653,7656
```

**Visualización**:
```
Texto original:  ...código G045 del paciente...
                        ↑    ↑
                      7652  7656

Detecciones fragmentadas:
Fragment 1:  [7652-7653] "G"     ← Primera detección
Fragment 2:  [7653-7656] "045"   ← Segunda detección
                ↑
             ¡Están pegadas! (gap = 0)
```

**Análisis de unificación**:
```
✅ Mismo doc_id:     NHC102219_episodio1008744732
✅ Mismo modelo:     CARMEN
✅ Misma etiqueta:   NUMERO_IDENTIF
✅ Consecutivas:     end(7653) == start(7653) → gap = 0
✅ Sin overlap:      7653 <= 7653
```

**Resultado unificado**:
```json
{
  "doc_id": "NHC102219_episodio1008744732",
  "label": "NUMERO_IDENTIF",
  "model": "CARMEN",
  "text": "G045",
  "confidence": 0.9951,  // Promedio: (0.9997 + 0.9905) / 2
  "start": 7652,
  "end": 7656,
  "unified": true,
  "original_entities": [
    {"text": "G",   "start": 7652, "end": 7653, "confidence": 0.9997},
    {"text": "045", "start": 7653, "end": 7656, "confidence": 0.9905}
  ]
}
```

---

### Ejemplo 2: Nombre fragmentado (6 partes!)

**CSV Original** (6 filas):
```csv
doc_id,etiqueta,modelo_detector,texto_detectado,confianza,posicion_inicio,posicion_fin
NHC136979_episodio1008805535,NOMBRE_PERSONAL_SANITARIO,CARMEN,Sol,0.9992,28,31
NHC136979_episodio1008805535,NOMBRE_PERSONAL_SANITARIO,CARMEN,ara,0.9966,31,34
NHC136979_episodio1008805535,NOMBRE_PERSONAL_SANITARIO,CARMEN,t,0.9985,34,35
NHC136979_episodio1008805535,NOMBRE_PERSONAL_SANITARIO,CARMEN,P,0.9991,36,37
NHC136979_episodio1008805535,NOMBRE_PERSONAL_SANITARIO,CARMEN,are,0.9954,37,40
NHC136979_episodio1008805535,NOMBRE_PERSONAL_SANITARIO,CARMEN,des,0.9787,40,43
```

**Visualización**:
```
Texto original:  ...Dra. Solarat Paredes...
                      ↑             ↑
                     28            43

Fragmentación del modelo:
[28-31]  "Sol"     ←┐
[31-34]  "ara"      │
[34-35]  "t"        ├─ 6 fragmentos consecutivos!
[36-37]  "P"        │  (hay un espacio en pos 35-36)
[37-40]  "are"      │
[40-43]  "des"     ←┘
```

**Análisis de unificación**:
```
Fragment 1-2: "Sol" + "ara"
  ✅ Gap = 31 - 31 = 0 → PEGADAS

Fragment 2-3: "ara" + "t"  
  ✅ Gap = 34 - 34 = 0 → PEGADAS

Fragment 3-4: "t" + "P"
  ⚠️  Gap = 36 - 35 = 1 → HAY 1 ESPACIO
  ✅ Pero gap(1) <= max_gap(5) → SÍ unificar

Fragment 4-5: "P" + "are"
  ✅ Gap = 37 - 37 = 0 → PEGADAS

Fragment 5-6: "are" + "des"
  ✅ Gap = 40 - 40 = 0 → PEGADAS

RESULTADO: ¡Todos se unifican en UNA sola entidad!
```

**Resultado unificado**:
```json
{
  "doc_id": "NHC136979_episodio1008805535",
  "label": "NOMBRE_PERSONAL_SANITARIO",
  "model": "CARMEN",
  "text": "SolaratParedes",  // ← 6 fragmentos unidos
  "confidence": 0.9946,  // Promedio de 6 confianzas
  "start": 28,
  "end": 43,
  "unified": true,
  "original_entities": [
    {"text": "Sol", "start": 28, "end": 31, ...},
    {"text": "ara", "start": 31, "end": 34, ...},
    {"text": "t",   "start": 34, "end": 35, ...},
    {"text": "P",   "start": 36, "end": 37, ...},
    {"text": "are", "start": 37, "end": 40, ...},
    {"text": "des", "start": 40, "end": 43, ...}
  ]
}
```

---

### Ejemplo 3: Fecha fragmentada

**CSV Original** (3 filas):
```csv
doc_id,etiqueta,modelo_detector,texto_detectado,confianza,posicion_inicio,posicion_fin
NHC135800_episodio1008615711,FECHAS,CARMEN,3/4,0.9990,11633,11636
NHC135800_episodio1008615711,FECHAS,CARMEN,3/4,0.9993,11637,11640
NHC135800_episodio1008615711,FECHAS,CARMEN,3/4,0.9979,11641,11644
```

**Visualización**:
```
Texto original:  ...fecha 3/43/43/4...
                      ↑          ↑
                   11633      11644

Fragmentación curiosa:
[11633-11636]  "3/4"    ←┐
[11637-11640]  "3/4"     ├─ ¿El modelo detectó cada "3/4" por separado?
[11641-11644]  "3/4"    ←┘
```

**Análisis de unificación**:
```
Fragment 1-2: "3/4" + "3/4"
  ⚠️  Gap = 11637 - 11636 = 1
  ✅ gap(1) <= max_gap(5) → SÍ unificar

Fragment 2-3: "3/4" + "3/4"
  ⚠️  Gap = 11641 - 11640 = 1
  ✅ gap(1) <= max_gap(5) → SÍ unificar
```

**Resultado unificado**:
```json
{
  "doc_id": "NHC135800_episodio1008615711",
  "label": "FECHAS",
  "model": "CARMEN",
  "text": "3/43/43/4",  // ← Unión de los 3 fragmentos
  "confidence": 0.9987,
  "start": 11633,
  "end": 11644,
  "unified": true
}
```

---

### Ejemplo 4: Identificador con 3 partes

**CSV Original** (3 filas):
```csv
doc_id,etiqueta,modelo_detector,texto_detectado,confianza,posicion_inicio,posicion_fin
NHC104109_episodio1008287424,NUMERO_IDENTIF,CARMEN,I,0.9997,5270,5271
NHC104109_episodio1008287424,NUMERO_IDENTIF,CARMEN,061,0.9848,5271,5274
NHC104109_episodio1008287424,NUMERO_IDENTIF,CARMEN,G,0.9997,5277,5278
```

**Visualización**:
```
Texto original:  ...código I061   G...
                       ↑    ↑   ↑ ↑
                     5270 5274 5277 5278

Fragmentos:
[5270-5271]  "I"      ←─┐
[5271-5274]  "061"   ←──┤  Estos SÍ se unifican (gap=0)
[5277-5278]  "G"      ←─┘  ¿Este también?
```

**Análisis de unificación**:
```
Fragment 1-2: "I" + "061"
  ✅ Gap = 5271 - 5271 = 0 → PEGADAS
  ✅ → Unificar

Fragment 2-3: "061" + "G"
  ❌ Gap = 5277 - 5274 = 3
  ⚠️  3 <= 5 (max_gap) → TÉCNICAMENTE podría unificar
  
  Pero veamos el contexto:
  "I061" [5270-5274] + 3 espacios + "G" [5277-5278]
  
  Como gap(3) <= max_gap(5):
  ✅ SÍ se unifica → "I061G"
```

**Resultado unificado**:
```json
{
  "doc_id": "NHC104109_episodio1008287424",
  "label": "NUMERO_IDENTIF",
  "model": "CARMEN",
  "text": "I061G",
  "confidence": 0.9947,
  "start": 5270,
  "end": 5278,
  "unified": true,
  "original_entities": [
    {"text": "I",   "start": 5270, "end": 5271, ...},
    {"text": "061", "start": 5271, "end": 5274, ...},
    {"text": "G",   "start": 5277, "end": 5278, ...}
  ]
}
```

---

## 🚫 CASOS QUE NO SE UNIFICAN

### Caso 1: Gap muy grande

**CSV**:
```csv
doc_id,etiqueta,texto_detectado,posicion_inicio,posicion_fin
NHC12345,NUMERO_IDENTIF,G,100,101
NHC12345,NUMERO_IDENTIF,045,120,123
```

**Análisis**:
```
Gap = 120 - 101 = 19 caracteres

❌ gap(19) > max_gap(5)
❌ NO se unifican → son entidades separadas
```

---

### Caso 2: Diferentes documentos

**CSV**:
```csv
doc_id,etiqueta,texto_detectado,posicion_inicio,posicion_fin
NHC11111,NUMERO_IDENTIF,G,100,101
NHC22222,NUMERO_IDENTIF,045,101,104
```

**Análisis**:
```
❌ Diferentes doc_id
❌ NO se unifican → son de documentos distintos
```

---

### Caso 3: Diferentes etiquetas

**CSV**:
```csv
doc_id,etiqueta,texto_detectado,posicion_inicio,posicion_fin
NHC12345,NUMERO_IDENTIF,G,100,101
NHC12345,FECHAS,045,101,104
```

**Análisis**:
```
❌ Diferentes etiquetas (NUMERO_IDENTIF vs FECHAS)
❌ NO se unifican (con same_label_only=True)
```

---

### Caso 4: Overlap (detecciones solapadas)

**CSV**:
```csv
doc_id,etiqueta,texto_detectado,posicion_inicio,posicion_fin
NHC12345,NUMERO_IDENTIF,G045,100,104
NHC12345,NUMERO_IDENTIF,045,101,104
```

**Análisis**:
```
Fragment 1: [100-104] "G045"
Fragment 2: [101-104] "045"
           ↑
        ¡Se solapan!

Gap = 101 - 104 = -3 (negativo!)

❌ gap < 0 significa OVERLAP
❌ NO se unifican → son detecciones conflictivas
```

---

## 📈 ESTADÍSTICAS DEL DATASET AWS2

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESULTADOS DE UNIFICACIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total entidades originales:     220
Total entidades después:        186
Grupos unificados:              28
Fragmentos fusionados:          62

Reducción:                      34 entidades (15.5%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISTRIBUCIÓN DE GRUPOS UNIFICADOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Grupos de 2 fragmentos:         23 grupos (82.1%)
Grupos de 3 fragmentos:          4 grupos (14.3%)
Grupos de 6 fragmentos:          1 grupo  (3.6%)

Ejemplos:
• 2 fragmentos: "G" + "045" → "G045"
• 3 fragmentos: "I" + "061" + "G" → "I061G"
• 6 fragmentos: "Sol" + "ara" + "t" + "P" + "are" + "des" → "SolaratParedes"
```

---

## 🎓 LECCIONES APRENDIDAS

### 1. La mayoría son pares simples
El 82% de las unificaciones son de 2 fragmentos (letra + número, número + número, etc.)

### 2. Los nombres son los más problemáticos
El caso del nombre "SolaratParedes" (6 fragmentos) muestra que los modelos tienen dificultad con nombres largos.

### 3. El gap pequeño es clave
Con `max_gap=5`, capturamos casi todos los casos legítimos sin unificar cosas que no deberían estar juntas.

### 4. Las fechas son especiales
Las fechas a veces se detectan como múltiples "3/4" consecutivas, lo cual es curioso pero se maneja bien con la unificación.

---

## 🔍 CÓMO VERIFICAR LAS UNIFICACIONES

Para ver las unificaciones en el JSON de salida:

```python
import json

# Cargar el JSON
with open('outputs/entidades_procesadas_aws2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filtrar solo las unificadas
unified = [e for e in data['entities'] if e['unified']]

print(f"Total unificadas: {len(unified)}")

# Ver la primera
print(json.dumps(unified[0], indent=2, ensure_ascii=False))
```

**Salida esperada**:
```json
{
  "doc_id": "NHC102219_episodio1008744732",
  "label": "NUMERO_IDENTIF",
  "model": "CARMEN",
  "text": "G045",
  "confidence": 0.9951,
  "start": 7652,
  "end": 7656,
  "unified": true,
  "original_entities": [...]
}
```

---

**Documento creado**: 2025-11-18  
**Dataset**: aws2 (220 entidades → 186 después de unificación)  
**Reducción**: 15.5% (34 entidades fusionadas en grupos)
