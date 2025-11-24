#!/usr/bin/env python3
"""
PIPELINE DE EVALUACIÓN CON JUEZ LLM - PASO 1: PREPROCESAMIENTO
===============================================================

Sistema de evaluación de entidades detectadas usando un LLM como juez.

OBJETIVO PRINCIPAL: MAXIMIZAR RECALL (minimizar FN)
- El juez debe favorecer TRUE en caso de duda
- No aplicar filtros previos de etiquetas
- Mantener precisión aceptable evitando FP excesivos

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASO 1 (ESTE SCRIPT): PREPROCESAMIENTO Y UNIFICACIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANTE: **SIN FILTRADO DE ETIQUETAS**

Este paso NO descarta ninguna entidad por su tipo de etiqueta.
Anteriormente se filtraban etiquetas como FAMILIARES_SUJETO_ASISTENCIA,
PROFESION, etc. Esa lógica ha sido **COMPLETAMENTE ELIMINADA**.

¿QUÉ HACE ESTE PASO?

1. CARGA COMPLETA DEL CSV
    Lee TODAS las entidades sin excepción
    No aplica ningún filtro por tipo de etiqueta
    Valida formato y rangos de valores
    Genera estadísticas de carga

2. UNIFICACIÓN DE FRAGMENTOS CONSECUTIVOS
   
   PROBLEMA: Los modelos NER a veces detectan UNA entidad como MÚLTIPLES fragmentos.
   
   Ejemplo del CSV real:
     Row 1: text="G",   start=7652, end=7653, label=NUMERO_IDENTIF
     Row 2: text="045", start=7653, end=7656, label=NUMERO_IDENTIF
   
   En realidad es: "G045" (un solo código médico)
   
   SOLUCIÓN: Detectar fragmentos consecutivos y unificarlos.
   
   Criterios para unificar:
    Mismo documento
    Mismo modelo detector
    Misma etiqueta
    Posiciones consecutivas (gap ≤ 5 caracteres)
    Sin overlap
   
   Resultado:
     → Entity unificada: text="G045", start=7652, end=7656, unified=True

3. ANÁLISIS ESTADÍSTICO
    Distribución por etiqueta
    Distribución por modelo (CARMEN vs MEDDOCAN)
    Distribución de confianza
    Entidades unificadas vs originales

4. EXPORTACIÓN A JSON
    Formato estructurado para el juez LLM
    Incluye metadata completa
    Preserva referencias a fragmentos originales

5. COLUMNA MANUAL_CORRECTION
   Esta columna almacena correcciones/etiquetas manuales humanas.
   
   ⚠️  IMPORTANTE - EXCLUSIÓN DEL LLM:
   - Esta columna se carga y preserva en todo el pipeline
   - Se incluye en exportaciones finales para métricas
   - NO se envía NUNCA al LLM en los prompts
   - NO contamina el contexto del modelo
   - Sirve solo para evaluación posterior

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE COMPLETO (6 PASOS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.  Preprocesar CSV + Unificar fragmentos (ESTE SCRIPT)
2.  Cargar etiquetas gold
3.  Configurar juez LLM
4.  Evaluar con juez LLM (TRUE/FALSE)
5.  Calcular métricas (TP, FP, FN, Precision, Recall, F1)
6.  Experimentar con diferentes chunking sizes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import json
import csv
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class Entity:
    """
    Representa una entidad detectada por los modelos.
    
    Attributes:
        doc_id: ID del documento
        label: Etiqueta de la entidad (ej: NOMBRE_SUJETO_ASISTENCIA)
        model: Modelo detector (CARMEN, MEDDOCAN)
        text: Texto detectado
        confidence: Nivel de confianza (0-1)
        start: Posición inicial en el texto
        end: Posición final en el texto
        unified: Si esta entidad fue unificada con otras
        manual_correction: Corrección/etiqueta manual humana (NO se envía al LLM)
    """
    doc_id: str
    label: str
    model: str
    text: str
    confidence: float
    start: int
    end: int
    unified: bool = False
    manual_correction: str = ""
    
    def to_dict(self) -> Dict:
        """Convierte la entidad a diccionario"""
        return asdict(self)
    
    def __repr__(self) -> str:
        return f"Entity({self.label}: '{self.text}' [{self.start}-{self.end}] conf={self.confidence:.3f})"


@dataclass
class ProcessingStats:
    """
    Estadísticas del procesamiento de entidades.
    """
    total_raw_entities: int = 0
    total_unified_entities: int = 0
    entities_merged: int = 0
    entities_by_label: Dict[str, int] = field(default_factory=dict)
    entities_by_model: Dict[str, int] = field(default_factory=dict)
    documents_processed: int = 0
    
    def to_dict(self) -> Dict:
        """Convierte las estadísticas a diccionario"""
        return asdict(self)


# ============================================================================
# FUNCIONES DE LOGGING Y DEBUG
# ============================================================================

def debug_print(message: str, level: str = "INFO"):
    """
    Imprime mensajes con timestamp y nivel de logging.
    
    Args:
        message: Mensaje a imprimir
        level: Nivel de log (DEBUG, INFO, WARN, ERROR)
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ============================================================================
# PASO 1: CARGA Y PREPROCESAMIENTO DEL CSV
# ============================================================================

def load_csv_detections(csv_path: str) -> Tuple[List[Entity], ProcessingStats]:
    """
    Carga las detecciones desde el CSV sin aplicar ningún filtrado.
    
    Este es el PRIMER PASO OBLIGATORIO del pipeline.
    NO se eliminan entidades por etiqueta, confianza u otro criterio.
    Se cargan TODAS las entidades detectadas.
    
    Args:
        csv_path: Ruta al archivo CSV con detecciones
        
    Returns:
        Tuple con:
        - Lista de entidades cargadas
        - Estadísticas de procesamiento
        
    Raises:
        FileNotFoundError: Si el CSV no existe
        ValueError: Si el CSV tiene formato incorrecto
    """
    debug_print(f"Cargando detecciones desde: {csv_path}", "INFO")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo CSV: {csv_path}")
    
    entities = []
    stats = ProcessingStats()
    
    # Intentar múltiples encodings
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    last_error = None
    file_content = None
    
    for encoding in encodings_to_try:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f, delimiter=';')
                
                # Verificar columnas requeridas
                required_columns = {'doc_id', 'etiqueta', 'modelo_detector', 
                                  'texto_detectado', 'confianza', 'posicion_inicio', 'posicion_fin'}
                
                if not required_columns.issubset(set(reader.fieldnames or [])):
                    raise ValueError(f"CSV debe contener columnas: {required_columns}")
                
                debug_print(f"  ✓ Archivo leído con encoding: {encoding}", "INFO")
                break  # Encoding exitoso, salir del loop
        except UnicodeDecodeError as e:
            last_error = e
            continue  # Probar siguiente encoding
    else:
        # Si ningún encoding funcionó
        raise UnicodeDecodeError(
            'multiple', b'', 0, 1, 
            f"No se pudo leer el archivo con ninguno de estos encodings: {encodings_to_try}. Último error: {last_error}"
        )
    
    try:
        with open(csv_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=';')
            
            # Verificar columnas requeridas ya realizada arriba
            required_columns = {'doc_id', 'etiqueta', 'modelo_detector', 
                              'texto_detectado', 'confianza', 'posicion_inicio', 'posicion_fin'}
            
            if not required_columns.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"CSV debe contener columnas: {required_columns}")
            
            # Verificar si existe la columna opcional manual_correction (varios nombres posibles)
            manual_col_names = ['manual_correction', 'Corrección manual', 'manual_label', 'correccion_manual']
            manual_col = None
            for col_name in manual_col_names:
                if col_name in (reader.fieldnames or []):
                    manual_col = col_name
                    break
            
            has_manual_correction = manual_col is not None
            if has_manual_correction:
                debug_print(f"Columna de corrección manual detectada: '{manual_col}' - se preservará", "INFO")
            
            for row_num, row in enumerate(reader, start=2):  # start=2 porque línea 1 es header
                try:
                    # Extraer datos básicos
                    doc_id = row['doc_id'].strip()
                    label = row['etiqueta'].strip()
                    model = row['modelo_detector'].strip()
                    text = row['texto_detectado']  # NO strip aquí, preservar espacios
                    
                    # Cargar manual_correction si existe, sino dejar vacío
                    manual_correction = row.get(manual_col, '').strip() if has_manual_correction else ''
                    
                    # Parsear numéricos con validación
                    try:
                        confidence = float(row['confianza'])
                        start = int(row['posicion_inicio'])
                        end = int(row['posicion_fin'])
                    except ValueError as e:
                        debug_print(
                            f"Línea {row_num}: Error parseando valores numéricos: {e}",
                            "WARN"
                        )
                        continue
                    
                    # Validar rangos
                    if not (0 <= confidence <= 1):
                        debug_print(
                            f"Línea {row_num}: Confianza fuera de rango [0,1]: {confidence}",
                            "WARN"
                        )
                        # No descartamos, pero normalizamos
                        confidence = max(0.0, min(1.0, confidence))
                    
                    if start < 0 or end < start:
                        debug_print(
                            f"Línea {row_num}: Posiciones inválidas: start={start}, end={end}",
                            "WARN"
                        )
                        continue
                    
                    # Crear entidad
                    entity = Entity(
                        doc_id=doc_id,
                        label=label,
                        model=model,
                        text=text,
                        confidence=confidence,
                        start=start,
                        end=end,
                        manual_correction=manual_correction
                    )
                    
                    entities.append(entity)
                    stats.total_raw_entities += 1
                    
                    # Actualizar estadísticas por etiqueta y modelo
                    stats.entities_by_label[label] = stats.entities_by_label.get(label, 0) + 1
                    stats.entities_by_model[model] = stats.entities_by_model.get(model, 0) + 1
                    
                except Exception as e:
                    debug_print(f"Línea {row_num}: Error procesando fila: {e}", "WARN")
                    continue
        
        # Contar documentos únicos
        unique_docs = set(e.doc_id for e in entities)
        stats.documents_processed = len(unique_docs)
        
        debug_print(f"✓ Cargadas {stats.total_raw_entities} entidades de {stats.documents_processed} documentos", "INFO")
        debug_print(f"  Etiquetas únicas: {len(stats.entities_by_label)}", "DEBUG")
        debug_print(f"  Modelos: {list(stats.entities_by_model.keys())}", "DEBUG")
        
        return entities, stats
        
    except Exception as e:
        debug_print(f"Error crítico cargando CSV: {e}", "ERROR")
        raise


# ============================================================================
# PASO 2: UNIFICACIÓN DE ENTIDADES FRAGMENTADAS
# ============================================================================

def should_merge_entities(entity1: Entity, entity2: Entity, 
                         max_gap: int = 5, 
                         same_label_only: bool = True) -> bool:
    """
    Determina si dos entidades consecutivas deberían unificarse.
    
    CRITERIOS ESTRICTOS DE UNIFICACIÓN:
    ===================================
    
    Una entidad debe unificarse con la siguiente SI Y SOLO SI:
    
    1.  Mismo documento (doc_id)
    2.  Mismo modelo detector (CARMEN o MEDDOCAN)
    3.  Misma etiqueta/tipo (si same_label_only=True)
    4.  Son CONSECUTIVAS en el texto (entity1.end <= entity2.start)
    5.  El GAP entre ellas es pequeño (≤ max_gap caracteres)
    6.  NO hay overlap (entity1.end <= entity2.start)
    
    CASOS TÍPICOS QUE SE UNIFICAN:
    ==============================
    
     Códigos fragmentados:
       - "G" (pos 100-101) + "045" (pos 101-104) → "G045"
       - "I" (pos 50-51) + "064" (pos 51-54) → "I064"
    
     Nombres fragmentados por tokenización:
       - "Sol" (pos 28-31) + "ara" (pos 31-34) + "t" (pos 34-35) → "Solarat"
       - "P" (pos 36-37) + "are" (pos 37-40) + "des" (pos 40-43) → "Paredes"
    
    Fechas fragmentadas:
       - "26" (pos 100-102) + "/" (pos 102-103) + "7" (pos 103-104) → "26/7"
       - "3/4" (pos 200-203) + "3/4" (pos 203-206) → "3/43/4" (si están pegadas)
    
     Identificadores con puntos o guiones:
       - "06" (pos 50-52) + "." (pos 52-53) + "2" (pos 53-54) → "06.2"
    
    CASOS QUE NO SE UNIFICAN:
    =========================
    
     Diferentes documentos → Son entidades independientes
     Diferentes modelos → Cada modelo tiene su propia detección
     Diferentes etiquetas (si same_label_only=True) → Son tipos distintos
     Gap grande (> max_gap) → No son consecutivas en el texto original
     Overlap (entity1.end > entity2.start) → Detecciones conflictivas
    
    EJEMPLO REAL DEL CSV:
    ====================
    
    Entrada:
      Row 1: doc_id=NHC102219, label=NUMERO_IDENTIF, text="G",   start=7652, end=7653
      Row 2: doc_id=NHC102219, label=NUMERO_IDENTIF, text="045", start=7653, end=7656
    
    Análisis:
      - Mismo doc_id 
      - Mismo modelo (CARMEN) 
      - Misma label (NUMERO_IDENTIF) 
      - Consecutivas: end(7653) == start(7653) → gap = 0 
      - No overlap: 7653 <= 7653 
    
    Resultado: UNIFICAR → "G045" (start=7652, end=7656)
    
    Args:
        entity1: Primera entidad (DEBE tener start < entity2.start)
        entity2: Segunda entidad
        max_gap: Máximo gap de caracteres para considerar unificación (default: 5)
                 Gap = 0 significa que están pegadas (entity1.end == entity2.start)
                 Gap > 0 permite caracteres intermedios (espacios, puntuación)
        same_label_only: Si True, solo unifica entidades con misma etiqueta (recomendado: True)
        
    Returns:
        True si las entidades deberían unificarse, False en caso contrario
    """
    # CRITERIO 1 y 2: Mismo documento y mismo modelo
    if entity1.doc_id != entity2.doc_id or entity1.model != entity2.model:
        return False
    
    # CRITERIO 3: Misma etiqueta (si es requerido)
    if same_label_only and entity1.label != entity2.label:
        return False
    
    # CRITERIO 4, 5 y 6: Calcular gap y validar que sean consecutivas sin overlap
    # Gap = cuántos caracteres hay entre el final de entity1 y el inicio de entity2
    gap = entity2.start - entity1.end
    
    # Gap < 0 significa OVERLAP (detecciones que se solapan) → NO unificar
    # Gap = 0 significa PEGADAS (una termina exactamente donde empieza la otra) → SÍ unificar
    # Gap > 0 y <= max_gap significa CERCANAS (hay caracteres entre ellas) → SÍ unificar si gap es pequeño
    # Gap > max_gap significa SEPARADAS (están lejos) → NO unificar
    
    if 0 <= gap <= max_gap:
        return True
    
    return False


def merge_entities(entities: List[Entity]) -> Entity:
    """
    Une múltiples entidades en una sola.
    
    La entidad unificada tendrá:
    - Texto combinado
    - Posición desde el start del primero hasta el end del último
    - Confianza promedio
    - Referencias a las entidades originales
    
    Args:
        entities: Lista de entidades a unir (deben estar ordenadas por posición)
        
    Returns:
        Entidad unificada
    """
    if not entities:
        raise ValueError("No se pueden unir 0 entidades")
    
    if len(entities) == 1:
        return entities[0]
    
    # Ordenar por posición por seguridad
    sorted_entities = sorted(entities, key=lambda e: e.start)
    
    # Tomar datos del primer elemento
    first = sorted_entities[0]
    last = sorted_entities[-1]
    
    # Calcular confianza promedio
    avg_confidence = sum(e.confidence for e in sorted_entities) / len(sorted_entities)
    
    # Crear entidad unificada
    # IMPORTANTE: manual_correction se toma del primer fragmento (si existe)
    unified = Entity(
        doc_id=first.doc_id,
        label=first.label,  # Usar label del primero
        model=first.model,
        text="",  # Se construirá después
        confidence=avg_confidence,
        start=first.start,
        end=last.end,
        unified=True,
        manual_correction=first.manual_correction  # Preservar corrección manual
    )
    
    # Construir texto combinado (necesitamos el texto original para gaps)
    # Por ahora, concatenamos los textos individuales
    # NOTA: Idealmente necesitaríamos el texto original del documento
    unified.text = "".join(e.text for e in sorted_entities)
    
    return unified


def unify_fragmented_entities(entities: List[Entity], 
                              max_gap: int = 5,
                              same_label_only: bool = True) -> Tuple[List[Entity], int]:
    """
    Unifica entidades que fueron detectadas de forma fragmentada por los modelos.
    
    OBJETIVO:
    =========
    Los modelos NER (CARMEN, MEDDOCAN) a veces detectan UNA entidad como MÚLTIPLES 
    fragmentos consecutivos. Esta función los detecta y los une en una sola entidad.
    
    ESTRATEGIA DE UNIFICACIÓN:
    ==========================
    
    1. AGRUPAR por documento
       - Procesar cada documento independientemente
       - Evitar unificaciones incorrectas entre documentos diferentes
    
    2. ORDENAR por posición
       - Ordenar entidades por (start, end) dentro de cada documento
       - Garantizar que procesamos en orden secuencial
    
    3. DETECTAR fragmentos consecutivos
       - Iterar por las entidades ordenadas
       - Usar should_merge_entities() para detectar fragmentos consecutivos
       - Acumular todos los fragmentos consecutivos en un "grupo"
    
    4. UNIFICAR grupos
       - Si un grupo tiene >1 entidad → unificarlas con merge_entities()
       - Si un grupo tiene 1 entidad → mantenerla sin cambios
    
    EJEMPLO DE PROCESAMIENTO:
    =========================
    
    Entrada (documento NHC102219):
      [
        Entity(label=NUMERO_IDENTIF, text="G",   start=7652, end=7653),
        Entity(label=NUMERO_IDENTIF, text="045", start=7653, end=7656),
        Entity(label=FECHAS,         text="26",  start=8000, end=8002),
        Entity(label=FECHAS,         text="/",   start=8002, end=8003),
        Entity(label=FECHAS,         text="7",   start=8003, end=8004)
      ]
    
    Procesamiento:
      Grupo 1: ["G", "045"] → consecutivas, misma label → UNIFICAR
      Grupo 2: ["26", "/", "7"] → consecutivas, misma label → UNIFICAR
    
    Salida:
      [
        Entity(label=NUMERO_IDENTIF, text="G045", start=7652, end=7656, unified=True),
        Entity(label=FECHAS,         text="26/7", start=8000, end=8004, unified=True)
      ]
    
    IMPORTANTE - SIN FILTRADO:
    ==========================
    Esta función NO descarta ninguna entidad.
    Solo las reorganiza unificando fragmentos.
    
    Si una entidad no tiene fragmentos consecutivos, se mantiene tal cual.
    TODAS las entidades (unificadas o no) aparecen en la salida.
    
    Args:
        entities: Lista completa de entidades del CSV (sin filtrar)
        max_gap: Gap máximo de caracteres para considerar dos entidades como consecutivas
                 - 0: Solo pega entidades exactamente adyacentes
                 - 5 (default): Permite hasta 5 caracteres entre fragmentos
        same_label_only: Si True, solo unifica fragmentos con la misma etiqueta
                        (Recomendado: True para evitar unificaciones incorrectas)
        
    Returns:
        Tuple con:
        - Lista de entidades después de unificación (puede ser más corta que la entrada)
        - Número total de entidades originales que fueron fusionadas
        
    Ejemplo de conteo:
        Entrada: 10 entidades
        Se unifican: "G" + "045" (2 entidades) y "26" + "/" + "7" (3 entidades)
        Salida: 7 entidades (10 - 2 - 3 + 1 + 1 = 7)
        total_merged: 5 (las 2 + 3 que se fusionaron)
    """
    debug_print(f"Iniciando unificación de entidades fragmentadas...", "INFO")
    debug_print(f"  Parámetros: max_gap={max_gap}, same_label_only={same_label_only}", "DEBUG")
    debug_print(f"  Entidades de entrada: {len(entities)}", "DEBUG")
    
    # PASO 1: AGRUPAR por documento
    entities_by_doc = defaultdict(list)
    for entity in entities:
        entities_by_doc[entity.doc_id].append(entity)
    
    debug_print(f"  Documentos únicos: {len(entities_by_doc)}", "DEBUG")
    
    unified_entities = []
    total_merged = 0
    total_groups_unified = 0
    
    # PASO 2: PROCESAR cada documento
    for doc_id, doc_entities in entities_by_doc.items():
        # PASO 2.1: ORDENAR por posición (start, end)
        sorted_entities = sorted(doc_entities, key=lambda e: (e.start, e.end))
        
        i = 0
        while i < len(sorted_entities):
            current = sorted_entities[i]
            group = [current]
            
            # PASO 3: DETECTAR fragmentos consecutivos
            # Intentamos agregar al grupo todas las entidades consecutivas
            j = i + 1
            while j < len(sorted_entities):
                next_entity = sorted_entities[j]
                
                # Verificar si el siguiente fragmento pertenece al grupo
                # Comparamos con el ÚLTIMO del grupo (group[-1]) no con el primero
                if should_merge_entities(group[-1], next_entity, max_gap, same_label_only):
                    group.append(next_entity)
                    j += 1
                else:
                    # Ya no hay más fragmentos consecutivos
                    break
            
            # PASO 4: UNIFICAR o mantener
            if len(group) > 1:
                # Hay múltiples fragmentos → UNIFICAR
                unified = merge_entities(group)
                unified_entities.append(unified)
                total_merged += len(group)
                total_groups_unified += 1
                
                # Log detallado del grupo unificado
                fragments_info = " + ".join([f'"{e.text}"' for e in group])
                debug_print(
                    f"  Doc {doc_id}: Unificadas {len(group)} entidades: {fragments_info} → '{unified.text}' "
                    f"(pos {unified.start}-{unified.end})",
                    "DEBUG"
                )
            else:
                # Entidad individual sin fragmentos → mantener como está
                unified_entities.append(current)
            
            # Avanzar al siguiente grupo
            i = j if j > i + 1 else i + 1
    
    # RESUMEN FINAL
    debug_print(
        f"✓ Unificación completada:",
        "INFO"
    )
    debug_print(f"    Entidades entrada:  {len(entities)}", "INFO")
    debug_print(f"    Entidades salida:   {len(unified_entities)}", "INFO")
    debug_print(f"    Grupos unificados:  {total_groups_unified}", "INFO")
    debug_print(f"    Fragmentos fusionados: {total_merged}", "INFO")
    debug_print(f"    Reducción: {len(entities) - len(unified_entities)} entidades", "INFO")
    
    return unified_entities, total_merged


# ============================================================================
# FUNCIONES DE ANÁLISIS Y ESTADÍSTICAS
# ============================================================================

def analyze_entities(entities: List[Entity]) -> Dict:
    """
    Genera análisis estadístico de las entidades procesadas.
    
    Args:
        entities: Lista de entidades a analizar
        
    Returns:
        Dict con estadísticas detalladas
    """
    analysis = {
        'total_entities': len(entities),
        'by_label': defaultdict(int),
        'by_model': defaultdict(int),
        'by_document': defaultdict(int),
        'confidence_distribution': {
            'very_high': 0,  # >= 0.95
            'high': 0,       # >= 0.80
            'medium': 0,     # >= 0.60
            'low': 0         # < 0.60
        },
        'unified_entities': 0,
        'text_length_stats': {
            'min': float('inf'),
            'max': 0,
            'avg': 0
        }
    }
    
    text_lengths = []
    
    for entity in entities:
        # Contadores por categoría
        analysis['by_label'][entity.label] += 1
        analysis['by_model'][entity.model] += 1
        analysis['by_document'][entity.doc_id] += 1
        
        # Distribución de confianza
        if entity.confidence >= 0.95:
            analysis['confidence_distribution']['very_high'] += 1
        elif entity.confidence >= 0.80:
            analysis['confidence_distribution']['high'] += 1
        elif entity.confidence >= 0.60:
            analysis['confidence_distribution']['medium'] += 1
        else:
            analysis['confidence_distribution']['low'] += 1
        
        # Entidades unificadas
        if entity.unified:
            analysis['unified_entities'] += 1
        
        # Estadísticas de longitud de texto
        text_len = len(entity.text)
        text_lengths.append(text_len)
        analysis['text_length_stats']['min'] = min(
            analysis['text_length_stats']['min'], text_len
        )
        analysis['text_length_stats']['max'] = max(
            analysis['text_length_stats']['max'], text_len
        )
    
    # Calcular promedio de longitud
    if text_lengths:
        analysis['text_length_stats']['avg'] = sum(text_lengths) / len(text_lengths)
    else:
        analysis['text_length_stats']['min'] = 0
    
    # Convertir defaultdicts a dicts normales
    analysis['by_label'] = dict(analysis['by_label'])
    analysis['by_model'] = dict(analysis['by_model'])
    analysis['by_document'] = dict(analysis['by_document'])
    
    return analysis


def print_analysis_summary(analysis: Dict):
    """
    Imprime resumen del análisis de forma legible.
    
    Args:
        analysis: Diccionario de análisis generado por analyze_entities
    """
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE ENTIDADES PROCESADAS")
    print(f"{'='*70}")
    
    print(f"\n📊 RESUMEN GENERAL")
    print(f"  Total de entidades: {analysis['total_entities']}")
    print(f"  Entidades unificadas: {analysis['unified_entities']}")
    print(f"  Documentos únicos: {len(analysis['by_document'])}")
    
    print(f"\n🏷️  DISTRIBUCIÓN POR ETIQUETA")
    for label, count in sorted(analysis['by_label'].items(), key=lambda x: -x[1]):
        pct = (count / analysis['total_entities'] * 100) if analysis['total_entities'] > 0 else 0
        print(f"  {label:40s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n🤖 DISTRIBUCIÓN POR MODELO")
    for model, count in analysis['by_model'].items():
        pct = (count / analysis['total_entities'] * 100) if analysis['total_entities'] > 0 else 0
        print(f"  {model:15s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n📈 DISTRIBUCIÓN DE CONFIANZA")
    conf_dist = analysis['confidence_distribution']
    total = analysis['total_entities']
    if total > 0:
        print(f"  Muy alta (≥0.95): {conf_dist['very_high']:5d} ({conf_dist['very_high']/total*100:5.1f}%)")
        print(f"  Alta (≥0.80):     {conf_dist['high']:5d} ({conf_dist['high']/total*100:5.1f}%)")
        print(f"  Media (≥0.60):    {conf_dist['medium']:5d} ({conf_dist['medium']/total*100:5.1f}%)")
        print(f"  Baja (<0.60):     {conf_dist['low']:5d} ({conf_dist['low']/total*100:5.1f}%)")
    
    print(f"\n📏 ESTADÍSTICAS DE LONGITUD DE TEXTO")
    stats = analysis['text_length_stats']
    print(f"  Mínima:  {stats['min']} caracteres")
    print(f"  Máxima:  {stats['max']} caracteres")
    print(f"  Promedio: {stats['avg']:.1f} caracteres")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# FUNCIONES DE EXPORTACIÓN
# ============================================================================

def save_processed_entities(entities: List[Entity], 
                           output_path: str,
                           stats: ProcessingStats,
                           analysis: Dict):
    """
    Guarda las entidades procesadas en formato JSON.
    
    Args:
        entities: Lista de entidades procesadas
        output_path: Ruta del archivo de salida
        stats: Estadísticas de procesamiento
        analysis: Análisis de las entidades
    """
    debug_print(f"Guardando entidades procesadas en: {output_path}", "INFO")
    
    output_data = {
        'metadata': {
            'generated_at': datetime.datetime.now().isoformat(),
            'total_entities': len(entities),
            'processing_stats': stats.to_dict(),
            'analysis': analysis
        },
        'entities': [entity.to_dict() for entity in entities]
    }
    
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        debug_print(f"✓ Entidades guardadas exitosamente", "INFO")
        
    except Exception as e:
        debug_print(f"Error guardando entidades: {e}", "ERROR")
        raise


# ============================================================================
# FUNCIÓN PRINCIPAL DE PREPROCESAMIENTO
# ============================================================================

def preprocess_entities(csv_path: str, 
                       output_path: str,
                       max_gap: int = 5,
                       same_label_only: bool = True) -> Tuple[List[Entity], Dict]:
    """
    Función principal de preprocesamiento del Paso 1.
    
    Ejecuta todo el pipeline de preprocesamiento:
    1. Carga CSV sin filtrar
    2. Unifica entidades fragmentadas
    3. Genera análisis
    4. Guarda resultados
    
    Args:
        csv_path: Ruta al CSV con detecciones
        output_path: Ruta para guardar entidades procesadas
        max_gap: Gap máximo para unificar entidades
        same_label_only: Si True, solo unifica entidades con misma etiqueta
        
    Returns:
        Tuple con:
        - Lista de entidades procesadas
        - Diccionario con análisis completo
    """
    debug_print("="*70, "INFO")
    debug_print("PASO 1: PREPROCESAMIENTO DE ENTIDADES", "INFO")
    debug_print("="*70, "INFO")
    
    # 1. Cargar CSV
    entities, stats = load_csv_detections(csv_path)
    
    # 2. Unificar entidades fragmentadas
    unified_entities, merged_count = unify_fragmented_entities(
        entities, max_gap, same_label_only
    )
    stats.total_unified_entities = len(unified_entities)
    stats.entities_merged = merged_count
    
    # 3. Generar análisis
    analysis = analyze_entities(unified_entities)
    
    # 4. Imprimir resumen
    print_analysis_summary(analysis)
    
    # 5. Guardar resultados
    save_processed_entities(unified_entities, output_path, stats, analysis)
    
    debug_print("="*70, "INFO")
    debug_print("✓ PREPROCESAMIENTO COMPLETADO", "INFO")
    debug_print("="*70, "INFO")
    
    return unified_entities, analysis


# ============================================================================
# CLI Y MAIN
# ============================================================================

def main():
    """Función principal del script"""
    parser = argparse.ArgumentParser(
        description='Pipeline de evaluación con juez LLM - Preprocesamiento',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Preprocesar detecciones básico
  python llm_judge_pipeline.py preprocess --csv detecciones_detalladas.csv --output entidades_procesadas.json
  
  # Con parámetros personalizados de unificación
  python llm_judge_pipeline.py preprocess --csv detecciones.csv --output salida.json --max-gap 10 --no-same-label

NOTA: Para cálculo de métricas, usar el script separado compute_llm_metrics.py
  
IMPORTANTE: 
- El preprocesamiento NO filtra ninguna entidad.
- La columna manual_correction se preserva pero NO se envía al LLM.
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Subcomando: preprocess
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Preprocesar CSV de detecciones'
    )
    preprocess_parser.add_argument(
        '--csv',
        required=True,
        help='Ruta al archivo CSV con detecciones'
    )
    preprocess_parser.add_argument(
        '--output',
        required=True,
        help='Ruta de salida para entidades procesadas (JSON)'
    )
    preprocess_parser.add_argument(
        '--max-gap',
        type=int,
        default=5,
        help='Gap máximo de caracteres para unificar entidades (default: 5)'
    )
    preprocess_parser.add_argument(
        '--no-same-label',
        action='store_true',
        help='Permitir unificación de entidades con diferentes etiquetas'
    )
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        try:
            preprocess_entities(
                csv_path=args.csv,
                output_path=args.output,
                max_gap=args.max_gap,
                same_label_only=not args.no_same_label
            )
        except Exception as e:
            debug_print(f"Error en preprocesamiento: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
