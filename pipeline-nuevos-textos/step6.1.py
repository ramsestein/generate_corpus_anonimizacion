#!/usr/bin/env python3
"""
PASO 6 DEL PIPELINE: Verificación de anonimización
Verifica documentos anonimizados detectando tokens de anonimización JJJ.
Opcionalmente puede usar modelos NER (MEDDOCAN y CARMEN) con --use-models.
"""

import os
import json
import csv
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
import torch

ANON_TOKEN = "JJJ"
# Valores de chunk y overlap definidos en el código por defecto.
# Cámbialos aquí si quieres otros valores.
CHUNK_TOKEN_SIZE = 384
CHUNK_TOKEN_OVERLAP = 32

# ============================================================================
# FILTRO POST-PROCESADO DE ETIQUETAS NO-PII
# ============================================================================
# Conjunto de etiquetas que NO se consideran PII y deben ser completamente
# ignoradas después de la extracción del modelo.
# 
# Estas etiquetas se descartan porque:
# - FAMILIARES_SUJETO_ASISTENCIA: Términos genéricos como "familia", "familiar"
#   que no identifican personas específicas
# - PROFESION: Ocupaciones genéricas que no son identificadores personales
# - OTROS_SUJETO_ASISTENCIA: Categoría demasiado genérica y ambigua
#
# Las entidades con estas etiquetas NO contarán como TP, FP ni aparecerán
# en métricas o resultados finales.
# ============================================================================
NON_PII_LABELS = {
    "FAMILIARES_SUJETO_ASISTENCIA",
    "PROFESION",
    "OTROS_SUJETO_ASISTENCIA",
}

def filter_non_pii_entities(entities: List[Dict]) -> List[Dict]:
    """
    Filtra etiquetas no-PII de la lista de entidades detectadas.
    
    Este filtro post-procesado descarta completamente las entidades cuyas
    etiquetas están en NON_PII_LABELS, independientemente de su confianza.
    
    Args:
        entities (List[Dict]): Lista de entidades detectadas por el modelo NER
        
    Returns:
        List[Dict]: Lista filtrada conteniendo solo entidades que representan PII real
        
    Ejemplo:
        >>> entities = [
        ...     {"entity_group": "NOMBRE_SUJETO_ASISTENCIA", "word": "Juan", "score": 0.99},
        ...     {"entity_group": "FAMILIARES_SUJETO_ASISTENCIA", "word": "familia", "score": 0.99},
        ...     {"entity_group": "PROFESION", "word": "médico", "score": 0.95}
        ... ]
        >>> filter_non_pii_entities(entities)
        [{"entity_group": "NOMBRE_SUJETO_ASISTENCIA", "word": "Juan", "score": 0.99}]
    """
    if not entities:
        return []
    
    # Filtrar entidades cuya etiqueta NO esté en el conjunto de no-PII
    filtered_entities = []
    discarded_count = 0
    
    for entity in entities:
        # Intentar obtener la etiqueta desde 'entity_group' o 'label'
        label = entity.get('entity_group') or entity.get('label')
        
        if label and label in NON_PII_LABELS:
            # Esta entidad NO es PII → descartarla completamente
            discarded_count += 1
            continue
        
        # Esta entidad SÍ es PII → mantenerla
        filtered_entities.append(entity)
    
    # Log informativo si se descartaron entidades
    if discarded_count > 0:
        debug_print(
            f"      Filtro no-PII: descartadas {discarded_count} entidades "
            f"(quedaron {len(filtered_entities)} entidades PII reales)",
            "DEBUG"
        )
    
    return filtered_entities

def debug_print(message: str, level: str = "INFO"):
    """Función para imprimir mensajes de debug con timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def setup_models(use_carmen: bool = True):
    """
    Configura y carga ambos modelos BSC.
    
    Returns:
        Tuple: (pipeline_meddocan, pipeline_carmen)
    """
    debug_print("Cargando modelos BSC para verificación...", "INFO")
    
    # Configurar device
    device = 0 if torch and torch.cuda.is_available() else -1
    debug_print(f"Device set to use {'cuda' if torch and torch.cuda.is_available() else 'cpu'}", "INFO")
    
    # Modelo MEDDOCAN
    debug_print("  - Cargando bsc-bio-ehr-es-meddocan...", "DEBUG")
    meddocan_model_path = "models/bsc-bio-ehr-es-meddocan"
    
    try:
        meddocan_tokenizer = AutoTokenizer.from_pretrained(meddocan_model_path)
        
        # Cargar modelo con configuración específica para evitar meta tensors
        meddocan_model = AutoModelForTokenClassification.from_pretrained(
            meddocan_model_path,
            dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,  # No usar device_map automático
            trust_remote_code=False
        )
        
        # Verificar que el modelo se cargó correctamente
        if hasattr(meddocan_model, 'config'):
            debug_print(f"      Modelo MEDDOCAN cargado: {meddocan_model.config.model_type}", "DEBUG")
        
        # Crear pipeline sin especificar device en el modelo
        pipeline_meddocan = hf_pipeline(
            "ner",
            model=meddocan_model,
            tokenizer=meddocan_tokenizer,
            aggregation_strategy="simple",
            device=-1  # Forzar CPU para evitar problemas con meta tensors
        )
    except Exception as e:
        debug_print(f"Error cargando modelo MEDDOCAN: {e}", "ERROR")
        raise e
    
    pipeline_carmen = None
    if use_carmen:
        # Modelo CARMEN
        debug_print("  - Cargando bsc-bio-ehr-es-carmen-anon...", "DEBUG")
        carmen_model_path = "models/bsc-bio-ehr-es-carmen-anon"

        try:
            carmen_tokenizer = AutoTokenizer.from_pretrained(carmen_model_path)

            # Cargar modelo con configuración específica para evitar meta tensors
            carmen_model = AutoModelForTokenClassification.from_pretrained(
                carmen_model_path,
                dtype=torch.float32,
                low_cpu_mem_usage=False,
                device_map=None,  # No usar device_map automático
                trust_remote_code=False
            )

            # Verificar que el modelo se cargó correctamente
            if hasattr(carmen_model, 'config'):
                debug_print(f"      Modelo CARMEN cargado: {carmen_model.config.model_type}", "DEBUG")

            # Crear pipeline sin especificar device en el modelo
            pipeline_carmen = hf_pipeline(
                "ner",
                model=carmen_model,
                tokenizer=carmen_tokenizer,
                aggregation_strategy="simple",
                device=-1  # Forzar CPU para evitar problemas con meta tensors
            )
        except Exception as e:
            debug_print(f"Error cargando modelo CARMEN: {e}", "ERROR")
            raise e

    debug_print("Modelos cargados exitosamente!", "INFO")
    return pipeline_meddocan, pipeline_carmen

def extract_entities_with_model(text: str, pipeline_model, model_name: str, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Extrae entidades de un texto usando un modelo específico.
    
    Args:
        text (str): Texto a procesar
        pipeline_model: Pipeline del modelo
        model_name (str): Nombre del modelo para logging
        confidence_threshold (float): Umbral mínimo de confianza
        
    Returns:
        List[Dict]: Lista de entidades detectadas
    """
    debug_print(f"    Analizando con {model_name}...", "DEBUG")
    
    try:
        # Cache text length para evitar llamadas repetidas
        text_len = len(text)
        
        # Obtener tokenizer del pipeline
        tokenizer = pipeline_model.tokenizer
        model_max_length = getattr(tokenizer, 'model_max_length', 512)
        
        # Usar valores definidos en el código (CHUNK_TOKEN_SIZE, CHUNK_TOKEN_OVERLAP)
        token_chunk_size = min(model_max_length - 20, CHUNK_TOKEN_SIZE)
        token_overlap = min(CHUNK_TOKEN_OVERLAP, token_chunk_size - 1)

        debug_print(f"      Usando CHUNK_TOKEN_SIZE={token_chunk_size}, CHUNK_TOKEN_OVERLAP={token_overlap}", "DEBUG")

        # Si el overlap configurado tuvo que reducirse por el tamaño de chunk, avisar
        if token_overlap != CHUNK_TOKEN_OVERLAP:
            debug_print(f"      AVISO: overlap configurado {CHUNK_TOKEN_OVERLAP} ajustado a {token_overlap} debido al tamaño de chunk ({token_chunk_size})", "WARN")
        
        # Tokenizar todo el texto para obtener offsets
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False
        )
        
        tokens = encoding['input_ids']
        offsets = encoding['offset_mapping']
        num_tokens = len(tokens)
        
        # Crear chunks basados en tokens
        chunks = []
        chunk_global_offsets = []
        
        if num_tokens <= token_chunk_size:
            # Texto corto, procesar completo
            chunks.append(text)
            chunk_global_offsets.append(0)
        else:
            # Dividir en chunks con overlap
            step = token_chunk_size - token_overlap
            for i in range(0, num_tokens, step):
                chunk_tokens = tokens[i:i + token_chunk_size]

                if not chunk_tokens:
                    break

                # Obtener offsets del primer y último token del chunk
                start_char = offsets[i][0]
                end_idx = min(i + len(chunk_tokens) - 1, len(offsets) - 1)
                end_char = offsets[end_idx][1]


                # Extraer texto del chunk (con límites ajustados)
                chunk_text = text[start_char:end_char]
                chunks.append(chunk_text)
                chunk_global_offsets.append(start_char)
        
        num_chunks = len(chunks)
        debug_print(f"      Texto dividido en {num_chunks} chunks (token-aligned)", "DEBUG")
        
      
        
        all_entities = []
        
        for i, (chunk, global_offset) in enumerate(zip(chunks, chunk_global_offsets)):
            debug_print(f"      Procesando chunk {i+1}/{num_chunks} (offset global: {global_offset})...", "DEBUG")
           
            # Procesar chunk: 
           # Extraer entidades del chunk
            try:
                entities = pipeline_model(chunk)
            except Exception as chunk_error:
                debug_print(f"        ERROR procesando chunk con {model_name}: {chunk_error}", "ERROR")
                continue
            
            # Verificar que entities es una lista válida
            if not isinstance(entities, list):
                debug_print(f"        WARNING: {model_name} devolvió resultado inválido: {type(entities)}", "WARN")
                continue
            
            debug_print(f"        Chunk devolvió {len(entities)} detecciones brutas", "DEBUG")
            
            # Ajustar posiciones y filtrar
            for entity in entities:
                try:
                    confidence = float(entity.get('score', 0.0))
                    
                    # Filtrar por umbral de confianza primero (early exit)
                    if confidence < confidence_threshold:
                        continue
                    
                    word = entity.get('word', '')
                    # Ajustar offsets del chunk al texto global
                    start_pos = int(entity.get('start', 0)) + global_offset
                    end_pos = int(entity.get('end', 0)) + global_offset
                except (ValueError, TypeError) as parse_error:
                    debug_print(f"        ERROR parseando entidad: {parse_error}", "ERROR")
                    continue
                
                # Validar bounds
                if not (0 <= start_pos < end_pos <= text_len):
                    continue
                
                # Extraer texto real para verificar
                try:
                    actual_text = text[start_pos:end_pos]
                except Exception:
                    actual_text = word
                
                # Verificar si es solo caracteres J (token de anonimización)
                if is_only_j_characters(actual_text):
                    # Detectó un token de anonimización - esto es esperado, NO reportar
                    debug_print(f"        Filtrada detección JJJ (esperada): '{actual_text}' (confianza: {confidence:.2f})", "DEBUG")
                    continue
                
                # Detección de algo que NO es JJJ - esto es sospechoso
                entity['start'] = start_pos
                entity['end'] = end_pos
                entity['score'] = confidence
                entity['chunk_id'] = i
                entity['model'] = model_name
                entity['actual_text'] = actual_text
                entity['is_j_only'] = False
                all_entities.append(entity)
                debug_print(f"        Detección sospechosa: '{actual_text}' (confianza: {confidence:.2f})", "WARN")
        
        # ========================================================================
        # APLICAR FILTRO POST-PROCESADO DE ETIQUETAS NO-PII
        # ========================================================================
        # Después de extraer todas las entidades del modelo, eliminamos aquellas
        # cuyas etiquetas no representan PII real (ej: FAMILIARES_SUJETO_ASISTENCIA,
        # PROFESION, OTROS_SUJETO_ASISTENCIA).
        #
        # Este filtro se aplica DESPUÉS de todos los demás filtros (confianza, JJJ, etc.)
        # para garantizar que estas etiquetas no aparezcan en ningún resultado final,
        # métrica, ni análisis.
        # ========================================================================
        entities_before_filter = len(all_entities)
        all_entities = filter_non_pii_entities(all_entities)
        entities_after_filter = len(all_entities)
        
        if entities_before_filter != entities_after_filter:
            debug_print(
                f"      {model_name}: {entities_before_filter} entidades brutas → "
                f"{entities_after_filter} entidades PII reales (descartadas {entities_before_filter - entities_after_filter} no-PII)",
                "INFO"
            )
        
        debug_print(f"      {model_name}: {len(all_entities)} entidades detectadas (confianza >= {confidence_threshold})", "INFO")
        return all_entities
        
    except Exception as e:
        debug_print(f"      ERROR al procesar con {model_name}: {e}", "ERROR")
        import traceback
        debug_print(f"      Traceback: {traceback.format_exc()}", "ERROR")
        return []



def is_only_j_characters(text: str) -> bool:
    """
    Verifica si un texto contiene solo caracteres 'J' y espacios.

    Args:
        text (str): Texto a verificar
        
    Returns:
        bool: True si solo contiene X y espacios
    """
    if not text:
        return False
       # Remover espacios y verificar si solo quedan J's
    cleaned_text = text.strip().replace(' ', '')
    return len(cleaned_text) > 0 and all(c == 'J' for c in cleaned_text)

def analyze_detected_entities(entities: List[Dict], text: str) -> Dict:
    """
    Analiza las entidades detectadas para clasificarlas.
    NO filtra, solo clasifica las detecciones según sean JJJ o no.
    
    Args:
        entities (List[Dict]): Entidades detectadas
        text (str): Texto original
        
    Returns:
        Dict: Análisis de las entidades
    """
    # Clasificar entidades (NO filtrar)
    non_j_entities = []
    j_only_entities = []
    
    for entity in entities:
        # Usar el campo is_j_only si ya fue calculado en extract_entities_with_model
        is_j = entity.get('is_j_only', False)
        if not is_j:
            # Fallback: calcular si no está presente
            word = entity.get('word', '')
            start = entity.get('start', 0)
            end = entity.get('end', 0)

        # Extraer el texto real de la posición

            try:
                actual_text = text[start:end] if start < len(text) and end <= len(text) else word
            except:
                actual_text = word
            is_j = is_only_j_characters(actual_text) or is_only_j_characters(word)
        
        if is_j:
            j_only_entities.append(entity)
        else:
            non_j_entities.append(entity)
    
    debug_print(f"      Clasificación: {len(j_only_entities)} detecciones JJJ, {len(non_j_entities)} otras", "DEBUG")
    
    analysis = {
        'total_entities': len(entities),  # Total sin filtrar
        'j_only_count': len(j_only_entities),
        'non_j_count': len(non_j_entities),
        'entities_by_label': {},
        'entities_by_confidence': {'high': 0, 'medium': 0, 'low': 0},
        'suspicious_entities': [],  # Solo entidades no-JJJ (potencialmente problemáticas)
        'anon_token_detections': [],  # Detecciones sobre el token de anonimización (JJJ)
        'other_detections': [],    # Otras detecciones (no JJJ)
        'j_only_entities': j_only_entities,  # Todas las detecciones JJJ
        'filtered_x_entities': len(j_only_entities)  # Compat: cuántas son JJJ
    }
    
    
    for entity in entities:
        label = entity.get('entity_group', entity.get('label', 'UNKNOWN'))
        confidence = entity.get('score', 0.0)
        word = entity.get('word', '')
        start = entity.get('start', 0)
        end = entity.get('end', 0)
        is_j = entity.get('is_j_only', False)
        
        # Contar por etiqueta
        if label not in analysis['entities_by_label']:
            analysis['entities_by_label'][label] = 0
        analysis['entities_by_label'][label] += 1
        
        # Contar por confianza
        if confidence >= 0.8:
            analysis['entities_by_confidence']['high'] += 1
        elif confidence >= 0.6:
            analysis['entities_by_confidence']['medium'] += 1
        else:
            analysis['entities_by_confidence']['low'] += 1
        
        # Extraer el texto real de la posición
        try:
            actual_text = entity.get('actual_text', text[start:end] if start < len(text) and end <= len(text) else word)
        except:
            actual_text = word
        
        entity_info = {
            'label': label,
            'word': word,
            'actual_text': actual_text,
            'confidence': confidence,
            'start': start,
            'end': end,
            'model': entity.get('model', 'unknown'),
            'is_j_only': is_j
        }
        
        # Clasificar la detección
        if is_j or (ANON_TOKEN and (ANON_TOKEN in actual_text or ANON_TOKEN in word)):
            analysis['anon_token_detections'].append(entity_info)
        else:
            analysis['other_detections'].append(entity_info)
            # Si no es JJJ, es sospechoso (no debería haber entidades reales en texto anonimizado)
            analysis['suspicious_entities'].append(entity_info)
    
    return analysis


def count_anon_markers(text: str) -> int:
    """
    Cuenta las apariciones del token de anonimización (ANON_TOKEN).
    """
    if not text:
        return 0

    count = 0
    if ANON_TOKEN:
        count += len(re.findall(re.escape(ANON_TOKEN), text))

    return count

def process_documents_batch(docs_batch: List[str], input_dir: str, output_dir: str, 
                           confidence_threshold: float = 0.7, use_carmen: bool = True) -> List[Dict]:
    """
    Procesa un lote de documentos con los modelos BSC en paralelo.
    
    Args:
        docs_batch: Lista de IDs de documentos
        input_dir: Directorio de entrada
        output_dir: Directorio de salida
        confidence_threshold: Umbral de confianza
        
    Returns:
        List[Dict]: Resultados del lote
    """
    # Cargar modelos en este worker
    pipeline_meddocan, pipeline_carmen = setup_models(use_carmen=use_carmen)
    
    results = []
    for doc_id in docs_batch:
        result = process_single_document(doc_id, input_dir, pipeline_meddocan, pipeline_carmen, 
                                       output_dir, confidence_threshold)
        results.append(result)
    return results

def process_single_document(doc_id: str, input_dir: str, pipeline_meddocan, pipeline_carmen, 
                           output_dir: str, confidence_threshold: float = 0.7) -> Dict:
    """
    Procesa un documento anonimizado con ambos modelos BSC.
    
    Args:
        doc_id (str): ID del documento
        input_dir (str): Directorio con documentos anonimizados
        pipeline_meddocan: Pipeline del modelo MEDDOCAN
        pipeline_carmen: Pipeline del modelo CARMEN
        output_dir (str): Directorio de salida
        confidence_threshold (float): Umbral de confianza
        
    Returns:
        Dict: Resultado del análisis
    """
    debug_print(f"Verificando anonimización: {doc_id}", "INFO")
    
    # Leer documento anonimizado - intentar con diferentes extensiones
    doc_file = os.path.join(input_dir, f"{doc_id}.txt.txt")
    if not os.path.exists(doc_file):
        doc_file = os.path.join(input_dir, f"{doc_id}.txt")
    
    if not os.path.exists(doc_file):
        debug_print(f"No se encontró el documento: {doc_id}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": "Documento no encontrado"
        }
    
    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            anonymized_text = f.read().strip()
    except Exception as e:
        debug_print(f"Error al leer documento {doc_id}: {e}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": f"Error al leer documento: {e}"
        }
    
    debug_print(f"  Texto anonimizado: {len(anonymized_text)} caracteres", "DEBUG")
    debug_print(f"  Conteo de marcas de anonimización ({ANON_TOKEN}): {count_anon_markers(anonymized_text)}", "DEBUG")
    
    # Analizar con MEDDOCAN
    meddocan_entities = extract_entities_with_model(
        anonymized_text, pipeline_meddocan, "MEDDOCAN", confidence_threshold
    )

    # Analizar con CARMEN (solo si pipeline está presente)
    if pipeline_carmen is not None:
        carmen_entities = extract_entities_with_model(
            anonymized_text, pipeline_carmen, "CARMEN", confidence_threshold
        )
    else:
        carmen_entities = []
    
    # Combinar todas las entidades
    all_entities = meddocan_entities + carmen_entities
    
    # Analizar las entidades detectadas
    meddocan_analysis = analyze_detected_entities(meddocan_entities, anonymized_text)
    carmen_analysis = analyze_detected_entities(carmen_entities, anonymized_text)
    combined_analysis = analyze_detected_entities(all_entities, anonymized_text)
    
    # Determinar si la anonimización fue exitosa
    anonymization_success = len(combined_analysis['suspicious_entities']) == 0
    
    # Verificar si hay entidades con confianza muy alta (> 0.99)
    high_confidence_entities = []
    for entity in all_entities:
        confidence = entity.get('score', entity.get('confidence', 0.0))
        if confidence > 0.99:
            high_confidence_entities.append({
                "label": entity.get('entity_group', entity.get('label', 'UNKNOWN')),
                "text": entity.get('word', ''),
                "confidence": confidence,
                "model": entity.get('model', 'UNKNOWN')
            })
    
    should_delete_document = len(high_confidence_entities) > 0
    
    # Crear resultado
    result = {
        "document_id": doc_id,
        "success": True,
        "anonymized_text": anonymized_text,
        "text_length": len(anonymized_text),
        "anonymized_count": count_anon_markers(anonymized_text),
        "confidence_threshold": confidence_threshold,
        "meddocan_results": {
            "entities_detected": len(meddocan_entities),
            "entities": meddocan_entities,
            "analysis": meddocan_analysis
        },
        "carmen_results": {
            "entities_detected": len(carmen_entities),
            "entities": carmen_entities,
            "analysis": carmen_analysis
        },
        "combined_analysis": combined_analysis,
        "anonymization_verification": {
            "success": anonymization_success,
            "total_detections": len(all_entities),
            "suspicious_detections": len(combined_analysis['suspicious_entities']),
            "anon_token_detections": len(combined_analysis.get('anon_token_detections', [])),
            "other_detections": len(combined_analysis['other_detections'])
        },
        "high_confidence_check": {
            "should_delete": should_delete_document,
            "high_confidence_entities": high_confidence_entities,
            "count": len(high_confidence_entities)
        }
    }
    
    # Guardar resultado detallado
    result_file = os.path.join(output_dir, f"{doc_id}_verification_result.json")
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        debug_print(f"  Resultado guardado: {result_file}", "DEBUG")
    except Exception as e:
        debug_print(f"  ERROR al guardar resultado: {e}", "WARN")
    
    # Mostrar resumen
    debug_print(f"  RESULTADO VERIFICACIÓN:", "INFO")
    debug_print(f"    - Anonimización exitosa: {'SÍ' if anonymization_success else 'NO'}", 
                "INFO" if anonymization_success else "WARN")
    debug_print(f"    - MEDDOCAN detectó: {len(meddocan_entities)} entidades", "INFO")
    debug_print(f"    - CARMEN detectó: {len(carmen_entities)} entidades", "INFO")
    debug_print(f"    - Detecciones sospechosas: {len(combined_analysis['suspicious_entities'])}", 
                "INFO" if len(combined_analysis['suspicious_entities']) == 0 else "WARN")
    
    # Mostrar detecciones sospechosas si las hay
    if combined_analysis['suspicious_entities']:
        debug_print(f"    DETECCIONES SOSPECHOSAS:", "WARN")
        for entity in combined_analysis['suspicious_entities']:
            debug_print(f"      - {entity['label']}: '{entity['actual_text']}' (confianza: {entity['confidence']:.2f})", "WARN")
    
    # Mostrar si hay entidades con confianza muy alta
    if high_confidence_entities:
        debug_print(f"    [!] ALTA CONFIANZA (>0.99): {len(high_confidence_entities)} entidades - DOCUMENTO MARCADO PARA ELIMINACION", "ERROR")
        for entity in high_confidence_entities:
            debug_print(f"      - {entity['label']}: '{entity['text'].strip()}' (confianza: {entity['confidence']:.4f}) [{entity['model']}]", "ERROR")
    
    return result

def create_verification_summary(results: List[Dict], output_dir: str):
    """
    Crea un reporte resumen de la verificación de anonimización.
    
    Args:
        results (List[Dict]): Lista de resultados de verificación
        output_dir (str): Directorio de salida
    """
    debug_print("Creando reporte resumen de verificación...", "INFO")
    
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    # Calcular estadísticas globales
    total_meddocan_detections = sum([r.get("meddocan_results", {}).get("entities_detected", 0) for r in successful_results])
    total_carmen_detections = sum([r.get("carmen_results", {}).get("entities_detected", 0) for r in successful_results])
    total_suspicious = sum([r.get("anonymization_verification", {}).get("suspicious_detections", 0) for r in successful_results])
    total_anon_token_detections = sum([r.get("anonymization_verification", {}).get("anon_token_detections", 0) for r in successful_results])
    
    perfect_anonymizations = len([r for r in successful_results 
                                 if r.get("anonymization_verification", {}).get("success", False)])
    
    # Calcular estadísticas de alta confianza
    high_confidence_docs = [r for r in successful_results 
                           if r.get("high_confidence_check", {}).get("should_delete", False)]
    total_high_confidence_entities = sum([r.get("high_confidence_check", {}).get("count", 0) 
                                         for r in successful_results])
    
    summary = {
        "pipeline_step": 4,
        "process": "anonymization_verification_with_bsc_models",
        "total_documents": len(results),
        "successful_verifications": len(successful_results),
        "failed_verifications": len(failed_results),
        "perfect_anonymizations": perfect_anonymizations,
        "imperfect_anonymizations": len(successful_results) - perfect_anonymizations,
        "global_statistics": {
            "total_meddocan_detections": total_meddocan_detections,
            "total_carmen_detections": total_carmen_detections,
            "total_suspicious_detections": total_suspicious,
            "total_anon_token_detections": total_anon_token_detections,
            "anonymization_success_rate": (perfect_anonymizations / len(successful_results) * 100) if successful_results else 0
        },
        "high_confidence_statistics": {
            "documents_to_delete": len(high_confidence_docs),
            "total_high_confidence_entities": total_high_confidence_entities,
            "document_ids_to_delete": [r["document_id"] for r in high_confidence_docs]
        },
        "documents": results
    }
    
    summary_file = os.path.join(output_dir, "step4_verification_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    debug_print(f"Reporte guardado: {summary_file}", "INFO")
    
    # Mostrar resumen en consola
    print(f"\n{'='*60}")
    print(f"PASO 4 - VERIFICACIÓN DE ANONIMIZACIÓN COMPLETADA")
    print(f"{'='*60}")
    print(f"Documentos verificados: {summary['total_documents']}")
    print(f"Verificaciones exitosas: {summary['successful_verifications']}")
    print(f"Verificaciones fallidas: {summary['failed_verifications']}")
    print(f"Anonimizaciones perfectas: {summary['perfect_anonymizations']}")
    print(f"Anonimizaciones imperfectas: {summary['imperfect_anonymizations']}")
    print(f"Tasa de éxito de anonimización: {summary['global_statistics']['anonymization_success_rate']:.1f}%")
    print(f"")
    print(f"DETECCIONES DE LOS MODELOS:")
    print(f"  - MEDDOCAN detectó: {summary['global_statistics']['total_meddocan_detections']} entidades")
    print(f"  - CARMEN detectó: {summary['global_statistics']['total_carmen_detections']} entidades")
    print(f"  - Detecciones sospechosas: {summary['global_statistics']['total_suspicious_detections']}")
    
    if summary['imperfect_anonymizations'] > 0:
        print(f"\nDOCUMENTOS CON DETECCIONES SOSPECHOSAS:")
        for result in successful_results:
            if not result.get("anonymization_verification", {}).get("success", False):
                suspicious_count = result.get("anonymization_verification", {}).get("suspicious_detections", 0)
                print(f"  - {result['document_id']}: {suspicious_count} detecciones sospechosas")
    
    # Mostrar documentos marcados para eliminación
    if summary['high_confidence_statistics']['documents_to_delete'] > 0:
        print(f"\n[!] DOCUMENTOS MARCADOS PARA ELIMINACION (Confianza > 0.99):")
        print(f"  Total: {summary['high_confidence_statistics']['documents_to_delete']} documentos")
        print(f"  Entidades de alta confianza detectadas: {summary['high_confidence_statistics']['total_high_confidence_entities']}")
        for doc_id in summary['high_confidence_statistics']['document_ids_to_delete']:
            print(f"    - {doc_id}")
    
    print(f"\nDirectorio de salida: {output_dir}")
    print(f"{'='*60}")

def create_detailed_detections_report(results: List[Dict], output_dir: str):
    """
    Crea un listado detallado CSV con todas las detecciones sospechosas.
    
    Formato: doc_id, etiqueta, modelo_detector, texto_detectado, confianza
    
    Args:
        results (List[Dict]): Lista de resultados de verificación
        output_dir (str): Directorio de salida
    """
    debug_print("Generando listado detallado de detecciones...", "INFO")
    
    detections_list = []
    
    for result in results:
        if not result.get("success", False):
            continue
            
        doc_id = result.get("document_id", "unknown")
        
        # Procesar detecciones de MEDDOCAN
        meddocan_entities = result.get("meddocan_results", {}).get("entities", [])
        for entity in meddocan_entities:
            # Los campos pueden venir como entity_group/score o label/confidence
            etiqueta = entity.get("entity_group", entity.get("label", "UNKNOWN"))
            confianza = entity.get("score", entity.get("confidence", 0.0))
            texto = entity.get("word", "").strip()
            
            detections_list.append({
                "doc_id": doc_id,
                "etiqueta": etiqueta,
                "modelo_detector": "MEDDOCAN",
                "texto_detectado": texto,
                "confianza": f"{confianza:.4f}",
                "posicion_inicio": entity.get("start", ""),
                "posicion_fin": entity.get("end", "")
            })
        
        # Procesar detecciones de CARMEN
        carmen_entities = result.get("carmen_results", {}).get("entities", [])
        for entity in carmen_entities:
            # Los campos pueden venir como entity_group/score o label/confidence
            etiqueta = entity.get("entity_group", entity.get("label", "UNKNOWN"))
            confianza = entity.get("score", entity.get("confidence", 0.0))
            texto = entity.get("word", "").strip()
            
            detections_list.append({
                "doc_id": doc_id,
                "etiqueta": etiqueta,
                "modelo_detector": "CARMEN",
                "texto_detectado": texto,
                "confianza": f"{confianza:.4f}",
                "posicion_inicio": entity.get("start", ""),
                "posicion_fin": entity.get("end", "")
            })
    
    # Ordenar por doc_id y luego por modelo
    detections_list.sort(key=lambda x: (x["doc_id"], x["modelo_detector"]))
    
    # Guardar como CSV
    csv_file = os.path.join(output_dir, "detecciones_detalladas.csv")
    try:
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            if detections_list:
                fieldnames = ["doc_id", "etiqueta", "modelo_detector", "texto_detectado", 
                            "confianza", "posicion_inicio", "posicion_fin"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(detections_list)
        
        debug_print(f"Listado detallado guardado: {csv_file}", "INFO")
        debug_print(f"  Total de detecciones: {len(detections_list)}", "INFO")
    except Exception as e:
        debug_print(f"ERROR al guardar listado detallado: {e}", "ERROR")
    
    # También guardar como JSON para mayor flexibilidad
    json_file = os.path.join(output_dir, "detecciones_detalladas.json")
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_detecciones": len(detections_list),
                "detecciones": detections_list
            }, f, indent=2, ensure_ascii=False)
        debug_print(f"Listado detallado JSON guardado: {json_file}", "INFO")
    except Exception as e:
        debug_print(f"ERROR al guardar listado JSON: {e}", "ERROR")
    
    return detections_list


def delete_high_confidence_documents(results: List[Dict], jsonl_file: str, report_only: bool = False):
    """
    Elimina documentos con alta confianza (>0.99) de todas las carpetas del pipeline
    y del archivo JSONL.
    
    Args:
        results (List[Dict]): Lista de resultados de verificación
        jsonl_file (str): Ruta al archivo JSONL
        report_only (bool): Si True, solo reporta sin eliminar
    """
    debug_print("Procesando eliminación de documentos con alta confianza...", "INFO")
    
    # Identificar documentos a eliminar
    docs_to_delete = []
    for result in results:
        if not result.get("success", False):
            continue
        
        if result.get("high_confidence_check", {}).get("should_delete", False):
            docs_to_delete.append(result["document_id"])
    
    if not docs_to_delete:
        debug_print("No hay documentos para eliminar", "INFO")
        return {"deleted": 0, "errors": []}
    
    if report_only:
        debug_print(f"[MODO REPORTE] Se eliminarían {len(docs_to_delete)} documentos (no se eliminan realmente)", "WARN")
        return {
            "deleted": 0,
            "documents_to_delete": docs_to_delete,
            "errors": [],
            "report_only": True
        }
    
    debug_print(f"Se eliminarán {len(docs_to_delete)} documentos", "WARN")
    
    deleted_count = 0
    errors = []
    
    # Directorios a limpiar
    directories = [
        "step4_corrected_documents",
        "step4_5_cleaned_documents", 
        "step5_anonymized_documents"
    ]
    
    # Eliminar archivos de cada directorio
    for doc_id in docs_to_delete:
        debug_print(f"  Eliminando documento: {doc_id}", "WARN")
        
        for directory in directories:
            file_path = os.path.join(directory, f"{doc_id}.txt")
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    debug_print(f"    [OK] Eliminado: {file_path}", "DEBUG")
                except Exception as e:
                    error_msg = f"Error eliminando {file_path}: {e}"
                    errors.append(error_msg)
                    debug_print(f"    [X] {error_msg}", "ERROR")
            else:
                debug_print(f"    - No existe: {file_path}", "DEBUG")
    
    # Eliminar entradas del JSONL
    if os.path.exists(jsonl_file):
        debug_print(f"  Actualizando archivo JSONL: {jsonl_file}", "INFO")
        try:
            # Leer todas las líneas
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Filtrar líneas que no corresponden a documentos eliminados
            kept_lines = []
            removed_count = 0
            
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    doc_id = data.get("id", "")
                    
                    if doc_id not in docs_to_delete:
                        kept_lines.append(line)
                    else:
                        removed_count += 1
                        debug_print(f"    [OK] Entrada JSONL eliminada: {doc_id}", "DEBUG")
                except json.JSONDecodeError:
                    # Mantener líneas que no se pueden parsear
                    kept_lines.append(line)
            
            # Guardar archivo actualizado
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                f.writelines(kept_lines)
            
            debug_print(f"    [OK] JSONL actualizado: {removed_count} entradas eliminadas", "INFO")
            
        except Exception as e:
            error_msg = f"Error actualizando JSONL: {e}"
            errors.append(error_msg)
            debug_print(f"    [X] {error_msg}", "ERROR")
    else:
        debug_print(f"  [!] Archivo JSONL no encontrado: {jsonl_file}", "WARN")
    
    # Resumen
    debug_print(f"Eliminación completada:", "INFO")
    debug_print(f"  - Archivos eliminados: {deleted_count}", "INFO")
    debug_print(f"  - Errores: {len(errors)}", "ERROR" if errors else "INFO")
    
    return {
        "deleted": deleted_count,
        "documents_deleted": docs_to_delete,
        "errors": errors
    }

def main():
    parser = argparse.ArgumentParser(description="PASO 6: Verificar anonimización con modelos BSC")
    parser.add_argument("--input-dir", default="step5_anonymized_documents",
                       help="Directorio con documentos anonimizados del paso 5")
    parser.add_argument("--output-dir", default="step6_validation_results",
                       help="Directorio de salida para resultados de verificación")
    parser.add_argument("--confidence-threshold", type=float, default=0.99,
                       help="Umbral mínimo de confianza para considerar detecciones (default: 0.99 = 99%%)")
    parser.add_argument("--jsonl-file", default="examples/jsonl_data/medical_annotations_fixed.jsonl",
                       help="Archivo JSONL con anotaciones")
    parser.add_argument("--max-docs", type=int, default=None,
                       help="Máximo número de documentos a procesar")
    parser.add_argument("--delete", action="store_true",
                       help="Si está presente, elimina archivos marcados; por defecto NO borra")
    parser.add_argument("--strict", action="store_true",
                       help="Modo estricto: exit code diferente si hay documentos dudosos")

    parser.add_argument(
        "--meddocan-only",
        action="store_true",
        help="Si se activa, solo se cargará y usará el modelo MEDDOCAN (se omitirá CARMEN)"
    )
    
    args = parser.parse_args()
    
    debug_print("=== PASO 6: VERIFICACIÓN DE ANONIMIZACIÓN CON MODELOS BSC ===", "INFO")
    debug_print(f"Directorio de entrada: {args.input_dir}", "INFO")
    debug_print(f"Directorio de salida: {args.output_dir}", "INFO")
    debug_print(f"Umbral de confianza: {args.confidence_threshold}", "INFO")
    
    # Verificar directorio de entrada
    if not os.path.exists(args.input_dir):
        print(f"ERROR: No existe el directorio de entrada: {args.input_dir}")
        return
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Encontrar documentos a procesar
    input_path = Path(args.input_dir)
    documents = list(input_path.glob("*.txt"))
    
    # Filtrar solo documentos (excluyendo archivos de reporte)
    valid_documents = []
    for doc in documents:
        # Manejar archivos con doble extensión .txt.txt
        doc_name = doc.name
        if doc_name.endswith('.txt.txt'):
            doc_id = doc_name[:-8]  # Quitar ambos .txt
        elif doc_name.endswith('.txt'):
            doc_id = doc_name[:-4]  # Quitar .txt
        else:
            doc_id = doc.stem
        
        if not doc_id.endswith('_summary') and not doc_id.endswith('_report'):
            valid_documents.append(doc_id)
    
    if args.max_docs:
        valid_documents = valid_documents[:args.max_docs]
    
    debug_print(f"Encontrados {len(valid_documents)} documentos a verificar", "INFO")
    
    if not valid_documents:
        print("ERROR: No se encontraron documentos válidos para procesar")
        return
    
    
    # Procesar documentos con modelos NER
    results = []
    
    debug_print("Cargando modelos NER...", "INFO")
    # Configurar paralelización (1 worker para evitar problemas con meta tensors)
    num_workers = 1 # Sin paralelización para evitar problemas de carga del modelo
    debug_print(f"Verificación secuencial con {num_workers} worker", "INFO")
    if args.meddocan_only:
        debug_print("Modo: MEDDOCAN only (CARMEN omitido)", "INFO")
    
    # Dividir documentos en lotes
    batch_size = max(1, len(valid_documents) // num_workers)
    doc_batches = [valid_documents[i:i + batch_size] for i in range(0, len(valid_documents), batch_size)]
    
    debug_print(f"Dividido en {len(doc_batches)} lotes de ~{batch_size} documentos cada uno", "INFO")
    
    # Procesar documentos en paralelo
    debug_print("Iniciando verificación con modelos NER...", "INFO")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Enviar lotes a workers
        future_to_batch = {
            executor.submit(process_documents_batch, batch, args.input_dir, args.output_dir, args.confidence_threshold, not args.meddocan_only): i 
            for i, batch in enumerate(doc_batches)
        }
        
        # Recopilar resultados
        completed_batches = 0
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                results.extend(batch_results)
                completed_batches += 1
                
                successful = len([r for r in batch_results if r.get("success", False)])
                perfect = len([r for r in batch_results if r.get("success", False) and 
                              r.get("anonymization_verification", {}).get("success", False)])
                debug_print(f"  Lote {batch_idx + 1}/{len(doc_batches)} completado: {successful}/{len(batch_results)} exitosos, {perfect} perfectos", "INFO")
                
            except Exception as e:
                debug_print(f"Error procesando lote {batch_idx}: {e}", "ERROR")
    
    # Crear reporte resumen
    create_verification_summary(results, args.output_dir)
    
    # Crear listado detallado de detecciones
    create_detailed_detections_report(results, args.output_dir)
    
    # Eliminar documentos con alta confianza (>0.99)
    # Por defecto no se eliminan; pasar --delete para permitir borrado.
    do_delete = bool(args.delete)
    deletion_result = delete_high_confidence_documents(results, args.jsonl_file, report_only=(not do_delete))

    debug_print("Paso 6 completado! Verificación de anonimización finalizada.", "INFO")
    
    # NUEVO: Exit codes diferenciados en modo strict
    if args.strict:
        # Verificar si hay errores de modelos
        failed_count = len([r for r in results if not r.get("success", False)])
        if failed_count > 0:
            debug_print(f"[STRICT] Errores de modelos detectados: {failed_count}", "ERROR")
            return 1
        
        # Verificar si hay documentos con alta confianza
        high_conf_count = len([r for r in results if r.get("high_confidence_check", {}).get("should_delete", False)])
        if high_conf_count > 0:
            debug_print(f"[STRICT] Documentos con alta confianza detectados: {high_conf_count}", "WARN")
            return 2
    
    return 0

if __name__ == "__main__":
    main()