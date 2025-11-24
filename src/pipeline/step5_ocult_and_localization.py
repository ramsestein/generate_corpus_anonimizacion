#!/usr/bin/env python3
"""
STEP 5 DEL PIPELINE: Anonimización con localización
Toma los documentos corregidos y reemplaza las cadenas de texto exactas por "XXX".
Además, localiza las posiciones (start, end) de cada entidad y las añade al JSON correspondiente
en medical_annotations.jsonl para tener el mapeo completo de localizaciones.
"""

import json
import csv
import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Set
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def debug_print(message: str, level: str = "INFO"):
    """Función para imprimir mensajes de debug con timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def load_available_labels(csv_file: str) -> Set[str]:
    """Carga las etiquetas disponibles del archivo CSV."""
    labels = set()
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Saltar header
            
            for row in csv_reader:
                if len(row) >= 1 and row[0].strip():
                    labels.add(row[0].strip())
                if len(row) >= 2 and row[1].strip():
                    labels.add(row[1].strip())
    
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {csv_file}")
        return set()
    
    # Remover etiquetas vacías o headers
    labels.discard("")
    labels.discard("MEDDOCAN")
    labels.discard("CARMEN-I")
    
    return labels

def load_all_annotations(jsonl_file: str) -> Dict[str, List[Dict]]:
    """
    Carga todas las anotaciones del archivo JSONL organizadas por ID.
    
    Returns:
        Dict[str, List[Dict]]: Diccionario con ID como clave y lista de entidades como valor
    """
    annotations_by_id = {}
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        doc_id = json_data.get('id', f'unknown_{line_num}')
                        entities = json_data.get('data', [])
                        
                        # Filtrar entidades válidas
                        valid_entities = []
                        for entity in entities:
                            text = entity.get('text', '').strip()
                            if text and text not in ['[No se proporcionó texto para extraer la edad]', '[valor generado]']:
                                valid_entities.append(entity)
                        
                        annotations_by_id[doc_id] = valid_entities
                        
                    except json.JSONDecodeError as e:
                        print(f"Error al parsear JSON en línea {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {jsonl_file}")
        return {}
    
    return annotations_by_id

def update_jsonl_with_locations(jsonl_file: str, doc_id: str, entity_locations: List[Dict]):
    """
    Actualiza el archivo JSONL añadiendo las posiciones de las entidades localizadas.
    
    Args:
        jsonl_file (str): Archivo JSONL a actualizar
        doc_id (str): ID del documento
        entity_locations (List[Dict]): Localizaciones de entidades
    """
    if not entity_locations:
        return
    
    debug_print(f"    Actualizando JSON con {len(entity_locations)} localizaciones...", "DEBUG")
    
    # Leer todas las entradas
    entries = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        debug_print(f"Error: No se encontró {jsonl_file}", "ERROR")
        return
    
    # Actualizar la entrada correspondiente
    for entry in entries:
        if entry.get('id') == doc_id:
            # Añadir localizaciones a cada entidad
            entities_data = entry.get('data', [])
            for entity in entities_data:
                entity_text = entity.get('text', '').strip()
                entity_label = entity.get('entity', '') or entity.get('etiqueta', '')
                
                # Buscar localización correspondiente
                for location in entity_locations:
                    if (location['etiqueta'] == entity_label and 
                        location['text'] == entity_text and 
                        location['occurrence_index'] == 1):  # Primera ocurrencia
                        
                        entity['start'] = location['start']
                        entity['end'] = location['end']
                        entity['anonymized_start'] = location['anonymized_start']
                        entity['anonymized_end'] = location['anonymized_end']
                        break
            break
    
    # Reescribir el archivo JSONL actualizado
    try:
        with open(jsonl_file, 'w', encoding='utf-8', errors='ignore') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        debug_print(f"    JSON actualizado con localizaciones", "DEBUG")
    except Exception as e:
        debug_print(f"Error actualizando JSON: {e}", "ERROR")

def clean_entity_names_with_regex(text: str, available_labels: Set[str]) -> str:
    """
    Elimina nombres literales de etiquetas del texto usando regex.
    
    Args:
        text (str): Texto del documento
        available_labels (Set[str]): Etiquetas disponibles
        
    Returns:
        str: Texto limpio sin nombres de etiquetas
    """
    cleaned_text = text
    
    # Lista de nombres de entidades a eliminar
    entity_names_to_remove = [
        "NOMBRE_SUJETO_ASISTENCIA",
        "NOMBRE_PERSONAL_SANITARIO", 
        "ID_SUJETO_ASISTENCIA",
        "CENTRO_SALUD",
        "HOSPITAL",
        "CALLE",
        "TERRITORIO",
        "PAIS",
        "FECHAS",
        "EDAD_SUJETO_ASISTENCIA",
        "SEXO_SUJETO_ASISTENCIA",
        "NUMERO_TELEFONO",
        "CORREO_ELECTRONICO",
        "PROFESION",
        "FAMILIARES_SUJETO_ASISTENCIA",
        "INSTITUCION",
        "URL_WEB",
        "NUMERO_FAX",
        "ID_CONTACTO_ASISTENCIAL",
        "ID_TITULACION_PERSONAL_SANITARIO",
        "ID_EMPLEO_PERSONAL_SANITARIO",
        "ID_ASEGURAMIENTO",
        "NUMERO_BENEF_PLAN_SALUD",
        "IDENTIF_BIOMETRICOS",
        "IDENTIF_DISPOSITIVOS_NRSERIE",
        "IDENTIF_VEHICULOS_NRSERIE_PLACAS",
        "DIREC_PROT_INTERNET",
        "OTROS_SUJETO_ASISTENCIA",
        "OTRO_NUMERO_IDENTIF"
    ]
    
    removed_count = 0
    for entity_name in entity_names_to_remove:
        if entity_name in available_labels:
            # Usar regex para eliminar el nombre de la etiqueta con contexto
            patterns = [
                rf'\blos\s+{re.escape(entity_name)}\b',  # "los FAMILIARES_SUJETO_ASISTENCIA"
                rf'\blas\s+{re.escape(entity_name)}\b',  # "las FECHAS"
                rf'\bel\s+{re.escape(entity_name)}\b',   # "el CENTRO_SALUD"
                rf'\bla\s+{re.escape(entity_name)}\b',   # "la INSTITUCION"
                rf'\ben\s+{re.escape(entity_name)}\b',   # "en HOSPITAL"
                rf'\bde\s+{re.escape(entity_name)}\b',   # "de TERRITORIO"
                rf'\bcon\s+{re.escape(entity_name)}\b',  # "con PROFESION"
                rf'\bpor\s+{re.escape(entity_name)}\b',  # "por PAIS"
                rf'\b{re.escape(entity_name)}\b'         # Solo el nombre de la etiqueta
            ]
            
            for pattern in patterns:
                if re.search(pattern, cleaned_text, re.IGNORECASE):
                    cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
                    removed_count += 1
                    break  # Solo eliminar una vez por entidad
    
    # Limpiar espacios múltiples
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    if removed_count > 0:
        debug_print(f"    Eliminados {removed_count} nombres de etiquetas con regex", "DEBUG")
    
    return cleaned_text

def anonymize_document_and_locate(text: str, entities: List[Dict], available_labels: Set[str], doc_id: str) -> Dict:
    """
    Anonimiza un documento reemplazando las cadenas exactas por "XXXXXX".
    
    Args:
        text (str): Texto del documento
        entities (List[Dict]): Lista de entidades del JSON
        available_labels (Set[str]): Etiquetas disponibles
        doc_id (str): ID del documento
        
    Returns:
        Dict: Resultado de la anonimización
    """
    debug_print(f"  Anonimizando y localizando entidades en documento {doc_id}...", "DEBUG")
    
    # Primero limpiar nombres de etiquetas con regex
    cleaned_text = clean_entity_names_with_regex(text, available_labels)
    
    anonymized_text = cleaned_text
    replacements_made = []
    entities_not_found = []
    entity_locations = []  # Para añadir al JSON original
    
    # Extraer textos únicos de las entidades válidas
    entity_texts = []
    for entity in entities:
        entity_text = entity.get('text', '').strip()
        entity_label = entity.get('entity', '') or entity.get('etiqueta', '')  # Compatibilidad
        
        if entity_label in available_labels and entity_text:
            entity_texts.append({
                'text': entity_text,
                'label': entity_label,
                'original_entity': entity
            })
    
    # Ordenar por longitud descendente para evitar reemplazos parciales
    entity_texts.sort(key=lambda x: len(x['text']), reverse=True)
    
    debug_print(f"    Buscando {len(entity_texts)} entidades para anonimizar...", "DEBUG")
    
    # Rastrear el offset causado por reemplazos anteriores
    offset = 0
    
    for entity_info in entity_texts:
        entity_text = entity_info['text']
        entity_label = entity_info['label']
        
        # Buscar todas las ocurrencias exactas en el texto LIMPIO (antes de reemplazos)
        original_positions = []
        start = 0
        while True:
            pos = cleaned_text.find(entity_text, start)
            if pos == -1:
                break
            original_positions.append(pos)
            start = pos + 1
        
        if original_positions:
            debug_print(f"      Localizando '{entity_text}' ({entity_label}): {len(original_positions)} ocurrencia(s)", "DEBUG")
            
            # Reemplazar en el texto anonimizado
            anonymized_text = anonymized_text.replace(entity_text, "XXX")
            
            # Calcular posiciones ajustadas por reemplazos anteriores
            adjusted_positions = []
            replacement_length_diff = len("XXX") - len(entity_text)
            
            for pos in original_positions:
                # Ajustar posición considerando reemplazos anteriores
                adjusted_start = pos + offset
                adjusted_end = adjusted_start + len("XXX")
                
                adjusted_positions.append({
                    'start': adjusted_start,
                    'end': adjusted_end,
                    'original_start': pos,
                    'original_end': pos + len(entity_text)
                })
                
                # Actualizar offset para futuros reemplazos
                offset += replacement_length_diff
            
            replacements_made.append({
                'original_text': entity_text,
                'label': entity_label,
                'occurrences': len(original_positions),
                'positions': adjusted_positions,
                'replacement': "XXX"
            })
            
            # Crear entrada de localización para el JSON
            for i, pos_info in enumerate(adjusted_positions):
                entity_locations.append({
                    'etiqueta': entity_label,
                    'text': entity_text,
                    'start': pos_info['original_start'],
                    'end': pos_info['original_end'],
                    'anonymized_start': pos_info['start'],
                    'anonymized_end': pos_info['end'],
                    'occurrence_index': i + 1
                })
        else:
            debug_print(f"      NO encontrado: '{entity_text}' ({entity_label})", "WARN")
            entities_not_found.append({
                'text': entity_text,
                'label': entity_label
            })
    
    return {
        'document_id': doc_id,
        'original_text': text,
        'anonymized_text': anonymized_text,
        'total_entities': len(entity_texts),
        'replacements_made': replacements_made,
        'entities_not_found': entities_not_found,
        'entity_locations': entity_locations,
        'replacement_count': len(replacements_made),
        'not_found_count': len(entities_not_found),
        'anonymization_success_rate': len(replacements_made) / len(entity_texts) * 100 if entity_texts else 100
    }

def process_documents_batch(docs_batch: List[str], input_dir: str, annotations_by_id: Dict[str, List[Dict]], 
                           available_labels: Set[str], output_dir: str, jsonl_file: str) -> List[Dict]:
    """
    Procesa un lote de documentos en paralelo.
    
    Args:
        docs_batch: Lista de IDs de documentos a procesar
        input_dir: Directorio de entrada
        annotations_by_id: Anotaciones por ID
        available_labels: Etiquetas disponibles
        output_dir: Directorio de salida
        jsonl_file: Archivo JSONL a actualizar
        
    Returns:
        List[Dict]: Resultados del lote
    """
    results = []
    for doc_id in docs_batch:
        result = process_single_document(doc_id, input_dir, annotations_by_id, available_labels, output_dir, jsonl_file)
        results.append(result)
    return results

def process_single_document(doc_id: str, input_dir: str, annotations_by_id: Dict[str, List[Dict]], 
                           available_labels: Set[str], output_dir: str, jsonl_file: str) -> Dict:
    """
    Procesa un documento individual para anonimización.
    
    Args:
        doc_id (str): ID del documento
        input_dir (str): Directorio con documentos del paso 2
        annotations_by_id (Dict): Anotaciones organizadas por ID
        available_labels (Set[str]): Etiquetas disponibles
        output_dir (str): Directorio de salida
        
    Returns:
        Dict: Resultado del procesamiento
    """
    debug_print(f"Procesando documento: {doc_id}", "INFO")
    
    # Leer el documento del paso 2
    doc_file = os.path.join(input_dir, f"{doc_id}.txt")
    if not os.path.exists(doc_file):
        debug_print(f"No se encontró el documento: {doc_file}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": "Documento no encontrado"
        }
    
    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
    except Exception as e:
        debug_print(f"Error al leer {doc_file}: {e}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": f"No se pudo leer el archivo: {e}"
        }
    
    debug_print(f"  Texto original: {len(original_text)} caracteres", "DEBUG")
    
    # Buscar las entidades correspondientes
    if doc_id not in annotations_by_id:
        debug_print(f"  ADVERTENCIA: No se encontraron anotaciones para {doc_id}", "WARN")
        return {
            "document_id": doc_id,
            "success": False,
            "error": f"No se encontraron anotaciones para {doc_id}"
        }
    
    entities = annotations_by_id[doc_id]
    debug_print(f"  Entidades a anonimizar: {len(entities)}", "DEBUG")
    
    # Anonimizar el documento y localizar posiciones
    anonymization_result = anonymize_document_and_locate(original_text, entities, available_labels, doc_id)
    
    # Guardar documento anonimizado
    anonymized_file = os.path.join(output_dir, f"{doc_id}.txt")
    try:
        with open(anonymized_file, 'w', encoding='utf-8') as f:
            f.write(anonymization_result['anonymized_text'])
        debug_print(f"  Documento anonimizado guardado: {anonymized_file}", "INFO")
    except Exception as e:
        debug_print(f"  ERROR al guardar documento anonimizado: {e}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": f"No se pudo guardar el documento: {e}"
        }
    
    # Actualizar el JSON con las localizaciones
    if anonymization_result['entity_locations']:
        update_jsonl_with_locations(jsonl_file, doc_id, anonymization_result['entity_locations'])
    
    # Mostrar estadísticas
    debug_print(f"  Estadísticas:", "INFO")
    debug_print(f"    - Entidades procesadas: {anonymization_result['total_entities']}", "INFO")
    debug_print(f"    - Reemplazos realizados: {anonymization_result['replacement_count']}", "INFO")
    debug_print(f"    - Entidades no encontradas: {anonymization_result['not_found_count']}", "INFO")
    debug_print(f"    - Localizaciones añadidas: {len(anonymization_result['entity_locations'])}", "INFO")
    debug_print(f"    - Tasa de éxito: {anonymization_result['anonymization_success_rate']:.1f}%", "INFO")
    
    return {
        "document_id": doc_id,
        "success": True,
        "original_file": doc_file,
        "anonymized_file": anonymized_file,
        "total_entities": anonymization_result['total_entities'],
        "replacements_made": anonymization_result['replacement_count'],
        "entities_not_found": anonymization_result['not_found_count'],
        "entity_locations_count": len(anonymization_result['entity_locations']),
        "anonymization_success_rate": anonymization_result['anonymization_success_rate'],
        "original_length": len(original_text),
        "anonymized_length": len(anonymization_result['anonymized_text'])
    }

def create_summary_report(results: List[Dict], output_dir: str, annotations_file: str):
    """
    Crea un reporte resumen de todo el proceso de anonimización del paso 3.
    
    Args:
        results (List[Dict]): Lista de resultados de procesamiento
        output_dir (str): Directorio de salida
        annotations_file (str): Archivo de anotaciones usado
    """
    debug_print("Creando reporte resumen del paso 3...", "INFO")
    
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    summary = {
        "pipeline_step": 3,
        "process": "final_document_anonymization",
        "annotations_source": annotations_file,
        "total_documents": len(results),
        "successful_documents": len(successful_results),
        "failed_documents": len(failed_results),
        "total_entities_processed": sum([r.get("total_entities", 0) for r in successful_results]),
        "total_replacements_made": sum([r.get("replacements_made", 0) for r in successful_results]),
        "total_entities_not_found": sum([r.get("entities_not_found", 0) for r in successful_results]),
        "average_anonymization_success_rate": sum([r.get("anonymization_success_rate", 0) for r in successful_results]) / len(successful_results) if successful_results else 0,
        "total_original_length": sum([r.get("original_length", 0) for r in successful_results]),
        "total_anonymized_length": sum([r.get("anonymized_length", 0) for r in successful_results]),
        "documents": results
    }
    
    summary_file = os.path.join(output_dir, "step3_anonymization_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    debug_print(f"Reporte guardado: {summary_file}", "INFO")
    
    # Mostrar resumen en consola
    print(f"\n{'='*60}")
    print(f"PASO 3 - ANONIMIZACIÓN FINAL COMPLETADA")
    print(f"{'='*60}")
    print(f"Documentos procesados: {summary['total_documents']}")
    print(f"Exitosos: {summary['successful_documents']}")
    print(f"Fallidos: {summary['failed_documents']}")
    print(f"Total entidades procesadas: {summary['total_entities_processed']}")
    print(f"Total reemplazos realizados: {summary['total_replacements_made']}")
    print(f"Total entidades no encontradas: {summary['total_entities_not_found']}")
    print(f"Tasa promedio de anonimización: {summary['average_anonymization_success_rate']:.1f}%")
    
    if summary['total_entities_processed'] > 0:
        global_success_rate = (summary['total_replacements_made'] / summary['total_entities_processed']) * 100
        print(f"Tasa global de éxito: {global_success_rate:.1f}%")
    
    print(f"Reduccion de texto: {summary['total_original_length']} -> {summary['total_anonymized_length']} caracteres")
    
    if summary['failed_documents'] > 0:
        print(f"\nDocumentos con problemas:")
        for result in failed_results:
            print(f"  - {result['document_id']}: {result.get('error', 'Error desconocido')}")
    
    print(f"\nDirectorio de salida: {output_dir}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="STEP 5: Anonimizar documentos y localizar posiciones de entidades")
    parser.add_argument("--input-dir", default="step4_5_cleaned_documents",
                       help="Directorio con documentos del paso 5")
    parser.add_argument("--jsonl-file", default="examples/jsonl_data/medical_annotations_fixed.jsonl", 
                       help="Archivo JSONL con anotaciones")
    parser.add_argument("--labels-file", default="examples/etiquetas_anonimizacion_meddocan_carmenI.csv",
                       help="Archivo CSV con etiquetas disponibles")
    parser.add_argument("--output-dir", default="step5_anonymized_documents",
                       help="Directorio de salida para documentos anonimizados")
    parser.add_argument("--max-docs", type=int, default=None,
                       help="Máximo número de documentos a procesar")
    
    args = parser.parse_args()
    
    debug_print("=== STEP 5: ANONIMIZACIÓN CON LOCALIZACIÓN ===", "INFO")
    debug_print(f"Directorio de entrada: {args.input_dir}", "INFO")
    debug_print(f"Directorio de salida: {args.output_dir}", "INFO")
    
    # Verificar que existe el directorio de entrada
    if not os.path.exists(args.input_dir):
        print(f"ERROR: No existe el directorio de entrada: {args.input_dir}")
        return
    
    # Verificar que existe el archivo de anotaciones
    if not os.path.exists(args.jsonl_file):
        print(f"ERROR: No existe el archivo de anotaciones: {args.jsonl_file}")
        return
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar etiquetas disponibles
    debug_print("Cargando etiquetas disponibles...", "INFO")
    available_labels = load_available_labels(args.labels_file)
    debug_print(f"Cargadas {len(available_labels)} etiquetas disponibles", "INFO")
    
    # Cargar anotaciones médicas
    debug_print("Cargando anotaciones médicas...", "INFO")
    annotations_by_id = load_all_annotations(args.jsonl_file)
    
    if not annotations_by_id:
        print("ERROR: No se pudieron cargar las anotaciones médicas")
        return
    
    debug_print(f"Cargadas anotaciones para {len(annotations_by_id)} documentos", "INFO")
    
    # Encontrar documentos a procesar
    input_path = Path(args.input_dir)
    documents = list(input_path.glob("*.txt"))
    
    # Filtrar solo documentos que tienen anotaciones (excluyendo archivos de reporte)
    valid_documents = []
    for doc in documents:
        doc_id = doc.stem
        if doc_id in annotations_by_id and not doc_id.endswith('_summary'):
            valid_documents.append(doc_id)
    
    if args.max_docs:
        valid_documents = valid_documents[:args.max_docs]
    
    debug_print(f"Encontrados {len(valid_documents)} documentos a anonimizar", "INFO")
    
    if not valid_documents:
        print("ERROR: No se encontraron documentos válidos para procesar")
        return
    
    # Configurar paralelización
    num_workers = min(mp.cpu_count(), 8, len(valid_documents))
    debug_print(f"Procesamiento paralelo con {num_workers} workers", "INFO")
    
    # Dividir documentos en lotes
    batch_size = max(1, len(valid_documents) // (num_workers * 2))
    doc_batches = [valid_documents[i:i + batch_size] for i in range(0, len(valid_documents), batch_size)]
    
    debug_print(f"Dividido en {len(doc_batches)} lotes de ~{batch_size} documentos cada uno", "INFO")
    
    # Procesar documentos en paralelo
    debug_print("Iniciando procesamiento paralelo...", "INFO")
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Enviar lotes a workers
        future_to_batch = {
            executor.submit(process_documents_batch, batch, args.input_dir, annotations_by_id, available_labels, args.output_dir, args.jsonl_file): i 
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
                debug_print(f"  Lote {batch_idx + 1}/{len(doc_batches)} completado: {successful}/{len(batch_results)} exitosos", "INFO")
                
            except Exception as e:
                debug_print(f"Error procesando lote {batch_idx}: {e}", "ERROR")
    
    # Crear reporte resumen
    create_summary_report(results, args.output_dir, args.jsonl_file)
    
    debug_print("Step 5 completado! Documentos anonimizados y JSON actualizado con localizaciones.", "INFO")

if __name__ == "__main__":
    main()
