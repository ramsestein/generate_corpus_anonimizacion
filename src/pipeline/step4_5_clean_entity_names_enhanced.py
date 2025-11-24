#!/usr/bin/env python3
"""
STEP 4.5 DEL PIPELINE: Limpieza avanzada de nombres de etiquetas + Reparación JSONL
Toma los documentos del step4 y elimina nombres literales de etiquetas que puedan haber quedado
en el texto, como "FAMILIARES_SUJETO_ASISTENCIA", "FECHAS", etc.
Además, al final repara el archivo JSONL y elimina documentos correspondientes a entradas no reparables.
"""

import os
import argparse
import re
import json
from pathlib import Path
from typing import Set, List, Dict
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
        import csv
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

def clean_entity_names_advanced(document_text: str, available_labels: Set[str]) -> str:
    """
    Elimina nombres literales de entidades del documento con patrones avanzados.
    
    Args:
        document_text (str): Texto del documento
        available_labels (Set[str]): Etiquetas disponibles
        
    Returns:
        str: Documento limpio sin nombres de entidades
    """
    cleaned_text = document_text
    
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
    
    # Eliminar nombres literales de entidades con patrones avanzados
    removed_count = 0
    for entity_name in entity_names_to_remove:
        if entity_name in available_labels:
            # Patrones comunes para eliminar con contexto
            patterns = [
                f"los {entity_name}",  # "los FAMILIARES_SUJETO_ASISTENCIA"
                f"las {entity_name}",  # "las FECHAS"
                f"el {entity_name}",   # "el CENTRO_SALUD"
                f"la {entity_name}",   # "la INSTITUCION"
                f"en {entity_name}",   # "en HOSPITAL"
                f"de {entity_name}",   # "de TERRITORIO"
                f"con {entity_name}",  # "con PROFESION"
                f"por {entity_name}",  # "por PAIS"
                f"en el apartado de {entity_name}",  # "en el apartado de FAMILIARES_SUJETO_ASISTENCIA"
                f"con fecha de {entity_name}",       # "con fecha de FECHAS"
                f"en la sección de {entity_name}",   # "en la sección de CENTRO_SALUD"
                f"dentro de {entity_name}",          # "dentro de TERRITORIO"
                f"relacionado con {entity_name}",     # "relacionado con PROFESION"
                f"correspondiente a {entity_name}",   # "correspondiente a PAIS"
                f"en relación con los {entity_name}", # "en relación con los FAMILIARES_SUJETO_ASISTENCIA"
                f"en relación con las {entity_name}", # "en relación con las FECHAS"
                f"en relación con el {entity_name}",  # "en relación con el CENTRO_SALUD"
                f"en relación con la {entity_name}",  # "en relación con la INSTITUCION"
                f"según los {entity_name}",          # "según los FAMILIARES_SUJETO_ASISTENCIA"
                f"según las {entity_name}",          # "según las FECHAS"
                f"según el {entity_name}",            # "según el CENTRO_SALUD"
                f"según la {entity_name}",            # "según la INSTITUCION"
                f"para los {entity_name}",           # "para los FAMILIARES_SUJETO_ASISTENCIA"
                f"para las {entity_name}",           # "para las FECHAS"
                f"para el {entity_name}",            # "para el CENTRO_SALUD"
                f"para la {entity_name}",             # "para la INSTITUCION"
                entity_name            # Solo el nombre de la etiqueta
            ]
            
            for pattern in patterns:
                if pattern in cleaned_text:
                    cleaned_text = cleaned_text.replace(pattern, "")
                    removed_count += 1
                    break  # Solo eliminar una vez por entidad
    
    # Limpiar espacios múltiples y saltos de línea extra
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text, removed_count

def process_single_document(doc_id: str, input_dir: str, output_dir: str, available_labels: Set[str]) -> Dict:
    """
    Procesa un documento individual para limpiar nombres de etiquetas.
    
    Args:
        doc_id (str): ID del documento
        input_dir (str): Directorio con documentos del step4
        output_dir (str): Directorio de salida
        available_labels (Set[str]): Etiquetas disponibles
        
    Returns:
        Dict: Resultado del procesamiento
    """
    debug_print(f"Procesando documento: {doc_id}", "INFO")
    
    # Leer el documento del step4
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
    
    # Limpiar nombres de etiquetas
    cleaned_text, removed_count = clean_entity_names_advanced(original_text, available_labels)
    
    # Guardar documento limpio
    cleaned_file = os.path.join(output_dir, f"{doc_id}.txt")
    try:
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        debug_print(f"  Documento limpio guardado: {cleaned_file}", "INFO")
    except Exception as e:
        debug_print(f"  ERROR al guardar documento limpio: {e}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": f"No se pudo guardar el documento: {e}"
        }
    
    # Mostrar estadísticas
    debug_print(f"  Estadísticas:", "INFO")
    debug_print(f"    - Nombres de etiquetas eliminados: {removed_count}", "INFO")
    debug_print(f"    - Longitud original: {len(original_text)} caracteres", "INFO")
    debug_print(f"    - Longitud final: {len(cleaned_text)} caracteres", "INFO")
    
    return {
        "document_id": doc_id,
        "success": True,
        "original_file": doc_file,
        "cleaned_file": cleaned_file,
        "entity_names_removed": removed_count,
        "original_length": len(original_text),
        "cleaned_length": len(cleaned_text)
    }

def process_documents_batch(docs_batch: List[str], input_dir: str, output_dir: str, available_labels: Set[str]) -> List[Dict]:
    """
    Procesa un lote de documentos en paralelo.
    
    Args:
        docs_batch: Lista de IDs de documentos a procesar
        input_dir: Directorio de entrada
        output_dir: Directorio de salida
        available_labels: Etiquetas disponibles
        
    Returns:
        List[Dict]: Resultados del lote
    """
    results = []
    for doc_id in docs_batch:
        result = process_single_document(doc_id, input_dir, output_dir, available_labels)
        results.append(result)
    return results

def extract_doc_id_from_broken_line(line: str) -> str:
    """
    Intenta extraer el ID del documento de una línea JSON malformada.
    Busca patrones como "id": "uuid" al inicio de la línea.
    """
    try:
        # Buscar patrones comunes de ID
        patterns = [
            r'"id":\s*"([a-f0-9\-]{36})"',  # UUID completo
            r'"id":\s*"([a-f0-9\-]{8})',     # UUID parcial al inicio
            r'id["\']?\s*:\s*["\']([a-f0-9\-]{36})',  # Variaciones de comillas
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
        
        # Si no encontramos un patrón, buscar cualquier UUID en la línea
        uuid_pattern = r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
        match = re.search(uuid_pattern, line)
        if match:
            return match.group(1)
            
    except Exception as e:
        debug_print(f"Error extrayendo ID de línea: {e}", "ERROR")
    
    return None

def fix_json_line(line: str) -> str:
    """
    Intenta reparar una línea JSON malformada
    """
    # Casos comunes de errores:
    
    # 1. Líneas que contienen múltiples objetos JSON concatenados
    if '}{' in line:
        # Dividir en múltiples objetos
        parts = line.split('}{')
        fixed_parts = []
        for i, part in enumerate(parts):
            if i == 0:
                part = part + '}'
            elif i == len(parts) - 1:
                part = '{' + part
            else:
                part = '{' + part + '}'
            
            # Verificar si es JSON válido
            try:
                json.loads(part)
                fixed_parts.append(part)
            except:
                pass
        
        # Si encontramos al menos una parte válida, devolver la primera
        if fixed_parts:
            return fixed_parts[0]
    
    # 2. Comillas no cerradas
    if line.count('"') % 2 != 0:
        # Intentar cerrar la última comilla
        if line.endswith('"'):
            line = line[:-1] + '"}'
        else:
            line = line + '"'
    
    # 3. Comas faltantes
    line = re.sub(r'}(\s*){', '},{', line)
    
    # 4. Comillas dobles faltantes en claves
    line = re.sub(r'(\w+):', r'"\1":', line)
    
    # 5. Limpiar caracteres especiales problemáticos
    line = line.replace('\x00', '').replace('\x01', '').replace('\x02', '')
    
    return line

def fix_jsonl_file(input_file: str, output_file: str, step4_5_dir: str) -> Dict:
    """
    Repara un archivo JSONL con errores de parsing y codificación.
    Elimina documentos de step4.5 correspondientes a entradas no reparables.
    """
    debug_print("=== INICIANDO REPARACIÓN DEL ARCHIVO JSONL ===", "INFO")
    debug_print(f"Reparando archivo: {input_file}", "INFO")
    debug_print(f"Directorio step4.5: {step4_5_dir}", "INFO")
    
    fixed_lines = []
    error_lines = []
    error_doc_ids = set()  # IDs de documentos que no se pudieron reparar
    line_num = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except UnicodeDecodeError:
        debug_print("Error de codificación UTF-8, intentando con latin-1...", "WARN")
        try:
            with open(input_file, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            debug_print(f"Error al leer el archivo: {e}", "ERROR")
            return {"success": False, "error": str(e)}
    
    # Dividir por líneas
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        try:
            # Intentar parsear como JSON
            json_data = json.loads(line)
            fixed_lines.append(line)
        except json.JSONDecodeError:
            # Intentar reparar la línea
            fixed_line = fix_json_line(line)
            if fixed_line:
                try:
                    json_data = json.loads(fixed_line)
                    fixed_lines.append(fixed_line)
                    debug_print(f"Línea {line_num} reparada exitosamente", "DEBUG")
                except json.JSONDecodeError:
                    error_lines.append((line_num, line))
                    debug_print(f"Línea {line_num} no se pudo reparar", "WARN")
                    
                    # Intentar extraer el ID del documento de la línea malformada
                    doc_id = extract_doc_id_from_broken_line(line)
                    if doc_id:
                        error_doc_ids.add(doc_id)
                        debug_print(f"  -> Documento {doc_id} será eliminado de step4.5", "WARN")
            else:
                error_lines.append((line_num, line))
                debug_print(f"Línea {line_num} no se pudo reparar", "WARN")
                
                # Intentar extraer el ID del documento de la línea malformada
                doc_id = extract_doc_id_from_broken_line(line)
                if doc_id:
                    error_doc_ids.add(doc_id)
                    debug_print(f"  -> Documento {doc_id} será eliminado de step4.5", "WARN")
    
    # Escribir archivo reparado
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    # Eliminar documentos correspondientes de step4.5
    deleted_docs = []
    if error_doc_ids and os.path.exists(step4_5_dir):
        debug_print(f"Eliminando {len(error_doc_ids)} documentos de {step4_5_dir}...", "INFO")
        for doc_id in error_doc_ids:
            doc_file = os.path.join(step4_5_dir, f"{doc_id}.txt")
            if os.path.exists(doc_file):
                try:
                    os.remove(doc_file)
                    deleted_docs.append(doc_id)
                    debug_print(f"  Eliminado: {doc_file}", "DEBUG")
                except Exception as e:
                    debug_print(f"  Error eliminando {doc_file}: {e}", "ERROR")
            else:
                debug_print(f"  No encontrado: {doc_file}", "WARN")
    
    result = {
        "success": True,
        "lines_processed": len(lines),
        "lines_repaired": len(fixed_lines),
        "lines_with_errors": len(error_lines),
        "documents_deleted": len(deleted_docs),
        "deleted_doc_ids": list(deleted_docs)
    }
    
    debug_print(f"Resumen de reparación JSONL:", "INFO")
    debug_print(f"  Líneas procesadas: {result['lines_processed']}", "INFO")
    debug_print(f"  Líneas reparadas: {result['lines_repaired']}", "INFO")
    debug_print(f"  Líneas con errores: {result['lines_with_errors']}", "INFO")
    debug_print(f"  Documentos eliminados de step4.5: {result['documents_deleted']}", "INFO")
    
    return result

def create_summary_report(results: List[Dict], output_dir: str, jsonl_fix_result: Dict = None):
    """
    Crea un reporte resumen del proceso de limpieza y reparación JSONL.
    
    Args:
        results (List[Dict]): Lista de resultados de procesamiento
        output_dir (str): Directorio de salida
        jsonl_fix_result (Dict): Resultado de la reparación JSONL
    """
    debug_print("Creando reporte resumen del paso 4.5...", "INFO")
    
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    summary = {
        "pipeline_step": 4.5,
        "process": "advanced_entity_name_cleaning_and_jsonl_fix",
        "total_documents": len(results),
        "successful_documents": len(successful_results),
        "failed_documents": len(failed_results),
        "total_entity_names_removed": sum([r.get("entity_names_removed", 0) for r in successful_results]),
        "total_original_length": sum([r.get("original_length", 0) for r in successful_results]),
        "total_cleaned_length": sum([r.get("cleaned_length", 0) for r in successful_results]),
        "jsonl_fix_result": jsonl_fix_result,
        "documents": results
    }
    
    summary_file = os.path.join(output_dir, "step4_5_cleaning_and_jsonl_fix_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    debug_print(f"Reporte guardado: {summary_file}", "INFO")
    
    # Mostrar resumen en consola
    print(f"\n{'='*60}")
    print(f"PASO 4.5 - LIMPIEZA AVANZADA Y REPARACIÓN JSONL COMPLETADA")
    print(f"{'='*60}")
    print(f"Documentos procesados: {summary['total_documents']}")
    print(f"Exitosos: {summary['successful_documents']}")
    print(f"Fallidos: {summary['failed_documents']}")
    print(f"Total nombres de etiquetas eliminados: {summary['total_entity_names_removed']}")
    
    if summary['total_original_length'] > 0:
        reduction_rate = ((summary['total_original_length'] - summary['total_cleaned_length']) / summary['total_original_length']) * 100
        print(f"Reducción de texto: {summary['total_original_length']} -> {summary['total_cleaned_length']} caracteres ({reduction_rate:.1f}%)")
    
    if jsonl_fix_result and jsonl_fix_result.get("success"):
        print(f"\nREPARACIÓN JSONL:")
        print(f"  Líneas procesadas: {jsonl_fix_result['lines_processed']}")
        print(f"  Líneas reparadas: {jsonl_fix_result['lines_repaired']}")
        print(f"  Líneas con errores: {jsonl_fix_result['lines_with_errors']}")
        print(f"  Documentos eliminados: {jsonl_fix_result['documents_deleted']}")
    
    if summary['failed_documents'] > 0:
        print(f"\nDocumentos con problemas:")
        for result in failed_results:
            print(f"  - {result['document_id']}: {result.get('error', 'Error desconocido')}")
    
    print(f"\nDirectorio de salida: {output_dir}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="STEP 4.5: Limpieza avanzada de nombres de etiquetas + Reparación JSONL")
    parser.add_argument("--input-dir", default="step4_corrected_documents",
                       help="Directorio con documentos del paso 4")
    parser.add_argument("--labels-file", default="examples/etiquetas_anonimizacion_meddocan_carmenI.csv",
                       help="Archivo CSV con etiquetas disponibles")
    parser.add_argument("--output-dir", default="step4_5_cleaned_documents",
                       help="Directorio de salida para documentos limpios")
    parser.add_argument("--jsonl-file", default="examples/jsonl_data/medical_annotations.jsonl",
                       help="Archivo JSONL a reparar")
    parser.add_argument("--jsonl-output", default="examples/jsonl_data/medical_annotations_fixed.jsonl",
                       help="Archivo JSONL reparado de salida")
    parser.add_argument("--max-docs", type=int, default=None,
                       help="Máximo número de documentos a procesar")
    parser.add_argument("--skip-jsonl-fix", action="store_true",
                       help="Omitir la reparación del JSONL")
    
    args = parser.parse_args()
    
    debug_print("=== STEP 4.5: LIMPIEZA AVANZADA DE NOMBRES DE ETIQUETAS + REPARACIÓN JSONL ===", "INFO")
    debug_print(f"Directorio de entrada: {args.input_dir}", "INFO")
    debug_print(f"Directorio de salida: {args.output_dir}", "INFO")
    debug_print(f"Archivo JSONL: {args.jsonl_file}", "INFO")
    debug_print(f"Archivo JSONL de salida: {args.jsonl_output}", "INFO")
    
    # Verificar que existe el directorio de entrada
    if not os.path.exists(args.input_dir):
        print(f"ERROR: No existe el directorio de entrada: {args.input_dir}")
        return
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar etiquetas disponibles
    debug_print("Cargando etiquetas disponibles...", "INFO")
    available_labels = load_available_labels(args.labels_file)
    debug_print(f"Cargadas {len(available_labels)} etiquetas disponibles", "INFO")
    
    # Encontrar documentos a procesar
    input_path = Path(args.input_dir)
    documents = list(input_path.glob("*.txt"))
    
    # Filtrar solo documentos válidos (excluyendo archivos de reporte)
    valid_documents = []
    for doc in documents:
        doc_id = doc.stem
        if not doc_id.endswith('_summary'):
            valid_documents.append(doc_id)
    
    if args.max_docs:
        valid_documents = valid_documents[:args.max_docs]
    
    debug_print(f"Encontrados {len(valid_documents)} documentos a limpiar", "INFO")
    
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
            executor.submit(process_documents_batch, batch, args.input_dir, args.output_dir, available_labels): i 
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
    
    # Reparar archivo JSONL si no se omite
    jsonl_fix_result = None
    if not args.skip_jsonl_fix:
        debug_print("Iniciando reparación del archivo JSONL...", "INFO")
        jsonl_fix_result = fix_jsonl_file(args.jsonl_file, args.jsonl_output, args.output_dir)
        
        if jsonl_fix_result.get("success"):
            debug_print("JSONL reparado exitosamente", "INFO")
        else:
            debug_print(f"Error reparando JSONL: {jsonl_fix_result.get('error', 'Error desconocido')}", "ERROR")
    else:
        debug_print("Reparación JSONL omitida por --skip-jsonl-fix", "INFO")
    
    # Crear reporte resumen
    create_summary_report(results, args.output_dir, jsonl_fix_result)
    
    debug_print("Step 4.5 completado! Documentos limpios y JSONL reparado.", "INFO")

if __name__ == "__main__":
    main()
