#!/usr/bin/env python3
"""
STEP 2 DEL PIPELINE: Limpieza del archivo JSONL
Limpia el archivo medical_annotations.jsonl eliminando:
1. Etiquetas duplicadas con el mismo texto exacto
2. Etiquetas con textos inválidos como "no tengo respuesta", "insertar referencia", etc.
"""

import json
import argparse
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

def is_invalid_text(text: str) -> bool:
    """
    Verifica si un texto es inválido y debe ser eliminado.
    
    Args:
        text (str): Texto a verificar
        
    Returns:
        bool: True si el texto es inválido
    """
    if not text or not text.strip():
        return True
    
    text_lower = text.lower().strip()
    
    # Patrones de textos inválidos
    invalid_patterns = [
        # Respuestas de IA que no son datos reales
        r"no tengo respuesta",
        r"no puedo proporcionar",
        r"insertar referencia",
        r"insertar aquí",
        r"ejemplo de",
        r"valor de ejemplo",
        r"texto de ejemplo",
        r"información no disponible",
        r"dato no especificado",
        r"pendiente de completar",
        r"a completar",
        r"por definir",
        r"sin especificar",
        
        # Respuestas genéricas de IA
        r"lo siento",
        r"disculpa",
        r"no es posible",
        r"no está disponible",
        r"consulte con",
        r"contacte con",
        
        # Textos placeholder
        r"xxx+",
        r"placeholder",
        r"temporal",
        r"borrador",
        
        # Respuestas evasivas
        r"según corresponda",
        r"según proceda",
        r"en función de",
        r"dependiendo de",
        
        # Textos muy cortos o genéricos
        r"^n/a$",
        r"^na$",
        r"^-+$",
        r"^\.*$",
        r"^_+$"
    ]
    
    # Verificar cada patrón
    for pattern in invalid_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Verificar si es solo números o caracteres especiales
    if len(text_lower) <= 2 and not text_lower.isalnum():
        return True
    
    # Verificar si contiene solo caracteres repetidos
    if len(set(text_lower.replace(' ', ''))) <= 1:
        return True
    
    return False

def clean_single_entry(entry: Dict) -> Dict:
    """
    Limpia una entrada individual del JSONL.
    
    Args:
        entry (Dict): Entrada del JSONL
        
    Returns:
        Dict: Entrada limpia con estadísticas
    """
    if 'data' not in entry or not isinstance(entry['data'], list):
        return {
            'cleaned_entry': entry,
            'removed_duplicates': 0,
            'removed_invalid': 0,
            'original_count': 0,
            'final_count': 0
        }
    
    original_entities = entry['data']
    original_count = len(original_entities)
    
    # Paso 1: Eliminar textos inválidos
    valid_entities = []
    removed_invalid = 0
    
    for entity in original_entities:
        text = entity.get('text', '').strip()
        if not is_invalid_text(text):
            valid_entities.append(entity)
        else:
            removed_invalid += 1
    
    # Paso 2: Eliminar duplicados exactos (misma etiqueta + mismo texto)
    unique_entities = []
    seen_combinations = set()
    removed_duplicates = 0
    
    for entity in valid_entities:
        etiqueta = entity.get('entity', '')
        text = entity.get('text', '').strip()
        
        # Crear clave única para la combinación etiqueta+texto
        combination_key = (etiqueta, text)
        
        if combination_key not in seen_combinations:
            unique_entities.append(entity)
            seen_combinations.add(combination_key)
        else:
            removed_duplicates += 1
    
    final_count = len(unique_entities)
    
    # Crear entrada limpia
    cleaned_entry = {
        'id': entry.get('id', ''),
        'data': unique_entities
    }
    
    return {
        'cleaned_entry': cleaned_entry,
        'removed_duplicates': removed_duplicates,
        'removed_invalid': removed_invalid,
        'original_count': original_count,
        'final_count': final_count
    }

def process_entries_batch(entries_batch: List[Dict]) -> List[Dict]:
    """
    Procesa un lote de entradas en paralelo.
    
    Args:
        entries_batch (List[Dict]): Lote de entradas a procesar
        
    Returns:
        List[Dict]: Resultados de limpieza del lote
    """
    results = []
    for entry in entries_batch:
        clean_result = clean_single_entry(entry)
        results.append(clean_result)
    return results

def clean_jsonl_file(input_file: str, output_file: str = None, num_workers: int = None) -> Dict:
    """
    Limpia todo el archivo JSONL.
    
    Args:
        input_file (str): Archivo JSONL de entrada
        output_file (str): Archivo JSONL de salida (si None, sobrescribe el original)
        
    Returns:
        Dict: Estadísticas de la limpieza
    """
    if output_file is None:
        output_file = input_file
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Máximo 8 workers
    
    debug_print(f"Limpieza paralela con {num_workers} workers", "INFO")
    debug_print(f"Archivo de entrada: {input_file}", "INFO")
    debug_print(f"Archivo de salida: {output_file}", "INFO")
    
    # Cargar todas las entradas
    debug_print("Cargando entradas del JSONL...", "INFO")
    entries = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        debug_print(f"Error parseando línea {line_num}: {e}", "ERROR")
                        continue
    except FileNotFoundError:
        debug_print(f"Error: No se encontró el archivo {input_file}", "ERROR")
        return {"error": f"Archivo no encontrado: {input_file}"}
    
    debug_print(f"Cargadas {len(entries)} entradas", "INFO")
    
    # Dividir en lotes para procesamiento paralelo
    batch_size = max(1, len(entries) // (num_workers * 4))  # 4 lotes por worker
    batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]
    
    debug_print(f"Dividido en {len(batches)} lotes de ~{batch_size} entradas cada uno", "INFO")
    
    # Procesar en paralelo
    debug_print("Iniciando procesamiento paralelo...", "INFO")
    all_clean_results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Enviar lotes a workers
        future_to_batch = {executor.submit(process_entries_batch, batch): i for i, batch in enumerate(batches)}
        
        # Recopilar resultados
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_clean_results.extend(batch_results)
                debug_print(f"  Lote {batch_idx + 1}/{len(batches)} completado", "DEBUG")
            except Exception as e:
                debug_print(f"Error procesando lote {batch_idx}: {e}", "ERROR")
    
    # Compilar estadísticas y entradas limpias
    debug_print("Compilando resultados...", "INFO")
    stats = {
        'total_entries_processed': len(entries),
        'total_entries_kept': 0,
        'total_entities_original': 0,
        'total_entities_final': 0,
        'total_duplicates_removed': 0,
        'total_invalid_removed': 0,
        'entries_with_changes': 0,
        'entries_completely_empty': 0,
        'detailed_stats': []
    }
    
    cleaned_entries = []
    
    for clean_result in all_clean_results:
        # Actualizar estadísticas
        stats['total_entities_original'] += clean_result['original_count']
        stats['total_entities_final'] += clean_result['final_count']
        stats['total_duplicates_removed'] += clean_result['removed_duplicates']
        stats['total_invalid_removed'] += clean_result['removed_invalid']
        
        # Verificar si hubo cambios
        if clean_result['removed_duplicates'] > 0 or clean_result['removed_invalid'] > 0:
            stats['entries_with_changes'] += 1
        
        # Verificar si la entrada quedó vacía
        if clean_result['final_count'] == 0:
            stats['entries_completely_empty'] += 1
        else:
            cleaned_entries.append(clean_result['cleaned_entry'])
            stats['total_entries_kept'] += 1
    
    # Escribir archivo limpio
    debug_print(f"Escribiendo archivo limpio con {len(cleaned_entries)} entradas...", "INFO")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in cleaned_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        debug_print(f"Error escribiendo archivo limpio: {e}", "ERROR")
        return {"error": f"Error escribiendo archivo: {e}"}
    
    debug_print("Limpieza completada!", "INFO")
    return stats

def show_cleaning_summary(stats: Dict):
    """Muestra un resumen de la limpieza realizada"""
    
    print(f"\n{'='*60}")
    print(f"STEP 2 - LIMPIEZA DE JSONL COMPLETADA")
    print(f"{'='*60}")
    print(f"Entradas procesadas: {stats['total_entries_processed']}")
    print(f"Entradas conservadas: {stats['total_entries_kept']}")
    print(f"Entradas eliminadas (vacías): {stats['entries_completely_empty']}")
    print(f"Entradas con cambios: {stats['entries_with_changes']}")
    print(f"")
    print(f"ENTIDADES:")
    print(f"  - Entidades originales: {stats['total_entities_original']}")
    print(f"  - Entidades finales: {stats['total_entities_final']}")
    print(f"  - Duplicados eliminados: {stats['total_duplicates_removed']}")
    print(f"  - Textos inválidos eliminados: {stats['total_invalid_removed']}")
    print(f"  - Total eliminaciones: {stats['total_duplicates_removed'] + stats['total_invalid_removed']}")
    
    if stats['total_entities_original'] > 0:
        reduction_rate = ((stats['total_entities_original'] - stats['total_entities_final']) / stats['total_entities_original']) * 100
        print(f"  - Tasa de reducción: {reduction_rate:.1f}%")
    
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="STEP 2: Limpiar archivo JSONL eliminando duplicados e inválidos")
    parser.add_argument("--input-file", default="examples/jsonl_data/medical_annotations.jsonl",
                       help="Archivo JSONL de entrada")
    parser.add_argument("--output-file", default=None,
                       help="Archivo JSONL de salida (si no se especifica, sobrescribe el original)")
    parser.add_argument("--backup", action="store_true",
                       help="Crear backup del archivo original antes de limpiar")
    parser.add_argument("--workers", type=int, default=None,
                       help="Número de workers paralelos (default: CPU cores, máx 8)")
    
    args = parser.parse_args()
    
    debug_print("=== STEP 2: LIMPIEZA DE ARCHIVO JSONL ===", "INFO")
    debug_print(f"Archivo de entrada: {args.input_file}", "INFO")
    
    # Verificar que existe el archivo de entrada
    if not Path(args.input_file).exists():
        print(f"ERROR: No existe el archivo de entrada: {args.input_file}")
        return
    
    # Crear backup si se solicita
    if args.backup:
        backup_file = f"{args.input_file}.backup"
        debug_print(f"Creando backup: {backup_file}", "INFO")
        try:
            import shutil
            shutil.copy2(args.input_file, backup_file)
            debug_print(f"Backup creado exitosamente", "INFO")
        except Exception as e:
            debug_print(f"Error creando backup: {e}", "WARN")
    
    # Limpiar archivo con paralelización
    stats = clean_jsonl_file(args.input_file, args.output_file, args.workers)
    
    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        return
    
    # Mostrar resumen
    show_cleaning_summary(stats)
    
    debug_print("Step 2 completado! Procede con step3 para anonimizar o step4 para generar documentos.", "INFO")

if __name__ == "__main__":
    main()
