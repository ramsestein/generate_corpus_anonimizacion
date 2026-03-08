#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2.5 DEL PIPELINE: Limpieza semántica con LabSE
Usa el modelo LabSE para comparar semánticamente todos los textos de cada tipo de etiqueta.
Elimina el porcentaje más diferente de la media semántica (5% por defecto).
"""

import json
import sys
import os

# Configurar codificación UTF-8 para Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def debug_print(message: str, level: str = "INFO"):
    """Función para imprimir mensajes de debug con timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def load_labse_model(model_path: str = "models/labse") -> SentenceTransformer:
    """
    Carga un modelo de embeddings semánticos (usando all-MiniLM-L6-v2 como alternativa estable).
    
    Args:
        model_path (str): Ruta al modelo (no se usa actualmente)
        
    Returns:
        SentenceTransformer: Modelo de embeddings cargado
    """
    debug_print("Cargando modelo de embeddings semánticos...", "INFO")
    
    # Usar all-MiniLM-L6-v2 que es más estable y rápido que LaBSE
    try:
        debug_print("Cargando all-MiniLM-L6-v2 (modelo más estable)...", "INFO")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        debug_print("Modelo de embeddings cargado exitosamente", "INFO")
        return model
    except Exception as e:
        debug_print(f"Error cargando all-MiniLM-L6-v2: {e}", "ERROR")
        # Como alternativa, intentar con LaBSE
        try:
            debug_print("Intentando con LaBSE...", "INFO")
            model = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
            debug_print("Modelo LaBSE cargado exitosamente", "INFO")
            return model
        except Exception as e2:
            debug_print(f"Error final: {e2}", "ERROR")
            raise e2

def group_entities_by_label(entries: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Agrupa todas las entidades por tipo de etiqueta.
    
    Args:
        entries (List[Dict]): Lista de entradas del JSONL
        
    Returns:
        Dict[str, List[Dict]]: Diccionario con etiquetas como claves y listas de entidades como valores
    """
    entities_by_label = {}
    total_entities = 0
    
    for entry in entries:
        entry_id = entry.get('id', 'unknown')
        entities = entry.get('data', [])
        
        for entity in entities:
            etiqueta = entity.get('entity', '')
            text = entity.get('text', '').strip()
            
            if etiqueta and text:
                total_entities += 1
                if etiqueta not in entities_by_label:
                    entities_by_label[etiqueta] = []
                
                entities_by_label[etiqueta].append({
                    'entry_id': entry_id,
                    'entity': etiqueta,
                    'text': text,
                    'original_entity': entity
                })
    
    debug_print(f"Total entidades agrupadas: {total_entities}", "DEBUG")
    debug_print(f"Etiquetas encontradas: {list(entities_by_label.keys())}", "DEBUG")
    
    return entities_by_label

def calculate_semantic_outliers(texts: List[str], model: SentenceTransformer, 
                               outlier_percentage: float = 0.05) -> List[int]:
    """
    Calcula qué textos son outliers semánticos usando LabSE.
    
    Args:
        texts (List[str]): Lista de textos a analizar
        model (SentenceTransformer): Modelo LabSE
        outlier_percentage (float): Porcentaje de outliers a identificar
        
    Returns:
        List[int]: Índices de los textos que son outliers
    """
    if len(texts) <= 2:
        return []  # No eliminar nada si hay muy pocos textos
    
    debug_print(f"    Calculando embeddings para {len(texts)} textos...", "DEBUG")
    
    # Generar embeddings
    embeddings = model.encode(texts, show_progress_bar=False)
    
    # Calcular matriz de similitud coseno
    similarity_matrix = cosine_similarity(embeddings)
    
    # Para cada texto, calcular su similitud promedio con todos los demás
    avg_similarities = []
    for i in range(len(texts)):
        # Excluir la similitud consigo mismo (que es 1.0)
        similarities_with_others = [similarity_matrix[i][j] for j in range(len(texts)) if i != j]
        avg_similarity = np.mean(similarities_with_others) if similarities_with_others else 0.0
        avg_similarities.append(avg_similarity)
    
    # Identificar outliers (los que tienen menor similitud promedio)
    num_outliers = max(1, int(len(texts) * outlier_percentage))
    
    # Obtener índices de los textos con menor similitud promedio
    outlier_indices = np.argsort(avg_similarities)[:num_outliers]
    
    debug_print(f"    Identificados {len(outlier_indices)} outliers semánticos", "DEBUG")
    
    return outlier_indices.tolist()

def process_label_batch(label_entities_pair: Tuple[str, List[Dict]], model_path: str, outlier_percentage: float) -> Tuple[str, Dict, Set]:
    """
    Procesa un tipo de etiqueta en paralelo.
    
    Args:
        label_entities_pair: Tupla (etiqueta, lista_entidades)
        model_path: Ruta al modelo LabSE
        outlier_percentage: Porcentaje de outliers
        
    Returns:
        Tuple: (etiqueta, resultado_limpieza, entidades_a_eliminar)
    """
    etiqueta, entities = label_entities_pair
    
    # Cargar modelo en este worker con configuración segura
    try:
        # Usar all-MiniLM-L6-v2 que es más estable
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    except Exception as e:
        # Si falla, intentar con LaBSE
        model = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
    
    entities_to_remove_local = set()
    
    if len(entities) <= 2:
        result = {
            'original_count': len(entities),
            'outliers_removed': 0,
            'final_count': len(entities),
            'outlier_texts': []
        }
        return etiqueta, result, entities_to_remove_local
    
    # Extraer textos únicos
    unique_texts = list(set([entity['text'] for entity in entities]))
    
    if len(unique_texts) <= 2:
        result = {
            'original_count': len(entities),
            'outliers_removed': 0,
            'final_count': len(entities),
            'outlier_texts': []
        }
        return etiqueta, result, entities_to_remove_local
    
    # Calcular outliers semánticos
    outlier_indices = calculate_semantic_outliers(unique_texts, model, outlier_percentage)
    outlier_texts = [unique_texts[i] for i in outlier_indices]
    
    # Marcar entidades para eliminación
    removed_count = 0
    for entity in entities:
        if entity['text'] in outlier_texts:
            entities_to_remove_local.add((entity['entry_id'], entity['entity'], entity['text']))
            removed_count += 1
    
    result = {
        'original_count': len(entities),
        'unique_texts': len(unique_texts),
        'outliers_removed': removed_count,
        'final_count': len(entities) - removed_count,
        'outlier_texts': outlier_texts,
        'outlier_percentage_actual': (removed_count / len(entities)) * 100 if entities else 0
    }
    
    return etiqueta, result, entities_to_remove_local

def clean_entities_by_semantic_similarity(entities_by_label: Dict[str, List[Dict]], 
                                        model_path: str,
                                        outlier_percentage: float = 0.05,
                                        num_workers: int = None) -> Tuple[Dict[str, Dict], Set]:
    """
    Limpia entidades usando similitud semántica por tipo de etiqueta.
    
    Args:
        entities_by_label (Dict): Entidades agrupadas por etiqueta
        model_path (str): Ruta al modelo LabSE
        outlier_percentage (float): Porcentaje de outliers a eliminar
        num_workers (int): Número de workers paralelos
        
    Returns:
        Tuple[Dict[str, Dict], Set]: Resultados de la limpieza y entidades a eliminar
    """
    if num_workers is None:
        num_workers = min(len(entities_by_label), 8)  # Máximo 8 workers o número de etiquetas
    
    debug_print(f"Análisis semántico paralelo con {num_workers} workers", "INFO")
    debug_print(f"Analizando {len(entities_by_label)} tipos de etiquetas...", "INFO")
    
    cleaning_results = {}
    all_entities_to_remove = set()
    
    # Preparar datos para procesamiento paralelo
    label_entity_pairs = list(entities_by_label.items())
    
    # Procesar etiquetas en paralelo usando ThreadPoolExecutor (mejor para I/O intensivo)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Enviar trabajos
        future_to_label = {
            executor.submit(process_label_batch, pair, model_path, outlier_percentage): pair[0] 
            for pair in label_entity_pairs
        }
        
        # Recopilar resultados
        completed = 0
        for future in as_completed(future_to_label):
            etiqueta = future_to_label[future]
            try:
                label_name, result, entities_to_remove_local = future.result()
                cleaning_results[label_name] = result
                all_entities_to_remove.update(entities_to_remove_local)
                
                completed += 1
                debug_print(f"  Completado: {etiqueta} ({completed}/{len(label_entity_pairs)})", "DEBUG")
                
                if result['outliers_removed'] > 0:
                    debug_print(f"    {etiqueta}: {result['outliers_removed']} outliers eliminados", "DEBUG")
                
            except Exception as e:
                debug_print(f"Error procesando etiqueta {etiqueta}: {e}", "ERROR")
                # Añadir resultado vacío para no fallar
                cleaning_results[etiqueta] = {
                    'original_count': 0,
                    'outliers_removed': 0,
                    'final_count': 0,
                    'outlier_texts': []
                }
    
    debug_print(f"Procesamiento paralelo completado", "INFO")
    return cleaning_results, all_entities_to_remove

def apply_semantic_cleaning(entries: List[Dict], entities_to_remove: Set[Tuple]) -> List[Dict]:
    """
    Aplica la limpieza semántica eliminando las entidades marcadas.
    
    Args:
        entries (List[Dict]): Entradas originales del JSONL
        entities_to_remove (Set[Tuple]): Conjunto de entidades a eliminar
        
    Returns:
        List[Dict]: Entradas limpias
    """
    cleaned_entries = []
    
    for entry in entries:
        entry_id = entry.get('id', 'unknown')
        original_entities = entry.get('data', [])
        
        # Filtrar entidades que no están en la lista de eliminación
        cleaned_entities = []
        for entity in original_entities:
            etiqueta = entity.get('entity', '')
            text = entity.get('text', '').strip()
            
            removal_key = (entry_id, etiqueta, text)
            if removal_key not in entities_to_remove:
                cleaned_entities.append(entity)
        
        # Solo conservar entradas que no quedaron vacías
        if cleaned_entities:
            cleaned_entries.append({
                'id': entry_id,
                'data': cleaned_entities
            })
    
    return cleaned_entries

def semantic_clean_jsonl(input_file: str, output_file: str = None, 
                        outlier_percentage: float = 0.05, model_path: str = "models/labse", 
                        num_workers: int = None) -> Dict:
    """
    Realiza limpieza semántica completa del archivo JSONL.
    
    Args:
        input_file (str): Archivo JSONL de entrada
        output_file (str): Archivo JSONL de salida
        outlier_percentage (float): Porcentaje de outliers a eliminar
        model_path (str): Ruta al modelo LabSE
        
    Returns:
        Dict: Estadísticas de la limpieza
    """
    if output_file is None:
        output_file = input_file
    
    debug_print(f"Iniciando limpieza semántica con LabSE", "INFO")
    debug_print(f"Archivo de entrada: {input_file}", "INFO")
    debug_print(f"Porcentaje de outliers a eliminar: {outlier_percentage * 100:.1f}%", "INFO")
    
    # Cargar modelo LabSE
    try:
        model = load_labse_model(model_path)
    except Exception as e:
        return {"error": f"No se pudo cargar el modelo LabSE: {e}"}
    
    # Cargar entradas del JSONL
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
                        debug_print(f"Error parseando línea {line_num}: {e}", "WARN")
                        continue
    except FileNotFoundError:
        return {"error": f"Archivo no encontrado: {input_file}"}
    
    debug_print(f"Cargadas {len(entries)} entradas", "INFO")
    
    # Agrupar entidades por etiqueta
    debug_print("Agrupando entidades por tipo de etiqueta...", "INFO")
    entities_by_label = group_entities_by_label(entries)
    
    debug_print(f"Encontrados {len(entities_by_label)} tipos de etiquetas diferentes", "INFO")
    
    # Realizar limpieza semántica en paralelo
    debug_print("Realizando análisis semántico paralelo...", "INFO")
    cleaning_results, entities_to_remove = clean_entities_by_semantic_similarity(
        entities_by_label, model_path, outlier_percentage, num_workers
    )
    
    # Aplicar limpieza
    debug_print("Aplicando limpieza semántica...", "INFO")
    cleaned_entries = apply_semantic_cleaning(entries, entities_to_remove)
    
    # Escribir archivo limpio
    debug_print(f"Escribiendo archivo limpio con {len(cleaned_entries)} entradas...", "INFO")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in cleaned_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        return {"error": f"Error escribiendo archivo: {e}"}
    
    # Calcular estadísticas globales
    total_original_entities = sum([result['original_count'] for result in cleaning_results.values()])
    total_final_entities = sum([result['final_count'] for result in cleaning_results.values()])
    total_removed = sum([result['outliers_removed'] for result in cleaning_results.values()])
    
    global_stats = {
        'original_entries': len(entries),
        'final_entries': len(cleaned_entries),
        'entries_removed': len(entries) - len(cleaned_entries),
        'total_original_entities': total_original_entities,
        'total_final_entities': total_final_entities,
        'total_entities_removed': total_removed,
        'outlier_percentage_used': outlier_percentage,
        'reduction_rate': (total_removed / total_original_entities * 100) if total_original_entities > 0 else 0,
        'cleaning_results_by_label': cleaning_results
    }
    
    debug_print("Limpieza semántica completada!", "INFO")
    return global_stats

def show_semantic_cleaning_summary(stats: Dict):
    """Muestra un resumen de la limpieza semántica realizada"""
    
    print(f"\n{'='*60}")
    print(f"STEP 2.5 - LIMPIEZA SEMÁNTICA COMPLETADA")
    print(f"{'='*60}")
    print(f"Entradas originales: {stats['original_entries']}")
    print(f"Entradas finales: {stats['final_entries']}")
    print(f"Entradas eliminadas (vacías): {stats['entries_removed']}")
    print(f"")
    print(f"ENTIDADES:")
    print(f"  - Entidades originales: {stats['total_original_entities']}")
    print(f"  - Entidades finales: {stats['total_final_entities']}")
    print(f"  - Entidades eliminadas: {stats['total_entities_removed']}")
    print(f"  - Tasa de reducción: {stats['reduction_rate']:.1f}%")
    print(f"  - Porcentaje de outliers usado: {stats['outlier_percentage_used'] * 100:.1f}%")
    print(f"")
    print(f"LIMPIEZA POR TIPO DE ETIQUETA:")
    
    # Mostrar estadísticas por etiqueta
    for etiqueta, result in stats['cleaning_results_by_label'].items():
        if result['outliers_removed'] > 0:
            print(f"  - {etiqueta}:")
            print(f"    * Original: {result['original_count']} -> Final: {result['final_count']}")
            print(f"    * Eliminados: {result['outliers_removed']} ({result['outlier_percentage_actual']:.1f}%)")
            if result['outlier_texts']:
                outlier_preview = []
                for t in result['outlier_texts'][:3]:
                    if len(t) > 30:
                        outlier_preview.append(f"'{t[:30]}...'")
                    else:
                        outlier_preview.append(f"'{t}'")
                print(f"    * Outliers: {', '.join(outlier_preview)}")
                if len(result['outlier_texts']) > 3:
                    print(f"      ... y {len(result['outlier_texts']) - 3} más")
    
    # Mostrar etiquetas sin cambios
    unchanged_labels = [etiqueta for etiqueta, result in stats['cleaning_results_by_label'].items() 
                       if result['outliers_removed'] == 0]
    
    if unchanged_labels:
        print(f"")
        print(f"Etiquetas sin cambios ({len(unchanged_labels)}): {', '.join(unchanged_labels[:10])}")
        if len(unchanged_labels) > 10:
            print(f"... y {len(unchanged_labels) - 10} más")
    
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="STEP 2.5: Limpieza semántica del JSONL con LabSE")
    parser.add_argument("--input-file", default="examples/jsonl_data/medical_annotations.jsonl",
                       help="Archivo JSONL de entrada")
    parser.add_argument("--output-file", default=None,
                       help="Archivo JSONL de salida (si no se especifica, sobrescribe el original)")
    parser.add_argument("--outlier-percentage", type=float, default=0.05,
                       help="Porcentaje de outliers semánticos a eliminar (default: 0.05 = 5%)")
    parser.add_argument("--model-path", default="models/labse",
                       help="Ruta al modelo LabSE")
    parser.add_argument("--backup", action="store_true",
                       help="Crear backup del archivo original antes de limpiar")
    parser.add_argument("--workers", type=int, default=None,
                       help="Número de workers paralelos (default: min(etiquetas, 8))")
    
    args = parser.parse_args()
    
    debug_print("=== STEP 2.5: LIMPIEZA SEMÁNTICA CON LABSE ===", "INFO")
    debug_print(f"Archivo de entrada: {args.input_file}", "INFO")
    debug_print(f"Porcentaje de outliers: {args.outlier_percentage * 100:.1f}%", "INFO")
    
    # Verificar que existe el archivo de entrada
    if not Path(args.input_file).exists():
        print(f"ERROR: No existe el archivo de entrada: {args.input_file}")
        return
    
    # Crear backup si se solicita
    if args.backup:
        backup_file = f"{args.input_file}.semantic_backup"
        debug_print(f"Creando backup: {backup_file}", "INFO")
        try:
            import shutil
            shutil.copy2(args.input_file, backup_file)
            debug_print(f"Backup creado exitosamente", "INFO")
        except Exception as e:
            debug_print(f"Error creando backup: {e}", "WARN")
    
    # Realizar limpieza semántica
    stats = semantic_clean_jsonl(
        args.input_file, 
        args.output_file, 
        args.outlier_percentage, 
        args.model_path,
        args.workers
    )
    
    if "error" in stats:
        print(f"ERROR: {stats['error']}")
        return
    
    # Mostrar resumen
    show_semantic_cleaning_summary(stats)
    
    debug_print("Step 2.5 completado! Procede con step3 para anonimizar o step4 para generar documentos.", "INFO")

if __name__ == "__main__":
    main()
