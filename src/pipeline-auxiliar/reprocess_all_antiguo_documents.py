#!/usr/bin/env python3
"""
Script para procesar TODOS los documentos de corpus/ANTIGUO/documents
con los modelos MEDDOCAN y CARMEN NER y generar un CSV completo.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import datetime
from tqdm import tqdm

def debug_print(message: str, level: str = "INFO"):
    """Función para imprimir mensajes de debug con timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def setup_models():
    """
    Configura y carga ambos modelos BSC.
    
    Returns:
        Tuple: (pipeline_meddocan, pipeline_carmen)
    """
    debug_print("🔧 Cargando modelos BSC (MEDDOCAN + CARMEN)...", "INFO")
    
    # Configurar device
    device = 0 if torch.cuda.is_available() else -1
    debug_print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}", "DEBUG")
    
    # Modelo MEDDOCAN
    debug_print("  - Cargando bsc-bio-ehr-es-meddocan...", "DEBUG")
    meddocan_model_path = "models/bsc-bio-ehr-es-meddocan"
    
    meddocan_tokenizer = AutoTokenizer.from_pretrained(meddocan_model_path)
    meddocan_model = AutoModelForTokenClassification.from_pretrained(
        meddocan_model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map=None
    )
    
    if device >= 0:
        meddocan_model = meddocan_model.cuda()
    
    pipe_meddocan = pipeline(
        "token-classification",
        model=meddocan_model,
        tokenizer=meddocan_tokenizer,
        aggregation_strategy="simple",
        device=device
    )
    
    # Modelo CARMEN
    debug_print("  - Cargando bsc-bio-ehr-es-carmen-anon...", "DEBUG")
    carmen_model_path = "models/bsc-bio-ehr-es-carmen-anon"
    
    carmen_tokenizer = AutoTokenizer.from_pretrained(carmen_model_path)
    carmen_model = AutoModelForTokenClassification.from_pretrained(
        carmen_model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map=None
    )
    
    if device >= 0:
        carmen_model = carmen_model.cuda()
    
    pipe_carmen = pipeline(
        "token-classification",
        model=carmen_model,
        tokenizer=carmen_tokenizer,
        aggregation_strategy="simple",
        device=device
    )
    
    debug_print("✅ Modelos cargados correctamente", "INFO")
    return pipe_meddocan, pipe_carmen

def process_document(doc_id: str, text: str, pipe_meddocan, pipe_carmen) -> List[Dict]:
    """
    Procesa un documento con ambos modelos NER.
    
    Returns:
        Lista de diccionarios con las detecciones
    """
    detections = []
    
    # Procesar con MEDDOCAN
    try:
        entities_meddocan = pipe_meddocan(text)
        for entity in entities_meddocan:
            detections.append({
                'doc_id': doc_id,
                'entity': entity['word'],
                'label': entity['entity_group'],
                'start': entity['start'],
                'end': entity['end'],
                'score': entity['score'],
                'model': 'meddocan'
            })
    except Exception as e:
        debug_print(f"Error en MEDDOCAN para {doc_id}: {e}", "ERROR")
    
    # Procesar con CARMEN
    try:
        entities_carmen = pipe_carmen(text)
        for entity in entities_carmen:
            detections.append({
                'doc_id': doc_id,
                'entity': entity['word'],
                'label': entity['entity_group'],
                'start': entity['start'],
                'end': entity['end'],
                'score': entity['score'],
                'model': 'carmen'
            })
    except Exception as e:
        debug_print(f"Error en CARMEN para {doc_id}: {e}", "ERROR")
    
    return detections

def main():
    # Rutas
    documents_dir = Path("corpus/ANTIGUO/documents")
    output_csv = Path("src/pipeline-auxiliar/step6_validation_judgeLLM/detecciones_detalladas_completo.csv")
    
    debug_print("🚀 Iniciando procesamiento de TODOS los documentos ANTIGUO", "INFO")
    
    # Cargar modelos
    pipe_meddocan, pipe_carmen = setup_models()
    
    # Obtener lista de todos los documentos
    doc_files = sorted(documents_dir.glob("*.txt"))
    debug_print(f"📄 Total de documentos encontrados: {len(doc_files)}", "INFO")
    
    # Procesar todos los documentos
    all_detections = []
    
    for doc_file in tqdm(doc_files, desc="Procesando documentos"):
        doc_id = doc_file.stem
        
        try:
            # Leer documento
            with open(doc_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Procesar con ambos modelos
            detections = process_document(doc_id, text, pipe_meddocan, pipe_carmen)
            all_detections.extend(detections)
            
        except Exception as e:
            debug_print(f"Error procesando {doc_id}: {e}", "ERROR")
            continue
    
    # Guardar CSV
    debug_print(f"💾 Guardando {len(all_detections)} detecciones en CSV...", "INFO")
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['doc_id', 'entity', 'label', 'start', 'end', 'score', 'model']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_detections)
    
    debug_print(f"✅ Procesamiento completado", "INFO")
    debug_print(f"  - Documentos procesados: {len(doc_files)}", "INFO")
    debug_print(f"  - Total detecciones: {len(all_detections)}", "INFO")
    debug_print(f"  - CSV guardado en: {output_csv}", "INFO")
    
    # Estadísticas por modelo
    meddocan_count = sum(1 for d in all_detections if d['model'] == 'meddocan')
    carmen_count = sum(1 for d in all_detections if d['model'] == 'carmen')
    debug_print(f"  - MEDDOCAN: {meddocan_count} detecciones", "INFO")
    debug_print(f"  - CARMEN: {carmen_count} detecciones", "INFO")
    
    # Estadísticas por etiqueta
    from collections import Counter
    label_counts = Counter(d['label'] for d in all_detections)
    debug_print(f"\n📊 Top 10 etiquetas más frecuentes:", "INFO")
    for label, count in label_counts.most_common(10):
        debug_print(f"  {label}: {count}", "INFO")

if __name__ == "__main__":
    main()
