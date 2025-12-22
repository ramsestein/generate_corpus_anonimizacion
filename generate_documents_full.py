#!/usr/bin/env python3
"""
generate_documents_full.py - Genera JSON completo de documentos para análisis
=============================================================================

Lee todos los documentos .txt del corpus ANTIGUO y crea un JSON con:
- doc_id: ID del documento (UUID)
- text: Contenido COMPLETO del documento
- text_length: Longitud en caracteres
- word_count: Número de palabras
- words: Lista de todas las palabras extraídas
- words_sample: Primeras 50 palabras
"""

import json
from pathlib import Path
from typing import Dict, List
import re
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def extract_entities(text: str) -> List[str]:
    """Extrae solo las entidades marcadas con [** **] (ground truth)."""
    # Encontrar todos los matches de [** ... **]
    entities = re.findall(r'\[\*\*([^\*]+)\*\*\]', text)
    # Limpiar y normalizar
    entities = [ent.strip() for ent in entities if ent.strip()]
    return entities


def extract_words(text: str) -> List[str]:
    """Extrae palabras significativas del texto (sin marcas de redacción)."""
    # Remover marcas de redacción [** **]
    text_clean = re.sub(r'\[\*\*([^\*]+)\*\*\]', r'\1', text)
    # Remover URLs, emails
    text_clean = re.sub(r'https?://\S+', '', text_clean)
    text_clean = re.sub(r'\S+@\S+', '', text_clean)
    # Obtener palabras (solo letras)
    words = re.findall(r'\b[a-záéíóúñàèìòùâêîôûäëïöüçA-Z]+\b', text_clean.lower())
    return words


def main():
    docs_dir = Path("corpus/ANTIGUO/documents")
    
    # Crear estructura JSON
    output = {
        'metadata': {
            'corpus': 'ANTIGUO',
            'total_documents': 0,
            'total_words': 0,
            'avg_words_per_doc': 0,
            'generation_timestamp': ''
        },
        'documents': {}
    }
    
    documents = output['documents']
    total_files = 0
    total_words = 0
    
    logger.info(f"Leyendo documentos desde {docs_dir.absolute()}")
    
    # Obtener lista completa de archivos
    txt_files = sorted(docs_dir.glob("*.txt"))
    logger.info(f"Total de archivos encontrados: {len(txt_files)}")
    
    # Procesar TODOS los archivos (no solo primeros 100)
    for idx, txt_file in enumerate(txt_files, 1):
        doc_id = txt_file.stem  # UUID sin .txt
        
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Extraer entidades (ground truth con [** **])
            entities = extract_entities(text)
            
            # Extraer palabras (texto limpio)
            words = extract_words(text)
            total_words += len(words)
            
            documents[doc_id] = {
                'doc_id': doc_id,
                'text_length': len(text),
                'word_count': len(words),
                'entity_count': len(entities),
                'entities': entities,  # SOLO ENTIDADES MARCADAS [** **]
                'text': text,  # TEXTO COMPLETO
                'words': words,  # TODAS LAS PALABRAS
                'words_sample': words[:50],  # Primeras 50 palabras
                'entities_sample': entities[:20],  # Primeras 20 entidades
            }
            
            total_files += 1
            
            # Mostrar progreso cada 1000 documentos
            if idx % 1000 == 0:
                logger.info(f"  {idx}/{len(txt_files)} documentos procesados...")
        
        except Exception as e:
            logger.error(f"Error leyendo {txt_file}: {e}")
    
    # Actualizar metadata
    avg_words = total_words // total_files if total_files > 0 else 0
    output['metadata']['total_documents'] = total_files
    output['metadata']['total_words'] = total_words
    output['metadata']['avg_words_per_doc'] = avg_words
    output['metadata']['generation_timestamp'] = __import__('datetime').datetime.now().isoformat()
    
    # Guardar JSON
    output_path = Path("outputs/documentos_corpus_antiguo_completo.json")
    output_path.parent.mkdir(exist_ok=True)
    
    logger.info(f"Guardando JSON en {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Mostrar resumen
    logger.info(f"\n{'='*70}")
    logger.info(f"RESUMEN DE PROCESAMIENTO")
    logger.info(f"{'='*70}")
    logger.info(f"Documentos procesados: {total_files}")
    logger.info(f"Palabras totales: {total_words:,}")
    logger.info(f"Promedio de palabras por documento: {avg_words}")
    logger.info(f"Archivo guardado: {output_path}")
    logger.info(f"{'='*70}\n")


if __name__ == '__main__':
    main()
