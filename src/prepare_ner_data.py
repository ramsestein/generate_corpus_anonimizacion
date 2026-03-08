"""
Script para preparar datos de NER desde el corpus.
Convierte documentos y anotaciones JSON a formato IOB para token classification.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import re
from tqdm import tqdm

def load_document(doc_path: Path) -> str:
    """Carga un documento de texto."""
    with open(doc_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_annotations(ann_path: Path) -> List[Dict]:
    """Carga las anotaciones de un archivo JSON."""
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('data', [])

def find_entity_spans(text: str, entity_text: str) -> List[Tuple[int, int]]:
    """
    Encuentra todas las ocurrencias de entity_text en text.
    Retorna lista de tuplas (start, end).
    """
    spans = []
    # Escapar caracteres especiales para regex
    escaped_text = re.escape(entity_text)
    # Buscar todas las ocurrencias
    for match in re.finditer(escaped_text, text, re.IGNORECASE):
        spans.append((match.start(), match.end()))
    return spans

def align_entities_to_text(text: str, annotations: List[Dict]) -> List[Tuple[int, int, str]]:
    """
    Alinea las entidades anotadas con el texto.
    Retorna lista de (start, end, entity_type) ordenada por posición.
    """
    entities = []
    for ann in annotations:
        entity_text = ann['text']
        entity_type = ann['entity']
        
        # Buscar todas las ocurrencias
        spans = find_entity_spans(text, entity_text)
        for start, end in spans:
            entities.append((start, end, entity_type))
    
    # Ordenar por posición de inicio
    entities.sort(key=lambda x: x[0])
    
    # Eliminar solapamientos (mantener la primera ocurrencia)
    filtered_entities = []
    last_end = 0
    for start, end, entity_type in entities:
        if start >= last_end:
            filtered_entities.append((start, end, entity_type))
            last_end = end
    
    return filtered_entities

def create_iob_labels(text: str, entities: List[Tuple[int, int, str]], 
                     tokenizer, label2id: Dict[str, int]) -> Tuple[List[str], List[int]]:
    """
    Convierte texto y entidades a formato IOB alineado con palabras.
    Retorna (tokens, labels) donde labels son IDs numéricos.
    """
    # Dividir texto en palabras (tokens simples)
    words = text.split()
    labels = ['O'] * len(words)
    
    # Crear mapeo de posición de caracter a palabra
    char_to_word = {}
    char_pos = 0
    for word_idx, word in enumerate(words):
        for char in word + ' ':  # Incluir espacio después
            char_to_word[char_pos] = word_idx
            char_pos += 1
    
    # Mapear cada entidad a palabras
    for entity_start, entity_end, entity_type in entities:
        # Encontrar palabras que se solapan con la entidad
        word_indices = set()
        for char_pos in range(entity_start, entity_end):
            if char_pos in char_to_word:
                word_indices.add(char_to_word[char_pos])
        
        word_indices = sorted(word_indices)
        
        if word_indices:
            # Asignar B- al primer token e I- a los siguientes
            labels[word_indices[0]] = f'B-{entity_type}'
            for idx in word_indices[1:]:
                labels[idx] = f'I-{entity_type}'
    
    # Convertir labels a IDs
    label_ids = []
    for label in labels:
        label_id = label2id.get(label, label2id['O'])
        label_ids.append(label_id)
    
    # Retornar palabras como tokens (serán retokenizadas después)
    return words, label_ids

def prepare_dataset(documents_dir: Path, entities_dir: Path, 
                   tokenizer, label2id: Dict[str, int],
                   output_file: Path = None) -> List[Dict]:
    """
    Prepara el dataset completo desde los directorios de documentos y entidades.
    """
    dataset = []
    doc_files = list(documents_dir.glob('*.txt'))
    
    print(f"Procesando {len(doc_files)} documentos...")
    
    for doc_path in tqdm(doc_files):
        doc_id = doc_path.stem
        ann_path = entities_dir / f"{doc_id}.json"
        
        if not ann_path.exists():
            continue
        
        try:
            # Cargar documento y anotaciones
            text = load_document(doc_path)
            annotations = load_annotations(ann_path)
            
            if not annotations:
                continue
            
            # Alinear entidades
            entities = align_entities_to_text(text, annotations)
            
            if not entities:
                continue
            
            # Crear labels IOB
            tokens, label_ids = create_iob_labels(text, entities, tokenizer, label2id)
            
            # Agregar al dataset (guardamos labels como lista de IDs)
            dataset.append({
                'id': doc_id,
                'text': text,
                'labels': label_ids,  # Labels como lista de IDs numéricos
                'entities': entities
            })
            
        except Exception as e:
            print(f"Error procesando {doc_id}: {e}")
            continue
    
    print(f"Dataset preparado: {len(dataset)} ejemplos")
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Dataset guardado en {output_file}")
    
    return dataset

if __name__ == "__main__":
    # Ejemplo de uso
    from transformers import AutoTokenizer
    
    documents_dir = Path("corpus/documents")
    entities_dir = Path("corpus/entidades")
    
    # Cargar tokenizer base (usaremos el del modelo meddocan)
    tokenizer = AutoTokenizer.from_pretrained("models/bsc-bio-ehr-es-meddocan")
    
    # Cargar label2id del modelo
    import json
    with open("models/bsc-bio-ehr-es-meddocan/config.json", 'r') as f:
        config = json.load(f)
        label2id = config['label2id']
    
    # Preparar dataset
    dataset = prepare_dataset(documents_dir, entities_dir, tokenizer, label2id,
                             output_file=Path("corpus/ner_dataset.json"))
    
    print(f"\nDataset listo con {len(dataset)} ejemplos")

