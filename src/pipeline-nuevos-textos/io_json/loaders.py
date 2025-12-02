#!/usr/bin/env python3
"""
Cargadores de datos - Funciones para leer archivos.

Soporta múltiples formatos: JSON, CSV, Excel, texto plano.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def load_json(file_path: str) -> Any:
    """
    Carga un archivo JSON.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        Contenido del JSON
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        json.JSONDecodeError: Si el JSON es inválido
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv(
    file_path: str,
    delimiter: str = ',',
    encoding: str = 'utf-8'
) -> List[Dict[str, Any]]:
    """
    Carga un archivo CSV como lista de diccionarios.
    
    Args:
        file_path: Ruta al archivo CSV
        delimiter: Delimitador de columnas
        encoding: Codificación del archivo
        
    Returns:
        Lista de diccionarios (cada fila es un dict)
    """
    try:
        import pandas as pd
        
        # Intentar múltiples delimitadores si falla
        for delim in [delimiter, ',', ';', '\t']:
            try:
                df = pd.read_csv(file_path, delimiter=delim, encoding=encoding)
                if len(df.columns) > 1:
                    return df.to_dict('records')
            except Exception:
                continue
        
        # Fallback con latin-1
        df = pd.read_csv(file_path, encoding='latin-1')
        return df.to_dict('records')
        
    except ImportError:
        # Sin pandas, usar csv
        import csv
        
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return list(reader)


def load_excel(file_path: str, sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Carga un archivo Excel como lista de diccionarios.
    
    Args:
        file_path: Ruta al archivo Excel
        sheet_name: Nombre de la hoja (None = primera hoja)
        
    Returns:
        Lista de diccionarios
    """
    try:
        import pandas as pd
        
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df.to_dict('records')
        
    except ImportError:
        raise ImportError("pandas y openpyxl son necesarios para leer Excel")


def load_document(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Carga el contenido de un documento de texto.
    
    Args:
        file_path: Ruta al archivo
        encoding: Codificación del archivo
        
    Returns:
        Contenido del documento como string
        
    Raises:
        FileNotFoundError: Si el archivo no existe
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    
    with open(path, 'r', encoding=encoding) as f:
        return f.read()


def load_ner_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Carga resultados NER desde JSON.
    
    Soporta múltiples formatos:
    - {"entities": [...]}
    - {"detecciones": [...]}
    - {"decisions": [...]}
    - [...]
    
    Args:
        file_path: Ruta al archivo JSON con resultados NER
        
    Returns:
        Lista de entidades
        
    Raises:
        ValueError: Si el formato no es reconocido
    """
    data = load_json(file_path)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Buscar en varias claves conocidas
        for key in ['entities', 'detecciones', 'decisions', 'results']:
            if key in data:
                return data[key]
        
        raise ValueError(
            f"Unrecognized JSON format in {file_path}. "
            "Expected 'entities', 'detecciones', 'decisions' or 'results' key"
        )
    else:
        raise ValueError(f"Unexpected data type in {file_path}: {type(data)}")


def load_entities(
    input_path: str,
    format: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Carga entidades desde cualquier formato soportado.
    
    Detecta automáticamente el formato basándose en la extensión.
    
    Args:
        input_path: Ruta al archivo de entrada
        format: Formato forzado ('json', 'csv', 'excel') o None para auto
        
    Returns:
        Lista de entidades normalizadas
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    # Detectar formato
    if format is None:
        ext = path.suffix.lower()
        if ext in ['.json']:
            format = 'json'
        elif ext in ['.csv']:
            format = 'csv'
        elif ext in ['.xls', '.xlsx']:
            format = 'excel'
        else:
            format = 'json'  # Default
    
    # Cargar según formato
    if format == 'json':
        return load_ner_results(input_path)
    elif format == 'csv':
        return load_csv(input_path)
    elif format == 'excel':
        return load_excel(input_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_documents_batch(
    docs_path: str,
    doc_ids: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Carga múltiples documentos de una carpeta.
    
    Args:
        docs_path: Ruta a la carpeta de documentos
        doc_ids: Lista de IDs de documentos a cargar (None = todos)
        
    Returns:
        Diccionario {doc_id: contenido}
    """
    docs_dir = Path(docs_path)
    
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_path}")
    
    documents = {}
    
    # Obtener archivos
    if doc_ids:
        # Solo los especificados
        for doc_id in doc_ids:
            # Buscar archivo
            for ext in ['', '.txt', '.txt.txt']:
                file_path = docs_dir / f"{doc_id}{ext}"
                if file_path.exists():
                    documents[doc_id] = load_document(str(file_path))
                    break
    else:
        # Todos los .txt
        for file_path in docs_dir.glob('*.txt'):
            doc_id = file_path.stem
            documents[doc_id] = load_document(str(file_path))
    
    logger.info(f"Loaded {len(documents)} documents from {docs_path}")
    return documents
