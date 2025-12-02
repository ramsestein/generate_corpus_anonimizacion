#!/usr/bin/env python3
"""
Guardadores de datos - Funciones para escribir archivos.

Guarda resultados en formato JSON estándar del pipeline.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def save_json(
    data: Any,
    file_path: str,
    indent: int = 2,
    ensure_ascii: bool = False
):
    """
    Guarda datos en un archivo JSON.
    
    Args:
        data: Datos a guardar
        file_path: Ruta del archivo de salida
        indent: Espacios de indentación
        ensure_ascii: Si True, escapa caracteres no-ASCII
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    
    logger.info(f"Saved JSON to {file_path}")


def save_entities(
    entities: List[Dict[str, Any]],
    output_path: str,
    include_metadata: bool = True
):
    """
    Guarda una lista de entidades en formato JSON.
    
    Alias de save_results para compatibilidad.
    
    Args:
        entities: Lista de entidades
        output_path: Ruta del archivo de salida
        include_metadata: Incluir sección de metadata
    """
    save_results(entities, output_path, include_metadata)


def save_results(
    results: List[Dict[str, Any]],
    output_path: str,
    include_metadata: bool = True
):
    """
    Guarda resultados del pipeline en formato JSON estándar.
    
    Formato de salida:
    {
        "metadata": {
            "generated_at": "2024-01-01T12:00:00",
            "total_entities": 100,
            "pipeline_version": "2.0"
        },
        "entities": [...]
    }
    
    Args:
        results: Lista de resultados (entidades procesadas)
        output_path: Ruta del archivo de salida
        include_metadata: Incluir sección de metadata
    """
    output_data = {}
    
    if include_metadata:
        output_data['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'total_entities': len(results),
            'pipeline_version': '2.0',
        }
    
    output_data['entities'] = results
    
    save_json(output_data, output_path)


def save_pipeline_results(
    decisions: List[Dict[str, Any]],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    stats: Optional[Dict[str, Any]] = None
):
    """
    Guarda resultados del pipeline con metadata y estadísticas.
    
    Formato de salida:
    {
        "metadata": {
            "generated_at": "...",
            "total_entities": 100,
            "stats": {...}
        },
        "decisions": [...]
    }
    
    Args:
        decisions: Lista de decisiones finales
        output_path: Ruta del archivo de salida
        metadata: Metadata adicional
        stats: Estadísticas del pipeline
    """
    output_data = {
        'metadata': metadata or {
            'generated_at': datetime.now().isoformat(),
            'total_entities': len(decisions)
        },
        'decisions': decisions
    }
    
    if stats:
        output_data['metadata']['stats'] = stats
    
    save_json(output_data, output_path)


def save_intermediate_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    stage_name: str,
    doc_id: Optional[str] = None
):
    """
    Guarda resultados intermedios de una etapa del pipeline.
    
    Útil para debugging y análisis de cada paso.
    
    Args:
        results: Resultados de la etapa
        output_dir: Directorio de salida
        stage_name: Nombre de la etapa (ej: "setfit", "dict_filter", "llm")
        doc_id: ID del documento (opcional)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if doc_id:
        filename = f"{stage_name}_{doc_id}_{timestamp}.json"
    else:
        filename = f"{stage_name}_{timestamp}.json"
    
    file_path = output_path / filename
    
    output_data = {
        'stage': stage_name,
        'timestamp': datetime.now().isoformat(),
        'total': len(results),
        'results': results
    }
    
    save_json(output_data, str(file_path))
    
    return str(file_path)


def export_to_csv(
    results: List[Dict[str, Any]],
    output_path: str,
    columns: Optional[List[str]] = None
):
    """
    Exporta resultados a CSV.
    
    Args:
        results: Lista de resultados
        output_path: Ruta del archivo CSV
        columns: Columnas a incluir (None = todas)
    """
    if not results:
        logger.warning("No results to export")
        return
    
    try:
        import pandas as pd
        
        df = pd.DataFrame(results)
        
        if columns:
            # Solo columnas especificadas que existan
            existing_cols = [c for c in columns if c in df.columns]
            df = df[existing_cols]
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Exported {len(results)} rows to {output_path}")
        
    except ImportError:
        # Sin pandas, usar csv
        import csv
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if columns is None:
            columns = list(results[0].keys())
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"Exported {len(results)} rows to {output_path}")
