#!/usr/bin/env python3
"""
Convierte detecciones_detalladas.csv a formato JSON para el pipeline.

Este script:
1. Lee el CSV con detecciones de entidades
2. Fusiona entidades continuas (mismo documento, posiciones consecutivas)
3. Genera un JSON compatible con el pipeline de anonimización

Uso:
    python convert_csv_to_pipeline_input.py <csv_input> <json_output>
    
Ejemplo:
 python pipeline-nuevos-textos/utils/csv_converter.py pipeline-auxiliar/step6_validation_judgeLLM/detecciones_detalladas.csv  entidades-pipeline.json"""

import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict


def read_csv_detections(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Lee el CSV de detecciones y lo convierte en lista de diccionarios.
    
    Args:
        csv_path: Ruta al archivo CSV
        
    Returns:
        Lista de detecciones
    """
    detections = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Intentar primero con coma (estándar)
        delimiter = ','
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # Si falla (pocas columnas), intentar detectar
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            f.seek(0)
            sample = f.read(1024)
            f.seek(0)
            try:
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
            except:
                delimiter = ';' # Fallback final
            
            reader = csv.DictReader(f, delimiter=delimiter)

        # Normalizar nombres de columnas
        if reader.fieldnames:
            fieldnames = [f.lower() for f in reader.fieldnames]
            reader.fieldnames = fieldnames
        
        for row in reader:
            # Mapear columnas
            doc_id = row.get('doc_id') or row.get('document_id')
            label = row.get('etiqueta') or row.get('label') or row.get('entity_type')
            model = row.get('modelo_detector') or row.get('model') or "UNKNOWN"
            text = row.get('texto_detectado') or row.get('text') or row.get('entity_text')
            
            try:
                conf = float(row.get('confianza') or row.get('confidence') or row.get('score') or 1.0)
            except:
                conf = 1.0
                
            try:
                start = int(row.get('posicion_inicio') or row.get('start') or row.get('start_char') or -1)
                end = int(row.get('posicion_fin') or row.get('end') or row.get('end_char') or -1)
            except:
                start = -1
                end = -1

            detection = {
                "doc_id": (doc_id or "").strip(),
                "label": (label or "").strip(),
                "model": (model or "").strip(),
                "text": (text or "").strip(),
                "confidence": conf,
                "start": start,
                "end": end
            }
            
            # Filtrar filas vacías
            if detection['doc_id'] and detection['text']:
                detections.append(detection)
    
    return detections


def merge_continuous_entities(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fusiona entidades continuas del mismo documento.
    
    Dos entidades son continuas si:
    - Pertenecen al mismo documento
    - Tienen la misma etiqueta
    - Las posiciones son consecutivas o con mínima separación (ej: salto de línea)
    
    Args:
        detections: Lista de detecciones sin fusionar
        
    Returns:
        Lista de entidades fusionadas
    """
    # Agrupar por documento
    by_document = defaultdict(list)
    for det in detections:
        by_document[det['doc_id']].append(det)
    
    # Ordenar por posición dentro de cada documento
    for doc_id in by_document:
        by_document[doc_id].sort(key=lambda x: x['start'])
    
    unified_entities = []
    
    for doc_id, doc_detections in by_document.items():
        if not doc_detections:
            continue
        
        # Iniciar con la primera entidad
        current = doc_detections[0].copy()
        current['unified'] = True
        current['original_count'] = 1
        
        for next_det in doc_detections[1:]:
            # Calcular distancia entre entidades (posicional estricta)
            gap = next_det['start'] - current['end']

            # Regla de fusión estricta: unir SÓLO si las posiciones son válidas
            # y la entidad siguiente comienza exactamente donde termina la actual.
            positions_valid = (
                isinstance(current.get('end'), int)
                and isinstance(next_det.get('start'), int)
                and current['end'] >= 0
                and next_det['start'] >= 0
            )

            adjacent = positions_valid and (gap == 0)

            if adjacent:
                # Fusionar: extender la entidad actual (solo adyacencia perfecta)
                current['end'] = next_det['end']
                current['text'] += next_det['text']
                # Promediar confianza
                current['confidence'] = (current['confidence'] + next_det['confidence']) / 2
                current['original_count'] += 1
                # Mantener el modelo original (preferir CARMEN sobre MEDDOCAN)
                if next_det.get('model') == 'CARMEN' and current.get('model') != 'CARMEN':
                    current['model'] = next_det['model']
            else:
                # No fusionar: guardar la entidad actual y empezar una nueva
                unified_entities.append(current)
                current = next_det.copy()
                current['unified'] = True
                current['original_count'] = 1
        
        # Añadir la última entidad
        unified_entities.append(current)
    
    return unified_entities


def generate_pipeline_json(entities: List[Dict[str, Any]], output_path: Path, csv_path: Path):
    """
    Genera el archivo JSON en formato compatible con el pipeline.
    
    Args:
        entities: Lista de entidades unificadas
        output_path: Ruta del archivo JSON de salida
        csv_path: Ruta del CSV original (para metadata)
    """
    # Calcular estadísticas
    total_entities = len(entities)
    by_label = defaultdict(int)
    by_model = defaultdict(int)
    by_document = defaultdict(int)
    
    for entity in entities:
        by_label[entity['label']] += 1
        by_model[entity['model']] += 1
        by_document[entity['doc_id']] += 1
    
    # Calcular entidades fusionadas
    total_merged = sum(e.get('original_count', 1) - 1 for e in entities)
    
    # Construir estructura JSON
    output_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'source_file': str(csv_path),
            'total_entities': total_entities,
            'processing_stats': {
                'total_raw_entities': total_entities + total_merged,
                'total_unified_entities': total_entities,
                'entities_merged': total_merged,
                'entities_by_label': dict(sorted(by_label.items())),
                'entities_by_model': dict(sorted(by_model.items())),
                'documents_processed': len(by_document)
            },
            'analysis': {
                'total_entities': total_entities,
                'by_label': dict(sorted(by_label.items())),
                'by_model': dict(sorted(by_model.items())),
                'by_document': dict(sorted(by_document.items()))
            }
        },
        'entities': [
            {
                'doc_id': e['doc_id'],
                'label': e['label'],
                'model': e['model'],
                'text': e['text'],
                'confidence': round(e['confidence'], 4),
                'start': e['start'],
                'end': e['end'],
                'unified': e.get('unified', False),
                'manual_correction': None
            }
            for e in entities
        ]
    }
    
    # Guardar JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ JSON generado: {output_path}")
    print(f"  - Total entidades: {total_entities}")
    print(f"  - Entidades fusionadas: {total_merged}")
    print(f"  - Documentos procesados: {len(by_document)}")


def main():
    parser = argparse.ArgumentParser(
        description='Convierte CSV de detecciones a JSON para el pipeline de anonimización'
    )
    parser.add_argument(
        'csv_input',
        type=str,
        help='Ruta al archivo CSV de entrada (detecciones_detalladas.csv)'
    )
    parser.add_argument(
        'json_output',
        type=str,
        help='Ruta al archivo JSON de salida'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='No fusionar entidades continuas'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_input)
    json_path = Path(args.json_output)
    
    # Validar que el CSV existe
    if not csv_path.exists():
        print(f"❌ Error: No se encuentra el archivo CSV: {csv_path}")
        return 1
    
    print(f"🔄 Procesando: {csv_path}")
    
    # Leer CSV
    print("  [1/3] Leyendo detecciones del CSV...")
    detections = read_csv_detections(csv_path)
    print(f"        → {len(detections)} detecciones leídas")
    
    # Fusionar entidades continuas
    if args.no_merge:
        print("  [2/3] Fusión de entidades desactivada")
        unified = detections
    else:
        print("  [2/3] Fusionando entidades continuas...")
        unified = merge_continuous_entities(detections)
        merged_count = len(detections) - len(unified)
        print(f"        → {len(unified)} entidades unificadas ({merged_count} fusiones)")
    
    # Generar JSON
    print("  [3/3] Generando JSON de salida...")
    generate_pipeline_json(unified, json_path, csv_path)
    
    if args.verbose:
        print("\n[RESUMEN POR ETIQUETA]")
        by_label = defaultdict(int)
        for e in unified:
            by_label[e['label']] += 1
        for label, count in sorted(by_label.items(), key=lambda x: -x[1]):
            print(f"  {label}: {count}")
    
    print("\n✅ Conversión completada")
    print(f"\nAhora puedes usar este archivo como entrada del pipeline:")
    print(f"  python src/pipeline-nuevos-textos/run_full_pipeline.py --input {json_path} --output resultados.json")
    
    return 0


if __name__ == '__main__':
    exit(main())
