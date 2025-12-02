import csv
from pathlib import Path

csv_path = Path('src/pipeline-axuliar/step6_validation_judgeLLM/detecciones_detalladas.csv')

def read_csv_detections(csv_path: Path):
    detections = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Intentar primero con coma (estándar)
        delimiter = ','
        reader = csv.DictReader(f, delimiter=delimiter)
        
        print(f"Fieldnames len: {len(reader.fieldnames) if reader.fieldnames else 0}")
        print(f"Fieldnames: {reader.fieldnames}")

        # Si falla (pocas columnas), intentar detectar
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            print("Fallback to sniffer")
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
        
        for i, row in enumerate(reader):
            # Mapear columnas
            doc_id = row.get('doc_id') or row.get('document_id')
            label = row.get('etiqueta') or row.get('label') or row.get('entity_type')
            model = row.get('modelo_detector') or row.get('model') or "UNKNOWN"
            text = row.get('texto_detectado') or row.get('text') or row.get('entity_text')
            
            detection = {
                "doc_id": (doc_id or "").strip(),
                "label": (label or "").strip(),
                "model": (model or "").strip(),
                "text": (text or "").strip(),
            }
            
            # Filtrar filas vacías
            if detection['doc_id'] and detection['text']:
                detections.append(detection)
            else:
                pass
                # print(f"Skipped row {i+2}: {row}")

    return detections

dets = read_csv_detections(csv_path)
print(f"Total detections: {len(dets)}")
