#!/usr/bin/env python3
"""
Combine todas las entidades JSON de `corpus/ANTIGUO/entidades` en un único JSON.
Salida por defecto: `outputs/combined_entidades_ANTIGUO.json`.
"""
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Combina JSON de entidades en un único archivo")
parser.add_argument('--input-dir', default='corpus/ANTIGUO/entidades', help='Directorio con archivos JSON de entidades')
parser.add_argument('--output-file', default='outputs/combined_entidades_ANTIGUO.json', help='Archivo JSON de salida')
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_file = Path(args.output_file)

if not input_dir.exists():
    print(f"ERROR: input dir does not exist: {input_dir}")
    raise SystemExit(1)

output_file.parent.mkdir(parents=True, exist_ok=True)

combined = []
file_count = 0
error_files = []

for p in sorted(input_dir.glob('*.json')):
    file_count += 1
    try:
        with p.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        error_files.append({'file': str(p), 'error': str(e)})
        continue

    # Normalize structure: keep file stem as doc_id
    combined.append({
        'doc_id': p.stem,
        'file': str(p),
        'entities': data
    })

# Also include a small summary
summary = {
    'source_dir': str(input_dir),
    'files_processed': file_count,
    'files_failed': len(error_files),
}

result = {
    'summary': summary,
    'errors': error_files,
    'combined': combined
}

with output_file.open('w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Wrote combined file: {output_file} (files processed: {file_count}, failures: {len(error_files)})")
