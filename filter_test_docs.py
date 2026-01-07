#!/usr/bin/env python3
"""
Script para filtrar detecciones_detalladas.csv y copiar solo las filas
de los 50 documentos seleccionados para pruebas.
"""

import csv
from pathlib import Path

# Archivos
input_csv = Path('step6_validation_results/aws2/detecciones_detalladas.csv')
output_csv = Path('step6_validation_results/aws2-pruebas/detecciones-seleccionadas.csv')
docs_file = Path('test_docs_50.txt')

# Cargar lista de documentos seleccionados
with open(docs_file, 'r', encoding='utf-8') as f:
    selected_docs = set(line.strip() for line in f if line.strip())

print(f"Documentos seleccionados: {len(selected_docs)}")

# Leer CSV original y filtrar
with open(input_csv, 'r', encoding='utf-8') as f_in:
    reader = csv.reader(f_in)
    header = next(reader)
    
    # Encontrar columna de doc_id (primera columna)
    rows_filtered = []
    rows_total = 0
    
    for row in reader:
        rows_total += 1
        doc_id = row[0] if row else ''
        
        if doc_id in selected_docs:
            rows_filtered.append(row)

print(f"Filas totales: {rows_total}")
print(f"Filas filtradas: {len(rows_filtered)}")

# Guardar CSV filtrado
output_csv.parent.mkdir(parents=True, exist_ok=True)
with open(output_csv, 'w', encoding='utf-8', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(header)
    writer.writerows(rows_filtered)

print(f"\nGuardado en: {output_csv}")
print(f"Total filas: {len(rows_filtered)} (de {rows_total})")
