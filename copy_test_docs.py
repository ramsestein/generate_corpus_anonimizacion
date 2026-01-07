#!/usr/bin/env python3
"""
Script para copiar los 50 documentos seleccionados de corpus/aws2 a corpus/aws2-prueba
"""

import shutil
from pathlib import Path

# Directorios
source_dir = Path('corpus/output/aws2')
dest_dir = Path('corpus/output/aws2-prueba')

# Lista de los 50 documentos
selected_docs = [
    "NHC5581723_episodio1008436709.txt",
    "NHC5566693_episodio1008774659.txt",
    "NHC5578853_episodio1008590097.txt",
    "NHC5563489_episodio1008370959.txt",
    "NHC5575687_episodio1008348431.txt",
    "NHC107102_episodio1008411916.txt",
    "NHC4198704_episodio1008285899.txt",
    "NHC5569653_episodio1008312191.txt",
    "NHC250452_episodio1008719568.txt",
    "NHC5577604_episodio1008637588.txt",
    "NHC5582910_episodio1008396001.txt",
    "NHC4034835_episodio1008530518.txt",
    "NHC5576593_episodio1008351936.txt",
    "NHC1158_episodio1008293935.txt",
    "NHC302538_episodio1008414306.txt",
    "NHC274523_episodio1008513588.txt",
    "NHC5564244_episodio1008274483.txt",
    "NHC5569798_episodio1008326489.txt",
    "NHC4085263_episodio1008717131.txt",
    "NHC5573424_episodio1008335354.txt",
    "NHC5582910_episodio1008394614.txt",
    "NHC37937_episodio1008254299.txt",
    "NHC5558558_episodio1008281141.txt",
    "NHC5585300_episodio1008687210.txt",
    "NHC4084564_episodio1008735773.txt",
    "NHC5565961_episodio1008280467.txt",
    "NHC179468_episodio1008393022.txt",
    "NHC19816_episodio1008346144.txt",
    "NHC338622_episodio1008518245.txt",
    "NHC5572932_episodio1008331708.txt",
    "NHC323932_episodio1008232100.txt",
    "NHC5580063_episodio1008372518.txt",
    "NHC5581074_episodio1008647757.txt",
    "NHC346940_episodio1008608640.txt",
    "NHC5578715_episodio1008372742.txt",
    "NHC5580111_episodio1008374350.txt",
    "NHC5575478_episodio1008341198.txt",
    "NHC295309_episodio1008536044.txt",
    "NHC4075917_episodio1008365969.txt",
    "NHC5569869_episodio1008313145.txt",
    "NHC5568921_episodio1008350100.txt",
    "NHC4158888_episodio1008657741.txt",
    "NHC5581074_episodio1008646905.txt",
    "NHC5576526_episodio1008353156.txt",
    "NHC5583668_episodio1008559582.txt",
    "NHC365276_episodio1008350813.txt",
    "NHC5582715_episodio1008397802.txt",
    "NHC5574823_episodio1008341183.txt",
    "NHC4067438_episodio1008262995.txt",
    "NHC4072273_episodio1008742093.txt",
]

# Crear directorio destino
dest_dir.mkdir(parents=True, exist_ok=True)

print(f"Documentos a copiar: {len(selected_docs)}")
print(f"Origen: {source_dir}")
print(f"Destino: {dest_dir}")
print()

copied = 0
not_found = []

for doc_name in selected_docs:
    # Los archivos tienen doble extension .txt.txt
    src_path = source_dir / (doc_name + ".txt") if not doc_name.endswith(".txt.txt") else source_dir / doc_name
    dst_path = dest_dir / doc_name
    
    if src_path.exists():
        shutil.copy2(src_path, dst_path)
        copied += 1
        print(f"  Copiado: {doc_name}")
    else:
        not_found.append(doc_name)
        print(f"  NO ENCONTRADO: {doc_name}")

print()
print(f"Copiados: {copied}/{len(selected_docs)}")
if not_found:
    print(f"No encontrados: {len(not_found)}")
    for doc in not_found[:5]:
        print(f"  - {doc}")
