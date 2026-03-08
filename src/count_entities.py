import os
import json
from pathlib import Path
from collections import Counter

def count_entities():
    entities_dir = Path("corpus/entidades")
    output_file = Path("corpus/entity_text_counts.json")

    if not entities_dir.exists():
        print(f"Error: El directorio {entities_dir} no existe.")
        return

    print(f"Analizando documentos en {entities_dir}...")
    
    # Contador de pares (texto, etiqueta)
    text_counter = Counter()
    # Contador de frecuencia de documentos para cada par (texto, etiqueta)
    doc_freq_counter = Counter()
    
    total_docs = 0
    total_entities = 0

    # Iterar sobre todos los archivos JSON en el directorio de entidades
    for ent_file in entities_dir.glob("*.json"):
        total_docs += 1
        doc_texts = set()
        try:
            with open(ent_file, 'r', encoding='utf-8') as f:
                ent_data = json.load(f)
                
            for item in ent_data.get("data", []):
                label = item.get("entity")
                text = item.get("text")
                if label and text:
                    key = (text, label)
                    text_counter[key] += 1
                    doc_texts.add(key)
                    total_entities += 1
            
            # Contar en cuántos documentos aparece cada par (texto, etiqueta)
            for key in doc_texts:
                doc_freq_counter[key] += 1
        except Exception as e:
            print(f"Error leyendo {ent_file}: {e}")

    if total_docs == 0:
        print("No se encontraron documentos JSON en el directorio.")
        return

    print(f"Se analizaron {total_docs} documentos.")
    print(f"Total de entidades encontradas: {total_entities}\n")

    # Mostrar el TOP 50 de textos más repetidos
    TOP_N = 50
    print(f"{'Texto':<40} | {'Entidad':<30} | {'Freq':<6} | {'% Docs':<8}")
    print("-" * 95)
    
    # Ordenar por frecuencia descendente
    all_sorted = text_counter.most_common()
    
    table_data = []
    
    for (text, entity), count in all_sorted:
        doc_freq = doc_freq_counter[(text, entity)]
        percentage = (doc_freq / total_docs) * 100
        
        row = {
            "text": text,
            "entity": entity,
            "frequency": count,
            "doc_frequency": doc_freq,
            "percentage_docs": round(percentage, 2)
        }
        table_data.append(row)
        
        # Solo imprimimos el TOP_N en consola
        if len(table_data) <= TOP_N:
            # Truncar texto si es muy largo para la tabla
            display_text = (text[:37] + '...') if len(text) > 40 else text
            print(f"{display_text:<40} | {entity:<30} | {count:<6} | {percentage:>6.2f}%")

    # Guardar resultados en un archivo JSON
    results = {
        "metadata": {
            "total_documents": total_docs,
            "total_entities": total_entities,
            "unique_text_pairs": len(all_sorted)
        },
        "entities": table_data
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResultados completos de {len(all_sorted)} textos guardados en: {output_file.absolute()}")
    except Exception as e:
        print(f"Error guardando los resultados: {e}")

if __name__ == "__main__":
    count_entities()
