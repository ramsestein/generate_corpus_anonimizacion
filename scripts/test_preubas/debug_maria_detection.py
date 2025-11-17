#!/usr/bin/env python3
"""
Script de debugging para investigar por qué no se detecta 'Maria' en prueba.txt
"""
import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Importar el módulo step6.1
from importlib.machinery import SourceFileLoader
step6 = SourceFileLoader('step6_1', str(REPO_ROOT / 'pipeline' / 'step6.1.py')).load_module()

def main():
    print("=" * 80)
    print("DEBUGGING: Detección de 'Maria' en prueba.txt")
    print("=" * 80)
    
    # Leer el archivo de prueba
    prueba_file = REPO_ROOT / 'scripts' / 'prueba.txt'
    
    if not prueba_file.exists():
        print(f"ERROR: No existe {prueba_file}")
        return 1
    
    with open(prueba_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"\n1. Archivo leído: {prueba_file}")
    print(f"   Longitud del texto: {len(text)} caracteres")
    print(f"   Primeros 100 caracteres: {repr(text[:100])}")
    print(f"   Últimos 100 caracteres: {repr(text[-100:])}")
    
    # Buscar 'Maria' manualmente
    import re
    maria_matches = list(re.finditer(r'Maria', text, re.IGNORECASE))
    print(f"\n2. Búsqueda manual de 'Maria':")
    print(f"   Encontradas {len(maria_matches)} ocurrencias:")
    for i, match in enumerate(maria_matches):
        start, end = match.span()
        context_start = max(0, start - 20)
        context_end = min(len(text), end + 20)
        context = text[context_start:context_end]
        print(f"   [{i+1}] Posición {start}-{end}: ...{repr(context)}...")
    
    # Cargar modelos
    print(f"\n3. Cargando modelos BSC...")
    try:
        meddocan_pipeline, carmen_pipeline = step6.setup_models()
        print("   ✓ Modelos cargados correctamente")
    except Exception as e:
        print(f"   ✗ ERROR cargando modelos: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Imprimir salida RAW de los pipelines (sin remapeo ni filtrado) para diagnosticar
    print("\n3.5 Salida RAW de MEDDOCAN (sin filtrado):")
    try:
        raw_meddocan = meddocan_pipeline(text)
        print(f"   Raw MEDDOCAN: {len(raw_meddocan)} items")
        for i, r in enumerate(raw_meddocan):
            print(f"    [{i}] {r}")
    except Exception as e:
        print(f"   ERROR obteniendo salida RAW MEDDOCAN: {e}")

    print("\n3.6 Salida RAW de CARMEN (sin filtrado):")
    try:
        raw_carmen = carmen_pipeline(text)
        print(f"   Raw CARMEN: {len(raw_carmen)} items")
        for i, r in enumerate(raw_carmen):
            print(f"    [{i}] {r}")
    except Exception as e:
        print(f"   ERROR obteniendo salida RAW CARMEN: {e}")

    # Probar una normalización leve: insertar espacio después de puntos si siguen palabras con mayúscula
    import re
    def normalize_spacing_for_names(s: str) -> str:
        # Añadir espacio después de un punto o paréntesis si el siguiente caracter es mayúscula (ej: 'confort.Maria' -> 'confort. Maria')
        return re.sub(r"([\.\:;\)])(?=[A-ZÁÉÍÓÚÑ])", r"\1 ", s)

    text_norm = normalize_spacing_for_names(text)
    if text_norm != text:
        print("\n4. Texto preprocesado aplicando normalize_spacing_for_names (se añadieron espacios donde faltaban)")
        # Mostrar contexto final donde aparecía Maria
        print(f"   Contexto original final: ...{repr(text[-60:])}...")
        print(f"   Contexto normalizado final: ...{repr(text_norm[-60:])}...")

        # Ejecutar RAW pipelines sobre el texto normalizado
        print("\n4.1 RAW MEDDOCAN sobre texto normalizado:")
        try:
            raw_meddocan_norm = meddocan_pipeline(text_norm)
            print(f"   Raw MEDDOCAN (norm): {len(raw_meddocan_norm)} items")
            for i, r in enumerate(raw_meddocan_norm):
                print(f"    [{i}] {r}")
        except Exception as e:
            print(f"   ERROR RAW MEDDOCAN (norm): {e}")

        print("\n4.2 Ejecutando extract_entities_with_model sobre texto normalizado (threshold=0.3):")
        ents_norm = step6.extract_entities_with_model(text_norm, meddocan_pipeline, "MEDDOCAN", confidence_threshold=0.3)
        print(f"   extract_entities_with_model (norm) devolvió {len(ents_norm)} entidades:")
        for i, ent in enumerate(ents_norm):
            print(f"   [{i}] {ent.get('entity_group','?'):15s} | conf={ent.get('score',0):.3f} | pos=({ent.get('start')},{ent.get('end')}) | text={repr(ent.get('actual_text',''))}")
    else:
        print("\n4. No se necesitó normalizar el texto (no se encontraron puntos pegados a mayúsculas).")

    # -------- Probar reemplazo de marcadores 'JJJ' por espacio para reducir ruido --------
    print("\n5. Probando reemplazo de 'JJJ' por espacio (text_nojjj) para reducir ruido en el texto)")
    text_nojjj = re.sub(r"\bJ{2,}\b", " ", text)
    print(f"   Muestra final texto_nojjj: ...{repr(text_nojjj[-60:])}...")

    print("\n5.1 RAW MEDDOCAN sobre text_nojjj:")
    try:
        raw_meddocan_nojjj = meddocan_pipeline(text_nojjj)
        print(f"   Raw MEDDOCAN (no JJJ): {len(raw_meddocan_nojjj)} items")
        for i, r in enumerate(raw_meddocan_nojjj):
            print(f"    [{i}] {r}")
    except Exception as e:
        print(f"   ERROR RAW MEDDOCAN (no JJJ): {e}")

    print("\n5.2 Ejecutando extract_entities_with_model sobre text_nojjj (threshold=0.3):")
    ents_nojjj = step6.extract_entities_with_model(text_nojjj, meddocan_pipeline, "MEDDOCAN", confidence_threshold=0.3)
    print(f"   extract_entities_with_model (no JJJ) devolvió {len(ents_nojjj)} entidades:")
    for i, ent in enumerate(ents_nojjj):
        print(f"   [{i}] {ent.get('entity_group','?'):15s} | conf={ent.get('score',0):.3f} | pos=({ent.get('start')},{ent.get('end')}) | text={repr(ent.get('actual_text',''))}")
    
    # Probar con MEDDOCAN
    print(f"\n4. Ejecutando detección con MEDDOCAN (threshold=0.3)...")
    print("-" * 80)
    meddocan_entities = step6.extract_entities_with_model(
        text, 
        meddocan_pipeline, 
        "MEDDOCAN", 
        confidence_threshold=0.3
    )
    print("-" * 80)
    print(f"\n   MEDDOCAN detectó {len(meddocan_entities)} entidades:")
    for i, ent in enumerate(meddocan_entities):
        print(f"   [{i+1}] {ent.get('entity_group', 'UNKNOWN'):15s} | "
              f"conf={ent.get('score', 0):.3f} | "
              f"pos=({ent.get('start')},{ent.get('end')}) | "
              f"text={repr(ent.get('actual_text', ''))}")
    
    # Buscar si alguna detección contiene 'Maria'
    maria_detections = [e for e in meddocan_entities if 'maria' in e.get('actual_text', '').lower()]
    if maria_detections:
        print(f"\n   ✓ Detecciones que contienen 'Maria': {len(maria_detections)}")
    else:
        print(f"\n   ✗ NINGUNA detección contiene 'Maria'")
    
    # Probar con CARMEN
    print(f"\n5. Ejecutando detección con CARMEN (threshold=0.3)...")
    print("-" * 80)
    carmen_entities = step6.extract_entities_with_model(
        text, 
        carmen_pipeline, 
        "CARMEN", 
        confidence_threshold=0.3
    )
    print("-" * 80)
    print(f"\n   CARMEN detectó {len(carmen_entities)} entidades:")
    for i, ent in enumerate(carmen_entities):
        print(f"   [{i+1}] {ent.get('entity_group', 'UNKNOWN'):15s} | "
              f"conf={ent.get('score', 0):.3f} | "
              f"pos=({ent.get('start')},{ent.get('end')}) | "
              f"text={repr(ent.get('actual_text', ''))}")
    
    # Buscar si alguna detección contiene 'Maria'
    maria_detections_carmen = [e for e in carmen_entities if 'maria' in e.get('actual_text', '').lower()]
    if maria_detections_carmen:
        print(f"\n   ✓ Detecciones que contienen 'Maria': {len(maria_detections_carmen)}")
    else:
        print(f"\n   ✗ NINGUNA detección contiene 'Maria'")
    
    # ANÁLISIS DETALLADO: Probar el pipeline directamente en cada chunk
    print(f"\n6. ANÁLISIS DETALLADO: Tokenización y chunks")
    print("=" * 80)
    
    # Obtener tokenizer
    tokenizer = meddocan_pipeline.tokenizer
    model_max_length = getattr(tokenizer, 'model_max_length', 512)
    token_chunk_size = min(model_max_length - 20, 512)
    token_overlap = 50
    
    print(f"   Configuración chunking:")
    print(f"   - model_max_length: {model_max_length}")
    print(f"   - token_chunk_size: {token_chunk_size}")
    print(f"   - token_overlap: {token_overlap}")
    
    # Tokenizar
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False
    )
    
    tokens = encoding['input_ids']
    offsets = encoding['offset_mapping']
    
    print(f"\n   Tokens totales: {len(tokens)}")
    
    # Encontrar tokens que corresponden a 'Maria'
    print(f"\n   Buscando tokens que contienen 'Maria':")
    for match in maria_matches:
        start, end = match.span()
        print(f"\n   'Maria' en posición {start}-{end}:")
        
        # Buscar qué tokens cubren esta posición
        covering_tokens = []
        for token_idx, (tok_start, tok_end) in enumerate(offsets):
            if not (tok_end <= start or tok_start >= end):  # Hay overlap
                token_text = text[tok_start:tok_end]
                covering_tokens.append((token_idx, tok_start, tok_end, token_text))
        
        print(f"      Cubierto por {len(covering_tokens)} tokens:")
        for tok_idx, tok_start, tok_end, tok_text in covering_tokens:
            print(f"      - Token {tok_idx}: pos=({tok_start},{tok_end}) text={repr(tok_text)}")
        
        # Determinar en qué chunk(s) cae
        print(f"\n      Chunks que incluyen esta posición:")
        if len(tokens) <= token_chunk_size:
            print(f"      - Chunk único (texto completo)")
        else:
            for i in range(0, len(tokens), token_chunk_size - token_overlap):
                chunk_start_tok = i
                chunk_end_tok = min(i + token_chunk_size - 1, len(tokens) - 1)
                chunk_start_char = offsets[chunk_start_tok][0]
                chunk_end_char = offsets[chunk_end_tok][1]
                
                # Verificar si este chunk contiene la posición de Maria
                if not (chunk_end_char <= start or chunk_start_char >= end):
                    chunk_text = text[chunk_start_char:chunk_end_char]
                    has_maria = 'Maria' in chunk_text or 'maria' in chunk_text
                    print(f"      - Chunk tokens {chunk_start_tok}-{chunk_end_tok}, "
                          f"chars {chunk_start_char}-{chunk_end_char}, "
                          f"len={len(chunk_text)}, "
                          f"contiene_Maria={has_maria}")
                    
                    if has_maria:
                        # Probar pipeline directamente en este chunk
                        print(f"\n      Probando MEDDOCAN directamente en este chunk:")
                        try:
                            direct_result = meddocan_pipeline(chunk_text)
                            print(f"         Pipeline devolvió {len(direct_result)} entidades:")
                            for dr in direct_result:
                                chunk_local_text = chunk_text[dr['start']:dr['end']]
                                print(f"         - {dr.get('entity_group', 'UNKNOWN'):15s} | "
                                      f"conf={dr.get('score', 0):.3f} | "
                                      f"chunk_pos=({dr['start']},{dr['end']}) | "
                                      f"text={repr(chunk_local_text)}")
                        except Exception as e:
                            print(f"         ERROR: {e}")
    
    print("\n" + "=" * 80)
    print("FIN DEL ANÁLISIS")
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
