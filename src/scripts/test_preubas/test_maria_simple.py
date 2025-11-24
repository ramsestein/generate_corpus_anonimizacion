#!/usr/bin/env python3
"""
Test directo: ¿El modelo detecta 'Maria' en texto simple?
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from importlib.machinery import SourceFileLoader
step6 = SourceFileLoader('step6_1', str(REPO_ROOT / 'pipeline' / 'step6.1.py')).load_module()

def test_simple_texts():
    """Probar con textos simples que contienen 'Maria'"""
    
    test_cases = [
        "Maria Fecha: JJJ",
        "La paciente Maria tiene 45 años.",
        "Maria García fue atendida en urgencias.",
        "Paciente: Maria Lopez",
        "Nombre: Maria",
        "confort.Maria",
        "Maria",
        "MARIA",
        "María",
    ]
    
    print("Cargando modelos...")
    meddocan_pipeline, carmen_pipeline = step6.setup_models()
    
    print("\n" + "=" * 80)
    print("PRUEBA: ¿Los modelos detectan 'Maria' en textos simples?")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Texto: {repr(text)}")
        print("-" * 80)
        
        # MEDDOCAN
        try:
            result_meddocan = meddocan_pipeline(text)
            print(f"MEDDOCAN: {len(result_meddocan)} detecciones")
            for ent in result_meddocan:
                detected_text = text[ent['start']:ent['end']]
                print(f"  - {ent['entity_group']:15s} | conf={ent['score']:.3f} | text={repr(detected_text)}")
        except Exception as e:
            print(f"MEDDOCAN: ERROR - {e}")
        
        # CARMEN
        try:
            result_carmen = carmen_pipeline(text)
            print(f"CARMEN:   {len(result_carmen)} detecciones")
            for ent in result_carmen:
                detected_text = text[ent['start']:ent['end']]
                print(f"  - {ent['entity_group']:15s} | conf={ent['score']:.3f} | text={repr(detected_text)}")
        except Exception as e:
            print(f"CARMEN:   ERROR - {e}")

if __name__ == '__main__':
    test_simple_texts()
