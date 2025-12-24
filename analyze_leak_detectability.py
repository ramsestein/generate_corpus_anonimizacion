#!/usr/bin/env python3
"""
Analiza la "detectabilidad" de las fugas de cada modelo.

Clasifica las fugas en:
1. FÁCIL: Detectable con regex (emails, teléfonos, fechas completas, NIFs)
2. MEDIO: Detectable con LLM pequeño (nombres obvios, apellidos comunes)
3. DIFÍCIL: Fragmentos, palabras genéricas, casos ambiguos

El mejor modelo es el que tiene más fugas FÁCILES/MEDIO y menos DIFÍCILES.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Patrones regex para detectabilidad FÁCIL
EASY_PATTERNS = {
    'email': re.compile(r'^[\w.+-]+@[\w-]+\.\w+$'),
    'phone': re.compile(r'^\+?\d[\d\s.-]{7,}$'),
    'nif_nie': re.compile(r'^\d{8}[A-Z]$|^[XYZ]\d{7}[A-Z]$'),
    'fecha_completa': re.compile(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$'),
    'codigo_postal': re.compile(r'^\d{5}$'),
}

# Palabras que indican fragmentos (DIFÍCIL)
FRAGMENT_INDICATORS = ['her', 'mano', 'ges', 'ra', 'se', 'ma', 'na', 'ama']

# Nombres/apellidos comunes españoles (MEDIO - LLM pequeño puede detectar)
COMMON_NAMES = {
    'nombres': ['juan', 'maría', 'josé', 'ana', 'pedro', 'carmen', 'antonio', 'laura'],
    'apellidos': ['garcía', 'martínez', 'lópez', 'gonzález', 'rodríguez', 'fernández']
}


def classify_leak_detectability(entity_text: str) -> str:
    """
    Clasifica una fuga por detectabilidad.
    
    Returns:
        'FACIL', 'MEDIO', o 'DIFICIL'
    """
    text = entity_text.lower().strip()
    
    # 1. FÁCIL: Detectable con regex
    for pattern_name, pattern in EASY_PATTERNS.items():
        if pattern.match(text):
            return 'FACIL'
    
    # 2. DIFÍCIL: Fragmentos muy cortos o conocidos
    if len(text) <= 2:
        return 'DIFICIL'
    
    if text in FRAGMENT_INDICATORS:
        return 'DIFICIL'
    
    # Fechas fragmentadas (difíciles)
    if re.match(r'^[./]\d{1,2}$|^\d{1,2}[./]$', text):
        return 'DIFICIL'
    
    # Números sueltos de 2-3 dígitos (edades, códigos fragmentados)
    if re.match(r'^\d{2,3}$', text):
        return 'DIFICIL'
    
    # 3. MEDIO: Nombres/apellidos comunes (LLM pequeño puede detectar)
    for name in COMMON_NAMES['nombres']:
        if name in text:
            return 'MEDIO'
    
    for apellido in COMMON_NAMES['apellidos']:
        if apellido in text:
            return 'MEDIO'
    
    # Palabras capitalizadas (posibles nombres)
    if text[0].isupper() and len(text) >= 4:
        return 'MEDIO'
    
    # Por defecto: DIFÍCIL (casos ambiguos)
    return 'DIFICIL'


def analyze_model_leaks(comparison_results_dir: Path):
    """
    Analiza las fugas de cada modelo y las clasifica por detectabilidad.
    """
    delta_file = comparison_results_dir / 'delta_analysis.md'
    
    if not delta_file.exists():
        print(f"No se encontró {delta_file}")
        return
    
    # Leer el archivo markdown
    content = delta_file.read_text(encoding='utf-8')
    
    # Extraer fugas de cada modelo
    leaks = {
        'Pipeline A': [],
        'Pipeline B': []
    }
    
    current_model = None
    for line in content.split('\n'):
        if 'Pipeline A' in line and 'PII Real Eliminado' in line:
            current_model = 'Pipeline A'
        elif 'Pipeline B' in line and 'PII Real Eliminado' in line:
            current_model = 'Pipeline B'
        elif current_model and line.startswith('| ') and '`' in line:
            # Extraer entidad (formato: | # | `entidad` | documento |)
            parts = line.split('|')
            if len(parts) >= 3:
                entity = parts[2].strip().strip('`')
                if entity and entity != 'Entidad':
                    leaks[current_model].append(entity)
    
    # Clasificar fugas por detectabilidad
    results = {}
    for model, entities in leaks.items():
        classification = defaultdict(list)
        for entity in entities:
            detectability = classify_leak_detectability(entity)
            classification[detectability].append(entity)
        
        results[model] = {
            'total': len(entities),
            'FACIL': len(classification['FACIL']),
            'MEDIO': len(classification['MEDIO']),
            'DIFICIL': len(classification['DIFICIL']),
            'examples': {
                'FACIL': classification['FACIL'][:5],
                'MEDIO': classification['MEDIO'][:5],
                'DIFICIL': classification['DIFICIL'][:10]
            }
        }
    
    # Imprimir resultados
    print("=" * 80)
    print("ANÁLISIS DE DETECTABILIDAD DE FUGAS")
    print("=" * 80)
    print()
    
    for model, stats in results.items():
        print(f"\n{model}:")
        print(f"  Total fugas (muestra): {stats['total']}")
        print(f"  FÁCIL (regex):         {stats['FACIL']} ({stats['FACIL']/stats['total']*100:.1f}%)")
        print(f"  MEDIO (LLM pequeño):   {stats['MEDIO']} ({stats['MEDIO']/stats['total']*100:.1f}%)")
        print(f"  DIFÍCIL (ambiguo):     {stats['DIFICIL']} ({stats['DIFICIL']/stats['total']*100:.1f}%)")
        
        print(f"\n  Ejemplos FÁCIL: {stats['examples']['FACIL']}")
        print(f"  Ejemplos MEDIO: {stats['examples']['MEDIO']}")
        print(f"  Ejemplos DIFÍCIL: {stats['examples']['DIFICIL']}")
    
    # Comparación
    print("\n" + "=" * 80)
    print("COMPARACIÓN")
    print("=" * 80)
    
    a_score = results['Pipeline A']['FACIL'] * 3 + results['Pipeline A']['MEDIO'] * 2
    b_score = results['Pipeline B']['FACIL'] * 3 + results['Pipeline B']['MEDIO'] * 2
    
    print(f"\nScore detectabilidad (FÁCIL*3 + MEDIO*2):")
    print(f"  Pipeline A: {a_score}")
    print(f"  Pipeline B: {b_score}")
    
    if a_score > b_score:
        print(f"\n🏆 GANADOR: Pipeline A (fugas más detectables)")
    elif b_score > a_score:
        print(f"\n🏆 GANADOR: Pipeline B (fugas más detectables)")
    else:
        print(f"\n🤝 EMPATE")
    
    print("\n" + "=" * 80)
    print("RECOMENDACIÓN")
    print("=" * 80)
    print("\nElige el modelo con:")
    print("  1. Más fugas FÁCIL (regex puede rescatar)")
    print("  2. Más fugas MEDIO (LLM pequeño puede rescatar)")
    print("  3. Menos fugas DIFÍCIL (imposibles de rescatar)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path('comparison_results_v3')
    
    analyze_model_leaks(results_dir)
