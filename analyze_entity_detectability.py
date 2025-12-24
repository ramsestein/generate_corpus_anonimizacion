#!/usr/bin/env python3
"""
Analiza qué tipo de entidades detecta cada modelo.

Clasifica las entidades en:
- FÁCIL: Nombres propios, fechas completas, emails, teléfonos, NIFs
- AMBIGUA: madre, familia, hospital, palabras genéricas, fragmentos
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter

# Patrones para entidades FÁCILES
EASY_PATTERNS = {
    'email': re.compile(r'[\w.+-]+@[\w-]+\.\w+'),
    'phone': re.compile(r'\+?\d[\d\s.-]{7,}'),
    'nif_nie': re.compile(r'\d{8}[A-Z]|[XYZ]\d{7}[A-Z]'),
    'fecha_completa': re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'),
    'nombre_capitalizado': re.compile(r'^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}$'),
}

# Palabras AMBIGUAS (difíciles de detectar sin contexto)
AMBIGUOUS_WORDS = {
    'familiares': ['madre', 'padre', 'hijo', 'hija', 'hermano', 'hermana', 
                   'abuelo', 'abuela', 'tío', 'tía', 'familia', 'familiar',
                   'cuidador', 'cuidadora', 'cuidadores', 'compañera', 'compañero'],
    'lugares_genericos': ['hospital', 'centro', 'clínica', 'madrid', 'barcelona'],
    'fragmentos': ['her', 'mano', 'ges', 'ra', 'se', 'ma', 'na', 'ama', 'eb', 'mar'],
    'numeros_cortos': re.compile(r'^\d{1,3}$'),  # Edades, códigos
    'fechas_fragmentadas': re.compile(r'^[./]\d{1,2}$|^\d{1,2}[./]$|^/\d{2}$'),
}


def classify_entity(text: str, label: str) -> str:
    """
    Clasifica una entidad como FÁCIL o AMBIGUA.
    
    Returns:
        'FACIL' o 'AMBIGUA'
    """
    text_lower = text.lower().strip()
    text_clean = text.strip()
    
    # 1. FÁCIL: Patrones regex claros
    for pattern_name, pattern in EASY_PATTERNS.items():
        if pattern.search(text_clean):
            return 'FACIL'
    
    # 2. AMBIGUA: Palabras genéricas
    for category, words in AMBIGUOUS_WORDS.items():
        if category in ['familiares', 'lugares_genericos']:
            if text_lower in words:
                return 'AMBIGUA'
        elif category == 'fragmentos':
            if text_lower in words or len(text_clean) <= 2:
                return 'AMBIGUA'
        elif category == 'numeros_cortos':
            if words.match(text_clean):
                return 'AMBIGUA'
        elif category == 'fechas_fragmentadas':
            if words.match(text_clean):
                return 'AMBIGUA'
    
    # 3. Por etiqueta
    if label in ['NOMBRE_SUJETO_ASISTENCIA', 'NOMBRE_PERSONAL_SANITARIO']:
        # Si es nombre y tiene mayúscula inicial, probablemente FÁCIL
        if text_clean[0].isupper() and len(text_clean) >= 3:
            return 'FACIL'
    
    if label == 'FAMILIARES_SUJETO_ASISTENCIA':
        # Familiares genéricos son AMBIGUOS
        return 'AMBIGUA'
    
    # Por defecto: FÁCIL (asumimos que si llegó aquí, es detectable)
    return 'FACIL'


def analyze_pipeline_output(json_file: Path, pipeline_name: str):
    """
    Analiza un archivo de salida del pipeline.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extraer entidades
    entities = []
    if isinstance(data, dict) and 'decisions' in data:
        entities = data['decisions']
    elif isinstance(data, list):
        entities = data
    
    # Clasificar entidades
    classification = defaultdict(list)
    label_distribution = Counter()
    
    for entity in entities:
        text = entity.get('entity_text', entity.get('text', ''))
        label = entity.get('label', entity.get('entity_label', ''))
        
        if not text:
            continue
        
        detectability = classify_entity(text, label)
        classification[detectability].append({
            'text': text,
            'label': label
        })
        label_distribution[label] += 1
    
    # Resultados
    total = len(entities)
    facil_count = len(classification['FACIL'])
    ambigua_count = len(classification['AMBIGUA'])
    
    print(f"\n{'=' * 80}")
    print(f"{pipeline_name}")
    print(f"{'=' * 80}")
    print(f"\nTotal entidades detectadas: {total}")
    print(f"\n  FÁCIL (nombres, fechas, emails):  {facil_count} ({facil_count/total*100:.1f}%)")
    print(f"  AMBIGUA (madre, familia, fragmentos): {ambigua_count} ({ambigua_count/total*100:.1f}%)")
    
    # Top 10 etiquetas
    print(f"\nTop 10 etiquetas más frecuentes:")
    for label, count in label_distribution.most_common(10):
        print(f"  {label}: {count}")
    
    # Ejemplos de cada tipo
    print(f"\nEjemplos FÁCIL (primeros 15):")
    for i, ent in enumerate(classification['FACIL'][:15], 1):
        print(f"  {i}. '{ent['text']}' ({ent['label']})")
    
    print(f"\nEjemplos AMBIGUA (primeros 15):")
    for i, ent in enumerate(classification['AMBIGUA'][:15], 1):
        print(f"  {i}. '{ent['text']}' ({ent['label']})")
    
    return {
        'total': total,
        'facil': facil_count,
        'ambigua': ambigua_count,
        'facil_pct': facil_count/total*100 if total > 0 else 0,
        'ambigua_pct': ambigua_count/total*100 if total > 0 else 0
    }


def main():
    base_dir = Path('outputs')
    
    # Analizar ambos pipelines
    results = {}
    
    print("\n" + "=" * 80)
    print("ANÁLISIS DE DETECTABILIDAD DE ENTIDADES")
    print("=" * 80)
    
    # Pipeline A
    pipeline_a = base_dir / 'salida.json'
    if pipeline_a.exists():
        results['Pipeline A'] = analyze_pipeline_output(
            pipeline_a, 
            "Pipeline A (salida.json)"
        )
    
    # Pipeline B
    pipeline_b = base_dir / 'resultados_recall_v2.json'
    if pipeline_b.exists():
        results['Pipeline B'] = analyze_pipeline_output(
            pipeline_b,
            "Pipeline B (resultados_recall_v2.json)"
        )
    
    # Comparación
    if len(results) == 2:
        print(f"\n{'=' * 80}")
        print("COMPARACIÓN")
        print(f"{'=' * 80}")
        
        a = results['Pipeline A']
        b = results['Pipeline B']
        
        print(f"\nPipeline A:")
        print(f"  FÁCIL: {a['facil']} ({a['facil_pct']:.1f}%)")
        print(f"  AMBIGUA: {a['ambigua']} ({a['ambigua_pct']:.1f}%)")
        
        print(f"\nPipeline B:")
        print(f"  FÁCIL: {b['facil']} ({b['facil_pct']:.1f}%)")
        print(f"  AMBIGUA: {b['ambigua']} ({b['ambigua_pct']:.1f}%)")
        
        print(f"\n{'=' * 80}")
        print("RECOMENDACIÓN")
        print(f"{'=' * 80}")
        
        if a['facil_pct'] > b['facil_pct']:
            print(f"\n🏆 Pipeline A detecta más entidades FÁCILES ({a['facil_pct']:.1f}% vs {b['facil_pct']:.1f}%)")
            print("   → Mejor para anonimización (menos ambigüedad)")
        elif b['facil_pct'] > a['facil_pct']:
            print(f"\n🏆 Pipeline B detecta más entidades FÁCILES ({b['facil_pct']:.1f}% vs {a['facil_pct']:.1f}%)")
            print("   → Mejor para anonimización (menos ambigüedad)")
        else:
            print(f"\n🤝 Ambos pipelines tienen similar distribución de detectabilidad")
        
        print(f"\nCriterio: Preferir el modelo que detecte más entidades FÁCILES")
        print("(nombres propios, fechas, emails) y menos AMBIGUAS (madre, familia)")


if __name__ == '__main__':
    main()
