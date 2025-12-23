#!/usr/bin/env python3
"""
Genera dataset de entrenamiento mejorado para SetFit basado en:
1. Análisis de fallos actuales (over-cleaning, noise leakage)
2. Guías de anotación
3. Dataset base existente

El objetivo es ganar precision sin perder recall.
"""

import json
import re
import csv
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
import random

# Configuración
SEED = 42
random.seed(SEED)

# =============================================================================
# REGLAS DE CONTEXTO BASADAS EN GUÍAS
# =============================================================================

# PII CLARO - Siempre debe clasificarse como PII (label=1)
PII_PATTERNS = {
    # FECHAS completas
    'fechas_completas': [
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # 20/09/2016
        r'\d{1,2}\.\d{1,2}\.\d{2,4}',  # 20.09.2016
        r'\d{1,2}-\d{1,2}-\d{2,4}',  # 20-09-2016
    ],
    # EDADES con unidad
    'edades': [
        r'\d+\s*(años?|meses|semanas|días|horas)',
        r'\d+a\b',  # 72a
    ],
    # NOMBRES completos o apellidos
    'nombres': [
        r'[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+',  # Nombre Apellido
    ],
    # FECHAS parciales con contexto temporal
    'fechas_parciales_temporales': [
        r'(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+\d{1,2}',
        r'\d{1,2}\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)',
    ],
    # AÑOS completos
    'años': [
        r'\b(19|20)\d{2}\b',  # 2024, 2001, etc.
    ],
}

# RUIDO CLARO - Debe filtrarse (label=0)
RUIDO_PATTERNS = {
    # Letras sueltas sin contexto
   
    # Palabras genéricas sin contexto identificativo
    'genericos': [
        r'^(familia|familiares|entorno familiar|cuidadores?)$',
    ],
  
}

# CONTEXTOS CRÍTICOS - Requieren análisis cuidadoso
CRITICAL_CONTEXTS = {
    # FAMILIARES con relación explícita = PII
    'familiares_con_relacion': {
        'keywords': ['hijo', 'hija', 'madre', 'padre', 'hermano', 'hermana', 'esposo', 'esposa', 
                     'abuelo', 'abuela', 'primo', 'tío', 'sobrino', 'cuidador'],
        'label': 1,  # PII
        'context_required': True
    },
    # Fechas parciales con contexto médico = PII
    'fechas_parciales_medicas': {
        'keywords': ['ingreso', 'alta', 'consulta', 'cita', 'intervención', 'episodio', 
                     'fecha', 'día', 'última', 'próxima'],
        'label': 1,  # PII
        'context_required': True
    },
    # Profesiones específicas = PII
    'profesiones_especificas': {
        'keywords': ['médico', 'enfermero', 'doctor', 'Dr.', 'Dra.'],
        'label': 1,  # PII (personal sanitario)
        'context_required': True
    },
}

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def normalize(text: str) -> str:
    """Normaliza texto."""
    if not text:
        return ''
    return ' '.join(str(text).lower().strip().split())

def format_training_example(entity: str, context: str) -> str:
    """Formato estándar para SetFit."""
    entity = entity.strip()
    context = context.strip()
    return f"ENTITY: {entity}\nSENTENCE: {context}"

def classify_by_pattern(entity: str) -> int:
    """Clasifica entidad por patrones."""
    entity_norm = normalize(entity)
    
    # Verificar PII patterns
    for pattern_group in PII_PATTERNS.values():
        for pattern in pattern_group:
            if re.search(pattern, entity_norm, re.IGNORECASE):
                return 1  # PII
    
    # Verificar RUIDO patterns
    for pattern_group in RUIDO_PATTERNS.values():
        for pattern in pattern_group:
            if re.search(pattern, entity_norm, re.IGNORECASE):
                return 0  # RUIDO
    
    # Casos ambiguos requieren contexto
    return -1

def has_critical_context(entity: str, context: str) -> int:
    """Verifica si tiene contexto crítico."""
    entity_norm = normalize(entity)
    context_norm = normalize(context)
    
    for ctx_name, ctx_config in CRITICAL_CONTEXTS.items():
        for keyword in ctx_config['keywords']:
            if keyword in entity_norm or keyword in context_norm:
                return ctx_config['label']
    
    return -1

# =============================================================================
# EXTRACCIÓN DE FALLOS
# =============================================================================

def extract_overcleaning_examples(delta_metrics_json: Path) -> List[Dict]:
    """Extrae ejemplos de PII que SetFit mató incorrectamente."""
    with open(delta_metrics_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    
    # Pipeline B over-cleaning (2,277 casos)
    pipeline_b = data.get('pipeline_b', {})
    
    # Necesitamos buscar en el reporte de errores
    return examples

def load_overcleaning_from_report(report_md: Path) -> List[Dict]:
    """Extrae ejemplos del reporte markdown."""
    examples = []
    
    with open(report_md, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar sección de over-cleaning de Pipeline B
    section_start = content.find('### Pipeline B (Base + SetFit B) - PII Real Eliminado')
    if section_start == -1:
        return examples
    
    section = content[section_start:section_start+2000]
    
    # Extraer tabla
    lines = section.split('\n')
    in_table = False
    for line in lines:
        if line.startswith('|') and not line.startswith('|---|'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4 and parts[1].isdigit():
                entity = parts[2].strip('`')
                doc_id = parts[3]
                examples.append({
                    'entity': entity,
                    'doc_id': doc_id,
                    'label': 1,  # Es PII que fue matado
                    'context': '',  # Necesitamos extraer del documento
                    'source': 'overcleaning_pipeline_b'
                })
    
    return examples

def load_noise_leakage(report_md: Path) -> List[Dict]:
    """Extrae ejemplos de ruido que pasó."""
    examples = []
    
    with open(report_md, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pipeline A filtró correctamente (estos son RUIDO confirmado)
    section_start = content.find('### Pipeline A (Base + SetFit A) filtró, Pipeline B')
    if section_start == -1:
        return examples
    
    section = content[section_start:section_start+1500]
    lines = section.split('\n')
    
    for line in lines:
        if line.startswith('|') and not line.startswith('|---|'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4 and parts[1].isdigit():
                entity = parts[2].strip('`')
                doc_id = parts[3]
                examples.append({
                    'entity': entity,
                    'doc_id': doc_id,
                    'label': 0,  # RUIDO confirmado
                    'context': '',
                    'source': 'noise_confirmed_by_a'
                })
    
    return examples

# =============================================================================
# GENERACIÓN DE EJEMPLOS SINTÉTICOS
# =============================================================================

def generate_synthetic_pii_examples() -> List[Dict]:
    """Genera ejemplos sintéticos de PII claro."""
    examples = []
    
    # FECHAS completas
    fechas = [
        ('15/03/2024', 'Fecha de ingreso: [15/03/2024]'),
        ('20.09.2016', 'Alta hospitalaria el [20.09.2016]'),
        ('23 de octubre de 1972', 'Fecha de nacimiento: [23 de octubre de 1972]'),
        ('12-05-2023', 'Última consulta: [12-05-2023]'),
        ('2024', 'Durante el año [2024] presentó...'),
        ('junio 22', 'Ingreso en [junio 22]'),
        ('15.08.2024', 'Intervención quirúrgica [15.08.2024]'),
        ('17.03', 'Cita programada para [17.03]'),
        ('17/3', 'Próxima revisión [17/3]'),
        ('8/11', 'Alta prevista [8/11]'),
        ('25.06.23', 'Última analítica [25.06.23]'),
        ('2001', 'Diagnosticado en [2001]'),
    ]
    
    for entity, context in fechas:
        examples.append({
            'entity': entity,
            'context': context,
            'label': 1,
            'source': 'synthetic_fechas',
            'category': 'FECHAS'
        })
    
    # EDADES con unidad
    edades = [
        ('68 años', 'Paciente de [68 años] con antecedentes...'),
        ('72a', 'Varón de [72a] que acude...'),
        ('62a', 'Mujer de [62a] ingresada por...'),
        ('3 meses', 'Lactante de [3 meses] con...'),
        ('40 años', 'Paciente de [40 años] de edad...'),
    ]
    
    for entity, context in edades:
        examples.append({
            'entity': entity,
            'context': context,
            'label': 1,
            'source': 'synthetic_edad',
            'category': 'EDAD_SUJETO_ASISTENCIA'
        })
    
    # NOMBRES y apellidos
    nombres = [
        ('María García', 'Paciente [María García] refiere...'),
        ('martínez', 'Apellido del paciente: [martínez]'),
        ('Carmen Ruiz', 'Dra. [Carmen Ruiz] realizó...'),
        ('López', 'Sr. [López] presenta...'),
    ]
    
    for entity, context in nombres:
        examples.append({
            'entity': entity,
            'context': context,
            'label': 1,
            'source': 'synthetic_nombres',
            'category': 'NOMBRE_SUJETO_ASISTENCIA'
        })
    
    # FAMILIARES con relación explícita (PII según guías)
    familiares = [
        ('hija', 'Acude acompañado de su [hija]'),
        ('madre', 'Refiere que su [madre] también...'),
        ('2 hermanas', 'Tiene [2 hermanas] vivas'),
        ('esposa', 'Su [esposa] refiere que...'),
        ('cuidadores', 'Los [cuidadores] principales son...'),
        ('familiares', 'Contactar con [familiares] cercanos'),
        ('hijo varón', 'Un [hijo varón] de 30 años'),
        ('una hija', 'Tiene [una hija] de 14 años'),
    ]
    
    for entity, context in familiares:
        examples.append({
            'entity': entity,
            'context': context,
            'label': 1,
            'source': 'synthetic_familiares',
            'category': 'FAMILIARES_SUJETO_ASISTENCIA'
        })
    
    return examples

def generate_synthetic_ruido_examples() -> List[Dict]:
    """Genera ejemplos sintéticos de RUIDO claro."""
    examples = []
    
    # Letras sueltas sin contexto
    letras_sueltas = [
        ('h', 'Apartado [h] del informe'),
        ('y', 'Conectores [y] conjunciones'),
        ('a', 'Letra [a] de la lista'),
        ('1', 'Punto [1] del listado'),
        ('12', 'Número [12] de la secuencia'),
    ]
    
    for entity, context in letras_sueltas:
        examples.append({
            'entity': entity,
            'context': context,
            'label': 0,
            'source': 'synthetic_ruido_letras',
            'category': 'RUIDO'
        })
    
    # Fragmentos mal extraídos
    fragmentos = [
        ('.08.2024', 'Fecha parcial [.08.2024] mal extraída'),
        ('.07.2024', 'Fragmento [.07.2024] incompleto'),
        ('di a', 'Texto fragmentado [di a]'),
        ('rero de 2 024', 'Extracción errónea [rero de 2 024]'),
        ('19.03.', 'Fecha incompleta [19.03.]'),
        ('14/7', 'Referencia numérica [14/7] sin contexto'),
    ]
    
    for entity, context in fragmentos:
        examples.append({
            'entity': entity,
            'context': context,
            'label': 0,
            'source': 'synthetic_ruido_fragmentos',
            'category': 'RUIDO'
        })
    
    # Palabras genéricas sin contexto identificativo
    genericos = [
        ('familia', 'Entorno [familia] sin especificar'),
        ('bebé', 'Referencia genérica [bebé]'),
        ('mare', 'Palabra [mare] sin contexto'),
        ('família', 'Mención genérica [família]'),
    ]
    
    for entity, context in genericos:
        examples.append({
            'entity': entity,
            'context': context,
            'label': 0,
            'source': 'synthetic_ruido_genericos',
            'category': 'RUIDO'
        })
    
    return examples

# =============================================================================
# CONSTRUCCIÓN DEL DATASET
# =============================================================================

def build_improved_dataset():
    """Construye dataset mejorado."""
    
    print("=" * 80)
    print("GENERACIÓN DE DATASET DE ENTRENAMIENTO MEJORADO")
    print("=" * 80)
    
    all_examples = []
    
    # 1. Ejemplos sintéticos de PII claro
    print("\n[1/6] Generando ejemplos sintéticos de PII...")
    pii_synthetic = generate_synthetic_pii_examples()
    all_examples.extend(pii_synthetic)
    print(f"  ✓ {len(pii_synthetic)} ejemplos de PII sintéticos")
    
    # 2. Ejemplos sintéticos de RUIDO claro
    print("\n[2/6] Generando ejemplos sintéticos de RUIDO...")
    ruido_synthetic = generate_synthetic_ruido_examples()
    all_examples.extend(ruido_synthetic)
    print(f"  ✓ {len(ruido_synthetic)} ejemplos de RUIDO sintéticos")
    
    # 3. Cargar over-cleaning (PII que fue matado)
    print("\n[3/6] Cargando ejemplos de over-cleaning...")
    delta_report = Path('comparison_results/delta_analysis.md')
    if delta_report.exists():
        overcleaning = load_overcleaning_from_report(delta_report)
        all_examples.extend(overcleaning)
        print(f"  ✓ {len(overcleaning)} ejemplos de PII mal clasificado")
    else:
        print("  ⚠️ No se encontró delta_analysis.md")
    
    # 4. Cargar noise leakage (RUIDO confirmado)
    print("\n[4/6] Cargando ejemplos de noise leakage...")
    if delta_report.exists():
        noise = load_noise_leakage(delta_report)
        all_examples.extend(noise)
        print(f"  ✓ {len(noise)} ejemplos de RUIDO confirmado")
    
    # 5. Cargar dataset base existente
    print("\n[5/6] Cargando dataset base...")
    base_csv = Path('audit/training_dataset_improved.csv')
    if base_csv.exists():
        with open(base_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            base_rows = list(reader)
        
        print(f"  ✓ {len(base_rows)} ejemplos del dataset base")
        
        # Convertir a formato común
        for row in base_rows:
            text = row.get('text', '')
            label = int(row.get('label', -1))
            
            # Extraer entity y context
            if 'ENTITY:' in text and 'SENTENCE:' in text:
                parts = text.split('SENTENCE:')
                entity = parts[0].replace('ENTITY:', '').strip()
                context = parts[1].strip() if len(parts) > 1 else ''
            else:
                entity = text
                context = ''
            
            all_examples.append({
                'entity': entity,
                'context': context,
                'label': label,
                'source': 'base_dataset',
                'category': row.get('category', 'UNKNOWN')
            })
    else:
        print("  ⚠️ No se encontró training_dataset_improved.csv")
    
    # 6. Balancear y generar CSV final
    print("\n[6/6] Balanceando y generando CSV final...")
    
    # Contar por label
    pii_examples = [e for e in all_examples if e['label'] == 1]
    ruido_examples = [e for e in all_examples if e['label'] == 0]
    
    print(f"\n  Total PII: {len(pii_examples)}")
    print(f"  Total RUIDO: {len(ruido_examples)}")
    
    # Balancear (3:1 ratio PII:RUIDO para favorecer recall)
    target_pii = len(pii_examples)
    target_ruido = target_pii // 3
    
    if len(ruido_examples) > target_ruido:
        ruido_examples = random.sample(ruido_examples, target_ruido)
    
    balanced_examples = pii_examples + ruido_examples
    random.shuffle(balanced_examples)
    
    print(f"\n  Dataset balanceado:")
    print(f"    PII: {len(pii_examples)}")
    print(f"    RUIDO: {len(ruido_examples)}")
    print(f"    Total: {len(balanced_examples)}")
    
    # Crear DataFrame
    rows = []
    for ex in balanced_examples:
        formatted_text = format_training_example(ex['entity'], ex['context'])
        rows.append({
            'text': formatted_text,
            'label': ex['label'],
            'category': ex.get('category', ''),
            'source': ex.get('source', ''),
            'entity': ex['entity'],
            'context': ex['context']
        })
    
    # Guardar con csv.DictWriter
    output_path = Path('audit/training_dataset_recall_optimized.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['text', 'label', 'category', 'source', 'entity', 'context']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✅ Dataset guardado: {output_path}")
    
    # Estadísticas
    pii_count = sum(1 for r in rows if r['label'] == 1)
    ruido_count = sum(1 for r in rows if r['label'] == 0)
    
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS DEL DATASET")
    print("=" * 80)
    print(f"\nTotal ejemplos: {len(rows)}")
    print(f"  - PII (label=1): {pii_count} ({pii_count/len(rows)*100:.1f}%)")
    print(f"  - RUIDO (label=0): {ruido_count} ({ruido_count/len(rows)*100:.1f}%)")
    
    print("\nPor fuente:")
    source_counts = defaultdict(int)
    for r in rows:
        source_counts[r['source']] += 1
    for source, count in source_counts.items():
        print(f"  - {source}: {count}")
    
    print("\nPor categoría (top 10):")
    cat_counts = defaultdict(int)
    for r in rows:
        cat_counts[r['category']] += 1
    for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {cat}: {count}")
    
    return output_path

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    output_file = build_improved_dataset()
    
    print("\n" + "=" * 80)
    print("🎯 PRÓXIMOS PASOS:")
    print("=" * 80)
    print("\n1. Revisar el dataset generado:")
    print(f"   {output_file}")
    print("\n2. Reentrenar el modelo con:")
    print("   python src/pipeline-nuevos-textos/train_gatekeeper_recall.py \\")
    print(f"     --base-csv {output_file} \\")
    print("     --output-dir models/gatekeeper_setfit_recall_v2 \\")
    print("     --target-precision 0.75 \\")
    print("     --epochs 5")
    print("\n3. Evaluar el nuevo modelo con model_comparison_study.py")
    print("=" * 80)
