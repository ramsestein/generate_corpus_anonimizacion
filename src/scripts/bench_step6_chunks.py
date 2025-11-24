#!/usr/bin/env python3
"""
Benchmark de chunking para step6.1 - Evaluación NER con métricas IoU
Genera grid de configuraciones (chunk_size × overlap) y evalúa con matching IoU.
"""

import sys
import json
import re
import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
from tqdm import tqdm
import time

# ============================================================================
# 1. CONFIGURACIÓN Y CARGA DE STEP6.1
# ============================================================================

def load_step61(step61_path: Path):
    """Importa step6.1.py dinámicamente."""
    spec = importlib.util.spec_from_file_location("step6_1_module", step61_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["step6_1_module"] = module
    spec.loader.exec_module(module)
    return module


def extract_entities_with_chunking(step61_module, text: str, pipeline_model,
                                   model_name: str, chunk_size: int, overlap: int, 
                                   confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Wrapper que modifica temporalmente las constantes de chunking en step6.1
    y ejecuta extract_entities_with_model.
    """
    # Guardar valores originales
    orig_chunk = getattr(step61_module, 'CHUNK_TOKEN_SIZE', 128)
    orig_overlap = getattr(step61_module, 'CHUNK_TOKEN_OVERLAP', 32)
    
    # Modificar temporalmente
    step61_module.CHUNK_TOKEN_SIZE = chunk_size
    step61_module.CHUNK_TOKEN_OVERLAP = overlap
    
    try:
        entities = step61_module.extract_entities_with_model(
            text, pipeline_model, model_name, confidence_threshold
        )
        return entities
    finally:
        # Restaurar
        step61_module.CHUNK_TOKEN_SIZE = orig_chunk
        step61_module.CHUNK_TOKEN_OVERLAP = orig_overlap


# ============================================================================
# 2. CARGA DE DOCUMENTOS Y GOLD ANNOTATIONS
# ============================================================================

def load_docs(data_dir: Path, refs_dir: Optional[Path], max_docs: Optional[int],
              file_ext: str = ".txt") -> List[Dict]:
    """Carga documentos con texto y gold spans."""
    
    # Obtener archivos de texto
    text_files = sorted(data_dir.glob(f"*{file_ext}"))
    
    if max_docs:
        text_files = text_files[:max_docs]
    
    if not text_files:
        print(f"ERROR: No se encontraron archivos {file_ext} en {data_dir}")
        sys.exit(1)
    
    docs = []
    for text_file in text_files:
        doc_id = text_file.stem
        
        # Leer texto
        try:
            with open(text_file, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            print(f"WARNING: Error leyendo {text_file}: {e}")
            continue
        
        # Leer gold spans
        if refs_dir and refs_dir.exists():
            gold_spans = parse_gold(refs_dir, doc_id, text)
        else:
            # Si no hay refs_dir, usar fallback: extraer marcas [** ... **] del texto
            gold_spans = extract_bracket_marks(text)
        
        docs.append({
            'doc_id': doc_id,
            'text': text,
            'gold_spans': gold_spans
        })
    
    return docs


def parse_gold(refs_dir: Path, doc_id: str, text: str) -> List[Tuple[int, int, str]]:
    """
    Parsea archivos de referencia gold en diferentes formatos:
    - .ann (Brat format)
    - .json (con campo 'entities': [{'start', 'end', 'label'}])
    - .refs (custom: start,end,label por línea)
    
    Fallback: extrae marcas [** ... **] del texto (marcas de anonimización) como gold.
    
    Returns:
        Lista de tuplas (start, end, label)
    """
    spans = []
    
    # Patrón 1: .ann (Brat format)
    ann_file = refs_dir / f"{doc_id}.ann"
    if ann_file.exists():
        try:
            with open(ann_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith('T'):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        tag_info = parts[1].split()
                        if len(tag_info) >= 3:
                            label = tag_info[0]
                            start = int(tag_info[1])
                            end = int(tag_info[2])
                            spans.append((start, end, label))
            return spans
        except Exception as e:
            print(f"WARNING: Error parseando {ann_file}: {e}")
    
    # Patrón 2: .json
    json_file = refs_dir / f"{doc_id}.json"
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8', errors='replace') as f:
                data = json.load(f)
                if 'entities' in data:
                    for ent in data['entities']:
                        spans.append((ent['start'], ent['end'], ent['label']))
            return spans
        except Exception as e:
            print(f"WARNING: Error parseando {json_file}: {e}")
    
    # Patrón 3: .refs (custom)
    refs_file = refs_dir / f"{doc_id}.refs"
    if refs_file.exists():
        try:
            with open(refs_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Asume formato: start,end,label
                    parts = line.split(',')
                    if len(parts) >= 3:
                        start = int(parts[0])
                        end = int(parts[1])
                        label = parts[2]
                        spans.append((start, end, label))
            return spans
        except Exception as e:
            print(f"WARNING: Error parseando {refs_file}: {e}")
    
    # Fallback: extraer marcas [** ... **] del texto
    spans = extract_bracket_marks(text)
    return spans


def extract_bracket_marks(text: str, default_label: str = "ANONIMIZADO") -> List[Tuple[int, int, str]]:
    """
    Extrae marcas de anonimización en formato [** ... **] del texto y las convierte
    en spans (start, end, label).

    Args:
        text: Texto que contiene marcas como [** Juan Pérez **]
        default_label: Etiqueta por defecto para asignar a las marcas extraídas

    Returns:
        Lista de tuplas (start, end, label)
    """
    spans = []
    pattern = r'\[\*\*[^"]*?\*\*\]'
    # Use a simpler pattern to capture anything between [** and **]
    pattern = r'\[\*\*[^"]*?\*\*\]'
    # More robust: allow any chars except closing bracket sequence
    pattern = r'\[\*\*[^^]*?\*\*\]'
    # However, above attempts may be overly specific; use the standard pattern used elsewhere
    pattern = r'\[\*\*[^"]*?\*\*\]'
    # Final reliable pattern: anything between [** and **]
    pattern = r"\[\*\*[\s\S]*?\*\*\]"

    for match in re.finditer(pattern, text):
        spans.append((match.start(), match.end(), default_label))

    return spans


# ============================================================================
# 3. CÁLCULO DE MÉTRICAS (TP/FP/FN)
# ============================================================================

def calculate_metrics(reference: List[Tuple[int, int, str]], 
                     detected: List[Tuple[int, int, str]], 
                     debug: bool = False, doc_id: str = "") -> Dict:
    """
    Calcula métricas de validación de anonimización.
    
    Lógica correcta (IGUAL QUE test_threshold_optimizer.py):
    - TP (Verdaderos Positivos): Marcas [** ... **] que fueron detectadas por el modelo
    - FP (Falsos Positivos): Detecciones sobre texto NO anonimizado (texto real detectado)
    - FN (Falsos Negativos): Marcas [** ... **] que NO fueron detectadas por el modelo
    
    Args:
        reference: lista de tuplas (start, end, label) - marcas [** ... **] gold
        detected: lista de tuplas (start, end, label) - detecciones del modelo
        debug: si True, imprime información de debug
        doc_id: identificador del documento (para debug)
        
    Returns:
        Dict con tp, fp, fn, precision, recall, f1
    """
    def overlaps(span1: Tuple[int, int, str], span2: Tuple[int, int, str]) -> bool:
        """Verifica si dos spans se solapan (por offsets de caracteres)."""
        start1, end1, _ = span1
        start2, end2, _ = span2
        return not (end1 <= start2 or end2 <= start1)

    if debug:
        print(f"\n[DEBUG METRICS] Doc: {doc_id}")
        print(f"[DEBUG] Total marcas [** ... **] (referencias): {len(reference)}")
        print(f"[DEBUG] Total detecciones del modelo: {len(detected)}")

    # Clasificar referencias: detectadas o no detectadas
    ref_matched: Set[int] = set()  # Índices de referencias que fueron detectadas
    det_matched: Set[int] = set()  # Índices de detecciones que solapan con [** ... **]
    
    # Para cada referencia [** ... **], buscar si hay alguna detección que solape
    for j, ref in enumerate(reference):
        for i, det in enumerate(detected):
            if i in det_matched:
                continue
            if overlaps(det, ref):
                ref_matched.add(j)
                det_matched.add(i)
                if debug and len(ref_matched) <= 3:
                    print(f"[DEBUG] TP: [** ... **] detectada en pos {ref[0]}-{ref[1]}")
                break

    # Clasificar detecciones no emparejadas como FP
    fp_detections = []
    for i, det in enumerate(detected):
        if i not in det_matched:
            fp_detections.append(det)
            if debug and len(fp_detections) <= 5:
                print(f"[DEBUG] FP (texto real detectado): pos {det[0]}-{det[1]}")

    # Clasificar referencias no emparejadas como FN
    fn_references = []
    for j, ref in enumerate(reference):
        if j not in ref_matched:
            fn_references.append(ref)
            if debug and len(fn_references) <= 3:
                print(f"[DEBUG] FN: [** ... **] NO detectada: pos {ref[0]}-{ref[1]}")

    # Calcular métricas
    tp = len(ref_matched)  # Marcas [** ... **] detectadas
    fp = len(fp_detections)  # Detecciones sobre texto real
    fn = len(fn_references)  # Marcas [** ... **] no detectadas
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if debug:
        print(f"[DEBUG] TP ([** ... **] detectadas): {tp}")
        print(f"[DEBUG] FP (texto real detectado): {fp}")
        print(f"[DEBUG] FN ([** ... **] no detectadas): {fn}")
        print(f"[DEBUG] Precision: {precision:.2%}")
        print(f"[DEBUG] Recall: {recall:.2%}")
        print(f"[DEBUG] F1: {f1:.2%}")
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_detected': len(detected),
        'total_reference': len(reference)
    }


def print_desglose(combo_name: str, metrics: Dict):
    """
    Imprime el desglose de métricas en el formato requerido.
    """
    tp = metrics['tp']
    fp = metrics['fp']
    fn = metrics['fn']
    total_ref = tp + fn  # Total de marcas reales
    total_det = tp + fp  # Total de detecciones
    
    # Calcular porcentajes:
    # TP% = TP sobre el total de marcas reales
    # FP% = FP sobre el total de detecciones  
    # FN% = FN sobre el total de marcas reales
    
    tp_pct = (tp / total_ref * 100) if total_ref > 0 else 0.0
    fp_pct = (fp / total_det * 100) if total_det > 0 else 0.0
    fn_pct = (fn / total_ref * 100) if total_ref > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"DESGLOSE PARA {combo_name}:")
    print(f"{'='*80}")
    print(f"  • TP: {tp_pct:.1f}% ({tp}) - Marcas [** ... **] detectadas (correcto)")
    print(f"  • FP: {fp_pct:.1f}% ({fp}) - Texto real detectado (FALLO DE SEGURIDAD)")
    print(f"  • FN: {fn_pct:.1f}% ({fn}) - Marcas [** ... **] no detectadas")
    print(f"\nMÉTRICAS ADICIONALES:")
    print(f"  • Precisión (Precision): {metrics['precision']:.4f} = {metrics['precision']*100:.2f}%")
    print(f"  • Recall (Sensibilidad): {metrics['recall']:.4f} = {metrics['recall']*100:.2f}%")
    print(f"  • F1-score: {metrics['f1']:.4f} = {metrics['f1']*100:.2f}%")
    print(f"\nTOTALES:")
    print(f"  • Total marcas reales (gold): {total_ref}")
    print(f"  • Total detecciones del modelo: {total_det}")
    print(f"{'='*80}\n")


# ============================================================================
# 4. EJECUCIÓN DE COMBINACIONES
# ============================================================================

def run_combo(step61_module, pipeline_meddocan, pipeline_carmen, 
              docs: List[Dict], chunk_size: int, overlap: int,
              models_cfg: str, out_dir: Path, allowed_labels: Optional[Set[str]] = None,
              threshold: float = 0.5) -> Dict:
    """
    Ejecuta una combinación (chunk_size, overlap) sobre todos los docs.
    Calcula métricas TP/FP/FN usando la misma lógica que test_threshold_optimizer.py
    Guarda CSV agregado y JSONL con detalles por documento.
    
    Args:
        threshold: umbral de confianza para filtrar detecciones (mismo que test_threshold_optimizer.py)
    """
    combo_name = f"chunk{chunk_size}_ov{overlap}"
    combo_dir = out_dir / "combos" / combo_name
    combo_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    results_doc = []
    results_combo = []
    
    # Acumuladores globales para métricas agregadas
    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_gold_count = 0
    global_pred_count = 0
    
    # Seleccionar el pipeline según models_cfg
    pipeline_model = pipeline_meddocan if models_cfg == 'meddocan' else pipeline_carmen
    model_name = models_cfg.upper()
    
    pbar = tqdm(docs, desc=combo_name, leave=False)
    for doc in pbar:
        doc_id = doc['doc_id']
        text = doc['text']
        gold_spans = doc['gold_spans']
        
        # Filtrar gold por etiquetas permitidas
        if allowed_labels:
            gold_spans = [(s, e, l) for s, e, l in gold_spans if l in allowed_labels]
        
        # Extraer entidades con step6.1 (usando threshold por defecto del modelo)
        pred_entities = extract_entities_with_chunking(
            step61_module, text, pipeline_model, model_name, chunk_size, overlap, threshold
        )
        
        # Convertir pred_entities a formato (start, end, label)
        pred_spans = []
        for ent in pred_entities:
            # Extraer label (puede venir como entity_group o label)
            label = ent.get('entity_group', ent.get('label', 'UNKNOWN'))
            pred_spans.append((ent['start'], ent['end'], label))
        
        # Filtrar pred por etiquetas permitidas
        if allowed_labels:
            pred_spans = [(s, e, l) for s, e, l in pred_spans if l in allowed_labels]
        
        # CALCULAR MÉTRICAS TP/FP/FN para este documento
        doc_metrics = calculate_metrics(gold_spans, pred_spans, debug=False, doc_id=doc_id)
        
        # Acumular métricas
        global_tp += doc_metrics['tp']
        global_fp += doc_metrics['fp']
        global_fn += doc_metrics['fn']
        global_gold_count += len(gold_spans)
        global_pred_count += len(pred_spans)
        
        # Guardar detalles para análisis posterior
        results_doc.append({
            'doc_id': doc_id,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'text': text,
            'gold_spans': [{'start': s, 'end': e, 'label': l} for s, e, l in gold_spans],
            'pred_spans': [{'start': s, 'end': e, 'label': l} for s, e, l in pred_spans],
            'tp': doc_metrics['tp'],
            'fp': doc_metrics['fp'],
            'fn': doc_metrics['fn'],
            'precision': doc_metrics['precision'],
            'recall': doc_metrics['recall'],
            'f1': doc_metrics['f1']
        })
        
        # Resumen por documento (CSV)
        results_combo.append({
            'doc_id': doc_id,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'gold_count': len(gold_spans),
            'pred_count': len(pred_spans),
            'tp': doc_metrics['tp'],
            'fp': doc_metrics['fp'],
            'fn': doc_metrics['fn'],
            'precision': doc_metrics['precision'],
            'recall': doc_metrics['recall'],
            'f1': doc_metrics['f1']
        })
    
    elapsed = time.time() - start_time
    
    # CALCULAR MÉTRICAS AGREGADAS para toda la combinación
    aggregate_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    aggregate_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    aggregate_f1 = (2 * aggregate_precision * aggregate_recall) / (aggregate_precision + aggregate_recall) \
                   if (aggregate_precision + aggregate_recall) > 0 else 0.0
    
    # Crear métricas agregadas
    aggregate_metrics = {
        'tp': global_tp,
        'fp': global_fp,
        'fn': global_fn,
        'precision': aggregate_precision,
        'recall': aggregate_recall,
        'f1': aggregate_f1
    }
    
    # Guardar JSONL con detalles (para analyze_benchmark.py)
    jsonl_file = combo_dir / "details_doc.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for record in results_doc:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Guardar CSV agregado
    import pandas as pd
    df_combo = pd.DataFrame(results_combo)
    csv_file = combo_dir / "summary_doc.csv"
    df_combo.to_csv(csv_file, index=False)
    
    # IMPRIMIR DESGLOSE CON EL FORMATO REQUERIDO
    print_desglose(combo_name, aggregate_metrics)
    
    print(f"✓ Combinación completada en {elapsed:.2f}s ({len(docs)/elapsed:.2f} docs/s)")
    print(f"  Docs procesados: {len(docs)}")
    
    return {
        'chunk_size': chunk_size,
        'overlap': overlap,
        'num_docs': len(docs),
        'tp': global_tp,
        'fp': global_fp,
        'fn': global_fn,
        'precision': aggregate_precision,
        'recall': aggregate_recall,
        'f1': aggregate_f1,
        'time_s': elapsed,
        'combo_name': combo_name
    }


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark grid chunk_size × overlap para step6.1 con métricas TP/FP/FN"
    )
    parser.add_argument('--data_dir', type=Path, required=True,
                       help="Directorio con documentos .txt")
    parser.add_argument('--refs_dir', type=Path, default=None,
                       help="Directorio con referencias gold (.ann/.json/.refs). "
                            "Si no se proporciona, usa marcas [** ... **] como fallback")
    parser.add_argument('--step61', type=Path, default=None,
                       help="Ruta a step6.1.py (default: pipeline-nuevos-textos/step6.1.py)")
    parser.add_argument('--max_docs', type=int, default=None,
                       help="Máximo número de documentos a procesar")
    parser.add_argument('--chunk_sizes', type=str, default="128,256,512",
                       help="Lista de chunk_sizes separados por coma (ej: 128,256)")
    parser.add_argument('--overlaps', type=str, default="0,32,64",
                       help="Lista de overlaps separados por coma (ej: 0,32)")
    parser.add_argument('--models_cfg', type=str, default="meddocan",
                       help="Configuración de modelos para step6.1 (ej: meddocan)")
    parser.add_argument('--out_dir', type=Path, required=True,
                       help="Directorio de salida para resultados")
    parser.add_argument('--labels', type=str, default=None,
                       help="Filtrar solo estas etiquetas (separadas por coma)")
    parser.add_argument('--threshold', type=float, default=0.5,
                       help="Umbral de confianza para detecciones (default: 0.5)")
    
    args = parser.parse_args()
    
    # Paths
    data_dir = args.data_dir.resolve()
    refs_dir = args.refs_dir.resolve() if args.refs_dir else None
    step61_path = args.step61 if args.step61 else Path("pipeline-nuevos-textos/step6.1.py")
    step61_path = step61_path.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Parsear configuraciones
    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(',')]
    overlaps = [int(x.strip()) for x in args.overlaps.split(',')]
    allowed_labels = set(args.labels.split(',')) if args.labels else None
    
    # Validar combinaciones
    valid_combos = []
    for cs in chunk_sizes:
        for ov in overlaps:
            if ov >= cs:
                print(f"WARNING: Skipping overlap={ov} >= chunk_size={cs}")
                continue
            if cs > 512:
                print(f"WARNING: Skipping chunk_size={cs} > 512 (max BERT)")
                continue
            valid_combos.append((cs, ov))
    
    if not valid_combos:
        print("ERROR: No hay combinaciones válidas")
        sys.exit(1)
    
    # Banner
    print("=" * 80)
    print("BENCHMARK STEP6.1 - GRID CHUNK_SIZE × OVERLAP CON MÉTRICAS TP/FP/FN")
    print("=" * 80)
    print(f"Data dir: {data_dir}")
    print(f"Refs dir: {refs_dir if refs_dir else 'USAR MARCAS [** ... **]'}")
    print(f"Step6.1: {step61_path}")
    print(f"Max docs: {args.max_docs if args.max_docs else 'TODOS'}")
    print(f"Models cfg: {args.models_cfg}")
    print(f"Threshold: {args.threshold}")
    print(f"Out dir: {out_dir}")
    print(f"Labels filter: {allowed_labels if allowed_labels else 'TODAS'}")
    print(f"\nCombinaciones válidas: {len(valid_combos)}")
    for cs, ov in valid_combos:
        print(f"  - chunk_size={cs}, overlap={ov}")
    print("=" * 80)
    
    # Cargar step6.1
    print(f"\nCargando step6.1.py...")
    step61_module = load_step61(step61_path)
    
    print("Configurando modelos...")
    pipeline_meddocan, pipeline_carmen = step61_module.setup_models()
    
    # Cargar documentos
    print("Cargando documentos...")
    docs = load_docs(data_dir, refs_dir, args.max_docs)
    print(f"Documentos cargados: {len(docs)}\n")
    
    # Ejecutar combinaciones
    summary_combos = []
    
    for chunk_size, overlap in valid_combos:
        print("\n" + "=" * 80)
        print(f"PROCESANDO: chunk_size={chunk_size}, overlap={overlap}")
        print("=" * 80)
        
        combo_result = run_combo(
            step61_module, pipeline_meddocan, pipeline_carmen,
            docs, chunk_size, overlap,
            args.models_cfg, out_dir, allowed_labels, args.threshold
        )
        summary_combos.append(combo_result)
    
    # Guardar resumen global
    import pandas as pd
    df_summary = pd.DataFrame(summary_combos)
    summary_file = out_dir / "summary_combos.csv"
    df_summary.to_csv(summary_file, index=False)
    
    # ========================================================================
    # RESUMEN COMPARATIVO: ORDENADO POR F1-SCORE (DE MEJOR A PEOR)
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO POR CHUNKING (ordenado por F1 de mayor a menor)")
    print("=" * 80)
    
    # Ordenar por F1-score descendente
    summary_combos_sorted = sorted(summary_combos, key=lambda x: x['f1'], reverse=True)
    
    print(f"\n{'RANKING':<8} {'CHUNK_SIZE':<12} {'OVERLAP':<10} {'F1':<10} {'PRECISION':<12} {'RECALL':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-" * 100)
    
    for idx, result in enumerate(summary_combos_sorted, 1):
        print(f"{idx:<8} "
              f"{result['chunk_size']:<12} "
              f"{result['overlap']:<10} "
              f"{result['f1']:.4f}    "
              f"{result['precision']:.4f}      "
              f"{result['recall']:.4f}    "
              f"{result['tp']:<6} "
              f"{result['fp']:<6} "
              f"{result['fn']:<6}")
    
    print("\n" + "=" * 80)
    print("MEJOR CONFIGURACIÓN:")
    print("=" * 80)
    best = summary_combos_sorted[0]
    print(f"  • Configuración: chunk_size={best['chunk_size']}, overlap={best['overlap']}")
    print(f"  • F1-score: {best['f1']:.4f} ({best['f1']*100:.2f}%)")
    print(f"  • Precision: {best['precision']:.4f} ({best['precision']*100:.2f}%)")
    print(f"  • Recall: {best['recall']:.4f} ({best['recall']*100:.2f}%)")
    print(f"  • TP={best['tp']}, FP={best['fp']}, FN={best['fn']}")
    print("=" * 80)
    
    print(f"\nArchivo resumen guardado: {summary_file}")
    print(f"Combinaciones procesadas: {len(summary_combos)}")
    
    print("\n" + "=" * 80)
    print("✓ BENCHMARK COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
