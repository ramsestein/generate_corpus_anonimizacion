#!/usr/bin/env python3
"""
Script para explorar el mejor confidence_threshold probando varios valores
sobre una muestra (primeros 200 documentos) del directorio de anónimos.

Calcula métricas (VP, FP, FN, Precision, Recall, F1) para cada threshold
comparando detecciones con entidades de referencia (marcas [** ... **]).

Las marcas [** ... **] indican spans ya anonimizados (silver-standard) que
se utilizan como referencia gold para evaluar la calidad del modelo.

Nota: este script importa dinámicamente `pipeline-nuevos-textos/step6.1.py`
para reutilizar `setup_models` y `extract_entities_with_model` sin asumir
que la carpeta es un paquete Python.
"""

import os
import argparse
import importlib.util
import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Set

THRESHOLDS = [0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
MAX_DOCS = 100
DEFAULT_BASE_THRESHOLD = 0.01  # threshold usado para extraer todas las detecciones inicialmente


def anonymize_text_with_spans(text: str, spans: List[Dict], token: str = "JJJ") -> str:
    """
    Anonimiza un texto reemplazando los spans especificados por el token dado.
    
    Args:
        text: Texto original
        spans: Lista de dicts con 'start' y 'end' (y opcionalmente 'label')
        token: Token de anonimización (por defecto "JJJ")
        
    Returns:
        Texto con spans reemplazados por el token
    """
    if not text or not spans:
        return text
    
    # Convertir spans a tuplas (start, end) y ordenar de derecha a izquierda
    span_tuples = []
    for span in spans:
        start = int(span.get('start', 0))
        end = int(span.get('end', 0))
        # Validar índices
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        if end > start:
            span_tuples.append((start, end))
    
    # Ordenar de derecha a izquierda para no desplazar índices
    span_tuples.sort(reverse=True)
    
    # Reemplazar
    result = text
    for start, end in span_tuples:
        result = result[:start] + token + result[end:]
    
    return result


def load_step6_module(repo_root: Path):
    """Carga dinámicamente el módulo step6.1.py desde pipeline-nuevos-textos."""
    module_path = repo_root / "pipeline-nuevos-textos" / "step6.1.py"
    if not module_path.exists():
        raise FileNotFoundError(f"No se encontró {module_path}")

    spec = importlib.util.spec_from_file_location("step6_mod", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def extract_reference_entities_from_brackets(text: str) -> List[Dict]:
    """
    Extrae las entidades de referencia de las marcas [** ... **] en el texto.
    
    Las marcas [** ... **] indican spans ya anonimizados (silver-standard) que
    se utilizarán como referencia para calcular métricas.
    
    Ejemplo:
        Input: "Paciente [**Juan Pérez**] nació en [**1970**]"
        Output: [
            {'start': 9, 'end': 25, 'text': '[**Juan Pérez**]'},
            {'start': 36, 'end': 46, 'text': '[**1970**]'}
        ]
    
    Args:
        text: texto con marcas [** ... **]
        
    Returns:
        Lista de dicts con start, end, text de cada marca [** ... **]
    """
    pattern = r'\[\*\*[^\]]*\*\*\]'
    matches = []
    for match in re.finditer(pattern, text):
        matches.append({
            'start': match.start(),
            'end': match.end(),
            'text': match.group()
        })
    return matches


def extract_reference_entities(text: str) -> List[Dict]:
    """
    Extrae las entidades de referencia del texto anonimizado.
    
    Esta función ahora delega a extract_reference_entities_from_brackets()
    para usar las marcas [** ... **] como referencia.
    
    Args:
        text: texto anonimizado con marcas [** ... **]
        
    Returns:
        Lista de dicts con start, end, text de cada marca [** ... **]
    """
    return extract_reference_entities_from_brackets(text)


def consolidate_entities(entities: List[Dict], threshold: float) -> List[Dict]:
    """
    Consolida una lista de entidades detectadas (posiblemente solapadas por chunking).

    - Filtra por score >= threshold.
    - Ordena por start.
    - Agrupa entidades solapadas y mantiene la de mayor score por grupo.

    Args:
        entities: lista de dicts con al menos 'start', 'end', 'score' (o 'confidence').
        threshold: umbral de confianza para filtrar detecciones.

    Returns:
        Lista consolidada de entidades.
    """
    if not entities:
        return []

    # Normalizar claves y filtrar
    cleaned = []
    for e in entities:
        score = float(e.get('score', e.get('confidence', 0.0) or 0.0))
        start = int(e.get('start', 0) or 0)
        end = int(e.get('end', 0) or 0)
        cleaned.append({
            'orig': e,
            'start': start,
            'end': end,
            'score': score
        })

    # Filtrar por threshold
    filtered = [c for c in cleaned if c['score'] >= threshold]

    # Ordenar por start asc, score desc para dar preferencia a mayor confianza
    filtered.sort(key=lambda x: (x['start'], -x['score']))

    if not filtered:
        return []

    # Agrupar solapamientos y elegir la mejor por grupo
    consolidated = []
    current_group = [filtered[0]]
    group_end = filtered[0]['end']

    for item in filtered[1:]:
        if item['start'] <= group_end:
            # solapa con el grupo actual
            current_group.append(item)
            group_end = max(group_end, item['end'])
        else:
            # procesar grupo actual
            best = max(current_group, key=lambda x: x['score'])
            consolidated.append(best['orig'])
            # iniciar nuevo grupo
            current_group = [item]
            group_end = item['end']

    # procesar último grupo
    if current_group:
        best = max(current_group, key=lambda x: x['score'])
        consolidated.append(best['orig'])

    return consolidated


def calculate_metrics(reference: List[Dict], detected: List[Dict], debug: bool = False, doc_id: str = "") -> Dict:
    """
    Calcula métricas de validación de anonimización.
    
    Lógica correcta:
    - TP (Verdaderos Positivos): Marcas [** ... **] que fueron detectadas por el modelo
    - FP (Falsos Positivos): Detecciones sobre texto NO anonimizado (texto real detectado)
    - FN (Falsos Negativos): Marcas [** ... **] que NO fueron detectadas por el modelo
    
    Args:
        reference: entidades de referencia (marcas [** ... **])
        detected: entidades detectadas por los modelos
        debug: si True, imprime información de debug
        doc_id: identificador del documento (para debug)
        
    Returns:
        Dict con tp, fp, fn, precision, recall, f1
    """
    def overlaps(e1: Dict, e2: Dict) -> bool:
        """Verifica si dos entidades se solapan (por offsets de caracteres)."""
        return not (e1['end'] <= e2['start'] or e2['end'] <= e1['start'])

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
                    print(f"[DEBUG] TP: [** ... **] detectada en pos {ref['start']}-{ref['end']} por detección: {det.get('word', '')[:30]}")
                break

    # Clasificar detecciones no emparejadas como FP
    fp_detections = []
    for i, det in enumerate(detected):
        if i not in det_matched:
            fp_detections.append(det)
            if debug and len(fp_detections) <= 5:
                print(f"[DEBUG] FP (texto real detectado): {det.get('word', det.get('text', ''))[:30]} (score={det.get('score', 0):.2f})")

    # Clasificar referencias no emparejadas como FN
    fn_references = []
    for j, ref in enumerate(reference):
        if j not in ref_matched:
            fn_references.append(ref)
            if debug and len(fn_references) <= 3:
                print(f"[DEBUG] FN: [** ... **] NO detectada: pos {ref['start']}-{ref['end']} '{ref['text']}'")

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


def find_document_ids(input_dir: Path, max_docs: int) -> List[str]:
    files = sorted([p.name for p in input_dir.glob('*.txt')])
    doc_ids = []
    for name in files:
        if name.endswith('.txt.txt'):
            doc_ids.append(name[:-8])
        elif name.endswith('.txt'):
            doc_ids.append(name[:-4])
        else:
            doc_ids.append(Path(name).stem)
        if len(doc_ids) >= max_docs:
            break
    return doc_ids


def main():
    parser = argparse.ArgumentParser(description='Test threshold optimizer (simulación).')
    parser.add_argument('--input-dir', default='corpus/output/aws2',
                        help='Directorio con documentos (por defecto una carpeta en corpus)')
    parser.add_argument('--max-docs', type=int, default=MAX_DOCS,
                        help=f'Máximo de documentos a procesar (por defecto {MAX_DOCS})')
    parser.add_argument('--debug', action='store_true',
                        help='Activar modo debug con prints detallados')
    parser.add_argument('--print-anon', action='store_true',
                        help='Incluir textos anonimizados en las salidas CSV/JSON')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = (repo_root / args.input_dir).resolve()

    if not input_dir.exists():
        print(f"ERROR: input-dir no encontrado: {input_dir}")
        return

    print(f"Usando input-dir: {input_dir}")

    # Cargar módulo step6.1
    step6 = load_step6_module(repo_root)

    # Cargar modelos (se hace una vez)
    print("Cargando modelos (esto puede tardar)...")
    pipeline_meddocan, pipeline_carmen = step6.setup_models()
    print("Modelos cargados.")

    # Listar documentos
    doc_ids = find_document_ids(input_dir, args.max_docs)
    print(f"Procesando {len(doc_ids)} documentos (muestra).\n")

    # Primera pasada: procesar cada documento UNA vez y almacenar todas las detecciones y referencias
    all_entities_by_doc = {}
    all_references_by_doc = {}
    all_texts_by_doc = {}  # Almacenar textos originales para anonimización
    
    print("Procesando documentos...")
    for idx, doc_id in enumerate(doc_ids, 1):
        txt_path = input_dir / f"{doc_id}.txt"
        if not txt_path.exists():
            txt_path = input_dir / f"{doc_id}.txt.txt"
        try:
            text = txt_path.read_text(encoding='utf-8').strip()
        except Exception:
            print(f"  WARNING: no se pudo leer {doc_id}, se salta")
            continue

        if idx % 10 == 0:
            print(f"  Procesado {idx}/{len(doc_ids)} documentos...")

        # Almacenar texto original si se requiere anonimización
        if args.print_anon:
            all_texts_by_doc[doc_id] = text

        # Extraer detecciones una sola vez con umbral base (incluir todo)
        med_entities = step6.extract_entities_with_model(text, pipeline_meddocan, 'MEDDOCAN', confidence_threshold=DEFAULT_BASE_THRESHOLD)
        car_entities = step6.extract_entities_with_model(text, pipeline_carmen, 'CARMEN', confidence_threshold=DEFAULT_BASE_THRESHOLD)
        all_entities = med_entities + car_entities
        all_entities_by_doc[doc_id] = all_entities
        
        if args.debug and idx == 1:
            print(f"\n[DEBUG] Documento: {doc_id}")
            print(f"[DEBUG] MEDDOCAN detectó {len(med_entities)} entidades")
            for i, ent in enumerate(med_entities[:3]):  # Mostrar solo las 3 primeras
                print(f"  [DEBUG] MEDDOCAN entity {i}: {ent}")
            print(f"[DEBUG] CARMEN detectó {len(car_entities)} entidades")
            for i, ent in enumerate(car_entities[:3]):
                print(f"  [DEBUG] CARMEN entity {i}: {ent}")
        
        # Extraer entidades de referencia (marcas [** ... **])
        reference_entities = extract_reference_entities(text)
        all_references_by_doc[doc_id] = reference_entities
        
        if args.debug and idx == 1:
            print(f"[DEBUG] Referencias (marcas [** ... **]): {len(reference_entities)}")
            for i, ref in enumerate(reference_entities[:3]):
                print(f"  [DEBUG] Referencia {i}: {ref}")
            print()
    
    print(f"Completado procesamiento de {len(all_entities_by_doc)} documentos.\n")

    # Crear carpeta de resultados
    results_dir = repo_root / 'scripts' / 'threshold_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_metrics = []

    # Para cada threshold, consolidar las detecciones y calcular métricas
    print("Calculando métricas por threshold...\n")
    for thresh in THRESHOLDS:
        print(f"Procesando threshold: {thresh}")
        
        # Acumuladores globales
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_detected = 0
        total_reference = 0
        per_threshold_results = {}

        for doc_id, entities in all_entities_by_doc.items():
            # Consolidar detecciones con el threshold actual
            consolidated = consolidate_entities(entities, threshold=thresh)
            
            # Obtener referencias para este documento
            references = all_references_by_doc.get(doc_id, [])
            
            # Generar textos anonimizados si está habilitado
            text_anon_pred = ""
            text_anon_gold = ""
            if args.print_anon and doc_id in all_texts_by_doc:
                original_text = all_texts_by_doc[doc_id]
                text_anon_pred = anonymize_text_with_spans(original_text, consolidated, token="JJJ")
                text_anon_gold = anonymize_text_with_spans(original_text, references, token="JJJ")
            
            # Debug: mostrar detalles solo para el primer documento y primer threshold
            is_debug = args.debug and doc_id == doc_ids[0] and thresh == THRESHOLDS[0]
            
            if is_debug:
                print(f"\n[DEBUG] Procesando {doc_id} con threshold {thresh}")
                print(f"[DEBUG] Referencias (marcas [** ... **]): {len(references)}")
                print(f"[DEBUG] Entidades consolidadas: {len(consolidated)}")
            
            # Calcular métricas para este documento
            doc_metrics = calculate_metrics(references, consolidated, debug=is_debug, doc_id=doc_id)
            
            # Acumular
            total_tp += doc_metrics['tp']
            total_fp += doc_metrics['fp']
            total_fn += doc_metrics['fn']
            total_detected += doc_metrics['total_detected']
            total_reference += doc_metrics['total_reference']
            
            # Guardar resultados por documento
            per_threshold_results[doc_id] = {
                'metrics': doc_metrics,
                'entities_count': len(consolidated),
                'text_anon_pred': text_anon_pred,
                'text_anon_gold': text_anon_gold
            }
            

        # Calcular métricas globales
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calcular porcentajes
        tp_pct = (total_tp / total_reference * 100) if total_reference > 0 else 0.0
        fp_pct = (total_fp / total_detected * 100) if total_detected > 0 else 0.0
        fn_pct = (total_fn / total_reference * 100) if total_reference > 0 else 0.0
        
        # MÉTRICAS CRÍTICAS DE ANONIMIZACIÓN
        # 1. Tasa de Fallo (Riesgo): % de detecciones que son sobre texto NO anonimizado
        tasa_fallo = (total_fp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        
        # 2. Precisión de Anonimización (Éxito): % de detecciones sobre [** ... **] (correctas)
        precision_anon = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        
        metrics_summary = {
            'threshold': thresh,
            'tp': total_tp,
            'tp_pct': tp_pct,
            'fp': total_fp,
            'fp_pct': fp_pct,
            'fn': total_fn,
            'fn_pct': fn_pct,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tasa_fallo_riesgo': tasa_fallo,
            'precision_anonimizacion': precision_anon,
            'total_detected': total_detected,
            'total_reference': total_reference,
            'num_docs': len(all_entities_by_doc)
        }
        
        summary_metrics.append(metrics_summary)
        
        # Guardar resultados detallados por threshold en JSON
        out_file = results_dir / f"threshold_{str(thresh).replace('.', '_')}.json"
        try:
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'threshold': thresh,
                    'global_metrics': metrics_summary,
                    'per_document': per_threshold_results
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  ERROR guardando JSON para threshold {thresh}: {e}")

        print(f"  TP={total_tp} ({tp_pct:.1f}%), FP={total_fp} ({fp_pct:.1f}%), FN={total_fn} ({fn_pct:.1f}%)")
        print(f"  Tasa Fallo (Riesgo)={tasa_fallo:.2%}, Precisión Anon={precision_anon:.2%}")
        print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n")

    # Guardar resumen en CSV
    csv_path = results_dir / 'threshold_comparison.csv'
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['threshold', 'tp', 'tp_pct', 'fp', 'fp_pct', 'fn', 'fn_pct',
                         'tasa_fallo_riesgo', 'precision_anonimizacion',
                         'precision', 'recall', 'f1', 'total_detected', 'total_reference', 'num_docs']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in summary_metrics:
                writer.writerow(metrics)
        print(f"✓ Resultados guardados en: {csv_path}")
    except Exception as e:
        print(f"ERROR guardando CSV: {e}")

    # Resumen final en consola
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS POR THRESHOLD")
    print("="*80)
    print(f"Documentos procesados: {len(all_entities_by_doc)}")
    print(f"Thresholds probados: {len(THRESHOLDS)}")
    print(f"Total referencias (marcas [** ... **]): {sum(len(refs) for refs in all_references_by_doc.values())}")
    print("\n")
    
    # Tabla resumen
    print(f"{'Threshold':<10} {'TP':<12} {'FP':<12} {'FN':<12} {'Tasa Fallo':<12} {'Prec. Anon':<12} {'F1':<10}")
    print("-" * 100)
    for m in summary_metrics:
        print(f"{m['threshold']:<10.2f} {m['tp']:>5} ({m['tp_pct']:>5.1f}%) "
              f"{m['fp']:>5} ({m['fp_pct']:>5.1f}%) "
              f"{m['fn']:>5} ({m['fn_pct']:>5.1f}%) "
              f"{m['tasa_fallo_riesgo']:<12.2%} {m['precision_anonimizacion']:<12.2%} {m['f1']:<10.4f}")
        
    MIN_RECALL = 0.80  
    candidatos = [m for m in summary_metrics if m['recall'] >= MIN_RECALL]

  
        # entre los que cumplen, elige el de MAYOR precisión; si empatan, el más alto
    best_threshold = max(candidatos, key=lambda m: (m['precision'], m['threshold']))
    



    print("\n" + "="*100)
    print(f"MEJOR THRESHOLD (menor riesgo): {best_threshold['threshold']:.2f}")
    print(f"")
    print(f"  MÉTRICAS CRÍTICAS DE ANONIMIZACIÓN:")
    print(f"  ┌─────────────────────────────────────────────────────────────────────────────────┐")
    print(f"  │ Tasa de Fallo (Riesgo):      {best_threshold['tasa_fallo_riesgo']:>6.2%}  - Detecciones sobre texto real   │")
    print(f"  │ Precisión de Anonimización:  {best_threshold['precision_anonimizacion']:>6.2%}  - Detecciones sobre [** ... **] │")
    print(f"  └─────────────────────────────────────────────────────────────────────────────────┘")
    print(f"")
    print(f"  DESGLOSE:")
    print(f"  • TP: {best_threshold['tp_pct']:.1f}% ({best_threshold['tp']:,}) - Marcas [** ... **] detectadas (correcto)")
    print(f"  • FP: {best_threshold['fp_pct']:.1f}% ({best_threshold['fp']:,}) - Texto real detectado (FALLO DE SEGURIDAD)")
    print(f"  • FN: {best_threshold['fn_pct']:.1f}% ({best_threshold['fn']:,}) - Marcas [** ... **] no detectadas")
    print(f"")
    print(f"  Métricas tradicionales: Precision={best_threshold['precision']:.4f}, Recall={best_threshold['recall']:.4f}, F1={best_threshold['f1']:.4f}")
    print("="*100)


if __name__ == '__main__':
    main()
