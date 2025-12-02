#!/usr/bin/env python3
"""
Calcula métricas detalladas (TP, FP, TN, FN) para los resultados de SetFit.
Compara las predicciones contra el ground truth del dataset.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Configuración
PROJECT_ROOT = Path(__file__).parent.parent.parent
SETFIT_RESULTS_CSV = PROJECT_ROOT / "outputs" / "setfit_context_resultados_20251202_081634.csv"
GROUND_TRUTH_FILE = PROJECT_ROOT / "entidades-procesadas-para-metricas.json"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "setfit_detailed_metrics.json"

def load_ground_truth(file_path):
    """Carga el ground truth desde el archivo JSON."""
    print(f"📂 Cargando ground truth desde: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # El archivo contiene un dict con 'entities'
    if isinstance(data, dict) and 'entities' in data:
        entities = data['entities']
    elif isinstance(data, list):
        entities = data
    else:
        raise ValueError("Formato de ground truth no reconocido")
    
    print(f"  ✅ Cargadas {len(entities)} entidades del ground truth")
    return entities


def build_ground_truth_labels(ground_truth_entities):
    """
    Construye un diccionario para mapear entidades a su clasificación binaria.
    
    Para cada entidad en el ground truth, determinamos si debe ser PII (1) o Ruido (0).
    Basándonos en la etiqueta NER de MEDDOCAN, todas las entidades detectadas
    son consideradas PII (1).
    """
    print("\n🔨 Construyendo mapeo de ground truth...")
    
    gt_labels = {}
    
    for ent in ground_truth_entities:
        doc_id = ent.get('doc_id', ent.get('document_id', ''))
        text = ent.get('text', ent.get('entity_text', '')).strip()
        label = ent.get('label', ent.get('ner_label', ''))
        
        # Crear clave única: doc_id + texto + label
        key = f"{doc_id}||{text}||{label}"
        
        # En MEDDOCAN, todas las entidades detectadas son PII
        # Pero podemos tener un campo explícito si existe
        if 'ground_truth' in ent or 'is_pii' in ent:
            is_pii = int(ent.get('ground_truth', ent.get('is_pii', 1)))
        else:
            # Por defecto, las entidades de MEDDOCAN son PII
            is_pii = 1
        
        gt_labels[key] = is_pii
    
    print(f"  ✅ Mapeadas {len(gt_labels)} entidades")
    print(f"  📊 PII: {sum(1 for v in gt_labels.values() if v == 1)}")
    print(f"  📊 Ruido: {sum(1 for v in gt_labels.values() if v == 0)}")
    
    return gt_labels


def calculate_metrics(csv_file, gt_labels):
    """Calcula métricas comparando predicciones SetFit con ground truth."""
    print(f"\n📊 Calculando métricas desde: {csv_file}")
    
    # Cargar resultados CSV
    df = pd.read_csv(csv_file)
    print(f"  ✅ Cargadas {len(df)} predicciones")
    
    y_true = []
    y_pred = []
    matched = 0
    unmatched = 0
    
    for idx, row in df.iterrows():
        doc_id = row['document_id']
        text = str(row['entity_text']).strip()
        label = row['ner_label']
        
        # SetFit prediction: 1 = PII, 0 = Ruido
        setfit_pred = int(row['setfit_prediction'])
        
        # Crear clave única
        key = f"{doc_id}||{text}||{label}"
        
        # Buscar en ground truth
        if key in gt_labels:
            gt_label = gt_labels[key]
            y_true.append(gt_label)
            y_pred.append(setfit_pred)
            matched += 1
        else:
            # Si no está en GT, asumir que es PII (MEDDOCAN detectó todo como PII)
            y_true.append(1)
            y_pred.append(setfit_pred)
            unmatched += 1
    
    print(f"  🔗 Matched con GT: {matched}")
    print(f"  ⚠️  Sin match en GT: {unmatched}")
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Classification report
    class_report = classification_report(
        y_true, y_pred,
        target_names=['Ruido (0)', 'PII (1)'],
        output_dict=True
    )
    
    # Resultados
    metrics = {
        'total_samples': len(y_true),
        'matched_with_gt': matched,
        'unmatched_with_gt': unmatched,
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
        },
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        },
        'classification_report': class_report,
        'class_distribution': {
            'ground_truth': {
                'pii': int(sum(y_true)),
                'ruido': int(len(y_true) - sum(y_true))
            },
            'predictions': {
                'pii': int(sum(y_pred)),
                'ruido': int(len(y_pred) - sum(y_pred))
            }
        }
    }
    
    return metrics, y_true, y_pred


def analyze_errors(csv_file, y_true, y_pred):
    """Analiza los errores por categoría NER."""
    print("\n🔍 Analizando errores por categoría...")
    
    df = pd.read_csv(csv_file)
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    
    # Identificar tipos de error
    df['error_type'] = 'Correct'
    df.loc[(df['y_true'] == 1) & (df['y_pred'] == 0), 'error_type'] = 'False Negative (FN)'
    df.loc[(df['y_true'] == 0) & (df['y_pred'] == 1), 'error_type'] = 'False Positive (FP)'
    
    # Agrupar por etiqueta NER
    error_analysis = {}
    
    for label in df['ner_label'].unique():
        mask = df['ner_label'] == label
        label_df = df[mask]
        
        total = len(label_df)
        tp = len(label_df[(label_df['y_true'] == 1) & (label_df['y_pred'] == 1)])
        fp = len(label_df[(label_df['y_true'] == 0) & (label_df['y_pred'] == 1)])
        fn = len(label_df[(label_df['y_true'] == 1) & (label_df['y_pred'] == 0)])
        tn = len(label_df[(label_df['y_true'] == 0) & (label_df['y_pred'] == 0)])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        error_analysis[label] = {
            'total': total,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
    
    return error_analysis


def print_summary(metrics, error_analysis):
    """Imprime resumen de métricas."""
    print("\n" + "=" * 70)
    print("📊 MÉTRICAS DETALLADAS DE SETFIT")
    print("=" * 70)
    
    cm = metrics['confusion_matrix']
    m = metrics['metrics']
    
    print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
    print(f"  True Positives (TP):  {cm['true_positives']:4d}  ✅ PII correctamente identificado")
    print(f"  False Positives (FP): {cm['false_positives']:4d}  ❌ Ruido clasificado como PII")
    print(f"  False Negatives (FN): {cm['false_negatives']:4d}  ❌ PII clasificado como Ruido")
    print(f"  True Negatives (TN):  {cm['true_negatives']:4d}  ✅ Ruido correctamente filtrado")
    
    print(f"\n📈 MÉTRICAS GLOBALES:")
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}  (de lo que predice PII, cuánto es correcto)")
    print(f"  Recall:    {m['recall']:.4f}  (de todo el PII real, cuánto detecta)")
    print(f"  F1-Score:  {m['f1_score']:.4f}  (balance entre precision y recall)")
    
    print(f"\n📊 DISTRIBUCIÓN:")
    gt = metrics['class_distribution']['ground_truth']
    pred = metrics['class_distribution']['predictions']
    print(f"  Ground Truth: PII={gt['pii']}, Ruido={gt['ruido']}")
    print(f"  Predicciones: PII={pred['pii']}, Ruido={pred['ruido']}")
    
    print(f"\n🏷️  TOP 5 PEORES CATEGORÍAS (por F1):")
    sorted_errors = sorted(error_analysis.items(), key=lambda x: x[1]['f1'])
    for label, stats in sorted_errors[:5]:
        print(f"  {label:30s} F1={stats['f1']:.3f} | TP={stats['tp']:3d} FP={stats['fp']:3d} FN={stats['fn']:3d}")
    
    print(f"\n🏆 TOP 5 MEJORES CATEGORÍAS (por F1):")
    for label, stats in sorted_errors[-5:][::-1]:
        print(f"  {label:30s} F1={stats['f1']:.3f} | TP={stats['tp']:3d} FP={stats['fp']:3d} FN={stats['fn']:3d}")
    
    print("=" * 70)


def main():
    print("🚀 Calculando métricas detalladas de SetFit...\n")
    
    # Verificar que existen los archivos
    if not SETFIT_RESULTS_CSV.exists():
        print(f"❌ No se encuentra el archivo de resultados: {SETFIT_RESULTS_CSV}")
        return 1
    
    if not GROUND_TRUTH_FILE.exists():
        print(f"❌ No se encuentra el archivo de ground truth: {GROUND_TRUTH_FILE}")
        return 1
    
    # Cargar ground truth
    gt_entities = load_ground_truth(GROUND_TRUTH_FILE)
    gt_labels = build_ground_truth_labels(gt_entities)
    
    # Calcular métricas
    metrics, y_true, y_pred = calculate_metrics(SETFIT_RESULTS_CSV, gt_labels)
    
    # Analizar errores por categoría
    error_analysis = analyze_errors(SETFIT_RESULTS_CSV, y_true, y_pred)
    
    # Imprimir resumen
    print_summary(metrics, error_analysis)
    
    # Guardar resultados
    output_data = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'files': {
            'setfit_results': str(SETFIT_RESULTS_CSV),
            'ground_truth': str(GROUND_TRUTH_FILE),
        },
        'metrics': metrics,
        'error_analysis_by_ner_label': error_analysis,
    }
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Métricas guardadas en: {OUTPUT_FILE}")
    
    return 0


if __name__ == '__main__':
    exit(main())
