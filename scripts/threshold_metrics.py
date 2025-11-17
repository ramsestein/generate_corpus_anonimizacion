#!/usr/bin/env python3
"""
Script para analizar y recalcular métricas de evaluación por threshold.
Ayuda a escoger el mejor threshold según distintos criterios.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Carga el CSV de métricas por threshold.
    
    Args:
        csv_path: Ruta al archivo CSV
        
    Returns:
        DataFrame con las métricas cargadas
    """
    if not csv_path.exists():
        print(f"ERROR: No se encontró el archivo {csv_path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ CSV cargado: {len(df)} filas, {len(df.columns)} columnas")
        return df
    except Exception as e:
        print(f"ERROR al leer CSV: {e}")
        sys.exit(1)


def recalc_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula todas las métricas desde las columnas básicas (tp, fp, fn, total_reference, total_detected).
    
    Añade columnas con prefijo 'recalc_' y columnas 'diff_' para comparar con originales.
    
    Args:
        df: DataFrame con columnas básicas
        
    Returns:
        DataFrame extendido con métricas recalculadas
    """
    df = df.copy()
    
    # Verificar columnas requeridas
    required = ['tp', 'fp', 'fn', 'total_reference', 'total_detected']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Faltan columnas requeridas: {missing}")
        sys.exit(1)
    
    # Recalcular porcentajes
    # tp_pct = (tp / total_reference) * 100
    df['recalc_tp_pct'] = (df['tp'] / df['total_reference']) * 100
    
    # fp_pct = (fp / total_detected) * 100
    # Este es el criterio elegido: FP sobre el total de detecciones
    df['recalc_fp_pct'] = (df['fp'] / df['total_detected']) * 100
    
    # fn_pct = (fn / total_reference) * 100
    df['recalc_fn_pct'] = (df['fn'] / df['total_reference']) * 100
    
    # Tasa de fallo (riesgo) = fp / total_detected
    df['recalc_tasa_fallo_riesgo'] = df['fp'] / df['total_detected']
    
    # Precisión de anonimización = 1 - tasa_fallo_riesgo
    # (Proporción de detecciones que son sobre marcas [** ... **])
    df['recalc_precision_anonimizacion'] = 1.0 - df['recalc_tasa_fallo_riesgo']
    
    # Métricas tradicionales
    # Precision = tp / (tp + fp)
    df['recalc_precision'] = df['tp'] / (df['tp'] + df['fp'])
    df['recalc_precision'] = df['recalc_precision'].fillna(0.0)
    
    # Recall = tp / (tp + fn)
    df['recalc_recall'] = df['tp'] / (df['tp'] + df['fn'])
    df['recalc_recall'] = df['recalc_recall'].fillna(0.0)
    
    # F1 = 2 * precision * recall / (precision + recall)
    df['recalc_f1'] = 2 * df['recalc_precision'] * df['recalc_recall'] / (df['recalc_precision'] + df['recalc_recall'])
    df['recalc_f1'] = df['recalc_f1'].fillna(0.0)
    
    # Calcular diferencias con columnas originales (si existen)
    for metric in ['tp_pct', 'fp_pct', 'fn_pct', 'tasa_fallo_riesgo', 'precision_anonimizacion', 'precision', 'recall', 'f1']:
        if metric in df.columns:
            df[f'diff_{metric}'] = df[metric] - df[f'recalc_{metric}']
    
    return df


def print_diff_summary(df: pd.DataFrame):
    """
    Imprime un resumen de las diferencias entre métricas originales y recalculadas.
    """
    diff_cols = [col for col in df.columns if col.startswith('diff_')]
    
    if not diff_cols:
        print("\nNo hay columnas originales para comparar.")
        return
    
    print("\n" + "=" * 80)
    print("RESUMEN DE DIFERENCIAS (original - recalculado)")
    print("=" * 80)
    
    for col in diff_cols:
        metric_name = col.replace('diff_', '')
        max_diff = df[col].abs().max()
        mean_diff = df[col].abs().mean()
        print(f"  {metric_name:30s}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    
    print("=" * 80)


def select_best_threshold(df: pd.DataFrame, objective: str, alpha: float = 0.5, 
                          beta: float = 0.3, gamma: float = 0.2, recall_min: float = 0.80) -> tuple:
    """
    Selecciona el mejor threshold según el objetivo especificado.
    
    Args:
        df: DataFrame con métricas recalculadas
        objective: Criterio de selección ('f1_max', 'precision_max', 'recall_max', 
                   'riesgo_min', 'fp_pct_min', 'custom', 'recall_precision_tradeoff')
        alpha, beta, gamma: Pesos para el objetivo 'custom'
        recall_min: Umbral mínimo de recall para 'recall_precision_tradeoff'
        
    Returns:
        Tupla (mejor_fila, df_ordenado)
    """
    df = df.copy()
    
    if objective == 'f1_max':
        df['_score'] = df['recalc_f1']
        ascending = False
        
    elif objective == 'precision_max':
        df['_score'] = df['recalc_precision']
        ascending = False
        
    elif objective == 'recall_max':
        df['_score'] = df['recalc_recall']
        ascending = False
        
    elif objective == 'riesgo_min':
        df['_score'] = df['recalc_tasa_fallo_riesgo']
        ascending = True
        
    elif objective == 'fp_pct_min':
        df['_score'] = df['recalc_fp_pct']
        ascending = True
        
    elif objective == 'custom':
        # score_custom = alpha * f1 + beta * precision - gamma * fp_pct
        df['_score'] = (alpha * df['recalc_f1'] + 
                       beta * df['recalc_precision'] - 
                       gamma * df['recalc_fp_pct'])
        df['score_custom'] = df['_score']  # Guardar para export
        ascending = False
    
    elif objective == 'recall_precision_tradeoff':
        """
        CRITERIO: Priorizar recall suficiente, luego maximizar precisión.
        
        Lógica:
        1. Priorizamos recall >= recall_min para reducir FN (falsos negativos).
           No queremos perder demasiadas marcas [** ... **] sin detectar.
        
        2. Entre los thresholds que cumplen ese mínimo de recall, 
           elegimos el de mejor precisión (menos FP, menos texto real detectado).
        
        3. Si hay empates en precisión, desempatamos con:
           - Mayor F1 score
           - Menor fp_pct (o menor tasa_fallo_riesgo)
        
        4. Si NINGÚN threshold alcanza recall_min:
           - Mostramos aviso claro.
           - Fallback: elegimos el de mejor F1 global (compromiso).
        """
        # Filtrar candidatos por recall mínimo
        candidates = df[df['recalc_recall'] >= recall_min].copy()
        
        if candidates.empty:
            # Ningún threshold cumple el recall mínimo
            print("\n" + "!" * 100)
            print("⚠️  ADVERTENCIA: NINGÚN threshold alcanza el recall mínimo requerido!")
            print(f"   Recall mínimo solicitado: {recall_min:.2f}")
            print(f"   Recall máximo disponible: {df['recalc_recall'].max():.4f}")
            print("   → Usando FALLBACK: se seleccionará el threshold con mejor F1 global.")
            print("!" * 100 + "\n")
            
            # Fallback: ordenar por F1 descendente
            df['_score'] = df['recalc_f1']
            df_sorted = df.sort_values('_score', ascending=False).reset_index(drop=True)
            best_row = df_sorted.iloc[0]
            return best_row, df_sorted
        
        else:
            # Hay candidatos: ordenar por precisión (desc), F1 (desc), fp_pct (asc)
            # Nota: sort_values ordena con [False, False, True] = [desc, desc, asc]
            candidates_sorted = candidates.sort_values(
                ['recalc_precision', 'recalc_f1', 'recalc_fp_pct'],
                ascending=[False, False, True]
            ).reset_index(drop=True)
            
            best_row = candidates_sorted.iloc[0]
            
            # Para mantener consistencia con otros objetivos, devolvemos candidates_sorted
            # (solo los que cumplen recall_min, ordenados por nuestro criterio)
            return best_row, candidates_sorted
        
    else:
        print(f"ERROR: Objetivo '{objective}' no reconocido.")
        print("Opciones válidas: f1_max, precision_max, recall_max, riesgo_min, fp_pct_min, custom, recall_precision_tradeoff")
        sys.exit(1)
    
    # Ordenar por score (para objetivos distintos de recall_precision_tradeoff)
    df_sorted = df.sort_values('_score', ascending=ascending).reset_index(drop=True)
    best_row = df_sorted.iloc[0]
    
    return best_row, df_sorted


def print_best_threshold_summary(best_row: pd.Series, objective: str, recall_min: float = None):
    """
    Imprime un resumen formateado del mejor threshold.
    
    Args:
        best_row: Serie con los datos del mejor threshold
        objective: Criterio utilizado para la selección
        recall_min: Umbral mínimo de recall usado (solo para recall_precision_tradeoff)
    """
    threshold = best_row['threshold']
    tp = int(best_row['tp'])
    fp = int(best_row['fp'])
    fn = int(best_row['fn'])
    
    tp_pct = best_row['recalc_tp_pct']
    fp_pct = best_row['recalc_fp_pct']
    fn_pct = best_row['recalc_fn_pct']
    
    tasa_fallo = best_row['recalc_tasa_fallo_riesgo'] * 100
    precision_anon = best_row['recalc_precision_anonimizacion'] * 100
    
    precision = best_row['recalc_precision']
    recall = best_row['recalc_recall']
    f1 = best_row['recalc_f1']
    
    # Mapeo de objetivos a descripción
    obj_desc = {
        'f1_max': 'máximo F1',
        'precision_max': 'máxima precisión',
        'recall_max': 'máximo recall',
        'riesgo_min': 'menor riesgo',
        'fp_pct_min': 'menor % FP',
        'custom': 'score custom',
        'recall_precision_tradeoff': f'recall suficiente (≥{recall_min:.2f}) y mejor precisión'
    }
    
    desc = obj_desc.get(objective, objective)
    
    print("\n" + "=" * 100)
    print(f"MEJOR THRESHOLD ({desc}): {threshold}")
    print()
    print("  MÉTRICAS CRÍTICAS DE ANONIMIZACIÓN:")
    print("  ┌" + "─" * 96 + "┐")
    print(f"  │ Tasa de Fallo (Riesgo):      {tasa_fallo:6.2f}%  - Detecciones sobre texto real" + " " * 28 + "│")
    print(f"  │ Precisión de Anonimización:  {precision_anon:6.2f}%  - Detecciones sobre [** ... **]" + " " * 22 + "│")
    print("  └" + "─" * 96 + "┘")
    print()
    print("  DESGLOSE:")
    print(f"  • TP: {tp_pct:5.1f}% ({tp:,}) - Marcas [** ... **] detectadas (correcto)")
    print(f"  • FP: {fp_pct:5.1f}% ({fp:,})   - Texto real detectado (FALLO DE SEGURIDAD)")
    print(f"  • FN: {fn_pct:5.1f}% ({fn:,})  - Marcas [** ... **] no detectadas")
    print()
    print(f"  Métricas tradicionales: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    print("=" * 100)


def print_top_thresholds(df_sorted: pd.DataFrame, objective: str, top_n: int = 5):
    """
    Imprime una tabla con los top N thresholds según el criterio.
    
    Args:
        df_sorted: DataFrame ordenado por el criterio
        objective: Criterio utilizado
        top_n: Número de thresholds a mostrar
    """
    print(f"\n{'=' * 100}")
    print(f"TOP {top_n} THRESHOLDS (ordenados por {objective})")
    print('=' * 100)
    
    cols_display = ['threshold', 'recalc_f1', 'recalc_precision', 'recalc_recall', 
                   'recalc_fp_pct', 'recalc_tasa_fallo_riesgo', 'tp', 'fp', 'fn']
    
    # Asegurar que todas las columnas existen
    cols_available = [col for col in cols_display if col in df_sorted.columns]
    
    df_top = df_sorted.head(top_n)[cols_available].copy()
    
    # Formatear para display
    if 'recalc_f1' in df_top.columns:
        df_top['F1'] = df_top['recalc_f1'].apply(lambda x: f"{x:.4f}")
    if 'recalc_precision' in df_top.columns:
        df_top['Precision'] = df_top['recalc_precision'].apply(lambda x: f"{x:.4f}")
    if 'recalc_recall' in df_top.columns:
        df_top['Recall'] = df_top['recalc_recall'].apply(lambda x: f"{x:.4f}")
    if 'recalc_fp_pct' in df_top.columns:
        df_top['FP%'] = df_top['recalc_fp_pct'].apply(lambda x: f"{x:.2f}")
    if 'recalc_tasa_fallo_riesgo' in df_top.columns:
        df_top['Riesgo'] = df_top['recalc_tasa_fallo_riesgo'].apply(lambda x: f"{x:.4f}")
    
    # Seleccionar columnas para display final
    display_cols = ['threshold']
    for col in ['F1', 'Precision', 'Recall', 'FP%', 'Riesgo', 'tp', 'fp', 'fn']:
        if col in df_top.columns:
            display_cols.append(col)
    
    print(df_top[display_cols].to_string(index=False))
    print('=' * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Analiza métricas de threshold y selecciona el mejor según distintos criterios"
    )
    
    parser.add_argument(
        '--csv-path',
        type=Path,
        default=Path('threshold_results/threshold_comparison.csv'),
        help='Ruta al CSV con métricas por threshold'
    )
    
    parser.add_argument(
        '--objective',
        type=str,
        default='f1_max',
        choices=['f1_max', 'precision_max', 'recall_max', 'riesgo_min', 'fp_pct_min', 'custom', 'recall_precision_tradeoff'],
        help='Criterio para seleccionar el mejor threshold'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Peso de F1 en objetivo custom (default: 0.5)'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=0.3,
        help='Peso de Precision en objetivo custom (default: 0.3)'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.2,
        help='Peso de FP%% en objetivo custom (default: 0.2)'
    )
    
    parser.add_argument(
        '--recall-min',
        type=float,
        default=0.75,
        help='Umbral mínimo de recall para el objetivo recall_precision_tradeoff (default: 0.80)'
    )
    
    parser.add_argument(
        '--out-csv',
        type=Path,
        default=None,
        help='Ruta para guardar CSV extendido con métricas recalculadas'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Número de top thresholds a mostrar (default: 5)'
    )
    
    args = parser.parse_args()
    
    # 1. Cargar datos
    print("\n" + "=" * 100)
    print("ANÁLISIS DE THRESHOLDS")
    print("=" * 100)
    df = load_data(args.csv_path)
    
    # 2. Recalcular métricas
    print("\nRecalculando métricas...")
    df = recalc_metrics(df)
    print("✓ Métricas recalculadas")
    
    # 3. Mostrar resumen de diferencias
    print_diff_summary(df)
    
    # 4. Seleccionar mejor threshold
    print(f"\nSeleccionando mejor threshold según criterio: {args.objective}")
    if args.objective == 'custom':
        print(f"  Pesos: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    elif args.objective == 'recall_precision_tradeoff':
        print(f"  Recall mínimo requerido: {args.recall_min:.2f}")
        print(f"  Estrategia: Filtrar por recall ≥ {args.recall_min:.2f}, luego maximizar precisión")
    
    best_row, df_sorted = select_best_threshold(
        df, args.objective, args.alpha, args.beta, args.gamma, args.recall_min
    )
    
    # 5. Imprimir resumen del mejor threshold
    print_best_threshold_summary(best_row, args.objective, args.recall_min)
    
    # 6. Imprimir top N thresholds
    print_top_thresholds(df_sorted, args.objective, args.top_n)
    
    # 7. Guardar CSV extendido si se solicita
    if args.out_csv:
        # Eliminar columna temporal _score si existe
        if '_score' in df_sorted.columns:
            df_sorted = df_sorted.drop(columns=['_score'])
        
        df_sorted.to_csv(args.out_csv, index=False)
        print(f"\n✓ CSV extendido guardado en: {args.out_csv}")
    
    print("\n✓ Análisis completado")


if __name__ == "__main__":
    main()