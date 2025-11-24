#!/usr/bin/env python3
"""
CÁLCULO DE MÉTRICAS PARA EVALUACIÓN DE LLM JUDGE
=================================================

Script independiente para calcular métricas de evaluación comparando
las predicciones del LLM contra la verdad terreno (etiquetas manuales).

ENTRADA:
--------
JSON con resultados del LLM que debe contener:
- manual_correction (o manual_label, manual, ground_truth): Verdad terreno
- llm_bool (o llm_response, llm_prediction, is_valid): Predicción del LLM

SALIDA:
-------
CSV con métricas calculadas (TP, FP, FN, TN, precision, recall, F1, etc.)

USO:
----
python compute_llm_metrics.py --json outputs/test_ollama_results.json --output outputs/llm_judge_metrics.csv
"""

import os
import sys
import csv
import json
import argparse
import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass


# ============================================================================
# CONFIGURACIÓN DE NOMBRES DE COLUMNAS
# ============================================================================

# Nombres posibles para la columna de verdad terreno (manual)
MANUAL_COLUMN_NAMES = ['manual_correction', 'Corrección manual', 'manual_label', 'manual', 'ground_truth', 'correccion_manual']

# Nombres posibles para la columna de predicción del LLM
LLM_COLUMN_NAMES = ['llm_bool', 'llm_response', 'llm_prediction', 'is_valid']


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class EvaluationMetrics:
    """
    Métricas de evaluación para clasificación binaria.
    
    Compara predicciones del LLM contra verdad terreno (manual_correction).
    
    Attributes:
        tp: True Positives (manual=TRUE y llm=TRUE)
        fp: False Positives (manual=FALSE y llm=TRUE)
        fn: False Negatives (manual=TRUE y llm=FALSE)
        tn: True Negatives (manual=FALSE y llm=FALSE)
        total: Total de ejemplos válidos evaluados
    """
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    total: int = 0
    
    @property
    def tp_pct(self) -> float:
        """Porcentaje de True Positives sobre el total"""
        return (self.tp / self.total * 100) if self.total > 0 else 0.0
    
    @property
    def fp_pct(self) -> float:
        """Porcentaje de False Positives sobre el total"""
        return (self.fp / self.total * 100) if self.total > 0 else 0.0
    
    @property
    def fn_pct(self) -> float:
        """Porcentaje de False Negatives sobre el total"""
        return (self.fn / self.total * 100) if self.total > 0 else 0.0
    
    @property
    def tn_pct(self) -> float:
        """Porcentaje de True Negatives sobre el total"""
        return (self.tn / self.total * 100) if self.total > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """
        Precision = TP / (TP + FP)

        "De las que el LLM dijo TRUE, ¿cuántas eran correctas?"
        """
        denominator = self.tp + self.fp
        numerator = self.tp
        return (numerator / denominator) if denominator > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """
        Recall = TP / (TP + FN)

        "De las que son TRUE, ¿cuántas detectó el LLM?"
        """
        numerator = self.tp
        denominator = self.tp + self.fn
        return (numerator / denominator) if denominator > 0 else 0.0
    
    @property
    def f1(self) -> float:
        """
        F1-Score = 2 * (precision * recall) / (precision + recall)
        
        Media armónica de precision y recall.
        """
        denominator = self.precision + self.recall
        return (2 * self.precision * self.recall / denominator) if denominator > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convierte las métricas a diccionario para exportación"""
        return {
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'tn': self.tn,
            'tp_pct': round(self.tp_pct, 2),
            'fp_pct': round(self.fp_pct, 2),
            'fn_pct': round(self.fn_pct, 2),
            'tn_pct': round(self.tn_pct, 2),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1': round(self.f1, 4),
            'total': self.total
        }


# ============================================================================
# FUNCIONES DE LOGGING
# ============================================================================

def log_message(message: str, level: str = "INFO"):
    """
    Imprime mensajes con timestamp y nivel de logging.
    
    Args:
        message: Mensaje a imprimir
        level: Nivel de log (DEBUG, INFO, WARN, ERROR)
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ============================================================================
# FUNCIONES DE PARSEO
# ============================================================================

def parse_bool_value(value: str) -> Optional[bool]:
    """
    Parsea un valor string a booleano.
    
    Valores aceptados para TRUE:
    - TRUE, True, true, 1, SI, Si, si, YES, Yes, yes, CORRECTO, VÁLIDO, VALIDO
    
    Valores aceptados para FALSE:
    - FALSE, False, false, 0, NO, No, no, INCORRECTO, INVÁLIDO, INVALIDO
    
    Args:
        value: String a parsear
        
    Returns:
        True, False o None si no se puede determinar
    """
    if not value or not isinstance(value, str):
        return None
    
    value_clean = value.strip().upper()
    
    # Valores que se consideran TRUE
    if value_clean in ('TRUE', '1', 'SI', 'YES', 'CORRECTO', 'VÁLIDO', 'VALIDO'):
        return True
    # Valores que se consideran FALSE
    elif value_clean in ('FALSE', '0', 'NO', 'INCORRECTO', 'INVÁLIDO', 'INVALIDO'):
        return False
    # Cualquier otro valor → None (se ignorará la fila)
    else:
        return None


# ============================================================================
# FUNCIÓN PRINCIPAL DE CÁLCULO DE MÉTRICAS
# ============================================================================

def calculate_metrics_from_json(json_path: str) -> EvaluationMetrics:
    """
    Lee un JSON con resultados del LLM y calcula métricas de evaluación.
    
    El JSON debe ser una lista de objetos, cada uno conteniendo (como mínimo):
    - Una clave de verdad terreno: manual_correction, manual_label, manual, o ground_truth
    - Una clave de predicción LLM: llm_bool, llm_response, llm_prediction, o is_valid
    
    Args:
        json_path: Ruta al JSON con resultados del LLM
        
    Returns:
        EvaluationMetrics con TP, FP, FN, TN y métricas derivadas
        
    Raises:
        FileNotFoundError: Si el JSON no existe
        ValueError: Si el JSON no tiene las columnas necesarias
    """
    log_message(f"Leyendo JSON: {json_path}")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró el archivo JSON: {json_path}")
    
    metrics = EvaluationMetrics()
    skipped_rows = 0
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # El JSON debe ser una lista de objetos
        if not isinstance(data, list):
            raise ValueError(
                f"El JSON debe ser una lista de objetos.\n"
                f"Tipo recibido: {type(data).__name__}"
            )
        
        if len(data) == 0:
            raise ValueError("El JSON está vacío, no hay datos para procesar")
        
        # Usar el primer objeto para detectar las claves disponibles
        first_item = data[0]
        if not isinstance(first_item, dict):
            raise ValueError(
                f"Cada elemento del JSON debe ser un objeto/diccionario.\n"
                f"Tipo del primer elemento: {type(first_item).__name__}"
            )
        
        # Detectar nombres de columnas
        fieldnames = list(first_item.keys())
        
        # Buscar columna de verdad terreno
        manual_col = None
        for col in MANUAL_COLUMN_NAMES:
            if col in fieldnames:
                manual_col = col
                break
        
        # Buscar columna de predicción LLM
        llm_col = None
        for col in LLM_COLUMN_NAMES:
            if col in fieldnames:
                llm_col = col
                break
        
        # Validar que encontramos ambas columnas
        if not manual_col:
            raise ValueError(
                f"JSON debe contener una clave de verdad terreno.\n"
                f"Claves esperadas: {', '.join(MANUAL_COLUMN_NAMES)}\n"
                f"Claves encontradas: {', '.join(fieldnames)}"
            )
        
        if not llm_col:
            raise ValueError(
                f"JSON debe contener una clave de predicción del LLM.\n"
                f"Claves esperadas: {', '.join(LLM_COLUMN_NAMES)}\n"
                f"Claves encontradas: {', '.join(fieldnames)}"
            )
        
        log_message(f"  → Clave verdad terreno: '{manual_col}'")
        log_message(f"  → Clave predicción LLM: '{llm_col}'")
        
        # Procesar cada objeto del JSON
        for row_num, row in enumerate(data, start=1):
            # Parsear valores booleanos
            manual_value = parse_bool_value(row.get(manual_col, ''))
            llm_value = parse_bool_value(row.get(llm_col, ''))
            
            # Saltar filas sin valores válidos
            if manual_value is None or llm_value is None:
                skipped_rows += 1
                continue
            
            # Contar según la matriz de confusión
            # TP: manual=TRUE y llm=TRUE → Correcto, detectó entidad válida
            if manual_value is True and llm_value is True:
                metrics.tp += 1
            # FP: manual=FALSE y llm=TRUE → Error, falso positivo
            elif manual_value is False and llm_value is True:
                metrics.fp += 1
            # FN: manual=TRUE y llm=FALSE → Error, falso negativo (missed)
            elif manual_value is True and llm_value is False:
                metrics.fn += 1
            # TN: manual=FALSE y llm=FALSE → Correcto, rechazó entidad inválida
            elif manual_value is False and llm_value is False:
                metrics.tn += 1
            
            metrics.total += 1
        
        # Mostrar advertencia si se saltaron filas
        if skipped_rows > 0:
            log_message(
                f"  ⚠ Se saltaron {skipped_rows} objetos sin valores válidos",
                "WARN"
            )
        
        log_message(f"  ✓ Procesadas {metrics.total} entradas válidas")
        
        return metrics
        
    except Exception as e:
        log_message(f"Error calculando métricas: {e}", "ERROR")
        raise


# ============================================================================
# FUNCIONES DE EXPORTACIÓN
# ============================================================================

def save_metrics_to_csv(metrics: EvaluationMetrics, output_path: str):
    """
    Guarda las métricas de evaluación en un archivo CSV.
    
    Genera un CSV con una sola fila que contiene todas las métricas.
    
    Args:
        metrics: Objeto EvaluationMetrics con las métricas calculadas
        output_path: Ruta del archivo CSV de salida
    """
    log_message(f"Guardando métricas en: {output_path}")
    
    try:
        # Crear directorio si no existe
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convertir métricas a diccionario
        metrics_dict = metrics.to_dict()
        
        # Escribir CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = list(metrics_dict.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(metrics_dict)
        
        log_message(f"  ✓ Métricas guardadas exitosamente")
        
    except Exception as e:
        log_message(f"Error guardando métricas: {e}", "ERROR")
        raise


def print_metrics_summary(metrics: EvaluationMetrics):
    """
    Imprime resumen de las métricas de evaluación en consola.
    
    Args:
        metrics: Objeto EvaluationMetrics con las métricas calculadas
    """
    print(f"\n{'='*70}")
    print(f"MÉTRICAS DE EVALUACIÓN")
    print(f"{'='*70}")
    
    print(f"\n📊 MATRIZ DE CONFUSIÓN")
    print(f"  Total de ejemplos evaluados: {metrics.total}")
    print(f"")
    print(f"  True Positives (TP):   {metrics.tp:5d} ({metrics.tp_pct:5.1f}%)")
    print(f"  False Positives (FP):  {metrics.fp:5d} ({metrics.fp_pct:5.1f}%)")
    print(f"  False Negatives (FN):  {metrics.fn:5d} ({metrics.fn_pct:5.1f}%)")
    print(f"  True Negatives (TN):   {metrics.tn:5d} ({metrics.tn_pct:5.1f}%)")
    
    print(f"\n📈 MÉTRICAS DE RENDIMIENTO")
    print(f"  Precision: {metrics.precision:.4f} ({metrics.precision*100:.2f}%)")
    print(f"  Recall:    {metrics.recall:.4f} ({metrics.recall*100:.2f}%)")
    print(f"  F1-Score:  {metrics.f1:.4f} ({metrics.f1*100:.2f}%)")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# CLI Y MAIN
# ============================================================================

def main():
    """Función principal del script"""
    parser = argparse.ArgumentParser(
        description='Cálculo de métricas de evaluación para LLM Judge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Calcular métricas desde JSON de resultados del LLM
  python compute_llm_metrics.py --json outputs/test_ollama_results.json --output outputs/llm_judge_metrics.csv
  
  # Con rutas relativas
  python compute_llm_metrics.py --json outputs/llm_results.json --output outputs/metrics.csv

CLAVES REQUERIDAS EN EL JSON DE ENTRADA:
=========================================

El JSON debe ser una lista de objetos, cada uno conteniendo estas claves
(el script las detecta automáticamente):

Verdad terreno (ground truth) - al menos una de:
  - manual_correction
  - manual_label
  - manual
  - ground_truth

Predicción del LLM - al menos una de:
  - llm_bool
  - llm_response
  - llm_prediction
  - is_valid

Valores aceptados: true/false, TRUE/FALSE, 1/0, SI/NO, YES/NO, etc.

MÉTRICAS CALCULADAS:
====================

- TP (True Positive):   manual=TRUE y llm=TRUE   → Correcto
- FP (False Positive):  manual=FALSE y llm=TRUE  → Falso positivo
- FN (False Negative):  manual=TRUE y llm=FALSE  → Falso negativo (missed)
- TN (True Negative):   manual=FALSE y llm=FALSE → Correcto

- Precision = TP / (TP + FP)
- Recall    = TP / (TP + FN)
- F1        = 2 * (Precision * Recall) / (Precision + Recall)
        """
    )
    
    parser.add_argument(
        '--json',
        required=True,
        help='Ruta al JSON con resultados del LLM (debe contener manual_correction y llm_bool o equivalentes)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Ruta de salida para el CSV de métricas'
    )
    
    args = parser.parse_args()
    
    try:
        log_message("="*70)
        log_message("CÁLCULO DE MÉTRICAS DE EVALUACIÓN")
        log_message("="*70)
        
        # 1. Calcular métricas desde JSON
        metrics = calculate_metrics_from_json(args.json)
        
        # 2. Imprimir resumen en consola
        print_metrics_summary(metrics)
        
        # 3. Guardar resultados en CSV
        save_metrics_to_csv(metrics, args.output)
        
        log_message("="*70)
        log_message("✓ CÁLCULO DE MÉTRICAS COMPLETADO")
        log_message("="*70)
        
    except Exception as e:
        log_message(f"Error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
