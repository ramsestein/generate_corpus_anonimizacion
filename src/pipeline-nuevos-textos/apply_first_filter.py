#!/usr/bin/env python3
"""
APLICACIÓN DE FILTRO DETERMINISTA (PRIMERA CAPA)
=================================================

Aplica EntityFastFilter a todas las entidades del JSON de resultados NER,
generando decisiones deterministas que reducen dramáticamente las llamadas al LLM.

FLUJO:
------
1. Carga test_results.json con entidades detectadas por NER
2. Para cada entidad,aplica EntityFastFilter.evaluate_candidate()
3. Genera outputs/first_filter_results.json con decisiones:
   - FORCE_ANONYMIZE: Anonimizar directamente (whitelist)
   - FORCE_IGNORE: Ignorar directamente (blacklist no-person)
   - ESCALATE_TO_LLM: Requiere validación semántica del LLM

Este script NO llama al LLM, solo aplica reglas deterministas.

USO:
----
python apply_first_filter.py --input outputs/test_results.json --output outputs/first_filter_results.json

AUTOR: Pipeline Anonimización Clínica
"""

import argparse
import json
import os
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Importar EntityFastFilter
try:
    from entity_fast_filter import EntityFastFilter, EnumDecision
except ImportError:
    # Si no está en el PATH, intentar importar desde directorio actual
    sys.path.insert(0, os.path.dirname(__file__))
    from entity_fast_filter import EntityFastFilter, EnumDecision


# ============================================================================
# FUNCIONES DE LOGGING
# ============================================================================

def log_message(message: str, level: str = "INFO"):
    """Imprime mensajes con timestamp y nivel."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def log_info(message: str):
    """Log de nivel INFO."""
    log_message(message, "INFO")


def log_warn(message: str):
    """Log de nivel WARN."""
    log_message(message, "WARN")


def log_error(message: str):
    """Log de nivel ERROR."""
    log_message(message, "ERROR")


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_ner_results(input_path: str) -> List[Dict]:
    """
    Carga el archivo JSON con resultados del NER.
    
    Args:
        input_path: Ruta al archivo JSON de resultados NER
        
    Returns:
        Lista de entidades detectadas por NER
    """
    log_info(f"Cargando resultados NER desde: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("El archivo debe contener una lista JSON")
    
    log_info(f"  → Cargadas {len(data)} entidades")
    return data


# ============================================================================
# APLICACIÓN DEL FILTRO
# ============================================================================

def apply_first_filter(
    ner_results: List[Dict],
    entity_filter: EntityFastFilter,
    verbose: bool = False
) -> List[Dict]:
    """
    Aplica el filtro determinista a todas las entidades.
    
    Args:
        ner_results: Lista de entidades detectadas por NER
        entity_filter: Instancia de EntityFastFilter
        verbose: Si True, muestra información detallada
        
    Returns:
        Lista de entidades con decisiones del filtro
    """
    log_info("Aplicando filtro determinista a las entidades...")
    
    filtered_results = []
    stats = {
        "FORCE_ANONYMIZE": 0,
        "FORCE_IGNORE": 0,
        "ESCALATE_TO_LLM": 0,
        "SKIPPED": 0
    }
    
    for idx, entity in enumerate(ner_results, 1):
        try:
            # Extraer campos requeridos
            document_id = entity.get("document_id", entity.get("doc_id", "unknown"))
            entity_text = entity.get("keyword", entity.get("entity_text", entity.get("text", "")))
            ner_label = entity.get("label", entity.get("ner_label", ""))
            start = entity.get("start", -1)
            end = entity.get("end", -1)
            
            # Validar que tenemos los campos mínimos
            if not entity_text or not ner_label:
                log_warn(f"  → Entidad {idx} omitida: falta entity_text o label")
                stats["SKIPPED"] += 1
                continue
            
            # Aplicar filtro determinista
            decision = entity_filter.evaluate_candidate(entity_text, ner_label)
            decision_str = decision.name
            
            # Incrementar estadísticas
            stats[decision_str] += 1
            
            # Crear registro filtrado
            filtered_entity = {
                "document_id": document_id,
                "entity_text": entity_text.strip(),
                "ner_label": ner_label,
                "decision": decision_str,
                "start": start,
                "end": end
            }
            
            # Añadir información adicional si está disponible
            if "context" in entity:
                filtered_entity["context"] = entity["context"]
            
            filtered_results.append(filtered_entity)
            
            # Logging verbose
            if verbose and idx % 100 == 0:
                log_info(f"  → Procesadas {idx}/{len(ner_results)} entidades...")
                
        except Exception as e:
            log_warn(f"  → Error procesando entidad {idx}: {e}")
            stats["SKIPPED"] += 1
            continue
    
    # Mostrar estadísticas
    log_info(f"  → Procesamiento completado:")
    log_info(f"     - FORCE_ANONYMIZE: {stats['FORCE_ANONYMIZE']}")
    log_info(f"     - FORCE_IGNORE:    {stats['FORCE_IGNORE']}")
    log_info(f"     - ESCALATE_TO_LLM: {stats['ESCALATE_TO_LLM']}")
    log_info(f"     - SKIPPED:         {stats['SKIPPED']}")
    
    total_processed = stats["FORCE_ANONYMIZE"] + stats["FORCE_IGNORE"] + stats["ESCALATE_TO_LLM"]
    if total_processed > 0:
        llm_reduction = (1 - stats["ESCALATE_TO_LLM"] / total_processed) * 100
        log_info(f"  → Reducción de llamadas LLM: {llm_reduction:.1f}%")
    
    return filtered_results


# ============================================================================
# EXPORTACIÓN
# ============================================================================

def save_filtered_results(results: List[Dict], output_path: str):
    """
    Guarda los resultados filtrados en un archivo JSON.
    
    Args:
        results: Lista de entidades con decisiones
        output_path: Ruta del archivo de salida
    """
    log_info(f"Guardando resultados filtrados en: {output_path}")
    
    # Crear directorio si no existe
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Guardar JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    log_info(f"  → Guardadas {len(results)} entidades filtradas")


# ============================================================================
# CLI Y MAIN
# ============================================================================

def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Aplica filtro determinista (EntityFastFilter) a entidades NER.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Aplicar filtro sin listas (solo ESCALATE_TO_LLM)
  python apply_first_filter.py --input outputs/test_results.json --output outputs/first_filter_results.json

  # Con listas de whitelist/blacklist
  python apply_first_filter.py -i outputs/test_results.json -o outputs/first_filter_results.json \\
      --whitelist data/hospitales.json data/lugares.json \\
      --blacklist data/medicamentos.json data/patologias.json

  # Con CIE10 para filtrado de patologías médicas
  python apply_first_filter.py -i outputs/test_results.json -o outputs/first_filter_results.json \\
      --whitelist data/hospitales.json data/lugares.json \\
      --blacklist data/medicamentos.json data/patologias.json \\
      --cie10 LISTAS/cie10.xls

  # Modo verbose
  python apply_first_filter.py -i outputs/test_results.json -o outputs/first_filter_results.json -v
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Ruta al JSON con resultados NER (ej: outputs/test_results.json)"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Ruta del JSON de salida con decisiones filtradas"
    )
    
    parser.add_argument(
        "--whitelist", "-w",
        nargs="+",
        help="Rutas a JSON de whitelist (hospitales, lugares) - case sensitive"
    )
    
    parser.add_argument(
        "--blacklist", "-b",
        nargs="+",
        help="Rutas a JSON de blacklist (medicamentos, patologías) - case insensitive"
    )
    
    parser.add_argument(
        "--cie10", "-c",
        help="Ruta al archivo Excel CIE10 (.xls o .xlsx) con códigos de diagnósticos médicos"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Muestra información detallada del procesamiento"
    )
    
    return parser.parse_args()


def main():
    """Función principal del script."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("APLICACIÓN DE FILTRO DETERMINISTA (PRIMERA CAPA)")
    print("="*70 + "\n")
    
    try:
        # 1. Inicializar EntityFastFilter
        log_info("Inicializando EntityFastFilter...")
        
        entity_filter = EntityFastFilter(
            whitelist_paths=args.whitelist,
            blacklist_paths=args.blacklist,
            cie10_path=args.cie10
        )
        
        stats = entity_filter.get_stats()
        log_info(f"  → {entity_filter}")
        log_info(f"     - WhiteList terms: {stats['whitelist_terms']}")
        log_info(f"     - BlackList terms: {stats['blacklist_terms']}")
        if stats.get('cie10_loaded'):
            log_info(f"     - CIE10 terms: {stats['cie10_terms']}")
        
        if stats['whitelist_terms'] == 0 and stats['blacklist_terms'] == 0:
            log_warn("  ⚠ No se cargaron listas, todas las entidades irán a ESCALATE_TO_LLM")
        
        # 2. Cargar resultados NER
        ner_results = load_ner_results(args.input)
        
        # 3. Aplicar filtro
        filtered_results = apply_first_filter(
            ner_results,
            entity_filter,
            verbose=args.verbose
        )
        
        # 4. Guardar resultados
        save_filtered_results(filtered_results, args.output)
        
        log_info("✅ Proceso completado exitosamente")
        print("\n" + "="*70 + "\n")
        return 0
        
    except FileNotFoundError as e:
        log_error(f"Archivo no encontrado: {e}")
        return 1
    except ValueError as e:
        log_error(f"Error de validación: {e}")
        return 1
    except Exception as e:
        log_error(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
