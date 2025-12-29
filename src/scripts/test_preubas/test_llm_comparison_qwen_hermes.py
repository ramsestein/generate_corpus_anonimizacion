#!/usr/bin/env python3
"""
Test de comparación: Qwen vs Hermes en casos donde SetFit falla.

Este script prueba ambos LLMs (qwen2.5:7b y hermes3:8b) en casos específicos
donde SetFit clasificó mal (falsos positivos o falsos negativos).

El LLM actúa como "rescate" después de SetFit, así que este test mide
cuál LLM es mejor para corregir los errores del modelo SetFit.

Uso:
    python test_llm_comparison_qwen_hermes.py
    python test_llm_comparison_qwen_hermes.py --debug
    python test_llm_comparison_qwen_hermes.py --test-id 5
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Añadir paths necesarios
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "pipeline-nuevos-textos"))

from llm_judge import run_llm_judge


# =============================================================================
# CASOS DE PRUEBA: Donde SetFit FALLA
# =============================================================================

# Estos son casos reales donde SetFit comete errores y necesita ayuda del LLM
SETFIT_FAILURE_CASES = [
    # CASO 1: Falso Positivo - "dolores" como síntoma, no nombre
    {
        "id": 1,
        "category": "FALSE_POSITIVE",
        "entity": "dolores",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "context": "Paciente refiere dolores abdominales intensos desde ayer",
        "expected_decision": "FILTER",  # No es PII, es síntoma
        "description": "SetFit detecta 'dolores' como NOMBRE pero es un síntoma",
        "setfit_wrongly_classified_as": "PII"
    },
    
    # CASO 2: Falso Positivo - "Mercedes" como marca de vehículo
    {
        "id": 2,
        "category": "FALSE_POSITIVE",
        "entity": "Mercedes",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "context": "Traslado realizado en ambulancia marca Mercedes",
        "expected_decision": "FILTER",
        "description": "SetFit detecta 'Mercedes' como NOMBRE pero es marca de vehículo",
        "setfit_wrongly_classified_as": "PII"
    },
    
    # CASO 3: Falso Positivo - "Madrid" como territorio genérico
    {
        "id": 3,
        "category": "FALSE_POSITIVE",
        "entity": "Madrid",
        "label": "TERRITORIO",
        "context": "Paciente procedente de Madrid",
        "expected_decision": "FILTER",
        "description": "SetFit detecta 'Madrid' como PII pero es ubicación genérica",
        "setfit_wrongly_classified_as": "PII"
    },
    
    # CASO 4: Falso Positivo - Número como valor clínico
    {
        "id": 4,
        "category": "FALSE_POSITIVE",
        "entity": "72",
        "label": "EDAD_SUJETO_ASISTENCIA",
        "context": "Presión arterial 72/40 mmHg, hipotensión severa",
        "expected_decision": "FILTER",
        "description": "SetFit detecta '72' como EDAD pero es valor de presión arterial",
        "setfit_wrongly_classified_as": "PII"
    },
    
    # CASO 5: Falso Positivo - Referencia genérica a familia
    {
        "id": 5,
        "category": "FALSE_POSITIVE",
        "entity": "familia",
        "label": "FAMILIARES_SUJETO_ASISTENCIA",
        "context": "Antecedentes de diabetes en la familia",
        "expected_decision": "FILTER",
        "description": "SetFit detecta 'familia' como PII pero es referencia genérica",
        "setfit_wrongly_classified_as": "PII"
    },
    
    # CASO 6: Falso Positivo - Letra suelta de apartado
    {
        "id": 6,
        "category": "FALSE_POSITIVE",
        "entity": "h",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "context": "Ver apartado h del informe de laboratorio",
        "expected_decision": "FILTER",
        "description": "SetFit detecta 'h' como NOMBRE pero es letra de sección",
        "setfit_wrongly_classified_as": "PII"
    },
    
    # CASO 7: Falso Positivo - Fragmento malformado de fecha
    {
        "id": 7,
        "category": "FALSE_POSITIVE",
        "entity": ".08.2024",
        "label": "FECHAS",
        "context": "Error en OCR genera fragmento .08.2024 incompleto",
        "expected_decision": "FILTER",
        "description": "SetFit detecta '.08.2024' como FECHA pero es fragmento malformado",
        "setfit_wrongly_classified_as": "PII"
    },
    
    # CASO 8: Verdadero Positivo que SetFit podría perder - Nombre con apellido
    {
        "id": 8,
        "category": "FALSE_NEGATIVE",
        "entity": "Dolores García",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "context": "La paciente Dolores García acude a consulta",
        "expected_decision": "KEEP",
        "description": "SetFit podría confundir 'Dolores' pero con apellido es claramente un nombre",
        "setfit_wrongly_classified_as": "RUIDO"
    },
    
    # CASO 9: Verdadero Positivo - Hospital específico
    {
        "id": 9,
        "category": "FALSE_NEGATIVE",
        "entity": "Hospital San Carlos de Madrid",
        "label": "HOSPITAL",
        "context": "Ingresa en Hospital San Carlos de Madrid por urgencias",
        "expected_decision": "KEEP",
        "description": "SetFit podría rechazar por tener 'Madrid' pero es nombre específico de hospital",
        "setfit_wrongly_classified_as": "RUIDO"
    },
    
    # CASO 10: Verdadero Positivo - Fecha con contexto médico
    {
        "id": 10,
        "category": "FALSE_NEGATIVE",
        "entity": "15/03/2024",
        "label": "FECHAS",
        "context": "Alta hospitalaria prevista para el 15/03/2024",
        "expected_decision": "KEEP",
        "description": "SetFit podría rechazar fecha pero tiene contexto médico identificable",
        "setfit_wrongly_classified_as": "RUIDO"
    },
    
    # CASO 11: Falso Positivo - Plural genérico de cuidadores
    {
        "id": 11,
        "category": "FALSE_POSITIVE",
        "entity": "cuidadores",
        "label": "FAMILIARES_SUJETO_ASISTENCIA",
        "context": "Educación sanitaria proporcionada a cuidadores",
        "expected_decision": "FILTER",
        "description": "SetFit detecta 'cuidadores' pero es referencia genérica plural",
        "setfit_wrongly_classified_as": "PII"
    },
    
    # CASO 12: Verdadero Positivo - Doctor con título
    {
        "id": 12,
        "category": "FALSE_NEGATIVE",
        "entity": "Dr. López",
        "label": "NOMBRE_PERSONAL_SANITARIO",
        "context": "Valorado por el Dr. López del servicio de cardiología",
        "expected_decision": "KEEP",
        "description": "SetFit podría rechazar pero es claramente personal sanitario",
        "setfit_wrongly_classified_as": "RUIDO"
    },
    
    # CASO 13: Falso Positivo - Fragmento incompleto de palabra
    {
        "id": 13,
        "category": "FALSE_POSITIVE",
        "entity": "arcía",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "context": "Error de tokenización genera arcía en lugar de García",
        "expected_decision": "FILTER",
        "description": "SetFit detecta 'arcía' pero es fragmento malformado",
        "setfit_wrongly_classified_as": "PII"
    },
    
    # CASO 14: Verdadero Positivo - Familiar específico con nombre
    {
        "id": 14,
        "category": "FALSE_NEGATIVE",
        "entity": "su hija María",
        "label": "FAMILIARES_SUJETO_ASISTENCIA",
        "context": "Acude acompañado de su hija María González",
        "expected_decision": "KEEP",
        "description": "SetFit podría rechazar pero es familiar específico identificable",
        "setfit_wrongly_classified_as": "RUIDO"
    },
    
    # CASO 15: Falso Positivo - Edad como protocolo médico
    {
        "id": 15,
        "category": "FALSE_POSITIVE",
        "entity": "65",
        "label": "EDAD_SUJETO_ASISTENCIA",
        "context": "Protocolo de screening para mayores de 65 años",
        "expected_decision": "FILTER",
        "description": "SetFit detecta '65' como edad del paciente pero es criterio de protocolo",
        "setfit_wrongly_classified_as": "PII"
    },
]


# =============================================================================
# FUNCIONES DE TESTING
# =============================================================================

def test_single_case_with_llm(
    test_case: Dict[str, Any],
    model: str,
    rules_path: str = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Prueba un caso individual con un LLM específico.
    
    Args:
        test_case: Caso de prueba
        model: Modelo LLM a usar (qwen2.5:7b, hermes3:8b, etc.)
        rules_path: Ruta a guías de anotación
        debug: Modo debug
        
    Returns:
        Resultado del test con métricas
    """
    entity = {
        "text": test_case["entity"],
        "entity_text": test_case["entity"],
        "label": test_case["label"],
        "context": test_case["context"],
    }
    
    config = {
        "model": model,
        "rules_path": rules_path,
        "debug": debug,
        "timeout": 120,
        "max_retries": 2,
    }
    
    start_time = time.time()
    
    try:
        results = run_llm_judge([entity], None, config)
        elapsed = time.time() - start_time
        
        if not results:
            return {
                "test_id": test_case["id"],
                "model": model,
                "success": False,
                "error": "No se obtuvo resultado del LLM",
                "time": elapsed
            }
        
        result = results[0]
        actual_decision = result.get("decision", "UNKNOWN")
        expected_decision = test_case["expected_decision"]
        
        # ¿El LLM corrigió el error de SetFit?
        correct = (actual_decision == expected_decision)
        
        return {
            "test_id": test_case["id"],
            "model": model,
            "category": test_case["category"],
            "description": test_case["description"],
            "entity": test_case["entity"],
            "label": test_case["label"],
            "expected_decision": expected_decision,
            "actual_decision": actual_decision,
            "llm_response": result.get("llm_response", ""),
            "llm_confidence": result.get("llm_confidence", 0.0),
            "llm_status": result.get("llm_status", ""),
            "correct": correct,
            "time": elapsed,
            "success": True,
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "test_id": test_case["id"],
            "model": model,
            "success": False,
            "error": str(e),
            "time": elapsed
        }


def run_comparison_test(
    models: List[str] = ["qwen2.5:7b", "hermes3:8b"],
    rules_path: str = None,
    debug: bool = False,
    test_id: int = None
) -> Dict[str, Any]:
    """
    Ejecuta la comparación completa entre modelos LLM.
    
    Args:
        models: Lista de modelos a comparar
        rules_path: Ruta a guías de anotación
        debug: Modo debug
        test_id: ID de test específico (opcional)
        
    Returns:
        Diccionario con resultados comparativos
    """
    print("=" * 80)
    print("🔬 TEST DE COMPARACIÓN: QWEN vs HERMES (Casos donde SetFit FALLA)")
    print("=" * 80)
    print(f"Modelos a comparar: {', '.join(models)}")
    print(f"Total casos: {len(SETFIT_FAILURE_CASES)}")
    print("=" * 80)
    
    # Filtrar casos si se especifica test_id
    test_cases = SETFIT_FAILURE_CASES
    if test_id:
        test_cases = [tc for tc in test_cases if tc["id"] == test_id]
        if not test_cases:
            print(f"❌ No se encontró caso con ID {test_id}")
            return {}
    
    # Resultados por modelo
    all_results = {model: [] for model in models}
    
    # Ejecutar tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"CASO #{test_case['id']} ({i}/{len(test_cases)}): {test_case['description']}")
        print(f"{'='*80}")
        print(f"Entidad: '{test_case['entity']}'")
        print(f"Etiqueta: {test_case['label']}")
        print(f"Contexto: {test_case['context']}")
        print(f"SetFit clasificó como: {test_case['setfit_wrongly_classified_as']}")
        print(f"Decisión esperada del LLM: {test_case['expected_decision']}")
        print(f"Categoría: {test_case['category']}")
        
        for model in models:
            print(f"\n  🤖 Probando con {model}...")
            result = test_single_case_with_llm(test_case, model, rules_path, debug)
            all_results[model].append(result)
            
            if result["success"]:
                status = "✅ CORRECTO" if result["correct"] else "❌ INCORRECTO"
                print(f"     Decisión: {result['actual_decision']} - {status}")
                print(f"     Confianza: {result['llm_confidence']:.2f}")
                print(f"     Tiempo: {result['time']:.2f}s")
            else:
                print(f"     ❌ ERROR: {result.get('error', 'Desconocido')}")
    
    # Calcular estadísticas comparativas
    print(f"\n{'='*80}")
    print("📊 RESUMEN COMPARATIVO")
    print(f"{'='*80}\n")
    
    comparison = {}
    for model in models:
        results = all_results[model]
        successful = [r for r in results if r["success"]]
        correct = [r for r in successful if r["correct"]]
        
        fp_cases = [r for r in successful if r.get("category") == "FALSE_POSITIVE"]
        fn_cases = [r for r in successful if r.get("category") == "FALSE_NEGATIVE"]
        
        fp_correct = [r for r in fp_cases if r["correct"]]
        fn_correct = [r for r in fn_cases if r["correct"]]
        
        total_time = sum(r["time"] for r in results)
        avg_time = total_time / len(results) if results else 0
        
        comparison[model] = {
            "total_tests": len(results),
            "successful": len(successful),
            "correct": len(correct),
            "incorrect": len(successful) - len(correct),
            "accuracy": (len(correct) / len(successful) * 100) if successful else 0,
            "fp_total": len(fp_cases),
            "fp_correct": len(fp_correct),
            "fp_accuracy": (len(fp_correct) / len(fp_cases) * 100) if fp_cases else 0,
            "fn_total": len(fn_cases),
            "fn_correct": len(fn_correct),
            "fn_accuracy": (len(fn_correct) / len(fn_cases) * 100) if fn_cases else 0,
            "total_time": total_time,
            "avg_time": avg_time,
        }
        
        print(f"🤖 {model.upper()}:")
        print(f"   Precisión general: {comparison[model]['accuracy']:.1f}% ({len(correct)}/{len(successful)})")
        print(f"   Falsos Positivos corregidos: {comparison[model]['fp_accuracy']:.1f}% ({len(fp_correct)}/{len(fp_cases)})")
        print(f"   Falsos Negativos corregidos: {comparison[model]['fn_accuracy']:.1f}% ({len(fn_correct)}/{len(fn_cases)})")
        print(f"   Tiempo promedio: {avg_time:.2f}s")
        print(f"   Tiempo total: {total_time:.2f}s\n")
    
    # Determinar ganador
    if len(models) == 2:
        model1, model2 = models
        stats1 = comparison[model1]
        stats2 = comparison[model2]
        
        print(f"{'='*80}")
        print("🏆 GANADOR")
        print(f"{'='*80}\n")
        
        if stats1["accuracy"] > stats2["accuracy"]:
            winner = model1
            margin = stats1["accuracy"] - stats2["accuracy"]
        elif stats2["accuracy"] > stats1["accuracy"]:
            winner = model2
            margin = stats2["accuracy"] - stats1["accuracy"]
        else:
            winner = "EMPATE"
            margin = 0
        
        if winner != "EMPATE":
            print(f"🥇 {winner.upper()} gana por {margin:.1f}% de precisión")
        else:
            print(f"🤝 EMPATE en precisión")
        
        print(f"\n{'='*80}\n")
    
    return {
        "models": models,
        "test_cases": len(test_cases),
        "results": all_results,
        "comparison": comparison,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test de comparación entre Qwen y Hermes en casos donde SetFit falla"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5:7b", "hermes3:8b"],
        help="Modelos a comparar (default: qwen2.5:7b hermes3:8b)"
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Ruta a guias-anotacion.json"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Modo debug con prompts completos"
    )
    parser.add_argument(
        "--test-id",
        type=int,
        help="Ejecutar solo un test específico"
    )
    parser.add_argument(
        "--output",
        default="llm_comparison_results.json",
        help="Archivo de salida para resultados JSON"
    )
    
    args = parser.parse_args()
    
    # Buscar archivo de reglas si no se especifica
    rules_path = args.rules
    if not rules_path:
        possible_paths = [
            project_root / "guias-anotacion.json",
            project_root.parent / "guias-anotacion.json",
        ]
        for path in possible_paths:
            if path.exists():
                rules_path = str(path)
                break
    
    if rules_path:
        print(f"📋 Usando reglas: {rules_path}\n")
    else:
        print(f"⚠️  No se encontraron reglas de anotación\n")
    
    # Ejecutar comparación
    results = run_comparison_test(
        models=args.models,
        rules_path=rules_path,
        debug=args.debug,
        test_id=args.test_id
    )
    
    # Guardar resultados
    if results and not args.test_id:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Resultados guardados en: {output_path}")
    
    print("\n✨ Comparación completada\n")
