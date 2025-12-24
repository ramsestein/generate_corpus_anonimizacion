#!/usr/bin/env python3
"""
Test Suite para el componente LLM optimizado.

Contiene casos "hardcore" que confunden a modelos pequeños:
- Nombres que parecen cosas (Dolores, Mercedes, Remedios)
- Cosas que parecen nombres (Madrid como gentilicio vs hospital)
- Números ambiguos (edad vs valor clínico)
"""

import sys
from pathlib import Path

# Agregar path para imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "pipeline-nuevos-textos"))

from llm_judge.judge_optimized import evaluate_with_optimized_llm


# =============================================================================
# CASOS DE PRUEBA HARDCORE
# =============================================================================

HARDCORE_TEST_CASES = [
    # CASO 1: "Dolores" - Nombre vs Síntoma
    {
        "id": 1,
        "entity": "Dolores",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "context": "La paciente Dolores García acude a consulta externa",
        "expected": True,  # PII (nombre propio)
        "description": "Dolores como NOMBRE (con apellido)"
    },
    {
        "id": 2,
        "entity": "dolores",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "context": "Paciente refiere dolores abdominales intensos desde hace 3 días",
        +
        "expected": False,  # RUIDO (síntoma)
        "description": "dolores como SÍNTOMA (minúscula, plural)"
    },
    
    # CASO 2: "Mercedes" - Nombre vs Marca
    {
        "id": 3,
        "entity": "Mercedes",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "context": "Dña. Mercedes Romero ingresa por urgencias",
        "expected": True,  # PII (nombre + título)
        "description": "Mercedes como NOMBRE (con Dña.)"
    },
    {
        "id": 4,
        "entity": "Mercedes",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "context": "Llegada en ambulancia marca Mercedes",
        "expected": False,  # RUIDO (marca de vehículo)
        "description": "Mercedes como MARCA (contexto vehículo)"
    },
    
    # CASO 3: "Madrid" - Hospital específico vs Gentilicio
    {
        "id": 5,
        "entity": "Madrid",
        "label": "HOSPITAL",
        "context": "Ingresa en Hospital Clínico San Carlos de Madrid",
        "expected": True,  # PII (parte del nombre del hospital)
        "description": "Madrid como parte de HOSPITAL específico"
    },
    {
        "id": 6,
        "entity": "Madrid",
        "label": "TERRITORIO",
        "context": "Paciente procedente de Madrid",
        "expected": False,  # RUIDO (gentilicio genérico)
        "description": "Madrid como TERRITORIO genérico"
    },
    
    # CASO 4: Número "72" - Edad vs Valor clínico
    {
        "id": 7,
        "entity": "72",
        "label": "EDAD_SUJETO_ASISTENCIA",
        "context": "Varón de 72 años que acude por disnea",
        "expected": True,  # PII (edad del paciente)
        "description": "72 como EDAD (con años)"
    },
    {
        "id": 8,
        "entity": "72",
        "label": "EDAD_SUJETO_ASISTENCIA",
        "context": "Presión arterial sistólica de 72 mmHg",
        "expected": False,  # RUIDO (valor de PA)
        "description": "72 como VALOR CLÍNICO (presión arterial)"
    },
    
    # CASO 5: "familia" - Referencia específica vs Genérica
    {
        "id": 9,
        "entity": "hija",
        "label": "FAMILIARES_SUJETO_ASISTENCIA",
        "context": "Acude acompañado de su hija María González",
        "expected": True,  # PII (familiar específico con nombre)
        "description": "hija como FAMILIAR (con nombre de persona)"
    },
    {
        "id": 10,
        "entity": "familia",
        "label": "FAMILIARES_SUJETO_ASISTENCIA",
        "context": "Presenta antecedentes de diabetes en la familia",
        "expected": False,  # RUIDO (referencia genérica)
        "description": "familia como REFERENCIA GENÉRICA (antecedentes)"
    },
    
    # CASO 6: "Dr. López" - Profesional sanitario
    {
        "id": 11,
        "entity": "López",
        "label": "NOMBRE_PERSONAL_SANITARIO",
        "context": "Valorado por el Dr. López del servicio de cardiología",
        "expected": True,  # PII (nombre de médico)
        "description": "López como PERSONAL SANITARIO (con Dr.)"
    },
    
    # CASO 7: Fecha parcial "15/03" - Con contexto médico
    {
        "id": 12,
        "entity": "15/03",
        "label": "FECHAS",
        "context": "Alta hospitalaria prevista para el 15/03",
        "expected": True,  # PII (fecha de evento médico)
        "description": "15/03 como FECHA (alta hospitalaria)"
    },
    
    # CASO 8: Letra suelta "h" - Ruido claro
    {
        "id": 13,
        "entity": "h",
        "label": "NOMBRE_SUJETO_ASISTENCIA",
        "context": "Apartado h del informe de laboratorio",
        "expected": False,  # RUIDO (letra de sección)
        "description": "h como LETRA DE APARTADO"
    },
    
    # CASO 9: Fragmento malformado ".08.2024"
    {
        "id": 14,
        "entity": ".08.2024",
        "label": "FECHAS",
        "context": "Fecha parcial .08.2024 mal extraída del documento",
        "expected": False,  # RUIDO (fragmento incompleto)
        "description": ".08.2024 como FRAGMENTO MALFORMADO"
    },
    
    # CASO 10: "cuidadores" - Genérico vs Específico
    {
        "id": 15,
        "entity": "cuidadores",
        "label": "FAMILIARES_SUJETO_ASISTENCIA",
        "context": "Educación sanitaria a cuidadores del paciente",
        "expected": False,  # RUIDO (plural genérico)
        "description": "cuidadores como REFERENCIA GENÉRICA"
    },
]


# =============================================================================
# FUNCIONES DE TESTING
# =============================================================================

def run_test_case(test_case: dict, model: str, rules_path: str = None, debug: bool = False) -> dict:
    """
    Ejecuta un caso de prueba individual.
    
    Args:
        test_case: Diccionario con datos del test
        model: Modelo Ollama a usar
        rules_path: Ruta al JSON de reglas
        debug: Mostrar prompts y respuestas
        
    Returns:
        Diccionario con resultado del test
    """
    print(f"\n{'='*80}")
    print(f"TEST CASO #{test_case['id']}: {test_case['description']}")
    print(f"{'='*80}")
    print(f"Entidad: '{test_case['entity']}'")
    print(f"Etiqueta: {test_case['label']}")
    print(f"Contexto: {test_case['context']}")
    print(f"Esperado: {'PII' if test_case['expected'] else 'RUIDO'}")
    
    try:
        result = evaluate_with_optimized_llm(
            entity_text=test_case['entity'],
            entity_label=test_case['label'],
            context=test_case['context'],
            model=model,
            rules_path=rules_path,
            debug=debug
        )
        
        is_pii = result.get('is_pii')
        thought = result.get('thought', '')
        status = result.get('status')
        
        # Verificar si acertó
        passed = (is_pii == test_case['expected']) if is_pii is not None else False
        
        print(f"\n📊 RESULTADO:")
        print(f"   LLM decidió: {'PII' if is_pii else 'RUIDO' if is_pii is False else 'ERROR'}")
        print(f"   Razonamiento: {thought}")
        print(f"   Status: {status}")
        print(f"   Tiempo: {result.get('processing_time', 0):.2f}s")
        print(f"   {'✅ PASS' if passed else '❌ FAIL'}")
        
        return {
            "test_id": test_case['id'],
            "description": test_case['description'],
            "passed": passed,
            "expected": test_case['expected'],
            "actual": is_pii,
            "thought": thought,
            "status": status,
            "time": result.get('processing_time', 0)
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return {
            "test_id": test_case['id'],
            "description": test_case['description'],
            "passed": False,
            "expected": test_case['expected'],
            "actual": None,
            "thought": f"ERROR: {e}",
            "status": "exception",
            "time": 0.0
        }


def run_all_tests(model: str = "llama3.2:latest", rules_path: str = None, debug: bool = False):
    """
    Ejecuta todos los casos de prueba hardcore.
    
    Args:
        model: Modelo Ollama a usar
        rules_path: Ruta al JSON de reglas
        debug: Mostrar información de debug
    """
    print("\n" + "="*80)
    print("🧪 TEST SUITE PARA LLM OPTIMIZADO")
    print("="*80)
    print(f"Modelo: {model}")
    print(f"Reglas: {rules_path or 'No especificadas'}")
    print(f"Total casos: {len(HARDCORE_TEST_CASES)}")
    print("="*80)
    
    results = []
    
    for test_case in HARDCORE_TEST_CASES:
        result = run_test_case(test_case, model, rules_path, debug)
        results.append(result)
    
    # RESUMEN FINAL
    print("\n" + "="*80)
    print("📈 RESUMEN DE RESULTADOS")
    print("="*80)
    
    passed = sum(1 for r in results if r['passed'])
    failed = len(results) - passed
    total_time = sum(r['time'] for r in results)
    
    print(f"\n✅ PASADOS: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
    print(f"❌ FALLIDOS: {failed}/{len(results)} ({failed/len(results)*100:.1f}%)")
    print(f"⏱️  TIEMPO TOTAL: {total_time:.2f}s")
    print(f"⚡ TIEMPO PROMEDIO: {total_time/len(results):.2f}s/test")
    
    # Detalles de fallos
    if failed > 0:
        print(f"\n❌ CASOS FALLIDOS:")
        for r in results:
            if not r['passed']:
                expected_str = "PII" if r['expected'] else "RUIDO"
                actual_str = "PII" if r['actual'] else "RUIDO" if r['actual'] is False else "ERROR"
                print(f"   #{r['test_id']}: {r['description']}")
                print(f"      Esperado: {expected_str}, Obtenido: {actual_str}")
                print(f"      Pensamiento: {r['thought'][:100]}...")
    
    print("\n" + "="*80)
    
    # Casos más lentos
    print("\n⏱️  TOP 3 CASOS MÁS LENTOS:")
    slowest = sorted(results, key=lambda x: x['time'], reverse=True)[:3]
    for i, r in enumerate(slowest, 1):
        print(f"   {i}. Test #{r['test_id']}: {r['time']:.2f}s - {r['description']}")
    
    print("\n" + "="*80)
    print("✨ Test suite completado")
    print("="*80)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test suite para el componente LLM optimizado"
    )
    parser.add_argument(
        "--model",
        default="llama3.2:latest",
        help="Modelo Ollama a usar (default: llama3.2:latest)"
    )
    parser.add_argument(
        "--rules",
        default="guias-anotacion.json",
        help="Ruta al archivo de reglas (default: guias-anotacion.json)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mostrar prompts y respuestas completas"
    )
    parser.add_argument(
        "--test-id",
        type=int,
        help="Ejecutar solo un test específico por ID"
    )
    
    args = parser.parse_args()
    
    # Verificar que existen las reglas
    rules_path = args.rules if Path(args.rules).exists() else None
    if not rules_path:
        print(f"⚠️  No se encontró {args.rules}, continuando sin reglas específicas")
    
    # Ejecutar tests
    if args.test_id:
        # Test individual
        test_case = next((t for t in HARDCORE_TEST_CASES if t['id'] == args.test_id), None)
        if test_case:
            run_test_case(test_case, args.model, rules_path, args.debug)
        else:
            print(f"❌ No se encontró test con ID {args.test_id}")
            sys.exit(1)
    else:
        # Todos los tests
        results = run_all_tests(args.model, rules_path, args.debug)
        
        # Exit code basado en resultados
        passed = sum(1 for r in results if r['passed'])
        sys.exit(0 if passed == len(results) else 1)
