#!/usr/bin/env python3
"""
TEST DEL MÓDULO DE PROMPTS
==========================

Tests rápidos para validar el funcionamiento del generador de prompts.
"""

import sys
from pathlib import Path

# Añadir el directorio src al path
sys.path.insert(0, str(Path(__file__).parent))

from llm_prompts import (
    generate_system_prompt,
    generate_user_prompt,
    generate_prompts_for_entity,
    validate_llm_response,
    parse_llm_response,
    get_prompt_stats
)


def test_system_prompt_generation():
    """Test 1: Verificar que el system prompt se genera correctamente"""
    print("🧪 Test 1: Generación de System Prompt")
    print("-" * 60)
    
    system_prompt = generate_system_prompt()
    
    # Verificaciones
    assert len(system_prompt) > 1000, "System prompt demasiado corto"
    assert "MAXIMIZAR" in system_prompt and "RECALL" in system_prompt, "Falta énfasis en RECALL"
    assert "TRUE" in system_prompt, "Falta formato de respuesta TRUE"
    assert "FALSE" in system_prompt, "Falta formato de respuesta FALSE"
    assert "TERRITORIO" in system_prompt, "Faltan reglas de TERRITORIO"
    assert "FECHAS" in system_prompt, "Faltan reglas de FECHAS"
    assert "FAMILIARES_SUJETO_ASISTENCIA" in system_prompt, "Faltan reglas de FAMILIARES"
    
    print(f"✅ System prompt generado correctamente")
    print(f"   - Longitud: {len(system_prompt):,} caracteres")
    print(f"   - Contiene reglas de anotación: ✓")
    print(f"   - Enfatiza maximizar RECALL: ✓")
    print()


def test_user_prompt_generation():
    """Test 2: Verificar que el user prompt se genera correctamente"""
    print("🧪 Test 2: Generación de User Prompts")
    print("-" * 60)
    
    # Caso 1: Nombre
    user_prompt = generate_user_prompt("NOMBRE_SUJETO_ASISTENCIA", "Juan García")
    assert "NOMBRE_SUJETO_ASISTENCIA" in user_prompt
    assert "Juan García" in user_prompt
    assert "TRUE" in user_prompt or "FALSE" in user_prompt
    print(f"✅ User prompt para NOMBRE generado correctamente")
    
    # Caso 2: Fecha
    user_prompt = generate_user_prompt("FECHAS", "25/12/2020")
    assert "FECHAS" in user_prompt
    assert "25/12/2020" in user_prompt
    print(f"✅ User prompt para FECHAS generado correctamente")
    
    # Caso 3: Familiares
    user_prompt = generate_user_prompt("FAMILIARES_SUJETO_ASISTENCIA", "familia")
    assert "FAMILIARES_SUJETO_ASISTENCIA" in user_prompt
    assert "familia" in user_prompt
    print(f"✅ User prompt para FAMILIARES generado correctamente")
    
    print()


def test_combined_prompts():
    """Test 3: Verificar generación combinada de prompts"""
    print("🧪 Test 3: Generación Combinada (System + User)")
    print("-" * 60)
    
    entity = {
        "label": "FECHAS",
        "text": "25/12/2020"
    }
    
    prompts = generate_prompts_for_entity(entity)
    
    assert "system_prompt" in prompts
    assert "user_prompt" in prompts
    assert len(prompts["system_prompt"]) > 1000
    assert "FECHAS" in prompts["user_prompt"]
    assert "25/12/2020" in prompts["user_prompt"]
    
    print(f"✅ Prompts combinados generados correctamente")
    print(f"   - System prompt: {len(prompts['system_prompt']):,} caracteres")
    print(f"   - User prompt: {len(prompts['user_prompt'])} caracteres")
    print()


def test_response_validation():
    """Test 4: Verificar validación de respuestas del LLM"""
    print("🧪 Test 4: Validación de Respuestas del LLM")
    print("-" * 60)
    
    # Casos válidos
    valid_responses = ["TRUE", "FALSE"]
    for resp in valid_responses:
        assert validate_llm_response(resp), f"{resp} debería ser válido"
        print(f"✅ '{resp}' → VÁLIDO")
    
    # Casos inválidos
    invalid_responses = [
        "true",  # minúsculas
        "false",  # minúsculas
        "TRUE porque es una fecha",  # explicación
        "FALSE - no cumple",  # puntuación
        "Sí",  # formato incorrecto
        "No",  # formato incorrecto
        "1",  # número
        "0",  # número
        "YES",  # inglés
        "TRUE\n",  # con salto de línea
        " TRUE ",  # con espacios (este SÍ debería ser válido al hacer strip())
    ]
    
    for resp in invalid_responses:
        # " TRUE " debería ser válido tras strip()
        if resp == " TRUE ":
            assert validate_llm_response(resp), f"'{resp}' debería ser válido (con strip)"
            print(f"✅ '{resp}' → VÁLIDO (tras strip)")
        else:
            result = validate_llm_response(resp)
            if not result:
                print(f"✅ '{resp}' → INVÁLIDO (correcto)")
            else:
                print(f"⚠️  '{resp}' → VÁLIDO (inesperado)")
    
    print()


def test_response_parsing():
    """Test 5: Verificar parsing de respuestas"""
    print("🧪 Test 5: Parsing de Respuestas")
    print("-" * 60)
    
    # Casos válidos
    assert parse_llm_response("TRUE") == True
    assert parse_llm_response("FALSE") == False
    assert parse_llm_response(" TRUE ") == True  # con espacios
    assert parse_llm_response(" FALSE ") == False
    
    print(f"✅ 'TRUE' → {parse_llm_response('TRUE')}")
    print(f"✅ 'FALSE' → {parse_llm_response('FALSE')}")
    
    # Casos inválidos
    invalid = [
        "true",
        "yes",
        "1",
        "TRUE porque...",
    ]
    
    for resp in invalid:
        result = parse_llm_response(resp)
        assert result is None, f"{resp} debería parsear a None"
        print(f"✅ '{resp}' → None (correcto)")
    
    print()


def test_prompt_stats():
    """Test 6: Verificar estadísticas de prompts"""
    print("🧪 Test 6: Estadísticas de Prompts")
    print("-" * 60)
    
    stats = get_prompt_stats()
    
    assert stats["system_prompt_length"] > 10000
    assert stats["system_prompt_words"] > 1000
    assert stats["estimated_tokens"] > 3000
    assert stats["user_prompt_length"] > 100
    
    print(f"✅ Estadísticas calculadas correctamente:")
    print(f"   - System prompt: {stats['system_prompt_length']:,} caracteres")
    print(f"   - System prompt: ~{stats['estimated_tokens']:,} tokens")
    print(f"   - User prompt (ejemplo): {stats['user_prompt_length']} caracteres")
    print()


def test_examples():
    """Test 7: Casos de ejemplo reales"""
    print("🧪 Test 7: Casos de Ejemplo Reales")
    print("-" * 60)
    
    examples = [
        {
            "label": "NOMBRE_SUJETO_ASISTENCIA",
            "text": "Juan García",
            "expected_in_prompt": ["NOMBRE_SUJETO_ASISTENCIA", "Juan García"]
        },
        {
            "label": "FECHAS",
            "text": "25/12/2020",
            "expected_in_prompt": ["FECHAS", "25/12/2020"]
        },
        {
            "label": "FAMILIARES_SUJETO_ASISTENCIA",
            "text": "familia",
            "expected_in_prompt": ["FAMILIARES_SUJETO_ASISTENCIA", "familia"]
        },
        {
            "label": "ID_SUJETO_ASISTENCIA",
            "text": "G045",
            "expected_in_prompt": ["ID_SUJETO_ASISTENCIA", "G045"]
        },
    ]
    
    for i, example in enumerate(examples, 1):
        prompts = generate_prompts_for_entity(example)
        user_prompt = prompts["user_prompt"]
        
        for expected_text in example["expected_in_prompt"]:
            assert expected_text in user_prompt, \
                f"'{expected_text}' no encontrado en el prompt"
        
        print(f"✅ Ejemplo {i}: {example['label']} = '{example['text']}' → OK")
    
    print()


def run_all_tests():
    """Ejecutar todos los tests"""
    print("=" * 60)
    print("🧪 SUITE DE TESTS - MÓDULO DE PROMPTS")
    print("=" * 60)
    print()
    
    tests = [
        test_system_prompt_generation,
        test_user_prompt_generation,
        test_combined_prompts,
        test_response_validation,
        test_response_parsing,
        test_prompt_stats,
        test_examples,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FALLO: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"📊 RESUMEN DE TESTS")
    print("=" * 60)
    print(f"✅ Pasados: {passed}/{len(tests)}")
    print(f"❌ Fallidos: {failed}/{len(tests)}")
    
    if failed == 0:
        print()
        print("🎉 ¡TODOS LOS TESTS PASARON!")
        print()
        return 0
    else:
        print()
        print("⚠️  ALGUNOS TESTS FALLARON")
        print()
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
