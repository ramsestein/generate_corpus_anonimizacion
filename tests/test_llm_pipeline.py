#!/usr/bin/env python3
"""
Script de prueba rápida del pipeline LLM Judge
Verifica que todo está configurado correctamente sin hacer llamadas reales al LLM
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "pipeline-nuevos-textos"))

def test_imports():
    """Verifica que todos los imports funcionan"""
    print("=" * 80)
    print("TEST 1: Verificando imports...")
    print("=" * 80)
    
    try:
        from llm_prompts import PROMPT_CONFIG, get_prompt_config, load_entity_rules
        print("  ✓ llm_prompts importado correctamente")
        
        from llm_entity_judge import (
            load_llm_credentials,
            extract_context_window,
            parse_llm_response,
            build_prompt_for_entity
        )
        print("  ✓ llm_entity_judge importado correctamente")
        
        from dotenv import load_dotenv
        print("  ✓ python-dotenv importado correctamente")
        
        import requests
        print("  ✓ requests importado correctamente")
        
        return True
    except ImportError as e:
        print(f"  ✗ Error de import: {e}")
        return False


def test_prompt_config():
    """Verifica la configuración de prompts"""
    print("\n" + "=" * 80)
    print("TEST 2: Verificando PROMPT_CONFIG...")
    print("=" * 80)
    
    try:
        from llm_prompts import PROMPT_CONFIG, get_prompt_config
        
        config = get_prompt_config()
        
        required_fields = ["name", "version", "description", "language", "template", 
                          "input_variables", "output_format", "expected_fields"]
        
        for field in required_fields:
            if field not in config:
                print(f"  ✗ Falta campo: {field}")
                return False
            print(f"  ✓ Campo '{field}': {str(config[field])[:50]}...")
        
        # Verificar variables en el template
        template = config["template"]
        required_vars = ["keyword", "context_text", "label", "location", "reglas_etiqueta"]
        
        for var in required_vars:
            placeholder = "{" + var + "}"
            if placeholder not in template:
                print(f"  ✗ Falta placeholder: {placeholder}")
                return False
            print(f"  ✓ Placeholder '{placeholder}' presente en template")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_entity_rules():
    """Verifica carga de reglas de entidades"""
    print("\n" + "=" * 80)
    print("TEST 3: Verificando carga de reglas...")
    print("=" * 80)
    
    try:
        from llm_prompts import load_entity_rules, get_entity_rules_for_label
        
        rules = load_entity_rules()
        print(f"  ✓ {len(rules)} tipos de entidades cargados")
        
        # Probar algunas etiquetas comunes
        test_labels = ["NOMBRE_SUJETO_ASISTENCIA", "FECHAS", "HOSPITAL"]
        
        for label in test_labels:
            if label in rules:
                rules_text = get_entity_rules_for_label(label)
                num_rules = len(rules[label])
                print(f"  ✓ {label}: {num_rules} reglas")
            else:
                print(f"  ✗ Etiqueta no encontrada: {label}")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_context_extraction():
    """Verifica extracción de contexto"""
    print("\n" + "=" * 80)
    print("TEST 4: Verificando extracción de contexto...")
    print("=" * 80)
    
    try:
        from llm_entity_judge import extract_context_window
        
        # Texto de prueba
        text = "El paciente García tiene 45 años y acude por dolor abdominal."
        entity_start = 12  # "García"
        entity_end = 18
        
        context = extract_context_window(text, entity_start, entity_end, 20, 20)
        
        if "García" in context:
            print(f"  ✓ Contexto extraído: '{context}'")
            return True
        else:
            print(f"  ✗ Entidad no encontrada en contexto: '{context}'")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_response_parsing():
    """Verifica parsing de respuestas del LLM"""
    print("\n" + "=" * 80)
    print("TEST 5: Verificando parsing de respuestas...")
    print("=" * 80)
    
    try:
        from llm_entity_judge import parse_llm_response
        
        test_cases = [
            ("TRUE", True),
            ("FALSE", False),
            ("true", True),  # case insensitive
            ("false", False),
            ("  TRUE  ", True),  # con espacios
            ("invalid", None),
        ]
        
        all_ok = True
        for input_val, expected in test_cases:
            result = parse_llm_response(input_val)
            if result == expected:
                print(f"  ✓ '{input_val}' → {result}")
            else:
                print(f"  ✗ '{input_val}' → {result} (esperado: {expected})")
                all_ok = False
        
        return all_ok
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_prompt_building():
    """Verifica construcción de prompts"""
    print("\n" + "=" * 80)
    print("TEST 6: Verificando construcción de prompts...")
    print("=" * 80)
    
    try:
        from llm_entity_judge import build_prompt_for_entity
        from llm_prompts import get_entity_rules_for_label
        
        # Datos de prueba
        keyword = "García"
        context = "El paciente García tiene 45 años"
        label = "NOMBRE_SUJETO_ASISTENCIA"
        location = "offset 12-18"
        rules = get_entity_rules_for_label(label)
        
        system_prompt, user_prompt = build_prompt_for_entity(
            keyword, context, label, location, rules
        )
        
        print(f"  ✓ System prompt generado ({len(system_prompt)} caracteres)")
        print(f"  ✓ User prompt generado ({len(user_prompt)} caracteres)")
        
        # Verificar que contiene elementos clave
        checks = [
            (keyword in system_prompt or keyword in user_prompt, "keyword presente"),
            (label in system_prompt or label in user_prompt, "label presente"),
            (len(rules) > 0, "reglas cargadas"),
        ]
        
        for check, desc in checks:
            if check:
                print(f"  ✓ {desc}")
            else:
                print(f"  ✗ {desc}")
                return False
        
        return True
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_check():
    """Verifica si existe .env (sin cargar credenciales)"""
    print("\n" + "=" * 80)
    print("TEST 7: Verificando archivo .env...")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    env_example_path = project_root / ".env.example"
    
    if env_path.exists():
        print(f"  ✓ Archivo .env encontrado: {env_path}")
        print("  ⚠ Nota: No se verifican las credenciales en este test")
        return True
    else:
        print(f"  ✗ Archivo .env NO encontrado: {env_path}")
        if env_example_path.exists():
            print(f"  ℹ Archivo .env.example disponible: {env_example_path}")
            print(f"  ℹ Copia .env.example a .env y configura tus credenciales")
        return False


def main():
    """Ejecuta todos los tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TEST SUITE - LLM JUDGE PIPELINE" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    
    tests = [
        ("Imports", test_imports),
        ("PROMPT_CONFIG", test_prompt_config),
        ("Reglas de Entidades", test_entity_rules),
        ("Extracción de Contexto", test_context_extraction),
        ("Parsing de Respuestas", test_response_parsing),
        ("Construcción de Prompts", test_prompt_building),
        ("Archivo .env", test_env_check),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Error fatal en test '{name}': {e}")
            results[name] = False
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE TESTS")
    print("=" * 80)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "-" * 80)
    print(f"Total: {total} tests")
    print(f"  ✓ Pasados: {passed}")
    print(f"  ✗ Fallidos: {failed}")
    print(f"  Porcentaje: {passed/total*100:.1f}%")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 ¡Todos los tests pasaron! El pipeline está listo para usar.")
        print("\nPróximos pasos:")
        print("  1. Configura .env con tus credenciales del LLM")
        print("  2. Ejecuta el pipeline con: python src/pipeline-nuevos-textos/llm_entity_judge.py")
        return 0
    else:
        print("\n⚠ Algunos tests fallaron. Revisa los errores arriba.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
