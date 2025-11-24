#!/usr/bin/env python3
"""
Script de prueba rápida para verificar que Ollama con gemma3:270m funciona correctamente
"""

import sys
import subprocess
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "pipeline-nuevos-textos"))


def test_ollama_installation():
    """Verifica que Ollama esté instalado"""
    print("=" * 80)
    print("TEST 1: Verificando instalación de Ollama")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        assert result.returncode == 0, "Ollama debe devolver código 0 al pedir --version"
        version = result.stdout.strip()
        print(f"  ✓ Ollama instalado: {version}")
            
    except FileNotFoundError:
        pytest_msg = "Ollama no encontrado en el PATH. Instala Ollama desde: https://ollama.ai"
        raise AssertionError(pytest_msg)
    except subprocess.TimeoutExpired:
        raise AssertionError("Timeout al ejecutar ollama --version")


def test_gemma_model():
    """Verifica que el modelo gemma3:270m esté disponible"""
    print("\n" + "=" * 80)
    print("TEST 2: Verificando modelo gemma3:270m")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, "Error al listar modelos con 'ollama list'"
        models = result.stdout
        assert ("gemma3:270m" in models) or ("gemma3" in models), (
            "Modelo gemma3:270m no encontrado. Ejecuta: ollama pull gemma3:270m\nModelos disponibles:\n"
            + models
        )
        print(f"  ✓ Modelo gemma3:270m disponible")
            
    except Exception as e:
        raise AssertionError(f"Error verificando modelos: {e}")


def test_ollama_call():
    """Prueba una llamada simple a Ollama"""
    print("\n" + "=" * 80)
    print("TEST 3: Probando llamada a Ollama con gemma3:270m")
    print("=" * 80)
    
    try:
        # Prompt de prueba simple
        test_prompt = "Responde SOLO con la palabra TRUE o FALSE: ¿Es 'García' un nombre de persona?"
        
        print(f"  Enviando prompt de prueba...")
        
        result = subprocess.run(
            ["ollama", "run", "gemma3:270m", test_prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Error al ejecutar 'ollama run': {result.stderr}"
        response = result.stdout.strip()
        print(f"  ✓ Respuesta recibida: '{response}'")
        assert ("TRUE" in response.upper()) or ("FALSE" in response.upper()), (
            "La respuesta debe contener TRUE o FALSE. Respuesta recibida: '" + response + "'"
        )
            
    except subprocess.TimeoutExpired:
        raise AssertionError("Timeout (>30s) - el modelo puede estar descargándose")
    except Exception as e:
        raise AssertionError(f"Error en llamada a Ollama: {e}")


def test_import_modules():
    """Verifica que los módulos del pipeline se importen correctamente"""
    print("\n" + "=" * 80)
    print("TEST 4: Verificando módulos del pipeline")
    print("=" * 80)
    
    try:
        from llm_entity_judge import (
            check_ollama_available,
            call_ollama_gemma,
            extract_context_window,
            parse_llm_response,
            build_prompt_for_entity
        )
        print(f"  ✓ Módulo llm_entity_judge importado correctamente")
        from llm_prompts import load_entity_rules, get_entity_rules_for_label
        print(f"  ✓ Módulo llm_prompts importado correctamente")
        # Aserciones simples para asegurar que las funciones existen
        assert callable(check_ollama_available)
        assert callable(call_ollama_gemma)
        assert callable(extract_context_window)
        assert callable(parse_llm_response)
        assert callable(build_prompt_for_entity)
        assert callable(load_entity_rules)
        assert callable(get_entity_rules_for_label)
    except ImportError as e:
        raise AssertionError(f"Error de importación: {e}")


def main():
    """Ejecuta todos los tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "TEST SUITE - OLLAMA + GEMMA3:270m" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    
    tests = [
        ("Instalación de Ollama", test_ollama_installation),
        ("Modelo gemma3:270m", test_gemma_model),
        ("Llamada a Ollama", test_ollama_call),
        ("Módulos del Pipeline", test_import_modules),
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
        print("\n🎉 ¡Todos los tests pasaron! El pipeline está listo para usar con Ollama.")
        print("\nPróximos pasos:")
        print("  1. Ejecuta el pipeline con: python src/pipeline-nuevos-textos/llm_entity_judge.py")
        print("  2. Usa los datos de ejemplo en: examples/entities_example.json")
        return 0
    else:
        print("\n⚠ Algunos tests fallaron. Revisa los errores arriba.")
        if not results.get("Instalación de Ollama"):
            print("\nℹ Instala Ollama desde: https://ollama.ai")
        if not results.get("Modelo gemma3:270m"):
            print("\nℹ Descarga el modelo con: ollama pull gemma3:270m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
