#!/usr/bin/env python3
"""
Test de regresion: verificar que casos que DEBEN ser PII siguen siendolo
y casos que deben ser RUIDO siguen siendo RUIDO.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src" / "pipeline-nuevos-textos"))

from llm_judge import LLMJudge

def main():
    judge = LLMJudge(model="qwen2.5:7b", template_name="strict_v3", timeout=60)
    
    # Casos que DEBEN ser TRUE (PII real)
    pii_cases = [
        # Fechas
        {"text": "12.03.2024", "label": "FECHAS", "context": "Fecha: [**12.03.2024**]", "expected": True},
        {"text": "15/03/24", "label": "FECHAS", "context": "IQ [**15/03/24**]: SVAo", "expected": True},
        {"text": "2020", "label": "FECHAS", "context": "tratamiento desde [**2020**]", "expected": True},
        # Nombres
        {"text": "Carlos", "label": "NOMBRE_SUJETO_ASISTENCIA", "context": "contacto: [**Carlos**] ([**hermano**])", "expected": True},
        {"text": "Carmen", "label": "NOMBRE_SUJETO_ASISTENCIA", "context": "La secretaria [**Carmen**] pedira", "expected": True},
        # Familiares
        {"text": "hermano", "label": "FAMILIARES_SUJETO_ASISTENCIA", "context": "[**Carlos**] ([**hermano**])", "expected": True},
        {"text": "esposa", "label": "FAMILIARES_SUJETO_ASISTENCIA", "context": "deambula con su [**esposa**]", "expected": True},
        # Sexo
        {"text": "Mujer", "label": "SEXO_SUJETO_ASISTENCIA", "context": "[**Mujer**] de [**63 años**]", "expected": True},
        {"text": "varon", "label": "SEXO_SUJETO_ASISTENCIA", "context": "[**varon**] de 68 años", "expected": True},
        # Edad
        {"text": "68 años", "label": "EDAD_SUJETO_ASISTENCIA", "context": "Pcte de [**68 años**] de edad", "expected": True},
        {"text": "52a", "label": "EDAD_SUJETO_ASISTENCIA", "context": "paciente de [**52a**] por Dolor", "expected": True},
        # Telefonos
        {"text": "634567891", "label": "NUMERO_TELEFONO", "context": "telefono: [**634567891**]", "expected": True},
        # Codigos
        {"text": "G045", "label": "NUMERO_IDENTIF", "context": "traslado [**G045**]", "expected": True},
        # Hospital
        {"text": "Hospital Central", "label": "HOSPITAL", "context": "ingresa en [**Hospital Central**]", "expected": True},
    ]
    
    # Casos que DEBEN ser FALSE (ruido)
    ruido_cases = [
        {"text": "paciente", "label": "NOMBRE_SUJETO_ASISTENCIA", "context": "El paciente refiere dolor", "expected": False},
        {"text": "hospital", "label": "HOSPITAL", "context": "traslado a hospital", "expected": False},
        {"text": "hace 2 meses", "label": "FECHAS", "context": "dolor desde hace 2 meses", "expected": False},
        {"text": "el medico", "label": "NOMBRE_PERSONAL_SANITARIO", "context": "el medico indica tratamiento", "expected": False},
        {"text": "ayer", "label": "FECHAS", "context": "ingreso ayer por urgencias", "expected": False},
        {"text": "familia", "label": "FAMILIARES_SUJETO_ASISTENCIA", "context": "antecedentes familiares de HTA", "expected": False},
    ]
    
    print("=" * 60)
    print("TEST DE REGRESION - VERIFICAR QUE NO SE ROMPE NADA")
    print("=" * 60)
    
    # Test PII
    print("\n--- CASOS PII (deben ser TRUE) ---")
    pii_pass = 0
    pii_fail = 0
    for case in pii_cases:
        try:
            result = judge.evaluate(case["text"], case["label"], case["context"][:200])
            if result.is_valid == case["expected"]:
                pii_pass += 1
                status = "OK"
            else:
                pii_fail += 1
                status = "FALLO"
        except Exception as e:
            pii_fail += 1
            status = f"ERROR: {e}"
        print(f"  [{status}] {case['label']}: \"{case['text']}\" -> {result.is_valid if 'result' in dir() else '?'}")
    
    # Test Ruido
    print("\n--- CASOS RUIDO (deben ser FALSE) ---")
    ruido_pass = 0
    ruido_fail = 0
    for case in ruido_cases:
        try:
            result = judge.evaluate(case["text"], case["label"], case["context"][:200])
            if result.is_valid == case["expected"]:
                ruido_pass += 1
                status = "OK"
            else:
                ruido_fail += 1
                status = "FALLO"
        except Exception as e:
            ruido_fail += 1
            status = f"ERROR: {e}"
        print(f"  [{status}] {case['label']}: \"{case['text']}\" -> {result.is_valid if 'result' in dir() else '?'}")
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  PII (TRUE esperado): {pii_pass}/{len(pii_cases)} OK ({pii_pass/len(pii_cases)*100:.0f}%)")
    print(f"  RUIDO (FALSE esperado): {ruido_pass}/{len(ruido_cases)} OK ({ruido_pass/len(ruido_cases)*100:.0f}%)")
    total_pass = pii_pass + ruido_pass
    total = len(pii_cases) + len(ruido_cases)
    print(f"  TOTAL: {total_pass}/{total} ({total_pass/total*100:.0f}%)")
    
    if pii_fail > 0 or ruido_fail > 0:
        print("\n  !! HAY REGRESIONES !!")
    else:
        print("\n  Todo OK - No hay regresiones")

if __name__ == '__main__':
    main()
