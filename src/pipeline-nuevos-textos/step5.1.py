#!/usr/bin/env python3
"""
STEP 5: Anonimización simple (versión corregida)
Lee archivos .txt y reemplaza cualquier texto marcado como [** ... **] por un token (por defecto "JJJ").
Genera un resumen con el número de archivos procesados y reemplazos realizados.
No lee CSV ni JSON.
"""

import argparse
import os
import re
import datetime
from pathlib import Path
from typing import Tuple, List, Dict


def debug_print(msg: str, level: str = "INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


# Regex para capturar contenido entre [** y **], no-greedy y admite saltos de línea
BRACKETED_RE = re.compile(r"\[\*\*(.*?)\*\*\]", flags=re.DOTALL)


def anonymize_text_remove_bracketed(text: str, token: str = "JJJ") -> Tuple[str, int]:
    """
    Reemplaza todas las ocurrencias de [** ... **] por `token`.
    Devuelve (texto_anonimizado, numero_reemplazos).
    """
    matches = BRACKETED_RE.findall(text)
    count = len(matches)
    if count == 0:
        return text, 0

    anonymized = BRACKETED_RE.sub(token, text)

    # Colapsar espacios múltiples que puedan quedar por el reemplazo
    anonymized = re.sub(r"\s{2,}", " ", anonymized).strip()
    return anonymized, count


def process_file(input_file: Path, output_file: Path, token: str) -> Dict:
    """
    Lee input_file, anonimiza, escribe output_file.
    Devuelve dict con métricas por archivo.
    """
    try:
        text = input_file.read_text(encoding='utf-8')
    except Exception as e:
        debug_print(f"Error leyendo {input_file}: {e}", "ERROR")
        return {"file": str(input_file), "success": False, "error": str(e)}

    anonymized, replaced = anonymize_text_remove_bracketed(text, token)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(anonymized, encoding='utf-8')
    except Exception as e:
        debug_print(f"Error escribiendo {output_file}: {e}", "ERROR")
        return {"file": str(input_file), "success": False, "error": str(e)}

    debug_print(f"Procesado {input_file.name}: {replaced} reemplazos", "INFO")
    return {"file": str(input_file), "success": True, "replacements": replaced, "output": str(output_file)}


def gather_txt_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*.txt"))


def main():
    parser = argparse.ArgumentParser(description="Anonimizar marcas [** ... **] en .txt")
    parser.add_argument("--input-dir", "-i", default="step4_5_cleaned_documents", help="Directorio con .txt")
    parser.add_argument("--output-dir", "-o", default="step5_anonymized_documents", help="Directorio de salida")
    parser.add_argument("--token", "-t", default="JJJ", help="Token de reemplazo (por defecto JJJ)")
    parser.add_argument("--max-files", type=int, default=None, help="Máximo de archivos a procesar")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists() or not input_path.is_dir():
        print(f"ERROR: directorio de entrada no existe: {input_path}")
        return

    files = gather_txt_files(input_path)
    if args.max_files:
        files = files[:args.max_files]

    if not files:
        print("No se encontraron archivos .txt para procesar.")
        return

    debug_print(f"Iniciando anonimización: {len(files)} archivos -> {output_path}", "INFO")

    results = []
    total_replacements = 0
    for f in files:
        out_file = output_path / f.name
        res = process_file(f, out_file, args.token)
        results.append(res)
        if res.get("success"):
            total_replacements += res.get("replacements", 0)

    # Resumen
    processed = sum(1 for r in results if r.get("success"))
    failed = [r for r in results if not r.get("success")]
    debug_print("=== RESUMEN ===", "INFO")
    print(f"Archivos procesados: {processed}/{len(results)}")
    print(f"Reemplazos totales: {total_replacements}")
    if failed:
        print("Errores en archivos:")
        for f in failed:
            print(f" - {f['file']}: {f.get('error')}")


if __name__ == "__main__":
    main()