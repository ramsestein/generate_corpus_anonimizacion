#!/usr/bin/env python3
"""
PIPELINE DE EVALUACIÓN CON JUEZ LLM - PASO 3: EVALUACIÓN CON OLLAMA
====================================================================

Este módulo ejecuta la evaluación completa de entidades usando Ollama
con el modelo local gemma3:270m.

FLUJO:
1. Cargar JSON de entidades detectadas (con doc_id, start, end, label, keyword)
2. Para cada entidad:
   - Leer doc_id, start, end, label, keyword desde el JSON
   - Construir document_path = docs_base_path + doc_id
   - Cargar documento desde document_path construido
   - Extraer contexto usando start/end + ventanas
   - Obtener reglas de la etiqueta
   - Construir prompts (system con reglas + user con palabra y contexto)
   - Llamar a Ollama con gemma3:270m
   - Interpretar respuesta (TRUE/FALSE)
3. Guardar resultados en CSV con document_path completo construido

REQUISITOS:
- Ollama instalado localmente
- Modelo gemma3:270m disponible (ollama pull gemma3:270m)
- JSON con doc_id (nombre del archivo), start, end, label, keyword en cada entidad
- Parámetro --docs con ruta base a los documentos
"""

import json
import csv
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Importar configuración de prompts
from llm_prompts import PROMPT_CONFIG, load_entity_rules, get_entity_rules_for_label


# ============================================================================
# CONFIGURACIÓN DE OLLAMA
# ============================================================================

def check_ollama_available() -> bool:
    """
    Verifica que Ollama esté disponible en el sistema.
    
    Returns:
        True si Ollama está disponible, False en caso contrario
    """
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ============================================================================
# LLAMADA A OLLAMA
# ============================================================================

def call_ollama_gemma(system_prompt: str, user_prompt: str, debug: bool = False) -> Tuple[str, str]:
    """
    Realiza una llamada a Ollama usando el modelo gemma3:270m.
    
    Args:
        system_prompt: Prompt de sistema con reglas de la etiqueta
        user_prompt: Prompt de usuario con palabra, contexto y consulta
        
    Returns:
        Respuesta del modelo como string
        
    Raises:
        subprocess.CalledProcessError: Si hay error en la llamada a Ollama
        ValueError: Si la respuesta no se puede parsear
    """
    # Construir el prompt completo combinando system y user
    # Ollama en CLI usa un formato más simple
    full_prompt = f"""{system_prompt}

---

{user_prompt}"""
    
    try:
        if debug:
            print("\n--- [DEBUG] Prompt de sistema (system_prompt) ---")
            print(system_prompt)
            print("--- [DEBUG] Prompt de usuario (user_prompt) ---")
            # Mostrar prompt completo para diagnóstico
            print(full_prompt)

        # Llamar a Ollama usando subprocess pasando el prompt por stdin
        # Usar text=True y pasar string como input para simplificar decoding
        # Añadir reintentos si la respuesta viene vacía (puede pasar por carga/streaming)
        max_retries = 2
        wait_seconds = 5
        last_stdout = ""
        last_stderr = ""
        for attempt in range(1, max_retries + 1):
            try:
                result = subprocess.run(
                    ["ollama", "run", "gemma3:270m"],
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True
                )

                stdout = result.stdout or ""
                stderr = result.stderr or ""

                last_stdout = stdout
                last_stderr = stderr

                if debug:
                    print("--- [DEBUG] Ollama stdout ---")
                    print(stdout[:4000] + ("...[truncated]" if len(stdout) > 4000 else ""))
                    if stderr:
                        print("--- [DEBUG] Ollama stderr ---")
                        print(stderr[:2000] + ("...[truncated]" if len(stderr) > 2000 else ""))

                response = stdout.strip()
                if response:
                    return stdout, response

                # Si no hay respuesta, reintentar
                if attempt < max_retries:
                    if debug:
                        print(f"[WARN] Ollama returned empty response (attempt {attempt}), retrying...")
                    import time
                    time.sleep(wait_seconds)
                    continue
                else:
                    # último intento fallido
                    raise ValueError(f"Ollama devolvió una respuesta vacía. stderr: {stderr}")

            except subprocess.TimeoutExpired:
                raise subprocess.CalledProcessError(
                    returncode=-1,
                    cmd="ollama",
                    output="Timeout al ejecutar Ollama (>120s)"
                )
            except subprocess.CalledProcessError as e:
                # Propagar error con información útil
                raise subprocess.CalledProcessError(
                    returncode=e.returncode,
                    cmd=e.cmd,
                    output=f"Error al ejecutar Ollama: {e.stderr or e.output}"
                )

        # Si se sale del loop (no debería), devolver último stdout
        return last_stdout, (last_stdout or "").strip()

    except FileNotFoundError:
        raise FileNotFoundError(
            "Ollama no está instalado o no está en el PATH.\n"
            "Instala Ollama desde: https://ollama.ai"
        )


# ============================================================================
# EXTRACCIÓN DE CONTEXTO
# ============================================================================

def extract_context_window(
    document_text: str,
    entity_start: int,
    entity_end: int,
    left_window: int = 80,
    right_window: int = 80
) -> str:
    """
    Extrae una ventana de contexto alrededor de la entidad en el documento.
    
    Args:
        document_text: Texto completo del documento
        entity_start: Posición de inicio de la entidad (caracteres)
        entity_end: Posición de fin de la entidad (caracteres)
        left_window: Caracteres a incluir antes de la entidad
        right_window: Caracteres a incluir después de la entidad
        
    Returns:
        String con el contexto (texto antes + entidad + texto después)
    """
    # Calcular límites
    start_pos = max(0, entity_start - left_window)
    end_pos = min(len(document_text), entity_end + right_window)
    
    # Extraer contexto
    context = document_text[start_pos:end_pos]
    
    return context


def load_document_text(doc_path: Path) -> str:
    """
    Carga el texto completo de un documento.
    
    Args:
        doc_path: Ruta al archivo del documento
        
    Returns:
        Texto completo del documento
        
    Raises:
        FileNotFoundError: Si el documento no existe
    """
    if not doc_path.exists():
        raise FileNotFoundError(f"Documento no encontrado: {doc_path}")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        return f.read()


# ============================================================================
# PROCESAMIENTO DE ENTIDADES
# ============================================================================

def parse_llm_response(llm_response: str) -> Optional[bool]:
    """
    Parsea la respuesta del LLM a un booleano.
    
    Args:
        llm_response: Respuesta del LLM
        
    Returns:
        True si es "TRUE", False si es "FALSE", None si es inválida
    """
    if not isinstance(llm_response, str):
        return None

    cleaned = llm_response.strip().upper()

    # Aceptar varias formas comunes de TRUE/FALSE
    true_tokens = {'TRUE', '1', 'SI', 'SÍ', 'YES', 'CORRECTO', 'VALIDO', 'VÁLIDO', 'VERDADERO'}
    false_tokens = {'FALSE', '0', 'NO', 'INCORRECTO', 'INVALIDO', 'INVÁLIDO', 'FALSO'}

    # Buscar palabras completas o substrings que indiquen respuesta
    # Primero tokens completos separados por espacio o nueva línea
    tokens = set(t.strip().upper() for t in cleaned.replace('\n', ' ').split())
    if tokens & true_tokens:
        return True
    if tokens & false_tokens:
        return False

    # Fallback: buscar substrings
    if any(tok in cleaned for tok in ('TRUE', 'VERDADERO', 'CORRECTO', 'SI', 'SÍ', 'YES')):
        return True
    if any(tok in cleaned for tok in ('FALSE', 'FALSO', 'NO', 'INCORRECTO', 'INVALIDO')):
        return False

    return None


def build_prompt_for_entity(
    keyword: str,
    context_text: str,
    label: str,
    location: str,
    rules_text: str
) -> Tuple[str, str]:
    """
    Construye los prompts de sistema y usuario para una entidad.
    
    El system_prompt contiene las reglas de la etiqueta.
    El user_prompt contiene la palabra, el contexto del documento y la consulta.
    
    Args:
        keyword: Palabra candidata a entidad
        context_text: Texto donde aparece la palabra (chunk del documento)
        label: Etiqueta de la entidad
        location: Ubicación/subtexto (offset en el documento)
        rules_text: Reglas formateadas de la etiqueta
        
    Returns:
        Tuple con (system_prompt, user_prompt)
    """
    # Usar la plantilla definida en llm_prompts.PROMPT_CONFIG para homogeneizar prompts
    template = PROMPT_CONFIG.get("template") if isinstance(PROMPT_CONFIG, dict) else None
    if template:
        system_prompt = template.format(
            reglas_etiqueta=rules_text,
            keyword=keyword,
            context_text=context_text,
            label=label,
            location=location
        )
        # Mantener user_prompt corto (el template ya incluye la información necesaria)
        user_prompt = 'Responde SOLO: "TRUE" o "FALSE".'
    else:
        # Fallback al prompt anterior si la plantilla no está disponible
        system_prompt = f"ERES UN VERIFICADOR ESTRICTO DE ENTIDADES.\n\nREGLAS:\n{rules_text}\n\nOBJETIVO: MAXIMIZAR PRECISION. \nResponde SOLO 'TRUE' o 'FALSE'."
        user_prompt = f"PALABRA CANDIDATA: \"{keyword}\"\n\nTEXTO: \"{context_text}\"\n\nETIQUETA: {label}\nUBICACIÓN: {location}\n\nResponde SOLO 'TRUE' o 'FALSE'."

    return system_prompt, user_prompt


def process_entity(
    entity: Dict,
    rules_text: str,
    docs_base_path: str,
    left_window: int = 80,
    right_window: int = 80,
    debug: bool = False
) -> Dict:
    """
    Procesa una entidad: construye document_path, carga documento, extrae contexto, llama al LLM y parsea respuesta.
    
    Args:
        entity: Diccionario con datos de la entidad (incluye doc_id, start, end, label, keyword)
        rules_text: Reglas formateadas de la etiqueta
        docs_base_path: Ruta base donde están los documentos
        left_window: Ventana de contexto izquierda
        right_window: Ventana de contexto derecha
        
    Returns:
        Dict con resultados de la evaluación (incluye document_path construido)
    """
    # Extraer datos de la entidad desde el JSON
    keyword = entity.get("text", entity.get("entity_text", entity.get("keyword", "")))
    label = entity.get("label", "")
    start = entity.get("start", 0)
    end = entity.get("end", start + len(keyword))
    doc_id = entity.get("document_id", entity.get("doc_id", "unknown"))
    
    # Validar que tenemos doc_id
    if not doc_id or doc_id == "unknown":
        return {
            "document_id": doc_id,
            "document_path": "",
            "entity_id": entity.get("id", entity.get("entity_id", "unknown")),
            "keyword": keyword,
            "label": label,
            "start": start,
            "end": end,
            "context": "",
            "llm_response": "",
            "is_valid": None,
            "manual_correction": entity.get("manual_correction", None),
            "status": "error: doc_id missing in JSON"
        }
    
    # CONSTRUIR document_path desde docs_base_path + doc_id
    document_path = os.path.join(docs_base_path, doc_id)
    # Si no existe, intentar con varias alternativas comunes:
    #  - doc_id + .txt
    #  - doc_id + .txt.txt (a veces hay doble extensión accidental)
    #  - buscar archivos en la carpeta cuyo nombre empiece por doc_id (insensible a mayúsculas)
    try_path = Path(document_path)
    if not try_path.exists():
        alt = Path(docs_base_path) / (str(doc_id) + '.txt')
        if alt.exists():
            document_path = str(alt)
        else:
            alt2 = Path(docs_base_path) / (str(doc_id) + '.txt.txt')
            if alt2.exists():
                document_path = str(alt2)
            else:
                # Búsqueda tolerante (no recursiva) por nombres que empiecen por doc_id
                try:
                    base = Path(docs_base_path)
                    for p in base.iterdir():
                        if p.is_file() and p.name.lower().startswith(str(doc_id).lower()):
                            document_path = str(p)
                            break
                except Exception:
                    # Si falla la iteración (p.ej. permisos), seguimos y se reportará como no encontrado
                    pass
    
    # Cargar documento desde document_path construido
    try:
        doc_path = Path(document_path)
        document_text = load_document_text(doc_path)
    except FileNotFoundError:
        return {
            "document_id": doc_id,
            "document_path": document_path,
            "entity_id": entity.get("id", entity.get("entity_id", "unknown")),
            "keyword": keyword,
            "label": label,
            "start": start,
            "end": end,
            "context": "",
            "llm_response": "",
            "is_valid": None,
            "manual_correction": entity.get("manual_correction", None),
            "status": f"error: document not found at {document_path}"
        }
    except Exception as e:
        return {
            "document_id": doc_id,
            "document_path": document_path,
            "entity_id": entity.get("id", entity.get("entity_id", "unknown")),
            "keyword": keyword,
            "label": label,
            "start": start,
            "end": end,
            "context": "",
            "llm_response": "",
            "is_valid": None,
            "manual_correction": entity.get("manual_correction", None),
            "status": f"error loading document: {str(e)}"
        }
    
    # Extraer contexto usando índices del JSON
    context_text = extract_context_window(
        document_text,
        start,
        end,
        left_window,
        right_window
    )
    
    # Construir ubicación
    location = f"offset {start}-{end}"
    
    # Construir prompts
    system_prompt, user_prompt = build_prompt_for_entity(
        keyword=keyword,
        context_text=context_text,
        label=label,
        location=location,
        rules_text=rules_text
    )
    
    try:
        # Llamar a Ollama con gemma3:270m
        llm_raw, llm_response = call_ollama_gemma(system_prompt, user_prompt, debug=debug)

        # Parsear respuesta
        is_valid = parse_llm_response(llm_response)
        
        # Construir resultado
        result = {
            "document_id": doc_id,
            "document_path": document_path,
            "entity_id": entity.get("id", entity.get("entity_id", "unknown")),
            "keyword": keyword,
            "label": label,
            "start": start,
            "end": end,
            "context": context_text,
            "llm_response": llm_response,
            "llm_raw": llm_raw,
            "is_valid": is_valid,
            "manual_correction": entity.get("manual_correction", None),
            "status": "success" if is_valid is not None else "invalid_response"
        }
        
    except Exception as e:
        # Error en el procesamiento
        result = {
            "document_id": doc_id,
            "document_path": document_path,
            "entity_id": entity.get("id", entity.get("entity_id", "unknown")),
            "keyword": keyword,
            "label": label,
            "start": start,
            "end": end,
            "context": context_text[:100] + "..." if len(context_text) > 100 else context_text,
            "llm_response": "",
            "is_valid": None,
            "manual_correction": entity.get("manual_correction", None),
            "status": f"error: {str(e)}"
        }
    
    return result


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_llm_entity_judgment(
    entities_json_path: str,
    docs_base_path: str,
    rules_file: str = "guias-anotacion.json",
    output_csv: str = "llm_entity_judgments.json",
    left_window: int = 80,
    right_window: int = 80,
    debug: bool = False
) -> None:
    """
    Ejecuta el pipeline completo de evaluación de entidades con LLM.
    
    IMPORTANTE: El JSON debe contener estos campos en cada entidad:
    - doc_id/document_id: SOLO nombre del archivo (ej: "12345.txt")
    - start: índice de inicio de la entidad en el documento
    - end: índice de fin de la entidad en el documento
    - label: tipo de entidad
    - text/entity_text/keyword: texto de la entidad
    
    El script construirá document_path = docs_base_path + doc_id
    
    Args:
        entities_json_path: Ruta al JSON con las entidades detectadas
        docs_base_path: Ruta base donde están los documentos (OBLIGATORIO)
        rules_file: Archivo JSON con las reglas de anotación
        output_csv: Ruta del archivo JSON de salida con resultados
        left_window: Caracteres de contexto a la izquierda
        right_window: Caracteres de contexto a la derecha
    """
    print("=" * 80)
    print("PIPELINE DE EVALUACIÓN CON OLLAMA (gemma3:270m)")
    print("=" * 80)
    
    # 1. Verificar Ollama
    print("\n[1/6] Verificando Ollama...")
    try:
        if not check_ollama_available():
            raise RuntimeError("Ollama no está disponible")
        print(f"  ✓ Ollama está instalado")
        print(f"  ✓ Modelo: gemma3:270m")
        print(f"  ℹ Asegúrate de haber ejecutado: ollama pull gemma3:270m")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print(f"  ℹ Instala Ollama desde: https://ollama.ai")
        print(f"  ℹ Luego ejecuta: ollama pull gemma3:270m")
        sys.exit(1)
    
    # 2. Cargar reglas
    print("\n[2/6] Cargando reglas de anotación...")
    try:
        rules = load_entity_rules(rules_file)
        print(f"  ✓ {len(rules)} tipos de entidades cargados")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)
    
    # 3. Cargar entidades
    print("\n[3/6] Cargando entidades detectadas...")
    entities_path = Path(entities_json_path)
    if not entities_path.exists():
        print(f"  ✗ Error: No se encontró {entities_json_path}")
        sys.exit(1)
    
    with open(entities_path, 'r', encoding='utf-8') as f:
        entities_data = json.load(f)
    
    # El JSON puede ser una lista de entidades o un dict con metadata
    if isinstance(entities_data, list):
        entities = entities_data
    elif isinstance(entities_data, dict) and "entities" in entities_data:
        entities = entities_data["entities"]
    else:
        print(f"  ✗ Error: Formato de JSON no reconocido")
        sys.exit(1)
    
    print(f"  ✓ {len(entities)} entidades cargadas")
    
    # 4. Validar que las entidades tienen doc_id y que docs_base_path existe
    print("\n[4/6] Validando campos del JSON y docs_base_path...")
    missing_doc_ids = []
    for idx, entity in enumerate(entities):
        doc_id = entity.get("document_id", entity.get("doc_id"))
        if not doc_id:
            missing_doc_ids.append(idx)
    
    if missing_doc_ids:
        print(f"  ⚠ Advertencia: {len(missing_doc_ids)} entidades sin 'doc_id'")
        print(f"    Índices: {missing_doc_ids[:10]}{'...' if len(missing_doc_ids) > 10 else ''}")
    else:
        print(f"  ✓ Todas las entidades tienen 'doc_id'")
    
    # Verificar que docs_base_path existe
    docs_base = Path(docs_base_path)
    if not docs_base.exists():
        print(f"  ✗ Error: La ruta base de documentos no existe: {docs_base_path}")
        sys.exit(1)
    else:
        print(f"  ✓ Ruta base de documentos: {docs_base_path}")
    
    # 5. Procesar entidades
    print("\n[5/6] Procesando entidades con LLM...")
    results = []
    total_entities = len(entities)
    
    for idx, entity in enumerate(entities, 1):
        doc_id = entity.get("document_id", entity.get("doc_id", "unknown"))
        label = entity.get("label", "unknown")
        keyword = entity.get("text", entity.get("entity_text", entity.get("keyword", "")))
        
        # Truncar keyword si es muy largo para el display
        display_keyword = keyword[:30] + "..." if len(keyword) > 30 else keyword
        print(f"  [{idx}/{total_entities}] {doc_id} | {label}: '{display_keyword}'", end=" ")
        
        # Obtener reglas de la etiqueta desde guias-anotacion.json
        try:
            rules_text = get_entity_rules_for_label(label, rules_file)
        except KeyError:
            print(f"✗ (etiqueta desconocida)")
            results.append({
                "document_id": doc_id,
                "document_path": "",
                "entity_id": entity.get("id", entity.get("entity_id", "unknown")),
                "keyword": keyword,
                "label": label,
                "start": entity.get("start", 0),
                "end": entity.get("end", 0),
                "context": "",
                "llm_response": "",
                "is_valid": None,
                "manual_correction": entity.get("manual_correction", None),
                "status": "error: unknown label"
            })
            continue
        
        # Procesar entidad (construye document_path internamente con docs_base_path + doc_id)
        result = process_entity(
            entity,
            rules_text,
            docs_base_path,
            left_window,
            right_window,
            debug=debug
        )
        
        results.append(result)
        
        # Mostrar resultado
        if result["status"] == "success":
            verdict = "✓ TRUE" if result["is_valid"] else "✗ FALSE"
            print(verdict)
        else:
            print(f"✗ ({result['status']})")
        # Mostrar EXACTAMENTE lo que devolvió Ollama (sin transcripciones)
        if "llm_raw" in result and result["llm_raw"]:
            print("--- RESPUESTA EXACTA DE OLLAMA (sin procesar) ---")
            print(result["llm_raw"])
    
    # 6. Guardar resultados
    print(f"\n[6/6] Guardando resultados en JSON...")
    output_path = Path(output_csv)
    # Cambiar extensión a .json si no lo es
    if output_path.suffix.lower() != '.json':
        output_path = output_path.with_suffix('.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Resultados guardados en: {output_path}")
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE EVALUACIÓN")
    print("=" * 80)
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    valid = sum(1 for r in results if r["is_valid"] is True)
    invalid = sum(1 for r in results if r["is_valid"] is False)
    errors = sum(1 for r in results if r["status"].startswith("error"))
    
    print(f"Total de entidades evaluadas: {total}")
    print(f"  ✓ Evaluaciones exitosas: {success} ({success/total*100:.1f}%)")
    print(f"  ✓ Entidades VÁLIDAS (TRUE): {valid} ({valid/total*100:.1f}%)")
    print(f"  ✗ Entidades INVÁLIDAS (FALSE): {invalid} ({invalid/total*100:.1f}%)")
    print(f"  ✗ Errores: {errors} ({errors/total*100:.1f}%)")
    print("=" * 80)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Función principal para ejecución desde CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluador de entidades con Ollama (gemma3:270m)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Evaluar entidades de un JSON (doc_id solo contiene nombre del archivo)
  python llm_entity_judge.py \\
    --entities entities.json \\
    --docs datasets/documents \\
    --output results.json

  # Con ventanas de contexto personalizadas
  python llm_entity_judge.py \\
    --entities entities.json \\
    --docs corpus/output/aws2 \\
    --output results.json \\
    --left-window 120 \\
    --right-window 120

Requisitos:
  - Ollama instalado: https://ollama.ai
  - Modelo gemma3:270m: ollama pull gemma3:270m
  - JSON con campos obligatorios por entidad:
    * doc_id/document_id: SOLO nombre del archivo (ej: "12345.txt")
    * start: índice de inicio en el documento
    * end: índice de fin en el documento
    * label: tipo de entidad
    * text/entity_text/keyword: texto de la entidad
  - Carpeta con documentos (parámetro --docs)
  
  El script construirá: document_path = docs_base_path + doc_id
        """
    )
    
    parser.add_argument(
        '--entities',
        required=True,
        help='Ruta al JSON con entidades detectadas (debe incluir doc_id con SOLO el nombre del archivo, start, end, label, keyword)'
    )
    
    parser.add_argument(
        '--rules-file',
        default='guias-anotacion.json',
        help='Archivo JSON con reglas de anotación (default: guias-anotacion.json)'
    )
    
    parser.add_argument(
        '--output',
        default='llm_entity_judgments.json',
        help='Archivo JSON de salida (default: llm_entity_judgments.json)'
    )
    
    parser.add_argument(
        '--left-window',
        type=int,
        default=80,
        help='Caracteres de contexto a la izquierda (default: 80)'
    )
    
    parser.add_argument(
        '--right-window',
        type=int,
        default=80,
        help='Caracteres de contexto a la derecha (default: 80)'
    )
    
    parser.add_argument(
        '--docs',
        required=True,
        help='Ruta base donde están los documentos (se combinará con doc_id del JSON para formar la ruta completa)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Activar modo debug: imprime prompts y salida de Ollama (útil para depuración)'
    )
    
    args = parser.parse_args()
    
    try:
        run_llm_entity_judgment(
            entities_json_path=args.entities,
            docs_base_path=args.docs,
            rules_file=args.rules_file,
            output_csv=args.output,
            left_window=args.left_window,
            right_window=args.right_window,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n\n✗ Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
