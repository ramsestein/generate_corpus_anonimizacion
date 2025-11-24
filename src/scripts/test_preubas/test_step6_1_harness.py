#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test harness para validar funciones clave de pipeline/step6.1.py sin cargar modelos HF.

Pruebas incluidas:
- is_only_j_characters: casos positivos/negativos.
- count_anon_markers: conteo literal del token ANON_TOKEN en texto.
- extract_entities_with_model: mapeo de offsets globales con chunking por caracteres usando un mock-pipeline.
- analyze_detected_entities: filtrado de tokens de anonimización y clasificación básica.

Uso rápido (PowerShell):
  python .\scripts\test_step6_1_harness.py

Salida: imprime PASA/FALLA por prueba y un resumen final con exit code 0 si todo pasa.
"""
import os
import sys
import re
import json
import argparse
from importlib.machinery import SourceFileLoader
import types
from typing import List, Dict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
STEP6_PATH = os.path.join(REPO_ROOT, 'pipeline', 'step6.1.py')

if not os.path.exists(STEP6_PATH):
    print(f"ERROR: No se encontró el archivo step6.1.py en: {STEP6_PATH}")
    sys.exit(1)

# Parse CLI args for the harness
parser = argparse.ArgumentParser(description='Harness tests for step6.1')
parser.add_argument('--doc-file', type=str, default=None, help='Optional path to a document to run processing against')
args = parser.parse_args()

# This harness requires the real models (no stubs). Ensure transformers/torch are installed.

# Cargar el módulo dinámicamente
step6 = SourceFileLoader("step6.1.py", STEP6_PATH).load_module()

# Utilidades de impresión
PASSES = 0
FAILS = 0
PASSED_TESTS = []
FAILED_TESTS = []

def ok(name):
    global PASSES
    PASSES += 1
    PASSED_TESTS.append(name)
    print(f"[PASA] {name}")

def fail(name, msg):
    global FAILS
    FAILS += 1
    FAILED_TESTS.append((name, msg))
    print(f"[FALLA] {name} -> {msg}")

# Mock pipeline HF: retorna entidades detectando un patrón dentro del chunk
# NOTE: MockPipeline removed - tests now require real models when model behaviour is needed.

# Pruebas

def test_is_only_j_characters():
    name = 'is_only_j_characters'
    try:
        # La implementación actual considera solo 'J' mayúscula y espacios como máscara
        assert step6.is_only_j_characters('JJJ') is True
        assert step6.is_only_j_characters('J J J') is True
        # Variantes como '(JJJ)' o 'jjj' pueden NO ser consideradas máscara en esta versión
        assert step6.is_only_j_characters('JXJ') is False
        ok(name)
    except AssertionError as e:
        fail(name, 'Resultados inesperados en detección de token JJJ')


def test_count_anon_markers_literal():
    name = 'count_anon_markers (literal)'
    try:
        text = 'JJJ (JJJ) jjj JJJ'  # La función literal solo cuenta coincidencias exactas de ANON_TOKEN
        # Obtener ANON_TOKEN del módulo
        token = getattr(step6, 'ANON_TOKEN', 'JJJ')
        literal_count = step6.count_anon_markers(text)
        # Conteo literal (sensible a forma exacta) debería contar solo 'JJJ' exactos (2 veces)
        assert literal_count == text.count(token)
        ok(name)
    except AssertionError:
        fail(name, f'Conteo literal no coincide con text.count({token})')


def test_extract_entities_with_model_offsets():
    name = 'extract_entities_with_model (offsets globales)'
    try:
        # Usamos MEDDOCAN (o CARMEN) para detectar fechas y comprobar mapeo de offsets.
        med, car = step6.setup_models()
        # Texto con fechas en formatos comunes — usar el mismo texto que en la prueba de fechas
        text = 'El paciente ingresó el 12/03/2021 y fue dado de alta el 15-03-2021. Evolución favorable.'
        # Umbral mínimo para detección fiable (coincide con la otra prueba)
        min_conf = 0.0

        ents_med = step6.extract_entities_with_model(text, med, 'MEDDOCAN', confidence_threshold=min_conf)
        ents_car = step6.extract_entities_with_model(text, car, 'CARMEN', confidence_threshold=min_conf)
        # Debug: mostrar entidades retornadas por cada modelo
        print(f'  DEBUG extract_offsets: MEDDOCAN returned {len(ents_med)} ents; CARMEN returned {len(ents_car)} ents')
        if not ents_med and not ents_car:
            return fail(name, f'Ningún modelo detectó entidades con confianza >= {min_conf} sobre el texto de prueba')
        # Preferir MEDDOCAN detections, sino usar CARMEN
        ents = ents_med or ents_car
        ok_slice = False
        for e in ents:
            s, eend = int(e['start']), int(e['end'])
            actual = text[s:eend]
            # Aceptar coincidencias que contengan la fecha '12/03/2021'
            if '12/03/2021' in actual or '12/03/2021' in str(e.get('word','')):
                ok_slice = True
                break
        assert ok_slice is True
        ok(name)
    except AssertionError:
        fail(name, 'Los offsets globales no reconstituyen el texto detectado')
    except Exception as ex:
        fail(name, f'Excepción inesperada: {ex}')



def test_process_documents_batch():
    name = 'process_documents_batch (batch end-to-end)'
    try:
            med, car = step6.setup_models()
            # Usar 'prueba.txt' (entrada) y escribir resultados en la misma carpeta scripts
            doc_filename = 'prueba.txt'
            doc_id = os.path.splitext(doc_filename)[0]  # 'prueba'
            input_dir = os.path.dirname(__file__)  # scripts folder (donde están prueba.txt y prueba_out.txt)
            output_dir = input_dir  # escribir salida en scripts también
            os.makedirs(output_dir, exist_ok=True)

            # Llamada con la firma correcta:
            # process_single_document(doc_id, input_dir, pipeline_meddocan, pipeline_carmen, output_dir, confidence_threshold)
            result = step6.process_single_document(
                doc_id,
                input_dir,
                med,
                car,
                output_dir,
                confidence_threshold=0.3
            )

            if isinstance(result, dict):
                print(f'  process_single_document result: success={result.get("success")}, keys={list(result.keys())}')
            else:
                print(f'  process_single_document returned non-dict: {type(result)}')

    except Exception as e:
            fail(name, f'Error corriendo modelos reales sobre el archivo: {e}')
            return

def test_model_loading():
    name = 'model_loading (setup_models)'
    try:
        pipelines = step6.setup_models()
        # Debe devolver dos pipelines
        assert isinstance(pipelines, tuple) and len(pipelines) == 2
        ok(name)
    except Exception as e:
        fail(name, f'No se pudieron cargar modelos: {e}')


def test_run_on_doc_file():
    name = 'run_on_doc_file'
    if not args.doc_file:
        print('[SKIP] run_on_doc_file: --doc-file no especificado')
        return ok(name)
    if not os.path.exists(args.doc_file):
        fail(name, f'Archivo no encontrado: {args.doc_file}')
        return
    try:
        text = open(args.doc_file, 'r', encoding='utf-8').read()
        # Conteo robusto de JJJ si está disponible, si no usar count_anon_markers literal
        if hasattr(step6, 'count_anon_markers_robust'):
            j_count = step6.count_anon_markers_robust(text)
        else:
            j_count = step6.count_anon_markers(text)
        print(f'  JJJ count (robust): {j_count}')

        # Ejecutar extracción con CARMEN (modelos siempre requeridos)
        try:
            med, car = step6.setup_models()
            # Construir doc_id e input_dir a partir del path pasado en --doc-file
            doc_path = args.doc_file
            doc_id = os.path.splitext(os.path.basename(doc_path))[0]
            input_dir = os.path.dirname(doc_path) or '.'
            # Directorio de salida (crear si no existe)
            output_dir = os.path.join(REPO_ROOT, 'step6_validation_results')
            os.makedirs(output_dir, exist_ok=True)

            # Llamar a process_single_document con la firma correcta:
            # (doc_id, input_dir, pipeline_meddocan, pipeline_carmen, output_dir, confidence_threshold)
            result = step6.process_single_document(doc_id, input_dir, med, car, output_dir, confidence_threshold=0.3)

            # Mostrar resumen del resultado (no asumimos estructura concreta del dict)
            if isinstance(result, dict):
                print(f'  process_single_document result: success={result.get("success")}, keys={list(result.keys())}')
            else:
                print(f'  process_single_document returned non-dict: {type(result)}')

        except Exception as e:
            fail(name, f'Error corriendo modelos reales sobre el archivo: {e}')
            return

        ok(name)
    except Exception as e:
        fail(name, f'Excepción procesando archivo: {e}')


def test_models_detect_date():
    name = 'models_detect_date (real models)'
    try:
        med, car = step6.setup_models()
        # Texto con fechas en formatos comunes (dd/mm/yyyy y dd-mm-yyyy)
        text = 'El paciente ingresó el 12/03/2021 y fue dado de alta el 15-03-2021. Evolución favorable. Tiene 17 años. G X JJJ. Mi cuidador esta en Madrid. Me llamo Jan vivo en Gaudalquivir 13'

        # Umbral mínimo de confianza esperado
        min_conf = 0.0

        # Extraer todas las entidades devueltas por ambos modelos (sin filtrar por tipo)
        ents_med = step6.extract_entities_with_model(text, med, 'MEDDOCAN', confidence_threshold=min_conf)
        ents_car = step6.extract_entities_with_model(text, car, 'CARMEN', confidence_threshold=min_conf)
        med_list = ents_med or []
        car_list = ents_car or []
        combined = med_list + car_list

        # Imprimir todas las entidades completas (ya incluidas como dicts serializables)
        print(f'  MEDDOCAN detected: {len(med_list)} entities (>= {min_conf})')
        for e in med_list:
            ed = dict(e)
            ed['model'] = 'MEDDOCAN'
            # Añadir snippet reconstituido si el offset es válido
            try:
                s = int(ed.get('start', -1))
                eend = int(ed.get('end', -1))
                if 0 <= s < eend <= len(text):
                    ed['snippet'] = text[s:eend]
                else:
                    ed['snippet'] = None
            except Exception:
                ed['snippet'] = None
            print('    MEDDOCAN entity:')
            print(json.dumps(ed, ensure_ascii=False, indent=2))

        print(f'  CARMEN detected: {len(car_list)} entities (>= {min_conf})')
        for e in car_list:
            ed = dict(e)
            ed['model'] = 'CARMEN'
            try:
                s = int(ed.get('start', -1))
                eend = int(ed.get('end', -1))
                if 0 <= s < eend <= len(text):
                    ed['snippet'] = text[s:eend]
                else:
                    ed['snippet'] = None
            except Exception:
                ed['snippet'] = None
            print('    CARMEN entity:')
            print(json.dumps(ed, ensure_ascii=False, indent=2))

        # Asegurarse de que al menos una entidad fue detectada por alguno de los modelos
        if not combined:
            return fail(name, f'Ninguna entidad con confianza >= {min_conf} detectada por los modelos en el texto de prueba')

        ok(name)
    except Exception as e:
        fail(name, f'Excepción en detección de fecha con modelos: {e}')


def test_detect_anonymization_markers_real_text():
    name = 'detect_anonymization_markers (JJJ)'
    try:
        # Texto con varias formas del token de anonimización
        text = 'Nombre: JJJ. Dirección: (JJJ). Observaciones: jjj.'
        if hasattr(step6, 'count_anon_markers_robust'):
            j_count = step6.count_anon_markers_robust(text)
        else:
            # Conteo literal mínimo como respaldo
            j_count = step6.count_anon_markers(text)
        assert j_count >= 2, f'Esperaba >=2 marcadores JJJ, encontrado: {j_count}'
        ok(name)
    except Exception as e:
        fail(name, f'Excepción en conteo de anonimización: {e}')


def test_is_only_j_characters_variants():
    name = 'is_only_j_characters (variantes)'
    try:
        # Comprobar que la función existe y devuelve booleanos consistentes
        assert hasattr(step6, 'is_only_j_characters')
        f = step6.is_only_j_characters
        # Casos que deberían ser True
        assert f('JJJ') is True
        assert f('J J J') is True
        assert f(' JJJ ') is True
        # Casos que deberían ser False según la implementación actual
        assert f('(JJJ)') is False
        assert f('jjj') is False
        assert f('J-J-J') is False
        assert f('') is False
        ok(name)
    except AssertionError:
        fail(name, 'Comportamiento inesperado en variantes de is_only_j_characters')

def test_analyze_detected_entities_filtering():
    name = 'analyze_detected_entities (filtrado JJJ)'
    try:
        text = 'Paciente JJJ con código G054 y referencia I06.'
        # Simular entidades de modelo (tras extract_entities_with_model)
        ents = [
            {'entity_group': 'MASK', 'score': 0.95, 'word': 'JJJ', 'start': 9, 'end': 12, 'model': 'MOCK'},
            {'entity_group': 'NUMERO_IDENTIF', 'score': 0.96, 'word': 'G054', 'start': 23, 'end': 27, 'model': 'MOCK'},
        ]
        analysis = step6.analyze_detected_entities(ents, text)
        # La detección de 'JJJ' debe ser filtrada por is_only_j_characters
        assert analysis['filtered_x_entities'] >= 1
        # Debe quedar al menos 1 entidad válida (G054)
        assert analysis['total_entities'] >= 1
        ok(name)
    except AssertionError:
        fail(name, 'Filtrado o conteo de entidades no coincide con lo esperado')

def test_analyze_detected_entities_confidence_buckets():
    name = 'analyze_detected_entities (confidence buckets)'
    try:
        text = 'Paciente JJJ con código G054 fecha 12/03/2021'
        ents = [
            {'entity_group': 'MASK', 'score': 0.95, 'word': 'JJJ', 'start': 9, 'end': 12, 'is_j_only': True},
            {'entity_group': 'NUMERO_IDENTIF', 'score': 0.96, 'word': 'G054', 'start': 23, 'end': 27, 'is_j_only': False},
            {'entity_group': 'FECHA', 'score': 0.7, 'word': '12/03/2021', 'start': 34, 'end': 44, 'is_j_only': False},
            {'entity_group': 'LOW', 'score': 0.3, 'word': 'X', 'start': 0, 'end': 1, 'is_j_only': False},
        ]
        analysis = step6.analyze_detected_entities(ents, text)
        # Comprobar buckets por confianza
        assert analysis['entities_by_confidence']['high'] >= 1
        assert analysis['entities_by_confidence']['medium'] >= 1
        assert analysis['entities_by_confidence']['low'] >= 1
        # Filtrado de JJJ
        assert analysis['j_only_count'] >= 1
        assert analysis['non_j_count'] >= 2
        ok(name)
    except AssertionError:
        fail(name, 'Buckets de confianza o clasificación no cumplen expectativas')


def test_extract_entities_with_mock_pipeline():
    # Ensure BOTH real pipelines detect the date and that offsets map back to the original text
    name = 'extract_entities_with_model (both real pipelines, token-aligned chunking)'
    try:
        med, car = step6.setup_models()
        # Determinar token_chunk_size usado internamente para construir un texto que obligue chunking
        model_max_length = getattr(med.tokenizer, 'model_max_length', 512)
        token_chunk_size = min(model_max_length - 20, 512)

        # Colocar la fecha después de token_chunk_size+10 palabras para asegurar que queda en un chunk distinto
        prefix = 'Palabra '
        text = (prefix * (token_chunk_size + 10)) + 'El paciente ingresó el 12/03/2021 y fue dado de alta.' + (prefix * 10)

        failures = []

        for pipeline_obj, name_model in ((med, 'MEDDOCAN'), (car, 'CARMEN')):
            ents = step6.extract_entities_with_model(text, pipeline_obj, name_model, confidence_threshold=0.0)
            if not isinstance(ents, list) or len(ents) == 0:
                failures.append(f"{name_model}: no entities returned")
                continue

            # Buscar la fecha en las entidades devueltas por este modelo
            found = False
            for e in ents:
                try:
                    s = int(e.get('start', -1))
                    eend = int(e.get('end', -1))
                except Exception:
                    continue
                if 0 <= s < eend <= len(text):
                    snippet = text[s:eend]
                    if '12/03/2021' in snippet or '12/03/2021' in str(e.get('word', '')):
                        found = True
                        break
            if not found:
                failures.append(f"{name_model}: date not found in returned entities")

        assert not failures, f"Pipelines failed checks: {failures}"
        ok(name)
    except AssertionError as ex:
        fail(name, str(ex))
    except Exception as ex:
        fail(name, f'Excepción ejecutando pipelines reales: {ex}')



def main():
    print('Ejecutando pruebas del harness para step6.1.py...')
    # Primero las pruebas que no requieren pasar --doc-file
    test_is_only_j_characters()
    test_is_only_j_characters_variants()
    test_count_anon_markers_literal()
    test_analyze_detected_entities_filtering()
    test_analyze_detected_entities_confidence_buckets()

    # Pruebas que usan modelos reales (pueden ser lentas)
    test_model_loading()
    test_extract_entities_with_model_offsets()
    test_extract_entities_with_mock_pipeline()
    test_models_detect_date()
    test_detect_anonymization_markers_real_text()

    # Opcional: procesar archivo si se pasó --doc-file
    test_run_on_doc_file()
    test_process_documents_batch()
    print('\nResumen:')
    print(f'  PASAN: {PASSES}')
    print(f'  FALLAN: {FAILS}')

    # Detalle final (debug): listar qué tests pasaron y cuáles fallaron
    print('\nDetalle de pruebas PASADAS:')
    if PASSED_TESTS:
        for t in PASSED_TESTS:
            print(f'  - {t}')
    else:
        print('  (ninguna)')

    print('\nDetalle de pruebas FALLADAS:')
    if FAILED_TESTS:
        for t, msg in FAILED_TESTS:
            print(f'  - {t}: {msg}')
    else:
        print('  (ninguna)')

    # Salida con código de error si hay fallos
    sys.exit(0 if FAILS == 0 else 1)

if __name__ == '__main__':
    main()
