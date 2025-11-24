#!/usr/bin/env python3
"""
Test de memoria para step6.1: carga modelos BSC y procesa `prueba.txt` con threshold=0.3.

Objetivo:
- Medir memoria del proceso antes/después de cargar cada modelo.
- Probar carga secuencial y carga simultánea (ambos modelos en memoria).
- Ejecutar `process_single_document` sobre `prueba.txt` con confidence_threshold=0.3.

Uso (PowerShell):
  python .\scripts\test_step6_memory.py

Notas:
- Si no tiene `psutil` instalado se mostrará una advertencia y se usará información de CUDA/Torch si está disponible.
- Este test no borra nada; solo escribe el resultado JSON en la carpeta `scripts` (si `process_single_document` lo hace).
"""
import os
import sys
import gc
import time
import traceback
from importlib.machinery import SourceFileLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
STEP6_PATH = os.path.join(REPO_ROOT, 'pipeline', 'step6.1.py')

if not os.path.exists(STEP6_PATH):
    print(f"ERROR: no se encontró {STEP6_PATH}")
    sys.exit(1)

# Cargar módulo step6
step6 = SourceFileLoader('step6.1', STEP6_PATH).load_module()

try:
    import psutil
    PSUTIL = True
except Exception:
    psutil = None
    PSUTIL = False

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
    import torch
except Exception as e:
    print('ERROR: faltan dependencias transformers/torch. Instálelas según requirements.txt')
    raise


def mem_mb():
    """Devuelve memoria residente en MB (si psutil disponible); si no, intenta dar alguna métrica de torch."""
    if PSUTIL:
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024 ** 2)
    else:
        # Fallback: si hay GPU, mostrar uso de GPU por torch
        if torch.cuda.is_available():
            try:
                mem = torch.cuda.memory_allocated(0) / (1024 ** 2)
                return mem
            except Exception:
                return -1
        return -1


def load_model(path):
    """Carga tokenizer, model y pipeline para un path local de HF.
    Retorna (tokenizer, model, pipeline)
    """
    print(f"Cargando modelo desde: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(
        path,
        dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map=None,
        trust_remote_code=False
    )
    pipe = hf_pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy='simple', device=-1)
    return tokenizer, model, pipe


def safe_del(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()


def main():
    scripts_dir = os.path.dirname(__file__)
    prueba_path = os.path.join(scripts_dir, 'prueba.txt')
    if not os.path.exists(prueba_path):
        print('ERROR: no se encontró scripts/prueba.txt')
        sys.exit(1)

    med_path = os.path.join(REPO_ROOT, 'models', 'bsc-bio-ehr-es-meddocan')
    car_path = os.path.join(REPO_ROOT, 'models', 'bsc-bio-ehr-es-carmen-anon')

    print('\n=== MEMORY TEST FOR step6.1 ===')
    print(f'PID: {os.getpid()}')
    print(f'psutil available: {PSUTIL}')
    print(f'Initial memory (MB): {mem_mb():.1f}')

    # 1) Cargar MEDDOCAN solo
    try:
        print('\n-- Cargando MEDDOCAN (solo) --')
        t0 = time.time()
        med_tok, med_model, med_pipe = load_model(med_path)
        t1 = time.time()
        print(f'  Tiempo carga MEDDOCAN: {t1-t0:.1f}s')
        print(f'  Memoria tras MEDDOCAN (MB): {mem_mb():.1f}')
    except Exception as e:
        print('ERROR al cargar MEDDOCAN:', e)
        traceback.print_exc()
        med_tok = med_model = med_pipe = None

    # Liberar MEDDOCAN
    safe_del(med_model, med_tok)
    print(f'  Memoria tras eliminar MEDDOCAN (GC) (MB): {mem_mb():.1f}')

    # 2) Cargar CARMEN solo
    try:
        print('\n-- Cargando CARMEN (solo) --')
        t0 = time.time()
        car_tok, car_model, car_pipe = load_model(car_path)
        t1 = time.time()
        print(f'  Tiempo carga CARMEN: {t1-t0:.1f}s')
        print(f'  Memoria tras CARMEN (MB): {mem_mb():.1f}')
    except Exception as e:
        print('ERROR al cargar CARMEN:', e)
        traceback.print_exc()
        car_tok = car_model = car_pipe = None

    safe_del(car_model, car_tok)
    print(f'  Memoria tras eliminar CARMEN (GC) (MB): {mem_mb():.1f}')

    # 3) Cargar ambos modelos en memoria (simular escenario worst-case)
    print('\n-- Cargando MEDDOCAN y luego CARMEN (ambos residiendo) --')
    med_tok = med_model = med_pipe = None
    car_tok = car_model = car_pipe = None
    try:
        t0 = time.time()
        med_tok, med_model, med_pipe = load_model(med_path)
        t1 = time.time()
        print(f'  Tiempo carga MEDDOCAN: {t1-t0:.1f}s')
        print(f'  Memoria tras MEDDOCAN (MB): {mem_mb():.1f}')

        t0 = time.time()
        car_tok, car_model, car_pipe = load_model(car_path)
        t1 = time.time()
        print(f'  Tiempo carga CARMEN: {t1-t0:.1f}s')
        print(f'  Memoria tras CARMEN (ambos cargados) (MB): {mem_mb():.1f}')

    except Exception as e:
        print('ERROR cargando ambos modelos:', e)
        traceback.print_exc()

    # 4) Ejecutar process_single_document sobre prueba.txt con threshold 0.3
    print('\n-- Ejecutando process_single_document sobre prueba.txt (threshold=0.3) --')
    try:
        doc_id = os.path.splitext(os.path.basename(prueba_path))[0]
        out_dir = scripts_dir
        # Usar pipelines si existen, si no fallback a setup_models
        pipeline_med = med_pipe
        pipeline_car = car_pipe
        if pipeline_med is None or pipeline_car is None:
            print('  Pipelines no disponibles, intentando usar step6.setup_models()')
            pipeline_med, pipeline_car = step6.setup_models()

        print(f'  Memoria antes de procesar doc (MB): {mem_mb():.1f}')
        res = step6.process_single_document(doc_id, scripts_dir, pipeline_med, pipeline_car, out_dir, confidence_threshold=0.3)
        print(f'  Memoria después de procesar doc (MB): {mem_mb():.1f}')
        print('  Resultado (summary keys):', list(res.keys()) if isinstance(res, dict) else type(res))

    except Exception as e:
        print('ERROR al procesar documento de prueba:', e)
        traceback.print_exc()

    # Limpieza final
    safe_del(med_model, med_tok, med_pipe, car_model, car_tok, car_pipe)
    print(f'\nMemoria final (MB): {mem_mb():.1f}')


if __name__ == '__main__':
    main()
