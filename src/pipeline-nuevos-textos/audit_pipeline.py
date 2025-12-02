#!/usr/bin/env python3
"""
audit_pipeline.py - Auditoría Completa del Pipeline de Anonimización
======================================================================

Este script ejecuta una auditoría exhaustiva del pipeline verificando:
1. ✅ Coherencia: Todos los módulos se conectan correctamente
2. ✅ Ejecución: El pipeline corre de principio a fin sin errores
3. ✅ Métricas: Los cálculos contra GT son correctos
4. ✅ Datos: Los formatos de entrada/salida son consistentes

USO:
    python audit_pipeline.py                    # Auditoría completa
    python audit_pipeline.py --dry-run          # Solo verificar estructura
    python audit_pipeline.py --skip-llm         # Saltar etapa LLM (más rápido)

SALIDA:
    - Logs detallados en consola
    - Informe JSON en outputs/audit_report.json
"""

import argparse
import json
import logging
import sys
import traceback
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Configurar logging estructurado
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)-8s] [%(name)-15s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent.parent.parent / "outputs" / "audit_log.txt",
            mode='w',
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger("AUDIT")

# Paths del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
PIPELINE_DIR = Path(__file__).parent
DEFAULT_INPUT = PROJECT_ROOT / "src" /  "entidades-pipeline.json"
DEFAULT_GT_DIR = PROJECT_ROOT / "corpus" / "ANTIGUO" / "entidades"
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "corpus" / "ANTIGUO" / "documents"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "audit_report.json"


class AuditReport:
    """Clase para recopilar resultados de auditoría."""
    
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.checks = []
        self.errors = []
        self.warnings = []
        self.metrics = {}
        self.pipeline_stats = {}
        
    def add_check(self, name: str, passed: bool, details: str = ""):
        self.checks.append({
            'name': name,
            'passed': passed,
            'details': details
        })
        icon = "✅" if passed else "❌"
        logger.info(f"{icon} CHECK: {name} - {'PASSED' if passed else 'FAILED'}")
        if details:
            logger.info(f"   └── {details}")
    
    def add_error(self, stage: str, error: str, trace: str = ""):
        self.errors.append({
            'stage': stage,
            'error': error,
            'traceback': trace
        })
        logger.error(f"ERROR en {stage}: {error}")
    
    def add_warning(self, message: str):
        self.warnings.append(message)
        logger.warning(f"⚠️  {message}")
    
    def to_dict(self):
        passed = sum(1 for c in self.checks if c['passed'])
        failed = sum(1 for c in self.checks if not c['passed'])
        return {
            'metadata': {
                'timestamp': self.timestamp,
                'total_checks': len(self.checks),
                'passed': passed,
                'failed': failed,
                'errors': len(self.errors),
                'warnings': len(self.warnings)
            },
            'checks': self.checks,
            'errors': self.errors,
            'warnings': self.warnings,
            'pipeline_stats': self.pipeline_stats,
            'metrics': self.metrics
        }
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Informe guardado en: {path}")


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación."""
    text = text.lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    return ' '.join(text.split())


def text_matches(text1: str, text2: str) -> bool:
    """Verifica si dos textos coinciden (exacto o parcial)."""
    n1 = normalize_text(text1)
    n2 = normalize_text(text2)
    if n1 == n2:
        return True
    if len(n1) >= 3 and len(n2) >= 3:
        return n1 in n2 or n2 in n1
    return False


# ==================== FASE 1: VERIFICAR ESTRUCTURA ====================

def check_structure(report: AuditReport):
    """Verifica que todos los archivos y módulos existen."""
    logger.info("=" * 60)
    logger.info("FASE 1: VERIFICAR ESTRUCTURA DEL PIPELINE")
    logger.info("=" * 60)
    
    # Verificar módulos
    modules = [
        'setfit_module/__init__.py',
        'setfit_module/api.py',
        'setfit_module/gatekeeper.py',
        'dict_filters/__init__.py',
        'dict_filters/api.py',
        'dict_filters/filter.py',
        'llm_judge/__init__.py',
        'llm_judge/api.py',
        'llm_judge/judge.py',
        'io_json/__init__.py',
        'io_json/loaders.py',
        'run_full_pipeline.py'
    ]
    
    for mod in modules:
        path = PIPELINE_DIR / mod
        exists = path.exists()
        report.add_check(f"Módulo: {mod}", exists, str(path))
    
    # Verificar datos de entrada
    report.add_check(
        "Archivo de entrada existe",
        DEFAULT_INPUT.exists(),
        str(DEFAULT_INPUT)
    )
    
    # Verificar GT
    gt_exists = DEFAULT_GT_DIR.exists()
    gt_count = len(list(DEFAULT_GT_DIR.glob("*.json"))) if gt_exists else 0
    report.add_check(
        "Directorio GT existe",
        gt_exists and gt_count > 0,
        f"{gt_count} archivos JSON en {DEFAULT_GT_DIR}"
    )
    
    # Verificar documentos
    docs_exists = DEFAULT_DOCUMENTS_DIR.exists()
    report.add_check(
        "Directorio documentos existe",
        docs_exists,
        str(DEFAULT_DOCUMENTS_DIR)
    )
    
    # Verificar modelo SetFit
    setfit_path = PROJECT_ROOT / "models" / "gatekeeper_setfit"
    setfit_exists = setfit_path.exists()
    report.add_check(
        "Modelo SetFit existe",
        setfit_exists,
        str(setfit_path)
    )
    
    # Verificar listas de diccionarios
    lists_path = PIPELINE_DIR / "dict_filters" / "lists"
    if lists_path.exists():
        list_files = list(lists_path.glob("*.json"))
        report.add_check(
            "Listas de diccionarios existen",
            len(list_files) > 0,
            f"{len(list_files)} archivos: {[f.name for f in list_files]}"
        )
    else:
        report.add_warning(f"Directorio de listas no encontrado: {lists_path}")


# ==================== FASE 2: VERIFICAR IMPORTACIONES ====================

def check_imports(report: AuditReport):
    """Verifica que todos los módulos se pueden importar."""
    logger.info("=" * 60)
    logger.info("FASE 2: VERIFICAR IMPORTACIONES")
    logger.info("=" * 60)
    
    # Agregar pipeline al path
    sys.path.insert(0, str(PIPELINE_DIR))
    
    modules_to_import = [
        ('setfit_module', 'run_setfit_filter'),
        ('dict_filters', 'apply_dict_filters'),
        ('llm_judge', 'run_llm_judge'),
        ('io_json', 'load_entities'),
    ]
    
    for mod_name, func_name in modules_to_import:
        try:
            mod = __import__(mod_name)
            has_func = hasattr(mod, func_name)
            report.add_check(
                f"Import {mod_name}.{func_name}",
                has_func,
                f"Función exportada: {has_func}"
            )
        except Exception as e:
            report.add_error(f"Import {mod_name}", str(e), traceback.format_exc())
            report.add_check(f"Import {mod_name}", False, str(e))


# ==================== FASE 3: VERIFICAR DATOS DE ENTRADA ====================

def check_input_data(report: AuditReport) -> dict:
    """Verifica y carga los datos de entrada."""
    logger.info("=" * 60)
    logger.info("FASE 3: VERIFICAR DATOS DE ENTRADA")
    logger.info("=" * 60)
    
    input_data = None
    
    try:
        with open(DEFAULT_INPUT, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Verificar estructura
        has_entities = 'entities' in input_data
        entities = input_data.get('entities', [])
        
        report.add_check(
            "Estructura entrada: campo 'entities'",
            has_entities,
            f"{len(entities)} entidades"
        )
        
        if entities:
            # Verificar campos requeridos
            sample = entities[0]
            required_fields = ['doc_id', 'text', 'label']
            has_fields = all(f in sample for f in required_fields)
            
            report.add_check(
                "Campos requeridos en entidades",
                has_fields,
                f"Campos presentes: {list(sample.keys())}"
            )
            
            # Verificar doc_ids únicos
            doc_ids = set(e.get('doc_id', '') for e in entities)
            report.add_check(
                "Documentos en entrada",
                len(doc_ids) > 0,
                f"{len(doc_ids)} documentos únicos"
            )
            
            # Verificar textos no vacíos
            empty_texts = sum(1 for e in entities if not e.get('text', '').strip())
            report.add_check(
                "Textos no vacíos",
                empty_texts == 0,
                f"{empty_texts} textos vacíos de {len(entities)}"
            )
            
            # Estadísticas por etiqueta
            labels = defaultdict(int)
            for e in entities:
                labels[e.get('label', 'UNKNOWN')] += 1
            
            logger.info(f"Distribución de etiquetas: {dict(labels)}")
            
    except Exception as e:
        report.add_error("Cargar entrada", str(e), traceback.format_exc())
        report.add_check("Datos de entrada válidos", False, str(e))
    
    return input_data


# ==================== FASE 4: VERIFICAR GROUND TRUTH ====================

def check_ground_truth(report: AuditReport, doc_ids: set = None) -> dict:
    """Verifica y carga el Ground Truth."""
    logger.info("=" * 60)
    logger.info("FASE 4: VERIFICAR GROUND TRUTH")
    logger.info("=" * 60)
    
    gt_by_doc = {}
    
    try:
        gt_files = list(DEFAULT_GT_DIR.glob("*.json"))
        report.add_check(
            "Archivos GT disponibles",
            len(gt_files) > 0,
            f"{len(gt_files)} archivos"
        )
        
        # Cargar GT para documentos de entrada
        loaded = 0
        errors = 0
        
        for gt_file in gt_files:
            doc_id = gt_file.stem
            
            # Si tenemos doc_ids, filtrar
            if doc_ids and doc_id not in doc_ids:
                continue
            
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Verificar estructura GT
                if 'data' in data:
                    entities = data['data']
                    texts = set()
                    for e in entities:
                        if 'text' in e:
                            texts.add(normalize_text(e['text']))
                        elif 'entity_text' in e:  # Formato alternativo
                            texts.add(normalize_text(e['entity_text']))
                    
                    gt_by_doc[doc_id] = texts
                    loaded += 1
                else:
                    errors += 1
                    if errors <= 3:
                        report.add_warning(f"GT sin campo 'data': {gt_file.name}")
                        
            except Exception as e:
                errors += 1
                if errors <= 3:
                    report.add_warning(f"Error leyendo GT {gt_file.name}: {e}")
        
        if doc_ids:
            matched = len(gt_by_doc)
            report.add_check(
                "GT para documentos de entrada",
                matched > 0,
                f"{matched}/{len(doc_ids)} documentos tienen GT"
            )
        
        # Estadísticas de GT
        total_entities = sum(len(texts) for texts in gt_by_doc.values())
        logger.info(f"GT cargado: {loaded} docs, {total_entities} entidades únicas")
        
    except Exception as e:
        report.add_error("Cargar GT", str(e), traceback.format_exc())
    
    return gt_by_doc


# ==================== FASE 5: EJECUTAR PIPELINE ====================

def run_pipeline(report: AuditReport, input_data: dict, skip_llm: bool = False) -> dict:
    """Ejecuta el pipeline completo con el NUEVO FLUJO INVERTIDO."""
    logger.info("=" * 60)
    logger.info("FASE 5: EJECUTAR PIPELINE (FLUJO INVERTIDO)")
    logger.info("=" * 60)
    
    sys.path.insert(0, str(PIPELINE_DIR))
    
    # Nueva estructura de resultados para flujo invertido
    results = {
        'input': [],
        # SetFit clasifica en PII (directo) o RUIDO (a rescate)
        'setfit_pii': [],
        'setfit_ruido': [],
        # Dict filters rescata
        'dict_rescued': [],
        'dict_filtered': [],
        'dict_escalated': [],
        # LLM rescata
        'llm_rescued': [],
        'llm_filtered': [],
        'final_output': []
    }
    
    entities = input_data.get('entities', [])
    logger.info(f"Entidades de entrada: {len(entities)}")
    results['input'] = entities
    
    # ===== ETAPA 1: SetFit - Clasificación binaria PII/RUIDO =====
    logger.info("-" * 40)
    logger.info("ETAPA 1: SetFit - Clasificación Binaria")
    logger.info("-" * 40)
    
    try:
        from setfit_module import run_setfit_filter
        
        # Configuración por defecto para SetFit
        setfit_config = {
            "model_path": str(PROJECT_ROOT / "models" / "gatekeeper_setfit"),
            "confidence_threshold": 0.75,
            "enable_noise_filter": False,
            "enable_pii_detector": False,
            "enable_fragment_filter": False,
            "enable_low_confidence_filter": False,
        }
        
        # Preparar entidades para SetFit
        setfit_input = []
        for e in entities:
            setfit_input.append({
                'text': e.get('text', ''),
                'label': e.get('label', ''),
                'doc_id': e.get('doc_id', ''),
                'context': e.get('context', ''),
                # Preservar campos originales
                **{k: v for k, v in e.items() if k not in ['text', 'label', 'doc_id', 'context']}
            })
        
        setfit_results = run_setfit_filter(setfit_input, None, setfit_config)
        
        for r in setfit_results:
            # Preservar doc_id original
            original = next((e for e in entities if e.get('text') == r.get('entity_text')), {})
            r['doc_id'] = original.get('doc_id', r.get('document_id', ''))
            r['text'] = r.get('entity_text', '')
            
            # Nuevo: usar 'classification' en vez de 'decision'
            if r.get('classification') == 'PII':
                results['setfit_pii'].append(r)
            else:  # RUIDO
                results['setfit_ruido'].append(r)
        
        logger.info(f"  PII (directo a salida): {len(results['setfit_pii'])}")
        logger.info(f"  RUIDO (a rescate): {len(results['setfit_ruido'])}")
        
        report.add_check(
            "SetFit ejecutado",
            True,
            f"PII={len(results['setfit_pii'])}, RUIDO={len(results['setfit_ruido'])}"
        )
        
    except Exception as e:
        report.add_error("SetFit", str(e), traceback.format_exc())
        report.add_check("SetFit ejecutado", False, str(e))
        return results
    
    # ===== ETAPA 2: Dict Filters - Rescate de RUIDO =====
    logger.info("-" * 40)
    logger.info("ETAPA 2: Dict Filters - Rescate")
    logger.info("-" * 40)
    
    try:
        from dict_filters import apply_dict_filters
        
        # Configuración por defecto para dict_filters
        dict_config = {
            "json_whitelist_paths": [],
            "json_blacklist_paths": [],
            "csv_base_path": None,
            "cie10_path": None,
        }
        
        # Procesar entidades marcadas como RUIDO por SetFit
        to_dict = results['setfit_ruido']
        
        if to_dict:
            dict_results = apply_dict_filters(to_dict, None, dict_config)
            
            for r in dict_results:
                decision = r.get('decision', 'ESCALATE')
                if decision == 'KEEP':
                    # Añadir trazabilidad
                    r['classification'] = 'PII'
                    r['classification_source'] = 'dict_whitelist'
                    results['dict_rescued'].append(r)
                elif decision == 'FILTER':
                    results['dict_filtered'].append(r)
                else:
                    results['dict_escalated'].append(r)
        
        logger.info(f"  RESCATADOS: {len(results['dict_rescued'])}")
        logger.info(f"  CONFIRMADOS RUIDO: {len(results['dict_filtered'])}")
        logger.info(f"  DUDOSOS (a LLM): {len(results['dict_escalated'])}")
        
        report.add_check(
            "Dict Filters ejecutado",
            True,
            f"Rescued={len(results['dict_rescued'])}, Filtered={len(results['dict_filtered'])}, Escalated={len(results['dict_escalated'])}"
        )
        
    except Exception as e:
        report.add_error("Dict Filters", str(e), traceback.format_exc())
        report.add_check("Dict Filters ejecutado", False, str(e))
        return results
    
    # ===== ETAPA 3: LLM Judge - Rescate final =====
    logger.info("-" * 40)
    logger.info("ETAPA 3: LLM Judge - Rescate Final")
    logger.info("-" * 40)
    
    if skip_llm:
        logger.info("  ⏭️  LLM omitido (--skip-llm)")
        # Si skip LLM, las escaladas se DESCARTAN (flujo invertido = conservador)
        report.add_check("LLM Judge omitido", True, f"{len(results['dict_escalated'])} entidades descartadas")
    else:
        try:
            from llm_judge import run_llm_judge
            
            # Configuración por defecto para LLM
            llm_config = {
                "model": "gemma3:270m",
                "rules_path": None,
                "timeout": 120,
                "max_retries": 2,
            }
            
            to_llm = results['dict_escalated']
            
            if to_llm:
                llm_results = run_llm_judge(to_llm, None, llm_config)
                
                for r in llm_results:
                    if r.get('decision') == 'KEEP':
                        # Añadir trazabilidad
                        r['classification'] = 'PII'
                        r['classification_source'] = 'llm_rescue'
                        results['llm_rescued'].append(r)
                    else:
                        results['llm_filtered'].append(r)
            
            logger.info(f"  LLM RESCATADOS: {len(results['llm_rescued'])}")
            logger.info(f"  LLM CONFIRMADOS RUIDO: {len(results['llm_filtered'])}")
            
            report.add_check(
                "LLM Judge ejecutado",
                True,
                f"Rescued={len(results['llm_rescued'])}, Filtered={len(results['llm_filtered'])}"
            )
            
        except Exception as e:
            report.add_error("LLM Judge", str(e), traceback.format_exc())
            report.add_check("LLM Judge ejecutado", False, str(e))
    
    # ===== Combinar salida final =====
    # Nuevo flujo: PII directo + rescatados por listas + rescatados por LLM
    results['final_output'] = (
        results['setfit_pii'] + 
        results['dict_rescued'] + 
        results['llm_rescued']
    )
    
    logger.info("-" * 40)
    logger.info(f"SALIDA FINAL: {len(results['final_output'])} entidades PII")
    logger.info("-" * 40)
    
    # Guardar estadísticas con nueva nomenclatura
    report.pipeline_stats = {
        'total_input': len(entities),
        'setfit_pii': len(results['setfit_pii']),
        'setfit_ruido': len(results['setfit_ruido']),
        'dict_rescued': len(results['dict_rescued']),
        'dict_filtered': len(results['dict_filtered']),
        'dict_escalated': len(results['dict_escalated']),
        'llm_rescued': len(results['llm_rescued']),
        'llm_filtered': len(results['llm_filtered']),
        'final_output': len(results['final_output'])
    }
    
    return results


# ==================== FASE 6: CALCULAR MÉTRICAS ====================

def calculate_metrics(report: AuditReport, pipeline_results: dict, gt_by_doc: dict):
    """Calcula métricas contra GT."""
    logger.info("=" * 60)
    logger.info("FASE 6: CALCULAR MÉTRICAS VS GROUND TRUTH")
    logger.info("=" * 60)
    
    # Construir set de KEEP
    keep_by_doc = defaultdict(set)
    for entity in pipeline_results['final_output']:
        doc_id = entity.get('doc_id', entity.get('document_id', ''))
        text = entity.get('text', entity.get('entity_text', ''))
        if doc_id and text:
            keep_by_doc[doc_id].add(normalize_text(text))
    
    # Calcular TP, FP, TN, FN
    tp = fp = tn = fn = 0
    examples = {'tp': [], 'fp': [], 'tn': [], 'fn': []}
    
    for entity in pipeline_results['input']:
        doc_id = entity.get('doc_id', '')
        text = entity.get('text', '')
        text_norm = normalize_text(text)
        label = entity.get('label', 'UNKNOWN')
        
        if doc_id not in gt_by_doc:
            continue
        
        gt_texts = gt_by_doc[doc_id]
        kept_texts = keep_by_doc[doc_id]
        
        # ¿Está en GT?
        in_gt = any(text_matches(text_norm, gt) for gt in gt_texts) if gt_texts else False
        
        # ¿Fue KEEP?
        was_kept = any(text_matches(text_norm, k) for k in kept_texts) if kept_texts else False
        
        if was_kept:
            if in_gt:
                tp += 1
                if len(examples['tp']) < 3:
                    examples['tp'].append({'text': text[:40], 'label': label, 'doc': doc_id[:8]})
            else:
                fp += 1
                if len(examples['fp']) < 5:
                    examples['fp'].append({'text': text[:40], 'label': label, 'doc': doc_id[:8]})
        else:
            if in_gt:
                fn += 1
                if len(examples['fn']) < 5:
                    examples['fn'].append({'text': text[:40], 'label': label, 'doc': doc_id[:8]})
            else:
                tn += 1
                if len(examples['tn']) < 3:
                    examples['tn'].append({'text': text[:40], 'label': label, 'doc': doc_id[:8]})
    
    # Calcular métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    
    report.metrics = {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'accuracy': round(accuracy, 4),
        'examples': examples
    }
    
    # Verificar métricas razonables
    report.add_check(
        "Métricas calculadas",
        (tp + fp + fn + tn) > 0,
        f"TP={tp}, FP={fp}, TN={tn}, FN={fn}"
    )
    
    report.add_check(
        "F1 > 0.5 (razonable)",
        f1 > 0.5,
        f"F1 = {f1:.4f}"
    )
    
    report.add_check(
        "Recall > 0.8 (conservador)",
        recall > 0.8,
        f"Recall = {recall:.4f}"
    )
    
    # Imprimir resultados
    logger.info("")
    logger.info("📊 MÉTRICAS FINALES:")
    logger.info(f"  TP (KEEP correcto): {tp}")
    logger.info(f"  FP (KEEP incorrecto): {fp}")
    logger.info(f"  TN (FILTER correcto): {tn}")
    logger.info(f"  FN (FILTER incorrecto - PII perdido): {fn}")
    logger.info("")
    logger.info(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
    logger.info(f"  Recall:    {recall:.4f} ({recall*100:.1f}%)")
    logger.info(f"  F1:        {f1:.4f} ({f1*100:.1f}%)")
    logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    if examples['fn']:
        logger.info("")
        logger.info("⚠️  EJEMPLOS DE FN (PII perdido):")
        for ex in examples['fn']:
            logger.info(f"  [{ex['doc']}] {ex['label']:25s} | \"{ex['text']}\"")
    
    if examples['fp']:
        logger.info("")
        logger.info("ℹ️  EJEMPLOS DE FP (ruido conservado):")
        for ex in examples['fp'][:3]:
            logger.info(f"  [{ex['doc']}] {ex['label']:25s} | \"{ex['text']}\"")


# ==================== FASE 7: VERIFICAR COHERENCIA ====================

def verify_coherence(report: AuditReport, pipeline_results: dict):
    """Verifica coherencia interna del pipeline con FLUJO INVERTIDO."""
    logger.info("=" * 60)
    logger.info("FASE 7: VERIFICAR COHERENCIA (FLUJO INVERTIDO)")
    logger.info("=" * 60)
    
    stats = report.pipeline_stats
    
    # Verificar que entrada = PII + RUIDO
    total_input = stats['total_input']
    total_setfit = stats['setfit_pii'] + stats['setfit_ruido']
    
    report.add_check(
        "SetFit procesa todas las entidades",
        total_input == total_setfit,
        f"Input={total_input}, SetFit procesó={total_setfit} (PII={stats['setfit_pii']}, RUIDO={stats['setfit_ruido']})"
    )
    
    # Verificar flujo de RUIDO hacia rescate
    setfit_ruido = stats['setfit_ruido']
    dict_total = stats['dict_rescued'] + stats['dict_filtered'] + stats['dict_escalated']
    
    report.add_check(
        "Dict recibe todo el RUIDO",
        setfit_ruido == dict_total or setfit_ruido == 0,
        f"SetFit RUIDO={setfit_ruido}, Dict procesó={dict_total}"
    )
    
    # Verificar flujo de ESCALATE hacia LLM
    dict_escalated = stats['dict_escalated']
    llm_total = stats['llm_rescued'] + stats['llm_filtered']
    
    report.add_check(
        "LLM recibe todas las escaladas",
        dict_escalated == llm_total or dict_escalated == 0,
        f"Dict ESCALATE={dict_escalated}, LLM procesó={llm_total}"
    )
    
    # Verificar salida final = PII directo + rescatados
    expected_output = stats['setfit_pii'] + stats['dict_rescued'] + stats['llm_rescued']
    actual_output = stats['final_output']
    
    report.add_check(
        "Salida final = PII + rescatados",
        expected_output == actual_output,
        f"Esperado={expected_output} (PII={stats['setfit_pii']}+Dict={stats['dict_rescued']}+LLM={stats['llm_rescued']}), Actual={actual_output}"
    )


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Auditoría completa del pipeline")
    parser.add_argument('--dry-run', action='store_true', help="Solo verificar estructura")
    parser.add_argument('--skip-llm', action='store_true', help="Omitir etapa LLM")
    parser.add_argument('--output', '-o', default=str(DEFAULT_OUTPUT), help="Archivo de salida")
    args = parser.parse_args()
    
    report = AuditReport()
    
    logger.info("🔍 INICIANDO AUDITORÍA DEL PIPELINE")
    logger.info(f"   Timestamp: {report.timestamp}")
    logger.info(f"   Dry run: {args.dry_run}")
    logger.info(f"   Skip LLM: {args.skip_llm}")
    
    # Fase 1: Estructura
    check_structure(report)
    
    # Fase 2: Importaciones
    check_imports(report)
    
    if args.dry_run:
        logger.info("\n⏭️  Modo dry-run: omitiendo ejecución del pipeline")
    else:
        # Fase 3: Datos de entrada
        input_data = check_input_data(report)
        
        if input_data:
            # Extraer doc_ids
            doc_ids = set(e.get('doc_id', '') for e in input_data.get('entities', []))
            
            # Fase 4: Ground Truth
            gt_by_doc = check_ground_truth(report, doc_ids)
            
            # Fase 5: Ejecutar pipeline
            pipeline_results = run_pipeline(report, input_data, skip_llm=args.skip_llm)
            
            if pipeline_results['final_output']:
                # Fase 6: Métricas
                calculate_metrics(report, pipeline_results, gt_by_doc)
                
                # Fase 7: Coherencia
                verify_coherence(report, pipeline_results)
    
    # Guardar informe
    report.save(Path(args.output))
    
    # Resumen final
    result = report.to_dict()
    passed = result['metadata']['passed']
    failed = result['metadata']['failed']
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📋 RESUMEN DE AUDITORÍA")
    logger.info("=" * 60)
    logger.info(f"   Total checks: {result['metadata']['total_checks']}")
    logger.info(f"   ✅ Passed: {passed}")
    logger.info(f"   ❌ Failed: {failed}")
    logger.info(f"   ⚠️  Warnings: {result['metadata']['warnings']}")
    logger.info(f"   🚫 Errors: {result['metadata']['errors']}")
    
    if result['metadata']['errors'] > 0:
        logger.info("\n🚫 ERRORES ENCONTRADOS:")
        for err in report.errors:
            logger.info(f"   - [{err['stage']}] {err['error']}")
    
    if failed > 0:
        logger.info("\n❌ PIPELINE TIENE PROBLEMAS - revisar errores arriba")
        return 1
    else:
        logger.info("\n✅ PIPELINE OK - todas las verificaciones pasaron")
        return 0


if __name__ == "__main__":
    exit(main())
