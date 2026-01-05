#!/usr/bin/env python3
"""
run_full_pipeline.py - Orquestador principal del pipeline de anonimización
===========================================================================

Este script ejecuta el pipeline completo de filtrado y anonimización:

1. Carga datos brutos y los convierte a formato JSON estándar
2. Ejecuta MEDDOCAN (opcional) para obtener entidades iniciales
3. Ejecuta SETFIT (Clasificación binaria) sobre los candidatos
   - PII: Se acepta
   - RUIDO: Se descarta (o pasa a LLM para rescate final)
4. Ejecuta LLM JUDGE (Opcional/Rescate final)
5. Guarda el resultado final en JSON

USO:
    python run_full_pipeline.py --input entidades.json --output resultados.json
    python run_full_pipeline.py --input entidades.json --skip-llm
    python run_full_pipeline.py --help

FLUJO OPTIMIZADO:
    Input -> [SetFit] --(PII)--> PII
                |               
            (Ruido)
                ↓
         [LLM Judge] --(Keep)--> PII
                |
            (Filter)
                ↓
            Discard

TRAZABILIDAD:
    Cada entidad lleva:
    - classification_source: 'setfit', 'llm_rescue'
    - classification: 'PII' (conservar) o 'RUIDO' (descartar)
"""

import argparse
import logging
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Añadir el directorio al path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Imports de módulos del pipeline
from io_json import load_entities, save_pipeline_results, normalize_entity
from setfit_module import run_setfit_filter
from llm_judge import run_llm_judge
from utils.csv_converter import read_csv_detections, merge_continuous_entities, generate_pipeline_json
from utils.token_healing import fix_entity_boundaries, batch_fix_entity_boundaries


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DEFAULT_CONFIG = {
    # SetFit - CONFIGURACIÓN SIMPLIFICADA
    "setfit": {
        "model_path": str(SCRIPT_DIR.parent.parent / "models" / "gatekeeper_setfit"),
        "confidence_threshold": 0.75,  # Subido para mejorar precisión
        "enable_pii_detector": False,  # SetFit clasifica todo
        "enable_low_confidence_filter": True,  # Filtrar baja confianza
    },
    # LLM
    "llm": {
        "model": "qwen2.5:7b",
        "rules_path": str(SCRIPT_DIR.parent.parent / "guias-anotacion.json"),
        "template_name": "default",
        "timeout": 120,
        "max_retries": 2,
    },
    # General
    "pipeline": {
        "skip_setfit": False,
        "skip_llm": False,
        "save_intermediate": False,
        "output_dir": str(SCRIPT_DIR.parent.parent / "outputs"),
    }
}


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(verbose: bool = False):
    """Configura el sistema de logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(levelname)-8s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# ============================================================================
# HELPER FUNCTIONS - Contexto de entidades
# ============================================================================

def load_documents(base_path: Path, valid_ids: Optional[set] = None, custom_docs_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Carga todos los documentos del directorio especificado o busca en rutas por defecto.
    
    Args:
        base_path: Ruta base del proyecto
        valid_ids: Set opcional de IDs de documentos a cargar. Si es None, carga todos.
        custom_docs_dir: Ruta personalizada al directorio de documentos.
    
    Returns:
        Dict {doc_id: texto_completo}
    """
    documents = {}
    
    # Si se proporciona una ruta personalizada, usarla prioritariamente
    possible_dirs = []
    if custom_docs_dir:
        custom_path = Path(custom_docs_dir)
        if not custom_path.is_absolute():
            possible_dirs.append(base_path / custom_path)
        else:
            possible_dirs.append(custom_path)
    
    # Buscar directorios de documentos por defecto
    possible_dirs.extend([
        # base_path / "corpus" / "documents",
       # base_path / "corpus" / "ANTIGUO" / "documents", 
       # base_path / "documents",
        base_path / "corpus" / "output" / "aws2",
    ])
    
    for docs_dir in possible_dirs:
        if docs_dir.exists():
            # Si tenemos IDs validos, solo buscamos esos archivos
            if valid_ids:
                count_found = 0
                for doc_id in valid_ids:
                    # Intentar encontrar el archivo con extension .txt
                    txt_file = docs_dir / f"{doc_id}.txt"
                    if txt_file.exists():
                        try:
                            text = txt_file.read_text(encoding="utf-8")
                            documents[doc_id] = text
                            count_found += 1
                        except Exception as e:
                            logging.warning(f"Error leyendo {txt_file}: {e}")
                
                if documents:
                    logging.info(f"Cargados {len(documents)} documentos específicos de {len(valid_ids)} requeridos desde {docs_dir}")
                    break
            else:
                # Comportamiento original: cargar todo (LENTO si hay muchos archivos)
                for txt_file in docs_dir.glob("*.txt"):
                    doc_id = txt_file.stem
                    try:
                        text = txt_file.read_text(encoding="utf-8")
                        documents[doc_id] = text
                    except Exception as e:
                        logging.warning(f"Error leyendo {txt_file}: {e}")
                
                if documents:
                    logging.info(f"Cargados {len(documents)} documentos desde {docs_dir}")
                    break
    
    return documents


def extract_sentence_context(
    text: str,
    start: int,
    end: int,
    entity_text: str,
    window_chars: int = 150
) -> str:
    """
    Extrae la frase o segmento del documento donde aparece la entidad.
    
    Args:
        text: Texto completo del documento
        start: Posición de inicio de la entidad
        end: Posición de fin de la entidad  
        entity_text: Texto de la entidad
        window_chars: Caracteres de contexto si no hay frase
        
    Returns:
        Frase o segmento de contexto
    """
    if not text or start < 0 or end > len(text):
        return ""
    
    # Verificar que las posiciones son correctas
    extracted = text[start:end]
    if extracted.strip() != entity_text.strip():
        # Las posiciones no coinciden, intentar buscar
        pos = text.find(entity_text)
        if pos != -1:
            start = pos
            end = pos + len(entity_text)
        else:
            # No encontrado, usar ventana
            context_start = max(0, start - window_chars)
            context_end = min(len(text), end + window_chars)
            return text[context_start:context_end].strip()
    
    # Buscar límites de frase (., !, ?, salto de línea)
    sentence_delimiters = r'[.!?\n]'
    
    # Buscar inicio de frase (hacia atrás)
    sentence_start = start
    for i in range(start - 1, max(0, start - 500), -1):
        if re.match(sentence_delimiters, text[i]):
            sentence_start = i + 1
            break
    else:
        sentence_start = max(0, start - window_chars)
    
    # Buscar fin de frase (hacia adelante)
    sentence_end = end
    for i in range(end, min(len(text), end + 500)):
        if re.match(sentence_delimiters, text[i]):
            sentence_end = i + 1
            break
    else:
        sentence_end = min(len(text), end + window_chars)
    
    # Extraer y limpiar
    sentence = text[sentence_start:sentence_end].strip()
    
    # Limitar longitud máxima
    max_length = 500
    if len(sentence) > max_length:
        # Centrar en la entidad
        entity_pos = start - sentence_start
        half_window = max_length // 2
        new_start = max(0, entity_pos - half_window)
        new_end = min(len(sentence), entity_pos + len(entity_text) + half_window)
        sentence = sentence[new_start:new_end].strip()
    
    return sentence


def add_context_to_entities(
    entities: List[Dict[str, Any]],
    documents: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Añade contexto a cada entidad extrayendo del documento original.
    
    Args:
        entities: Lista de entidades
        documents: Dict de documentos {doc_id: texto}
        
    Returns:
        Lista de entidades con campo 'context' añadido
    """
    entities_with_context = []
    missing_docs = set()
    
    for entity in entities:
        doc_id = entity.get('doc_id', entity.get('document_id', ''))
        entity_text = entity.get('text', entity.get('entity_text', ''))
        start = entity.get('start', entity.get('posicion_inicio', -1))
        end = entity.get('end', entity.get('posicion_fin', -1))
        
        # Obtener documento
        doc_text = documents.get(doc_id)
        
        if doc_text and start >= 0 and end > start:
            # Extraer contexto
            context = extract_sentence_context(doc_text, start, end, entity_text)
            entity['context'] = context
        else:
            if not doc_text and doc_id not in missing_docs:
                missing_docs.add(doc_id)
            entity['context'] = ""
        
        entities_with_context.append(entity)
    
    if missing_docs:
        logging.warning(f"No se encontraron {len(missing_docs)} documentos para extraer contexto")
    
    return entities_with_context


def apply_token_healing_to_entities(
    entities: List[Dict[str, Any]],
    documents: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Aplica token healing a las entidades para corregir fronteras.
    
    Expande las coordenadas (start, end) de una entidad para capturar
    palabras completas en lugar de fragmentos. Solo expande si no hay
    espacio en los bordes (indicando que la palabra está cortada).
    
    Args:
        entities: Lista de entidades
        documents: Dict de documentos {doc_id: texto}
        
    Returns:
        Lista de entidades con coordenadas reparadas
    """
    healed_entities = []
    stats_healing = {
        "total_processed": 0,
        "total_fixed": 0,
        "errors": 0,
    }
    
    for entity in entities:
        doc_id = entity.get('doc_id', entity.get('document_id', ''))
        doc_text = documents.get(doc_id)
        
        if not doc_text:
            healed_entities.append(entity)
            continue
        
        try:
            stats_healing["total_processed"] += 1
            healed = fix_entity_boundaries(entity, doc_text)
            
            if healed.get("boundary_fixed", False):
                stats_healing["total_fixed"] += 1
            
            healed_entities.append(healed)
            
        except Exception as e:
            logging.warning(f"Error en token healing para entidad {entity.get('text')}: {e}")
            stats_healing["errors"] += 1
            healed_entities.append(entity)
    
    return healed_entities


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

class FullPipeline:
    """
    Orquestador del pipeline completo de anonimización.
    
    FLUJO OPTIMIZADO (SetFit -> LLM):
    1. SetFit (Modelo):
       - PII: KEEP -> Va a salida
       - Ruido: RUIDO -> Se descarta (o rescate LLM)
       
    2. LLM (Rescate final):
       - Si habilitado, revisa el RUIDO del modelo.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el pipeline.
        
        Args:
            config: Configuración personalizada (se combina con DEFAULT_CONFIG)
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            for key, value in config.items():
                if key in self.config and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        self.logger = logging.getLogger(__name__)
        
        # Estadísticas
        self.stats = {
            "total_input": 0,
            # Paso 2: SetFit
            "setfit_pii": 0,       # Modelo: Aceptado
            "setfit_ruido": 0,     # Modelo: Rechazado
            # Paso 3: LLM
            "llm_rescued": 0,      # LLM: Rescatado del ruido del modelo
            "llm_filtered": 0,     # LLM: Confirmado ruido
            # Resultado final
            "final_output": 0,
            "execution_time": 0.0,
        }
    
    def run(
        self,
        entities: List[Dict[str, Any]],
        document_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta el pipeline completo.
        
        Args:
            entities: Lista de entidades de entrada
            document_text: Texto del documento (opcional)
            
        Returns:
            Lista de entidades con decisiones finales
        """
        start_time = time.time()
        
        self.logger.info("=" * 70)
        self.logger.info("PIPELINE DE ANONIMIZACIÓN - INICIANDO (MODO OPTIMIZADO)")
        self.logger.info("=" * 70)
        
        self.stats["total_input"] = len(entities)
        self.logger.info(f"Entidades de entrada: {len(entities)}")
        
        # Normalizar entidades
        normalized = [normalize_entity(e) for e in entities]
        
        # ====================================================================
        # PASO 0: Cargar documentos y añadir contexto
        # ====================================================================
        self.logger.info("\n[PREPARACIÓN] Cargando documentos y extrayendo contexto...")
        
        # Cargar todos los documentos necesarios
        base_path = SCRIPT_DIR.parent.parent
        required_doc_ids = set()
        for e in normalized:
            doc_id = e.get('doc_id') or e.get('document_id')
            if doc_id:
                required_doc_ids.add(str(doc_id))
        
        self.logger.info(f"  -> Identificados {len(required_doc_ids)} documentos únicos requeridos")
        documents = load_documents(base_path, valid_ids=required_doc_ids, custom_docs_dir=self.config["pipeline"].get("docs_dir"))
        
        if documents:
            self.logger.info(f"  -> Documentos cargados: {len(documents)}")
            # Añadir contexto a todas las entidades
            normalized = add_context_to_entities(normalized, documents)
            
            # Contar cuántas tienen contexto
            with_context = sum(1 for e in normalized if e.get('context'))
            self.logger.info(f"  -> Entidades con contexto: {with_context}/{len(normalized)}")
        else:
            self.logger.warning("  -> No se pudieron cargar documentos, SetFit usará solo el texto de la entidad")
        
        # ====================================================================
        # PASO 0.5: Token Healing - Reparar fronteras de entidades
        # ====================================================================
        if documents:
            self.logger.info("\n[REPARACIÓN] Aplicando token healing para corregir fronteras...")
            
            normalized = apply_token_healing_to_entities(normalized, documents)
            
            # Contar cuántas fueron reparadas
            healed_count = sum(1 for e in normalized if e.get('boundary_fixed', False))
            self.logger.info(f"  -> Entidades reparadas: {healed_count}/{len(normalized)}")
        
        # ====================================================================
        # PASO 1: Dict Filters (PRE-FILTER)
        # ====================================================================
        
        pre_filtered_candidates = []
        final_kept = []  # Entidades PII
        final_ruido = []  # Entidades RUIDO (para estadísticas y comparación)
        
        # Dict filters eliminado - Todo pasa directamente a SetFit
        self.logger.info("\n[PASO 1/3] Dict Filters... ELIMINADO")
        pre_filtered_candidates = normalized
            
        # ====================================================================
        # PASO 2: SetFit - Clasificación binaria de Candidatos
        # ====================================================================
        to_llm = []
        
        if not self.config["pipeline"].get("skip_setfit", False) and pre_filtered_candidates:
            self.logger.info(f"\n[PASO 2/3] SetFit ({len(pre_filtered_candidates)} candidatos)...")
            
            try:
                setfit_results = run_setfit_filter(
                    pre_filtered_candidates,
                    document_text,
                    self.config["setfit"]
                )
                
                pii_entities = [e for e in setfit_results if e.get("classification") == "PII"]
                ruido_entities = [e for e in setfit_results if e.get("classification") == "RUIDO"]
                
                self.stats["setfit_pii"] = len(pii_entities)
                self.stats["setfit_ruido"] = len(ruido_entities)
                
                self.logger.info(f"  -> PII (Aceptados): {len(pii_entities)}")
                self.logger.info(f"  -> RUIDO (Rechazados): {len(ruido_entities)}")
                
                # PII se añade a final_kept
                final_kept.extend(pii_entities)
                
                # RUIDO se guarda por separado para estadísticas
                final_ruido.extend(ruido_entities)
                
                # Ruido pasa a LLM si queremos ser conservadores (Rescate final)
                to_llm = ruido_entities
                
            except Exception as e:
                self.logger.error(f"  Error en SetFit: {e}")
                self.logger.info("  Continuando sin SetFit - candidatos pasan a LLM...")
                to_llm = pre_filtered_candidates
        elif pre_filtered_candidates:
            self.logger.info("\n[PASO 2/3] SetFit... OMITIDO")
            # Si se salta SetFit, ¿qué hacemos con los candidatos?
            # En modo "seguro", asumimos PII.
             # Si también saltamos LLM, entonces en este flujo "listas -> fin", 
            if self.config["pipeline"].get("skip_llm", False):
                 for e in pre_filtered_candidates:
                    e["classification"] = "PII"
                    e["classification_source"] = "default_no_setfit_escalated"
                 final_kept.extend(pre_filtered_candidates)
                 to_llm = []
            else:
                 to_llm = pre_filtered_candidates
        else:
             self.logger.info("\n[PASO 2/3] SetFit... OMITIDO (Sin candidatos)")
        
        # ====================================================================
        # PASO 3: LLM Judge - Rescate final de casos dudosos
        # ====================================================================
        if not self.config["pipeline"].get("skip_llm", False) and to_llm:
            self.logger.info(f"\n[PASO 3/3] LLM Judge - Rescate final ({len(to_llm)} entidades)...")
            self.logger.info(f"  -> Modelo LLM: {self.config['llm']['model']}")
            self.logger.info(f"  -> Timeout: {self.config['llm']['timeout']}s")
            self.logger.info(f"  -> Max retries: {self.config['llm']['max_retries']}")
            
            # Mostrar preview de las entidades que se van a evaluar
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("  -> Entidades a evaluar:")
                for i, e in enumerate(to_llm[:5], 1):  # Primeras 5
                    self.logger.debug(f"     {i}. {e.get('text', 'N/A')[:50]}")
                if len(to_llm) > 5:
                    self.logger.debug(f"     ... y {len(to_llm) - 5} más")
            
            self.logger.info("  -> Iniciando evaluación con LLM...")
            
            # Debug: mostrar ejemplo de lo que se envía al LLM
            if to_llm and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("  -> Enviando al LLM:")
                for i, e in enumerate(to_llm[:2], 1):
                    self.logger.debug(f"     {i}. '{e.get('text', 'N/A')[:40]}' | label={e.get('label', 'N/A')}")
            
            try:
                llm_results = run_llm_judge(
                    to_llm,
                    document_text,
                    self.config["llm"]
                )
                self.logger.info("  -> Evaluación LLM completada")
                
                # Debug: mostrar ejemplo de lo que respondió el LLM
                if llm_results and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("  -> Respuesta LLM:")
                    for i, e in enumerate(llm_results[:2], 1):
                        decision = e.get('decision', 'N/A')
                        reason = e.get('llm_reasoning', 'N/A')[:60]
                        self.logger.debug(f"     {i}. '{e.get('text', 'N/A')[:40]}' → {decision} ({reason})")
                
                llm_rescued = [e for e in llm_results if e.get("decision") == "KEEP"]
                llm_filtered = [e for e in llm_results if e.get("decision") == "FILTER"]
                
                # Añadir trazabilidad a los rescatados
                for e in llm_rescued:
                    e["classification"] = "PII"
                    e["classification_source"] = "llm_rescue"
                
                # Añadir trazabilidad a los filtrados
                for e in llm_filtered:
                    e["classification"] = "RUIDO"
                    e["classification_source"] = "llm_filtered"
                
                self.stats["llm_rescued"] = len(llm_rescued)
                self.stats["llm_filtered"] = len(llm_filtered)
                
                self.logger.info(f"  -> RESCATADOS (LLM): {len(llm_rescued)}")
                self.logger.info(f"  -> DESCARTADOS (LLM): {len(llm_filtered)}")
                
                # Debug: mostrar ejemplos de rescatados
                if llm_rescued and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("  -> Ejemplos rescatados por LLM:")
                    for i, e in enumerate(llm_rescued[:3], 1):
                        self.logger.debug(f"     {i}. {e.get('text', 'N/A')[:50]} (razón: {e.get('llm_reasoning', 'N/A')[:80]})")
                
                # Añadir rescatados a PII
                final_kept.extend(llm_rescued)
                
                # Añadir filtrados a RUIDO
                final_ruido.extend(llm_filtered)
                
            except Exception as e:
                self.logger.error(f"  Error en LLM: {e}")
                self.logger.info("  Manteniendo entidades ambiguas por seguridad...")
                for entity in to_llm:
                    entity["classification"] = "PII"
                    entity["classification_source"] = "default_on_error"
                final_kept.extend(to_llm)
        else:
            if to_llm:
                self.logger.info(f"\n[PASO 3/3] LLM Judge... OMITIDO ({len(to_llm)} entidades DESCARTADAS)")
                # Marcar como RUIDO las que quedaron sin evaluar
                for e in to_llm:
                    e["classification"] = "RUIDO"
                    e["classification_source"] = "setfit_filtered"
                final_ruido.extend(to_llm)
            else:
                self.logger.info("\n[PASO 3/3] LLM Judge... OMITIDO")
        
        # ====================================================================
        # RESULTADO FINAL
        # ====================================================================
        # Combinar PII y RUIDO para tener todas las entidades con su clasificación
        all_results = final_kept + final_ruido
        
        self.stats["final_output"] = len(final_kept)
        self.stats["final_ruido"] = len(final_ruido)
        self.stats["execution_time"] = time.time() - start_time
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PIPELINE COMPLETADO")
        self.logger.info("=" * 70)
        self._print_stats()
        
        return all_results
    
    def _print_stats(self):
        """Imprime estadísticas del pipeline con el nuevo flujo."""
        s = self.stats
        
        self.logger.info(f"\n[ESTADÍSTICAS DEL PIPELINE]")
        self.logger.info(f"  Entrada total:  {s['total_input']} entidades")
        self.logger.info(f"  Salida PII:     {s['final_output']} entidades")
        self.logger.info(f"  Salida RUIDO:   {s.get('final_ruido', 0)} entidades")
        
        if s['total_input'] > 0:
            reduction = (s.get('final_ruido', 0) / s['total_input']) * 100
            self.logger.info(f"  Ruido filtrado: {reduction:.1f}%")
        
        self.logger.info(f"\n[FLUJO POR ETAPA]")
        self.logger.info(f"  1. SetFit:       PII={s['setfit_pii']} (Keep)   | Ruido={s['setfit_ruido']} (Drop/LLM)")
        self.logger.info(f"  2. LLM:          Rescatados={s['llm_rescued']} | Ruido={s['llm_filtered']}")
        
        self.logger.info(f"\n[TRAZABILIDAD SALIDA]")
        setfit_pii = s['setfit_pii']
        llm_rescued = s['llm_rescued']
        self.logger.info(f"  SetFit PII:               {setfit_pii}")
        self.logger.info(f"  LLM Rescued:              {llm_rescued}")
        self.logger.info(f"  TOTAL PII:                {setfit_pii + llm_rescued}")

        self.logger.info(f"\n[TIEMPO] {s['execution_time']:.2f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Devuelve las estadísticas del pipeline."""
        return self.stats.copy()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline completo de anonimización de entidades",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Ejecutar pipeline completo
  python run_full_pipeline.py --input entidades.json --output resultados.json
  
  # Sin LLM (solo SetFit + listas)
  python run_full_pipeline.py --input entidades.json --output resultados.json --skip-llm
  
  # Solo SetFit
  python run_full_pipeline.py --input entidades.json --output resultados.json --skip-dict --skip-llm
  
  # Con verbose
  python run_full_pipeline.py --input entidades.json --output resultados.json -v
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=False,  # Ahora es opcional, busca CSV por defecto
        help='Archivo JSON de entrada con entidades (o CSV para conversión automática)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Archivo JSON de salida con resultados'
    )
    
    parser.add_argument(
        '--config', '-c',
        default=None,
        help='Archivo JSON de configuración personalizada'
    )
    
    parser.add_argument(
        '--skip-setfit',
        action='store_true',
        help='Omitir paso de SetFit'
    )
    
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Omitir paso de LLM'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Guardar resultados intermedios de cada etapa'
    )
    
    parser.add_argument(
        '--docs-dir',
        default=None,
        help='Directorio personalizado para buscar los documentos .txt'
    )
    
    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()
    
    # Configurar logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validar entrada o buscar por defecto
    input_path = args.input
    
    # Si no se especifica input, buscar el CSV por defecto
    if not input_path:
        default_csv = SCRIPT_DIR.parent / "pipeline" / "step6_validation_results" / "detecciones_detalladas.csv"
        if default_csv.exists():
            logger.info(f"Input no especificado. Detectado CSV por defecto: {default_csv}")
            input_path = str(default_csv)
        else:
            logger.error(f"No se especificó input y no se encontró el CSV por defecto: {default_csv}")
            return 1

    # Si el input es un CSV, convertirlo a JSON
    if input_path.lower().endswith('.csv'):
        logger.info(f"Input es CSV. Iniciando conversión automática...")
        try:
            # 1. Leer CSV
            detections = read_csv_detections(Path(input_path))
            logger.info(f"  -> Leídas {len(detections)} detecciones")
            
            # 2. Fusionar entidades
            unified = merge_continuous_entities(detections)
            logger.info(f"  -> Fusionadas a {len(unified)} entidades unificadas")
            
            # 3. Generar JSON temporal
            temp_json_path = Path(args.output).parent / "temp_pipeline_input.json"
            if not temp_json_path.parent.exists():
                temp_json_path.parent.mkdir(parents=True)
                
            generate_pipeline_json(unified, temp_json_path, Path(input_path))
            logger.info(f"  -> JSON temporal generado: {temp_json_path}")
            
            # Actualizar input para el resto del pipeline
            input_path = str(temp_json_path)
            
        except Exception as e:
            logger.error(f"Error durante la conversión de CSV: {e}")
            return 1
            
    if not Path(input_path).exists():
        logger.error(f"Archivo de entrada no encontrado: {input_path}")
        return 1
    
    # Cargar configuración
    config = DEFAULT_CONFIG.copy()
    
    if args.config and Path(args.config).exists():
        import json
        # Cargar configuración tolerando BOM en Windows
        try:
            with open(args.config, encoding='utf-8') as f:
                custom_config = json.load(f)
        except json.JSONDecodeError:
            with open(args.config, encoding='utf-8-sig') as f:
                custom_config = json.load(f)
        for key, value in custom_config.items():
            if key in config and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    
    # Aplicar flags de CLI
    config["pipeline"]["skip_setfit"] = args.skip_setfit
    config["pipeline"]["skip_llm"] = args.skip_llm
    config["pipeline"]["save_intermediate"] = args.save_intermediate
    config["pipeline"]["docs_dir"] = args.docs_dir
    
    try:
        # Cargar entidades
        logger.info(f"Cargando entidades desde {input_path}")
        entities = load_entities(input_path)
        logger.info(f"Cargadas {len(entities)} entidades")
        
        # Ejecutar pipeline
        pipeline = FullPipeline(config)
        results = pipeline.run(entities)
        
        # Guardar resultados
        logger.info(f"\nGuardando resultados en {args.output}")
        
        # Contar PII y RUIDO en los resultados
        pii_count = sum(1 for e in results if e.get('classification') == 'PII')
        ruido_count = sum(1 for e in results if e.get('classification') == 'RUIDO')
        
        save_pipeline_results(
            results,
            args.output,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "input_file": input_path,
                "total_entities": len(results),
                "pii_entities": pii_count,
                "ruido_entities": ruido_count,
                "pipeline_version": "2.0-optimized",
            },
            stats=pipeline.get_stats()
        )
        
        logger.info("Pipeline completado exitosamente")
        return 0
        
    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
