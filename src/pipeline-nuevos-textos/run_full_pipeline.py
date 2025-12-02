#!/usr/bin/env python3
"""
run_full_pipeline.py - Orquestador principal del pipeline de anonimización
===========================================================================

Este script ejecuta el pipeline completo de filtrado y anonimización:

1. Carga datos brutos y los convierte a formato JSON estándar
2. Ejecuta MEDDOCAN (opcional) para obtener entidades iniciales
3. Pasa las entidades por SetFit para filtrar ruido
4. Aplica filtros de diccionario (blacklist/whitelist)
5. Valida casos ambiguos con el LLM
6. Guarda el resultado final en JSON

USO:
    python run_full_pipeline.py --input entidades.json --output resultados.json
    python run_full_pipeline.py --input entidades.json --skip-llm
    python run_full_pipeline.py --help

FLUJO DEL PIPELINE:
    ┌─────────────┐
    │  Input JSON │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   SetFit    │ → Filtra ruido obvio
    │  Gatekeeper │ → Detecta PII obvio
    └──────┬──────┘
           │ (entities con is_pii=True o decision=ESCALATE)
    ┌──────▼──────┐
    │ Dict Filter │ → Whitelist: KEEP
    │ (Listas)    │ → Blacklist: FILTER
    └──────┬──────┘
           │ (entities con decision=ESCALATE)
    ┌──────▼──────┐
    │  LLM Judge  │ → Validación final
    │ (Ollama)    │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Output JSON │
    └─────────────┘
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
from dict_filters import apply_dict_filters
from llm_judge import run_llm_judge


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DEFAULT_CONFIG = {
    # SetFit
    "setfit": {
        "model_path": str(SCRIPT_DIR.parent.parent / "models" / "gatekeeper_setfit"),
        "confidence_threshold": 0.75,
        "enable_noise_filter": False,
        "enable_pii_detector": False,
        "enable_fragment_filter": False,
        "enable_low_confidence_filter": False,
    },
    # Dict Filters
    "dict_filters": {
        "json_whitelist_paths": [
            str(SCRIPT_DIR.parent.parent / "data" / "hospitales.json"),
            str(SCRIPT_DIR.parent.parent / "data" / "lugares.json"),
        ],
        "json_blacklist_paths": [
            str(SCRIPT_DIR.parent.parent / "data" / "medicamentos.json"),
            str(SCRIPT_DIR.parent.parent / "data" / "patologias.json"),
        ],
        "csv_base_path": str(SCRIPT_DIR.parent.parent / "LISTAS"),
        "cie10_path": str(SCRIPT_DIR.parent.parent / "LISTAS" / "cie10.xls"),
        "min_length_per_label": {
            "NUMERO_IDENTIF": 1,
            "FECHAS": 4,
            "NUMERO_TELEFONO": 4,
            "NUMERO_FAX": 4,
        },
        "ignore_single_char": True,  # Mantener, filtra ruido real de 1 carácter
        "ignore_numeric_only": False,
    },
    # LLM
    "llm": {
        "model": "gemma3:270m",
        "rules_path": str(SCRIPT_DIR.parent.parent / "guias-anotacion.json"),
        "template_name": "default",
        "timeout": 120,
        "max_retries": 2,
    },
    # General
    "pipeline": {
        "skip_setfit": False,
        "skip_dict_filters": False,
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

def load_documents(base_path: Path) -> Dict[str, str]:
    """
    Carga todos los documentos del directorio corpus/documents.
    
    Returns:
        Dict {doc_id: texto_completo}
    """
    documents = {}
    
    # Buscar directorios de documentos
    possible_dirs = [
        base_path / "corpus" / "documents",
        base_path / "corpus" / "ANTIGUO" / "documents", 
        base_path / "documents",
        base_path / "corpus" / "output",
    ]
    
    for docs_dir in possible_dirs:
        if docs_dir.exists():
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


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

class FullPipeline:
    """
    Orquestador del pipeline completo de anonimización.
    
    Ejecuta los pasos en orden:
    1. SetFit → Clasificación inicial PII/Ruido
    2. Dict Filters → Whitelist/Blacklist
    3. LLM → Validación de casos ambiguos
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
            "setfit_kept": 0,
            "setfit_filtered": 0,
            "dict_kept": 0,
            "dict_filtered": 0,
            "dict_escalated": 0,
            "llm_kept": 0,
            "llm_filtered": 0,
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
        self.logger.info("PIPELINE DE ANONIMIZACIÓN - INICIANDO")
        self.logger.info("=" * 70)
        
        self.stats["total_input"] = len(entities)
        self.logger.info(f"Entidades de entrada: {len(entities)}")
        
        # Normalizar entidades
        normalized = [normalize_entity(e) for e in entities]
        
        # ====================================================================
        # PASO 0: Cargar documentos y añadir contexto
        # ====================================================================
        self.logger.info("\n[PREPARACIÓN] Cargando documentos y extrayendo contexto...")
        
        # Cargar todos los documentos
        base_path = SCRIPT_DIR.parent.parent
        documents = load_documents(base_path)
        
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
        # PASO 1: SetFit
        # ====================================================================
        if not self.config["pipeline"].get("skip_setfit", False):
            self.logger.info("\n[PASO 1/3] SetFit Gatekeeper...")
            
            try:
                setfit_results = run_setfit_filter(
                    normalized,
                    document_text,
                    self.config["setfit"]
                )
                
                # Separar resultados
                kept = [e for e in setfit_results if e.get("decision") == "KEEP"]
                filtered = [e for e in setfit_results if e.get("decision") == "FILTER"]
                
                self.stats["setfit_kept"] = len(kept)
                self.stats["setfit_filtered"] = len(filtered)
                
                self.logger.info(f"  -> KEEP (PII detectado): {len(kept)}")
                self.logger.info(f"  -> FILTER (Ruido, pasa a listas): {len(filtered)}")
                
                # KEEP directo
                final_kept = kept
                # FILTER pasa a dict_filters
                to_dict_filters = filtered
                
            except Exception as e:
                self.logger.error(f"  Error en SetFit: {e}")
                self.logger.info("  Continuando sin SetFit...")
                final_kept = normalized
                to_dict_filters = []
        else:
            self.logger.info("\n[PASO 1/3] SetFit Gatekeeper... OMITIDO")
            final_kept = normalized
            to_dict_filters = []
        
        # ====================================================================
        # PASO 2: Dict Filters (para entidades que SetFit filtró como ruido)
        # ====================================================================
        if not self.config["pipeline"].get("skip_dict_filters", False) and to_dict_filters:
            self.logger.info(f"\n[PASO 2/3] Filtros de Diccionario ({len(to_dict_filters)} entidades de ruido SetFit)...")
            
            try:
                dict_results = apply_dict_filters(
                    to_dict_filters,
                    document_text,
                    self.config["dict_filters"]
                )
                
                # Separar resultados
                dict_kept = [e for e in dict_results if e.get("decision") == "KEEP"]
                dict_filtered = [e for e in dict_results if e.get("decision") == "FILTER"]
                dict_escalated = [e for e in dict_results if e.get("decision") == "ESCALATE"]
                
                self.stats["dict_kept"] = len(dict_kept)
                self.stats["dict_filtered"] = len(dict_filtered)
                self.stats["dict_escalated"] = len(dict_escalated)
                
                self.logger.info(f"  -> Whitelist (KEEP): {len(dict_kept)}")
                self.logger.info(f"  -> Blacklist (FILTER): {len(dict_filtered)}")
                self.logger.info(f"  -> Sin match (ESCALATE a LLM): {len(dict_escalated)}")
                
                # Añadir los de whitelist a final_kept
                final_kept.extend(dict_kept)
                # Los escalados van al LLM
                to_llm = dict_escalated
                
            except Exception as e:
                self.logger.error(f"  Error en Dict Filters: {e}")
                self.logger.info("  Continuando sin filtros...")
                # Si falla dict_filters, enviar todo al LLM
                to_llm = to_dict_filters
        elif to_dict_filters:
            self.logger.info(f"\n[PASO 2/3] Filtros de Diccionario... OMITIDO ({len(to_dict_filters)} van al LLM)")
            to_llm = to_dict_filters
        else:
            self.logger.info("\n[PASO 2/3] Filtros de Diccionario... OMITIDO (sin entidades)")
            to_llm = []
        
        # ====================================================================
        # PASO 3: LLM Judge (para entidades escaladas desde dict_filters)
        # ====================================================================
        if not self.config["pipeline"].get("skip_llm", False) and to_llm:
            self.logger.info(f"\n[PASO 3/3] LLM Judge ({len(to_llm)} entidades escaladas)...")
            
            try:
                llm_results = run_llm_judge(
                    to_llm,
                    document_text,
                    self.config["llm"]
                )
                
                # Separar resultados
                llm_kept = [e for e in llm_results if e.get("decision") == "KEEP"]
                llm_filtered = [e for e in llm_results if e.get("decision") == "FILTER"]
                
                self.stats["llm_kept"] = len(llm_kept)
                self.stats["llm_filtered"] = len(llm_filtered)
                
                self.logger.info(f"  -> KEEP (LLM dice que SÍ es PII): {len(llm_kept)}")
                self.logger.info(f"  -> FILTER (LLM confirma ruido): {len(llm_filtered)}")
                
                # Añadir a final
                final_kept.extend(llm_kept)
                
            except Exception as e:
                self.logger.error(f"  Error en LLM: {e}")
                self.logger.info("  Manteniendo entidades ambiguas por seguridad...")
                # En caso de error, mantener las entidades
                for e in to_llm:
                    e["decision"] = "KEEP"
                    e["decision_source"] = "default_on_error"
                final_kept.extend(to_llm)
        else:
            if to_llm:
                self.logger.info(f"\n[PASO 3/3] LLM Judge... OMITIDO ({len(to_llm)} entidades descartadas)")
                # Si skip LLM, las entidades escaladas se descartan
            else:
                self.logger.info("\n[PASO 3/3] LLM Judge... OMITIDO (sin entidades)")
        
        # ====================================================================
        # RESULTADO FINAL
        # ====================================================================
        self.stats["final_output"] = len(final_kept)
        self.stats["execution_time"] = time.time() - start_time
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PIPELINE COMPLETADO")
        self.logger.info("=" * 70)
        self._print_stats()
        
        return final_kept
    
    def _print_stats(self):
        """Imprime estadísticas del pipeline."""
        s = self.stats
        
        self.logger.info(f"\n[ESTADISTICAS]")
        self.logger.info(f"  Entrada:  {s['total_input']} entidades")
        self.logger.info(f"  Salida:   {s['final_output']} entidades")
        
        if s['total_input'] > 0:
            reduction = (1 - s['final_output'] / s['total_input']) * 100
            self.logger.info(f"  Reducción: {reduction:.1f}%")
        
        self.logger.info(f"\n[POR ETAPA]")
        self.logger.info(f"  SetFit:  {s['setfit_kept']} kept, {s['setfit_filtered']} escalados a LLM")
        self.logger.info(f"  LLM:     {s['llm_kept']} kept, {s['llm_filtered']} filtered")

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
        required=True,
        help='Archivo JSON de entrada con entidades'
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
        '--skip-dict',
        action='store_true',
        help='Omitir paso de filtros de diccionario'
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
    
    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()
    
    # Configurar logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validar entrada
    if not Path(args.input).exists():
        logger.error(f"Archivo de entrada no encontrado: {args.input}")
        return 1
    
    # Cargar configuración
    config = DEFAULT_CONFIG.copy()
    
    if args.config and Path(args.config).exists():
        import json
        with open(args.config) as f:
            custom_config = json.load(f)
        for key, value in custom_config.items():
            if key in config and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    
    # Aplicar flags de CLI
    config["pipeline"]["skip_setfit"] = args.skip_setfit
    config["pipeline"]["skip_dict_filters"] = args.skip_dict
    config["pipeline"]["skip_llm"] = args.skip_llm
    config["pipeline"]["save_intermediate"] = args.save_intermediate
    
    try:
        # Cargar entidades
        logger.info(f"Cargando entidades desde {args.input}")
        entities = load_entities(args.input)
        logger.info(f"Cargadas {len(entities)} entidades")
        
        # Ejecutar pipeline
        pipeline = FullPipeline(config)
        results = pipeline.run(entities)
        
        # Guardar resultados
        logger.info(f"\nGuardando resultados en {args.output}")
        save_pipeline_results(
            results,
            args.output,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "input_file": args.input,
                "total_entities": len(results),
                "pipeline_version": "2.0",
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
