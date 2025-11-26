#!/usr/bin/env python3
"""
EntityFastFilter: Filtro determinista SIMPLE para entidades NER.

Este módulo implementa un sistema de filtrado basado EXCLUSIVAMENTE en:
- WhiteList: Coincidencia EXACTA → FORCE_ANONYMIZE
- BlackList: Coincidencia EXACTA (case-insensitive) → FORCE_IGNORE
- Default: → ESCALATE_TO_LLM

SIN:
- Regex
- Heurísticas
- Normalización de texto
- Limpieza de dosis
- Filtros por longitud
- Listas de palabras comunes
- Coincidencias parciales/substring

Fuentes de datos:
    - Archivos JSON con listas de términos.
    - Archivos CSV/Excel en carpeta LISTAS/ (via CsvListManager).
    - Archivo CIE10 (.xls/.xlsx) con códigos de diagnósticos médicos.

Author: Pipeline Anonimización Clínica
Version: 2.0.0 (Simplificado - Sin Overfitting)
"""

from __future__ import annotations

import json
import logging
import unittest
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import pandas as pd

# Import condicional para evitar dependencia circular
if TYPE_CHECKING:
    from csv_list_manager import CsvListManager

# Configurar logging
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERACIÓN DE DECISIONES
# ============================================================================

class EnumDecision(Enum):
    """
    Decisiones posibles del filtro rápido de entidades.

    Attributes:
        FORCE_ANONYMIZE: Anonimizar directamente sin LLM (whitelist match exacto).
        FORCE_IGNORE: Ignorar directamente sin LLM (blacklist match exacto).
        ESCALATE_TO_LLM: Requiere validación semántica del LLM.
    """
    FORCE_ANONYMIZE = auto()
    FORCE_IGNORE = auto()
    ESCALATE_TO_LLM = auto()


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class EntityFastFilter:
    """
    Filtro determinista SIMPLE para entidades NER.

    Utiliza SOLO coincidencias EXACTAS en whitelist y blacklist.
    NO usa regex, heurísticas, ni normalización.

    Attributes:
        whitelist_set: Set de términos de whitelist (case-sensitive).
        blacklist_set: Set de términos de blacklist (lowercase para comparación).

    Flujo de decisión:
        1. Si texto EXACTO está en whitelist → FORCE_ANONYMIZE
        2. Si texto.lower() EXACTO está en blacklist → FORCE_IGNORE
        3. En cualquier otro caso → ESCALATE_TO_LLM

    Example:
        >>> filter = EntityFastFilter(
        ...     whitelist_paths=["hospitales.json", "lugares.json"],
        ...     blacklist_paths=["medicamentos.json", "patologias.json"]
        ... )
        >>> decision = filter.evaluate_candidate("Hospital Clínic", "HOSPITAL")
        >>> decision == EnumDecision.FORCE_ANONYMIZE  # Solo si "Hospital Clínic" está EXACTO en whitelist
    """

    # ========================================================================
    # COLUMNAS CONOCIDAS PARA DETECCIÓN AUTOMÁTICA EN ARCHIVOS EXCEL
    # ========================================================================
    
    CIE10_DESCRIPTION_COLUMNS = [
        "DESCRIPCION CODIGOS DE CUATRO CARACTERES",
        "DESRIPCION CATEGORIAS DE TRES CARACTERES",
        "DESCRIPCION",
        "DESCRIPCIÓN",
        "Description",
        "Name",
        "Term",
        "Término",
        "Nombre",
    ]
    
    CIE10_CODE_COLUMNS = [
        "COD_3",
        "COD_4",
        "CODIGO",
        "CÓDIGO",
        "Code",
    ]

    def __init__(
        self,
        whitelist_paths: Optional[List[str]] = None,
        blacklist_paths: Optional[List[str]] = None,
        whitelist_data: Optional[List[str]] = None,
        blacklist_data: Optional[List[str]] = None,
        csv_lists_path: Optional[str] = None,
        cie10_path: Optional[str] = None
    ) -> None:
        """
        Inicializa el filtro con listas blancas y negras.

        Args:
            whitelist_paths: Rutas a JSONs de whitelist (hospitales, lugares).
                             Coincidencia EXACTA case-sensitive.
            blacklist_paths: Rutas a JSONs de blacklist (medicamentos, patologías).
                             Coincidencia EXACTA case-insensitive.
            whitelist_data: Datos directos de whitelist.
            blacklist_data: Datos directos de blacklist.
            csv_lists_path: Ruta a carpeta con archivos CSV/Excel.
            cie10_path: Ruta al archivo Excel CIE10 (.xls o .xlsx).

        Raises:
            FileNotFoundError: Si algún archivo JSON o CIE10 no existe.
            json.JSONDecodeError: Si algún JSON es inválido.
        """
        # Sets para almacenar términos
        # WhiteList: Case Sensitive - se guarda tal cual
        self.whitelist_set: Set[str] = set()
        # BlackList: Case Insensitive - se guarda en lowercase
        self.blacklist_set: Set[str] = set()

        # Referencia al CsvListManager (si se usa)
        self._csv_manager: Optional[CsvListManager] = None
        
        # Términos CIE10 cargados
        self._cie10_terms: Set[str] = set()
        self._cie10_path: Optional[str] = cie10_path
        self._cie10_stats: Dict[str, Any] = {
            "loaded": False,
            "total_terms": 0,
            "duplicates_removed": 0,
            "empty_removed": 0,
            "columns_detected": []
        }

        # Cargar whitelist desde archivos JSON
        if whitelist_paths:
            for path in whitelist_paths:
                terms = self._load_json_terms(path, lowercase=False)
                self.whitelist_set.update(terms)

        # Añadir datos directos de whitelist
        if whitelist_data:
            for term in whitelist_data:
                if term and term.strip():
                    self.whitelist_set.add(term.strip())

        # Cargar blacklist desde archivos JSON
        if blacklist_paths:
            for path in blacklist_paths:
                terms = self._load_json_terms(path, lowercase=True)
                self.blacklist_set.update(terms)

        # Añadir datos directos de blacklist
        if blacklist_data:
            for term in blacklist_data:
                if term and term.strip():
                    self.blacklist_set.add(term.strip().lower())

        # Cargar listas desde CSV/Excel si se especifica la ruta
        if csv_lists_path:
            self._load_csv_lists(csv_lists_path)

        # Cargar CIE10 si se especifica la ruta
        if cie10_path:
            cie10_terms = self._load_cie10_excel(cie10_path)
            self._cie10_terms = cie10_terms
            # CIE10 va a blacklist (términos médicos que no son datos personales)
            self.blacklist_set.update(cie10_terms)
            logger.info(f"CIE10 loaded: {len(cie10_terms)} terms added to blacklist")

        # Guardar contadores para debugging
        self._whitelist_count = len(self.whitelist_set)
        self._blacklist_count = len(self.blacklist_set)
        
        logger.info(f"EntityFastFilter initialized: whitelist={self._whitelist_count}, blacklist={self._blacklist_count}")

    def _load_json_terms(self, path: str, lowercase: bool = False) -> Set[str]:
        """
        Carga términos desde un archivo JSON.

        Args:
            path: Ruta al archivo JSON.
            lowercase: Si True, convierte todos los términos a minúsculas.

        Returns:
            Set de términos cargados.
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.warning(f"Archivo JSON no encontrado: {path}")
            return set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando JSON {path}: {e}")
            return set()

        terms: Set[str] = set()
        self._extract_strings_recursive(data, terms)

        if lowercase:
            terms = {t.lower() for t in terms}

        logger.debug(f"Loaded {len(terms)} terms from {path}")
        return terms

    def _extract_strings_recursive(self, obj: Any, accumulator: Set[str]) -> None:
        """
        Extrae strings de una estructura JSON anidada.

        Args:
            obj: Objeto JSON (puede ser dict, list, str, etc.)
            accumulator: Set donde acumular los strings encontrados.
        """
        if isinstance(obj, str):
            cleaned = obj.strip()
            if cleaned:
                accumulator.add(cleaned)

        elif isinstance(obj, dict):
            for value in obj.values():
                self._extract_strings_recursive(value, accumulator)

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._extract_strings_recursive(item, accumulator)

    def _load_csv_lists(self, csv_path: str) -> None:
        """
        Carga listas desde archivos CSV/Excel usando CsvListManager.

        Args:
            csv_path: Ruta a la carpeta con archivos CSV/Excel.
        """
        try:
            from csv_list_manager import CsvListManager

            self._csv_manager = CsvListManager(
                base_path=csv_path,
                lowercase_whitelist=False,
                lowercase_blacklist=True
            )
            self._csv_manager.load()

            # Añadir términos de CSV a los sets
            self.whitelist_set.update(self._csv_manager.whitelist_global)
            self.blacklist_set.update(self._csv_manager.blacklist_global)

            logger.info(
                f"CSV lists loaded: "
                f"+{len(self._csv_manager.whitelist_global)} whitelist, "
                f"+{len(self._csv_manager.blacklist_global)} blacklist"
            )

        except ImportError:
            logger.warning(
                "CsvListManager no disponible. "
                "Instala las dependencias: pip install pandas openpyxl xlrd"
            )
        except FileNotFoundError as e:
            logger.warning(f"Carpeta CSV no encontrada: {e}")
        except Exception as e:
            logger.error(f"Error cargando listas CSV: {e}")

    def _load_cie10_excel(self, cie10_path: str) -> Set[str]:
        """
        Carga términos médicos desde un archivo Excel CIE10.

        Args:
            cie10_path: Ruta al archivo Excel (.xls o .xlsx).

        Returns:
            Set de términos médicos (lowercase para comparación case-insensitive).
        """
        file_path = Path(cie10_path)
        
        if not file_path.exists():
            error_msg = f"Archivo CIE10 no encontrado: {cie10_path}"
            logger.warning(error_msg)
            return set()

        try:
            # Determinar engine según extensión
            suffix = file_path.suffix.lower()
            if suffix == '.xls':
                engine = 'xlrd'
            else:
                engine = 'openpyxl'

            df = pd.read_excel(file_path, engine=engine)
            logger.info(f"CIE10 Excel loaded: {len(df)} rows, columns: {list(df.columns)}")

        except Exception as e:
            logger.error(f"Error leyendo Excel CIE10: {e}")
            return set()

        # Detectar columnas de descripción
        description_columns = []
        for col in df.columns:
            col_str = str(col).strip()
            # Buscar columnas de descripción conocidas
            for desc_col in self.CIE10_DESCRIPTION_COLUMNS:
                if desc_col.lower() in col_str.lower():
                    description_columns.append(col)
                    break

        if not description_columns:
            # Fallback: usar todas las columnas que no sean de código
            for col in df.columns:
                col_str = str(col).strip()
                is_code_col = any(
                    code_col.lower() in col_str.lower()
                    for code_col in self.CIE10_CODE_COLUMNS
                )
                if not is_code_col and df[col].dtype == 'object':
                    description_columns.append(col)

        self._cie10_stats["columns_detected"] = [str(c) for c in description_columns]
        logger.info(f"CIE10 description columns detected: {description_columns}")

        # Extraer términos
        terms: Set[str] = set()
        empty_count = 0
        
        for col in description_columns:
            for value in df[col].dropna():
                value_str = str(value).strip()
                if value_str:
                    # Guardar en lowercase para comparación case-insensitive
                    terms.add(value_str.lower())
                else:
                    empty_count += 1

        self._cie10_stats["loaded"] = True
        self._cie10_stats["total_terms"] = len(terms)
        self._cie10_stats["empty_removed"] = empty_count

        logger.info(f"CIE10 terms extracted: {len(terms)}")
        return terms

    # ========================================================================
    # MÉTODOS DE COINCIDENCIA EXACTA
    # ========================================================================

    def is_in_whitelist(self, text: str) -> bool:
        """
        Verifica si el texto está EXACTAMENTE en la whitelist.

        Coincidencia EXACTA y case-sensitive.

        Args:
            text: Texto a buscar.

        Returns:
            True si el texto EXACTO está en whitelist.
        """
        if not text:
            return False
        return text.strip() in self.whitelist_set

    def is_in_blacklist(self, text: str) -> bool:
        """
        Verifica si el texto está EXACTAMENTE en la blacklist.

        Coincidencia EXACTA pero case-insensitive (se compara en lowercase).

        Args:
            text: Texto a buscar.

        Returns:
            True si el texto EXACTO (ignorando mayúsculas) está en blacklist.
        """
        if not text:
            return False
        return text.strip().lower() in self.blacklist_set

    def match_in_whitelist(self, text: str) -> Optional[str]:
        """
        Busca coincidencia EXACTA en whitelist.

        Args:
            text: Texto a buscar.

        Returns:
            El término si hay coincidencia exacta, None si no.
        """
        if not text:
            return None
        text_stripped = text.strip()
        if text_stripped in self.whitelist_set:
            return text_stripped
        return None

    def match_in_blacklist(self, text: str) -> Optional[str]:
        """
        Busca coincidencia EXACTA en blacklist (case-insensitive).

        Args:
            text: Texto a buscar.

        Returns:
            El término si hay coincidencia exacta (case-insensitive), None si no.
        """
        if not text:
            return None
        text_lower = text.strip().lower()
        if text_lower in self.blacklist_set:
            return text_lower
        return None

    # ========================================================================
    # MÉTODO PRINCIPAL DE EVALUACIÓN
    # ========================================================================

    def evaluate_candidate(
        self,
        entity_text: str,
        ner_label: str
    ) -> EnumDecision:
        """
        Evalúa una entidad candidata y decide el flujo de procesamiento.

        Flujo de decisión SIMPLE (sin heurísticas):
        1. Si texto EXACTO en whitelist → FORCE_ANONYMIZE
        2. Si texto EXACTO en blacklist (case-insensitive) → FORCE_IGNORE
        3. En cualquier otro caso → ESCALATE_TO_LLM

        NO aplica:
        - Regex
        - Normalización de texto
        - Filtros por longitud
        - Listas de palabras comunes
        - Limpieza de dosis
        - Coincidencias parciales

        Args:
            entity_text: Texto de la entidad detectada por NER.
            ner_label: Etiqueta asignada por el modelo NER (no se usa en esta versión).

        Returns:
            EnumDecision indicando el flujo a seguir.

        Example:
            >>> filter.evaluate_candidate("Hospital Clínic", "HOSPITAL")
            EnumDecision.FORCE_ANONYMIZE  # Solo si "Hospital Clínic" está EXACTO en whitelist

            >>> filter.evaluate_candidate("ibuprofeno", "MEDICATION")
            EnumDecision.FORCE_IGNORE  # Solo si "ibuprofeno" está EXACTO en blacklist

            >>> filter.evaluate_candidate("García", "PERSON")
            EnumDecision.ESCALATE_TO_LLM  # No está en ninguna lista
        """
        if not entity_text:
            return EnumDecision.ESCALATE_TO_LLM

        text = entity_text.strip()
        
        if not text:
            return EnumDecision.ESCALATE_TO_LLM

        # ═══════════════════════════════════════════════════════════════════
        # PASO 1: WhiteList - Coincidencia EXACTA (case-sensitive)
        # ═══════════════════════════════════════════════════════════════════
        if text in self.whitelist_set:
            logger.debug(f"FORCE_ANONYMIZE: Whitelist exact match: '{text}'")
            return EnumDecision.FORCE_ANONYMIZE

        # ═══════════════════════════════════════════════════════════════════
        # PASO 2: BlackList - Coincidencia EXACTA (case-insensitive)
        # ═══════════════════════════════════════════════════════════════════
        if text.lower() in self.blacklist_set:
            logger.debug(f"FORCE_IGNORE: Blacklist exact match: '{text}'")
            return EnumDecision.FORCE_IGNORE

        # ═══════════════════════════════════════════════════════════════════
        # PASO 3: Default - Escalar al LLM
        # ═══════════════════════════════════════════════════════════════════
        logger.debug(f"ESCALATE_TO_LLM: No exact match for: '{text}'")
        return EnumDecision.ESCALATE_TO_LLM

    def evaluate_candidate_detailed(
        self,
        entity_text: str,
        ner_label: str
    ) -> Dict[str, Any]:
        """
        Evalúa una entidad y devuelve información detallada del match.

        Útil para debugging y análisis.

        Args:
            entity_text: Texto de la entidad detectada por NER.
            ner_label: Etiqueta asignada por el modelo NER.

        Returns:
            Dict con:
                - decision: EnumDecision
                - whitelist_match: término que hizo match o None
                - blacklist_match: término que hizo match o None
                - reason: explicación de la decisión
        """
        if not entity_text:
            return {
                "decision": EnumDecision.ESCALATE_TO_LLM,
                "whitelist_match": None,
                "blacklist_match": None,
                "reason": "Texto vacío"
            }

        text = entity_text.strip()
        
        if not text:
            return {
                "decision": EnumDecision.ESCALATE_TO_LLM,
                "whitelist_match": None,
                "blacklist_match": None,
                "reason": "Texto vacío después de strip"
            }

        # Verificar whitelist (exacto, case-sensitive)
        whitelist_match = text if text in self.whitelist_set else None
        
        # Verificar blacklist (exacto, case-insensitive)
        text_lower = text.lower()
        blacklist_match = text_lower if text_lower in self.blacklist_set else None

        if whitelist_match:
            return {
                "decision": EnumDecision.FORCE_ANONYMIZE,
                "whitelist_match": whitelist_match,
                "blacklist_match": blacklist_match,
                "reason": f"Whitelist exact match: '{whitelist_match}'"
            }

        if blacklist_match:
            return {
                "decision": EnumDecision.FORCE_IGNORE,
                "whitelist_match": None,
                "blacklist_match": blacklist_match,
                "reason": f"Blacklist exact match: '{blacklist_match}'"
            }

        return {
            "decision": EnumDecision.ESCALATE_TO_LLM,
            "whitelist_match": None,
            "blacklist_match": None,
            "reason": "No exact match in any list, escalating to LLM"
        }

    # ========================================================================
    # MÉTODOS DE UTILIDAD
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del filtro.

        Returns:
            Dict con contadores de términos cargados.
        """
        return {
            "whitelist_terms": self._whitelist_count,
            "blacklist_terms": self._blacklist_count,
            "cie10_loaded": self._cie10_stats["loaded"],
            "cie10_terms": self._cie10_stats["total_terms"],
        }

    def debug_lists(self) -> None:
        """
        Imprime información de debugging sobre las listas cargadas.
        """
        print("\n" + "="*70)
        print("ENTITY FAST FILTER - DEBUG INFO")
        print("="*70)
        print(f"Whitelist terms: {self._whitelist_count}")
        print(f"Blacklist terms: {self._blacklist_count}")
        print(f"CIE10 loaded: {self._cie10_stats['loaded']}")
        if self._cie10_stats['loaded']:
            print(f"CIE10 terms: {self._cie10_stats['total_terms']}")
            print(f"CIE10 columns: {self._cie10_stats['columns_detected']}")
        print("="*70 + "\n")

    def __repr__(self) -> str:
        """Representación string del filtro."""
        cie10_info = f", cie10={self._cie10_stats['total_terms']}" if self._cie10_stats["loaded"] else ""
        return (
            f"EntityFastFilter("
            f"whitelist={self._whitelist_count}, "
            f"blacklist={self._blacklist_count}{cie10_info})"
        )


# ============================================================================
# TESTS UNITARIOS
# ============================================================================

class TestEntityFastFilter(unittest.TestCase):
    """Tests unitarios para EntityFastFilter."""

    def setUp(self):
        """Configura el filtro con datos de prueba."""
        self.filter = EntityFastFilter(
            whitelist_data=["Hospital Clínic", "Hospital de la Santa Creu"],
            blacklist_data=["ibuprofeno", "paracetamol", "diabetes mellitus"]
        )

    def test_whitelist_exact_match(self):
        """Test: whitelist solo con coincidencia EXACTA."""
        # Coincidencia exacta → FORCE_ANONYMIZE
        self.assertEqual(
            self.filter.evaluate_candidate("Hospital Clínic", "HOSPITAL"),
            EnumDecision.FORCE_ANONYMIZE
        )
        
        # No coincidencia exacta (substring) → ESCALATE_TO_LLM
        self.assertEqual(
            self.filter.evaluate_candidate("Ingreso en Hospital Clínic", "HOSPITAL"),
            EnumDecision.ESCALATE_TO_LLM
        )
        
        # No coincidencia exacta (case different) → ESCALATE_TO_LLM
        self.assertEqual(
            self.filter.evaluate_candidate("hospital clínic", "HOSPITAL"),
            EnumDecision.ESCALATE_TO_LLM
        )

    def test_blacklist_exact_match_case_insensitive(self):
        """Test: blacklist con coincidencia EXACTA case-insensitive."""
        # Coincidencia exacta (mismo case) → FORCE_IGNORE
        self.assertEqual(
            self.filter.evaluate_candidate("ibuprofeno", "MEDICATION"),
            EnumDecision.FORCE_IGNORE
        )
        
        # Coincidencia exacta (diferente case) → FORCE_IGNORE
        self.assertEqual(
            self.filter.evaluate_candidate("IBUPROFENO", "MEDICATION"),
            EnumDecision.FORCE_IGNORE
        )
        
        self.assertEqual(
            self.filter.evaluate_candidate("Ibuprofeno", "MEDICATION"),
            EnumDecision.FORCE_IGNORE
        )
        
        # No coincidencia exacta (con dosis) → ESCALATE_TO_LLM
        self.assertEqual(
            self.filter.evaluate_candidate("ibuprofeno 600mg", "MEDICATION"),
            EnumDecision.ESCALATE_TO_LLM
        )

    def test_no_match_escalates_to_llm(self):
        """Test: sin coincidencia → escalar a LLM."""
        self.assertEqual(
            self.filter.evaluate_candidate("García", "PERSON"),
            EnumDecision.ESCALATE_TO_LLM
        )
        
        self.assertEqual(
            self.filter.evaluate_candidate("12345678A", "NUMERO_IDENTIF"),
            EnumDecision.ESCALATE_TO_LLM
        )

    def test_empty_text(self):
        """Test: texto vacío → escalar a LLM."""
        self.assertEqual(
            self.filter.evaluate_candidate("", "PERSON"),
            EnumDecision.ESCALATE_TO_LLM
        )
        
        self.assertEqual(
            self.filter.evaluate_candidate("   ", "PERSON"),
            EnumDecision.ESCALATE_TO_LLM
        )
        
        self.assertEqual(
            self.filter.evaluate_candidate(None, "PERSON"),
            EnumDecision.ESCALATE_TO_LLM
        )

    def test_detailed_evaluation(self):
        """Test: evaluación detallada devuelve información correcta."""
        result = self.filter.evaluate_candidate_detailed("Hospital Clínic", "HOSPITAL")
        self.assertEqual(result["decision"], EnumDecision.FORCE_ANONYMIZE)
        self.assertEqual(result["whitelist_match"], "Hospital Clínic")
        
        result = self.filter.evaluate_candidate_detailed("ibuprofeno", "MEDICATION")
        self.assertEqual(result["decision"], EnumDecision.FORCE_IGNORE)
        self.assertEqual(result["blacklist_match"], "ibuprofeno")
        
        result = self.filter.evaluate_candidate_detailed("García", "PERSON")
        self.assertEqual(result["decision"], EnumDecision.ESCALATE_TO_LLM)
        self.assertIsNone(result["whitelist_match"])
        self.assertIsNone(result["blacklist_match"])


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Ejecutar tests
    unittest.main(verbosity=2)
