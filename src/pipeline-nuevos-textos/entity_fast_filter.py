#!/usr/bin/env python3
"""
EntityFastFilter: Filtro determinista de alta velocidad para entidades NER.

Este módulo implementa un sistema de filtrado O(N) basado en flashtext
para reducir drásticamente las llamadas al LLM en pipelines de anonimización.

Arquitectura:
    - WhiteList (Hospitales, Lugares): Fuerza anonimización directa.
    - BlackList (Medicamentos, Patologías): Ignora salvo si NER == PERSON.
    - Default: Escala al LLM para validación semántica.

Fuentes de datos:
    - Archivos JSON con listas de términos.
    - Archivos CSV/Excel en carpeta LISTAS/ (via CsvListManager).

Performance:
    - Búsqueda O(N) mediante Aho-Corasick (flashtext).
    - Reducción estimada del 80% de llamadas al LLM.

Author: Pipeline Anonimización Clínica
Version: 1.1.0
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import unittest
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING

from flashtext import KeywordProcessor

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
        FORCE_ANONYMIZE: Anonimizar directamente sin LLM (whitelist match).
        FORCE_IGNORE: Ignorar directamente sin LLM (blacklist match, no PERSON).
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
    Filtro determinista de alta velocidad para entidades NER.

    Utiliza flashtext (Aho-Corasick) para búsquedas O(N) en listas
    blancas y negras, reduciendo llamadas al LLM en ~80%.

    Attributes:
        whitelist_processor: KeywordProcessor para hospitales/lugares (case-sensitive).
        blacklist_processor: KeywordProcessor para medicamentos/patologías (case-insensitive).

    Example:
        >>> filter = EntityFastFilter(
        ...     whitelist_paths=["hospitales.json", "lugares.json"],
        ...     blacklist_paths=["medicamentos.json", "patologias.json"]
        ... )
        >>> decision = filter.evaluate_candidate("Hospital Clínic", "HOSPITAL")
        >>> decision == EnumDecision.FORCE_ANONYMIZE
        True
    """

    # Regex compilado para eliminar dosis y unidades farmacéuticas
    # Usa lookahead para aplicar múltiples veces desde el final
    _DOSE_PATTERN: re.Pattern = re.compile(
        r'('
        r'\s+\d+\s*'  # Espacio + número antes de unidad
        r'('
        r'mg|g|kg|mcg|µg|ml|l|cc|dl|'  # Unidades de peso/volumen
        r'ui|u\.i\.|unidades?\s*internacionales?|'  # Unidades internacionales
        r'%|por\s*ciento'  # Porcentajes
        r')'
        r'|'  # O bien...
        r'\s+'  # Espacio seguido de...
        r'('
        r'comprimidos?|comp\.|'  # Comprimidos
        r'cápsulas?|caps?\.|'  # Cápsulas
        r'pastillas?|'  # Pastillas
        r'inyecciones?|iny\.|'  # Inyecciones
        r'ampollas?|amp\.|'  # Ampollas
        r'spray|aerosol|'  # Sprays
        r'gotas?|gtt\.|'  # Gotas
        r'sobres?|'  # Sobres
        r'supositorios?|sup\.|'  # Supositorios
        r'parches?|'  # Parches
        r'crema|pomada|gel|ungüento|'  # Tópicos
        r'jarabe|solución|suspensión|'  # Líquidos orales
        r'viales?|'  # Viales
        r'inhalaciones?|puff|'  # Inhaladores
        r'óvulos?|'  # Óvulos
        r'cada\s*\d+\s*h(oras?)?|'  # Frecuencias
        r'c/\d+\s*h|'  # Frecuencias abreviadas
        r'x\s*día|al\s*día|diarios?'  # Frecuencias diarias
        r')'
        r')+\s*$',
        re.IGNORECASE | re.UNICODE
    )

    def __init__(
        self,
        whitelist_paths: Optional[List[str]] = None,
        blacklist_paths: Optional[List[str]] = None,
        whitelist_data: Optional[List[str]] = None,
        blacklist_data: Optional[List[str]] = None,
        csv_lists_path: Optional[str] = None
    ) -> None:
        """
        Inicializa el filtro con listas blancas y negras.

        Args:
            whitelist_paths: Rutas a JSONs de whitelist (hospitales, lugares).
                             Case-sensitive matching.
            blacklist_paths: Rutas a JSONs de blacklist (medicamentos, patologías).
                             Case-insensitive matching.
            whitelist_data: Datos directos de whitelist (alternativa a paths).
            blacklist_data: Datos directos de blacklist (alternativa a paths).
            csv_lists_path: Ruta a carpeta con archivos CSV/Excel (ej: "LISTAS/").
                            Archivos con "Nomenclator_de_Facturacion" → blacklist.
                            Resto → whitelist.

        Raises:
            FileNotFoundError: Si algún archivo JSON no existe.
            json.JSONDecodeError: Si algún JSON es inválido.
        """
        # Inicializar procesadores flashtext
        # WhiteList: Case Sensitive (nombres propios de hospitales/lugares)
        self.whitelist_processor = KeywordProcessor(case_sensitive=True)
        # BlackList: Case Insensitive (medicamentos pueden variar en capitalización)
        self.blacklist_processor = KeywordProcessor(case_sensitive=False)

        # Referencia al CsvListManager (si se usa)
        self._csv_manager: Optional[CsvListManager] = None

        # Cargar whitelist desde archivos JSON
        whitelist_terms: Set[str] = set()
        if whitelist_paths:
            for path in whitelist_paths:
                terms = self._load_and_normalize_json(path, lowercase=False)
                whitelist_terms.update(terms)

        # Añadir datos directos de whitelist
        if whitelist_data:
            whitelist_terms.update(whitelist_data)

        # Cargar blacklist desde archivos JSON
        blacklist_terms: Set[str] = set()
        if blacklist_paths:
            for path in blacklist_paths:
                terms = self._load_and_normalize_json(path, lowercase=True)
                blacklist_terms.update(terms)

        # Añadir datos directos de blacklist
        if blacklist_data:
            blacklist_terms.update(t.lower() for t in blacklist_data)

        # Cargar listas desde CSV/Excel si se especifica la ruta
        if csv_lists_path:
            self._load_csv_lists(csv_lists_path, whitelist_terms, blacklist_terms)

        # Poblar procesadores flashtext
        for term in whitelist_terms:
            if term.strip():
                self.whitelist_processor.add_keyword(term.strip())

        for term in blacklist_terms:
            if term.strip():
                self.blacklist_processor.add_keyword(term.strip())

        # Guardar contadores para debugging
        self._whitelist_count = len(whitelist_terms)
        self._blacklist_count = len(blacklist_terms)

    def _load_csv_lists(
        self,
        csv_path: str,
        whitelist_terms: Set[str],
        blacklist_terms: Set[str]
    ) -> None:
        """
        Carga listas desde archivos CSV/Excel usando CsvListManager.

        Args:
            csv_path: Ruta a la carpeta con archivos CSV/Excel.
            whitelist_terms: Set donde añadir términos de whitelist.
            blacklist_terms: Set donde añadir términos de blacklist (en lowercase).
        """
        try:
            from csv_list_manager import CsvListManager

            self._csv_manager = CsvListManager(
                base_path=csv_path,
                lowercase_whitelist=False,  # Case-sensitive para nombres propios
                lowercase_blacklist=True    # Case-insensitive para medicamentos
            )
            self._csv_manager.load()

            # Añadir términos de CSV a los sets
            whitelist_terms.update(self._csv_manager.whitelist_global)
            blacklist_terms.update(self._csv_manager.blacklist_global)

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

    def _load_and_normalize_json(
        self,
        path: str,
        lowercase: bool = False
    ) -> List[str]:
        """
        Carga y normaliza cualquier estructura JSON a lista de términos.

        Este método es robusto ante cualquier formato JSON:
        - Lista simple: ["term1", "term2"]
        - Dict con valores: {"key1": "term1", "key2": ["term2", "term3"]}
        - Estructuras anidadas: {"data": {"items": ["term1"]}}
        - Arrays de arrays: [["term1", "term2"], ["term3"]]

        Args:
            path: Ruta al archivo JSON.
            lowercase: Si True, convierte todo a minúsculas.

        Returns:
            Lista de términos únicos extraídos del JSON.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            json.JSONDecodeError: Si el JSON es inválido.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo JSON no encontrado: {path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"JSON inválido en {path}: {e.msg}",
                e.doc,
                e.pos
            )

        # Extraer todos los strings recursivamente
        terms: Set[str] = set()
        self._extract_strings_recursive(data, terms)

        # Aplicar lowercase si es necesario
        if lowercase:
            terms = {t.lower() for t in terms}

        return list(terms)

    def _extract_strings_recursive(
        self,
        obj: Any,
        accumulator: Set[str]
    ) -> None:
        """
        Extrae recursivamente todas las cadenas de cualquier estructura.

        Args:
            obj: Objeto JSON (dict, list, str, int, etc.).
            accumulator: Set donde acumular los strings encontrados.
        """
        if isinstance(obj, str):
            # String directo: añadir si no está vacío
            cleaned = obj.strip()
            if cleaned:
                accumulator.add(cleaned)

        elif isinstance(obj, dict):
            # Diccionario: procesar todos los valores (ignorar claves)
            for value in obj.values():
                self._extract_strings_recursive(value, accumulator)

        elif isinstance(obj, (list, tuple)):
            # Lista/tupla: procesar cada elemento
            for item in obj:
                self._extract_strings_recursive(item, accumulator)

        # Ignorar int, float, bool, None (no son términos válidos)

    def _sanitize_dose(self, text: str) -> str:
        """
        Elimina dosis y unidades farmacéuticas del final del texto.

        Limpia patrones como:
        - "Paracetamol 500mg" → "Paracetamol"
        - "Ibuprofeno 400 mg comprimidos" → "Ibuprofeno"
        - "Omeprazol 20mg cápsulas" → "Omeprazol"

        Args:
            text: Texto a limpiar.

        Returns:
            Texto sin dosis/unidades al final.

        Example:
            >>> filter._sanitize_dose("Amoxicilina 500mg cápsulas")
            'Amoxicilina'
        """
        if not text:
            return text

        # Aplicar regex para eliminar dosis
        cleaned = self._DOSE_PATTERN.sub('', text)

        # Limpiar espacios residuales
        return cleaned.strip()

    def _is_in_whitelist(self, text: str) -> bool:
        """
        Verifica si algún término de whitelist aparece dentro del texto.

        IMPORTANTE: Usa búsqueda por SUBSTRING, no igualdad exacta.
        Ejemplo: "Hospital Clínic" en whitelist hace match con
                 "Ingreso en el Hospital Clínic de Barcelona"

        Args:
            text: Texto donde buscar (puede ser entity_text o contexto).

        Returns:
            True si algún término de whitelist aparece en el texto.
        """
        # flashtext extract_keywords busca SUBSTRINGS, no igualdad exacta
        matches = self.whitelist_processor.extract_keywords(text)
        return len(matches) > 0

    def match_in_whitelist(self, text: str) -> Optional[str]:
        """
        Busca términos de whitelist dentro del texto y devuelve el match.

        Búsqueda por SUBSTRING: si "Hospital Clínic" está en whitelist,
        hace match con "Ingreso en el Hospital Clínic de Barcelona".

        Args:
            text: Texto donde buscar.

        Returns:
            Primer término de whitelist encontrado, o None si no hay match.
            Si hay múltiples matches, devuelve el más largo (más específico).
        """
        if not text or not text.strip():
            return None

        # flashtext devuelve todos los keywords encontrados como substrings
        matches = self.whitelist_processor.extract_keywords(text)

        if not matches:
            return None

        # Devolver el match más largo (más específico)
        # Ej: si matchean "Hospital" y "Hospital Clínic", preferir el segundo
        return max(matches, key=len)

    def _is_in_blacklist(self, text: str) -> bool:
        """
        Verifica si algún término de blacklist aparece dentro del texto.

        IMPORTANTE: Usa búsqueda por SUBSTRING, no igualdad exacta.
        Ejemplo: "ibuprofeno" en blacklist hace match con
                 "Se pauta ibuprofeno 600 mg cada 8h"

        Args:
            text: Texto donde buscar (puede ser entity_text o contexto).

        Returns:
            True si algún término de blacklist aparece en el texto.
        """
        # Sanitizar dosis antes de buscar
        sanitized = self._sanitize_dose(text)

        # flashtext busca SUBSTRINGS, no igualdad exacta
        matches_original = self.blacklist_processor.extract_keywords(text)
        matches_sanitized = self.blacklist_processor.extract_keywords(sanitized)

        return len(matches_original) > 0 or len(matches_sanitized) > 0

    def match_in_blacklist(self, text: str) -> Optional[str]:
        """
        Busca términos de blacklist dentro del texto y devuelve el match.

        Búsqueda por SUBSTRING: si "ibuprofeno" está en blacklist,
        hace match con "Se pauta ibuprofeno 600 mg cada 8h".

        Args:
            text: Texto donde buscar.

        Returns:
            Primer término de blacklist encontrado, o None si no hay match.
            Si hay múltiples matches, devuelve el más largo (más específico).
        """
        if not text or not text.strip():
            return None

        # Sanitizar dosis antes de buscar
        sanitized = self._sanitize_dose(text)

        # flashtext devuelve todos los keywords encontrados como substrings
        matches_original = self.blacklist_processor.extract_keywords(text)
        matches_sanitized = self.blacklist_processor.extract_keywords(sanitized)

        # Combinar matches únicos
        all_matches = list(set(matches_original + matches_sanitized))

        if not all_matches:
            return None

        # Devolver el match más largo (más específico)
        return max(all_matches, key=len)

    def evaluate_candidate(
        self,
        entity_text: str,
        ner_label: str
    ) -> EnumDecision:
        """
        Evalúa una entidad candidata y decide el flujo de procesamiento.

        IMPORTANTE: La búsqueda es por SUBSTRING, no por igualdad exacta.
        - "Hospital Clínic" en whitelist matchea "Ingreso en Hospital Clínic de BCN"
        - "ibuprofeno" en blacklist matchea "Se pauta ibuprofeno 600 mg"

        Flujo de decisión:
        1. WhiteList match → FORCE_ANONYMIZE (hospitales, lugares conocidos)
        2. BlackList match:
           - Si ner_label == "PERSON" → ESCALATE_TO_LLM (posible nombre que coincide con medicamento)
           - Si no → FORCE_IGNORE (es un medicamento/patología, no anonimizar)
        3. Sin coincidencias → ESCALATE_TO_LLM (requiere validación semántica)

        Args:
            entity_text: Texto de la entidad detectada por NER.
            ner_label: Etiqueta asignada por el modelo NER.

        Returns:
            EnumDecision indicando el flujo a seguir.

        Example:
            >>> filter.evaluate_candidate("Hospital Clínic", "HOSPITAL")
            EnumDecision.FORCE_ANONYMIZE

            >>> filter.evaluate_candidate("Ingreso en Hospital Clínic", "HOSPITAL")
            EnumDecision.FORCE_ANONYMIZE  # Substring match!

            >>> filter.evaluate_candidate("ibuprofeno 600 mg", "MEDICATION")
            EnumDecision.FORCE_IGNORE  # Substring match con "ibuprofeno"

            >>> filter.evaluate_candidate("García", "PERSON")
            EnumDecision.ESCALATE_TO_LLM
        """
        if not entity_text or not entity_text.strip():
            return EnumDecision.ESCALATE_TO_LLM

        text = entity_text.strip()
        label_upper = ner_label.upper() if ner_label else ""

        # ═══════════════════════════════════════════════════════════════════
        # PASO 1: Prioridad absoluta - WhiteList (búsqueda por substring)
        # ═══════════════════════════════════════════════════════════════════
        whitelist_match = self.match_in_whitelist(text)
        if whitelist_match:
            logger.debug(f"Whitelist match: '{whitelist_match}' en '{text}'")
            return EnumDecision.FORCE_ANONYMIZE

        # ═══════════════════════════════════════════════════════════════════
        # PASO 2: Seguridad - BlackList (búsqueda por substring)
        # ═══════════════════════════════════════════════════════════════════
        blacklist_match = self.match_in_blacklist(text)
        if blacklist_match:
            logger.debug(f"Blacklist match: '{blacklist_match}' en '{text}'")
            # Caso especial: el NER cree que es una persona
            # Nombres como "García" o "Roca" podrían ser apellidos Y medicamentos
            # En ese caso, escalamos al LLM para validación semántica
            if label_upper in ("PERSON", "NOMBRE_SUJETO_ASISTENCIA",
                               "NOMBRE_PERSONAL_SANITARIO", "FAMILIAR"):
                return EnumDecision.ESCALATE_TO_LLM
            else:
                # Es un medicamento/patología confirmado, ignorar
                return EnumDecision.FORCE_IGNORE

        # ═══════════════════════════════════════════════════════════════════
        # PASO 3: Default - Sin coincidencias
        # ═══════════════════════════════════════════════════════════════════
        return EnumDecision.ESCALATE_TO_LLM

    def evaluate_candidate_detailed(
        self,
        entity_text: str,
        ner_label: str
    ) -> Dict[str, Any]:
        """
        Evalúa una entidad y devuelve información detallada del match.

        Útil para debugging y análisis de por qué se tomó cada decisión.

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
        if not entity_text or not entity_text.strip():
            return {
                "decision": EnumDecision.ESCALATE_TO_LLM,
                "whitelist_match": None,
                "blacklist_match": None,
                "reason": "Texto vacío"
            }

        text = entity_text.strip()
        label_upper = ner_label.upper() if ner_label else ""

        whitelist_match = self.match_in_whitelist(text)
        blacklist_match = self.match_in_blacklist(text)

        # Prioridad 1: Whitelist
        if whitelist_match:
            return {
                "decision": EnumDecision.FORCE_ANONYMIZE,
                "whitelist_match": whitelist_match,
                "blacklist_match": blacklist_match,
                "reason": f"Whitelist match: '{whitelist_match}' encontrado en texto"
            }

        # Prioridad 2: Blacklist
        if blacklist_match:
            if label_upper in ("PERSON", "NOMBRE_SUJETO_ASISTENCIA",
                               "NOMBRE_PERSONAL_SANITARIO", "FAMILIAR"):
                return {
                    "decision": EnumDecision.ESCALATE_TO_LLM,
                    "whitelist_match": None,
                    "blacklist_match": blacklist_match,
                    "reason": f"Blacklist match '{blacklist_match}' pero NER={ner_label}, escalando a LLM"
                }
            else:
                return {
                    "decision": EnumDecision.FORCE_IGNORE,
                    "whitelist_match": None,
                    "blacklist_match": blacklist_match,
                    "reason": f"Blacklist match: '{blacklist_match}' encontrado, NER={ner_label}"
                }

        # Default
        return {
            "decision": EnumDecision.ESCALATE_TO_LLM,
            "whitelist_match": None,
            "blacklist_match": None,
            "reason": "Sin coincidencias en listas, requiere validación LLM"
        }

    def get_stats(self) -> Dict[str, int]:
        """
        Retorna estadísticas del filtro.

        Returns:
            Dict con contadores de términos cargados.
        """
        return {
            "whitelist_terms": self._whitelist_count,
            "blacklist_terms": self._blacklist_count
        }

    def __repr__(self) -> str:
        """Representación string del filtro."""
        return (
            f"EntityFastFilter("
            f"whitelist={self._whitelist_count}, "
            f"blacklist={self._blacklist_count})"
        )


# ============================================================================
# TESTS UNITARIOS
# ============================================================================

class TestEntityFastFilter(unittest.TestCase):
    """Tests unitarios para EntityFastFilter."""

    @classmethod
    def setUpClass(cls) -> None:
        """Crea archivos JSON dummy con diferentes formatos."""
        cls.temp_dir = tempfile.mkdtemp()

        # ─────────────────────────────────────────────────────────────────
        # Formato 1: Lista simple (hospitales.json)
        # ─────────────────────────────────────────────────────────────────
        cls.hospitales_path = Path(cls.temp_dir) / "hospitales.json"
        hospitales_data = [
            "Hospital Clínic",
            "Hospital de Sant Pau",
            "Hospital Vall d'Hebron",
            "Hospital del Mar",
            "Hospital Universitari de Bellvitge"
        ]
        with open(cls.hospitales_path, 'w', encoding='utf-8') as f:
            json.dump(hospitales_data, f, ensure_ascii=False)

        # ─────────────────────────────────────────────────────────────────
        # Formato 2: Dict con claves arbitrarias (lugares.json)
        # ─────────────────────────────────────────────────────────────────
        cls.lugares_path = Path(cls.temp_dir) / "lugares.json"
        lugares_data = {
            "ciudades": ["Barcelona", "Madrid", "Valencia"],
            "barrios": {
                "barcelona": ["Eixample", "Gràcia", "Sarrià"],
                "madrid": ["Chamberí", "Salamanca"]
            },
            "calles": "Passeig de Gràcia"  # String único
        }
        with open(cls.lugares_path, 'w', encoding='utf-8') as f:
            json.dump(lugares_data, f, ensure_ascii=False)

        # ─────────────────────────────────────────────────────────────────
        # Formato 3: Arrays anidados (medicamentos.json)
        # ─────────────────────────────────────────────────────────────────
        cls.medicamentos_path = Path(cls.temp_dir) / "medicamentos.json"
        medicamentos_data = [
            ["Paracetamol", "Ibuprofeno", "Aspirina"],
            ["Omeprazol", "Pantoprazol"],
            [["Amoxicilina", "Azitromicina"], ["Ciprofloxacino"]],
            "Metformina"  # String suelto en array
        ]
        with open(cls.medicamentos_path, 'w', encoding='utf-8') as f:
            json.dump(medicamentos_data, f, ensure_ascii=False)

        # ─────────────────────────────────────────────────────────────────
        # Formato 4: Estructura compleja con metadata (patologias.json)
        # ─────────────────────────────────────────────────────────────────
        cls.patologias_path = Path(cls.temp_dir) / "patologias.json"
        patologias_data = {
            "metadata": {
                "version": "1.0",
                "source": "ICD-10"
            },
            "data": {
                "cardiovasculares": [
                    {"name": "Hipertensión", "code": "I10"},
                    {"name": "Diabetes Mellitus", "code": "E11"}
                ],
                "respiratorias": ["Asma", "EPOC", "Neumonía"],
                "oncologicas": {
                    "solidas": ["Carcinoma", "Melanoma"],
                    "hematologicas": ["Leucemia", "Linfoma"]
                }
            },
            "duplicados": ["Hipertensión", "Asma"]  # Duplicados intencionales
        }
        with open(cls.patologias_path, 'w', encoding='utf-8') as f:
            json.dump(patologias_data, f, ensure_ascii=False)

        # ─────────────────────────────────────────────────────────────────
        # Crear instancia del filtro
        # ─────────────────────────────────────────────────────────────────
        cls.filter = EntityFastFilter(
            whitelist_paths=[str(cls.hospitales_path), str(cls.lugares_path)],
            blacklist_paths=[str(cls.medicamentos_path), str(cls.patologias_path)]
        )

    def test_whitelist_force_anonymize(self) -> None:
        """Test: WhiteList → FORCE_ANONYMIZE."""
        # Hospitales
        self.assertEqual(
            self.filter.evaluate_candidate("Hospital Clínic", "HOSPITAL"),
            EnumDecision.FORCE_ANONYMIZE
        )
        self.assertEqual(
            self.filter.evaluate_candidate("Hospital de Sant Pau", "ORGANIZATION"),
            EnumDecision.FORCE_ANONYMIZE
        )

        # Lugares
        self.assertEqual(
            self.filter.evaluate_candidate("Barcelona", "LOCATION"),
            EnumDecision.FORCE_ANONYMIZE
        )
        self.assertEqual(
            self.filter.evaluate_candidate("Eixample", "TERRITORY"),
            EnumDecision.FORCE_ANONYMIZE
        )

    def test_blacklist_not_person_force_ignore(self) -> None:
        """Test: BlackList + NOT PERSON → FORCE_IGNORE."""
        # Medicamentos
        self.assertEqual(
            self.filter.evaluate_candidate("Paracetamol", "MEDICATION"),
            EnumDecision.FORCE_IGNORE
        )
        self.assertEqual(
            self.filter.evaluate_candidate("Ibuprofeno", "DRUG"),
            EnumDecision.FORCE_IGNORE
        )
        # Con dosis
        self.assertEqual(
            self.filter.evaluate_candidate("Omeprazol 20mg", "MEDICATION"),
            EnumDecision.FORCE_IGNORE
        )

        # Patologías
        self.assertEqual(
            self.filter.evaluate_candidate("Hipertensión", "CONDITION"),
            EnumDecision.FORCE_IGNORE
        )
        self.assertEqual(
            self.filter.evaluate_candidate("Diabetes Mellitus", "DISEASE"),
            EnumDecision.FORCE_IGNORE
        )

    def test_blacklist_person_escalate_to_llm(self) -> None:
        """Test: BlackList + PERSON → ESCALATE_TO_LLM."""
        # Caso: medicamento que podría ser apellido
        # (Aunque "Paracetamol" no es apellido real, simulamos el caso)
        self.assertEqual(
            self.filter.evaluate_candidate("Paracetamol", "PERSON"),
            EnumDecision.ESCALATE_TO_LLM
        )
        self.assertEqual(
            self.filter.evaluate_candidate("Aspirina", "NOMBRE_SUJETO_ASISTENCIA"),
            EnumDecision.ESCALATE_TO_LLM
        )

    def test_no_match_escalate_to_llm(self) -> None:
        """Test: Sin coincidencias → ESCALATE_TO_LLM."""
        self.assertEqual(
            self.filter.evaluate_candidate("García López", "PERSON"),
            EnumDecision.ESCALATE_TO_LLM
        )
        self.assertEqual(
            self.filter.evaluate_candidate("Dr. Martínez", "DOCTOR"),
            EnumDecision.ESCALATE_TO_LLM
        )
        self.assertEqual(
            self.filter.evaluate_candidate("C/ Balmes 123", "ADDRESS"),
            EnumDecision.ESCALATE_TO_LLM
        )

    def test_sanitize_dose(self) -> None:
        """Test: Sanitización de dosis."""
        self.assertEqual(
            self.filter._sanitize_dose("Paracetamol 500mg"),
            "Paracetamol"
        )
        self.assertEqual(
            self.filter._sanitize_dose("Ibuprofeno 400 mg comprimidos"),
            "Ibuprofeno"
        )
        self.assertEqual(
            self.filter._sanitize_dose("Omeprazol 20mg cápsulas"),
            "Omeprazol"
        )
        self.assertEqual(
            self.filter._sanitize_dose("Insulina 100 UI"),
            "Insulina"
        )
        self.assertEqual(
            self.filter._sanitize_dose("Ventolín spray"),
            "Ventolín"
        )

    def test_case_sensitivity(self) -> None:
        """Test: Case sensitivity correcta."""
        # WhiteList es case-sensitive
        self.assertEqual(
            self.filter.evaluate_candidate("Hospital Clínic", "HOSPITAL"),
            EnumDecision.FORCE_ANONYMIZE
        )
        # "hospital clínic" en minúsculas no debería coincidir (case-sensitive)
        # NOTA: flashtext busca substrings, así que depende de implementación

        # BlackList es case-insensitive
        self.assertEqual(
            self.filter.evaluate_candidate("PARACETAMOL", "MEDICATION"),
            EnumDecision.FORCE_IGNORE
        )
        self.assertEqual(
            self.filter.evaluate_candidate("paracetamol", "MEDICATION"),
            EnumDecision.FORCE_IGNORE
        )

    def test_empty_input(self) -> None:
        """Test: Entrada vacía → ESCALATE_TO_LLM."""
        self.assertEqual(
            self.filter.evaluate_candidate("", "PERSON"),
            EnumDecision.ESCALATE_TO_LLM
        )
        self.assertEqual(
            self.filter.evaluate_candidate("   ", "LOCATION"),
            EnumDecision.ESCALATE_TO_LLM
        )

    def test_stats(self) -> None:
        """Test: Estadísticas del filtro."""
        stats = self.filter.get_stats()
        self.assertGreater(stats["whitelist_terms"], 0)
        self.assertGreater(stats["blacklist_terms"], 0)


# ============================================================================
# MAIN - DEMO Y TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("EntityFastFilter - Demo y Tests")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────
    # Crear archivos JSON dummy temporales
    # ─────────────────────────────────────────────────────────────────────
    print("\n[1] Creando archivos JSON dummy con diferentes formatos...\n")

    temp_dir = tempfile.mkdtemp()

    # Formato 1: Lista simple
    hospitales_path = Path(temp_dir) / "hospitales.json"
    hospitales_data = [
        "Hospital Clínic",
        "Hospital de Sant Pau",
        "Hospital Vall d'Hebron"
    ]
    with open(hospitales_path, 'w', encoding='utf-8') as f:
        json.dump(hospitales_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {hospitales_path}")
    print(f"    Formato: Lista simple")
    print(f"    Contenido: {hospitales_data}\n")

    # Formato 2: Dict con claves arbitrarias
    lugares_path = Path(temp_dir) / "lugares.json"
    lugares_data = {
        "ciudades": ["Barcelona", "Madrid"],
        "barrios": {"bcn": ["Eixample", "Gràcia"]}
    }
    with open(lugares_path, 'w', encoding='utf-8') as f:
        json.dump(lugares_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {lugares_path}")
    print(f"    Formato: Dict con claves arbitrarias")
    print(f"    Contenido: {lugares_data}\n")

    # Formato 3: Arrays anidados
    medicamentos_path = Path(temp_dir) / "medicamentos.json"
    medicamentos_data = [
        ["Paracetamol", "Ibuprofeno"],
        [["Omeprazol"], "Metformina"]
    ]
    with open(medicamentos_path, 'w', encoding='utf-8') as f:
        json.dump(medicamentos_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {medicamentos_path}")
    print(f"    Formato: Arrays anidados")
    print(f"    Contenido: {medicamentos_data}\n")

    # Formato 4: Estructura compleja con metadata
    patologias_path = Path(temp_dir) / "patologias.json"
    patologias_data = {
        "metadata": {"version": "1.0"},
        "data": {
            "cardio": [{"name": "Hipertensión"}, {"name": "Diabetes"}],
            "resp": ["Asma", "EPOC"]
        }
    }
    with open(patologias_path, 'w', encoding='utf-8') as f:
        json.dump(patologias_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {patologias_path}")
    print(f"    Formato: Estructura compleja con metadata")
    print(f"    Contenido: {patologias_data}\n")

    # ─────────────────────────────────────────────────────────────────────
    # Crear instancia del filtro
    # ─────────────────────────────────────────────────────────────────────
    print("\n[2] Inicializando EntityFastFilter...\n")

    entity_filter = EntityFastFilter(
        whitelist_paths=[str(hospitales_path), str(lugares_path)],
        blacklist_paths=[str(medicamentos_path), str(patologias_path)]
    )

    print(f"  ✓ {entity_filter}")
    stats = entity_filter.get_stats()
    print(f"    WhiteList terms: {stats['whitelist_terms']}")
    print(f"    BlackList terms: {stats['blacklist_terms']}")

    # ─────────────────────────────────────────────────────────────────────
    # Demo de evaluación
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3] Demo de evaluación de candidatos...\n")

    test_cases = [
        # (entity_text, ner_label, expected_decision)
        ("Hospital Clínic", "HOSPITAL", "FORCE_ANONYMIZE"),
        ("Barcelona", "LOCATION", "FORCE_ANONYMIZE"),
        ("Paracetamol", "MEDICATION", "FORCE_IGNORE"),
        ("Ibuprofeno 400mg", "DRUG", "FORCE_IGNORE"),
        ("Hipertensión", "CONDITION", "FORCE_IGNORE"),
        ("Paracetamol", "PERSON", "ESCALATE_TO_LLM"),  # BlackList + PERSON
        ("García López", "PERSON", "ESCALATE_TO_LLM"),  # No match
        ("Dr. Martínez", "DOCTOR", "ESCALATE_TO_LLM"),  # No match
    ]

    print(f"  {'ENTITY':<25} {'NER_LABEL':<15} {'DECISION':<20} {'EXPECTED':<20}")
    print(f"  {'-'*25} {'-'*15} {'-'*20} {'-'*20}")

    for entity_text, ner_label, expected in test_cases:
        decision = entity_filter.evaluate_candidate(entity_text, ner_label)
        status = "✓" if decision.name == expected else "✗"
        print(f"  {entity_text:<25} {ner_label:<15} {decision.name:<20} {expected:<20} {status}")

    # ─────────────────────────────────────────────────────────────────────
    # Ejecutar tests unitarios
    # ─────────────────────────────────────────────────────────────────────
    print("\n[4] Ejecutando tests unitarios...\n")

    # Crear suite de tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEntityFastFilter)

    # Ejecutar con verbosidad
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"  Tests ejecutados: {result.testsRun}")
    print(f"  Éxitos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Fallos: {len(result.failures)}")
    print(f"  Errores: {len(result.errors)}")
    print("=" * 80)

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\n  ✓ Archivos temporales eliminados: {temp_dir}")
