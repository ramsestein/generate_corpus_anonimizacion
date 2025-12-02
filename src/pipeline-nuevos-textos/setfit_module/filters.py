#!/usr/bin/env python3
"""
Filtros auxiliares para el clasificador SetFit.

Contiene los detectores de ruido, PII obvio y fragmentos
como clases separadas para facilitar testing y extensión.
"""

import re
import logging
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Resultado de aplicar un filtro."""
    matched: bool
    reason: str = ""
    confidence: float = 1.0


class NoiseFilter:
    """
    Filtro de ruido obvio (pre-SetFit).
    
    Detecta entidades que claramente no son PII:
    - Palabras comunes (paciente, hospital, etc.)
    - Títulos aislados (Dr., Sra., etc.)
    - Fragmentos muy cortos
    - Puntuación
    """
    
    DEFAULT_COMMON_WORDS = {
        'paciente', 'hospital', 'servicio', 'unidad', 'centro', 'clinica', 'consulta',
        'medico', 'enfermera', 'enfermero', 'doctor', 'doctora', 'cirujano',
        'españa', 'madrid', 'barcelona', 'valencia', 'sevilla', 'bilbao',
        'nacional', 'general', 'universitario', 'regional', 'provincial',
        'urgencias', 'emergencias', 'consultas', 'externas', 'hospitalizacion',
        'medicina', 'cirugia', 'pediatria', 'cardiologia', 'neurologia',
        'tratamiento', 'diagnostico', 'evolucion', 'anamnesis', 'exploracion',
        'informe', 'historia', 'clinica', 'medica', 'alta', 'ingreso',
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        'del', 'de', 'en', 'con', 'por', 'para', 'ante', 'sobre',
        'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
        'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
        'primero', 'segundo', 'tercero', 'cuarto', 'quinto',
        'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
        'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
        'lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo',
    }
    
    DEFAULT_ISOLATED_TITLES = {'dr', 'dra', 'sr', 'sra', 'don', 'doña', 'dña', 'd', 'prof', 'lic'}
    
    DEFAULT_FRAGMENTS = {
        '+', 'b', 'c', 'hc', 'cp', 'tf', 'tel', 'nº', 'n°', 'num',
        'calle', 'c/', 'avda', 'av', 'pza', 'plaza',
    }
    
    def __init__(
        self,
        common_words: Optional[Set[str]] = None,
        isolated_titles: Optional[Set[str]] = None,
        fragments: Optional[Set[str]] = None,
        min_length: int = 2
    ):
        """
        Inicializa el filtro de ruido.
        
        Args:
            common_words: Palabras comunes a filtrar
            isolated_titles: Títulos aislados a filtrar
            fragments: Fragmentos conocidos a filtrar
            min_length: Longitud mínima para no ser considerado muy corto
        """
        self.common_words = common_words or self.DEFAULT_COMMON_WORDS
        self.isolated_titles = isolated_titles or self.DEFAULT_ISOLATED_TITLES
        self.fragments = fragments or self.DEFAULT_FRAGMENTS
        self.min_length = min_length
    
    def is_noise(self, entity_text: str, entity_label: str = "") -> FilterResult:
        """
        Determina si una entidad es ruido obvio.
        
        Args:
            entity_text: Texto de la entidad
            entity_label: Etiqueta de la entidad (opcional)
        
        Returns:
            FilterResult con matched=True si es ruido
        """
        normalized = entity_text.lower().strip()
        
        # Palabras comunes
        if normalized in self.common_words:
            return FilterResult(True, f"common_word:{normalized}")
        
        # Títulos aislados
        if normalized in self.isolated_titles:
            return FilterResult(True, f"isolated_title:{normalized}")
        
        # Muy corto
        if len(entity_text.strip()) <= self.min_length and not entity_text.strip().isdigit():
            return FilterResult(True, f"too_short:{entity_text}")
        
        # Solo números de 1-2 dígitos
        if re.match(r'^\d{1,2}$', entity_text.strip()):
            return FilterResult(True, f"single_number:{entity_text}")
        
        # Solo puntuación
        if re.match(r'^[^\w]+$', entity_text):
            return FilterResult(True, f"punctuation_only:{entity_text}")
        
        # Fragmentos conocidos
        if normalized in self.fragments:
            return FilterResult(True, f"known_fragment:{normalized}")
        
        return FilterResult(False)
    
    def add_common_word(self, word: str):
        """Añade una palabra a la lista de palabras comunes."""
        self.common_words.add(word.lower())
    
    def add_fragment(self, fragment: str):
        """Añade un fragmento a la lista de fragmentos."""
        self.fragments.add(fragment.lower())


class PIIDetector:
    """
    Detector de PII obvio (bypass SetFit).
    
    Detecta entidades que claramente son PII mediante patrones regex:
    - Emails
    - Teléfonos
    - DNI/NIE
    - IBAN
    - Tarjetas de crédito
    - Fechas completas
    """
    
    DEFAULT_PATTERNS = {
        'email': r'^[\w.+-]+@[\w-]+\.[\w.-]+$',
        'phone_es': r'^(\+34\s?)?(6|7|8|9)\d{2}[\s.-]?\d{3}[\s.-]?\d{3}$',
        'phone_intl': r'^\+\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}$',
        'dni_nif': r'^[0-9]{8}[A-Z]$',
        'nie': r'^[XYZ][0-9]{7}[A-Z]$',
        'iban_es': r'^ES\d{22}$',
        'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$',
        'ss_number': r'^\d{12}$',
        'postal_code': r'^\d{5}$',
        'license_plate': r'^[0-9]{4}[A-Z]{3}$|^[A-Z]{1,2}[0-9]{4}[A-Z]{2}$',
        'full_date': r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',
    }
    
    def __init__(self, patterns: Optional[dict] = None):
        """
        Inicializa el detector de PII.
        
        Args:
            patterns: Diccionario de patrones regex {nombre: patron}
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS.copy()
        self._compiled = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }
    
    def is_pii(self, entity_text: str, entity_label: str = "") -> FilterResult:
        """
        Determina si una entidad es PII obvio.
        
        Args:
            entity_text: Texto de la entidad
            entity_label: Etiqueta de la entidad (opcional)
        
        Returns:
            FilterResult con matched=True si es PII obvio
        """
        text = entity_text.strip()
        
        # Probar cada patrón
        for pattern_name, regex in self._compiled.items():
            if regex.match(text):
                return FilterResult(True, pattern_name)
        
        # Nombres compuestos (al menos 2 palabras con mayúscula)
        if entity_label in ['NOMBRE_SUJETO_ASISTENCIA', 'NOMBRE_PERSONAL_SANITARIO']:
            words = text.split()
            if len(words) >= 2 and all(w[0].isupper() for w in words if len(w) > 1):
                return FilterResult(True, "compound_name")
        
        return FilterResult(False)
    
    def add_pattern(self, name: str, pattern: str):
        """Añade un nuevo patrón de PII."""
        self.patterns[name] = pattern
        self._compiled[name] = re.compile(pattern, re.IGNORECASE)


class FragmentDetector:
    """
    Detector de fragmentos de entidades.
    
    Detecta entidades que son parte de otras entidades más largas:
    - Textos muy cortos
    - Substrings de otras entidades del documento
    """
    
    def __init__(self, min_length: int = 2):
        """
        Inicializa el detector de fragmentos.
        
        Args:
            min_length: Longitud mínima para no ser considerado fragmento
        """
        self.min_length = min_length
    
    def is_fragment(
        self,
        entity_text: str,
        entity_label: str = "",
        full_document: Optional[str] = None,
        other_entities: Optional[List[str]] = None
    ) -> FilterResult:
        """
        Determina si una entidad es un fragmento.
        
        Args:
            entity_text: Texto de la entidad
            entity_label: Etiqueta de la entidad (opcional)
            full_document: Documento completo (opcional)
            other_entities: Otras entidades del documento (opcional)
        
        Returns:
            FilterResult con matched=True si es fragmento
        """
        normalized = entity_text.lower().strip()
        
        # Muy corto
        if len(normalized) <= self.min_length:
            return FilterResult(True, "too_short")
        
        # Parte de otra entidad
        if other_entities:
            for other in other_entities:
                other_norm = other.lower().strip()
                if normalized != other_norm and normalized in other_norm:
                    return FilterResult(True, f"subset_of:{other}")
        
        # Parte de un patrón más largo en el documento
        if full_document:
            pattern = rf'\b\w+\s+{re.escape(entity_text)}\s+\w+\b'
            matches = re.findall(pattern, full_document, re.IGNORECASE)
            if matches:
                return FilterResult(True, f"part_of:{matches[0]}")
        
        return FilterResult(False)
