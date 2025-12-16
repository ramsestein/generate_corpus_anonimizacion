#!/usr/bin/env python3
"""
DictFilter - Filtro determinista basado en listas.

Evalúa entidades usando listas blancas/negras y devuelve decisiones:
- FORCE_ANONYMIZE: Anonimizar (match en whitelist)
- FORCE_IGNORE: Ignorar (match en blacklist/CIE10)
- ESCALATE: Requiere evaluación adicional (LLM)
"""

import logging
from enum import Enum, auto
from typing import Dict, Optional, Any
from dataclasses import dataclass

from .list_loader import ListLoader

logger = logging.getLogger(__name__)


class FilterDecision(Enum):
    """Decisiones posibles del filtro determinista."""
    FORCE_ANONYMIZE = auto()  # Anonimizar directamente (whitelist)
    FORCE_IGNORE = auto()      # Ignorar directamente (blacklist/CIE10)
    ESCALATE = auto()          # Requiere LLM


@dataclass
class FilterResult:
    """Resultado detallado de una evaluación del filtro."""
    decision: FilterDecision
    reason: str
    matched_list: str = "none"
    matched_term: Optional[str] = None
    heuristic_applied: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "decision": self.decision.name,
            "reason": self.reason,
            "matched_list": self.matched_list,
            "matched_term": self.matched_term,
            "heuristic_applied": self.heuristic_applied,
        }


class DictFilter:
    """
    Filtro determinista basado en listas blancas/negras.
    
    Evalúa entidades usando:
    1. Coincidencias EXACTAS en listas
    2. Matching consciente de etiquetas (whitelist solo aplica a labels geográficos)
    
    Devuelve decisiones determinísticas para la mayoría de entidades.
    """
    
    def __init__(
        self,
        list_loader: ListLoader,
        min_length_per_label: Optional[Dict[str, int]] = None,
        ignore_single_char: bool = True,
        ignore_numeric_only: bool = False
    ):
        """
        Inicializa el filtro determinista.
        
        Args:
            list_loader: Instancia de ListLoader con listas cargadas
            min_length_per_label: Longitud mínima por tipo de etiqueta
            ignore_single_char: Si True, ignora entidades de 1 carácter
            ignore_numeric_only: Si True, ignora entidades solo numéricas
        """
        self.list_loader = list_loader
        self.ignore_single_char = ignore_single_char
        self.ignore_numeric_only = ignore_numeric_only
        
        # Labels compatibles con la whitelist de lugares/hospitales
        # (NO incluye nombres personales)
        self.geo_labels = {
            "HOSPITAL", "INSTITUCION", "CENTRO_DE_SALUD",
            "PAIS", "TERRITORIO", "CALLE", "DIRECCION",
            "OTROS_SUJETO_ASISTENCIA",  # Puede ser lugar
            "PROFESION",  # "Grado" es un municipio
        }
        
        self.stats = {
            "total_evaluated": 0,
            "force_anonymize": 0,
            "force_ignore": 0,
            "escalate": 0,
            "by_list": {"whitelist": 0, "blacklist": 0, "cie10": 0},
            "by_heuristic": {"single_char": 0, "numeric_only": 0},
            "skipped_label_mismatch": 0,
        }
        
        logger.info("DictFilter initialized")
    
    def evaluate(self, entity_text: str, ner_label: str) -> FilterResult:
        """
        Evalúa una entidad y devuelve la decisión del filtro.
        
        Flujo de decisión:
        1. Aplicar heurísticas (solo numéricas)
        2. Verificar whitelist → FORCE_ANONYMIZE
        3. Verificar blacklist → FORCE_IGNORE
        4. Verificar CIE10 → FORCE_IGNORE
        5. Ninguna coincidencia → ESCALATE
        
        Args:
            entity_text: Texto de la entidad
            ner_label: Etiqueta NER
            
        Returns:
            FilterResult con la decisión y detalles
        """
        self.stats["total_evaluated"] += 1
        text = entity_text.strip()
        
        # HEURÍSTICAS
        
        # Solo numérico
        if self.ignore_numeric_only and text.isdigit():
            self.stats["force_ignore"] += 1
            self.stats["by_heuristic"]["numeric_only"] += 1
            return FilterResult(
                decision=FilterDecision.FORCE_IGNORE,
                reason="Entidad solo numérica",
                heuristic_applied="numeric_only"
            )
        
        # WHITELIST (solo si label es compatible con geolocalización)
        # Evita que "Iglesias" (apellido) sea rescatado por ser un municipio
        is_name_label = ner_label in {
            "NOMBRE_SUJETO_ASISTENCIA", "NOMBRE_PERSONAL_SANITARIO",
            "FAMILIARES_SUJETO_ASISTENCIA", "SEXO_SUJETO_ASISTENCIA"
        }
        
        if self.list_loader.is_in_whitelist(text):
            if is_name_label:
                # Es un nombre personal, NO rescatamos por whitelist geográfica
                self.stats["skipped_label_mismatch"] += 1
                # Continuar a blacklist/CIE10/escalate normalmente
            else:
                self.stats["force_anonymize"] += 1
                self.stats["by_list"]["whitelist"] += 1
                return FilterResult(
                    decision=FilterDecision.FORCE_ANONYMIZE,
                    reason="Coincidencia exacta en whitelist (label compatible)",
                    matched_list="whitelist",
                    matched_term=text
                )
        
        # BLACKLIST
        if self.list_loader.is_in_blacklist(text):
            self.stats["force_ignore"] += 1
            self.stats["by_list"]["blacklist"] += 1
            return FilterResult(
                decision=FilterDecision.FORCE_IGNORE,
                reason="Coincidencia exacta en blacklist",
                matched_list="blacklist",
                matched_term=text
            )
        
        # CIE10
        if self.list_loader.is_in_cie10(text):
            self.stats["force_ignore"] += 1
            self.stats["by_list"]["cie10"] += 1
            return FilterResult(
                decision=FilterDecision.FORCE_IGNORE,
                reason="Coincidencia exacta en CIE10",
                matched_list="cie10",
                matched_term=text
            )
        
        # ESCALATE
        self.stats["escalate"] += 1
        return FilterResult(
            decision=FilterDecision.ESCALATE,
            reason="No hay coincidencias determinísticas",
            matched_list="none"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del filtro."""
        total = self.stats["total_evaluated"]
        if total == 0:
            return self.stats
        
        deterministic = self.stats["force_anonymize"] + self.stats["force_ignore"]
        
        return {
            **self.stats,
            "deterministic_resolution": {
                "count": deterministic,
                "percentage": (deterministic / total) * 100
            },
            "escalate_rate": {
                "count": self.stats["escalate"],
                "percentage": (self.stats["escalate"] / total) * 100
            }
        }
    
    def reset_stats(self):
        """Reinicia las estadísticas."""
        self.stats = {
            "total_evaluated": 0,
            "force_anonymize": 0,
            "force_ignore": 0,
            "escalate": 0,
            "by_list": {"whitelist": 0, "blacklist": 0, "cie10": 0},
            "by_heuristic": {"single_char": 0, "numeric_only": 0},
            "skipped_label_mismatch": 0,
        }
