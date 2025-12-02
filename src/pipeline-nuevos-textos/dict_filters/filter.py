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
    2. Heurísticas opcionales (longitud mínima, formato)
    
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
        self.min_length_per_label = min_length_per_label or {}
        self.ignore_single_char = ignore_single_char
        self.ignore_numeric_only = ignore_numeric_only
        
        # Estadísticas
        self.stats = {
            "total_evaluated": 0,
            "force_anonymize": 0,
            "force_ignore": 0,
            "escalate": 0,
            "by_list": {"whitelist": 0, "blacklist": 0, "cie10": 0},
            "by_heuristic": {"single_char": 0, "numeric_only": 0, "min_length": 0}
        }
        
        logger.info("DictFilter initialized")
    
    def evaluate(self, entity_text: str, ner_label: str) -> FilterResult:
        """
        Evalúa una entidad y devuelve la decisión del filtro.
        
        Flujo de decisión:
        1. Aplicar heurísticas
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
        
        # Carácter único
        if self.ignore_single_char and len(text) <= 1:
            self.stats["force_ignore"] += 1
            self.stats["by_heuristic"]["single_char"] += 1
            return FilterResult(
                decision=FilterDecision.FORCE_IGNORE,
                reason="Entidad de un solo carácter",
                heuristic_applied="single_char"
            )
        
        # Solo numérico
        if self.ignore_numeric_only and text.isdigit():
            self.stats["force_ignore"] += 1
            self.stats["by_heuristic"]["numeric_only"] += 1
            return FilterResult(
                decision=FilterDecision.FORCE_IGNORE,
                reason="Entidad solo numérica",
                heuristic_applied="numeric_only"
            )
        
        # Longitud mínima por etiqueta
        if ner_label in self.min_length_per_label:
            min_len = self.min_length_per_label[ner_label]
            if len(text) < min_len:
                self.stats["force_ignore"] += 1
                self.stats["by_heuristic"]["min_length"] += 1
                return FilterResult(
                    decision=FilterDecision.FORCE_IGNORE,
                    reason=f"Longitud {len(text)} < mínimo {min_len} para {ner_label}",
                    heuristic_applied="min_length"
                )
        
        # WHITELIST
        if self.list_loader.is_in_whitelist(text):
            self.stats["force_anonymize"] += 1
            self.stats["by_list"]["whitelist"] += 1
            return FilterResult(
                decision=FilterDecision.FORCE_ANONYMIZE,
                reason="Coincidencia exacta en whitelist",
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
            "by_heuristic": {"single_char": 0, "numeric_only": 0, "min_length": 0}
        }
