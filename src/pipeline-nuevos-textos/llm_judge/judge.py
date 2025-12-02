#!/usr/bin/env python3
"""
LLMJudge - Evaluador de entidades con LLM.

Realiza llamadas al LLM (Ollama) para validar entidades ambiguas.
"""

import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .prompts import PromptBuilder, get_entity_rules_for_label

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Resultado de la evaluación del juez LLM."""
    is_valid: Optional[bool]
    confidence: float
    raw_response: str
    status: str
    processing_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
            "status": self.status,
            "processing_time": self.processing_time,
            "details": self.details,
        }


class LLMJudge:
    """
    Evaluador de entidades usando LLM.
    
    Llama a Ollama con el modelo configurado para validar
    si una entidad es realmente PII o un falso positivo.
    """
    
    # Tokens que indican TRUE
    TRUE_TOKENS = {'TRUE', '1', 'SI', 'SÍ', 'YES', 'CORRECTO', 'VALIDO', 'VÁLIDO', 'VERDADERO'}
    
    # Tokens que indican FALSE
    FALSE_TOKENS = {'FALSE', '0', 'NO', 'INCORRECTO', 'INVALIDO', 'INVÁLIDO', 'FALSO'}
    
    def __init__(
        self,
        model: str = "gemma3:270m",
        rules_path: Optional[str] = None,
        template_name: str = "default",
        timeout: int = 120,
        max_retries: int = 2
    ):
        """
        Inicializa el juez LLM.
        
        Args:
            model: Nombre del modelo Ollama
            rules_path: Ruta al archivo de reglas de anotación
            template_name: Nombre de la plantilla de prompts
            timeout: Timeout en segundos para llamadas a Ollama
            max_retries: Número máximo de reintentos
        """
        self.model = model
        self.rules_path = rules_path
        self.template_name = template_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Prompt builder
        self.prompt_builder = PromptBuilder(
            template_name=template_name,
            rules_path=rules_path
        )
        
        # Estadísticas
        self.stats = {
            "total_evaluated": 0,
            "valid": 0,
            "invalid": 0,
            "errors": 0,
            "total_time": 0.0,
        }
        
        logger.info(f"LLMJudge initialized with model={model}")
    
    def _check_ollama_available(self) -> bool:
        """Verifica que Ollama esté disponible."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        debug: bool = False
    ) -> Tuple[str, str]:
        """
        Realiza una llamada a Ollama.
        
        Args:
            system_prompt: Prompt de sistema
            user_prompt: Prompt de usuario
            debug: Mostrar prompts en consola
            
        Returns:
            Tuple (raw_output, cleaned_response)
        """
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        
        if debug:
            logger.debug(f"System prompt:\n{system_prompt}")
            logger.debug(f"User prompt:\n{user_prompt}")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                result = subprocess.run(
                    ["ollama", "run", self.model],
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    check=True
                )
                
                stdout = result.stdout or ""
                response = stdout.strip()
                
                if response:
                    return stdout, response
                
                if attempt < self.max_retries:
                    logger.warning(f"Empty response (attempt {attempt}), retrying...")
                    time.sleep(2)
                    
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Timeout ({self.timeout}s) al ejecutar Ollama")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Error al ejecutar Ollama: {e.stderr or e.output}")
        
        raise RuntimeError("Ollama devolvió respuesta vacía después de reintentos")
    
    def _parse_response(self, response: str) -> Optional[bool]:
        """
        Parsea la respuesta del LLM a booleano.
        
        Args:
            response: Respuesta del LLM
            
        Returns:
            True, False, o None si no se puede parsear
        """
        if not isinstance(response, str):
            return None
        
        cleaned = response.strip().upper()
        
        # Buscar tokens
        tokens = set(t.strip().upper() for t in cleaned.replace('\n', ' ').split())
        
        if tokens & self.TRUE_TOKENS:
            return True
        if tokens & self.FALSE_TOKENS:
            return False
        
        # Fallback: buscar substrings
        if any(tok in cleaned for tok in ('TRUE', 'VERDADERO', 'CORRECTO', 'YES')):
            return True
        if any(tok in cleaned for tok in ('FALSE', 'FALSO', 'INCORRECTO', 'NO')):
            return False
        
        return None
    
    def evaluate(
        self,
        entity_text: str,
        entity_label: str,
        context: str,
        debug: bool = False
    ) -> JudgeResult:
        """
        Evalúa una entidad con el LLM.
        
        Args:
            entity_text: Texto de la entidad
            entity_label: Etiqueta de la entidad
            context: Contexto donde aparece la entidad
            debug: Mostrar información de depuración
        
        Returns:
            JudgeResult con el veredicto
        """
        start_time = time.time()
        self.stats["total_evaluated"] += 1
        
        try:
            # Obtener reglas
            rules_text = ""
            if self.rules_path:
                try:
                    rules_text = get_entity_rules_for_label(entity_label, self.rules_path)
                except KeyError:
                    logger.warning(f"No rules found for label: {entity_label}")
            
            # Construir prompts
            system_prompt, user_prompt = self.prompt_builder.build(
                keyword=entity_text,
                context=context,
                label=entity_label,
                rules_override=rules_text if rules_text else None
            )
            
            # Llamar a Ollama
            raw_output, response = self._call_ollama(system_prompt, user_prompt, debug)
            
            # Parsear respuesta
            is_valid = self._parse_response(response)
            
            processing_time = time.time() - start_time
            self.stats["total_time"] += processing_time
            
            if is_valid is True:
                self.stats["valid"] += 1
            elif is_valid is False:
                self.stats["invalid"] += 1
            else:
                self.stats["errors"] += 1
            
            return JudgeResult(
                is_valid=is_valid,
                confidence=1.0 if is_valid is not None else 0.0,
                raw_response=response,
                status="success" if is_valid is not None else "parse_error",
                processing_time=processing_time,
                details={"raw_output": raw_output[:500]}
            )
            
        except Exception as e:
            self.stats["errors"] += 1
            processing_time = time.time() - start_time
            
            return JudgeResult(
                is_valid=None,
                confidence=0.0,
                raw_response="",
                status=f"error: {str(e)}",
                processing_time=processing_time,
                details={"exception": str(e)}
            )
    
    def evaluate_batch(
        self,
        entities: list,
        debug: bool = False
    ) -> list:
        """
        Evalúa un lote de entidades.
        
        Args:
            entities: Lista de dicts con entity_text, entity_label, context
            debug: Mostrar información de depuración
        
        Returns:
            Lista de JudgeResult
        """
        results = []
        total = len(entities)
        
        for i, entity in enumerate(entities, 1):
            if debug:
                logger.info(f"Processing {i}/{total}...")
            
            result = self.evaluate(
                entity_text=entity.get("entity_text", entity.get("text", "")),
                entity_label=entity.get("entity_label", entity.get("label", "")),
                context=entity.get("context", entity.get("sentence_context", "")),
                debug=debug
            )
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Devuelve estadísticas del juez."""
        total = self.stats["total_evaluated"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "avg_time": self.stats["total_time"] / total,
            "valid_rate": self.stats["valid"] / total * 100,
            "invalid_rate": self.stats["invalid"] / total * 100,
            "error_rate": self.stats["errors"] / total * 100,
        }
    
    def reset_stats(self):
        """Reinicia las estadísticas."""
        self.stats = {
            "total_evaluated": 0,
            "valid": 0,
            "invalid": 0,
            "errors": 0,
            "total_time": 0.0,
        }
