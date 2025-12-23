#!/usr/bin/env python3
"""
LLMJudge OPTIMIZADO para Small Language Models (SLMs).

Implementa 3 técnicas avanzadas:
1. Entity Highlighting: Marcar visualmente la entidad en el texto
2. Constrained Decoding: Forzar JSON con Chain of Thought
3. Few-Shot Contrastivo: Ejemplos que diferencian PII vs RUIDO

Autor: Optimizado para modelos pequeños (Llama-3-8B, Mistral, Phi-3, Gemma)
"""

import subprocess
import logging
import time
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITARIAS - Extracción robusta de JSON
# =============================================================================

def extract_json_from_response(raw_response: str) -> Optional[Dict[str, Any]]:
    """
    Extrae JSON de respuesta del LLM que puede contener texto extra.
    
    Los modelos pequeños a veces añaden texto antes/después del JSON.
    Esta función usa regex para capturar solo lo que hay entre { y }.
    
    Args:
        raw_response: Respuesta cruda del LLM
        
    Returns:
        Dict parseado o None si no hay JSON válido
        
    Ejemplos:
        >>> extract_json_from_response('Aquí está: {"label": "PII", "thought": "..."}')
        {'label': 'PII', 'thought': '...'}
        
        >>> extract_json_from_response('{"label": "RUIDO"} - explicación extra')
        {'label': 'RUIDO'}
    """
    if not raw_response:
        return None
    
    # Buscar JSON entre { y } (más externo)
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
    if not match:
        logger.warning(f"No se encontró JSON en respuesta: {raw_response[:100]}")
        return None
    
    json_str = match.group(0)
    
    try:
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"Error al parsear JSON extraído: {e}\nJSON: {json_str[:200]}")
        return None


def highlight_entity_in_text(text: str, entity: str, marker: str = "<<<>>>") -> str:
    """
    Resalta la entidad en el texto con marcadores especiales.
    
    Técnica: Entity Highlighting para mejorar el foco de atención del SLM.
    
    Args:
        text: Texto completo
        entity: Entidad a resaltar
        marker: Marcador a usar (se divide en inicio y fin)
        
    Returns:
        Texto con la entidad resaltada
        
    Ejemplos:
        >>> highlight_entity_in_text("Dolor abdominal", "Dolor", "<<<>>>")
        "<<< DOLOR >>> abdominal"
    """
    if not entity or entity not in text:
        return text
    
    # Dividir marcador en inicio/fin
    mid = len(marker) // 2
    start_marker = marker[:mid]
    end_marker = marker[mid:]
    
    # Resaltar la entidad (case-insensitive, pero mantener mayúsculas originales)
    entity_upper = entity.upper()
    highlighted = f"{start_marker} {entity_upper} {end_marker}"
    
    # Reemplazar en el texto (case-insensitive)
    pattern = re.compile(re.escape(entity), re.IGNORECASE)
    result = pattern.sub(highlighted, text, count=1)
    
    return result


# =============================================================================
# PROMPT OPTIMIZADO - Few-Shot + CoT
# =============================================================================

SYSTEM_PROMPT_OPTIMIZED = """ERES UN CLASIFICADOR DE ENTIDADES MÉDICAS ESPECIALIZADO EN PII.

TU TAREA:
Analizar si una palabra/frase marcada en un texto médico es información personal (PII) que debe anonimizarse o es RUIDO (falso positivo).

⚠️ CRÍTICO: La clasificación DEBE basarse en el CONTEXTO, NO en la longitud o apariencia de la palabra.

ETIQUETA OBJETIVO: {label}

REGLAS OFICIALES:
{rules}

METODOLOGÍA DE ANÁLISIS:
1. PRIMERO: Lee el contexto completo alrededor de la palabra marcada
2. SEGUNDO: Identifica si el contexto indica función clínica o identificativa
3. TERCERO: Decide basándote en el USO, no en la forma de la palabra

IMPORTANTE - EJEMPLOS CONTRASTIVOS POR CONTEXTO:

EJEMPLO 1: "Dolores" - EL CONTEXTO DEFINE TODO
❌ RUIDO: "Paciente refiere <<< DOLORES >>> abdominales intensos"
   → Contexto: síntoma descrito por el paciente
   → Función: descripción clínica (NO identifica persona)
✅ PII: "La paciente <<< DOLORES >>> García acude a consulta"
   → Contexto: identificación de persona con apellido
   → Función: nombre propio (SÍ identifica persona)

EJEMPLO 2: "Madrid" - CONTEXTO DIFERENCIA GENTILICIO VS LUGAR
❌ RUIDO: "Paciente procedente de <<< MADRID >>>"
   → Contexto: origen geográfico genérico
   → Función: no identifica de forma única
✅ PII: "Ingresa en Hospital <<< MADRID >>> Norte"
   → Contexto: nombre específico de institución
   → Función: identifica lugar específico

EJEMPLO 3: "72" - CONTEXTO MÉDICO VS CONTEXTO DE IDENTIFICACIÓN
❌ RUIDO: "Presión arterial sistólica de <<< 72 >>> mmHg"
   → Contexto: medición clínica con unidades (mmHg)
   → Función: valor diagnóstico (NO identifica paciente)
   ⚠️ CLAVE: "mmHg", "mg/dl", "lpm" = VALORES CLÍNICOS
❌ RUIDO: "Glucemia <<< 72 >>> mg/dl"
   → Contexto: resultado de laboratorio con unidades
   → Función: valor clínico (NO edad)
✅ PII: "Varón de <<< 72 >>> años que acude..."
   → Contexto: descripción demográfica con "años"
   → Función: edad del paciente (SÍ identifica)

EJEMPLO 3: "72" - CONTEXTO MÉDICO VS CONTEXTO DE IDENTIFICACIÓN
❌ RUIDO: "Presión arterial sistólica de <<< 72 >>> mmHg"
   → Contexto: medición clínica con unidades (mmHg)
   → Función: valor diagnóstico (NO identifica paciente)
   ⚠️ CLAVE: "mmHg", "mg/dl", "lpm" = VALORES CLÍNICOS
❌ RUIDO: "Glucemia <<< 72 >>> mg/dl"
   → Contexto: resultado de laboratorio con unidades
   → Función: valor clínico (NO edad)
✅ PII: "Varón de <<< 72 >>> años que acude..."
   → Contexto: descripción demográfica con "años"
   → Función: edad del paciente (SÍ identifica)

SEÑALES DE CONTEXTO CLÍNICO (→ RUIDO):
- Unidades médicas: mmHg, mg/dl, lpm, °C, cm, kg
- Verbos de medición: "de X mmHg", "con X mg/dl", "mide X cm"
- Términos diagnósticos: glucemia, presión, frecuencia, temperatura
- Referencias genéricas: "antecedentes en la familia", "educación a cuidadores"

SEÑALES DE CONTEXTO IDENTIFICATIVO (→ PII):
- Edad: "de X años", "X años de edad", "paciente de X años", "varón/mujer de X años"
  ⚠️ IMPORTANTE: Si ves "años" junto al número → SIEMPRE es PII (edad del paciente)
- Nombres: con apellido, con título (Dr., Dña., Sr.)
- Familiares específicos: "su hijo X", "acompañado de su hija X" (con nombre propio)
- Lugares específicos: "Hospital X de [ciudad]" → PII si es nombre de institución

FORMATO DE RESPUESTA OBLIGATORIO (JSON):
Debes responder SIEMPRE en este formato JSON con dos campos:
{{
  "thought": "1. Analizo el contexto: [describe qué rodea a la palabra]. 2. Identifico función: [clínica o identificativa]. 3. Decisión: [PII o RUIDO porque...]",
  "label": "PII" o "RUIDO"
}}

⚠️ CRÍTICO:
- Primero ANALIZA EL CONTEXTO (campo "thought")
- Luego DECIDE basándote en la FUNCIÓN (campo "label")
- NO agregues texto fuera del JSON
- SOLO valores permitidos: "PII" o "RUIDO"
"""

USER_PROMPT_OPTIMIZED = """CASO A ANALIZAR:

TEXTO:
{highlighted_text}

INSTRUCCIONES:
1. La palabra/frase marcada con <<< >>> es la entidad candidata
2. Analiza el contexto inmediato
3. Compara con los ejemplos contrastivos
4. Responde en formato JSON con "thought" y "label"

Responde AHORA:"""


# =============================================================================
# RESULTADO CON METADATOS
# =============================================================================

@dataclass
class OptimizedJudgeResult:
    """Resultado optimizado con metadatos de razonamiento."""
    is_pii: Optional[bool]  # True = PII, False = RUIDO, None = error
    confidence: float
    thought: str  # Razonamiento del modelo (CoT)
    raw_response: str
    status: str  # "success", "parse_error", "timeout", "error"
    processing_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "is_pii": self.is_pii,
            "confidence": self.confidence,
            "thought": self.thought,
            "raw_response": self.raw_response,
            "status": self.status,
            "processing_time": self.processing_time,
            "details": self.details,
            "decision": "KEEP" if self.is_pii else "FILTER"
        }


# =============================================================================
# JUEZ OPTIMIZADO
# =============================================================================

class OptimizedLLMJudge:
    """
    Juez LLM optimizado para Small Language Models.
    
    Características:
    - Entity Highlighting para mejorar foco
    - Chain of Thought (CoT) para mejor precisión
    - Few-Shot Contrastivo para contextos ambiguos
    - Parser robusto con regex para JSON malformado
    """
    
    def __init__(
        self,
        model: str = "llama3.2:latest",
        rules_path: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 2,
        entity_marker: str = "<<<>>>"
    ):
        """
        Inicializa el juez optimizado.
        
        Args:
            model: Modelo Ollama (llama3.2, mistral, phi3, gemma2)
            rules_path: Ruta al JSON de reglas de anotación
            timeout: Timeout en segundos
            max_retries: Reintentos en caso de error
            entity_marker: Marcador para resaltar entidad (default: <<<>>>)
        """
        self.model = model
        self.rules_path = rules_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.entity_marker = entity_marker
        
        # Cargar reglas si existen
        self.rules_cache: Dict[str, str] = {}
        if rules_path and Path(rules_path).exists():
            self._load_rules()
        
        # Estadísticas
        self.stats = {
            "total_evaluated": 0,
            "pii_detected": 0,
            "ruido_detected": 0,
            "parse_errors": 0,
            "timeouts": 0,
            "total_time": 0.0,
        }
        
        logger.info(f"OptimizedLLMJudge initialized: model={model}, marker={entity_marker}")
    
    def _load_rules(self):
        """Carga las reglas de anotación desde JSON."""
        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extraer texto de reglas por etiqueta
            if "anotaciones" in data:
                for anotacion in data["anotaciones"]:
                    label = anotacion.get("etiqueta", "")
                    descripcion = anotacion.get("descripcion", "")
                    ejemplos = anotacion.get("ejemplos", [])
                    
                    rules_text = f"{descripcion}\n\nEjemplos:\n"
                    for ej in ejemplos[:5]:  # Limitar a 5 ejemplos
                        rules_text += f"- {ej}\n"
                    
                    self.rules_cache[label] = rules_text
                    
            logger.info(f"Reglas cargadas para {len(self.rules_cache)} etiquetas")
        except Exception as e:
            logger.error(f"Error al cargar reglas: {e}")
    
    def _get_rules_for_label(self, label: str) -> str:
        """Obtiene las reglas para una etiqueta."""
        return self.rules_cache.get(label, f"No hay reglas específicas para {label}.")
    
    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Llama a Ollama con manejo de errores.
        
        Args:
            system_prompt: Prompt de sistema
            user_prompt: Prompt de usuario
            
        Returns:
            Respuesta cruda del modelo
            
        Raises:
            RuntimeError: Si falla la llamada después de reintentos
        """
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        
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
                
                response = result.stdout.strip()
                if response:
                    return response
                
                if attempt < self.max_retries:
                    logger.warning(f"Respuesta vacía (intento {attempt}), reintentando...")
                    time.sleep(2)
                    
            except subprocess.TimeoutExpired:
                self.stats["timeouts"] += 1
                raise RuntimeError(f"Timeout ({self.timeout}s) en Ollama")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Error en Ollama: {e.stderr or e.output}")
        
        raise RuntimeError("Ollama devolvió respuesta vacía tras reintentos")
    
    def evaluate(
        self,
        entity_text: str,
        entity_label: str,
        context: str,
        debug: bool = False
    ) -> OptimizedJudgeResult:
        """
        Evalúa una entidad con el LLM optimizado.
        
        Args:
            entity_text: Texto de la entidad a evaluar
            entity_label: Etiqueta de la entidad (ej: NOMBRE_SUJETO_ASISTENCIA)
            context: Contexto donde aparece la entidad (frase/párrafo)
            debug: Si True, imprime prompts y respuesta
            
        Returns:
            OptimizedJudgeResult con decisión y razonamiento
        """
        start_time = time.time()
        self.stats["total_evaluated"] += 1
        
        try:
            # 1. ENTITY HIGHLIGHTING - Marcar la entidad en el contexto
            highlighted_context = highlight_entity_in_text(
                context, 
                entity_text, 
                self.entity_marker
            )
            
            # 2. OBTENER REGLAS
            rules_text = self._get_rules_for_label(entity_label)
            
            # 3. CONSTRUIR PROMPTS con Few-Shot + CoT
            system_prompt = SYSTEM_PROMPT_OPTIMIZED.format(
                label=entity_label,
                rules=rules_text
            )
            
            user_prompt = USER_PROMPT_OPTIMIZED.format(
                highlighted_text=highlighted_context
            )
            
            if debug:
                print("=" * 80)
                print("SYSTEM PROMPT:")
                print(system_prompt)
                print("\nUSER PROMPT:")
                print(user_prompt)
                print("=" * 80)
            
            # 4. LLAMAR A OLLAMA
            raw_response = self._call_ollama(system_prompt, user_prompt)
            
            if debug:
                print(f"\nRAW RESPONSE:\n{raw_response}\n")
            
            # 5. PARSEAR RESPUESTA con extractor robusto
            parsed_json = extract_json_from_response(raw_response)
            
            if parsed_json is None:
                # Fallback: intentar parsear sin JSON
                self.stats["parse_errors"] += 1
                return OptimizedJudgeResult(
                    is_pii=None,
                    confidence=0.0,
                    thought="ERROR: No se pudo extraer JSON de la respuesta",
                    raw_response=raw_response,
                    status="parse_error",
                    processing_time=time.time() - start_time,
                    details={"error": "json_extraction_failed"}
                )
            
            # 6. EXTRAER DECISIÓN
            thought = parsed_json.get("thought", "")
            label_decision = parsed_json.get("label", "").upper()
            
            is_pii = None
            if label_decision == "PII":
                is_pii = True
                self.stats["pii_detected"] += 1
            elif label_decision == "RUIDO":
                is_pii = False
                self.stats["ruido_detected"] += 1
            else:
                self.stats["parse_errors"] += 1
            
            processing_time = time.time() - start_time
            self.stats["total_time"] += processing_time
            
            return OptimizedJudgeResult(
                is_pii=is_pii,
                confidence=1.0 if is_pii is not None else 0.0,
                thought=thought,
                raw_response=raw_response,
                status="success" if is_pii is not None else "invalid_label",
                processing_time=processing_time,
                details={
                    "parsed_json": parsed_json,
                    "highlighted_context": highlighted_context
                }
            )
            
        except Exception as e:
            logger.error(f"Error al evaluar entidad: {e}")
            return OptimizedJudgeResult(
                is_pii=None,
                confidence=0.0,
                thought=f"ERROR: {str(e)}",
                raw_response="",
                status="error",
                processing_time=time.time() - start_time,
                details={"exception": str(e)}
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de uso."""
        return {
            **self.stats,
            "avg_time_per_eval": (
                self.stats["total_time"] / self.stats["total_evaluated"] 
                if self.stats["total_evaluated"] > 0 
                else 0.0
            )
        }


# =============================================================================
# FUNCIÓN DE API SIMPLIFICADA
# =============================================================================

def evaluate_with_optimized_llm(
    entity_text: str,
    entity_label: str,
    context: str,
    model: str = "llama3.2:latest",
    rules_path: Optional[str] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Función simplificada para evaluar una entidad con el juez optimizado.
    
    Args:
        entity_text: Texto de la entidad
        entity_label: Etiqueta de la entidad
        context: Contexto donde aparece
        model: Modelo Ollama a usar
        rules_path: Ruta al JSON de reglas (opcional)
        debug: Mostrar información de debug
        
    Returns:
        Diccionario con decisión y metadatos
        
    Ejemplo:
        >>> result = evaluate_with_optimized_llm(
        ...     entity_text="Dolores",
        ...     entity_label="NOMBRE_SUJETO_ASISTENCIA",
        ...     context="La paciente Dolores García acude a consulta",
        ...     model="llama3.2:latest"
        ... )
        >>> print(result['is_pii'])  # True
        >>> print(result['thought'])  # "Dolores es nombre propio..."
    """
    judge = OptimizedLLMJudge(
        model=model,
        rules_path=rules_path
    )
    
    result = judge.evaluate(
        entity_text=entity_text,
        entity_label=entity_label,
        context=context,
        debug=debug
    )
    
    return result.to_dict()
