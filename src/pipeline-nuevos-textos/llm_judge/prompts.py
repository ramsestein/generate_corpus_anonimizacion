#!/usr/bin/env python3
"""
Plantillas de prompts para el juez LLM.

Contiene las plantillas de system y user prompts,
así como funciones para cargar reglas de anotación.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# PLANTILLAS DE PROMPTS
# ============================================================================

PROMPT_TEMPLATES = {
    "default": {
        "name": "Clasificador de entidad por palabra y contexto",
        "version": "1.0",
        "description": "Juez LLM que determina si una palabra pertenece a una etiqueta según reglas.",
        "system": """ERES UN DETECTOR PROFESIONAL DE ENTIDADES.
DEBES DECIDIR SI UNA PALABRA, DENTRO DE UN TEXTO, ES UNA ENTIDAD VÁLIDA DE LA ETIQUETA INDICADA.

AQUÍ ESTÁN LAS REGLAS OFICIALES DE LA ETIQUETA:
{rules}

DATOS DE ENTRADA:
PALABRA: {keyword}
TEXTO DONDE APARECE: {context}
ETIQUETA: {label}

TAREA:
DECIDE SI LA PALABRA, EN ESTE TEXTO, CUMPLE LAS REGLAS DE LA ETIQUETA.

RESPONDE EXCLUSIVAMENTE:
"TRUE"  la palabra es una entidad válida.
"FALSE" la palabra NO es una entidad válida.""",
        "user": 'Responde SOLO: "TRUE" o "FALSE".',
    },
    
    "paranoid": {
        "name": "Auditor Paranoico con Guías Meddocan",
        "version": "2.0",
        "description": "Juez LLM más estricto para verificar anonimización.",
        "system": """ACTÚA COMO UN AUDITOR DE PRIVACIDAD CLÍNICA (DE-IDENTIFICATION).
TU OBJETIVO ES VERIFICAR SI UNA PALABRA DETECTADA DEBE SER ANONIMIZADA.

ESTÁS ANALIZANDO LA CATEGORÍA: **{label}**

A CONTINUACIÓN, LAS REGLAS ESTRICTAS Y EJEMPLOS PARA ESTA CATEGORÍA:
{rules}

CONTEXTO DEL DOCUMENTO:
... {context} ...

CASO A ANALIZAR:
Palabra/Frase candidata: "{keyword}"

TAREA:
1. Compara la palabra candidata y su contexto con los EJEMPLOS de las reglas.
2. Si la palabra encaja en la definición o se parece a los ejemplos, DEBES ANONIMIZARLA.
3. PRECAUCIÓN:
   - Si es "NOMBRE_...", distingue apellidos de nombres comunes.
   - Si es "TERRITORIO", distingue lugares específicos de gentilicios genéricos.
   - "OTROS_SUJETO_ASISTENCIA" incluye cualquier detalle que haga al paciente reconocible.

VEREDICTO FINAL:
Responde "TRUE" si debe ser ocultado según las reglas.
Responde "FALSE" si es un falso positivo.""",
        "user": 'Responde SOLO: "TRUE" o "FALSE".',
    },
    
    "simple": {
        "name": "Clasificador Simple",
        "version": "1.0",
        "description": "Prompt minimalista para respuestas rápidas.",
        "system": """Eres un clasificador de entidades médicas.

Reglas para {label}:
{rules}

Palabra: "{keyword}"
Contexto: "{context}"

¿Esta palabra es una entidad válida de tipo {label}?""",
        "user": 'Responde TRUE o FALSE.',
    },
}


class PromptBuilder:
    """
    Constructor de prompts para el juez LLM.
    
    Combina plantillas con datos específicos de cada entidad.
    """
    
    def __init__(
        self,
        template_name: str = "default",
        rules_path: Optional[str] = None
    ):
        """
        Inicializa el constructor de prompts.
        
        Args:
            template_name: Nombre de la plantilla a usar
            rules_path: Ruta al archivo de reglas de anotación
        """
        if template_name not in PROMPT_TEMPLATES:
            raise ValueError(f"Template '{template_name}' not found")
        
        self.template = PROMPT_TEMPLATES[template_name]
        self.rules_path = rules_path
        self._rules_cache: Optional[Dict] = None
    
    @property
    def rules(self) -> Dict:
        """Carga y cachea las reglas de anotación."""
        if self._rules_cache is None and self.rules_path:
            self._rules_cache = load_entity_rules(self.rules_path)
        return self._rules_cache or {}
    
    def build(
        self,
        keyword: str,
        context: str,
        label: str,
        rules_override: Optional[str] = None
    ) -> tuple:
        """
        Construye los prompts de sistema y usuario.
        
        Args:
            keyword: Palabra/entidad a evaluar
            context: Texto de contexto donde aparece
            label: Etiqueta de la entidad
            rules_override: Reglas personalizadas (opcional)
        
        Returns:
            Tuple (system_prompt, user_prompt)
        """
        # Obtener reglas
        if rules_override:
            rules_text = rules_override
        else:
            rules_text = get_entity_rules_for_label(label, self.rules_path) if self.rules_path else ""
        
        # Construir system prompt
        system_prompt = self.template["system"].format(
            keyword=keyword,
            context=context,
            label=label,
            rules=rules_text
        )
        
        # User prompt
        user_prompt = self.template["user"]
        
        return system_prompt, user_prompt
    
    def get_template_info(self) -> Dict:
        """Devuelve información sobre la plantilla actual."""
        return {
            "name": self.template["name"],
            "version": self.template["version"],
            "description": self.template["description"],
        }


# ============================================================================
# FUNCIONES DE UTILIDAD PARA REGLAS
# ============================================================================

def load_entity_rules(rules_path: str) -> Dict[str, List[str]]:
    """
    Carga las reglas de anotación desde el archivo JSON.

    Args:
        rules_path: Ruta al archivo JSON con las reglas

    Returns:
        Dict con las reglas por tipo de entidad

    Raises:
        FileNotFoundError: Si el archivo no existe
        json.JSONDecodeError: Si el JSON es inválido
    """
    path = Path(rules_path)
    
    # Buscar en varias ubicaciones
    if not path.exists():
        # Buscar en la raíz del proyecto
        project_root = Path(__file__).parent.parent.parent.parent
        alt_path = project_root / rules_path
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"No se encontró el archivo de reglas: {rules_path}")

    with open(path, 'r', encoding='utf-8') as f:
        rules = json.load(f)

    return rules


def get_entity_rules_for_label(label: str, rules_path: str) -> str:
    """
    Obtiene las reglas formateadas para una etiqueta específica.

    Args:
        label: Etiqueta de la entidad
        rules_path: Archivo JSON con las reglas

    Returns:
        String con las reglas formateadas

    Raises:
        KeyError: Si la etiqueta no existe en las reglas
    """
    rules = load_entity_rules(rules_path)

    if label not in rules:
        # Intentar buscar variantes
        for key in rules.keys():
            if label.upper() in key.upper() or key.upper() in label.upper():
                label = key
                break
        else:
            raise KeyError(f"Etiqueta '{label}' no encontrada en las reglas")

    # Formatear las reglas
    formatted_rules = []
    for i, rule in enumerate(rules[label], 1):
        formatted_rules.append(f"{i}. {rule}")

    return "\n".join(formatted_rules)
