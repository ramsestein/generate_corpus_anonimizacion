"""
Módulo que define la plantilla de prompt para el juez LLM.

Este módulo proporciona la configuración de la plantilla de prompt utilizada
por el juez LLM para determinar si una palabra, dentro de un contexto dado,
pertenece a una etiqueta oficial según reglas específicas.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# CONFIGURACIÓN DE LA PLANTILLA DEL PROMPT
# ============================================================================

PROMPT_CONFIG = {
    "name": "Clasificador de entidad por palabra y contexto",
    "version": "1.0",
    "description": "Juez LLM que determina si una palabra, dentro de un contexto dado, pertenece a una etiqueta oficial según reglas.",
    "language": "es",
    "template": """ERES UN DETECTOR PROFESIONAL DE ENTIDADES.
DEBES DECIDIR SI UNA PALABRA, DENTRO DE UN TEXTO, ES UNA ENTIDAD VÁLIDA DE LA ETIQUETA INDICADA TEN ENCUENTA EL CONTEXTO CIRCUNDANTE.

AQUÍ ESTÁN LAS REGLAS OFICIALES DE LA ETIQUETA:
{reglas_etiqueta}

DATOS DE ENTRADA:
PALABRA: {keyword}
TEXTO DONDE APARECE: {context_text}
ETIQUETA: {label}

Si tienes dudas sobre la anotación, revisa las reglas proporcionadas.

TAREA:
DECIDE SI LA PALABRA, EN ESTE TEXTO, CUMPLE LAS REGLAS DE LA ETIQUETA.

RESPONDE EXCLUSIVAMENTE:
"TRUE"  la palabra es una entidad válida.
"FALSE" la palabra NO es una entidad válida.""",
    "input_variables": ["keyword", "context_text", "label", "location", "reglas_etiqueta"],
    "output_format": "text",
    "expected_fields": ["TRUE or FALSE"]
}

PROMPT_CONFIG_nue= {
    "name": "Auditor Paranoico con Guías Meddocan",
    "version": "3.0",
    "template": """ACTÚA COMO UN AUDITOR DE PRIVACIDAD CLÍNICA (DE-IDENTIFICATION).
TU OBJETIVO ES VERIFICAR SI UNA PALABRA DETECTADA DEBE SER ANONIMIZADA.

ESTÁS ANALIZANDO LA CATEGORÍA: **{label}**

A CONTINUACIÓN, LAS REGLAS ESTRICTAS Y EJEMPLOS PARA ESTA CATEGORÍA:
{reglas_etiqueta}


CONTEXTO DEL DOCUMENTO:
... {context_text} ...

CASO A ANALIZAR:
Palabra/Frase candidata: "{keyword}"

TAREA:
1. Compara la palabra candidata y su contexto con los EJEMPLOS de las reglas anteriores.
2. Si la palabra encaja en la definición o se parece a los ejemplos, DEBES ANONIMIZARLA.
3. PRECAUCIÓN:
   - Si la categoría es "NOMBRE_...", distingue apellidos de nombres comunes (ej: "Roca" vs "roca").
   - Si la categoría es "TERRITORIO", distingue lugares específicos (anotar) de gentilicios genéricos (a veces no se anotan, revisa contexto).
   - RECUERDA: "OTROS_SUJETO_ASISTENCIA" incluye cualquier detalle raro (tatuajes, profesiones únicas) que hagan al paciente reconocible.

VEREDICTO FINAL:
Responde "TRUE" si debe ser ocultado según las reglas.
Responde "FALSE" si es un falso positivo (medicamento, anatomía, verbo, etc).
"""
}

# ============================================================================
# FUNCIONES DE UTILIDAD PARA REGLAS
# ============================================================================

def load_entity_rules(rules_file: str = "guias-anotacion.json") -> Dict[str, List[str]]:
    """
    Carga las reglas de anotación desde el archivo JSON.

    Args:
        rules_file: Ruta al archivo JSON con las reglas

    Returns:
        Dict con las reglas por tipo de entidad

    Raises:
        FileNotFoundError: Si el archivo no existe
        json.JSONDecodeError: Si el JSON es inválido
    """
    # Buscar el archivo en la raíz del proyecto
    project_root = Path(__file__).parent.parent.parent
    rules_path = project_root / rules_file

    if not rules_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de reglas: {rules_path}\n"
            f"Buscado en: {rules_path.absolute()}"
        )

    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)

    return rules


def get_entity_rules_for_label(label: str, rules_file: str = "guias-anotacion.json") -> str:
    """
    Obtiene las reglas formateadas para una etiqueta específica.

    Útil para rellenar el placeholder {reglas_etiqueta} en el template.

    Args:
        label: Etiqueta de la entidad (ej: "NOMBRE_SUJETO_ASISTENCIA")
        rules_file: Archivo JSON con las reglas

    Returns:
        String con las reglas formateadas para esa etiqueta

    Raises:
        KeyError: Si la etiqueta no existe en las reglas
    """
    rules = load_entity_rules(rules_file)

    if label not in rules:
        raise KeyError(f"Etiqueta '{label}' no encontrada en las reglas")

    # Formatear las reglas de esta etiqueta
    formatted_rules = []
    for i, rule in enumerate(rules[label], 1):
        formatted_rules.append(f"{i}. {rule}")

    return "\n".join(formatted_rules)


def get_prompt_config() -> Dict:
    """
    Devuelve la configuración de la plantilla del prompt.

    Returns:
        Dict con la configuración completa del prompt
    """
    return PROMPT_CONFIG.copy()
