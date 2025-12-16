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
    
    "strict_v3": {
        "name": "Validador Estricto V3 - Precision First",
        "version": "3.0",
        "description": "Juez LLM de alta precision con criterios de evidencia explicitos. Prioriza reducir FP.",
        "system": """ERES UN VALIDADOR ESTRICTO DE ENTIDADES PII EN DOCUMENTOS CLINICOS.

TU TAREA:
Determinar si la KEYWORD proporcionada, dentro del CONTEXTO dado, corresponde REALMENTE a la ETIQUETA indicada.

REGLAS DE DECISION:
1. SOLO puedes usar la informacion del CONTEXTO proporcionado.
2. NO puedes inferir, completar ni asumir informacion no presente.
3. La KEYWORD debe cumplir EXPLICITAMENTE las reglas de la ETIQUETA.
4. En caso de ambiguedad o duda -> FALSE.

CRITERIO BASE:
- TRUE: La keyword cumple CLARAMENTE las reglas Y el contexto lo DEMUESTRA.
- FALSE: En cualquier otro caso.

ETIQUETA A VALIDAR: {label}

REGLAS OFICIALES DE ESTA ETIQUETA:
{rules}

SEÑALES DE EVIDENCIA PARA {label}:
{evidence_signals}

SEÑALES DE RECHAZO (-> FALSE):
- Palabras genericas sin contexto identificativo
- Terminos clinicos o medicos comunes
- Plurales o referencias genericas
- Fragmentos incompletos o ambiguos
- Numeros aislados sin patron ni referencia

DATOS DEL CASO:
KEYWORD: "{keyword}"
CONTEXTO: "{context}"

INSTRUCCION FINAL:
Analiza si la KEYWORD, en este CONTEXTO, cumple las reglas de {label}.
Responde EXCLUSIVAMENTE: TRUE o FALSE.""",
        "user": 'Responde SOLO una palabra: TRUE o FALSE.',
    },
}


# ============================================================================
# SEÑALES DE EVIDENCIA POR ETIQUETA
# ============================================================================

EVIDENCE_SIGNALS = {
    # Nombres personales
    "NOMBRE_SUJETO_ASISTENCIA": """- Nombre propio en mayusculas con estructura de nombre personal
- Precedido de titulo (D., Dona, Sr., Sra.) o posesivo
- Contexto de identificacion del paciente
- NO: palabras genericas como "paciente", "usuario", "enfermo"
- NO: pronombres o referencias anonimas""",

    "NOMBRE_PERSONAL_SANITARIO": """- Nombre propio en mayusculas con estructura de nombre personal
- Precedido de titulo profesional (Dr., Dra., Enf., Lic.)
- Asociado a rol clinico (medico, enfermera, especialista)
- Firma o identificacion profesional explicita
- NO: roles genericos sin nombre ("el medico", "la enfermera")
- NO: especialidades sin nombre propio""",

    "FAMILIARES_SUJETO_ASISTENCIA": """- Referencia a familiar CONCRETO del paciente con posesivo
- Estructura: posesivo + relacion ("su madre", "padre del paciente")
- Nombre propio de familiar si presente
- NO: plurales genericos ("familiares", "la familia")
- NO: secciones informativas ("antecedentes familiares")
- NO: referencias sin posesivo ni vinculo explicito""",

    # Instituciones y lugares
    "HOSPITAL": """- Nombre propio de centro sanitario completo
- Estructura: "Hospital/Centro/Clinica" + nombre especifico
- Abreviaturas reconocibles (H., HU., HC.)
- NO: terminos genericos ("el hospital", "centro de salud")
- NO: tipos de servicio sin nombre de institucion""",

    "INSTITUCION": """- Nombre propio de institucion completo
- Organizacion identificable (ministerio, universidad, colegio)
- NO: referencias genericas ("la institucion", "el organismo")""",

    "CENTRO_DE_SALUD": """- Nombre propio del centro de atencion primaria
- Estructura: "C.S." o "Centro de Salud" + nombre especifico
- NO: termino generico sin nombre propio""",

    "TERRITORIO": """- Nombre propio geografico especifico (ciudad, region, pais)
- Contexto de procedencia, residencia o nacimiento
- NO: gentilicios ("espanol", "catalan")
- NO: referencias genericas ("la comunidad", "la region")""",

    "PAIS": """- Nombre propio de pais especifico
- Contexto de nacionalidad, procedencia, residencia
- NO: gentilicios ni adjetivos de nacionalidad""",

    "CALLE": """- Direccion postal completa o parcial
- Estructura: tipo via + nombre + numero (Calle/Av./C/ + nombre)
- Incluye numero de portal, piso, puerta
- Contexto de domicilio o residencia
- NO: tipos de via sin nombre especifico
- NO: numeros aislados sin contexto de direccion""",

    # Identificadores
    "ID_SUJETO_ASISTENCIA": """- Patron alfanumerico estructurado de identificacion
- Precedido de referencia explicita (DNI, NIE, pasaporte, NHC)
- Formato reconocible (letras+numeros, guiones)
- NO: numeros aislados sin contexto de identificador
- NO: palabras genericas ("identificacion", "numero")""",

    "ID_TITULACION_PERSONAL_SANITARIO": """- Numero de colegiado o identificador profesional
- Precedido de referencia (n. colegiado, ID profesional)
- NO: numeros sin contexto de titulacion""",

    "ID_ASEGURAMIENTO": """- Numero de poliza o tarjeta sanitaria
- Precedido de referencia (poliza, TSI, NASS, asegurado)
- Patron estructurado de aseguradora
- NO: numeros aislados sin contexto de seguro""",

    "NUMERO_IDENTIF": """- Cualquier codigo o numero de identificacion
- Con referencia explicita a su tipo
- NO: numeros sin contexto identificativo""",

    "ID_CONTACTO_ASISTENCIAL": """- Numero de episodio o contacto asistencial
- Precedido de referencia (NHC, episodio, ingreso, HC)
- Patron de historia clinica
- NO: numeros aislados""",

    # Contacto
    "NUMERO_TELEFONO": """- Numero telefonico completo o parcial
- Precedido de referencia (tel., telefono, contacto, movil)
- Formato telefonico reconocible (9 digitos, prefijo +34)
- NO: prefijos aislados (+34, 91)
- NO: numeros cortos sin contexto telefonico""",

    "NUMERO_FAX": """- Numero de fax completo
- Precedido de referencia explicita (fax)
- Formato telefonico
- NO: numeros sin contexto de fax""",

    "CORREO_ELECTRONICO": """- Direccion de correo electronico completa
- Formato usuario@dominio
- NO: dominios aislados
- NO: palabra "email" sin direccion""",

    "URL_WEB": """- Direccion web completa o parcial
- Formato reconocible (www., http://, .com, .es)
- NO: palabras genericas ("pagina web", "sitio")""",

    # Fechas y edad
    "FECHAS": """- Formato de fecha explicito (DD/MM/AAAA, dia-mes-ano)
- Expresion temporal clinica (fecha de nacimiento, ingreso, alta)
- Fecha completa o parcial con contexto temporal
- NO: anos aislados sin contexto de fecha personal
- NO: meses o dias aislados
- NO: referencias temporales genericas ("hace 2 meses")""",

    "EDAD_SUJETO_ASISTENCIA": """- Edad del paciente explicita
- Estructura: numero + anos/meses (45 anos, 3 meses)
- Contexto de descripcion del paciente
- NO: numeros aislados sin unidad temporal
- NO: rangos de edad genericos""",

    # Otros
    "SEXO_SUJETO_ASISTENCIA": """- Sexo del paciente explicito
- Terminos: varon, mujer, hombre, femenino, masculino
- Contexto de descripcion del paciente
- NO: pronombres (el, ella)
- NO: terminos en contexto no identificativo""",

    "PROFESION": """- Profesion u ocupacion del paciente
- Precedido de "trabaja como", "profesion:", "oficio"
- NO: profesiones del personal sanitario
- NO: terminos genericos ("trabajador", "empleado")""",

    "OTROS_SUJETO_ASISTENCIA": """- Cualquier dato que haga al paciente identificable
- Circunstancias raras o unicas
- Caracteristicas fisicas distintivas
- Relaciones especiales o publicidad
- NO: datos demograficos comunes
- NO: informacion clinica estandar""",
}

# Señal por defecto para etiquetas no mapeadas
DEFAULT_EVIDENCE_SIGNAL = """- La keyword debe cumplir claramente la definicion de la etiqueta
- El contexto debe demostrar inequivocamente que es PII
- NO: palabras genericas o fragmentos ambiguos"""


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
        
        # Obtener senales de evidencia especificas para la etiqueta
        evidence_signals = EVIDENCE_SIGNALS.get(label, DEFAULT_EVIDENCE_SIGNAL)
        
        # Construir system prompt
        system_prompt = self.template["system"].format(
            keyword=keyword,
            context=context,
            label=label,
            rules=rules_text,
            evidence_signals=evidence_signals
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
