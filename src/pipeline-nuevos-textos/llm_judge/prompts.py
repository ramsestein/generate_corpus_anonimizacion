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
        "name": "Validador Estricto V3 - Recall Optimizado",
        "version": "3.1",
        "description": "Juez LLM optimizado para reducir FN. En caso de duda, marca como PII.",
        "system": """ERES UN VALIDADOR DE ENTIDADES PII EN DOCUMENTOS CLINICOS.

ETIQUETA: {label}

REGLAS OFICIALES:
{rules}

CRITERIOS DE VALIDACION:
TRUE si:
{evidence_signals}

FALSE solo si:
- Palabras genericas SIEMPRE FALSE: "paciente", "usuario", "enfermo", "hospital" (sin nombre propio)
- Termino medico o clinico comun sin valor identificativo
- Numero completamente aislado sin ningun patron ni contexto
- Referencias temporales vagas: "ayer", "hace X dias/meses"

REGLA DE ORO: Si hay DUDA sobre algo que NO esta en la lista de FALSE, responde TRUE.
Es preferible anonimizar de mas que dejar escapar PII real.

CASO A EVALUAR:
Keyword: "{keyword}"
Contexto: "{context}"

Analiza con la informacion del contexto. En caso de duda, responde TRUE.
Responde EXCLUSIVAMENTE: TRUE o FALSE.""",
        "user": 'Responde SOLO: TRUE o FALSE.',
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
- En datos de contacto familiar: "[**Carlos**] ([**hermano**])" - SI es PII
- Nombres de familiares en formularios de contacto - SI es PII
- NO: palabras genericas como "paciente", "usuario", "enfermo"
- NO: pronombres o referencias anonimas""",

    "NOMBRE_PERSONAL_SANITARIO": """- Nombre propio en mayusculas con estructura de nombre personal
- Precedido de titulo profesional (Dr., Dra., Enf., Lic.)
- Asociado a rol clinico (medico, enfermera, especialista)
- Firma o identificacion profesional explicita
- NO: roles genericos sin nombre ("el medico", "la enfermera")
- NO: especialidades sin nombre propio""",

    "FAMILIARES_SUJETO_ASISTENCIA": """- Relaciones familiares: madre, padre, hermano/a, hijo/a, esposo/a, marido, mujer, abuelo/a, tio/a
- Con o sin posesivo: "su madre", "la hermana", "hermano" - SI es PII
- Formato formulario: "([**relacion**])" o "nombre ([**relacion**])" - SI es PII
- En datos de contacto: "[**654789123**] ([**hermana**])" - SI es PII
- Relacion + nombre: "[**hermana Carmen**]", "[**hijo Carlos**]" - SI es PII
- En contexto de acompañante: "deambula con su esposa" - SI es PII
- En contexto de informar: "informo a la familia (esposa)" - SI es PII
- Terminos fill, marit, mare, pare (catalan) - SI es PII
- NO: "antecedentes familiares" como seccion de formulario
- NO: "la familia" como grupo generico sin identificar individuos""",

    # Instituciones y lugares
    "HOSPITAL": """- Nombre propio de centro sanitario completo o parcial
- Estructura: "Hospital/Centro/Clinica" + nombre especifico (Hospital Central, H. de Barcelona)
- Abreviaturas reconocibles (H., HU., HC.)
- Nombres propios de hospitales aunque sean comunes ("Hospital Central"): SI es PII
- NO: termino "hospital" solo sin nombre
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
- Codigos medicos alfanumericos: H025, H042, E041, J201, G033
- Codigos de ubicacion hospitalaria: G045, I105, 28B - SI es PII
- Formato letra+numeros en contexto de traslado/cama/habitacion - SI es PII
- Con referencia explicita o implicita a su tipo
- Patrones tipo letra+numeros en contexto clinico: SI es PII
- NO: numeros completamente aislados sin patron alfanumerico
- NO: codigos CIE-10 de diagnostico (ej: I25.1)""",

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
    "FECHAS": """- Formato de fecha explicito: DD/MM/AAAA, DD.MM.AAAA, DD/M, D.M.AA
- Formatos europeos con puntos: 15.3.19, 20.8.20, 12.03.2024 - SI es PII
- Formato abreviado: DD/MM, MM/AA, mes/año (marzo/24, 22/09/24)
- Meses textuales con año: "septiembre 2020", "agosto 2023" - SI es PII
- Anos aislados en contexto de historial: 2019, 2020, 2021 - SI es PII
- Patron "Fecha: [**...**]" en notas clinicas - SIEMPRE es PII
- Dias de semana en contexto de cita: "miercoles", "martes 19/03" - SI es PII
- Horas programadas en contexto de evento: "a las 10h", "14h" - SI es PII
- NO: referencias temporales vagas ("hace 2 meses", "ayer")
- NO: horas de constantes vitales sin contexto de programacion""",

    "EDAD_SUJETO_ASISTENCIA": """- Edad del paciente explicita
- Estructura: numero + anos/meses/semanas (45 anos, 3 meses, 68a, 72a)
- Abreviaturas de edad: 68a, 72a, 82a (numero + "a") en contexto de paciente: SI es PII
- Expresiones como "varon de 68", "mujer de 75a": SI es PII
- Contexto de descripcion del paciente
- NO: numeros completamente aislados sin unidad ni patron
- NO: rangos de edad de protocolos ("mayores de 65")""",

    # Otros
    "SEXO_SUJETO_ASISTENCIA": """- Terminos: varon, mujer, hombre, femenino, masculino, Mujer, Hombre
- Patron demografico inicial: "[**Mujer/Hombre**] de [**X años**]" - SIEMPRE es PII
- Al inicio de nota clinica describiendo al paciente - SIEMPRE es PII
- Cualquier uso seguido de edad: "mujer de 28 años", "varon de 65a" - SI es PII
- En contexto de descripcion del paciente - SI es PII
- NO: pronombres (el, ella)
- NO: uso puramente gramatical sin descripcion del paciente""",

    "PROFESION": """- Profesion u ocupacion del paciente
- Precedido de "trabaja como", "profesion:", "oficio"
- NO: profesiones del personal sanitario
- NO: terminos genericos ("trabajador", "empleado")""",

    "OTROS_SUJETO_ASISTENCIA": """- Cualquier dato que haga al paciente identificable
- Circunstancias raras o unicas
- Caracteristicas fisicas distintivas
- Relaciones especiales o publicidad
- Patron "X años" o "Xa" donde X es numero: SI es PII (edad)
- Meses textuales con año: "septiembre 2020", "marzo 2024": SI es PII (fecha)
- Habitos personales: exfumador, fumador, bebedor: SI es PII
- Palabras en mayusculas que parecen nombres propios en contexto personal
- NO: datos demograficos sin contexto identificativo
- NO: informacion clinica estandar sin valor personal""",
}

DEFAULT_EVIDENCE_SIGNAL = """- La keyword encaja en algun patron de PII conocido:
  * Fechas: DD/MM/AA, DD.MM.AAAA, meses con año
  * Edades: numero + años, numero + "a" (68a, 72a)
  * Nombres propios en mayusculas
  * Relaciones familiares: hermano, madre, esposa, hijo
  * Telefonos: 9 digitos consecutivos
  * Codigos alfanumericos: letra + numeros (G045, I105)
  * Sexo/genero seguido de edad: "Mujer de 65 años"
- El contexto sugiere que identifica a una persona
- En caso de duda, considerar PII (TRUE)
- NO: palabras claramente genericas sin valor identificativo"""


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
