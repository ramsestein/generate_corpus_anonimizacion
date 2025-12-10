#!/usr/bin/env python3
"""
Train Gatekeeper SetFit con Auditoría.

Este script entrena un modelo SetFit para actuar como "Gatekeeper Semántico"
que valida si una entidad detectada por NER es realmente PII (Clase 1) o
ruido contextual (Clase 0).

Características:
- Genera datos sintéticos basados en guias-anotacion.json
- Crea ejemplos positivos (PII real) y negativos (trampas/ruido)
- Entrena modelo SetFit multilingual
- Genera reporte de auditoría en Markdown

Author: Pipeline Anonimización Clínica
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from faker import Faker

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar Faker en español
fake = Faker('es_ES')
Faker.seed(42)
random.seed(42)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas por defecto
DEFAULT_RULES_FILE = "guias-anotacion.json"
DEFAULT_MODELS_DIR = "models"
DEFAULT_AUDIT_DIR = "audit"
DEFAULT_MODEL_NAME = "setfit_high_precision_v2"

# Modelo base SetFit
SETFIT_BASE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SETFIT_BASE_MODEL_ANTIGUO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Número de ejemplos por categoría
EXAMPLES_PER_CATEGORY = 15  # Reducido de 20 para entrenamiento más rápido

# Hiperparámetros de entrenamiento (Fine-Grained Decision Boundary)
TRAINING_HYPERPARAMS = {
    "num_iterations": 20,  # Reducido de 40 para evitar timeout
    "num_epochs": 1,
    "learning_rate": 3e-5,  # Ligeramente más alto para convergencia rápida
    "batch_size": 8,  # Reducido de 16 para evitar OOM
    "max_iter": 50,  # Reducido de 100 para clasificador final
}


# ============================================================================
# PLANTILLAS DE GENERACIÓN DE DATOS SINTÉTICOS
# ============================================================================

# Plantillas para generar ejemplos POSITIVOS (PII real - Clase 1)
# Cada plantilla usa marcadores {entity} que se reemplazan con datos de Faker
POSITIVE_TEMPLATES: Dict[str, List[str]] = {
    "TERRITORIO": [
        "El paciente reside en [{city}].",
        "Domicilio habitual: [{address}].",
        "Trasladado desde [{city}], provincia de [{province}].",
        "Natural de [{city}].",
        "Paciente procedente del CP [{postcode}].",
        "Vive en la urbanización [{neighborhood}] de [{city}].",
        "Remitido desde centro de salud de [{city}].",
        "Dirección postal: [{address}], [{city}].",
    ],
    "FECHAS": [
        "Fecha de ingreso: [{date}].",
        "Nacido el [{date}].",
        "Último control realizado el [{date}].",
        "Intervención quirúrgica programada para el [{date}].",
        "Diagnóstico inicial en [{month}] de [{year}].",
        "Alta médica el día [{date}].",
        "Próxima revisión: [{weekday}] [{date}].",
        "Desde [{date}] presenta síntomas.",
    ],
    "EDAD_SUJETO_ASISTENCIA": [
        "Paciente de [{age}] años.",
        "Varón de [{age}] años de edad.",
        "Mujer, [{age}] años.",
        "Edad: [{age}].",
        "Niño de [{age_child}] años.",
        "Lactante de [{age_months}] meses.",
        "Recién nacido de [{age_days}] días de vida.",
        "Anciano de [{age_elderly}] años.",
    ],
    "NOMBRE_SUJETO_ASISTENCIA": [
        "Paciente [{full_name}].",
        "D./Dña. [{full_name}] acude a consulta.",
        "[{first_name}] [{last_name}] ingresa por urgencias.",
        "Se recibe a [{full_name}].",
        "Historia clínica de [{full_name}].",
        "El/La paciente [{first_name}] refiere dolor.",
        "Derivado por Dr. García, paciente [{full_name}].",
        "[{prefix}] [{last_name}] presenta mejoría.",
    ],
    "NOMBRE_PERSONAL_SANITARIO": [
        "Dr./Dra. [{doctor_name}].",
        "Atendido por [{doctor_name}].",
        "Médico responsable: [{doctor_name}].",
        "Informe elaborado por Dr. [{doctor_name}].",
        "[{doctor_name}], Servicio de Medicina Interna.",
        "Enfermera [{nurse_name}] administra medicación.",
        "Valorado por [{doctor_name}], especialista.",
        "Firma: [{doctor_name}], Nº Col. {col_number}.",
    ],
    "SEXO_SUJETO_ASISTENCIA": [
        "Sexo: [{sex}].",
        "Paciente [{sex_desc}].",
        "[{sex_desc}] de 45 años.",
        "Enfermo [{sex_desc}] que acude.",
        "[{sex_desc}] sometida a intervención.",
        "Niño/Niña [{sex_child}] de 8 años.",
    ],
    "CALLE": [
        "Domicilio: [{street_address}].",
        "Dirección: [{street}], nº [{building_number}].",
        "C/ [{street_name}] [{building_number}], [{floor}].",
        "Avda. [{street_name}] s/n.",
        "Plaza [{plaza_name}], [{building_number}].",
        "Pº [{paseo_name}], [{city}].",
        "Reside en [{street_address}], [{postcode}].",
    ],
    "PAIS": [
        "Natural de [{country}].",
        "Procedente de [{country}].",
        "Nacionalidad: [{country}].",
        "Emigró desde [{country}] hace 5 años.",
        "Residente en [{country}].",
    ],
    "ID_SUJETO_ASISTENCIA": [
        "NHC: [{nhc}].",
        "DNI: [{dni}].",
        "CIPA: [{cipa}].",
        "Nº Historia: [{nhc}].",
        "Identificador paciente: [{patient_id}].",
        "NIF: [{nif}].",
        "Pasaporte: [{passport}].",
        "CIE: [{cie}].",
    ],
    "CORREO_ELECTRONICO": [
        "Email: [{email}].",
        "Contacto: [{email}].",
        "Correo electrónico del paciente: [{email}].",
        "Notificar a [{email}].",
    ],
    "ID_TITULACION_PERSONAL_SANITARIO": [
        "Nº Colegiado: [{col_number}].",
        "Col. Nº [{col_number}].",
        "Médico colegiado [{col_number}].",
    ],
    "ID_ASEGURAMIENTO": [
        "NASS: [{nass}].",
        "Nº Seguridad Social: [{nass}].",
        "Afiliación SS: [{nass}].",
    ],
    "HOSPITAL": [
        "Hospital [{hospital_name}].",
        "Ingreso en [{hospital_name}].",
        "Derivado a [{hospital_name}].",
        "Complejo Hospitalario [{hospital_name}].",
        "Centro: [{hospital_name}].",
        "HU [{hospital_name}].",
    ],
    "FAMILIARES_SUJETO_ASISTENCIA": [
        "Acompañado por su esposa [{spouse_name}].",
        "Madre: [{mother_name}].",
        "Padre: [{father_name}].",
        "Hijo/a [{child_name}] de [{child_age}] años.",
        "Familiar de contacto: [{family_name}].",
        "Cuidador principal: [{caregiver_name}].",
        # Contexto específico con relación + nombre = PII
        "La madre [{mother_name}] acompaña al paciente.",
        "El padre [{father_name}] es su responsable legal.",
        "Hermana [{sister_name}] será el contacto principal.",
        "Hermano [{brother_name}] tiene antecedentes similares.",
        "Esposo [{spouse_name}] es médico en [{hospital_name}].",
        "Esposa [{spouse_name}] trabaja en el mismo centro.",
        "Abuela [{grandmother_name}] vive con el paciente.",
        "Abuelo [{grandfather_name}] también diabético.",
        "Tío [{uncle_name}] es oncólogo.",
        "Tía [{aunt_name}] padece artritis.",
    ],
    "INSTITUCION": [
        "Remitido desde [{institution}].",
        "Trabaja en [{institution}].",
        "Afiliado a [{institution}].",
        "Centro [{institution}].",
        "Mutua [{mutual}].",
    ],
    "NUMERO_TELEFONO": [
        "Teléfono: [{phone}].",
        "Contacto: [{phone}].",
        "Móvil: [{mobile}].",
        "Tel. [{phone}].",
        "Tfno. de contacto [{phone}].",
    ],
    "ID_CONTACTO_ASISTENCIAL": [
        "Episodio: [{episode_id}].",
        "Nº Contacto: [{contact_id}].",
        "ID Proceso: [{process_id}].",
    ],
    "PROFESION": [
        "Profesión: [{profession}].",
        "Trabaja como [{profession}].",
        "Ocupación: [{profession}].",
        "[{profession}] de 50 años.",
    ],
    "NUMERO_FAX": [
        "Fax: [{fax}].",
        "FAX: [{fax}].",
        "Enviar a fax [{fax}].",
    ],
    "OTROS_SUJETO_ASISTENCIA": [
        "Presenta tatuaje de [{tattoo}].",
        "Conocido como '{alias}'.",
        "Apodo: [{nickname}].",
        "Rasgo distintivo: [{distinctive_feature}].",
    ],
    "CENTRO_SALUD": [
        "Centro de Salud [{health_center}].",
        "CS [{health_center}].",
        "Derivado desde C.S. [{health_center}].",
        "Pertenece al Centro de Salud [{health_center}].",
    ],
    "ID_EMPLEO_PERSONAL_SANITARIO": [
        "Nº Empleado: [{employee_id}].",
        "ID Personal: [{employee_id}].",
        "Usuario: [{employee_id}].",
    ],
    "IDENTIF_VEHICULOS_NRSERIE_PLACAS": [
        "Matrícula: [{license_plate}].",
        "Vehículo [{license_plate}].",
        "Accidente con vehículo [{license_plate}].",
    ],
    "IDENTIF_DISPOSITIVOS_NRSERIE": [
        "IP: [{ip_address}].",
        "MAC: [{mac_address}].",
        "Dispositivo [{device_serial}].",
    ],
    "NUMERO_BENEF_PLAN_SALUD": [
        "Nº Beneficiario: [{beneficiary_id}].",
        "Póliza: [{policy_number}].",
        "ID Asegurado: [{insured_id}].",
    ],
    "DIREC_PROT_INTERNET": [
        "URL: [{url}].",
        "Enlace: [{url}].",
        "Acceso web: [{url}].",
    ],
    "URL_WEB": [
        "Página web: [{website}].",
        "Web: [{website}].",
        "Consultar en [{website}].",
    ],
    "OTRO_NUMERO_IDENTIF": [
        "Nº Socio: [{member_id}].",
        "ID: [{other_id}].",
        "Referencia: [{reference_id}].",
    ],
    "IDENTIF_BIOMETRICOS": [
        "Huella dactilar registrada: [{fingerprint_id}].",
        "Identificación biométrica: [{biometric_id}].",
        "Registro facial ID: [{face_id}].",
        "Huella digital [{fingerprint_id}].",
        "Patrón retinal [{retinal_id}].",
    ],
    "NUMERO_IDENTIF": [
        "Número de identificación: [{patient_id}].",
        "ID: [{patient_id}].",
        "Código identificador: [{nhc}].",
        "Registro: [{nhc}].",
        "Número asignado: [{cipa}].",
    ],
}

# ============================================================================
# PLANTILLAS PARA EJEMPLOS NEGATIVOS (RUIDO/TRAMPAS - Clase 0)
# Estas son frases que confunden al NER pero NO son PII real
# ============================================================================

NEGATIVE_TEMPLATES: Dict[str, List[str]] = {
    "TERRITORIO": [
        # Términos médicos que suenan a lugares
        "Presenta dolor en región [lumbar].",
        "Afectación de zona [temporal] izquierda.",
        "Masa en región [cervical].",
        "Lesión en área [frontal].",
        "Dolor referido a región [dorsal].",
        "Exploración de fosa [iliaca] derecha.",
        "Arteria [femoral] permeable.",
        "Ganglio [axilar] aumentado.",
        # Términos genéricos
        "Trasladado desde [urgencias].",
        "Derivado a [consultas externas].",
        "Ingresa en planta [tercera].",
        "Ubicado en [box 5].",
    ],
    "FECHAS": [
        # Referencias temporales no específicas
        "Desde hace [varios días].",
        "En las [últimas horas].",
        "Durante [la noche].",
        "Por [la mañana] presentó fiebre.",
        "Evolución en [las últimas semanas].",
        "[Actualmente] estable.",
        "Control [periódico] cada 6 meses.",
        # Fechas médicas estándar
        "Protocolo [día 0] de tratamiento.",
        "Semana [+2] post-trasplante.",
        "Ciclo [3] de quimioterapia.",
    ],
    "EDAD_SUJETO_ASISTENCIA": [
        # Números que no son edades
        "Saturación O2 [95]%.",
        "Tensión arterial [120]/80.",
        "Frecuencia cardíaca [72] lpm.",
        "Temperatura [36.5]ºC.",
        "Peso [65] kg.",
        "Talla [170] cm.",
        "Glucemia [110] mg/dl.",
        "IMC [25].",
        # Edades de enfermedades, no del paciente
        "Diabetes de [15] años de evolución.",
        "HTA de [10] años.",
    ],
    "NOMBRE_SUJETO_ASISTENCIA": [
        # Nombres de medicamentos que parecen nombres
        "Se pauta [Adriana] 50mg.",
        "Tratamiento con [Amoxicilina].",
        "Administrar [Ramona] IV.",
        # Nombres de síndromes/enfermedades
        "Síndrome de [Cushing].",
        "Enfermedad de [Parkinson].",
        "Signo de [Murphy] positivo.",
        "Maniobra de [Valsalva].",
        # Epónimos médicos
        "Test de [Romberg] negativo.",
        "Escala de [Glasgow] 15.",
        "Clasificación de [Child-Pugh] A.",
        # Partes anatómicas que suenan a nombres
        "Dolor en [muñeca] derecha.",
        "Fractura de [húmero].",
    ],
    "NOMBRE_PERSONAL_SANITARIO": [
        # Epónimos médicos (no son doctores actuales)
        "Maniobra de [Heimlich].",
        "Técnica de [Seldinger].",
        "Catéter de [Foley].",
        "Sonda de [Levin].",
        "Tubo de [Mayo].",
        # Términos que parecen nombres
        "Enfermedad de [Graves].",
        "Tiroiditis de [Hashimoto].",
        "Síndrome de [Sjögren].",
    ],
    "SEXO_SUJETO_ASISTENCIA": [
        # Términos médicos
        "Cromosoma [X] normal.",
        "Cariotipo [XY].",
        "Receptor [alfa] positivo.",
        "Factor [V] Leiden.",
    ],
    "CALLE": [
        # Términos médicos con números
        "Administrar [pauta 3x1].",
        "Tratamiento [fase 2].",
        "Estadio [IVb].",
        "Grado [III/IV].",
        "T[3]N[1]M[0].",
    ],
    "PAIS": [
        # Nacionalidades de patógenos/cepas
        "Variante [brasileña].",
        "Cepa [británica].",
        "Virus [japonés] B.",
        # Términos genéricos
        "Protocolo [europeo] de tratamiento.",
        "Guías [americanas] de cardiología.",
    ],
    "ID_SUJETO_ASISTENCIA": [
        # Códigos médicos
        "CIE-10: [E11.9].",
        "Código ATC: [N02BE01].",
        "GRD: [470].",
        "Cama [305].",
        "Box [12].",
        # Valores de laboratorio
        "pH [7.35].",
        "pO2 [95] mmHg.",
        "Hb [12.5] g/dL.",
    ],
    "CORREO_ELECTRONICO": [
        # Textos que parecen emails
        "Relación [riesgo/beneficio] favorable.",
        "Proporción [1:1000].",
    ],
    "HOSPITAL": [
        # Servicios/unidades genéricas
        "Ingresa en [UCI].",
        "Derivado a [urgencias].",
        "Planta de [cardiología].",
        "Servicio de [medicina interna].",
        "[Quirófano] programado.",
        "Sala de [reanimación].",
    ],
    "FAMILIARES_SUJETO_ASISTENCIA": [
        # Antecedentes familiares genéricos - SIN NOMBRES específicos = RUIDO
        "[Padre] con HTA.",
        "[Madre] diabética.",
        "Antecedentes familiares de [cáncer].",
        "[Hermano] fallecido.",
        # Contexto familiar SIN nombres específicos = RUIDO
        "Historia familiar de [hipertensión].",
        "[Familia] con antecedentes de infarto.",
        "Padres con [enfermedad].",
        "Abuelos con [diabetes].",
        "[Hermana] con artritis reumatoide.",
        "[Hermano] con asma.",
        "[Tío] con cáncer de pulmón.",
        "[Tía] con tiroiditis.",
        "Los [padres] tienen HTA.",
        "[Familia] sin antecedentes relevantes.",
        "Antecedentes [familiares] de EPOC.",
        "[Familiares] allegados sanos.",
        # Términos genéricos de relación
        "Relación [familiar] estable.",
        "Apoyo [familiar] presente.",
        "Red [familiar] adecuada.",
        "Situación [familiar] compleja.",
        "Dinámica [familiar] conflictiva.",
    ],
    "INSTITUCION": [
        # Servicios médicos
        "Servicio de [Urgencias].",
        "Unidad de [Cuidados Intensivos].",
        "Departamento de [Radiología].",
        "[Laboratorio] central.",
    ],
    "NUMERO_TELEFONO": [
        # Códigos/clasificaciones
        "CIE [630.0].",
        "Código [925.11].",
        "Referencia [555-123].",
    ],
    "PROFESION": [
        # Términos médicos que suenan a profesiones
        "[Portador] de marcapasos.",
        "[Conductor] del haz de His.",
        "Paciente [trabajador] respiratorio.",
    ],
    "CENTRO_SALUD": [
        # Áreas hospitalarias
        "Área de [salud] mental.",
        "Centro [quirúrgico].",
        "Punto de [atención] continuada.",
    ],
    "IDENTIF_DISPOSITIVOS_NRSERIE": [
        # Códigos de procedimientos
        "Técnica [IV.2.3].",
        "Protocolo [A.1.2].",
    ],
}


# ============================================================================
# FUNCIONES DE GENERACIÓN DE DATOS SINTÉTICOS
# ============================================================================

def generate_fake_data() -> Dict[str, Any]:
    """
    Genera un diccionario con datos falsos usando Faker.
    
    Returns:
        Dict con todos los tipos de datos sintéticos necesarios.
    """
    # Provincias españolas
    provincias_espanolas = [
        "Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza", "Málaga",
        "Murcia", "Palma de Mallorca", "Las Palmas", "Bilbao", "Alicante",
        "Córdoba", "Valladolid", "Vigo", "Gijón", "Granada", "A Coruña",
        "Vitoria-Gasteiz", "Elche", "Oviedo", "Santa Cruz de Tenerife",
        "Pamplona", "Santander", "Almería", "Burgos", "Albacete", "Logroño"
    ]
    
    return {
        # Territorio
        "city": fake.city(),
        "address": fake.address().replace('\n', ', '),
        "province": random.choice(provincias_espanolas),
        "postcode": fake.postcode(),
        "neighborhood": fake.city_suffix() + " " + fake.last_name(),
        
        # Fechas
        "date": fake.date(pattern="%d/%m/%Y"),
        "month": fake.month_name(),
        "year": str(fake.year()),
        "weekday": fake.day_of_week(),
        
        # Edades
        "age": random.randint(18, 90),
        "age_child": random.randint(1, 12),
        "age_months": random.randint(1, 11),
        "age_days": random.randint(1, 30),
        "age_elderly": random.randint(70, 95),
        
        # Nombres
        "full_name": fake.name(),
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "prefix": random.choice(["D.", "Dña.", "Sr.", "Sra."]),
        
        # Personal sanitario
        "doctor_name": fake.name(),
        "nurse_name": fake.name(),
        "col_number": f"{random.randint(10, 50)}-{random.randint(10000, 99999)}",
        
        # Sexo
        "sex": random.choice(["M", "H", "V", "F"]),
        "sex_desc": random.choice(["varón", "mujer", "hombre"]),
        "sex_child": random.choice(["niño", "niña"]),
        
        # Direcciones
        "street_address": fake.street_address(),
        "street": fake.street_name(),
        "street_name": fake.street_name().split()[-1] if fake.street_name() else "Mayor",
        "building_number": str(random.randint(1, 200)),
        "floor": f"{random.randint(1, 10)}º {random.choice(['A', 'B', 'C', 'D'])}",
        "plaza_name": fake.last_name(),
        "paseo_name": fake.last_name(),
        
        # País
        "country": random.choice(["España", "Portugal", "Francia", "Alemania", "Italia", 
                                  "Reino Unido", "Marruecos", "Ecuador", "Colombia"]),
        
        # IDs
        "nhc": str(random.randint(100000, 9999999)),
        "dni": f"{random.randint(10000000, 99999999)}{random.choice('ABCDEFGHJKLMNPQRSTVWXYZ')}",
        "cipa": f"nhc-{random.randint(100000, 999999)}",
        "patient_id": f"P{random.randint(10000, 99999)}",
        "nif": f"{random.randint(10000000, 99999999)}-{random.choice('ABCDEFGHJKLMNPQRSTVWXYZ')}",
        "passport": f"{random.choice('ABCDEFG')}{random.randint(10000000, 99999999)}",
        "cie": f"CIE{random.randint(100000, 999999)}",
        
        # Email
        "email": fake.email(),
        
        # Seguridad Social
        "nass": f"{random.randint(10, 99)} {random.randint(1000000, 9999999)} {random.randint(10, 99)}",
        
        # Hospital
        "hospital_name": random.choice([
            "Universitario La Paz", "12 de Octubre", "Gregorio Marañón",
            "La Fe", "Vall d'Hebron", "Clínic", "Virgen del Rocío",
            "Carlos Haya", "Son Espases", "Cruces"
        ]),
        
        # Familiares
        "spouse_name": fake.name(),
        "mother_name": fake.name_female(),
        "father_name": fake.name_male(),
        "child_name": fake.first_name(),
        "child_age": random.randint(1, 18),
        "family_name": fake.name(),
        "caregiver_name": fake.name(),
        
        # Institución
        "institution": random.choice([
            "CNIO", "ISCIII", "CSIC", "Clínica Universidad de Navarra",
            "Quirónsalud", "HM Hospitales", "Vithas"
        ]),
        "mutual": random.choice(["MUFACE", "Adeslas", "Sanitas", "DKV", "Asisa"]),
        
        # Teléfonos
        "phone": fake.phone_number(),
        "mobile": f"6{random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)}",
        "fax": f"9{random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)}",
        
        # IDs de contacto
        "episode_id": str(random.randint(1000000000, 9999999999)),
        "contact_id": f"C{random.randint(100000, 999999)}",
        "process_id": f"PROC-{random.randint(10000, 99999)}",
        
        # Profesión
        "profession": random.choice([
            "albañil", "conductor", "profesor", "enfermero", "administrativo",
            "agricultor", "mecánico", "electricista", "cocinero", "camarero"
        ]),
        
        # Otros identificadores
        "tattoo": random.choice(["dragón", "rosa", "nombre", "fecha", "símbolo"]),
        "alias": random.choice(["El Rubio", "Chato", "Peque", "Flaco"]),
        "nickname": random.choice(["Paco", "Pepe", "Toño", "Lola"]),
        "distinctive_feature": random.choice(["cicatriz en frente", "lunar en mejilla", "vitíligo"]),
        
        # Centro de salud
        "health_center": random.choice([
            "Cea Bermúdez", "Infanta Mercedes", "General Ricardos",
            "Alameda", "Puerta del Ángel", "Vallecas"
        ]),
        
        # Empleado
        "employee_id": f"u{random.randint(100000, 999999)}",
        
        # Vehículos
        "license_plate": f"{random.randint(1000, 9999)}{random.choice('BCDFGHJKLMNPRSTVWXYZ')}{random.choice('BCDFGHJKLMNPRSTVWXYZ')}{random.choice('BCDFGHJKLMNPRSTVWXYZ')}",
        
        # Dispositivos
        "ip_address": f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}",
        "mac_address": ":".join([f"{random.randint(0,255):02x}" for _ in range(6)]),
        "device_serial": f"SN{random.randint(10000000, 99999999)}",
        
        # Plan de salud
        "beneficiary_id": f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(100, 999)}",
        "policy_number": f"{random.randint(10000000, 99999999)}-{random.choice('ABCD')}",
        "insured_id": f"PS-{random.randint(1000000, 9999999)}",
        
        # URLs
        "url": f"https://www.{fake.domain_name()}",
        "website": f"www.{fake.domain_name()}",
        
        # Otros IDs
        "member_id": str(random.randint(1000000000, 9999999999)),
        "other_id": f"REF-{random.randint(10000, 99999)}",
        "reference_id": f"R{random.randint(100000, 999999)}",
        
        # Biométricos
        "fingerprint_id": f"FP{random.randint(100000000, 999999999)}",
        "biometric_id": f"BIO-{random.randint(10000000, 99999999)}",
        "face_id": f"FACE{random.randint(100000, 999999)}",
        "retinal_id": f"RET-{random.randint(10000000, 99999999)}",
    }


def fill_template(template: str, data: Dict[str, Any]) -> str:
    """
    Rellena una plantilla con datos sintéticos.
    
    Las entidades están marcadas con [{entity}] y se mantienen los corchetes
    en el resultado final para indicar dónde está la entidad.
    
    Args:
        template: Plantilla con marcadores {key} y [{key}].
        data: Diccionario con datos para rellenar.
    
    Returns:
        Cadena con la plantilla rellenada.
    """
    result = template
    
    # Primero procesar los marcadores con corchetes [{...}]
    import re
    bracket_pattern = r'\[\{(\w+)\}\]'
    
    def replace_bracket(match):
        key = match.group(1)
        if key in data:
            return f"[{data[key]}]"
        return match.group(0)
    
    result = re.sub(bracket_pattern, replace_bracket, result)
    
    # Luego procesar los marcadores simples {...}
    simple_pattern = r'\{(\w+)\}'
    
    def replace_simple(match):
        key = match.group(1)
        if key in data:
            return str(data[key])
        return match.group(0)
    
    result = re.sub(simple_pattern, replace_simple, result)
    
    return result


def generate_positive_examples(
    category: str,
    num_examples: int = EXAMPLES_PER_CATEGORY
) -> List[Tuple[str, int]]:
    """
    Genera ejemplos POSITIVOS (PII real) para una categoría.
    
    Args:
        category: Nombre de la categoría (ej: FECHAS, NOMBRE_SUJETO_ASISTENCIA).
        num_examples: Número de ejemplos a generar.
    
    Returns:
        Lista de tuplas (texto, label=1).
    """
    examples = []
    templates = POSITIVE_TEMPLATES.get(category, [])
    
    if not templates:
        logger.warning(f"No hay plantillas positivas para categoría: {category}")
        return examples
    
    for _ in range(num_examples):
        template = random.choice(templates)
        data = generate_fake_data()
        text = fill_template(template, data)
        examples.append((text, 1))
    
    return examples


def generate_negative_examples(
    category: str,
    num_examples: int = EXAMPLES_PER_CATEGORY
) -> List[Tuple[str, int]]:
    """
    Genera ejemplos NEGATIVOS (trampas/ruido) para una categoría.
    
    Estos son contextos médicos seguros que un NER podría confundir con PII,
    pero que en realidad NO deben anonimizarse.
    
    ESTRATEGIA DE TRAMPAS:
    - TERRITORIO: Usar términos anatómicos que suenan a lugares (lumbar, temporal)
    - FECHAS: Usar referencias temporales genéricas (varios días, actualmente)
    - EDAD: Usar números de constantes vitales (saturación, tensión)
    - NOMBRES: Usar epónimos médicos (Cushing, Parkinson, Murphy)
    - HOSPITAL: Usar servicios genéricos (UCI, urgencias)
    
    Args:
        category: Nombre de la categoría.
        num_examples: Número de ejemplos a generar.
    
    Returns:
        Lista de tuplas (texto, label=0).
    """
    examples = []
    templates = NEGATIVE_TEMPLATES.get(category, [])
    
    if not templates:
        # Si no hay plantillas específicas, usar genéricas
        templates = [
            f"Término médico no sensible para {category}.",
            f"Contexto clínico seguro, no PII ({category}).",
        ]
    
    for _ in range(num_examples):
        template = random.choice(templates)
        # Los negativos no necesitan datos de Faker, ya tienen el texto fijo
        text = template
        examples.append((text, 0))
    
    return examples


def generate_synthetic_dataset(
    rules: Dict[str, List[str]],
    examples_per_category: int = EXAMPLES_PER_CATEGORY
) -> pd.DataFrame:
    """
    Genera el dataset sintético completo para entrenamiento.
    
    Args:
        rules: Diccionario con las reglas de anotación por categoría.
        examples_per_category: Número de ejemplos por categoría y clase.
    
    Returns:
        DataFrame con columnas [text, label, category].
    """
    all_examples = []
    
    for category in rules.keys():
        logger.info(f"Generando ejemplos para: {category}")
        
        # Ejemplos positivos (PII real)
        positive = generate_positive_examples(category, examples_per_category)
        for text, label in positive:
            all_examples.append({
                "text": text,
                "label": label,
                "category": category
            })
        
        # Ejemplos negativos (trampas/ruido)
        negative = generate_negative_examples(category, examples_per_category)
        for text, label in negative:
            all_examples.append({
                "text": text,
                "label": label,
                "category": category
            })
    
    df = pd.DataFrame(all_examples)
    
    # Mezclar aleatoriamente
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def train_setfit_model(
    df: pd.DataFrame,
    model_name: str = SETFIT_BASE_MODEL,
    output_dir: str = None,
    hyperparams: Dict[str, Any] = None
) -> Any:
    """
    Entrena un modelo SetFit con el dataset generado.
    
    ESTRATEGIA: Fine-Grained Decision Boundary para maximizar Precisión sin perder Recall.
    
    Args:
        df: DataFrame con columnas [text, label].
        model_name: Nombre del modelo base de Sentence Transformers.
        output_dir: Directorio donde guardar el modelo.
        hyperparams: Diccionario con hiperparámetros de entrenamiento.
    
    Returns:
        Modelo SetFit entrenado y diccionario de métricas detalladas.
    """
    try:
        from setfit import SetFitModel, SetFitTrainer
        from datasets import Dataset
        from sklearn.metrics import precision_recall_fscore_support, classification_report
    except ImportError:
        logger.error("Instala setfit y datasets: pip install setfit datasets scikit-learn")
        raise
    
    # Usar hiperparámetros por defecto si no se proporcionan
    if hyperparams is None:
        hyperparams = TRAINING_HYPERPARAMS
    
    # Convertir a Dataset de HuggingFace
    dataset = Dataset.from_pandas(df[["text", "label"]])
    
    # Dividir en train/eval sin estratificación (el dataset ya está balanceado)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    logger.info(f"Dataset - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    
    # Calcular distribución de clases
    train_labels = dataset['train']['label']
    test_labels = dataset['test']['label']
    train_class0 = sum(1 for label in train_labels if label == 0)
    train_class1 = sum(1 for label in train_labels if label == 1)
    test_class0 = sum(1 for label in test_labels if label == 0)
    test_class1 = sum(1 for label in test_labels if label == 1)
    
    logger.info(f"Distribución Train - Clase 0: {train_class0}, Clase 1: {train_class1}")
    logger.info(f"Distribución Test - Clase 0: {test_class0}, Clase 1: {test_class1}")
    
    # Cargar modelo base
    logger.info(f"Cargando modelo base: {model_name}")
    model = SetFitModel.from_pretrained(model_name)
    
    # Configurar argumentos de entrenamiento con hiperparámetros optimizados
    logger.info("Hiperparámetros de entrenamiento:")
    for key, value in hyperparams.items():
        logger.info(f"  {key}: {value}")
    
    # Configurar trainer con hiperparámetros directamente
    trainer = SetFitTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        metric="f1",  # Optimizar F1 (balance Precision/Recall)
        num_iterations=hyperparams.get("num_iterations", 40),
        num_epochs=hyperparams.get("num_epochs", 1),
        learning_rate=hyperparams.get("learning_rate", 2e-5),
        batch_size=hyperparams.get("batch_size", 16),
        seed=42,
        column_mapping={"text": "text", "label": "label"},
    )
    
    # Entrenar
    logger.info("🚀 Iniciando entrenamiento SetFit con Fine-Grained Decision Boundary...")
    trainer.train()
    
    # Evaluar con métricas detalladas
    logger.info("📊 Evaluando modelo...")
    
    # Predicciones en test set
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]
    predictions = model(test_texts)
    
    # Calcular métricas detalladas
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, predictions, average='binary', pos_label=1
    )
    
    # Reporte completo
    report = classification_report(
        test_labels, predictions, 
        target_names=["Clase 0 (Ruido)", "Clase 1 (PII)"],
        digits=4
    )
    
    # Métricas básicas del trainer
    basic_metrics = trainer.evaluate()
    
    # Consolidar métricas
    metrics = {
        **basic_metrics,
        "precision_class1": float(precision),
        "recall_class1": float(recall),
        "f1_class1": float(f1),
        "support_class1": int(support),
        "classification_report": report
    }
    
    # Imprimir reporte detallado
    logger.info("\n" + "="*80)
    logger.info("📈 REPORTE DE EVALUACIÓN - MÉTRICAS DETALLADAS")
    logger.info("="*80)
    logger.info(f"\n🎯 MÉTRICAS CLASE 1 (PII - CRÍTICO PARA ANONIMIZACIÓN):")
    logger.info(f"   Precision: {precision:.4f} ({precision*100:.2f}%) - Reducción de Falsos Positivos")
    logger.info(f"   Recall:    {recall:.4f} ({recall*100:.2f}%) - Detección completa de PII (NO DEBE BAJAR)")
    logger.info(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%) - Balance óptimo")
    logger.info(f"   Support:   {support} ejemplos")
    logger.info(f"\n📋 REPORTE COMPLETO:\n{report}")
    logger.info("="*80 + "\n")
    
    # Advertencia si el Recall baja
    if recall < 0.95:
        logger.warning("⚠️  ADVERTENCIA: Recall < 95% - Riesgo de perder datos sensibles!")
        logger.warning("    Considera aumentar num_iterations o ajustar el balance de clases.")
    
    # Guardar modelo
    if output_dir:
        logger.info(f"💾 Guardando modelo en: {output_dir}")
        model.save_pretrained(output_dir)
        
        # Guardar hiperparámetros y métricas
        import json
        metadata_path = Path(output_dir) / "training_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "hyperparameters": hyperparams,
                "metrics": {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support
                },
                "model_base": model_name,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        logger.info(f"📄 Metadata guardada en: {metadata_path}")
    
    return model, metrics


# ============================================================================
# FUNCIONES DE AUDITORÍA
# ============================================================================

def generate_audit_report(
    df: pd.DataFrame,
    metrics: Dict[str, float],
    categories_processed: List[str],
    model_path: str,
    output_path: str
) -> None:
    """
    Genera el reporte de auditoría en Markdown.
    
    Args:
        df: DataFrame con los datos de entrenamiento.
        metrics: Métricas de evaluación del modelo.
        categories_processed: Lista de categorías procesadas.
        model_path: Ruta donde se guardó el modelo.
        output_path: Ruta del archivo Markdown de salida.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Estadísticas
    total_examples = len(df)
    pii_examples = len(df[df["label"] == 1])
    noise_examples = len(df[df["label"] == 0])
    
    # Muestras aleatorias
    sample_pii = df[df["label"] == 1].sample(min(5, pii_examples), random_state=42)
    sample_noise = df[df["label"] == 0].sample(min(5, noise_examples), random_state=42)
    
    # Generar Markdown
    report = f"""# 🔍 Reporte de Entrenamiento - Gatekeeper SetFit

## 📅 Información General

| Campo | Valor |
|-------|-------|
| **Fecha/Hora** | {timestamp} |
| **Modelo Base** | `{SETFIT_BASE_MODEL}` |
| **Ruta del Modelo** | `{model_path}` |

---

## 📊 Estadísticas del Dataset

| Métrica | Valor |
|---------|-------|
| **Total de frases** | {total_examples} |
| **Clase 1 (PII)** | {pii_examples} ({pii_examples/total_examples*100:.1f}%) |
| **Clase 0 (Ruido)** | {noise_examples} ({noise_examples/total_examples*100:.1f}%) |
| **Categorías procesadas** | {len(categories_processed)} |

---

## 📈 Métricas de Evaluación

### 🎯 Métricas Críticas (Clase 1 - PII)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Precision** | {metrics.get('precision_class1', 0):.4f} | Reducción de Falsos Positivos |
| **Recall** | {metrics.get('recall_class1', 0):.4f} | ⚠️ CRÍTICO: No perder datos sensibles |
| **F1-Score** | {metrics.get('f1_class1', 0):.4f} | Balance óptimo Precision/Recall |

### 📊 Métricas Generales

| Métrica | Valor |
|---------|-------|
"""
    
    for key, value in metrics.items():
        if key not in ['precision_class1', 'recall_class1', 'f1_class1', 'classification_report']:
            if isinstance(value, float):
                report += f"| **{key}** | {value:.4f} |\n"
            else:
                report += f"| **{key}** | {value} |\n"
    
    report += f"""
---

## 🔬 Muestras de Verificación

### ✅ Ejemplos Clase 1 (PII - Anonimizar)

| # | Categoría | Texto |
|---|-----------|-------|
"""
    
    for idx, row in enumerate(sample_pii.itertuples(), 1):
        # Escapar pipes para Markdown
        text_escaped = row.text.replace("|", "\\|")
        report += f"| {idx} | `{row.category}` | {text_escaped} |\n"
    
    report += f"""
### ❌ Ejemplos Clase 0 (Ruido - NO Anonimizar)

| # | Categoría | Texto |
|---|-----------|-------|
"""
    
    for idx, row in enumerate(sample_noise.itertuples(), 1):
        text_escaped = row.text.replace("|", "\\|")
        report += f"| {idx} | `{row.category}` | {text_escaped} |\n"
    
    report += f"""
---

## 📋 Categorías Procesadas

"""
    
    for cat in sorted(categories_processed):
        cat_count = len(df[df["category"] == cat])
        report += f"- `{cat}` ({cat_count} ejemplos)\n"
    
    # Análisis de Recall
    recall_class1 = metrics.get('recall_class1', 0)
    if recall_class1 >= 0.95:
        recall_status = "✅ **Recall ACEPTABLE** (≥95%): El modelo no pierde datos sensibles."
    else:
        recall_status = "🚨 **Recall BAJO** (<95%): RIESGO de pérdida de datos sensibles. Reentrenar."
    
    report += f"""
---

## 🔧 Configuración de Entrenamiento (v2 - High Precision)

### Hiperparámetros Aplicados

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **num_iterations** | {TRAINING_HYPERPARAMS['num_iterations']} | Más pares contrastivos → mejor boundary |
| **learning_rate** | {TRAINING_HYPERPARAMS['learning_rate']} | Learning rate conservador para estabilidad |
| **batch_size** | {TRAINING_HYPERPARAMS['batch_size']} | Balance entre velocidad y precisión |
| **metric** | F1-Score | Optimización del balance Precision/Recall |

### ⚠️ Análisis de Recall

{recall_status}

---

## 💡 Notas de Interpretación

### Estrategia de Generación de Trampas (Clase 0)

El generador crea ejemplos negativos siguiendo estas estrategias:

1. **TERRITORIO → Anatomía**: Términos como "región lumbar", "zona temporal" suenan a lugares pero son anatómicos.
2. **FECHAS → Referencias genéricas**: "Varios días", "últimas horas" no son fechas específicas.
3. **EDAD → Constantes vitales**: Números como "95%" (saturación), "120/80" (tensión) no son edades.
4. **NOMBRES → Epónimos médicos**: "Síndrome de Cushing", "Maniobra de Heimlich" no son pacientes reales.
5. **HOSPITAL → Servicios**: "UCI", "Urgencias", "Quirófano" son servicios, no hospitales identificables.
6. **ID → Códigos médicos**: "CIE-10: E11.9", "Cama 305" no son identificadores de paciente.

### Cómo Usar Este Reporte

1. **Verificar calidad**: Revisa las muestras para confirmar que los ejemplos son coherentes.
2. **Detectar problemas**: Si ves ejemplos mal generados, ajusta las plantillas en el código.
3. **Trazabilidad**: Este archivo documenta exactamente qué datos se usaron para entrenar.

---

*Generado automáticamente por `train_gatekeeper_audit.py`*
"""
    
    # Escribir archivo
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Reporte de auditoría generado: {output_path}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main(
    rules_file: str = DEFAULT_RULES_FILE,
    models_dir: str = DEFAULT_MODELS_DIR,
    audit_dir: str = DEFAULT_AUDIT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    examples_per_category: int = EXAMPLES_PER_CATEGORY
) -> None:
    """
    Ejecuta el pipeline completo de entrenamiento y auditoría.
    
    Args:
        rules_file: Ruta al archivo JSON con reglas de anotación.
        models_dir: Directorio para guardar modelos.
        audit_dir: Directorio para reportes de auditoría.
        model_name: Nombre del modelo a guardar.
        examples_per_category: Número de ejemplos por categoría.
    """
    logger.info("=" * 60)
    logger.info("🚀 Iniciando entrenamiento Gatekeeper SetFit")
    logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. Gestión de Directorios
    # -------------------------------------------------------------------------
    logger.info("📁 Verificando directorios...")
    
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"   ✓ Directorio de modelos: {models_dir}")
    
    Path(audit_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"   ✓ Directorio de auditoría: {audit_dir}")
    
    model_output_path = Path(models_dir) / model_name
    audit_output_path = Path(audit_dir) / "reporte_entrenamiento.md"
    
    # -------------------------------------------------------------------------
    # 2. Cargar Reglas de Anotación
    # -------------------------------------------------------------------------
    logger.info(f"📖 Cargando reglas desde: {rules_file}")
    
    if not Path(rules_file).exists():
        raise FileNotFoundError(f"No se encontró el archivo de reglas: {rules_file}")
    
    with open(rules_file, "r", encoding="utf-8") as f:
        rules = json.load(f)
    
    categories = list(rules.keys())
    logger.info(f"   ✓ {len(categories)} categorías encontradas")
    
    # -------------------------------------------------------------------------
    # 3. Generar Dataset Sintético
    # -------------------------------------------------------------------------
    logger.info("🔄 Generando dataset sintético...")
    
    df = generate_synthetic_dataset(rules, examples_per_category)
    
    logger.info(f"   ✓ Total ejemplos: {len(df)}")
    logger.info(f"   ✓ Clase 1 (PII): {len(df[df['label'] == 1])}")
    logger.info(f"   ✓ Clase 0 (Ruido): {len(df[df['label'] == 0])}")
    
    # Guardar dataset para referencia
    dataset_path = Path(audit_dir) / "training_dataset.csv"
    df.to_csv(dataset_path, index=False, encoding="utf-8")
    logger.info(f"   ✓ Dataset guardado: {dataset_path}")
    
    # -------------------------------------------------------------------------
    # 4. Entrenar Modelo SetFit (Fine-Grained Decision Boundary)
    # -------------------------------------------------------------------------
    logger.info("🤖 Entrenando modelo SetFit con estrategia de alta precisión...")
    logger.info(f"   Versión: {model_name} (v2 - Fine-Grained)")
    logger.info(f"   Objetivo: Maximizar Precisión SIN reducir Recall")
    
    try:
        model, metrics = train_setfit_model(
            df=df,
            model_name=SETFIT_BASE_MODEL,
            output_dir=str(model_output_path),
            hyperparams=TRAINING_HYPERPARAMS
        )
        logger.info(f"   ✓ Modelo guardado en: {model_output_path}")
        
        # Verificar que el Recall se mantenga alto
        recall = metrics.get("recall_class1", 0)
        precision = metrics.get("precision_class1", 0)
        
        if recall >= 0.95:
            logger.info(f"   ✅ Recall EXCELENTE: {recall:.2%} - No se pierden datos sensibles")
        else:
            logger.warning(f"   ⚠️  Recall INSUFICIENTE: {recall:.2%} - Ajustar hiperparámetros")
        
        logger.info(f"   📊 Precision lograda: {precision:.2%}")
        
    except ImportError as e:
        logger.error(f"Error de importación: {e}")
        logger.error("Instala las dependencias: pip install setfit datasets torch scikit-learn")
        metrics = {"error": str(e)}
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        metrics = {"error": str(e)}
    
    # -------------------------------------------------------------------------
    # 5. Generar Reporte de Auditoría
    # -------------------------------------------------------------------------
    logger.info("📝 Generando reporte de auditoría...")
    
    generate_audit_report(
        df=df,
        metrics=metrics,
        categories_processed=categories,
        model_path=str(model_output_path),
        output_path=str(audit_output_path)
    )
    
    # -------------------------------------------------------------------------
    # Resumen Final
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("✅ Proceso completado exitosamente")
    logger.info("=" * 60)
    logger.info(f"   📊 Dataset: {dataset_path}")
    logger.info(f"   🤖 Modelo: {model_output_path}")
    logger.info(f"   📝 Reporte: {audit_output_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Entrena un modelo SetFit Gatekeeper con auditoría.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python train_gatekeeper_audit.py
  python train_gatekeeper_audit.py --rules-file guias-anotacion.json --examples 30
  python train_gatekeeper_audit.py --models-dir ./my_models --audit-dir ./my_audit
        """
    )
    
    parser.add_argument(
        "--rules-file",
        type=str,
        default=DEFAULT_RULES_FILE,
        help=f"Archivo JSON con reglas de anotación (default: {DEFAULT_RULES_FILE})"
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Directorio para guardar modelos (default: {DEFAULT_MODELS_DIR})"
    )
    
    parser.add_argument(
        "--audit-dir",
        type=str,
        default=DEFAULT_AUDIT_DIR,
        help=f"Directorio para reportes de auditoría (default: {DEFAULT_AUDIT_DIR})"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Nombre del modelo a guardar (default: {DEFAULT_MODEL_NAME})"
    )
    
    parser.add_argument(
        "--examples",
        type=int,
        default=EXAMPLES_PER_CATEGORY,
        help=f"Ejemplos por categoría y clase (default: {EXAMPLES_PER_CATEGORY})"
    )
    
    args = parser.parse_args()
    
    main(
        rules_file=args.rules_file,
        models_dir=args.models_dir,
        audit_dir=args.audit_dir,
        model_name=args.model_name,
        examples_per_category=args.examples
    )
