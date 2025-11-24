#!/usr/bin/env python3
"""
PASO 2 DEL PIPELINE: Verificador iterativo de etiquetas faltantes
Verifica si todas las etiquetas del JSON están presentes en el documento.
Si faltan etiquetas, envía a DeepSeek para corrección.
Repite hasta que no haya etiquetas faltantes o se alcance el máximo de iteraciones.
"""

import json
import csv
import argparse
import requests
import os
import re
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Set, Tuple
import datetime

def debug_print(message: str, level: str = "INFO"):
    """Función para imprimir mensajes de debug con timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def load_available_labels(csv_file: str) -> Set[str]:
    """Carga las etiquetas disponibles del archivo CSV."""
    labels = set()
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Saltar header
            
            for row in csv_reader:
                if len(row) >= 1 and row[0].strip():
                    labels.add(row[0].strip())
                if len(row) >= 2 and row[1].strip():
                    labels.add(row[1].strip())
    
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {csv_file}")
        return set()
    
    # Remover etiquetas vacías o headers
    labels.discard("")
    labels.discard("MEDDOCAN")
    labels.discard("CARMEN-I")
    
    return labels

def load_all_annotations(jsonl_file: str) -> Dict[str, List[Dict]]:
    """
    Carga todas las anotaciones del archivo JSONL organizadas por ID.
    
    Returns:
        Dict[str, List[Dict]]: Diccionario con ID como clave y lista de entidades como valor
    """
    annotations_by_id = {}
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        doc_id = json_data.get('id', f'unknown_{line_num}')
                        entities = json_data.get('data', [])
                        annotations_by_id[doc_id] = entities
                    except json.JSONDecodeError as e:
                        print(f"Error al parsear JSON en línea {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {jsonl_file}")
        return {}
    
    return annotations_by_id

def check_existing_corrected_documents(output_dir: str) -> Set[str]:
    """
    Verifica qué documentos ya han sido corregidos en el directorio de salida del step4.
    
    Args:
        output_dir (str): Directorio donde se guardan los documentos corregidos
        
    Returns:
        Set[str]: Conjunto de IDs de documentos que ya han sido corregidos
    """
    existing_docs = set()
    
    if not os.path.exists(output_dir):
        debug_print(f"Directorio de salida no existe: {output_dir}", "INFO")
        return existing_docs
    
    try:
        # Buscar archivos .txt que corresponden a documentos corregidos
        for filename in os.listdir(output_dir):
            if filename.endswith('.txt') and not filename.startswith('step'):
                # Extraer el ID del documento del nombre del archivo
                doc_id = filename[:-4]  # Remover .txt
                existing_docs.add(doc_id)
        
        debug_print(f"Encontrados {len(existing_docs)} documentos ya corregidos", "INFO")
        
    except Exception as e:
        debug_print(f"Error al verificar documentos corregidos existentes: {e}", "ERROR")
    
    return existing_docs

def filter_remaining_documents_to_correct(input_dir: str, annotations_by_id: Dict[str, List[Dict]], 
                                         existing_corrected_docs: Set[str]) -> List[str]:
    """
    Filtra los documentos para procesar solo los que faltan por corregir.
    
    Args:
        input_dir (str): Directorio con documentos del paso anterior
        annotations_by_id (Dict[str, List[Dict]]): Anotaciones organizadas por ID
        existing_corrected_docs (Set[str]): IDs de documentos ya corregidos
        
    Returns:
        List[str]: Lista de IDs de documentos que faltan por corregir
    """
    remaining_docs = []
    
    # Encontrar documentos disponibles en el directorio de entrada
    input_path = Path(input_dir)
    available_docs = list(input_path.glob("*.txt"))
    
    for doc in available_docs:
        doc_id = doc.stem
        
        # Solo incluir si:
        # 1. Tiene anotaciones correspondientes
        # 2. No ha sido corregido aún
        if doc_id in annotations_by_id and doc_id not in existing_corrected_docs:
            remaining_docs.append(doc_id)
    
    debug_print(f"Documentos restantes por corregir: {len(remaining_docs)}", "INFO")
    
    return remaining_docs

def clean_entity_names_from_document(document_text: str, available_labels: Set[str]) -> str:
    """
    Elimina nombres literales de entidades del documento.
    
    Args:
        document_text (str): Texto del documento
        available_labels (Set[str]): Etiquetas disponibles
        
    Returns:
        str: Documento limpio sin nombres de entidades
    """
    cleaned_text = document_text
    
    # Lista de nombres de entidades a eliminar
    entity_names_to_remove = [
        "NOMBRE_SUJETO_ASISTENCIA",
        "NOMBRE_PERSONAL_SANITARIO", 
        "ID_SUJETO_ASISTENCIA",
        "CENTRO_SALUD",
        "HOSPITAL",
        "CALLE",
        "TERRITORIO",
        "PAIS",
        "FECHAS",
        "EDAD_SUJETO_ASISTENCIA",
        "SEXO_SUJETO_ASISTENCIA",
        "NUMERO_TELEFONO",
        "CORREO_ELECTRONICO",
        "PROFESION",
        "FAMILIARES_SUJETO_ASISTENCIA",
        "INSTITUCION",
        "URL_WEB",
        "NUMERO_FAX",
        "ID_CONTACTO_ASISTENCIAL",
        "ID_TITULACION_PERSONAL_SANITARIO",
        "ID_EMPLEO_PERSONAL_SANITARIO",
        "ID_ASEGURAMIENTO",
        "NUMERO_BENEF_PLAN_SALUD",
        "IDENTIF_BIOMETRICOS",
        "IDENTIF_DISPOSITIVOS_NRSERIE",
        "IDENTIF_VEHICULOS_NRSERIE_PLACAS",
        "DIREC_PROT_INTERNET",
        "OTROS_SUJETO_ASISTENCIA",
        "OTRO_NUMERO_IDENTIF"
    ]
    
    # Eliminar nombres literales de entidades
    removed_count = 0
    for entity_name in entity_names_to_remove:
        if entity_name in available_labels:
            # Buscar y eliminar el nombre literal de la entidad
            if entity_name in cleaned_text:
                cleaned_text = cleaned_text.replace(entity_name, "")
                removed_count += 1
    
    # Limpiar espacios múltiples y saltos de línea extra
    import re
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    if removed_count > 0:
        debug_print(f"    Eliminados {removed_count} nombres literales de entidades", "DEBUG")
    
    return cleaned_text

def check_missing_entities(document_text: str, expected_entities: List[Dict], available_labels: Set[str]) -> List[Dict]:
    """
    Verifica qué entidades del JSON faltan en el documento.
    
    Args:
        document_text (str): Texto del documento
        expected_entities (List[Dict]): Entidades que deberían estar presentes
        available_labels (Set[str]): Etiquetas disponibles
        
    Returns:
        List[Dict]: Lista de entidades faltantes
    """
    missing_entities = []
    
    for entity in expected_entities:
        etiqueta = entity.get('entity', '')
        texto = entity.get('text', '').strip()
        
        # Solo verificar entidades válidas
        if etiqueta in available_labels and texto and texto not in ['[No se proporcionó texto para extraer la edad]', '[valor generado]']:
            # Buscar el texto exacto en el documento (case sensitive)
            if texto not in document_text:
                missing_entities.append(entity)
    
    return missing_entities

def create_correction_prompt(document_text: str, missing_entities: List[Dict], available_labels: Set[str]) -> str:
    """
    Crea un prompt para corregir el documento añadiendo las entidades faltantes.
    
    Args:
        document_text (str): Texto actual del documento
        missing_entities (List[Dict]): Entidades que faltan
        available_labels (Set[str]): Etiquetas disponibles
        
    Returns:
        str: Prompt para DeepSeek
    """
    
    # Extraer textos de entidades faltantes
    missing_texts = []
    missing_labels = set()
    
    for entity in missing_entities:
        etiqueta = entity.get('entity', '')
        texto = entity.get('text', '').strip()
        
        if etiqueta in available_labels and texto:
            missing_texts.append(texto)
            missing_labels.add(etiqueta)
    
    missing_labels_list = sorted(list(missing_labels))
    
    prompt = f"""Eres un médico especialista que debe CORREGIR un documento clínico en español. El documento actual está INCOMPLETO porque le faltan algunos datos específicos que DEBEN aparecer.

DOCUMENTO ACTUAL (INCOMPLETO):
{document_text}

DATOS FALTANTES QUE DEBES AÑADIR (usar estos textos EXACTAMENTE como están escritos):
{chr(10).join([f"- {texto}" for texto in missing_texts])}

ETIQUETAS DE LOS DATOS FALTANTES:
{', '.join(missing_labels_list)}

INSTRUCCIONES PARA LA CORRECCIÓN:

1. MANTÉN TODO EL CONTENIDO ACTUAL del documento
2. AÑADE los datos faltantes de manera natural e integrada
3. Los datos faltantes deben aparecer como texto normal, NO como listas o marcadores
4. Integra cada dato faltante en el contexto médico apropiado
5. Si es necesario, crea nuevos párrafos o secciones para incluir los datos
6. Mantén el estilo médico profesional y la coherencia del documento
7. NO elimines ni modifiques el contenido existente
8. NO uses etiquetas XML como <ETIQUETA>texto</ETIQUETA>

RESTRICCIONES CRÍTICAS:
- USA EXACTAMENTE los textos de la lista de datos faltantes
- NO inventes, modifiques o cambies estos textos
- NO añadas información adicional que no esté en la lista
- El documento corregido debe sonar natural y profesional
- Todos los datos faltantes DEBEN aparecer en el documento final

FORMATO DEL TEXTO CORREGIDO:
- Escribe ÚNICAMENTE el documento médico completo y corregido
- NO incluyas explicaciones, comentarios o notas adicionales
- NO uses formato de marcado o etiquetas
- El texto debe fluir naturalmente como un documento médico real

Genera el documento médico COMPLETO Y CORREGIDO que incluya tanto el contenido actual como los datos faltantes."""

    return prompt

async def call_deepseek_api_async(session: aiohttp.ClientSession, prompt: str, api_key: str, model: str = "deepseek-chat") -> str:
    """
    Llama a la API de DeepSeek de manera asíncrona para corregir el documento.
    
    Args:
        session (aiohttp.ClientSession): Sesión HTTP asíncrona
        prompt (str): Prompt para el modelo
        api_key (str): Clave API de DeepSeek
        model (str): Modelo a usar
        
    Returns:
        str: Respuesta generada por DeepSeek
    """
    
    url = "https://api.deepseek.com/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 2500,
        "temperature": 0.3  # Menor temperatura para correcciones más precisas
    }
    
    try:
        async with session.post(url, headers=headers, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result['choices'][0]['message']['content']
            
    except aiohttp.ClientError as e:
        debug_print(f"Error en la llamada a la API: {e}", "ERROR")
        return ""
    except KeyError as e:
        debug_print(f"Error en la respuesta de la API: {e}", "ERROR")
        return ""

def call_deepseek_api(prompt: str, api_key: str, model: str = "deepseek-chat") -> str:
    """Llama a la API de DeepSeek para generar/corregir el documento."""
    
    url = "https://api.deepseek.com/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 2500,
        "temperature": 0.3  # Menor temperatura para correcciones más precisas
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.RequestException as e:
        debug_print(f"Error en la llamada a la API: {e}", "ERROR")
        return ""
    except KeyError as e:
        debug_print(f"Error en la respuesta de la API: {e}", "ERROR")
        return ""

async def process_document_iteratively_async(session: aiohttp.ClientSession, doc_id: str, input_dir: str, expected_entities: List[Dict], 
                                            available_labels: Set[str], api_key: str, model: str, 
                                            max_iterations: int = 5) -> Dict:
    """
    Procesa un documento iterativamente de manera asíncrona hasta que todas las entidades estén presentes.
    
    Args:
        session (aiohttp.ClientSession): Sesión HTTP asíncrona
        doc_id (str): ID del documento
        input_dir (str): Directorio con los documentos del paso anterior
        expected_entities (List[Dict]): Entidades que deben estar presentes
        available_labels (Set[str]): Etiquetas disponibles
        api_key (str): Clave API de DeepSeek
        model (str): Modelo a usar
        max_iterations (int): Máximo número de iteraciones
        
    Returns:
        Dict: Resultado del procesamiento
    """
    
    debug_print(f"Procesando documento {doc_id}", "DEBUG")
    
    # Cargar documento inicial
    doc_file = os.path.join(input_dir, f"{doc_id}.txt")
    if not os.path.exists(doc_file):
        debug_print(f"No se encontró el documento: {doc_file}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": "Documento no encontrado",
            "iterations": 0
        }
    
    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            current_document = f.read().strip()
    except Exception as e:
        debug_print(f"Error al leer documento {doc_id}: {e}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": f"Error al leer documento: {e}",
            "iterations": 0
        }
    
    # Limpiar nombres literales de entidades del documento
    current_document = clean_entity_names_from_document(current_document, available_labels)
    
    # Filtrar entidades válidas esperadas
    valid_expected_entities = []
    for entity in expected_entities:
        etiqueta = entity.get('entity', '')
        texto = entity.get('text', '').strip()
        
        if etiqueta in available_labels and texto and texto not in ['[No se proporcionó texto para extraer la edad]', '[valor generado]']:
            valid_expected_entities.append(entity)
    
    # Capturar entidades faltantes al inicio (antes de cualquier iteración)
    initial_missing_entities = check_missing_entities(current_document, valid_expected_entities, available_labels)
    initial_missing_count = len(initial_missing_entities)
    
    iteration_history = []
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        debug_print(f"  --- ITERACIÓN {iteration} para {doc_id} ---", "DEBUG")
        
        # Verificar entidades faltantes
        missing_entities = check_missing_entities(current_document, valid_expected_entities, available_labels)
        
        if not missing_entities:
            debug_print(f"    ¡Todas las entidades están presentes! Completado en {iteration-1} iteraciones.", "DEBUG")
            break
        
        # Crear prompt de corrección
        correction_prompt = create_correction_prompt(current_document, missing_entities, available_labels)
        
        # Enviar a DeepSeek para corrección (llamada asíncrona)
        corrected_document = await call_deepseek_api_async(session, correction_prompt, api_key, model)
        
        if not corrected_document:
            debug_print(f"    Error en la corrección en iteración {iteration}", "ERROR")
            break
        
        # Guardar historial de la iteración
        iteration_info = {
            "iteration": iteration,
            "missing_entities_count": len(missing_entities),
            "missing_entities": missing_entities,
            "document_length_before": len(current_document),
            "document_length_after": len(corrected_document)
        }
        iteration_history.append(iteration_info)
        
        # Actualizar documento actual
        current_document = corrected_document
    
    # Verificación final
    final_missing = check_missing_entities(current_document, valid_expected_entities, available_labels)
    
    success = len(final_missing) == 0
    debug_print(f"  RESULTADO: {'ÉXITO' if success else 'INCOMPLETO'} - {len(final_missing)} entidades faltantes", 
                "DEBUG" if success else "WARN")
    
    return {
        "document_id": doc_id,
        "success": success,
        "iterations": iteration,
        "expected_entities_count": len(valid_expected_entities),
        "initial_missing_count": initial_missing_count,
        "final_missing_count": len(final_missing),
        "final_missing_entities": final_missing,
        "final_document": current_document,
        "iteration_history": iteration_history,
        "needed_correction": iteration > 1
    }

async def process_batch_documents_iteratively_async(batch_doc_ids: List[str], input_dir: str, annotations_by_id: Dict[str, List[Dict]], 
                                                   available_labels: Set[str], api_key: str, model: str, 
                                                   max_iterations: int = 5, max_concurrent: int = 2) -> List[Dict]:
    """
    Procesa un lote de documentos iterativamente de manera asíncrona y paralela.
    
    Args:
        batch_doc_ids (List[str]): Lista de IDs de documentos a procesar
        input_dir (str): Directorio con los documentos del paso anterior
        annotations_by_id (Dict[str, List[Dict]]): Anotaciones organizadas por ID
        available_labels (Set[str]): Etiquetas disponibles
        api_key (str): Clave API de DeepSeek
        model (str): Modelo a usar
        max_iterations (int): Máximo número de iteraciones
        max_concurrent (int): Máximo número de documentos concurrentes
        
    Returns:
        List[Dict]: Lista de resultados del procesamiento
    """
    
    debug_print(f"Procesando lote de {len(batch_doc_ids)} documentos de manera paralela (máx {max_concurrent} concurrentes)...", "INFO")
    
    # Crear semáforo para limitar documentos concurrentes
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(doc_id):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                expected_entities = annotations_by_id[doc_id]
                return await process_document_iteratively_async(
                    session,
                    doc_id,
                    input_dir,
                    expected_entities,
                    available_labels,
                    api_key,
                    model,
                    max_iterations
                )
    
    # Crear todas las tareas
    tasks = [process_with_semaphore(doc_id) for doc_id in batch_doc_ids]
    
    # Ejecutar todas las tareas en paralelo
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Manejar excepciones
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            debug_print(f"Error procesando documento {batch_doc_ids[i]}: {result}", "ERROR")
            processed_results.append({
                "document_id": batch_doc_ids[i],
                "success": False,
                "error": str(result),
                "iterations": 0
            })
        else:
            processed_results.append(result)
    
    return processed_results

def process_batch_documents_iteratively(batch_doc_ids: List[str], input_dir: str, annotations_by_id: Dict[str, List[Dict]], 
                                      available_labels: Set[str], api_key: str, model: str, 
                                      max_iterations: int = 5) -> List[Dict]:
    """
    Procesa un lote de documentos iterativamente.
    
    Args:
        batch_doc_ids (List[str]): Lista de IDs de documentos a procesar
        input_dir (str): Directorio con los documentos del paso anterior
        annotations_by_id (Dict[str, List[Dict]]): Anotaciones organizadas por ID
        available_labels (Set[str]): Etiquetas disponibles
        api_key (str): Clave API de DeepSeek
        model (str): Modelo a usar
        max_iterations (int): Máximo número de iteraciones
        
    Returns:
        List[Dict]: Lista de resultados del procesamiento
    """
    results = []
    
    debug_print(f"Procesando lote de {len(batch_doc_ids)} documentos...", "INFO")
    
    for doc_id in batch_doc_ids:
        expected_entities = annotations_by_id[doc_id]
        result = process_document_iteratively(
            doc_id,
            input_dir,
            expected_entities,
            available_labels,
            api_key,
            model,
            max_iterations
        )
        results.append(result)
    
    return results

def process_document_iteratively(doc_id: str, input_dir: str, expected_entities: List[Dict], 
                                available_labels: Set[str], api_key: str, model: str, 
                                max_iterations: int = 5) -> Dict:
    """
    Procesa un documento iterativamente hasta que todas las entidades estén presentes.
    
    Args:
        doc_id (str): ID del documento
        input_dir (str): Directorio con los documentos del paso 1
        expected_entities (List[Dict]): Entidades que deben estar presentes
        available_labels (Set[str]): Etiquetas disponibles
        api_key (str): Clave API de DeepSeek
        model (str): Modelo a usar
        max_iterations (int): Máximo número de iteraciones
        
    Returns:
        Dict: Resultado del procesamiento
    """
    
    debug_print(f"Procesando documento {doc_id}", "INFO")
    
    # Cargar documento inicial
    doc_file = os.path.join(input_dir, f"{doc_id}.txt")
    if not os.path.exists(doc_file):
        debug_print(f"No se encontró el documento: {doc_file}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": "Documento no encontrado",
            "iterations": 0
        }
    
    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            current_document = f.read().strip()
    except Exception as e:
        debug_print(f"Error al leer documento {doc_id}: {e}", "ERROR")
        return {
            "document_id": doc_id,
            "success": False,
            "error": f"Error al leer documento: {e}",
            "iterations": 0
        }
    
    # Limpiar nombres literales de entidades del documento
    debug_print(f"  Limpiando nombres literales de entidades...", "DEBUG")
    current_document = clean_entity_names_from_document(current_document, available_labels)
    
    # Filtrar entidades válidas esperadas
    valid_expected_entities = []
    for entity in expected_entities:
        etiqueta = entity.get('entity', '')
        texto = entity.get('text', '').strip()
        
        if etiqueta in available_labels and texto and texto not in ['[No se proporcionó texto para extraer la edad]', '[valor generado]']:
            valid_expected_entities.append(entity)
    
    debug_print(f"  Entidades esperadas: {len(valid_expected_entities)}", "DEBUG")
    
    # Capturar entidades faltantes al inicio (antes de cualquier iteración)
    initial_missing_entities = check_missing_entities(current_document, valid_expected_entities, available_labels)
    initial_missing_count = len(initial_missing_entities)
    debug_print(f"  Entidades faltantes al inicio: {initial_missing_count}", "DEBUG")
    
    iteration_history = []
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        debug_print(f"  --- ITERACIÓN {iteration} ---", "INFO")
        
        # Verificar entidades faltantes
        missing_entities = check_missing_entities(current_document, valid_expected_entities, available_labels)
        
        debug_print(f"    Entidades faltantes: {len(missing_entities)}", "INFO")
        
        if not missing_entities:
            debug_print(f"    ¡Todas las entidades están presentes! Completado en {iteration-1} iteraciones.", "INFO")
            break
        
        # Mostrar entidades faltantes
        for entity in missing_entities:
            debug_print(f"      Falta: {entity.get('entity', '')} = '{entity.get('text', '')}'", "DEBUG")
        
        # Crear prompt de corrección
        correction_prompt = create_correction_prompt(current_document, missing_entities, available_labels)
        
        # Enviar a DeepSeek para corrección
        debug_print(f"    Enviando a DeepSeek para corrección...", "DEBUG")
        corrected_document = call_deepseek_api(correction_prompt, api_key, model)
        
        if not corrected_document:
            debug_print(f"    Error en la corrección en iteración {iteration}", "ERROR")
            break
        
        # Guardar historial de la iteración
        iteration_info = {
            "iteration": iteration,
            "missing_entities_count": len(missing_entities),
            "missing_entities": missing_entities,
            "document_length_before": len(current_document),
            "document_length_after": len(corrected_document)
        }
        iteration_history.append(iteration_info)
        
        # Actualizar documento actual
        current_document = corrected_document
        debug_print(f"    Documento actualizado (longitud: {len(current_document)} caracteres)", "DEBUG")
    
    # Verificación final
    final_missing = check_missing_entities(current_document, valid_expected_entities, available_labels)
    
    success = len(final_missing) == 0
    debug_print(f"  RESULTADO: {'ÉXITO' if success else 'INCOMPLETO'} - {len(final_missing)} entidades faltantes", 
                "INFO" if success else "WARN")
    
    return {
        "document_id": doc_id,
        "success": success,
        "iterations": iteration,
        "expected_entities_count": len(valid_expected_entities),
        "initial_missing_count": initial_missing_count,
        "final_missing_count": len(final_missing),
        "final_missing_entities": final_missing,
        "final_document": current_document,
        "iteration_history": iteration_history,
        "needed_correction": iteration > 1
    }


def clean_failed_documents_from_jsonl(jsonl_file: str, failed_document_ids: List[str]):
    """
    Elimina las referencias de documentos fallidos del archivo JSONL.
    
    Args:
        jsonl_file (str): Ruta al archivo JSONL original
        failed_document_ids (List[str]): Lista de IDs de documentos que fallaron
    """
    if not failed_document_ids:
        debug_print("No hay documentos fallidos para eliminar del JSONL", "INFO")
        return
    
    debug_print(f"Eliminando {len(failed_document_ids)} documentos fallidos del JSONL...", "INFO")
    
    # Crear backup del archivo original
    backup_file = jsonl_file + ".backup"
    import shutil
    shutil.copy2(jsonl_file, backup_file)
    debug_print(f"Backup creado: {backup_file}", "DEBUG")
    
    # Leer todas las líneas y filtrar las que no corresponden a documentos fallidos
    kept_lines = []
    removed_count = 0
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        doc_id = json_data.get('id', f'unknown_{line_num}')
                        
                        if doc_id not in failed_document_ids:
                            kept_lines.append(line)
                        else:
                            removed_count += 1
                            debug_print(f"  Eliminado documento fallido: {doc_id}", "DEBUG")
                            
                    except json.JSONDecodeError as e:
                        debug_print(f"Error al parsear JSON en línea {line_num}: {e}", "WARN")
                        # Mantener líneas con errores de parsing
                        kept_lines.append(line)
        
        # Escribir el archivo limpio
        with open(jsonl_file, 'w', encoding='utf-8') as file:
            for line in kept_lines:
                file.write(line + '\n')
        
        debug_print(f"JSONL limpiado: {removed_count} documentos eliminados, {len(kept_lines)} mantenidos", "INFO")
        
    except Exception as e:
        debug_print(f"Error al limpiar JSONL: {e}", "ERROR")
        # Restaurar backup en caso de error
        shutil.copy2(backup_file, jsonl_file)
        debug_print("Archivo JSONL restaurado desde backup", "WARN")

def create_summary_report(results: List[Dict], output_dir: str, total_dataset_size: int = None, existing_corrected_count: int = None, batch_size: int = None):
    """Crea un reporte resumen del paso 2."""
    
    debug_print("Creando reporte resumen del paso 2...", "INFO")
    
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    corrected_docs = [r for r in successful_results if r.get("needed_correction", False)]
    
    summary = {
        "pipeline_step": 2,
        "process": "iterative_verification_and_correction",
        "total_documents_in_dataset": total_dataset_size,
        "existing_corrected_documents": existing_corrected_count,
        "documents_processed_this_run": len(results),
        "successful_documents": len(successful_results),
        "failed_documents": len(failed_results),
        "documents_needing_correction": len(corrected_docs),
        "documents_perfect_from_step1": len(successful_results) - len(corrected_docs),
        "batch_size_used": batch_size,
        "total_batches": (len(results) + batch_size - 1) // batch_size if batch_size else 1,
        "average_iterations": sum([r.get("iterations", 0) for r in results]) / len(results) if results else 0,
        "total_entities_expected": sum([r.get("expected_entities_count", 0) for r in results]),
        "total_entities_missing_initial": sum([r.get("initial_missing_count", 0) for r in results]),
        "total_entities_missing_final": sum([r.get("final_missing_count", 0) for r in results]),
        "entities_corrected": sum([r.get("initial_missing_count", 0) for r in results]) - sum([r.get("final_missing_count", 0) for r in results]),
        "success_rate": len(successful_results) / len(results) * 100 if results else 0,
        "documents": results
    }
    
    summary_file = os.path.join(output_dir, "step3_verification_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    debug_print(f"Reporte guardado: {summary_file}", "INFO")
    
    # Mostrar resumen en consola
    print(f"\n{'='*60}")
    print(f"PASO 2 - VERIFICACIÓN Y CORRECCIÓN COMPLETADA")
    print(f"{'='*60}")
    if total_dataset_size is not None:
        print(f"Documentos totales en dataset: {total_dataset_size}")
    if existing_corrected_count is not None:
        print(f"Documentos ya corregidos: {existing_corrected_count}")
    print(f"Documentos procesados en esta ejecución: {summary['documents_processed_this_run']}")
    print(f"Exitosos: {summary['successful_documents']}")
    print(f"Fallidos: {summary['failed_documents']}")
    print(f"Documentos que necesitaron corrección: {summary['documents_needing_correction']}")
    print(f"Documentos perfectos desde paso 1: {summary['documents_perfect_from_step1']}")
    if batch_size is not None:
        print(f"Tamaño de lote usado: {batch_size}")
        print(f"Total de lotes procesados: {summary['total_batches']}")
    print(f"Promedio de iteraciones: {summary['average_iterations']:.1f}")
    print(f"Tasa de éxito: {summary['success_rate']:.1f}%")
    print(f"Total entidades esperadas: {summary['total_entities_expected']}")
    print(f"Total entidades faltantes al inicio: {summary['total_entities_missing_initial']}")
    print(f"Total entidades aún faltantes: {summary['total_entities_missing_final']}")
    print(f"Entidades corregidas: {summary['entities_corrected']}")
    
    if summary['failed_documents'] > 0:
        print(f"\nDocumentos con problemas:")
        for result in failed_results:
            print(f"  - {result['document_id']}: {result.get('error', 'Error desconocido')}")
        print(f"\nNOTA: Los documentos fallidos han sido eliminados del archivo JSONL original.")
        print(f"Se ha creado un backup del archivo original con extensión .backup")
    
    print(f"\nDirectorio de salida: {output_dir}")
    print(f"{'='*60}")

async def main_async():
    parser = argparse.ArgumentParser(description="PASO 2: Verificar y corregir documentos iterativamente")
    parser.add_argument("--input-dir", default="step3_generated_documents",
                       help="Directorio con documentos del paso 4")
    parser.add_argument("--jsonl-file", default="examples/jsonl_data/medical_annotations.jsonl", 
                       help="Archivo JSONL con anotaciones")
    parser.add_argument("--labels-file", default="examples/etiquetas_anonimizacion_meddocan_carmenI.csv",
                       help="Archivo CSV con etiquetas disponibles")
    parser.add_argument("--api-key-file", default="api_keys", 
                       help="Archivo que contiene la clave API de DeepSeek")
    parser.add_argument("--output-dir", default="step4_corrected_documents",
                       help="Directorio de salida para documentos corregidos")
    parser.add_argument("--model", default="deepseek-chat",
                       help="Modelo de DeepSeek a usar")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Máximo número de iteraciones por documento")
    parser.add_argument("--max-docs", type=int, default=None,
                       help="Máximo número de documentos a procesar")
    parser.add_argument("--batch-size", type=int, default=3,
                       help="Número de documentos a procesar en cada lote")
    parser.add_argument("--max-concurrent", type=int, default=3,
                       help="Máximo número de documentos concurrentes (menor que step3 debido a corrección iterativa)")
    
    args = parser.parse_args()
    
    debug_print("=== PASO 2: VERIFICACIÓN Y CORRECCIÓN ITERATIVA (ASÍNCRONO) ===", "INFO")
    debug_print(f"Directorio de entrada: {args.input_dir}", "INFO")
    debug_print(f"Directorio de salida: {args.output_dir}", "INFO")
    debug_print(f"Máximo iteraciones por documento: {args.max_iterations}", "INFO")
    
    # Verificar directorio de entrada
    if not os.path.exists(args.input_dir):
        print(f"ERROR: No existe el directorio de entrada: {args.input_dir}")
        return
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar etiquetas disponibles
    debug_print("Cargando etiquetas disponibles...", "INFO")
    available_labels = load_available_labels(args.labels_file)
    debug_print(f"Cargadas {len(available_labels)} etiquetas disponibles", "INFO")
    
    # Cargar anotaciones
    debug_print("Cargando anotaciones del archivo JSONL...", "INFO")
    annotations_by_id = load_all_annotations(args.jsonl_file)
    
    if not annotations_by_id:
        print("ERROR: No se pudieron cargar anotaciones del archivo JSONL")
        return
    
    # Verificar qué documentos ya han sido corregidos
    debug_print("Verificando documentos ya corregidos...", "INFO")
    existing_corrected_docs = check_existing_corrected_documents(args.output_dir)
    
    # Filtrar solo los documentos que faltan por corregir
    remaining_documents = filter_remaining_documents_to_correct(
        args.input_dir, 
        annotations_by_id, 
        existing_corrected_docs
    )
    
    if not remaining_documents:
        debug_print("¡Todos los documentos ya han sido corregidos!", "INFO")
        debug_print(f"Total documentos en el dataset: {len(annotations_by_id)}", "INFO")
        debug_print(f"Documentos ya corregidos: {len(existing_corrected_docs)}", "INFO")
        return
    
    # Aplicar límite máximo si se especifica
    valid_documents = remaining_documents
    if args.max_docs:
        valid_documents = valid_documents[:args.max_docs]
    
    debug_print(f"Documentos totales en dataset: {len(annotations_by_id)}", "INFO")
    debug_print(f"Documentos ya corregidos: {len(existing_corrected_docs)}", "INFO")
    debug_print(f"Documentos restantes por corregir: {len(valid_documents)}", "INFO")
    debug_print(f"Tamaño de lote: {args.batch_size}", "INFO")
    debug_print(f"Máximo documentos concurrentes: {args.max_concurrent}", "INFO")
    
    # Cargar clave API
    try:
        with open(args.api_key_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            api_key = None
            for line in lines:
                if line.startswith('deepseek='):
                    api_key = line.split('=', 1)[1].strip()
                    break
            
            if not api_key:
                print("ERROR: No se encontró la clave API de DeepSeek en el archivo")
                return
                
    except FileNotFoundError:
        print(f"ERROR: No se pudo encontrar el archivo de clave API: {args.api_key_file}")
        return
    
    # Procesar documentos en lotes
    results = []
    total_batches = (len(valid_documents) + args.batch_size - 1) // args.batch_size
    
    debug_print(f"Procesando {len(valid_documents)} documentos en {total_batches} lotes de máximo {args.batch_size} documentos", "INFO")
    
    for batch_num in range(total_batches):
        start_idx = batch_num * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(valid_documents))
        batch_doc_ids = valid_documents[start_idx:end_idx]
        
        debug_print(f"\n--- PROCESANDO LOTE {batch_num + 1}/{total_batches} (documentos {start_idx + 1}-{end_idx}) ---", "INFO")
        
        # Procesar el lote de manera asíncrona
        batch_results = await process_batch_documents_iteratively_async(
            batch_doc_ids,
            args.input_dir,
            annotations_by_id,
            available_labels,
            api_key,
            args.model,
            args.max_iterations,
            args.max_concurrent
        )
        
        results.extend(batch_results)
        
        # Guardar documentos corregidos del lote
        for result in batch_results:
            if result.get("success", False):
                doc_id = result["document_id"]
                doc_file = os.path.join(args.output_dir, f"{doc_id}.txt")
                with open(doc_file, 'w', encoding='utf-8') as f:
                    f.write(result["final_document"])
                debug_print(f"  Documento guardado: {doc_file}", "DEBUG")
        
        # Mostrar progreso del lote
        successful_in_batch = len([r for r in batch_results if r.get("success", False)])
        total_successful = len([r for r in results if r.get("success", False)])
        debug_print(f"LOTE COMPLETADO: {len(batch_doc_ids)} procesados ({successful_in_batch} exitosos)", "INFO")
        debug_print(f"PROGRESO TOTAL: {end_idx}/{len(valid_documents)} documentos procesados ({total_successful} exitosos)", "INFO")
    
    # Crear reporte resumen
    create_summary_report(results, args.output_dir, len(annotations_by_id), len(existing_corrected_docs), args.batch_size)
    
    # Limpiar documentos fallidos del JSONL
    failed_document_ids = [r["document_id"] for r in results if not r.get("success", False)]
    clean_failed_documents_from_jsonl(args.jsonl_file, failed_document_ids)
    
    debug_print("Paso 2 completado! Pipeline finalizado.", "INFO")


def main():
    """Función principal que ejecuta la versión asíncrona."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
