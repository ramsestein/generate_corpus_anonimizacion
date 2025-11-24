#!/usr/bin/env python3
"""
STEP 1 DEL PIPELINE: Generador de anotaciones médicas JSONL
Genera un archivo medical_annotations.jsonl con el número especificado de entradas.
Cada entrada contiene múltiples etiquetas con sus textos correspondientes.
"""

import os
import json
import random
import csv
import time
import requests
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def debug_print(message: str, level: str = "INFO"):
    """Función para imprimir mensajes de debug con timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

class MedicalAnnotationsGenerator:
    """Generador de anotaciones médicas JSONL usando DeepSeek API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.available_labels = []
        self.load_labels()
    
    def load_labels(self):
        """Carga las etiquetas disponibles del CSV"""
        csv_path = Path("examples/etiquetas_anonimizacion_meddocan_carmenI.csv")
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # Saltar header
                
                labels_set = set()
                for row in reader:
                    if len(row) >= 1 and row[0].strip():  # MEDDOCAN label
                        label = row[0].strip()
                        if label and label != 'MEDDOCAN':
                            labels_set.add(label)
                    if len(row) >= 2 and row[1].strip():  # CARMEN-I label
                        label = row[1].strip()
                        if label and label != 'CARMEN-I':
                            labels_set.add(label)
                
                self.available_labels = sorted(list(labels_set))
            
            debug_print(f"Cargadas {len(self.available_labels)} etiquetas disponibles", "INFO")
            debug_print(f"Etiquetas: {', '.join(self.available_labels[:10])}{'...' if len(self.available_labels) > 10 else ''}", "DEBUG")
            
        except Exception as e:
            debug_print(f"Error cargando etiquetas: {str(e)}", "ERROR")
            # Etiquetas por defecto si falla la carga
            self.available_labels = [
                "TERRITORIO", "FECHAS", "EDAD_SUJETO_ASISTENCIA", 
                "NOMBRE_SUJETO_ASISTENCIA", "NOMBRE_PERSONAL_SANITARIO",
                "HOSPITAL", "INSTITUCION", "NUMERO_TELEFONO", "PROFESION",
                "CENTRO_SALUD", "CALLE", "CORREO_ELECTRONICO", "PAIS",
                "SEXO_SUJETO_ASISTENCIA", "FAMILIARES_SUJETO_ASISTENCIA"
            ]
    
    def generate_text_for_label_batch(self, labels: List[str]) -> List[str]:
        """Genera textos para múltiples etiquetas en paralelo usando DeepSeek"""
        
        def generate_single_text(label: str) -> str:
            # Crear prompt específico para cada etiqueta
            prompt = f"""Genera SOLO el valor específico para la etiqueta médica de anonimización: {label}

IMPORTANTE: Responde ÚNICAMENTE con el valor realista, sin contexto ni explicaciones.

Ejemplos por tipo de etiqueta:
- FECHAS: "15 de marzo de 2024"
- TERRITORIO: "Madrid"
- NOMBRE_SUJETO_ASISTENCIA: "Juan Pérez García"
- HOSPITAL: "Hospital Universitario La Paz"
- NUMERO_TELEFONO: "91-234-5678"
- CALLE: "Calle de la Princesa 25"
- PROFESION: "enfermera"
- CORREO_ELECTRONICO: "maria.lopez@correo.es"
- EDAD_SUJETO_ASISTENCIA: "45 años"
- SEXO_SUJETO_ASISTENCIA: "masculino"
- CENTRO_SALUD: "Centro de Salud El Carmen"
- ID_SUJETO_ASISTENCIA: "12345678A"
- IDENTIF_DISPOSITIVOS_NRSERIE: "SN-78451209"

Para {label}, genera solo el valor realista:"""
            
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.8,
                        "max_tokens": 100
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    # Limpiar el contenido de comillas o caracteres extra
                    content = content.strip('"\'').strip()
                    return content if content else self.get_fallback_text(label)
                else:
                    return self.get_fallback_text(label)
                    
            except Exception as e:
                return self.get_fallback_text(label)
        
        # Ejecutar llamadas en paralelo
        max_workers = min(getattr(self, 'api_workers', 5), len(labels))
        results = [None] * len(labels)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Enviar todas las llamadas
            future_to_index = {
                executor.submit(generate_single_text, label): i 
                for i, label in enumerate(labels)
            }
            
            # Recopilar resultados en orden
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = self.get_fallback_text(labels[index])
        
        return results
    
    def get_fallback_text(self, label: str) -> str:
        """Proporciona texto de respaldo para cada etiqueta"""
        fallback_texts = {
            "TERRITORIO": random.choice(["Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao"]),
            "FECHAS": random.choice(["15 de marzo de 2024", "enero de 2023", "12/05/2022", "abril de 2024"]),
            "EDAD_SUJETO_ASISTENCIA": random.choice(["45 años", "67 años", "34 años", "52 años", "28 años"]),
            "NOMBRE_SUJETO_ASISTENCIA": random.choice(["Juan Pérez García", "María López Martínez", "Carlos Fernández Silva"]),
            "NOMBRE_PERSONAL_SANITARIO": random.choice(["Dra. Elena Rodríguez Martínez", "Dra. Elena Martínez Rodríguez", "Dra. Elena Martínez Ruiz"]),
            "HOSPITAL": random.choice(["Hospital Universitario La Paz", "Hospital Universitario Ramón y Cajal", "Clínica Universidad de Navarra"]),
            "INSTITUCION": random.choice(["Ministerio de Sanidad", "Servicio de Neurología", "Instituto de Salud Carlos III"]),
            "NUMERO_TELEFONO": random.choice(["91-234-5678", "93-456-7890", "96-123-4567"]),
            "PROFESION": random.choice(["enfermera", "enfermero", "cirujano", "médico"]),
            "CORREO_ELECTRONICO": random.choice(["maria.lopez@correo.es", "juan.perez@ejemplo.com", "paciente.nombre@dominio.es"]),
            "PAIS": random.choice(["España", "Francia", "Italia", "Portugal"]),
            "CALLE": random.choice(["Calle de la Princesa 25", "Avenida de la Paz 45", "Plaza España 67"]),
            "SEXO_SUJETO_ASISTENCIA": random.choice(["masculino", "femenino"]),
            "FAMILIARES_SUJETO_ASISTENCIA": random.choice(["hermana", "madre", "padre", "hermano"]),
            "CENTRO_SALUD": random.choice(["Centro de Salud El Carmen", "Centro de Salud Los Ángeles", "Centro de Salud La Fortuna"]),
            "ID_SUJETO_ASISTENCIA": random.choice(["12345678A", "87654321B", "11223344C"]),
            "IDENTIF_DISPOSITIVOS_NRSERIE": random.choice(["SN-78451209", "123456789ABC", "SN-8472-AB6"]),
            "IDENTIF_BIOMETRICOS": random.choice(["12345678A", "12345678Z", "87654321B"]),
            "ID_EMPLEO_PERSONAL_SANITARIO": random.choice(["12345678A", "EMP-123456", "ID_EMPLEO_PS_123456"]),
            "IDENTIF_VEHICULOS_NRSERIE_PLACAS": random.choice(["1234 ABC", "5678 XYZ", "ABC-1234"]),
            "NUMERO_FAX": random.choice(["91-876-5432", "+34 91 123 45 67", "+34 91 456 78 90"]),
            "URL_WEB": random.choice(["https://temu.bio", "https://meddocan.herokuapp.com/", "http://www.meddocan.es"]),
            "DIREC_PROT_INTERNET": random.choice(["https://www.hospitalgeneral.com", "https://www.ejemplo.es"]),
            "ID_CONTACTO_ASISTENCIAL": random.choice(["HC-2024-789123", "ID-2024-789123", "HC-784512"]),
            "NUMERO_BENEF_PLAN_SALUD": random.choice(["123456789", "1234567890", "123456789012"]),
            "ID_TITULACION_PERSONAL_SANITARIO": random.choice(["12345678A", "Grado en Medicina", "Licenciado en Medicina"]),
            "ID_ASEGURAMIENTO": random.choice(["12345678A", "1234567890", "A123456789"]),
            "OTROS_SUJETO_ASISTENCIA": random.choice(["hermana del paciente", "familiar", "otro sujeto de asistencia"]),
            "OTRO_NUMERO_IDENTIF": random.choice(["NHC-2024-789", "ID_123456", ""])
        }
        
        return fallback_texts.get(label, f"ejemplo_{label.lower()}")
    
    def generate_single_entry(self, existing_ids: Set[str] = None) -> Dict[str, Any]:
        """Genera una única entrada JSONL con múltiples entidades"""
        
        if existing_ids is None:
            existing_ids = set()
        
        # Usar los parámetros configurados o valores por defecto
        min_entities = getattr(self, 'min_entities', 3)
        max_entities = getattr(self, 'max_entities', 12)
        
        # Seleccionar un número aleatorio de etiquetas diferentes
        num_different_labels = random.randint(min_entities, min(max_entities, len(self.available_labels)))
        selected_labels = random.sample(self.available_labels, num_different_labels)
        
        entities = []
        
        # Preparar lista de todas las etiquetas que necesitamos generar
        labels_to_generate = []
        label_counts = []
        
        for label in selected_labels:
            # Para cada etiqueta, generar entre 1 y 2 ejemplos (ocasionalmente 3)
            num_examples = random.choices([1, 2, 3], weights=[60, 35, 5])[0]
            label_counts.append((label, num_examples))
            
            # Añadir a la lista de etiquetas a generar
            for _ in range(num_examples):
                labels_to_generate.append(label)
        
        # Generar todos los textos en paralelo (máximo 5 llamadas simultáneas)
        debug_print(f"    Generando {len(labels_to_generate)} textos en paralelo...", "DEBUG")
        generated_texts = self.generate_text_for_label_batch(labels_to_generate)
        
        # Crear entidades con los textos generados
        for i, label in enumerate(labels_to_generate):
            text = generated_texts[i]
            entities.append({
                "entity": label,
                "text": text
            })
        
        # Generar ID único para la entrada (verificar que no exista)
        max_attempts = 10
        for attempt in range(max_attempts):
            entry_id = str(uuid.uuid4())
            if entry_id not in existing_ids:
                existing_ids.add(entry_id)  # Añadir a la lista para futuras verificaciones
                break
            debug_print(f"ID duplicado detectado en intento {attempt + 1}, generando nuevo...", "WARN")
        else:
            # Si después de 10 intentos no se encuentra un ID único, usar timestamp
            import time as time_module
            entry_id = f"entry_{int(time_module.time() * 1000000)}"
            existing_ids.add(entry_id)
            debug_print(f"Usando ID basado en timestamp: {entry_id}", "WARN")
        
        return {
            "id": entry_id,
            "data": entities
        }
    
    def generate_all_entries(self, num_entries: int, output_file: str = None, append_mode: bool = True):
        """Genera todas las entradas JSONL y las guarda en un archivo"""
        
        if output_file is None:
            output_dir = Path("examples/jsonl_data")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / "medical_annotations.jsonl"
        else:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Verificar si el archivo existe y cargar IDs existentes
        existing_entries = 0
        existing_ids = set()
        if append_mode and output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                entry_id = entry.get('id', '')
                                if entry_id:
                                    existing_ids.add(entry_id)
                                existing_entries += 1
                            except json.JSONDecodeError:
                                existing_entries += 1  # Contar línea aunque no se pueda parsear
                debug_print(f"Archivo existente encontrado con {existing_entries} entradas", "INFO")
                debug_print(f"IDs existentes cargados: {len(existing_ids)}", "DEBUG")
            except Exception as e:
                debug_print(f"Error leyendo archivo existente: {e}", "WARN")
                existing_entries = 0
                existing_ids = set()
        
        # Si no es append_mode, eliminar archivo existente
        if not append_mode and output_file.exists():
            debug_print("Modo sobrescritura: eliminando archivo existente", "INFO")
            output_file.unlink()
            existing_entries = 0
        
        action = "Añadiendo" if append_mode and existing_entries > 0 else "Creando"
        
        debug_print(f"{action} {num_entries} nuevas entradas JSONL...", "INFO")
        debug_print(f"Archivo: {output_file}", "INFO")
        if existing_entries > 0:
            debug_print(f"Entradas existentes: {existing_entries}", "INFO")
        
        successful_entries = 0
        failed_entries = 0
        
        # Generar y escribir una entrada a la vez
        for i in tqdm(range(num_entries), desc="Generando entradas"):
            try:
                # Generar una entrada (pasando IDs existentes para verificar unicidad)
                entry = self.generate_single_entry(existing_ids)
                
                # Escribir inmediatamente al archivo JSONL
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f.flush()  # Asegurar que se escriba inmediatamente
                
                successful_entries += 1
                
                # Mostrar progreso cada 10 entradas
                if (i + 1) % 10 == 0:
                    debug_print(f"Progreso: {i + 1}/{num_entries} entradas generadas y guardadas", "INFO")
                    
            except Exception as e:
                debug_print(f"Error generando entrada {i + 1}: {str(e)}", "ERROR")
                failed_entries += 1
                continue
        
        # Mostrar estadísticas finales
        total_entries_now = existing_entries + successful_entries
        debug_print(f"Generación completada!", "INFO")
        debug_print(f"Nuevas entradas exitosas: {successful_entries}", "INFO")
        debug_print(f"Nuevas entradas fallidas: {failed_entries}", "INFO")
        debug_print(f"Total entradas en archivo: {total_entries_now}", "INFO")
        debug_print(f"Archivo actualizado: {output_file}", "INFO")
        
        return successful_entries, failed_entries, str(output_file), existing_entries

def load_api_key(api_key_file: str = "api_keys") -> str:
    """Carga la API key desde el archivo especificado"""
    try:
        with open(api_key_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("deepseek="):
                    return line.split("=", 1)[1].strip()
        return None
    except FileNotFoundError:
        debug_print(f"No se encontró el archivo {api_key_file}", "ERROR")
        return None

def validate_generated_file(jsonl_file: str) -> Dict[str, Any]:
    """Valida el archivo JSONL generado"""
    debug_print(f"Validando archivo generado: {jsonl_file}", "INFO")
    
    if not os.path.exists(jsonl_file):
        return {"error": "Archivo no encontrado"}
    
    total_entries = 0
    total_entities = 0
    labels_used = set()
    entries_by_entity_count = {}
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    total_entries += 1
                    
                    # Contar entidades en esta entrada
                    entities = entry.get('data', [])
                    entry_entity_count = len(entities)
                    total_entities += entry_entity_count
                    
                    # Contar por número de entidades
                    if entry_entity_count not in entries_by_entity_count:
                        entries_by_entity_count[entry_entity_count] = 0
                    entries_by_entity_count[entry_entity_count] += 1
                    
                    # Recopilar etiquetas usadas
                    for entity in entities:
                        label = entity.get('entity', '')
                        if label:
                            labels_used.add(label)
                            
                except json.JSONDecodeError as e:
                    debug_print(f"Error en línea {line_num}: {e}", "WARN")
                    continue
    
    except Exception as e:
        return {"error": f"Error validando archivo: {e}"}
    
    validation_result = {
        "total_entries": total_entries,
        "total_entities": total_entities,
        "average_entities_per_entry": total_entities / total_entries if total_entries > 0 else 0,
        "unique_labels_used": len(labels_used),
        "labels_used": sorted(list(labels_used)),
        "entries_by_entity_count": entries_by_entity_count,
        "file_size_bytes": os.path.getsize(jsonl_file)
    }
    
    return validation_result

def create_summary_report(successful_entries: int, failed_entries: int, output_file: str, 
                         validation_result: Dict, output_dir: str, existing_entries: int = 0):
    """Crea un reporte resumen de la generación"""
    
    summary = {
        "pipeline_step": 1,
        "process": "medical_annotations_generation",
        "generation_results": {
            "new_successful_entries": successful_entries,
            "new_failed_entries": failed_entries,
            "existing_entries": existing_entries,
            "total_entries_after": existing_entries + successful_entries,
            "total_attempted": successful_entries + failed_entries,
            "success_rate": (successful_entries / (successful_entries + failed_entries) * 100) if (successful_entries + failed_entries) > 0 else 0
        },
        "output_file": output_file,
        "validation_results": validation_result
    }
    
    summary_file = os.path.join(output_dir, "step1_annotations_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    debug_print(f"Reporte guardado: {summary_file}", "INFO")
    
    # Mostrar resumen en consola
    print(f"\n{'='*60}")
    print(f"STEP 1 - GENERACIÓN DE ANOTACIONES COMPLETADA")
    print(f"{'='*60}")
    print(f"Nuevas entradas generadas: {successful_entries}")
    print(f"Entradas existentes: {summary['generation_results']['existing_entries']}")
    print(f"Total entradas en archivo: {summary['generation_results']['total_entries_after']}")
    print(f"Entradas fallidas: {failed_entries}")
    print(f"Tasa de éxito: {summary['generation_results']['success_rate']:.1f}%")
    print(f"")
    print(f"ESTADÍSTICAS DEL ARCHIVO GENERADO:")
    print(f"  - Total entradas: {validation_result.get('total_entries', 0)}")
    print(f"  - Total entidades: {validation_result.get('total_entities', 0)}")
    print(f"  - Promedio entidades por entrada: {validation_result.get('average_entities_per_entry', 0):.1f}")
    print(f"  - Etiquetas únicas utilizadas: {validation_result.get('unique_labels_used', 0)}")
    print(f"  - Tamaño del archivo: {validation_result.get('file_size_bytes', 0):,} bytes")
    print(f"")
    print(f"DISTRIBUCIÓN POR NÚMERO DE ENTIDADES:")
    for entity_count, entry_count in sorted(validation_result.get('entries_by_entity_count', {}).items()):
        print(f"  - {entity_count} entidades: {entry_count} entradas")
    print(f"")
    print(f"Archivo generado: {output_file}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="STEP 1: Generar anotaciones médicas JSONL")
    parser.add_argument("--num-entries", type=int, default=13000,
                       help="Número de entradas a generar en el JSONL (default: 250)")
    parser.add_argument("--output-file", default="examples/jsonl_data/medical_annotations.jsonl",
                       help="Archivo de salida para las anotaciones JSONL")
    parser.add_argument("--labels-file", default="examples/etiquetas_anonimizacion_meddocan_carmenI.csv",
                       help="Archivo CSV con etiquetas disponibles")
    parser.add_argument("--api-key-file", default="api_keys",
                       help="Archivo que contiene la clave API de DeepSeek")
    # Step1 no genera carpeta de resultados
    parser.add_argument("--min-entities", type=int, default=3,
                       help="Mínimo número de entidades por entrada (default: 3)")
    parser.add_argument("--max-entities", type=int, default=12,
                       help="Máximo número de entidades por entrada (default: 12)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Sobrescribir archivo existente en lugar de añadir (default: añadir)")
    parser.add_argument("--api-workers", type=int, default=5,
                       help="Número de llamadas paralelas a la API (default: 5)")
    
    args = parser.parse_args()
    
    debug_print("=== STEP 1: GENERACIÓN DE ANOTACIONES MÉDICAS ===", "INFO")
    debug_print(f"Número de entradas a generar: {args.num_entries}", "INFO")
    debug_print(f"Archivo de salida: {args.output_file}", "INFO")
    debug_print(f"Rango de entidades por entrada: {args.min_entities}-{args.max_entities}", "INFO")
    
    # Step1 no genera directorio de reportes
    
    # Cargar API key
    debug_print("Cargando clave API de DeepSeek...", "INFO")
    api_key = load_api_key(args.api_key_file)
    
    if not api_key:
        print("ERROR: No se encontró la clave API de DeepSeek")
        print(f"Asegúrate de que el archivo {args.api_key_file} existe y contiene 'deepseek=tu_api_key'")
        return
    
    debug_print(f"API key cargada exitosamente: {api_key[:10]}...", "INFO")
    
    # Crear generador
    generator = MedicalAnnotationsGenerator(api_key)
    
    # Configurar parámetros de generación
    generator.min_entities = args.min_entities
    generator.max_entities = args.max_entities
    generator.api_workers = args.api_workers
    
    debug_print(f"Llamadas paralelas a API: {args.api_workers}", "INFO")
    
    # Generar todas las entradas JSONL
    append_mode = not args.overwrite  # Si --overwrite está presente, no usar append
    successful, failed, output_path, existing = generator.generate_all_entries(args.num_entries, args.output_file, append_mode=append_mode)
    
    if successful == 0:
        print("ERROR: No se pudo generar ninguna entrada")
        return
    
    # Validar archivo generado
    debug_print("Validando archivo generado...", "INFO")
    validation_result = validate_generated_file(output_path)
    
    if "error" in validation_result:
        print(f"ERROR en validación: {validation_result['error']}")
        return
    
    # Step1 no genera reporte resumen
    
    debug_print("Step 1 completado! Procede con step2 para generar documentos.", "INFO")

if __name__ == "__main__":
    main()
