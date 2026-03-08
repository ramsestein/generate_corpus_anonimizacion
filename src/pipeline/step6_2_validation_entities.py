#!/usr/bin/env python3
"""
PASO 6.2 DEL PIPELINE: Corrección de Entidades Altamente Frecuentes (Optimizado)

Este script:
1. Carga los textos de entidades más frecuentes (>1% de documentos).
2. Genera un pool de 10 alternativas sintéticas por cada texto usando DeepSeek.
3. Sustituye estos textos en todos los documentos del corpus de forma eficiente
   y en tiempo real, informando de cada cambio.
"""

import os
import json
import asyncio
import aiohttp
import random
from typing import Dict, List, Set, Tuple
from pathlib import Path
from collections import defaultdict

def debug_print(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}", flush=True)

def load_api_key(filepath: str, service: str = "deepseek") -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(f"{service}="):
                    return line.strip().split('=', 1)[1]
    except Exception as e:
        debug_print(f"Error cargando API key: {e}", "ERROR")
    return ""

def load_high_frequency_texts(counts_file: Path, threshold_pct: float = 1.0) -> List[Tuple[str, str]]:
    if not counts_file.exists():
        debug_print(f"No se encontró {counts_file}. Ejecuta src/count_entities.py primero.", "ERROR")
        return []

    try:
        with open(counts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pairs = []
        for item in data.get("entities", []):
            if item.get("percentage_docs", 0) > threshold_pct:
                pairs.append((item["text"], item["entity"]))
        
        debug_print(f"Detectados {len(pairs)} textos que superan el {threshold_pct}% de frecuencia.", "INFO")
        return pairs
    except Exception as e:
        debug_print(f"Error cargando recuentos: {e}", "ERROR")
        return []

async def generate_pool_for_text(session: aiohttp.ClientSession, api_key: str, text: str, entity: str) -> List[str]:
    """Genera 10 alternativas para un texto específico."""
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    prompt = f"""Eres un experto en datos clínicos sintéticos.
El texto '{text}' para la entidad '{entity}' se repite demasiado en el corpus.
Genera una lista de 10 alternativas clínicas creíbles, variadas y concisas que puedan reemplazarlo.

REGLAS:
1. SOLO devuelve la lista de textos, uno por línea.
2. Sin numeración, sin explicaciones, sin comillas.
3. Deben ser verosímiles para un historial médico.

Lista de 10 alternativas:"""

    try:
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.8
        }
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                resp = await response.json()
                content = resp['choices'][0]['message']['content'].strip()
                alternatives = [line.strip() for line in content.split('\n') if line.strip()]
                # Filtrar el propio texto original si aparece
                return [a for a in alternatives if a.lower() != text.lower()][:10]
            else:
                debug_print(f"Error API ({response.status}) para '{text}'", "WARN")
    except Exception as e:
        debug_print(f"Error generando pool para '{text}': {e}", "ERROR")
    return []

async def process_document(doc_id: str, pools: Dict[Tuple[str, str], List[str]], 
                         entities_dir: Path, docs_dir: Path) -> int:
    """Procesa un documento usando el pool de alternativas pre-generadas."""
    json_path = entities_dir / f"{doc_id}.json"
    txt_path = docs_dir / f"{doc_id}.txt"
    
    if not json_path.exists() or not txt_path.exists(): return 0
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f: json_data = json.load(f)
        with open(txt_path, 'r', encoding='utf-8') as f: texto_doc = f.read()
    except: return 0

    mod_count = 0
    # Diccionario local para mantener consistencia dentro del mismo documento
    doc_replacements = {}

    for item in json_data.get("data", []):
        lbl = item.get("entity")
        txt = item.get("text")
        key = (txt, lbl)
        
        if key in pools and pools[key]:
            if key not in doc_replacements:
                doc_replacements[key] = random.choice(pools[key])
            
            new_text = doc_replacements[key]
            if new_text and new_text != txt:
                # Reemplazo en el texto del documento (solo si existe el texto literal)
                if txt in texto_doc:
                    texto_doc = texto_doc.replace(txt, new_text)
                    debug_print(f"[{doc_id}] Sustituido '{txt}' -> '{new_text}' ({lbl})", "LOG")
                    item["text"] = new_text
                    mod_count += 1

    if mod_count > 0:
        try:
            with open(json_path, 'w', encoding='utf-8') as f: json.dump(json_data, f, ensure_ascii=False, indent=2)
            with open(txt_path, 'w', encoding='utf-8') as f: f.write(texto_doc)
        except: pass
    return mod_count

async def main_async():
    api_key = load_api_key("api_keys")
    if not api_key: 
        debug_print("No se encontró API KEY.", "ERROR")
        return

    entities_dir = Path("corpus/entidades")
    docs_dir = Path("corpus/documents")
    counts_file = Path("corpus/entity_text_counts.json")

    high_freq_pairs = load_high_frequency_texts(counts_file, threshold_pct=1.0)
    if not high_freq_pairs: return

    # FASE 1: Generar Pool de Alternativas
    debug_print(f"Fase 1: Generando pool de alternativas para {len(high_freq_pairs)} textos frecuentes...", "INFO")
    pools = {}
    async with aiohttp.ClientSession() as session:
        # Procesar en pequeños lotes de 5 para no saturar la API
        sem_api = asyncio.Semaphore(5)
        async def pool_task(text, entity):
            async with sem_api:
                res = await generate_pool_for_text(session, api_key, text, entity)
                return (text, entity), res
        
        tasks = [pool_task(t, e) for t, e in high_freq_pairs]
        results = await asyncio.gather(*tasks)
        for key, alt_list in results:
            if alt_list:
                pools[key] = alt_list
                debug_print(f"Pool listo para '{key[0]}' ({len(alt_list)} opciones)", "INFO")

    if not pools:
        debug_print("No se pudieron generar pools de alternativas. Abortando.", "ERROR")
        return

    # FASE 2: Procesar Documentos
    doc_ids = [p.stem for p in entities_dir.glob("*.json")]
    debug_print(f"\nFase 2: Aplicando sustituciones en {len(doc_ids)} documentos...", "INFO")
    
    total_mods = 0
    # Procesamos uno a uno o en lotes pequeños para el "tiempo real"
    for i, doc_id in enumerate(doc_ids):
        mods = await process_document(doc_id, pools, entities_dir, docs_dir)
        total_mods += mods
        if (i+1) % 100 == 0:
            debug_print(f"Progreso global: {i+1}/{len(doc_ids)} documentos procesados...", "PROGRESS")

    debug_print(f"\nFinalizado! Se realizaron {total_mods} sustituciones en total.", "SUCCESS")

if __name__ == "__main__":
    asyncio.run(main_async())
