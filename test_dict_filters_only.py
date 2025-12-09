#!/usr/bin/env python3
"""
Test script to run ONLY dictionary filters on entities.
Skips SetFit and LLM - directly applies whitelist/blacklist filtering.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "src" / "pipeline-nuevos-textos"))

from dict_filters import apply_dict_filters
from io_json import load_entities

def main():
    print("=" * 70)
    print("TEST: DICTIONARY FILTERS ONLY")
    print("=" * 70)
    
    # Load entities
    input_file = SCRIPT_DIR / "entidades-rpueba.json"
    print(f"\n[1/3] Loading entities from: {input_file}")
    entities = load_entities(str(input_file))
    print(f"      Loaded {len(entities)} entities")
    
    # Configure dict filters
    config = {
        "json_whitelist_paths": [
            str(SCRIPT_DIR / "data" / "hospitales.json"),
            str(SCRIPT_DIR / "data" / "lugares.json"),
        ],
        "json_blacklist_paths": [
            str(SCRIPT_DIR / "data" / "medicamentos.json"),
            str(SCRIPT_DIR / "data" / "patologias.json"),
        ],
        "csv_base_path": str(SCRIPT_DIR / "LISTAS"),
        "cie10_path": str(SCRIPT_DIR / "LISTAS" / "cie10.xls"),
        "min_length_per_label": {
            "NUMERO_IDENTIF": 1,
            "FECHAS": 4,
            "NUMERO_TELEFONO": 4,
            "NUMERO_FAX": 4,
        },
        "ignore_single_char": True,
        "ignore_numeric_only": False,
    }
    
    print(f"\n[2/3] Applying dictionary filters...")
    print(f"      Whitelist sources: {len(config['json_whitelist_paths'])} JSON files")
    print(f"      Blacklist sources: {len(config['json_blacklist_paths'])} JSON files")
    print(f"      CSV base path: {config['csv_base_path']}")
    
    # Apply filters
    results = apply_dict_filters(entities, None, config)
    
    # Analyze results
    print(f"\n[3/3] Analyzing results...")
    
    kept = [e for e in results if e.get("decision") == "KEEP"]
    filtered = [e for e in results if e.get("decision") == "FILTER"]
    escalated = [e for e in results if e.get("decision") == "ESCALATE"]
    
    print(f"\n      Total processed: {len(results)}")
    print(f"      ✓ KEEP (rescued by whitelist): {len(kept)}")
    print(f"      ✗ FILTER (blocked by blacklist): {len(filtered)}")
    print(f"      ? ESCALATE (no match, needs LLM): {len(escalated)}")
    
    # Save results
    output_file = SCRIPT_DIR / "outputs" / "dict_filters_only_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source_file": str(input_file),
            "total_entities": len(results),
            "filter": "dict_only",
        },
        "statistics": {
            "total": len(results),
            "kept": len(kept),
            "filtered": len(filtered),
            "escalated": len(escalated),
        },
        "entities": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n      Results saved to: {output_file}")
    
    # Show sample errors
    print(f"\n" + "=" * 70)
    print("SAMPLE FILTERED ENTITIES (False Negatives - should be kept):")
    print("=" * 70)
    
    for i, entity in enumerate(filtered[:10], 1):
        print(f"\n{i}. Text: '{entity.get('entity_text', entity.get('text', 'N/A'))}'")
        print(f"   Label: {entity.get('label', 'N/A')}")
        print(f"   Matched: {entity.get('matched_list', 'N/A')} - {entity.get('matched_term', 'N/A')}")
        print(f"   Reason: {entity.get('filter_reason', 'N/A')}")
    
    print(f"\n" + "=" * 70)
    print("SAMPLE KEPT ENTITIES (rescued by whitelist):")
    print("=" * 70)
    
    for i, entity in enumerate(kept[:10], 1):
        print(f"\n{i}. Text: '{entity.get('entity_text', entity.get('text', 'N/A'))}'")
        print(f"   Label: {entity.get('label', 'N/A')}")
        print(f"   Matched: {entity.get('matched_list', 'N/A')} - {entity.get('matched_term', 'N/A')}")
    
    print(f"\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
