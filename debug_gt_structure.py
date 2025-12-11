#!/usr/bin/env python3
import json
from pathlib import Path

gt_path = Path('outputs/combined_entidades_ANTIGUO.json')
if gt_path.exists():
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    
    print(f"GT Type: {type(gt).__name__}")
    print(f"GT size: {len(gt)} items")
    
    if isinstance(gt, dict):
        keys_sample = list(gt.keys())[:3]
        print(f"\nSample keys (first 3): {keys_sample}")
        
        for k in keys_sample:
            v = gt[k]
            print(f"\nKey: '{k[:50]}'")
            print(f"  Value type: {type(v).__name__}")
            if isinstance(v, list):
                print(f"  Length: {len(v)}")
                if len(v) > 0:
                    print(f"  First item type: {type(v[0]).__name__}")
                    if isinstance(v[0], str):
                        print(f"  First item (text): {v[0][:100]}")
                    elif isinstance(v[0], dict):
                        print(f"  First item keys: {list(v[0].keys())}")
                        print(f"  First item: {v[0]}")
            elif isinstance(v, dict):
                print(f"  Keys: {list(v.keys())[:5]}")
                if 'entities' in v:
                    print(f"  Has 'entities' field with {len(v.get('entities', []))} items")
                if 'data' in v:
                    print(f"  Has 'data' field with {len(v.get('data', []))} items")
    elif isinstance(gt, list):
        print(f"\nList with {len(gt)} items")
        if len(gt) > 0:
            print(f"First item type: {type(gt[0]).__name__}")
            print(f"First item: {str(gt[0])[:200]}")
else:
    print("GT file not found!")
