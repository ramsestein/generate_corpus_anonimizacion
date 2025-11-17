#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparador de chunking: compara detecciones NER entre chunking alineado a tokens y chunking por caracteres.
Uso:
    python ./scripts/compare_chunking.py --text "El paciente ingresó..."
    python ./scripts/compare_chunking.py --doc-file ./corpus/step5_anonymized_documents/aws2_anonimizado/NHC....txt

Salida: imprime entidades por chunk y un resumen de diferencias (solo-token, solo-char).
"""
import argparse
import os
import json
import re
from importlib.machinery import SourceFileLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
STEP6_PATH = os.path.join(REPO_ROOT, 'pipeline', 'step6.1.py')
STEP6V_PATH = os.path.join(REPO_ROOT, 'pipeline', 'step6_validation.py')

if not os.path.exists(STEP6_PATH):
    print('ERROR: no se encontró pipeline/step6.1.py')
    raise SystemExit(1)

step6 = SourceFileLoader('step6_mod', STEP6_PATH).load_module()
step6v = SourceFileLoader('step6v_mod', STEP6V_PATH).load_module()


def build_char_chunks(text, max_chunk_chars=1800, overlap_chars=300):
    chunks = []
    if len(text) <= max_chunk_chars:
        return [(text, 0)]
    start_idx = 0
    text_len = len(text)
    while start_idx < text_len:
        end_idx = min(start_idx + max_chunk_chars, text_len)
        chunks.append((text[start_idx:end_idx], start_idx))
        if end_idx == text_len:
            break
        start_idx = end_idx - overlap_chars
    return chunks


def build_chunks_step6_1(text):
    """Reproduce la lógica de chunking que usa pipeline/step6.1.py
    Usa split por palabras, chunk_size = max_length//2 con max_length=512,
    y overlap de 50 palabras (se une chunk_size + 50 palabras).
    Devuelve lista de (chunk_text, chunk_start_word_index).
    """
    # Produce chunks by WORD COUNT but preserve the exact character spans
    # from the original text. This avoids losing newlines/multiple-spaces and
    # ensures the chunk_text is an exact substring of `text` and the
    # returned start index is the correct character offset.
    max_length = 512
    if len(text) <= max_length:
        return [(text, 0)]

    # Find word spans (non-whitespace sequences) with their character offsets
    word_spans = [m.span() for m in re.finditer(r"\S+", text)]
    num_words = len(word_spans)
    chunk_size = max(1, max_length // 2)
    overlap = 50

    chunks = []
    i = 0
    while i < num_words:
        start_word = i
        end_word = min(i + chunk_size + overlap - 1, num_words - 1)

        char_start = word_spans[start_word][0]
        char_end = word_spans[end_word][1]

        chunk_text = text[char_start:char_end]
        chunks.append((chunk_text, char_start))

        if end_word == num_words - 1:
            break

        i += chunk_size

    return chunks


def build_chunks_step6_validation(text):
    """Reproduce la lógica de chunking que usa pipeline/step6_validation.py.
    IMPORTANTE: El archivo `step6_validation.py` acumula el offset usando
    el número de palabras (offset += len(chunk.split())) en lugar de
    offsets en caracteres. Para reproducir exactamente su comportamiento
    devolvemos (chunk_text, chunk_start) donde chunk_start es el número
    acumulado de palabras procesadas (no un offset en caracteres).
    Esto permite replicar las posiciones erróneas que su código produciría
    y comparar resultados tal y como saldrían de ese script.
    """
    max_length = 512
    if len(text) <= max_length:
        return [(text, 0)]

    words = text.split()
    chunk_size = max(1, max_length // 2)
    overlap = 50

    chunks = []
    i = 0
    cumulative_words = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size + overlap]
        chunk_text = " ".join(chunk_words)

        # Here we RETURN chunk_start as the cumulative number of words seen so far
        # to mimic `offset += len(chunk.split())` used in step6_validation.py
        chunks.append((chunk_text, cumulative_words))

        cumulative_words += len(chunk_words)
        if i + chunk_size + overlap >= len(words):
            break
        i += chunk_size

    return chunks


def build_token_chunks(text, tokenizer, token_chunk_size=512, token_overlap=50):
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.get('offset_mapping', [])
    if not offsets:
        raise ValueError('tokenizer no devolvió offsets')
    num_tokens = len(offsets)
    step = max(1, token_chunk_size - token_overlap)
    chunks = []
    i = 0
    while i < num_tokens:
        start_tok = i
        end_tok = min(i + token_chunk_size, num_tokens)
        # char start
        char_start = offsets[start_tok][0]
        # char end: find last token with end>0
        j = end_tok - 1
        char_end = offsets[j][1]
        while j >= start_tok and char_end == 0:
            j -= 1
            if j >= start_tok:
                char_end = offsets[j][1]
        if char_end == 0:
            char_end = offsets[end_tok - 1][1]
        if char_end <= char_start:
            i += step
            continue
        chunks.append((text[char_start:char_end], char_start))
        if end_tok >= num_tokens:
            break
        i += step
    return chunks


def run_on_chunks(chunks, pipeline_model, confidence_threshold=0.1):
    all_entities = []
    for idx, (chunk_text, chunk_start) in enumerate(chunks):
        ents = pipeline_model(chunk_text)
        # normalize
        for e in (ents or []):
            try:
                s = int(e.get('start', 0)) + chunk_start
                ed = int(e.get('end', 0)) + chunk_start
            except Exception:
                continue
            label = e.get('entity_group', e.get('label', ''))
            score = float(e.get('score', e.get('confidence', 0.0) or 0.0))
            word = e.get('word', '')
            if score < confidence_threshold:
                continue
            all_entities.append({
                'start': s,
                'end': ed,
                'label': label,
                'score': score,
                'word': word,
                'chunk_id': idx
            })
    return all_entities


def normalize_entities(ent_list):
    # return set of tuples for comparison
    s = set()
    for e in ent_list:
        s.add((e['label'], int(e['start']), int(e['end']), e.get('word','').strip()))
    return s


def pretty_print_entities(ents, text, title):
    print('\n' + '='*30)
    print(title)
    print('='*30)
    for e in ents:
        snippet = text[e['start']:e['end']] if 0 <= e['start'] < e['end'] <= len(text) else e.get('word','')
        print(f"- {e['label']} [{e['start']}-{e['end']}] score={e['score']:.3f} snippet='{snippet}' chunk={e.get('chunk_id')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc-file', type=str, default=None)
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--confidence', type=float, default=0.1)
    args = parser.parse_args()

    # If neither --text nor --doc-file is provided, try to read the sample file scripts/prueba.txt
    sample_path = os.path.join(os.path.dirname(__file__), 'prueba.txt')

    if args.doc_file:
        if not os.path.exists(args.doc_file):
            print('Archivo no encontrado:', args.doc_file)
            raise SystemExit(1)
        with open(args.doc_file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        if os.path.exists(sample_path):
            with open(sample_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f'Leyendo texto de muestra: {sample_path}')
        else:
            print('Provide --text or --doc-file, or create scripts/prueba.txt')
            raise SystemExit(1)

    med, car = step6.setup_models()
    models = [(med, 'MEDDOCAN'), (car, 'CARMEN')]

    for pipeline_model, model_name in models:
        print('\n' + '#' * 60)
        print(f'Procesando con modelo: {model_name}')
        print('#' * 60)

        # token chunks
        tokenizer = getattr(pipeline_model, 'tokenizer', None)
        if tokenizer is None:
            print(f'Tokenizer no disponible en el pipeline {model_name}; saltando token-chunks')
            model_max_len = 512
            token_chunk_size = 512
            token_overlap = 50
            token_chunks = []
        else:
            model_max_len = getattr(tokenizer, 'model_max_length', None) or getattr(tokenizer, 'model_max_len', 512)
            token_chunk_size = min(max(int(model_max_len) - 20, 1), 512)
            token_overlap = 50
            token_chunks = build_token_chunks(text, tokenizer, token_chunk_size=token_chunk_size, token_overlap=token_overlap)

        char_chunks = build_char_chunks(text, max_chunk_chars=1800, overlap_chars=300)
        # Use the chunk builders from the actual pipeline modules so behavior
        # matches the real scripts (token-aligned / word-split differences).
        try:
            step6_chunks = step6.build_word_split_chunks(text)
        except Exception:
            # fallback to local implementation if missing
            step6_chunks = build_chunks_step6_1(text)

        try:
            # step6_validation uses word-count-based offsets; call its helper
            step6v_chunks = step6v.build_chunks_wordcount_offsets(text)
        except Exception:
            # fallback to local behavior that mimics it
            step6v_chunks = build_chunks_step6_validation(text)

        print(f'Token chunks: {len(token_chunks)}, Char chunks: {len(char_chunks)}, step6.1 chunks: {len(step6_chunks)}, step6_validation chunks: {len(step6v_chunks)}')

        token_entities = run_on_chunks(token_chunks, pipeline_model, confidence_threshold=args.confidence) if token_chunks else []
        char_entities = run_on_chunks(char_chunks, pipeline_model, confidence_threshold=args.confidence)
        step6_entities = run_on_chunks(step6_chunks, pipeline_model, confidence_threshold=args.confidence)
        step6v_entities = run_on_chunks(step6v_chunks, pipeline_model, confidence_threshold=args.confidence)

        pretty_print_entities(token_entities, text, f'{model_name} - Token-aligned entities')
        pretty_print_entities(char_entities, text, f'{model_name} - Char-aligned entities')
        pretty_print_entities(step6_entities, text, f'{model_name} - step6.1 (word-split) entities')
        pretty_print_entities(step6v_entities, text, f'{model_name} - step6_validation (word-split) entities')

        set_token = normalize_entities(token_entities)
        set_char = normalize_entities(char_entities)
        set_step6 = normalize_entities(step6_entities)
        set_step6v = normalize_entities(step6v_entities)

        # Pairwise diffs
        def print_diff(a_set, b_set, a_name, b_name):
            only_a = a_set - b_set
            only_b = b_set - a_set
            print(f"\n---- DIFF {a_name} vs {b_name} ----")
            print(f"{a_name}: {len(a_set)} entities, {b_name}: {len(b_set)} entities")
            print(f"Only in {a_name}: {len(only_a)}")
            for item in list(only_a)[:20]:
                print(f"  {a_name}_ONLY:", item)
            print(f"Only in {b_name}: {len(only_b)}")
            for item in list(only_b)[:20]:
                print(f"  {b_name}_ONLY:", item)

        # Show diffs of interest
        if token_chunks:
            print_diff(set_token, set_step6, f'{model_name}-TOKEN', f'{model_name}-STEP6')
            print_diff(set_char, set_step6, f'{model_name}-CHAR', f'{model_name}-STEP6')
        else:
            print_diff(set_char, set_step6, f'{model_name}-CHAR', f'{model_name}-STEP6')
        print_diff(set_step6, set_step6v, f'{model_name}-STEP6', f'{model_name}-STEP6V')

if __name__ == '__main__':
    main()
