#!/usr/bin/env python3
"""
Comprehensive test for chunking and entity mapping in pipeline/step6.1.extract_entities_with_model

This script runs two complementary checks:
 1) Chunk integrity: verifies that token-aligned chunking (the strategy used in step6.1)
     does not split an alphanumeric code like 'G054' across two chunks. If chunking
     as implemented would split the code across chunk boundaries, the test prints
     detailed diagnostics and fails.
 2) Entity mapping: using a lightweight MockPipeline (returns an entity only when
     the ENTIRE code appears inside the chunk text), verifies that extract_entities_with_model
     correctly maps chunk-local start/end to global offsets.

The test is intentionally defensive and prints concise diagnostics to help fix
the chunking logic if it fails. Optionally the script can run the real models
(`--use-models`) to also validate the real pipelines (requires the models and
transformers in the environment).

Run examples:
  python .\scripts\test_step6_chunking_and_entities.py            # only mock/token tests
  python .\scripts\test_step6_chunking_and_entities.py --use-models --threshold 0.3

Exit codes:
  0: all checks passed
  1: failing assertion(s) — diagnostics printed

"""
import os
import sys
import argparse
from importlib.machinery import SourceFileLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
STEP6_PATH = os.path.join(REPO_ROOT, 'pipeline', 'step6.1.py')
step6 = SourceFileLoader('step6.1', STEP6_PATH).load_module()


class MockTokenizer:
    """One-token-per-character tokenizer used to exercise boundary splits deterministically."""
    def __init__(self, model_max_length=100):
        self.model_max_length = model_max_length

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False, truncation=False):
        input_ids = list(range(len(text)))
        offsets = [(i, i+1) for i in range(len(text))]
        return {'input_ids': input_ids, 'offset_mapping': offsets}


class MockPipeline:
    """Pipeline that returns an entity only when the full code string appears inside chunk_text.

    The returned entity uses chunk-local start/end (like HF pipelines do) so extract_entities_with_model
    must remap to global offsets.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, chunk_text):
        ents = []
        idx = chunk_text.find('G054')
        if idx != -1:
            ents.append({'entity_group': 'CODE', 'score': 0.95, 'word': 'G054', 'start': idx, 'end': idx + 4})
        return ents


def compute_token_chunks_from_tokenizer(tokenizer, text):
    """Replicate the token-aligned chunking used in step6.1 to inspect boundaries.

    Returns list of dicts: {start_tok, end_tok, start_char, end_char, chunk_text}
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
    tokens = encoding.get('input_ids', [])
    offsets = encoding.get('offset_mapping', [])
    model_max_length = getattr(tokenizer, 'model_max_length', 512) or 512
    token_chunk_size = min(model_max_length - 20, 512)
    token_overlap = 50

    chunks = []
    if len(tokens) <= token_chunk_size:
        start_tok = 0
        end_tok = len(tokens) - 1 if tokens else -1
        start_char = offsets[start_tok][0] if start_tok <= end_tok and offsets else 0
        end_char = offsets[end_tok][1] if end_tok >= 0 and offsets else len(text)
        chunks.append({'start_tok': start_tok, 'end_tok': end_tok, 'start_char': start_char, 'end_char': end_char, 'chunk_text': text[start_char:end_char]})
        return chunks

    step = token_chunk_size - token_overlap
    if step <= 0:
        # defensively fall back to single chunk
        chunks.append({'start_tok': 0, 'end_tok': len(tokens) - 1, 'start_char': 0, 'end_char': len(text), 'chunk_text': text})
        return chunks

    for i in range(0, len(tokens), step):
        start_tok = i
        end_tok = min(i + token_chunk_size - 1, len(tokens) - 1)
        start_char = offsets[start_tok][0]
        end_char = offsets[end_tok][1]
        chunk_text = text[start_char:end_char]
        chunks.append({'start_tok': start_tok, 'end_tok': end_tok, 'start_char': start_char, 'end_char': end_char, 'chunk_text': chunk_text})

    return chunks


def build_test_text(prefix_len, suffix_len, code='G054'):
    return ('A' * prefix_len) + code + ('B' * suffix_len)


def run_mock_chunking_entity_test(prefix_len=79, suffix_len=200):
    """Run a deterministic mock test which is expected to FAIL if chunking splits the code across chunks."""
    print('\n-- Running mock chunking + entity mapping test')
    text = build_test_text(prefix_len, suffix_len)
    # prepare mock tokenizer/pipeline
    mock_tok = MockTokenizer(model_max_length=100)
    mock_pipe = MockPipeline(mock_tok)

    # Compute chunks using the same logic used in the pipeline
    chunks = compute_token_chunks_from_tokenizer(mock_tok, text)

    # Find tokens that cover the code
    code_start = text.find('G054')
    code_end = code_start + 4
    print(f"Test text length={len(text)}; code span=({code_start},{code_end})")

    # Determine which chunks partially overlap the code (for diagnostics)
    overlapping_chunks = []
    fully_covering = []
    for ci, c in enumerate(chunks):
        if not (c['end_char'] <= code_start or c['start_char'] >= code_end):
            overlapping_chunks.append((ci, c))
        # chunk fully contains the code span
        if c['start_char'] <= code_start and c['end_char'] >= code_end:
            fully_covering.append((ci, c))

    print(f"Chunks computed: {len(chunks)}; overlapping chunks: {[ci for ci,_ in overlapping_chunks]}; fully covering chunks: {[ci for ci,_ in fully_covering]}")
    for ci, c in overlapping_chunks:
        print(f"  chunk {ci} chars=({c['start_char']},{c['end_char']}) tokens=({c['start_tok']},{c['end_tok']}) len={len(c['chunk_text'])}")

    # The code is OK if at least one chunk fully contains the entire code span.
    if len(fully_covering) == 0:
        print('\nFAIL: No single chunk fully contains G054 — chunking breaks alphanumeric tokens')
        return 1

    # Now call the actual function under test with the mock pipeline; it must detect the code
    ents = step6.extract_entities_with_model(text, mock_pipe, 'MOCK', confidence_threshold=0.5)
    print(f"extract_entities_with_model returned {len(ents)} entity(ies)")
    if not ents:
        print('\nFAIL: extract_entities_with_model did not detect G054 (likely chunk-split).')
        # print chunk neighbourhood to help debugging
        for ci, c in fully_covering:
            start = max(0, c['start_char'] - 10)
            end = min(len(text), c['end_char'] + 10)
            print(f"  chunk {ci} context: ...{text[start:end]}...")
        return 1

    # Validate mapping
    e = ents[0]
    actual = e.get('actual_text', None)
    start = e.get('start')
    end = e.get('end')
    print(f"  Detected entity actual_text={repr(actual)} span=({start},{end}) model={e.get('model')}")
    if actual != 'G054' or text[start:end] != 'G054':
        print('\nFAIL: entity mapping incorrect. expected G054 at global slice')
        return 1

    print('\nPASS: mock chunking and mapping test OK — G054 preserved and mapped correctly')
    return 0


def run_optional_model_check(text, threshold=0.3):
    """Run extract_entities_with_model using the real MEDDOCAN and CARMEN pipelines (if available).

    This step is optional and may require heavy dependencies and the models directory.
    It prints the detections but does not assert correctness automatically (because models vary).
    """
    print('\n-- Running optional real-model checks (may be slow)')
    try:
        meddocan_pipe, carmen_pipe = step6.setup_models()
    except Exception as e:
        print('Could not load real models:', e)
        return 2

    for pipe, name in ((meddocan_pipe, 'MEDDOCAN'), (carmen_pipe, 'CARMEN')):
        ents = step6.extract_entities_with_model(text, pipe, name, confidence_threshold=threshold)
        print(f"{name} returned {len(ents)} suspicious entities (threshold={threshold}):")
        for e in ents:
            print(f"  model={e.get('model')} span=({e.get('start')},{e.get('end')}) score={e.get('score'):.3f} text={repr(e.get('actual_text'))}")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Optional text file to use instead of synthetic test')
    parser.add_argument('--use-models', action='store_true', help='Run optional checks with real MEDDOCAN/CARMEN models')
    parser.add_argument('--threshold', type=float, default=0.3, help='Confidence threshold for model checks')
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print('File not found:', args.file)
            sys.exit(1)
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # default synthetic case crafted to exercise boundary-splitting
        text = build_test_text(prefix_len=79, suffix_len=200)

    rc = run_mock_chunking_entity_test(prefix_len=79, suffix_len=200)
    if rc != 0:
        print('\n=== MOCK TEST FAILED — see diagnostics above. This indicates chunking may split alphanumeric tokens. ===')
        sys.exit(1)

    if args.use_models:
        rc2 = run_optional_model_check(text, threshold=args.threshold)
        if rc2 != 0:
            print('\nNote: model checks failed or could not run.')

    print('\nAll requested checks completed successfully.')
    sys.exit(0)


if __name__ == '__main__':
    main()
