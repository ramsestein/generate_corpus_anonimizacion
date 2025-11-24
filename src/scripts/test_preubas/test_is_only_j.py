#!/usr/bin/env python3
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent
step6 = SourceFileLoader('step6.1', str(REPO_ROOT / 'pipeline' / 'step6.1.py')).load_module()

cases = [
    'JJJ',
    'JJJ\nEUR',
    'J J J',
    ' Maria ',
    'JJ',
    'J',
    '',
    'G054',
]

for c in cases:
    print(repr(c), '->', step6.is_only_j_characters(c))
