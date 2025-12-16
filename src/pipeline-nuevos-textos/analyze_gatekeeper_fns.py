#!/usr/bin/env python3
"""Analyze Gatekeeper false negatives (FN) using real corpus candidates + GT.

Goal
- Identify FN patterns specifically attributable to the SetFit Gatekeeper stage.
- FN here = candidate that matches a GT entity (PII) but Gatekeeper predicts RUIDO.

Inputs
- candidates JSON: outputs/temp_pipeline_input.json (entities detected by upstream NER)
- ground truth JSON: outputs/combined_entidades_ANTIGUO.json (entities per document)
- documents dir: corpus/ANTIGUO/documents (to extract sentence context)

Output
- outputs/gatekeeper_fn_analysis.json (summary + grouped stats + examples)

This script does NOT modify the pipeline; it only analyzes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SENTENCE_DELIMITERS = re.compile(r"[.!?;:\n]")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    # remove common markup noise observed in outputs
    text = text.replace("**", "")
    text = text.replace("<>", "")
    return text


def token_set(text: str) -> set:
    """Tokenize to a simple alnum-ish set (Spanish friendly)."""
    text = normalize_text(text)
    return set(re.findall(r"[a-z0-9áéíóúñü]+", text)) if text else set()


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def fuzzy_match_pred_to_gt(pred_text: str, gt_texts: List[str], threshold: float = 0.6) -> Optional[str]:
    """Returns the matched GT text if pred matches any GT (fuzzy)."""
    pred_norm = normalize_text(pred_text)
    if not pred_norm:
        return None

    # exact / substring
    for gt in gt_texts:
        gt_norm = normalize_text(gt)
        if not gt_norm:
            continue
        if pred_norm == gt_norm:
            return gt
        if pred_norm in gt_norm:
            # allow reasonably sized fragment
            if len(pred_norm) >= 4 or (len(gt_norm) > 0 and (len(pred_norm) / len(gt_norm)) >= 0.2):
                return gt
        if gt_norm in pred_norm:
            if len(gt_norm) >= 4 or (len(pred_norm) > 0 and (len(gt_norm) / len(pred_norm)) >= 0.5):
                return gt

    # token overlap (robust to punctuation)
    pred_tokens = set(re.findall(r"[a-z0-9áéíóúñü]+", pred_norm))
    if not pred_tokens:
        return None

    best_gt = None
    best_score = 0.0
    for gt in gt_texts:
        gt_norm = normalize_text(gt)
        gt_tokens = set(re.findall(r"[a-z0-9áéíóúñü]+", gt_norm))
        score = jaccard(pred_tokens, gt_tokens)
        if score > best_score:
            best_score = score
            best_gt = gt

    if best_score >= threshold:
        return best_gt
    return None


def find_documents_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Documents dir not found: {p}")

    candidates = [
        Path("corpus/ANTIGUO/documents"),
        Path("corpus/documents"),
        Path("documents"),
        Path("corpus/output"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No documents dir found. Pass --documents-dir")


def load_doc_text(documents_dir: Path, doc_id: str) -> Optional[str]:
    txt = documents_dir / f"{doc_id}.txt"
    if not txt.exists():
        return None
    try:
        return txt.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return txt.read_text(encoding="latin-1", errors="replace")


def extract_sentence_context(doc_text: str, start: int, end: int, window_chars: int = 150) -> str:
    """Extract sentence around [start,end) using delimiters, fallback to window."""
    if not doc_text:
        return ""
    start = max(0, min(start, len(doc_text)))
    end = max(0, min(end, len(doc_text)))
    if start > end:
        start, end = end, start

    # sentence boundaries by nearest delimiter
    left = doc_text.rfind(".", 0, start)
    left = max(left, doc_text.rfind("\n", 0, start))
    left = max(left, doc_text.rfind("!", 0, start))
    left = max(left, doc_text.rfind("?", 0, start))
    left = max(left, doc_text.rfind(";", 0, start))
    left = max(left, doc_text.rfind(":", 0, start))
    left = 0 if left < 0 else left + 1

    right_candidates = [
        doc_text.find(".", end),
        doc_text.find("\n", end),
        doc_text.find("!", end),
        doc_text.find("?", end),
        doc_text.find(";", end),
        doc_text.find(":", end),
    ]
    right = min([c for c in right_candidates if c != -1], default=-1)
    right = len(doc_text) if right == -1 else right

    sentence = doc_text[left:right].strip()
    if len(sentence) < 10:
        # fallback: fixed window
        l2 = max(0, start - window_chars)
        r2 = min(len(doc_text), end + window_chars)
        sentence = doc_text[l2:r2].strip()

    sentence = re.sub(r"\s+", " ", sentence)
    return sentence


def detect_pattern(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return "empty"
    if re.search(r"@", t) and re.search(r"\.", t):
        return "email_like"
    if re.search(r"https?://", t) or re.search(r"www\.", t):
        return "url_like"
    if re.fullmatch(r"[0-9]{8}[a-z]", t):
        return "dni_like"
    if re.search(r"\d", t) and re.search(r"[a-z]", t):
        return "alnum_mix"
    if re.search(r"\d", t):
        return "has_digit"
    if len(t.split()) == 1:
        return "single_token"
    return "multi_token"


@dataclass
class Example:
    doc_id: str
    label: str
    entity_text: str
    gt_match: str
    start: int
    end: int
    context: str
    p_pii: float
    predicted_is_pii: bool


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Gatekeeper FNs (candidate matches GT but predicted noise)")
    ap.add_argument(
        "--mode",
        choices=["gt"],
        default="gt",
        help=(
            "Modo de análisis. 'gt' = FN si una entidad del ground truth es clasificada como ruido por el gatekeeper. "
            "(Es el caso que describe el problema: entidades presentes en GT que el gatekeeper filtra.)"
        ),
    )
    ap.add_argument("--ground-truth", default="outputs/combined_entidades_ANTIGUO.json")
    ap.add_argument("--documents-dir", default=None)
    ap.add_argument("--model-path", default="models/gatekeeper_setfit_v2")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--window-chars", type=int, default=150)
    ap.add_argument(
        "--max-positives",
        type=int,
        default=20000,
        help="Máximo número de entidades GT a evaluar (muestreo determinista). 0 = sin límite.",
    )
    ap.add_argument(
        "--sample-mod",
        type=int,
        default=1,
        help="Submuestreo determinista: conserva 1 de cada N (default: 1 = todo).",
    )
    ap.add_argument("--max-examples", type=int, default=2000, help="Limit stored FN examples")
    ap.add_argument("--output", default="outputs/gatekeeper_fn_analysis.json")
    args = ap.parse_args()

    gt_path = Path(args.ground_truth)
    documents_dir = find_documents_dir(args.documents_dir)

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    # Load GT map: doc_id -> list of gt texts (PII)
    gt_obj = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_map: Dict[str, List[str]] = {}
    gt_label_map: Dict[Tuple[str, str], str] = {}
    for item in gt_obj.get("combined", []):
        doc_id = item.get("doc_id")
        data = (item.get("entities") or {}).get("data") or []
        texts = []
        for e in data:
            txt = e.get("text")
            lab = e.get("entity")
            if txt:
                texts.append(txt)
                gt_label_map[(doc_id, normalize_text(txt))] = lab
        gt_map[doc_id] = texts

    # Lazy import SetFit
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(args.model_path)

    batch_size = 256
    batch_inputs: List[str] = []
    batch_meta: List[Tuple[str, str, str, str, int, int, str]] = []
    # meta tuple: (doc_id, label, entity_text, gt_match, start, end, context)

    fn_examples: List[Example] = []
    counts_by_label = Counter()
    counts_by_pattern = Counter()
    counts_by_len_bucket = Counter()
    counts_by_token_bucket = Counter()

    total_pos = 0
    total_fn = 0

    # Cache doc texts (avoid re-reading a lot)
    doc_cache: Dict[str, Optional[str]] = {}

    def flush_batch():
        nonlocal total_fn
        if not batch_inputs:
            return

        probas = model.predict_proba(batch_inputs)
        for (doc_id, label, entity_text, gt_match, start, end, context), proba in zip(batch_meta, probas):
            p1 = float(proba[1]) if len(proba) > 1 else float(max(proba))
            pred_is_pii = p1 >= args.threshold

            if not pred_is_pii:
                total_fn += 1
                counts_by_label[label] += 1
                pat = detect_pattern(entity_text)
                counts_by_pattern[pat] += 1
                l = len(normalize_text(entity_text))
                counts_by_len_bucket[len_bucket(l)] += 1
                tok = len(normalize_text(entity_text).split())
                counts_by_token_bucket[token_bucket(tok)] += 1

                if len(fn_examples) < args.max_examples:
                    fn_examples.append(
                        Example(
                            doc_id=doc_id,
                            label=label,
                            entity_text=entity_text,
                            gt_match=gt_match,
                            start=start,
                            end=end,
                            context=context,
                            p_pii=p1,
                            predicted_is_pii=False,
                        )
                    )

        batch_inputs.clear()
        batch_meta.clear()

    def len_bucket(n: int) -> str:
        if n <= 3:
            return "1-3"
        if n <= 6:
            return "4-6"
        if n <= 10:
            return "7-10"
        if n <= 20:
            return "11-20"
        return "21+"

    def token_bucket(n: int) -> str:
        if n <= 1:
            return "1"
        if n == 2:
            return "2"
        if n <= 4:
            return "3-4"
        if n <= 8:
            return "5-8"
        return "9+"

    # GT-only mode: iterate real GT entities, locate them in documents to extract context
    seen_pos = 0
    for item in gt_obj.get("combined", []):
        doc_id = item.get("doc_id")
        if not doc_id:
            continue

        if doc_id not in doc_cache:
            doc_cache[doc_id] = load_doc_text(documents_dir, doc_id)
        doc_text = doc_cache[doc_id] or ""

        data = (item.get("entities") or {}).get("data") or []
        for e in data:
            entity_text = e.get("text") or ""
            label = e.get("entity") or "UNKNOWN"

            # deterministic subsampling (fast + reproducible)
            if args.sample_mod and args.sample_mod > 1:
                key = f"{doc_id}|{entity_text}".encode("utf-8", errors="ignore")
                if (sum(key) % args.sample_mod) != 0:
                    continue

            if args.max_positives and args.max_positives > 0 and seen_pos >= args.max_positives:
                break

            total_pos += 1
            seen_pos += 1

            # Find span in doc to extract a sentence; if not found, fallback to window around 0
            start = -1
            end = -1
            if doc_text and entity_text:
                # try exact then case-insensitive
                start = doc_text.find(entity_text)
                if start == -1:
                    start = doc_text.lower().find(entity_text.lower())
                if start != -1:
                    end = start + len(entity_text)

            if start == -1:
                start = 0
                end = 0

            context = extract_sentence_context(doc_text, start, end, window_chars=args.window_chars)
            input_text = f"ENTITY: {entity_text}\nSENTENCE: {context}"

            batch_inputs.append(input_text)
            batch_meta.append((doc_id, label, entity_text, entity_text, start, end, context))
            if len(batch_inputs) >= batch_size:
                flush_batch()

        if args.max_positives and args.max_positives > 0 and seen_pos >= args.max_positives:
            break

    flush_batch()

    fn_rate = (total_fn / total_pos) if total_pos else 0.0

    # add some top terms for quick pattern intuition
    top_fn_terms = Counter([normalize_text(e.entity_text) for e in fn_examples]).most_common(30)

    out = {
        "inputs": {
            "ground_truth": str(gt_path),
            "documents_dir": str(documents_dir),
            "model_path": args.model_path,
            "threshold": args.threshold,
            "window_chars": args.window_chars,
            "mode": args.mode,
        },
        "summary": {
            "total_positive_candidates": total_pos,
            "total_gatekeeper_fn": total_fn,
            "fn_rate": fn_rate,
        },
        "groupings": {
            "by_label": counts_by_label.most_common(40),
            "by_pattern": counts_by_pattern.most_common(20),
            "by_length_bucket": counts_by_len_bucket.most_common(),
            "by_token_bucket": counts_by_token_bucket.most_common(),
            "top_fn_terms": top_fn_terms,
        },
        "examples": [
            {
                "doc_id": e.doc_id,
                "label": e.label,
                "entity_text": e.entity_text,
                "gt_match": e.gt_match,
                "start": e.start,
                "end": e.end,
                "p_pii": e.p_pii,
                "context": e.context,
            }
            for e in fn_examples[: min(200, len(fn_examples))]
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Gatekeeper FN Analysis]")
    print(f"  positives (GT):       {total_pos}")
    print(f"  FN (pred=RUIDO):     {total_fn}  ({fn_rate:.2%})")
    print(f"  saved: {out_path}")
    print("  top labels:")
    for lab, c in counts_by_label.most_common(10):
        print(f"    {lab:30s} {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
