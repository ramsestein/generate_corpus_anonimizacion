#!/usr/bin/env python3
"""FN analysis for SetFit Gatekeeper using precomputed SetFit CSV outputs.

Why this exists
- Running SetFit over ~100k+ examples is slow.
- The repo already contains a CSV with contextualized inputs + SetFit predictions:
    outputs/setfit_context_resultados_*.csv

We use that CSV + GT to find gatekeeper-specific FNs:
- FN_gatekeeper = entity is in GT for that doc (PII) BUT gatekeeper would NOT keep it.
- gatekeeper_keep rule (matches runtime in setfit_module/gatekeeper.py):
    keep iff (setfit_prediction==1 AND confidence >= threshold)

Outputs
- outputs/gatekeeper_fn_analysis_from_csv.json

Notes
- This attributes errors to the gatekeeper decision (thresholding), not to upstream NER coverage.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("**", "")
    text = text.replace("<>", "")
    return text


def detect_pattern(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return "empty"
    if re.search(r"https?://", t) or re.search(r"www\.", t):
        return "url_like"
    if re.search(r"@", t) and re.search(r"\.", t):
        return "email_like"
    if re.fullmatch(r"[0-9]{8}[a-z]", t):
        return "dni_like"
    if re.search(r"\d", t) and re.search(r"[a-z]", t):
        return "alnum_mix"
    if re.search(r"\d", t):
        return "has_digit"
    if len(t.split()) == 1:
        return "single_token"
    return "multi_token"


def fuzzy_match_pred_to_gt(pred_text: str, gt_texts: List[str], threshold: float = 0.6) -> Optional[str]:
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
            if len(pred_norm) >= 4 or (len(gt_norm) > 0 and (len(pred_norm) / len(gt_norm)) >= 0.2):
                return gt
        if gt_norm in pred_norm:
            if len(gt_norm) >= 4 or (len(pred_norm) > 0 and (len(gt_norm) / len(pred_norm)) >= 0.5):
                return gt

    # token overlap
    pred_tokens = set(re.findall(r"[a-z0-9áéíóúñü]+", pred_norm))
    if not pred_tokens:
        return None

    best_gt = None
    best_score = 0.0
    for gt in gt_texts:
        gt_norm = normalize_text(gt)
        gt_tokens = set(re.findall(r"[a-z0-9áéíóúñü]+", gt_norm))
        if not gt_tokens:
            continue
        inter = len(pred_tokens & gt_tokens)
        union = len(pred_tokens | gt_tokens)
        score = inter / union if union else 0.0
        if score > best_score:
            best_score = score
            best_gt = gt

    if best_score >= threshold:
        return best_gt
    return None


def build_gt_maps(gt_path: Path) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], str]]:
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

    return gt_map, gt_label_map


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Gatekeeper FNs from precomputed SetFit CSV")
    ap.add_argument("--setfit-csv", default="outputs/setfit_context_resultados_20251202_081634.csv")
    ap.add_argument("--ground-truth", default="outputs/combined_entidades_ANTIGUO.json")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--output", default="outputs/gatekeeper_fn_analysis_from_csv.json")
    ap.add_argument("--max-examples", type=int, default=200)
    args = ap.parse_args()

    csv_path = Path(args.setfit_csv)
    gt_path = Path(args.ground_truth)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    gt_map, gt_label_map = build_gt_maps(gt_path)

    df = pd.read_csv(csv_path)

    required_cols = {"document_id", "entity_text", "ner_label", "setfit_prediction", "confidence", "contextualized_input"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    # Determine if candidate is GT-positive
    gt_match = []
    gt_label = []
    is_gt_pii = []

    for doc_id, entity_text in zip(df["document_id"].astype(str), df["entity_text"].astype(str)):
        texts = gt_map.get(doc_id)
        if not texts:
            gt_match.append("")
            gt_label.append("")
            is_gt_pii.append(False)
            continue

        m = fuzzy_match_pred_to_gt(entity_text, texts, threshold=0.6)
        if m:
            gt_match.append(m)
            is_gt_pii.append(True)
            gt_label.append(gt_label_map.get((doc_id, normalize_text(m)), ""))
        else:
            gt_match.append("")
            gt_label.append("")
            is_gt_pii.append(False)

    df["gt_match"] = gt_match
    df["gt_label"] = gt_label
    df["is_gt_pii"] = is_gt_pii

    # Gatekeeper decision rule
    df["gatekeeper_keep"] = (df["setfit_prediction"].astype(int) == 1) & (df["confidence"].astype(float) >= float(args.threshold))

    # Gatekeeper FN = in GT but filtered out
    df_fn = df[(df["is_gt_pii"] == True) & (df["gatekeeper_keep"] == False)].copy()

    # Groupings
    by_ner = Counter(df_fn["ner_label"].astype(str)).most_common(30)
    by_gt_label = Counter([x for x in df_fn["gt_label"].astype(str) if x]).most_common(30)
    by_pattern = Counter([detect_pattern(x) for x in df_fn["entity_text"].astype(str)]).most_common(20)

    lengths = df_fn["entity_text"].astype(str).map(lambda s: len(normalize_text(s)))
    tokens = df_fn["entity_text"].astype(str).map(lambda s: len(normalize_text(s).split()))

    def bucket_len(n: int) -> str:
        if n <= 3:
            return "1-3"
        if n <= 6:
            return "4-6"
        if n <= 10:
            return "7-10"
        if n <= 20:
            return "11-20"
        return "21+"

    def bucket_tok(n: int) -> str:
        if n <= 1:
            return "1"
        if n == 2:
            return "2"
        if n <= 4:
            return "3-4"
        if n <= 8:
            return "5-8"
        return "9+"

    by_len_bucket = Counter(lengths.map(bucket_len)).most_common()
    by_tok_bucket = Counter(tokens.map(bucket_tok)).most_common()

    # Confidence diagnostics: how far below threshold are most FNs?
    conf = df_fn["confidence"].astype(float)
    conf_summary = {
        "count": int(conf.shape[0]),
        "min": float(conf.min()) if conf.shape[0] else None,
        "p10": float(conf.quantile(0.10)) if conf.shape[0] else None,
        "p50": float(conf.quantile(0.50)) if conf.shape[0] else None,
        "p90": float(conf.quantile(0.90)) if conf.shape[0] else None,
        "max": float(conf.max()) if conf.shape[0] else None,
    }

    examples = []
    for _, row in df_fn.head(args.max_examples).iterrows():
        ci = str(row.get("contextualized_input") or "")
        sentence = ""
        if "SENTENCE:" in ci:
            sentence = ci.split("SENTENCE:", 1)[1].strip().replace("\n", " ")
        examples.append(
            {
                "doc_id": str(row["document_id"]),
                "ner_label": str(row["ner_label"]),
                "entity_text": str(row["entity_text"]),
                "gt_label": str(row.get("gt_label") or ""),
                "gt_match": str(row.get("gt_match") or ""),
                "setfit_prediction": int(row["setfit_prediction"]),
                "confidence": float(row["confidence"]),
                "sentence": sentence,
            }
        )

    out = {
        "inputs": {
            "setfit_csv": str(csv_path),
            "ground_truth": str(gt_path),
            "threshold": float(args.threshold),
            "rule": "keep iff (pred==1 and confidence>=threshold)",
        },
        "summary": {
            "rows_total": int(df.shape[0]),
            "rows_in_gt": int(df["is_gt_pii"].sum()),
            "gatekeeper_fn": int(df_fn.shape[0]),
            "gatekeeper_fn_rate_over_gt": float(df_fn.shape[0] / max(1, int(df["is_gt_pii"].sum()))),
        },
        "confidence_summary_fn": conf_summary,
        "groupings": {
            "by_ner_label": by_ner,
            "by_gt_label": by_gt_label,
            "by_pattern": by_pattern,
            "by_length_bucket": by_len_bucket,
            "by_token_bucket": by_tok_bucket,
        },
        "examples": examples,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Gatekeeper FN Analysis from CSV]")
    print(f"  total rows: {df.shape[0]}")
    print(f"  rows in GT: {int(df['is_gt_pii'].sum())}")
    print(f"  gatekeeper FN: {df_fn.shape[0]} (rate={out['summary']['gatekeeper_fn_rate_over_gt']:.2%})")
    print(f"  saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
