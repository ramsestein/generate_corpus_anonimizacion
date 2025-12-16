#!/usr/bin/env python3
"""Train a recall-optimized SetFit Gatekeeper model.

This script is intentionally aligned with production inference input format:

  ENTITY: <entity_text>\nSENTENCE: <context>

It starts from an existing base dataset (typically synthetic) and can inject
"hard positives" mined from real Gatekeeper false negatives.

Outputs:
- A SetFit model directory compatible with `SetFitModel.from_pretrained()`.
- `training_metadata.json` with metrics and a recommended decision threshold.

Example:
  python src/pipeline-nuevos-textos/train_gatekeeper_recall.py \
    --base-csv audit/training_dataset.csv \
    --fn-analysis-json outputs/gatekeeper_fn_analysis_sample20k.json \
    --max-fn-examples 5000 \
    --repeat-fn 3 \
    --output-dir models/gatekeeper_setfit_recall_v1 \
    --target-precision 0.75
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


BRACKET_SPAN_RE = re.compile(r"\[(.+?)\]")


def _format_input(entity_text: str, sentence_context: str) -> str:
    entity_text = (entity_text or "").strip()
    sentence_context = (sentence_context or "").strip()
    return f"ENTITY: {entity_text}\nSENTENCE: {sentence_context}"


def _row_to_entity_and_sentence(text: str) -> Tuple[str, str]:
    """Extract the first bracketed span as entity; sentence is text with brackets removed."""
    text = (text or "").strip()
    if not text:
        return "", ""

    m = BRACKET_SPAN_RE.search(text)
    if not m:
        # No explicit entity marker; keep sentence and use a placeholder entity.
        return "<NO_ENTITY>", text

    entity = m.group(1).strip()
    # Replace only the first bracketed instance to avoid altering other placeholders too much.
    sentence = text[: m.start()] + entity + text[m.end() :]
    return entity, sentence


def load_base_dataset_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must contain columns 'text' and 'label': {path}")
    df = df[["text", "label"] + (["category"] if "category" in df.columns else [])].copy()
    df["label"] = df["label"].astype(int)
    return df


def downsample_base_df(
    base_df: pd.DataFrame,
    base_pos_max: Optional[int],
    base_neg_max: Optional[int],
    seed: int,
) -> pd.DataFrame:
    if base_pos_max is None and base_neg_max is None:
        return base_df

    pos_df = base_df[base_df["label"] == 1]
    neg_df = base_df[base_df["label"] == 0]

    if base_pos_max is not None:
        pos_df = pos_df.sample(n=min(int(base_pos_max), len(pos_df)), random_state=seed)
    if base_neg_max is not None:
        neg_df = neg_df.sample(n=min(int(base_neg_max), len(neg_df)), random_state=seed)

    out = pd.concat([pos_df, neg_df], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def load_fn_examples(fn_analysis_json: Path) -> List[Dict[str, Any]]:
    with open(fn_analysis_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"FN analysis JSON does not contain a list 'examples': {fn_analysis_json}")
    return examples


def build_training_rows(
    base_df: pd.DataFrame,
    fn_examples: Optional[List[Dict[str, Any]]],
    max_fn_examples: int,
    repeat_fn: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)

    rows: List[Dict[str, Any]] = []

    # Base dataset
    for _, r in base_df.iterrows():
        entity, sentence = _row_to_entity_and_sentence(str(r.get("text", "")))
        rows.append(
            {
                "text": _format_input(entity, sentence),
                "label": int(r.get("label", 0)),
                "source": "base_csv",
            }
        )

    # Hard positives from FN analysis
    if fn_examples:
        # Prefer sampling across labels by shuffling the whole list.
        fn_examples = list(fn_examples)
        rng.shuffle(fn_examples)
        fn_examples = fn_examples[: max(0, int(max_fn_examples))]

        for ex in fn_examples:
            entity_text = str(ex.get("entity_text", "")).strip()
            context = str(ex.get("context", "")).strip()
            if not entity_text or not context:
                continue

            for _ in range(max(1, int(repeat_fn))):
                rows.append(
                    {
                        "text": _format_input(entity_text, context),
                        "label": 1,
                        "source": "hard_positive_fn",
                    }
                )

    out = pd.DataFrame(rows)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


@dataclass
class ThresholdSearchResult:
    threshold: float
    precision: float
    recall: float
    f1: float


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def choose_threshold(
    y_true: List[int],
    p_pos: List[float],
    target_precision: float,
) -> ThresholdSearchResult:
    best: Optional[ThresholdSearchResult] = None

    # Grid search thresholds; keep it simple and deterministic.
    for i in range(1, 100):
        t = i / 100.0
        tp = fp = fn = 0
        for yt, pp in zip(y_true, p_pos, strict=True):
            yp = 1 if pp >= t else 0
            if yt == 1 and yp == 1:
                tp += 1
            elif yt == 0 and yp == 1:
                fp += 1
            elif yt == 1 and yp == 0:
                fn += 1

        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)

        # Prefer thresholds that hit target precision; within that, maximize recall.
        if precision >= target_precision:
            cand = ThresholdSearchResult(threshold=t, precision=precision, recall=recall, f1=f1)
            if best is None or cand.recall > best.recall or (
                math.isclose(cand.recall, best.recall) and cand.f1 > best.f1
            ):
                best = cand

    if best is not None:
        return best

    # Fallback: maximize F1 if target precision is not achievable.
    for i in range(1, 100):
        t = i / 100.0
        tp = fp = fn = 0
        for yt, pp in zip(y_true, p_pos, strict=True):
            yp = 1 if pp >= t else 0
            if yt == 1 and yp == 1:
                tp += 1
            elif yt == 0 and yp == 1:
                fp += 1
            elif yt == 1 and yp == 0:
                fn += 1
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        cand = ThresholdSearchResult(threshold=t, precision=precision, recall=recall, f1=f1)
        if best is None or cand.f1 > best.f1:
            best = cand

    assert best is not None
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="Train recall-optimized SetFit Gatekeeper")
    ap.add_argument("--base-csv", default="audit/training_dataset.csv")
    ap.add_argument("--fn-analysis-json", default=None)
    ap.add_argument("--max-fn-examples", type=int, default=5000)
    ap.add_argument("--repeat-fn", type=int, default=3)
    ap.add_argument("--output-dir", default="models/gatekeeper_setfit_recall_v1")
    ap.add_argument(
        "--base-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    ap.add_argument(
        "--base-pos-max",
        type=int,
        default=600,
        help="Max positive rows to keep from base CSV (for speed). Use 0/negative to disable cap.",
    )
    ap.add_argument(
        "--base-neg-max",
        type=int,
        default=600,
        help="Max negative rows to keep from base CSV (for speed). Use 0/negative to disable cap.",
    )
    ap.add_argument("--num-iterations", type=int, default=20)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--target-precision",
        type=float,
        default=0.75,
        help="Threshold tuning tries to keep precision >= this, maximizing recall.",
    )
    args = ap.parse_args()

    base_csv = Path(args.base_csv)
    if not base_csv.exists():
        raise FileNotFoundError(f"Base CSV not found: {base_csv}")

    fn_examples: Optional[List[Dict[str, Any]]] = None
    if args.fn_analysis_json:
        fn_path = Path(args.fn_analysis_json)
        if not fn_path.exists():
            raise FileNotFoundError(f"FN analysis JSON not found: {fn_path}")
        fn_examples = load_fn_examples(fn_path)

    base_df = load_base_dataset_csv(base_csv)
    base_pos_max = None if args.base_pos_max is None or args.base_pos_max <= 0 else args.base_pos_max
    base_neg_max = None if args.base_neg_max is None or args.base_neg_max <= 0 else args.base_neg_max
    base_df = downsample_base_df(base_df, base_pos_max=base_pos_max, base_neg_max=base_neg_max, seed=args.seed)

    train_df = build_training_rows(
        base_df=base_df,
        fn_examples=fn_examples,
        max_fn_examples=args.max_fn_examples,
        repeat_fn=args.repeat_fn,
        seed=args.seed,
    )

    pos = int((train_df["label"] == 1).sum())
    neg = int((train_df["label"] == 0).sum())
    print(f"Dataset size: {len(train_df)} (pos={pos}, neg={neg})")
    if fn_examples:
        print(
            "Hard positives: "
            f"max_fn_examples={args.max_fn_examples}, repeat_fn={args.repeat_fn} (source=FN analysis JSON)"
        )

    # Lazy imports to avoid import-time issues in non-ML environments.
    from datasets import Dataset
    from setfit import SetFitModel, SetFitTrainer

    dataset = Dataset.from_dict({"text": train_df["text"].tolist(), "label": train_df["label"].tolist()})
    dataset = dataset.train_test_split(test_size=0.2, seed=args.seed)

    model = SetFitModel.from_pretrained(args.base_model)

    print(
        "Training SetFit... "
        f"(iterations={args.num_iterations}, epochs={args.num_epochs}, batch={args.batch_size})"
    )
    trainer = SetFitTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        metric="f1",
        num_iterations=args.num_iterations,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        column_mapping={"text": "text", "label": "label"},
    )

    trainer.train()

    print("Training finished. Calibrating threshold on held-out split...")

    # Evaluate + threshold tuning
    test_texts = dataset["test"]["text"]
    y_true = [int(x) for x in dataset["test"]["label"]]

    # In SetFit, predict_proba returns 2 columns [p0, p1]
    proba = model.predict_proba(test_texts)
    p_pos = [float(p[1]) for p in proba]

    thr = choose_threshold(y_true=y_true, p_pos=p_pos, target_precision=float(args.target_precision))

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "base_csv": str(base_csv).replace("\\\\", "/"),
        "fn_analysis_json": str(Path(args.fn_analysis_json)).replace("\\\\", "/") if args.fn_analysis_json else None,
        "dataset_size": int(len(train_df)),
        "dataset_label_counts": {
            "pos": int((train_df["label"] == 1).sum()),
            "neg": int((train_df["label"] == 0).sum()),
        },
        "hyperparameters": {
            "base_model": args.base_model,
            "num_iterations": args.num_iterations,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "recommended_threshold": {
            "target_precision": float(args.target_precision),
            "threshold": float(thr.threshold),
            "precision": float(thr.precision),
            "recall": float(thr.recall),
            "f1": float(thr.f1),
        },
    }

    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved model to: {output_dir}")
    print(
        "Recommended threshold: "
        f"t={thr.threshold:.2f} (precision={thr.precision:.3f}, recall={thr.recall:.3f}, f1={thr.f1:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
