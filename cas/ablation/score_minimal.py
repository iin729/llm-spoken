"""
score_minimal.py
================

Compute the three metrics the paper actually needs for the CAS ablation table:
  * level_accuracy  (predicted Orality Level matching the target level)
  * level_Bias      (signed mean error of predicted level vs target level)
  * BERTScore_F1    (semantic preservation against the gold spoken label)

Plus a few sanity diagnostics:
  * n_samples
  * mean_target_level / mean_pred_level
  * mean_applied_alpha  (if the prediction JSONL was produced by inject_ablation.py)

Designed to need ONLY:
  - eval_oscore_level  (already verified importable on this host)
  - stat_metric8       (idem)
  - bert_score         (already in pip list)

Usage:
    python score_minimal.py path/to/final_predictions.jsonl
    # writes path/to/eval_all_metrics.json with a `metrics` dict

The output schema matches what summarize_results.py expects, so the existing
LaTeX-row pipeline keeps working.
"""

import argparse
import json
import os
import sys
from typing import Optional, List

# Add repo root to sys.path so `from orality import ...` works regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from orality import eval_oscore_and_level, stat_all_metrics


def text_to_level(text: str) -> Optional[int]:
    if not isinstance(text, str) or not text.strip():
        return None
    stats = stat_all_metrics(text)
    if not stats:
        return None
    _, level = eval_oscore_and_level(stats)
    return int(level)


def load_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def score(path: str, bertscore_batch: int = 64) -> dict:
    rows = list(load_jsonl(path))
    preds, refs, targets, alphas = [], [], [], []
    for r in rows:
        p = r.get("predict")
        l = r.get("label")
        if isinstance(p, str) and isinstance(l, str):
            preds.append(p)
            refs.append(l)
            targets.append(r.get("target_level"))
            if "applied_coeff" in r:
                alphas.append(float(r["applied_coeff"]))

    n = len(preds)
    if n == 0:
        return {"total_samples": 0, "metrics": {}}

    # Predicted level for every output
    print(f"[level] scoring {n} predictions ...")
    pred_levels = []
    for p in preds:
        lvl = text_to_level(p)
        pred_levels.append(lvl)

    # Pair-wise comparison against the requested target_level (from inject_ablation
    # output) if present, else fall back to the gold-label-derived level.
    paired_diffs = []
    n_correct = 0
    n_paired = 0
    for i, plvl in enumerate(pred_levels):
        if plvl is None:
            continue
        tgt = targets[i]
        if tgt is None:
            # derive from the gold label
            tgt = text_to_level(refs[i])
            if tgt is None:
                continue
        n_paired += 1
        if plvl == int(tgt):
            n_correct += 1
        paired_diffs.append(plvl - int(tgt))

    level_accuracy = (n_correct / n_paired) if n_paired else 0.0
    level_bias = (sum(paired_diffs) / len(paired_diffs)) if paired_diffs else 0.0
    level_mae = (sum(abs(d) for d in paired_diffs) / len(paired_diffs)) if paired_diffs else 0.0

    # BERTScore F1
    print(f"[bertscore] computing F1 for {n} pairs ...")
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(preds, refs, lang="en", batch_size=bertscore_batch, verbose=False)
    bert_f1 = float(F1.mean())

    summary = {
        "total_samples": n,
        "metrics": {
            "n_paired": n_paired,
            "level_accuracy": level_accuracy,
            "level_Bias":     level_bias,
            "level_MAE":      level_mae,
            "BERTScore_F1":   bert_f1,
            "mean_target_level": (sum(int(t) for t in targets if t is not None) /
                                   max(1, sum(1 for t in targets if t is not None))),
            "mean_pred_level":   (sum(p for p in pred_levels if p is not None) /
                                   max(1, sum(1 for p in pred_levels if p is not None))),
            "mean_applied_alpha": (sum(alphas) / len(alphas)) if alphas else None,
        },
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_path", help="path to final_predictions.jsonl")
    p.add_argument("--bert_batch", type=int, default=64)
    args = p.parse_args()

    summary = score(args.input_path, bertscore_batch=args.bert_batch)
    out_dir = os.path.dirname(os.path.abspath(args.input_path))
    out_file = os.path.join(out_dir, "eval_all_metrics.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {out_file}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
