"""
aggregate_results.py
====================

Walk the per-variant prediction folders produced by run_full_ablation.sh and
roll them up into a single CSV that maps directly to one row of the ablation
table in the paper.

Output columns:
    variant, n, mean_target_level, mean_applied_alpha, L-Acc, L-Acc_bias, BERTScore_F1

Usage:
    python aggregate_results.py \
        --results_root ./results \
        --model_name Qwen2.5-3B-Instruct \
        --out ./results/ablation_summary.csv

Assumes each `abl_<variant>_<model>/final_predictions.jsonl` has already been
scored with eval2.py / eval_predictions.py and that scoring wrote a sibling
file `final_predictions_metrics.json` containing the L-Acc / L-Acc_bias /
BERTScore-F1 summary. If you use a different scorer, edit `_load_metrics`.
"""
import argparse
import json
import os
import csv


METRICS_FILENAME_CANDIDATES = [
    "final_predictions_metrics.json",
    "eval_all_metrics.json",
    "metrics_summary.json",
]


def _load_metrics(variant_dir):
    """Return dict of summary metrics or None if no metrics file is found."""
    for name in METRICS_FILENAME_CANDIDATES:
        path = os.path.join(variant_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def _scan_predictions(jsonl_path):
    """Return n, mean target level, mean applied alpha from the predictions file."""
    if not os.path.exists(jsonl_path):
        return 0, None, None
    n = 0
    sum_t = 0.0
    sum_a = 0.0
    have_alpha = 0
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            sum_t += float(r.get("target_level", 0))
            if "applied_coeff" in r:
                sum_a += float(r["applied_coeff"])
                have_alpha += 1
    mt = sum_t / n if n else None
    ma = sum_a / have_alpha if have_alpha else None
    return n, mt, ma


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default="./results")
    p.add_argument("--model_name", default="Qwen2.5-3B-Instruct")
    p.add_argument("--variants", nargs="+",
                   default=["full_cas", "no_calib", "no_multilayer",
                            "actadd", "random_layers", "random_vec"])
    p.add_argument("--out", default="./results/ablation_summary.csv")
    args = p.parse_args()

    rows = []
    for v in args.variants:
        vdir = os.path.join(args.results_root, f"abl_{v}_{args.model_name}")
        pred_file = os.path.join(vdir, "final_predictions.jsonl")
        n, mean_t, mean_a = _scan_predictions(pred_file)
        m = _load_metrics(vdir) or {}
        rows.append({
            "variant": v,
            "n": n,
            "mean_target_level": f"{mean_t:.3f}" if mean_t is not None else "-",
            "mean_applied_alpha": f"{mean_a:.3f}" if mean_a is not None else "-",
            "L-Acc": m.get("L_Acc", m.get("L-Acc", "-")),
            "L-Acc_bias": m.get("L_Acc_bias", m.get("L-Acc_bias", "-")),
            "BERTScore_F1": m.get("BERTScore_F1", m.get("bertscore_f1", "-")),
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.out} with {len(rows)} rows.")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
