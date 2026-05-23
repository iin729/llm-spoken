"""
summarize_results.py
====================

Walk results/{actadd_*,abl_*}/eval_all_metrics.json and produce:

  (A) results/main_table_actadd_rows.txt   <- LaTeX rows for Table 2 (CAS)
  (B) results/ablation_table_rows.txt      <- LaTeX rows for the new component
                                              ablation table on Qwen-3B
  (C) results/all_metrics.csv              <- one row per variant, all metrics

The LaTeX rows are formatted to match the existing acl_latex_v9_36.tex layout
so you can paste them in directly without manual reformatting.
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, Any, List, Optional


# ----- key extraction helpers --------------------------------------------
def get_m(metrics: Dict[str, Any], key: str, default=None):
    v = metrics.get(key, default)
    if isinstance(v, (int, float)):
        return v
    return default


def fmt_num(v: Optional[float], digits: int = 3, default: str = "--") -> str:
    if v is None:
        return default
    return f"{v:.{digits}f}"


def load_metrics_for(variant_dir: str) -> Dict[str, Any]:
    """Return the dict of summary metrics from <dir>/eval_all_metrics.json."""
    path = os.path.join(variant_dir, "eval_all_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        summary = json.load(f)
    return summary.get("metrics", {})


def scan_predictions_for_alpha(variant_dir: str) -> Optional[float]:
    """Mean applied alpha across the prediction file, if available."""
    path = os.path.join(variant_dir, "final_predictions.jsonl")
    if not os.path.exists(path):
        return None
    s, n = 0.0, 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if "applied_coeff" in r:
                    s += float(r["applied_coeff"])
                    n += 1
            except Exception:
                continue
    return (s / n) if n else None


def n_samples(variant_dir: str) -> int:
    path = os.path.join(variant_dir, "final_predictions.jsonl")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


# ----- main pipelines -----------------------------------------------------
def collect_actadd(results_root: str) -> List[Dict[str, Any]]:
    rows = []
    for d in sorted(glob.glob(os.path.join(results_root, "actadd_*"))):
        if not os.path.isdir(d):
            continue
        backbone = os.path.basename(d)[len("actadd_"):]
        m = load_metrics_for(d)
        alpha = scan_predictions_for_alpha(d)
        rows.append({
            "variant": "actadd",
            "backbone": backbone,
            "n": n_samples(d),
            "alpha": alpha,
            "BERT_F1": get_m(m, "BERTScore_F1"),
            "L_Acc":   get_m(m, "level_accuracy"),
            "L_Bias":  get_m(m, "level_Bias"),
            "L_MAE":   get_m(m, "level_MAE"),
            "O_Bias":  get_m(m, "O_score_Bias"),
        })
    return rows


def collect_cas_capped(results_root: str) -> List[Dict[str, Any]]:
    """Rows from results/cas_capped_<backbone>/ -- CAS with --alpha_cap set."""
    rows = []
    for d in sorted(glob.glob(os.path.join(results_root, "cas_capped_*"))):
        if not os.path.isdir(d):
            continue
        backbone = os.path.basename(d)[len("cas_capped_"):]
        m = load_metrics_for(d)
        alpha = scan_predictions_for_alpha(d)
        rows.append({
            "variant": "cas_capped",
            "backbone": backbone,
            "n": n_samples(d),
            "alpha": alpha,
            "BERT_F1": get_m(m, "BERTScore_F1"),
            "L_Acc":   get_m(m, "level_accuracy"),
            "L_Bias":  get_m(m, "level_Bias"),
            "L_MAE":   get_m(m, "level_MAE"),
            "O_Bias":  get_m(m, "O_score_Bias"),
        })
    return rows


def collect_ablation(results_root: str, model_name: str = "Qwen2.5-3B-Instruct") -> List[Dict[str, Any]]:
    order = ["full_cas", "no_calib", "no_multilayer",
             "actadd", "random_layers", "random_vec"]
    rows = []
    for v in order:
        d = os.path.join(results_root, f"abl_{v}_{model_name}")
        m = load_metrics_for(d)
        alpha = scan_predictions_for_alpha(d)
        rows.append({
            "variant": v,
            "backbone": model_name,
            "n": n_samples(d),
            "alpha": alpha,
            "BERT_F1": get_m(m, "BERTScore_F1"),
            "L_Acc":   get_m(m, "level_accuracy"),
            "L_Bias":  get_m(m, "level_Bias"),
            "L_MAE":   get_m(m, "level_MAE"),
            "O_Bias":  get_m(m, "O_score_Bias"),
        })
    return rows


# ----- LaTeX row formatting ----------------------------------------------
PRETTY_BACKBONE = {
    "Qwen2.5-3B-Instruct":     "Qwen-3B",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen2.5-7B-Instruct":     "Qwen-7B",
    "Llama-3.2-1B-Instruct":   "Llama-1B",
}


def fmt_lacc_bias(L_Acc: Optional[float], L_Bias: Optional[float]) -> str:
    if L_Acc is None:
        return "--"
    if L_Bias is None:
        return fmt_num(L_Acc, 3)
    return f"{L_Acc:.3f}$_{{{L_Bias:+.2f}}}$"


def render_main_table_rows(actadd_rows: List[Dict[str, Any]],
                           capped_rows: List[Dict[str, Any]] = None) -> str:
    """Emit ActAdd (and optionally alpha-capped CAS) rows for Table 2."""
    lines = []
    lines.append("% ----- ActAdd rows for Table 2 (CAS) -----")
    lines.append("% Paste under the open-source block, right above each backbone's CAS row")
    for r in actadd_rows:
        bb = PRETTY_BACKBONE.get(r["backbone"], r["backbone"])
        bert = fmt_num(r["BERT_F1"], 3)
        lacc = fmt_lacc_bias(r["L_Acc"], r["L_Bias"])
        lines.append(
            f"% {bb}: n={r['n']}, mean alpha={fmt_num(r['alpha'], 3)}"
        )
        lines.append(
            f" & ActAdd & -- & {bert} & {lacc} \\\\"
        )

    if capped_rows:
        lines.append("")
        lines.append("% ----- alpha-capped CAS rows (replaces uncapped CAS rows where applicable) -----")
        for r in capped_rows:
            bb = PRETTY_BACKBONE.get(r["backbone"], r["backbone"])
            bert = fmt_num(r["BERT_F1"], 3)
            lacc = fmt_lacc_bias(r["L_Acc"], r["L_Bias"])
            lines.append(
                f"% {bb}: n={r['n']}, mean applied alpha={fmt_num(r['alpha'], 3)} "
                f"(this is CAS with --alpha_cap enforced)"
            )
            lines.append(
                f" & \\textbf{{CAS}}$_{{\\text{{ours}}}}$ & -- & "
                f"{bert} & \\textbf{{{lacc}}} \\\\"
            )
    return "\n".join(lines) + "\n"


def render_ablation_table_rows(ablation_rows: List[Dict[str, Any]]) -> str:
    """Emit a clean 6-row ablation table block."""
    header = (
        "% ----- Ablation rows: CAS component decomposition on Qwen-2.5-3B -----\n"
        "% Columns: Variant | mean alpha | L-Acc_bias | BERT-F1\n"
        "\\begin{tabular}{l c c c}\n"
        "\\toprule\n"
        "\\textbf{Variant} & $\\bar{\\alpha}$ & \\textbf{L-Acc$_{\\text{bias}}$} & \\textbf{BERT} \\\\\n"
        "\\midrule\n"
    )
    pretty_variant = {
        "full_cas":      "\\textbf{CAS (ours)}",
        "no_calib":      "$-$ Calibration",
        "no_multilayer": "$-$ Multi-layer",
        "actadd":        "$-$ Both (= ActAdd)",
        "random_layers": "$-$ Localization (rnd L)",
        "random_vec":    "$-$ Learned vec (rnd v)",
    }
    body = []
    for r in ablation_rows:
        v = pretty_variant.get(r["variant"], r["variant"])
        alpha_str = fmt_num(r["alpha"], 2)
        lacc = fmt_lacc_bias(r["L_Acc"], r["L_Bias"])
        bert = fmt_num(r["BERT_F1"], 3)
        body.append(f"{v} & {alpha_str} & {lacc} & {bert} \\\\")
    return header + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"


# ----- CSV dump ------------------------------------------------------------
def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w") as f:
            f.write("(no rows)\n")
        return
    cols = ["variant", "backbone", "n", "alpha", "BERT_F1",
            "L_Acc", "L_Bias", "L_MAE", "O_Bias"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


# ----- entry point --------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default="./results")
    p.add_argument("--ablation_model", default="Qwen2.5-3B-Instruct")
    p.add_argument("--out_dir", default="./results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    actadd_rows = collect_actadd(args.results_root)
    capped_rows = collect_cas_capped(args.results_root)
    abl_rows = collect_ablation(args.results_root, args.ablation_model)

    csv_path = os.path.join(args.out_dir, "all_metrics.csv")
    write_csv(csv_path, actadd_rows + capped_rows + abl_rows)

    main_tex = render_main_table_rows(actadd_rows, capped_rows)
    abl_tex = render_ablation_table_rows(abl_rows)

    main_path = os.path.join(args.out_dir, "main_table_actadd_rows.txt")
    abl_path = os.path.join(args.out_dir, "ablation_table_rows.txt")
    with open(main_path, "w") as f:
        f.write(main_tex)
    with open(abl_path, "w") as f:
        f.write(abl_tex)

    print(f"Wrote {csv_path}")
    print(f"Wrote {main_path}")
    print(f"Wrote {abl_path}")
    print()
    print("=" * 70)
    print("ACTADD ROWS (paste into Table 2 of the paper):")
    print("=" * 70)
    print(main_tex)
    print()
    print("=" * 70)
    print("ABLATION TABLE (use as Table N for the new component decomposition):")
    print("=" * 70)
    print(abl_tex)


if __name__ == "__main__":
    main()
