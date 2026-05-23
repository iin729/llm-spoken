# -*- coding: utf-8 -*-
"""
Aggregate Table-4 ablation results into one markdown table.

Walks every variant's saved directory under saves/Qwen2.5-3B-Instruct/, extracts:
  * Params  -- by summing trainable tensors in affine_reft_params.pt (mask buffers excluded)
  * L-Acc   -- from any JSON file in eval_results/ (and parent dir), with optional log-file fallback
  * BERT    -- same source list

The exact key names emitted by eval2.py vary; we try a list of common spellings.
If a metric can't be found automatically the cell prints "—" and the script
notes the variant in stderr so you can fill it in by hand.

The LAS (ours) column is read from affine_pipeline_custom/ (the main-experiment
output dir from run_affine_pipeline.sh, dataset=all).

Usage:
  python aggregate_ablation_results.py \
      --saves_root ../saves/Qwen2.5-3B-Instruct \
      --log_file ablation_0514_0130.log \
      --output ablation_table.md
"""

import os
import re
import json
import glob
import argparse
import sys
from typing import Optional, List

import torch


# (tag, display_name, subdir_under_saves_root)
VARIANT_ORDER = [
    ("las_ours",   "LAS (ours)",  "affine_pipeline_custom"),
    ("rnd_lh",     "Rnd L+H",     "ablation_rnd_lh"),
    ("no_layers",  "$-$Layers",   "ablation_no_layers"),
    ("no_heads",   "$-$Heads",    "ablation_no_heads"),
    ("no_scale",   "$-$Scale",    "ablation_no_scale"),
    ("no_bias",    "$-$Bias",     "ablation_no_bias"),
    ("lora",       "LoRA(L+H)",   "ablation_lora"),
    ("ia3",        "IA$^3$(L+H)", "ablation_ia3"),
    ("reft",       "ReFT(L+H)",   "ablation_reft"),
]

# Plausible key names for the two text-quality metrics that eval2.py might use
LACC_KEYS = [
    "L-Acc", "l_acc", "l-acc", "L_acc", "lacc", "LAcc",
    "level_acc", "level_accuracy", "Level Accuracy", "level-acc",
    "acc", "accuracy",
]
BERT_KEYS = [
    "BERT", "bert", "bertscore", "BERTScore", "bertscore_f1",
    "BERT_F1", "bert-f1", "bert_score", "bert-score", "BERT-F1",
]


# =============================================================================
# Params counter (works for both AblationController and original AffineReFTController)
# =============================================================================
def count_trainable_params(state_dict_path: str) -> Optional[int]:
    if not os.path.exists(state_dict_path):
        return None
    try:
        sd = torch.load(state_dict_path, map_location="cpu")
    except Exception as e:
        print(f"[WARN] failed to load {state_dict_path}: {e}", file=sys.stderr)
        return None
    total = 0
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        # exclude mask buffers (both controllers prefix them with "mask")
        if "mask" in k.lower():
            continue
        # exclude floating-point buffers added by parametrizations (e.g. "tau")
        # — but DO count `parametrizations.weight.original` (the trainable A matrix)
        if k.endswith(".tau") or k.endswith("/tau"):
            continue
        total += v.numel()
    return total


# =============================================================================
# Metric scraper: read every JSON in eval_results/ and bubble up matching keys
# =============================================================================
def _walk_json_for_key(obj, key_candidates: List[str]):
    """Depth-first search for any of `key_candidates` (case-insensitive) in a nested dict/list."""
    targets_lc = [k.lower() for k in key_candidates]
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(k, str) and k.lower() in targets_lc:
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str):
                        m = re.search(r"-?\d+(?:\.\d+)?", v)
                        if m:
                            return float(m.group(0))
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def find_metric_in_dir(eval_dir: str, parent_dir: str, keys: List[str]) -> Optional[float]:
    """Scan all JSON files under eval_dir and parent_dir for any of `keys`."""
    candidates = []
    for d in (eval_dir, parent_dir):
        if d and os.path.isdir(d):
            candidates.extend(sorted(glob.glob(os.path.join(d, "*.json"))))
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        v = _walk_json_for_key(data, keys)
        if v is not None:
            return v
    return None


# =============================================================================
# Log-file fallback: locate the section for a given tag, then regex-scrape
# =============================================================================
def parse_metric_from_log(log_path: str, tag: str, keys: List[str]) -> Optional[float]:
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Special-case LAS (main run): look for affine_pipeline_custom section,
    # otherwise look for our ablation tag markers
    if tag == "las_ours":
        start_pat = r"affine_pipeline_custom\b"
    else:
        start_pat = rf"\[ablation\] tag={re.escape(tag)}\b|\[SMOKE\] tag={re.escape(tag)}\b"
    m = re.search(start_pat, content)
    if not m:
        return None
    start = m.start()
    # End of section = next [ablation] tag= marker OR EOF
    rest = content[start + 1:]
    m_end = re.search(r"\[ablation\] tag=", rest)
    end = start + 1 + m_end.start() if m_end else len(content)
    section = content[start:end]

    # Try each key spelling and grab the first plausible number on the same line
    for key in keys:
        pattern = rf"{re.escape(key)}\s*[:=]?\s*(-?\d+(?:\.\d+)?)"
        m = re.search(pattern, section, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


# =============================================================================
# Formatting
# =============================================================================
def format_params(n: Optional[int]) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    return f"{n/1000:.0f}K"


def format_metric(v: Optional[float], decimals: int = 3) -> str:
    if v is None:
        return "—"
    # eval2.py may emit either 0–1 fractions or 0–100 percentages; normalize to fractions
    if v > 1.5:
        v = v / 100.0
    return f"{v:.{decimals}f}"


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saves_root", default="../saves/Qwen2.5-3B-Instruct",
                    help="Path to saves/Qwen2.5-3B-Instruct (or wherever the variant dirs live)")
    ap.add_argument("--log_file", default=None,
                    help="Optional: the tee log file from run_ablation_qwen.sh — used as a fallback "
                         "for L-Acc / BERT if no JSON contains them")
    ap.add_argument("--output", default="ablation_table.md")
    ap.add_argument("--include_extra", action="store_true",
                    help="Also output BLEU-4 and ROUGE-L rows from eval_ablation.py's metrics.json")
    args = ap.parse_args()

    saves_root = os.path.abspath(args.saves_root)
    print(f"Scanning: {saves_root}", file=sys.stderr)
    if args.log_file:
        print(f"Log fallback: {os.path.abspath(args.log_file)}", file=sys.stderr)

    headers = ["Metric"]
    rows = {
        "Params $\\downarrow$": [],
        "L-Acc $\\uparrow$": [],
        "BERT $\\uparrow$": [],
    }
    if args.include_extra:
        rows["BLEU-4 $\\uparrow$"] = []
        rows["ROUGE-L $\\uparrow$"] = []

    missing_report = []

    for tag, display, subdir in VARIANT_ORDER:
        var_dir = os.path.join(saves_root, subdir)
        params_pt = os.path.join(var_dir, "affine_reft_params.pt")
        eval_dir = os.path.join(var_dir, "eval_results")

        params = count_trainable_params(params_pt)

        lacc = find_metric_in_dir(eval_dir, var_dir, LACC_KEYS)
        bert = find_metric_in_dir(eval_dir, var_dir, BERT_KEYS)
        if lacc is None and args.log_file:
            lacc = parse_metric_from_log(args.log_file, tag, LACC_KEYS)
        if bert is None and args.log_file:
            bert = parse_metric_from_log(args.log_file, tag, BERT_KEYS)

        bleu = rouge = None
        if args.include_extra:
            bleu = find_metric_in_dir(eval_dir, var_dir, ["bleu-4", "bleu", "BLEU", "BLEU-4"])
            rouge = find_metric_in_dir(eval_dir, var_dir, ["rouge-l", "rougeL", "ROUGE-L", "rouge_l"])

        headers.append(display)
        rows["Params $\\downarrow$"].append(format_params(params))
        rows["L-Acc $\\uparrow$"].append(format_metric(lacc))
        rows["BERT $\\uparrow$"].append(format_metric(bert))
        if args.include_extra:
            rows["BLEU-4 $\\uparrow$"].append(format_metric(bleu, decimals=2))
            rows["ROUGE-L $\\uparrow$"].append(format_metric(rouge, decimals=2))

        missing = []
        if params is None:
            missing.append("Params")
        if lacc is None:
            missing.append("L-Acc")
        if bert is None:
            missing.append("BERT")
        if missing:
            missing_report.append(f"  - {tag:12s} ({var_dir}): missing {missing}")

        print(
            f"  {tag:12s} | Params={format_params(params):>7s} | "
            f"L-Acc={format_metric(lacc):>6s} | BERT={format_metric(bert):>6s}",
            file=sys.stderr,
        )

    # ---- emit markdown ----
    md = []
    md.append("# LAS Design Ablations (Qwen-2.5-3B-Instruct, WSPairs test)\n")
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    for metric, vals in rows.items():
        md.append(f"| **{metric}** | " + " | ".join(vals) + " |")
    md_text = "\n".join(md) + "\n"

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md_text)
    print()
    print(md_text)
    print(f"Saved -> {os.path.abspath(args.output)}", file=sys.stderr)

    if missing_report:
        print(file=sys.stderr)
        print("Variants with missing values (fill in manually if needed):", file=sys.stderr)
        for line in missing_report:
            print(line, file=sys.stderr)
        print(
            "\nIf L-Acc/BERT are missing everywhere, run aggregate_ablation_results.py with "
            "--log_file pointing at the tee log so the script can scrape eval2.py's stdout, "
            "or open eval_results/ and tell me what filename eval2.py wrote so I can add it "
            "to the scanner.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
