#!/usr/bin/env bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# =============================================================================
# score_iso_ablation.sh
# =============================================================================
# Minimal scorer for the Iso ablation row of paper Table 3.
#
# What it does
# ------------
#   Runs `score_minimal.py` on ./results/abl_iso_cas_<MODEL_NAME>/final_predictions.jsonl,
#   producing eval_all_metrics.json with only the three numbers the paper's
#   ablation table reports:
#       BERTScore_F1
#       level_accuracy
#       level_Bias
#
# Why not ΔR_r?
# -------------
#   Per the latest paper plan, ΔR_r (reward-model evaluator) is reported only
#   for the two MAIN tables (Tables 1 & 2), not the ablation table.  So we
#   don't load Mistral-7B here — keeps the scoring fast (~2 min total) and
#   doesn't need access to the evaluator_validation pipeline at all.
#
# Why not the other ablation variants?
# ------------------------------------
#   They were already scored.  This script only fills in the new Iso row.
#   If you ever want to re-score everything, use `bash score_all.sh` (which
#   walks every results/abl_*/ dir).
#
# Layout assumed
# --------------
#   cas_paper_ablation/
#     score_iso_ablation.sh     (this file)
#     score_minimal.py          (computes BERT + level metrics)
#     results/
#       abl_iso_cas_<MODEL>/    (the dir run_iso_ablation.sh produced)
#         final_predictions.jsonl
#
# Launch
# ------
#   cd ${REPO_ROOT}/cas/ablation
#   bash score_iso_ablation.sh
#
# Override (other backbones / non-default variant tag):
#   MODEL_NAME=Mistral-7B-Instruct-v0.3 bash score_iso_ablation.sh
#   VARIANT_TAG=iso_cas              bash score_iso_ablation.sh
#   PRED_DIR=/abs/path/to/some/dir   bash score_iso_ablation.sh
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-3B-Instruct}"
VARIANT_TAG="${VARIANT_TAG:-iso_cas}"
PRED_DIR="${PRED_DIR:-$HERE/results/abl_${VARIANT_TAG}_${MODEL_NAME}}"
PRED_FILE="$PRED_DIR/final_predictions.jsonl"

# Sanity
if [ ! -f "$HERE/score_minimal.py" ]; then
    echo "[fatal] score_minimal.py not found alongside this sh: $HERE/"
    exit 2
fi
if [ ! -f "$PRED_FILE" ]; then
    echo "[fatal] no predictions at: $PRED_FILE"
    echo "        Did run_iso_ablation.sh finish?  Or override PRED_DIR=..."
    exit 2
fi

ROWS=$(wc -l < "$PRED_FILE" | awk '{print $1}')
echo "================================================================"
echo " Scoring variant : $VARIANT_TAG"
echo " Backbone        : $MODEL_NAME"
echo " Input           : $PRED_FILE  ($ROWS rows)"
echo " Output          : $PRED_DIR/eval_all_metrics.json"
echo "================================================================"

# Run the minimal scorer (only level + BERTScore; no PPL, no BLEU, no RM)
python -u "$HERE/score_minimal.py" "$PRED_FILE"

# Pull just the three numbers we need and render the paper-ready row.
echo
echo "================================================================"
echo " Paper-ready row (drop into the Iso line of Table 3)"
echo "================================================================"
python - "$PRED_DIR/eval_all_metrics.json" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
m = json.loads(p.read_text()).get("metrics", {})
bert  = m.get("BERTScore_F1")
lacc  = m.get("level_accuracy")
lbias = m.get("level_Bias")

def _f(x, w=3):
    return "—" if x is None else f"{x:.{w}f}"

print(f"  BERTScore_F1     = {_f(bert)}")
print(f"  level_accuracy   = {_f(lacc)}")
print(f"  level_Bias       = {('—' if lbias is None else f'{lbias:+.3f}')}")
print()
print("  Markdown form  :  ", end="")
if bert is not None and lacc is not None and lbias is not None:
    print(f"| Iso (isotonic α(t)) | {bert:.3f} | {lacc:.3f}$_{{{lbias:+.2f}}}$ |")
else:
    print("(missing fields — inspect eval_all_metrics.json)")

print()
print("  LaTeX form     :  ", end="")
if bert is not None and lacc is not None and lbias is not None:
    print(f"Iso & {bert:.3f} & {lacc:.3f}$_{{{lbias:+.2f}}}$ \\\\")
else:
    print("(missing fields)")
PY

echo
echo "================================================================"
echo " Done.  Full JSON: $PRED_DIR/eval_all_metrics.json"
echo "================================================================"
