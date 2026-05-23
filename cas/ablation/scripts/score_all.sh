#!/usr/bin/env bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# =============================================================================
# score_all.sh
# =============================================================================
# Walks every results/actadd_<bb>/final_predictions.jsonl and
# results/abl_<variant>_<bb>/final_predictions.jsonl and runs an evaluator
# on each, producing <dir>/eval_all_metrics.json.
#
# Two evaluator options, in order of preference:
#   1. Full evaluator (eval_predictions.py)   -- BLEU / ROUGE / BERTScore / PPL / level metrics
#   2. Minimal evaluator (score_minimal.py)   -- only the metrics the paper actually needs:
#                                                level_accuracy, level_Bias, BERTScore_F1
# Pick #1 if you have the full one available; #2 is a clean fallback that
# uses only modules already verified on this host.
#
# Override via env var, e.g.:
#   EVAL_SCRIPT=/abs/path/to/eval_predictions.py bash score_all.sh
#   EVAL_SCRIPT=./score_minimal.py             bash score_all.sh
# =============================================================================
set -e

cd "$(dirname "$0")"

# ---- pick an evaluator ----
if [ -z "$EVAL_SCRIPT" ]; then
    # Auto-discover, in priority order
    CANDIDATES=(
        "./eval/eval2.py"                                                  # local new layout
        "./eval/eval_predictions.py"                                       # local new layout
        "${REPO_ROOT}/orality/eval_predictions.py"       # known absolute path
        "${REPO_ROOT}/orality/eval_predictions.py"
        "${REPO_ROOT}/orality/eval_predictions.py"
        "../eval_predictions.py"
        "./score_minimal.py"                                               # zero-dep fallback
    )
    for C in "${CANDIDATES[@]}"; do
        if [ -f "$C" ]; then
            EVAL_SCRIPT="$C"
            break
        fi
    done
fi

if [ -z "$EVAL_SCRIPT" ] || [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: no evaluator found. Set EVAL_SCRIPT explicitly, e.g."
    echo "  EVAL_SCRIPT=./score_minimal.py bash score_all.sh"
    exit 1
fi

echo "Using evaluator: $EVAL_SCRIPT"
echo ""

# ---- gather prediction files ----
RESULTS_DIR="./results"
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: no $RESULTS_DIR directory. Run the experiments first."
    exit 1
fi

PREDS=$(find "$RESULTS_DIR" -name "final_predictions.jsonl" \
        \( -path "*/actadd_*/*" -o -path "*/abl_*/*" -o -path "*/cas_capped_*/*" \) | sort)

if [ -z "$PREDS" ]; then
    echo "ERROR: no final_predictions.jsonl files found under $RESULTS_DIR."
    exit 1
fi

echo "Found $(echo "$PREDS" | wc -l) prediction files to score:"
echo "$PREDS" | sed 's|^|  |'
echo ""

# ---- score each ----
for PRED in $PREDS; do
    DIR=$(dirname "$PRED")
    NAME=$(basename "$DIR")
    OUT="$DIR/eval_all_metrics.json"
    if [ -f "$OUT" ]; then
        echo "[skip] $NAME : already scored ($OUT)"
        continue
    fi
    echo "================================================================"
    echo "[score] $NAME"
    echo "================================================================"
    python -u "$EVAL_SCRIPT" "$PRED"
    echo ""
done

echo "================================================================"
echo "All evaluations complete. Summary files:"
find "$RESULTS_DIR" \( -path "*/actadd_*" -o -path "*/abl_*" \) -name "eval_all_metrics.json" | sort
echo ""
echo "Next: python summarize_results.py"
echo "================================================================"
