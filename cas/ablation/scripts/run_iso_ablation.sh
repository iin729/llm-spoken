#!/usr/bin/env bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# =============================================================================
# run_iso_ablation.sh
# =============================================================================
# Single-variant ablation for paper Table 3:
#
#     Iso  =  L_oral (multi-layer) + real vectors + ISOTONIC regression
#             on the same (t, α*_t) anchor points used by full_cas, in
#             place of the quadratic α(t) fit (Barlow et al., 1972).
#
# Everything else is identical to full_cas, so the difference vs. the
# `full_cas` row of Table 3 isolates the *fit family* (parametric quadratic
# vs. non-parametric monotone) alone.
#
# Defaults: Qwen-2.5-3B-Instruct, matching scripts/cas_paper_ablation/
# run_full_ablation.sh.  To run on Mistral-7B or Qwen-7B, override env vars
# at the call site, e.g.:
#
#   MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3 \
#   VECTOR_PATH=${REPO_ROOT}/cas/results/Mistral-7B-Instruct-v0.3_L5_minus_L0/style_vectors_l5_minus_l0.pt \
#   ORAL_LAYERS=15-25  ALPHA_CAP=1.5 \
#       bash run_iso_ablation.sh
#
# Output:
#   ./results/abl_iso_cas_<MODEL_NAME>/experiment_results.xlsx
#   ./results/abl_iso_cas_<MODEL_NAME>/coeff_function_params.json     (type="isotonic")
#   ./results/abl_iso_cas_<MODEL_NAME>/final_predictions.jsonl
# =============================================================================
set -e

# ---- environment ----
NUM_GPUS="${NUM_GPUS:-2}"

# Absolute paths so the script works regardless of cwd.
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
VEC_DIR="${VEC_DIR:-${REPO_ROOT}/cas/results}"
TEST_DATA="${TEST_DATA:-${DATA_DIR}/test_level_sft.json}"
CALIB_DATA="${CALIB_DATA:-${DATA_DIR}/ablation_level_sft.json}"

# ---- backbone (override via env to run on other models) ----
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
VECTOR_PATH="${VECTOR_PATH:-${VEC_DIR}/Qwen2.5-3B-Instruct_L5_minus_L0/style_vectors_l5_minus_l0.pt}"
ORAL_LAYERS="${ORAL_LAYERS:-12-28}"
ALPHA_CAP="${ALPHA_CAP:-4.0}"

MODEL_NAME=$(basename "$MODEL_PATH")
VARIANT_TAG="iso_cas"
OUT_DIR="./results/abl_${VARIANT_TAG}_${MODEL_NAME}"
PRED_FILE="${OUT_DIR}/final_predictions.jsonl"
mkdir -p "$OUT_DIR"

echo "################################################################"
echo "## variant     = $VARIANT_TAG"
echo "## model       = $MODEL_NAME"
echo "## model_path  = $MODEL_PATH"
echo "## vector_path = $VECTOR_PATH"
echo "## oral_layers = $ORAL_LAYERS"
echo "## alpha_cap   = $ALPHA_CAP"
echo "## output_dir  = $OUT_DIR"
echo "## num_gpus    = $NUM_GPUS"
echo "################################################################"

# Make sure the .py we depend on (and its sibling inject_ablation.py for the
# shared helpers) are next to this sh.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in inject_ablation_iso.py inject_ablation.py; do
    if [ ! -f "$HERE/$f" ]; then
        echo "[fatal] $f not found alongside this sh: $HERE/"
        exit 2
    fi
done
cd "$HERE"

# ------------------------------------------------------------------
# Step 1/3: calibration sweep (same alpha grid as full_cas)
# ------------------------------------------------------------------
echo ">>> [1/3] Calibration sweep ..."
CUDA_VISIBLE_DEVICES=0 python -u inject_ablation_iso.py \
    --mode calibrate \
    --model_path "$MODEL_PATH" \
    --vector_path "$VECTOR_PATH" \
    --data_path "$CALIB_DATA" \
    --output_dir "$OUT_DIR" \
    --oral_layers "$ORAL_LAYERS"

# ------------------------------------------------------------------
# Step 2/3: analyze -> isotonic regression on (t, α*_t)
# ------------------------------------------------------------------
echo ">>> [2/3] Analyze (isotonic regression, PAVA / sklearn) ..."
python -u inject_ablation_iso.py \
    --mode analyze \
    --model_path "$MODEL_PATH" \
    --vector_path "$VECTOR_PATH" \
    --data_path "$CALIB_DATA" \
    --output_dir "$OUT_DIR" \
    --oral_layers "$ORAL_LAYERS"

# ------------------------------------------------------------------
# Step 3/3: chunked prediction on test set with iso lookup
# ------------------------------------------------------------------
echo ">>> [3/3] Prediction on test set ($NUM_GPUS GPUs) ..."
rm -f "${PRED_FILE}.part"*
for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python -u inject_ablation_iso.py \
        --mode predict \
        --model_path "$MODEL_PATH" \
        --vector_path "$VECTOR_PATH" \
        --data_path "$TEST_DATA" \
        --output_dir "$OUT_DIR" \
        --prediction_file "$PRED_FILE" \
        --oral_layers "$ORAL_LAYERS" \
        --alpha_cap "$ALPHA_CAP" \
        --num_chunks "$NUM_GPUS" \
        --chunk_idx "$i" \
        > "${OUT_DIR}/pred_gpu${i}.log" 2>&1 &
done
wait
cat "${PRED_FILE}.part"* > "$PRED_FILE"
rm "${PRED_FILE}.part"*

echo ""
echo "================================================================"
echo "Iso ablation complete."
echo "  predictions  : $PRED_FILE  ($(wc -l < "$PRED_FILE") rows)"
echo "  fit params   : $OUT_DIR/coeff_function_params.json"
echo "  calib sweep  : $OUT_DIR/experiment_results.xlsx"
echo ""
echo "Next: drop final_predictions.jsonl into your eval pipeline"
echo "  (e.g. python ../eval/eval_predictions.py $PRED_FILE)"
echo "and score ΔR_r with score_with_reward_model.py if desired."
echo "================================================================"
