#!/usr/bin/env bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# =============================================================================
# run_full_ablation.sh
# =============================================================================
# Runs the full CAS component ablation on Qwen-2.5-3B-Instruct.
# Six variants, one row of the ablation table each:
#
#   full_cas       : L_oral + alpha*(t)            (= our CAS)
#   no_calib       : L_oral + best fixed alpha     (- calibration)
#   no_multilayer  : single best layer + alpha*(t) (- multi-layer)
#   actadd         : single best layer + fixed     (= Turner et al. 2023)
#   random_layers  : random non-oral layers + alpha*(t)
#   random_vec     : L_oral + random Gaussian vectors + alpha*(t)
#
# Each variant produces:
#   results/abl_<variant>/experiment_results.xlsx
#   results/abl_<variant>/coeff_function_params.json
#   results/abl_<variant>/final_predictions.jsonl
# Downstream: feed final_predictions.jsonl into eval2.py to fill the table row.
# =============================================================================
set -e

# ---- environment ----
NUM_GPUS=4

# Absolute paths so the script works regardless of cwd
DATA_DIR="${REPO_ROOT}/data"
VEC_DIR="${REPO_ROOT}/cas/results"
TEST_DATA="${DATA_DIR}/test_level_sft.json"
CALIB_DATA="${DATA_DIR}/ablation_level_sft.json"

# ---- backbone ----
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
VECTOR_PATH="${VEC_DIR}/Qwen2.5-3B-Instruct_L5_minus_L0/style_vectors_l5_minus_l0.pt"
MODEL_NAME=$(basename "$MODEL_PATH")

# Localization config for Qwen-2.5-3B-Instruct
ORAL_LAYERS="12-28"          # L_oral (from causal patching elbow, used by CAS production)
BEST_SINGLE_LAYER="20"        # Single best layer from the Task-2 layer sweep
ALL_MODEL_LAYERS="0-35"       # 36-layer backbone

VARIANTS=("full_cas" "no_calib" "no_multilayer" "actadd" "random_layers" "random_vec")

# =============================================================================

run_variant () {
    local VARIANT="$1"
    local OUT_DIR="./results/abl_${VARIANT}_${MODEL_NAME}"
    local PRED_FILE="${OUT_DIR}/final_predictions.jsonl"
    mkdir -p "$OUT_DIR"

    echo "################################################################"
    echo "## variant = $VARIANT  |  model = $MODEL_NAME"
    echo "## output  = $OUT_DIR"
    echo "################################################################"

    # ------------------------------------------------------------------
    # Step 1/3: calibration sweep
    # ------------------------------------------------------------------
    echo ">>> [1/3] Calibration sweep ..."
    CUDA_VISIBLE_DEVICES=0 python -u inject_ablation.py \
        --variant "$VARIANT" \
        --mode calibrate \
        --model_path "$MODEL_PATH" \
        --vector_path "$VECTOR_PATH" \
        --data_path "$CALIB_DATA" \
        --output_dir "$OUT_DIR" \
        --oral_layers "$ORAL_LAYERS" \
        --best_single_layer "$BEST_SINGLE_LAYER" \
        --all_model_layers "$ALL_MODEL_LAYERS"

    # ------------------------------------------------------------------
    # Step 2/3: analyze (poly2 fit OR best fixed alpha)
    # ------------------------------------------------------------------
    echo ">>> [2/3] Analyze ..."
    python -u inject_ablation.py \
        --variant "$VARIANT" \
        --mode analyze \
        --model_path "$MODEL_PATH" \
        --vector_path "$VECTOR_PATH" \
        --data_path "$CALIB_DATA" \
        --output_dir "$OUT_DIR" \
        --oral_layers "$ORAL_LAYERS" \
        --best_single_layer "$BEST_SINGLE_LAYER" \
        --all_model_layers "$ALL_MODEL_LAYERS" \
        --fixed_alpha_target 3.0

    # ------------------------------------------------------------------
    # Step 3/3: chunked prediction on test set
    # ------------------------------------------------------------------
    echo ">>> [3/3] Prediction on test set ($NUM_GPUS GPUs) ..."
    rm -f "${PRED_FILE}.part"*
    for ((i=0; i<NUM_GPUS; i++)); do
        CUDA_VISIBLE_DEVICES=$i python -u inject_ablation.py \
            --variant "$VARIANT" \
            --mode predict \
            --model_path "$MODEL_PATH" \
            --vector_path "$VECTOR_PATH" \
            --data_path "$TEST_DATA" \
            --output_dir "$OUT_DIR" \
            --prediction_file "$PRED_FILE" \
            --oral_layers "$ORAL_LAYERS" \
            --best_single_layer "$BEST_SINGLE_LAYER" \
            --all_model_layers "$ALL_MODEL_LAYERS" \
            --num_chunks "$NUM_GPUS" \
            --chunk_idx "$i" \
            > "${OUT_DIR}/pred_gpu${i}.log" 2>&1 &
    done
    wait
    cat "${PRED_FILE}.part"* > "$PRED_FILE"
    rm "${PRED_FILE}.part"*

    echo ">>> $VARIANT predictions: $PRED_FILE"
    echo ""
}

for V in "${VARIANTS[@]}"; do
    run_variant "$V"
done

echo "================================================================"
echo "Full ablation complete. Six prediction files ready:"
for V in "${VARIANTS[@]}"; do
    echo "  ./results/abl_${V}_${MODEL_NAME}/final_predictions.jsonl"
done
echo ""
echo "Next: run eval2.py / eval_predictions.py on each to fill ablation table."
echo "================================================================"
