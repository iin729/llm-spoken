#!/usr/bin/env bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# =============================================================================
# run_actadd_baseline.sh
# =============================================================================
# Runs the ActAdd baseline (Turner et al., 2023) across the three open-source
# backbones in our main CAS results table (Table 2 of the paper).
#
# ActAdd recipe in this script:
#   - Single grid-searched best layer (per backbone, set below).
#   - Fixed scalar coefficient (selected by inject_ablation.py's analyze step,
#     which picks the alpha whose mean predicted level is closest to the
#     midpoint target level = 3 on the calibration set).
#
# Prerequisite: style vectors must have been extracted per backbone, into the
# `results/<backbone>_L5_minus_L0/style_vectors_l5_minus_l0.pt` paths used
# below. (Produced by cas_origin/extract_style_vectors.py or equivalent.)
#
# Outputs (per backbone): ./results/actadd_<backbone>/final_predictions.jsonl
# which can then be fed into eval2.py / eval_predictions.py for the table row.
# =============================================================================
set -e

# ---- environment ----
NUM_GPUS=4

# Absolute paths so the script works regardless of cwd
DATA_DIR="${REPO_ROOT}/data"
VEC_DIR="${REPO_ROOT}/cas/results"
TEST_DATA="${DATA_DIR}/test_level_sft.json"
CALIB_DATA="${DATA_DIR}/ablation_level_sft.json"

# ---- per-backbone configuration ----
# Each entry: MODEL_PATH | VECTOR_PATH | ORAL_LAYERS | BEST_SINGLE_LAYER | ALL_MODEL_LAYERS
# BEST_SINGLE_LAYER is the empirical best single layer found by the Task-2 layer
# sweep of cas/ablation/01_steering_oral_layers.py (the top of the hockey-stick).
# Update if your sweep places it elsewhere.
BACKBONES=(
    "Qwen/Qwen2.5-3B-Instruct|${VEC_DIR}/Qwen2.5-3B-Instruct_L5_minus_L0/style_vectors_l5_minus_l0.pt|12-28|20|0-35"
    "mistralai/Mistral-7B-Instruct-v0.3|${VEC_DIR}/Mistral-7B-Instruct-v0.3_L5_minus_L0/style_vectors_l5_minus_l0.pt|15-25|18|0-31"
    "Qwen/Qwen2.5-7B-Instruct|${VEC_DIR}/Qwen2.5-7B-Instruct_L5_minus_L0/style_vectors_l5_minus_l0.pt|12-27|20|0-27"
)

# =============================================================================

for entry in "${BACKBONES[@]}"; do
    IFS='|' read -r MODEL_PATH VECTOR_PATH ORAL BEST_LAYER ALL_LAYERS <<< "$entry"
    MODEL_NAME=$(basename "$MODEL_PATH")
    OUT_DIR="./results/actadd_${MODEL_NAME}"
    PRED_FILE="${OUT_DIR}/final_predictions.jsonl"
    mkdir -p "$OUT_DIR"

    echo "################################################################"
    echo "## ActAdd baseline on $MODEL_NAME"
    echo "##   single layer: $BEST_LAYER  |  L_oral (ref): $ORAL"
    echo "##   output: $OUT_DIR"
    echo "################################################################"

    # ------------------------------------------------------------------
    # Step 1/3: calibration sweep on the single best layer
    # ------------------------------------------------------------------
    echo ">>> [1/3] Calibration sweep on best single layer $BEST_LAYER ..."
    CUDA_VISIBLE_DEVICES=0 python -u inject_ablation.py \
        --variant actadd \
        --mode calibrate \
        --model_path "$MODEL_PATH" \
        --vector_path "$VECTOR_PATH" \
        --data_path "$CALIB_DATA" \
        --output_dir "$OUT_DIR" \
        --oral_layers "$ORAL" \
        --best_single_layer "$BEST_LAYER" \
        --all_model_layers "$ALL_LAYERS"

    # ------------------------------------------------------------------
    # Step 2/3: pick best fixed alpha (selected for target level 3)
    # ------------------------------------------------------------------
    echo ">>> [2/3] Selecting best fixed alpha ..."
    python -u inject_ablation.py \
        --variant actadd \
        --mode analyze \
        --model_path "$MODEL_PATH" \
        --vector_path "$VECTOR_PATH" \
        --data_path "$CALIB_DATA" \
        --output_dir "$OUT_DIR" \
        --oral_layers "$ORAL" \
        --best_single_layer "$BEST_LAYER" \
        --all_model_layers "$ALL_LAYERS" \
        --fixed_alpha_target 3.0

    # ------------------------------------------------------------------
    # Step 3/3: chunked prediction on test set (4 GPUs)
    # ------------------------------------------------------------------
    echo ">>> [3/3] Prediction on test set ($NUM_GPUS GPUs) ..."
    rm -f "${PRED_FILE}.part"*
    for ((i=0; i<NUM_GPUS; i++)); do
        CUDA_VISIBLE_DEVICES=$i python -u inject_ablation.py \
            --variant actadd \
            --mode predict \
            --model_path "$MODEL_PATH" \
            --vector_path "$VECTOR_PATH" \
            --data_path "$TEST_DATA" \
            --output_dir "$OUT_DIR" \
            --prediction_file "$PRED_FILE" \
            --oral_layers "$ORAL" \
            --best_single_layer "$BEST_LAYER" \
            --all_model_layers "$ALL_LAYERS" \
            --num_chunks "$NUM_GPUS" \
            --chunk_idx "$i" \
            > "${OUT_DIR}/pred_gpu${i}.log" 2>&1 &
    done
    wait
    cat "${PRED_FILE}.part"* > "$PRED_FILE"
    rm "${PRED_FILE}.part"*

    echo ">>> Predictions: $PRED_FILE"
    echo ""
done

echo "================================================================"
echo "ActAdd baseline complete on all backbones."
echo "Next: run eval2.py / eval_predictions.py on each results/actadd_<backbone>/final_predictions.jsonl"
echo "================================================================"
