#!/usr/bin/env bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# =============================================================================
# run_cas_capped_mistral.sh
# =============================================================================
# Rerun CAS on Mistral-7B-Instruct-v0.3 with --alpha_cap 1.5 so the applied
# steering coefficient respects the BERT >= 0.85 envelope the paper already
# promises in prose (Mistral is alpha-sensitive).
#
# Output: results/cas_capped_Mistral-7B-Instruct-v0.3/final_predictions.jsonl
# This is the new Mistral row for Table 2 (CAS), replacing the existing one
# which used uncapped alpha and therefore destroyed BERTScore (0.794).
# =============================================================================
set -e

cd "$(dirname "$0")"

# ---- environment ----
NUM_GPUS=4
DATA_DIR="${REPO_ROOT}/data"
VEC_DIR="${REPO_ROOT}/cas/results"
TEST_DATA="${DATA_DIR}/test_level_sft.json"
CALIB_DATA="${DATA_DIR}/ablation_level_sft.json"

# ---- Mistral-7B-Instruct-v0.3 configuration ----
MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.3"
VECTOR_PATH="${VEC_DIR}/Mistral-7B-Instruct-v0.3_L5_minus_L0/style_vectors_l5_minus_l0.pt"
ORAL_LAYERS="15-25"
BEST_SINGLE_LAYER="18"
ALL_MODEL_LAYERS="0-31"
ALPHA_CAP="1.5"

MODEL_NAME=$(basename "$MODEL_PATH")
OUT_DIR="./results/cas_capped_${MODEL_NAME}"
PRED_FILE="${OUT_DIR}/final_predictions.jsonl"
mkdir -p "$OUT_DIR"

echo "################################################################"
echo "## CAS on $MODEL_NAME with alpha_cap=$ALPHA_CAP"
echo "##   L_oral: $ORAL_LAYERS   |   output: $OUT_DIR"
echo "################################################################"

# ------------------------------------------------------------------
# Step 1/3: calibration sweep  (sweep alpha unconstrained on calibration set)
# ------------------------------------------------------------------
echo ">>> [1/3] Calibration sweep ..."
CUDA_VISIBLE_DEVICES=0 python -u inject_ablation.py \
    --variant full_cas \
    --mode calibrate \
    --model_path "$MODEL_PATH" \
    --vector_path "$VECTOR_PATH" \
    --data_path "$CALIB_DATA" \
    --output_dir "$OUT_DIR" \
    --oral_layers "$ORAL_LAYERS" \
    --best_single_layer "$BEST_SINGLE_LAYER" \
    --all_model_layers "$ALL_MODEL_LAYERS"

# ------------------------------------------------------------------
# Step 2/3: fit alpha*(t) polynomial  (unconstrained fit; cap applied later)
# ------------------------------------------------------------------
echo ">>> [2/3] Fitting alpha*(t) polynomial ..."
python -u inject_ablation.py \
    --variant full_cas \
    --mode analyze \
    --model_path "$MODEL_PATH" \
    --vector_path "$VECTOR_PATH" \
    --data_path "$CALIB_DATA" \
    --output_dir "$OUT_DIR" \
    --oral_layers "$ORAL_LAYERS" \
    --best_single_layer "$BEST_SINGLE_LAYER" \
    --all_model_layers "$ALL_MODEL_LAYERS"

# ------------------------------------------------------------------
# Step 3/3: chunked prediction with alpha_cap=1.5 enforced at inference
# ------------------------------------------------------------------
echo ">>> [3/3] Prediction with alpha_cap=$ALPHA_CAP ($NUM_GPUS GPUs) ..."
rm -f "${PRED_FILE}.part"*
for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python -u inject_ablation.py \
        --variant full_cas \
        --mode predict \
        --model_path "$MODEL_PATH" \
        --vector_path "$VECTOR_PATH" \
        --data_path "$TEST_DATA" \
        --output_dir "$OUT_DIR" \
        --prediction_file "$PRED_FILE" \
        --oral_layers "$ORAL_LAYERS" \
        --best_single_layer "$BEST_SINGLE_LAYER" \
        --all_model_layers "$ALL_MODEL_LAYERS" \
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
echo "Mistral CAS (alpha_cap=$ALPHA_CAP) done:"
echo "  $PRED_FILE"
echo ""
echo "Score it with:"
echo "  bash score_all.sh    # picks up the new directory automatically"
echo "or run score_minimal.py / eval2.py directly on the file."
echo "================================================================"
