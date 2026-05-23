#!/usr/bin/env bash
set -e

# 3. row GPU count
NUM_GPUS=4

# ==============================================================================
# datasetconfig
# ==============================================================================
# predictionusetest
TEST_DATA="../../../dataset/add_stat/test_level_sft.json"

# calibration/fituse (file)
CALIB_DATA="../../../dataset/add_stat/ablation_level_sft.json"

# ==============================================================================
# column (: model path|vectorpath|layers)
#MODEL_NAMES=(
#  "Qwen2.5-3B-Instruct"
#  "Mistral-7B-Instruct-v0.3"
#  "Llama-3.2-1B-Instruct"
#  "Qwen2.5-7B-Instruct"
#  "Qwen2.5-14B-Instruct"
#  "Qwen2.5-32B-Instruct"
##  "Qwen2.5-72B-Instruct"
#)
# ==============================================================================
MODELS_TO_RUN=(
    "Qwen/Qwen2.5-3B-Instruct|./results/Qwen2.5-3B-Instruct_L5_minus_L0/style_vectors_l5_minus_l0.pt|12-28"
    "meta-llama/Llama-3.2-1B-Instruct|./results/Llama-3.2-1B-Instruct_L5_minus_L0/style_vectors_l5_minus_l0.pt|5-10"
    "mistralai/Mistral-7B-Instruct-v0.3|./results/Mistral-7B-Instruct-v0.3_L5_minus_L0/style_vectors_l5_minus_l0.pt|15-25"
    "Qwen/Qwen2.5-7B-Instruct|./results/Qwen2.5-7B-Instruct_L5_minus_L0/style_vectors_l5_minus_l0.pt|12-28"
    "Qwen/Qwen2.5-14B-Instruct|./results/Qwen2.5-14B-Instruct_L5_minus_L0/style_vectors_l5_minus_l0.pt|12-28"
    "Qwen/Qwen2.5-32B-Instruct|./results/Qwen2.5-32B-Instruct_L5_minus_L0/style_vectors_l5_minus_l0.pt|12-28"
    # "/path/to/another-model|./vecs/style_vectors.pt|10-25"
)

# ==============================================================================
# loop
# ==============================================================================

for entry in "${MODELS_TO_RUN[@]}"; do
    IFS='|' read -r MODEL_PATH VECTOR_PATH LAYERS <<< "$entry"

    MODEL_NAME=$(basename "$MODEL_PATH")
    OUTPUT_DIR="./results/${MODEL_NAME}_experiment"
    PRED_FILE="${OUTPUT_DIR}/final_predictions.jsonl"

    echo "################################################################"
    echo "Processing Model: $MODEL_NAME"
    echo "Layers: $LAYERS"
    echo "Output Dir: $OUTPUT_DIR"
    echo "################################################################"

    mkdir -p "$OUTPUT_DIR"

    # --------------------------------------------------------------------------
    # Step 1: Calibration (datagenerate)
    # --------------------------------------------------------------------------
    echo ">>> [1/4] Calibration (Generating Sweeps)..."
    CUDA_VISIBLE_DEVICES=0 python -u 02_cas_injector.py \
        --mode calibrate \
        --model_path "$MODEL_PATH" \
        --vector_path "$VECTOR_PATH" \
        --data_path "$CALIB_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --selected_layers "$LAYERS"

    # --------------------------------------------------------------------------
    # Step 2: Analysis (fitfunction)
    # --------------------------------------------------------------------------
    echo ">>> [2/4] Analysis (Fitting Linear Function)..."
    python -u 02_cas_injector.py \
        --mode analyze \
        --model_path "$MODEL_PATH" \
        --vector_path "$VECTOR_PATH" \
        --data_path "$CALIB_DATA" \
        --output_dir "$OUTPUT_DIR"

    # --------------------------------------------------------------------------
    # Step 3: Parallel Prediction (4 GPUs)
    # --------------------------------------------------------------------------
    echo ">>> [3/4] Parallel Prediction (4 GPUs)..."

    rm -f "${PRED_FILE}.part"*

    for ((i=0; i<NUM_GPUS; i++)); do
        echo "   Launching Chunk $i on GPU $i..."
        CUDA_VISIBLE_DEVICES=$i python -u 02_cas_injector.py \
            --mode predict \
            --model_path "$MODEL_PATH" \
            --vector_path "$VECTOR_PATH" \
            --data_path "$TEST_DATA" \
            --output_dir "$OUTPUT_DIR" \
            --prediction_file "$PRED_FILE" \
            --selected_layers "$LAYERS" \
            --num_chunks $NUM_GPUS \
            --chunk_idx $i \
            > "${OUTPUT_DIR}/pred_log_gpu${i}.log" 2>&1 &
    done
#                --force_slope 0.5 \
#            --force_intercept 0.0 \

    wait
    echo "   All chunks finished."

 # shard
    cat "${PRED_FILE}.part"* > "$PRED_FILE"
    rm "${PRED_FILE}.part"*
    echo "   Merged predictions to: $PRED_FILE"

    # --------------------------------------------------------------------------
 # Step 4: Evaluation (envevaluation)
    # --------------------------------------------------------------------------
    echo ">>> [4/4] Evaluating Metrics..."

 # rowevaluation ../eval/eval_predictions.py
    python -u ../eval/eval_predictions.py "$PRED_FILE"

    echo "Model $MODEL_NAME Completed!"
    echo "----------------------------------------------------------------"

done

echo "All Models Processed Successfully."