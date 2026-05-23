#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

# calibration/usedataset written / spoken_level_N
DATA_PATH="../../../dataset/add_stat/ablation_level_sft.json"

TARGET_LEVEL=5
BASE_LEVEL=0

NUM_SAMPLES=300

MODEL_NAMES=(
  "Qwen2.5-3B-Instruct"
  "Mistral-7B-Instruct-v0.3"
  "Llama-3.2-1B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct"
  "Qwen2.5-32B-Instruct"
#  "Qwen2.5-72B-Instruct"
)

MODEL_PATHS=(
  "Qwen/Qwen2.5-3B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "meta-llama/Llama-3.2-1B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
#  "Qwen/Qwen2.5-72B-Instruct"
)

# after batch sizeand
BATCH_SIZES=(96 48 128 48 32 16)
#8)

for i in "${!MODEL_NAMES[@]}"; do
  MODEL_NAME="${MODEL_NAMES[$i]}"
  MODEL_PATH="${MODEL_PATHS[$i]}"
  BATCH_SIZE="${BATCH_SIZES[$i]}"

  job_name="${MODEL_NAME}_L${TARGET_LEVEL}_minus_L${BASE_LEVEL}"
  OUTPUT_PATH="./results/${job_name}"
  LOG_DIR="$OUTPUT_PATH/logs"

  mkdir -p "$OUTPUT_PATH" "$LOG_DIR"

  echo "=================================================="
  echo "Starting Extraction (Pure PyTorch Mode)"
  echo "Model: $MODEL_PATH"
  echo "Batch size: $BATCH_SIZE"
  echo "Target-Level: $TARGET_LEVEL | Base-Level: $BASE_LEVEL"
  echo "=================================================="

  echo "target_level=${TARGET_LEVEL}" > "$OUTPUT_PATH/layers.txt"
  echo "base_level=${BASE_LEVEL}" >> "$OUTPUT_PATH/layers.txt"

  python -u 01_extract_style_vectors.py \
      --model_path "$MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --output_path "$OUTPUT_PATH" \
      --num_samples $NUM_SAMPLES \
      --batch_size $BATCH_SIZE \
      --target_level $TARGET_LEVEL \
      --base_level $BASE_LEVEL \
      > "$LOG_DIR/extraction.log" 2>&1

  echo "Done! Check vectors in $OUTPUT_PATH"
done
