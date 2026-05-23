#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# IA3 baseline on WSPairs main table.
# Produces the IA3 rows of Table 2 in the paper. IA3 is matched to LAS's scope
# (down_proj only) to isolate localization from raw capacity.
#
# Override with: REPO_ROOT, MODEL_PATH, TRAIN_FILE, EVAL_FILE, OUTPUT_DIR,
#                CUDA_VISIBLE_DEVICES.
# -----------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/train.json}"
EVAL_FILE="${EVAL_FILE:-$REPO_ROOT/data/test.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/baselines/ia3/runs/$(basename "$MODEL_PATH")_ia3_ep10}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT/baselines/ia3"

echo "=============================="
echo "IA3 training (target=down_proj)"
echo "  Model: $MODEL_PATH"
echo "  Train: $TRAIN_FILE"
echo "  Out  : $OUTPUT_DIR"
echo "=============================="

accelerate launch --multi_gpu --num_processes "$NUM_GPUS" train_ia3.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 1e-3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --oral_token_weight 1.0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 1 \
    --bf16 True \
    --remove_unused_columns False \
    --report_to none \
    --ia3_target_modules "down_proj" \
    --ia3_feedforward_modules "down_proj"

echo "IA3 training finished. Adapter: $OUTPUT_DIR"

# Optional: evaluate the trained adapter.
EVAL_OUT="${OUTPUT_DIR}/eval_results"
mkdir -p "$EVAL_OUT"
python eval_simple.py \
    --model_name_or_path "$MODEL_PATH" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$EVAL_OUT" \
    --lora_adapter_path "$OUTPUT_DIR" \
    --per_device_eval_batch_size 16 \
    --max_seq_length 1024 \
    --generation_max_new_tokens 500 \
    --generation_num_beams 1 \
    --generation_do_sample False \
    --device_map auto
