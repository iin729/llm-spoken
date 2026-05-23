#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Full Supervised Fine-Tuning baseline (SFT_ref) on WSPairs main table.
# Produces the SFT_ref rows of Table 2 in the paper.
#
# Override with env vars: REPO_ROOT, MODEL_PATH, TRAIN_FILE, EVAL_FILE,
#                          OUTPUT_DIR, CUDA_VISIBLE_DEVICES.
# -----------------------------------------------------------------------------

set -e

# Resolve REPO_ROOT relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# ---- Config ----------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/train.json}"
EVAL_FILE="${EVAL_FILE:-$REPO_ROOT/data/test.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/baselines/sft/runs/$(basename "$MODEL_PATH")_sft_ep10}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT/baselines/sft"

echo "=============================="
echo "SFT training"
echo "  Model: $MODEL_PATH"
echo "  Train: $TRAIN_FILE"
echo "  Eval : $EVAL_FILE"
echo "  Out  : $OUTPUT_DIR"
echo "=============================="

accelerate launch --multi_gpu --num_processes "$NUM_GPUS" train_full.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 1e-5 \
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
    --group_by_length True \
    --ddp_find_unused_parameters False \
    --optim adamw_torch_fused

echo "SFT training finished. Checkpoint: $OUTPUT_DIR"
