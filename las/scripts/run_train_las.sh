#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Train LAS (Learned Activation Steering) on three backbones.
# Produces the LAS rows of Table 2 in the paper (main table, in-domain).
#
# For each backbone we:
#   * load the pre-identified oral heads (mech_analysis/heads_idx/<model>.json)
#   * intervene on oral layers (--direft_layers) discovered by causal patching
#   * train an element-wise affine on those components (~0.004-0.006% of weights)
#
# Override with: REPO_ROOT, MODEL_PATH, CUDA_VISIBLE_DEVICES.
# To train only one backbone, set BACKBONE=qwen3b | llama1b | mistral7b.
# -----------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# ---- Common config ---------------------------------------------------------
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/train.json}"
EVAL_FILE="${EVAL_FILE:-$REPO_ROOT/data/test.json}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/las/runs}"

# Per-backbone configs (model name, HF id, heads file, oral layers, LR, BS, GA)
declare -A MODEL_PATH HEADS_FILE LAYERS LR BS GA
MODEL_PATH[qwen3b]="Qwen/Qwen2.5-3B-Instruct"
HEADS_FILE[qwen3b]="$REPO_ROOT/mech_analysis/heads_idx/qwen.json"
LAYERS[qwen3b]="20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35"
LR[qwen3b]="5e-3"; BS[qwen3b]=2; GA[qwen3b]=8

MODEL_PATH[llama1b]="meta-llama/Llama-3.2-1B-Instruct"
HEADS_FILE[llama1b]="$REPO_ROOT/mech_analysis/heads_idx/llama.json"
LAYERS[llama1b]="5,6,7,8,9,10,11,12,13,14,15"
LR[llama1b]="1e-3"; BS[llama1b]=4; GA[llama1b]=4

MODEL_PATH[mistral7b]="mistralai/Mistral-7B-Instruct-v0.3"
HEADS_FILE[mistral7b]="$REPO_ROOT/mech_analysis/heads_idx/mistral.json"
LAYERS[mistral7b]="15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
LR[mistral7b]="5e-3"; BS[mistral7b]=2; GA[mistral7b]=8

BACKBONES="${BACKBONE:-qwen3b llama1b mistral7b}"

# ---- Run ------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

cd "$REPO_ROOT/las"

for B in $BACKBONES; do
    HF_ID="${MODEL_PATH[$B]}"
    HF_NAME=$(basename "$HF_ID")
    HEADS="${HEADS_FILE[$B]}"
    DIREFT_LAYERS="${LAYERS[$B]}"
    LRATE="${LR[$B]}"
    BSIZE="${BS[$B]}"
    GACC="${GA[$B]}"
    OUTPUT_DIR="$SAVE_ROOT/${HF_NAME}_las"

    mkdir -p "$OUTPUT_DIR"
    echo "================================================================="
    echo "Training LAS on $HF_ID"
    echo "  oral layers : $DIREFT_LAYERS"
    echo "  oral heads  : $HEADS"
    echo "  out dir     : $OUTPUT_DIR"
    echo "================================================================="

    accelerate launch --multi_gpu --num_processes "$NUM_GPUS" train_las.py \
        --model_name_or_path "$HF_ID" \
        --train_file "$TRAIN_FILE" \
        --head_path "$HEADS" \
        --direft_layers "$DIREFT_LAYERS" \
        --output_dir "$OUTPUT_DIR" \
        --reft_lr "$LRATE" \
        --style_loss_weight 0.0 \
        --oral_token_weight 1.0 \
        --per_device_train_batch_size "$BSIZE" \
        --gradient_accumulation_steps "$GACC" \
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

    echo "LAS training finished for $B."
done
