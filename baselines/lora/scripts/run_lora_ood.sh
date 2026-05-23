#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Leave-one-domain-out LoRA training (Table fig:eff_ood in the paper).
#
# NOTE: This script expects OOD splits that we do NOT bundle in the repo (size).
# To build them locally, run:
#   python -m data.tools.split_by_source  \
#       --in $REPO_ROOT/data/train.json   \
#       --out_dir $REPO_ROOT/data/ood
# After that you'll have:
#   data/ood/train_4{meditod,spokenwoz,house_sale}.json   (train on 2 of 3)
#   data/ood/test_{meditod,spokenwoz,house_sale}.json     (held-out domain)
# -----------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
HF_NAME=$(basename "$MODEL_PATH")
DATA_DIR="${OOD_DATA_DIR:-$REPO_ROOT/data/ood}"

DATASETS=("meditod" "spokenwoz" "house_sale")
cd "$REPO_ROOT/baselines/lora"

for DATASET in "${DATASETS[@]}"; do
    TRAIN_FILE="$DATA_DIR/train_4${DATASET}.json"
    EVAL_FILE="$DATA_DIR/test_${DATASET}.json"
    OUTPUT_DIR="$REPO_ROOT/baselines/lora/runs/${HF_NAME}_lora_ood_${DATASET}"

    if [ ! -f "$TRAIN_FILE" ]; then
        echo "[skip] OOD train file missing: $TRAIN_FILE"
        echo "       Build it from data/train.json first (see header)."
        continue
    fi

    mkdir -p "$OUTPUT_DIR"
    echo "=== LoRA OOD train | $HF_NAME | held-out: $DATASET ==="

    accelerate launch --multi_gpu --num_processes "$NUM_GPUS" train_lora.py \
        --model_name_or_path "$MODEL_PATH" \
        --train_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --learning_rate 5e-5 \
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
        --lora_r 4 \
        --lora_alpha 8 \
        --lora_dropout 0 \
        --lora_target_modules "down_proj" \
        --group_by_length True \
        --ddp_find_unused_parameters False \
        --optim adamw_torch_fused
done
