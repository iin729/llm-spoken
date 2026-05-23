#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Leave-one-domain-out IA3 training (Table fig:eff_ood in the paper).
#
# NOTE: OOD splits are not bundled in the repo. Build them locally with:
#   python -m data.tools.split_by_source  \
#       --in $REPO_ROOT/data/train.json   \
#       --out_dir $REPO_ROOT/data/ood
# After that you'll have data/ood/{train_4*,test_*}.json
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
cd "$REPO_ROOT/baselines/ia3"

for DATASET in "${DATASETS[@]}"; do
    TRAIN_FILE="$DATA_DIR/train_4${DATASET}.json"
    EVAL_FILE="$DATA_DIR/test_${DATASET}.json"
    OUTPUT_DIR="$REPO_ROOT/baselines/ia3/runs/${HF_NAME}_ia3_ood_${DATASET}"

    if [ ! -f "$TRAIN_FILE" ]; then
        echo "[skip] OOD train file missing: $TRAIN_FILE"
        echo "       Build it from data/train.json first (see header)."
        continue
    fi

    mkdir -p "$OUTPUT_DIR"
    echo "=== IA3 OOD train | $HF_NAME | held-out: $DATASET ==="

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
done
