#!/bin/bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Prompt-mode ablation on the SOTA backbone (Mistral-7B-Instruct-v0.3).
# PM1 (raw) is already done — test_acc=0.9646. Here we sweep PM2 (instruction)
# and PM3 (guidelines) to see if prompt design pushes it further.
#
# How to launch detached:
#   nohup bash run_reward_mistral7b_pm23.sh \
#       > run_0521_3.log 2>&1 < /dev/null &
#   disown
#   tail -f run_0521_3.log

set -euo pipefail

############################################
# Resolve paths relative to THIS script
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$SCRIPT_DIR"

############################################
# GPU / environment
############################################
: "${CUDA_VISIBLE_DEVICES:=0,1}"
export CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  NUM_GPUS=$NUM_GPUS"
export TOKENIZERS_PARALLELISM=false
: "${HF_HOME:=$HOME/.cache/huggingface}"
export HF_HOME

############################################
# Data
############################################
: "${DATA_ROOT:=${REPO_ROOT}/data}"
: "${TRAIN_SRC:=$DATA_ROOT/train_level_sft.json}"
: "${DEV_SRC:=$DATA_ROOT/dev_level_sft.json}"
: "${TEST_SRC:=$DATA_ROOT/test_level_sft.json}"

TRAIN_PAIRS="$SCRIPT_DIR/train_pairs.json"
DEV_PAIRS="$SCRIPT_DIR/dev_pairs.json"
TEST_PAIRS="$SCRIPT_DIR/test_pairs.json"

if [ ! -f "$TRAIN_PAIRS" ] || [ ! -f "$DEV_PAIRS" ] || [ ! -f "$TEST_PAIRS" ]; then
    echo "[data] building pair files (one-time) ..."
    python "$PY_DIR/build_pairs_from_splits.py" \
        --train_json "$TRAIN_SRC" \
        --dev_json   "$DEV_SRC" \
        --test_json  "$TEST_SRC" \
        --out_dir    "$SCRIPT_DIR"
fi

############################################
# Backbone — Mistral-7B-Instruct-v0.3 (current SOTA)
############################################
: "${MODEL_PATH:=mistralai/Mistral-7B-Instruct-v0.3}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "[warn] Backbone path not found locally: $MODEL_PATH"
fi

############################################
# Output roots (same layout as previous scripts)
############################################
: "${OUT_ROOT:=$SCRIPT_DIR/rm_ckpt}"
: "${CKPT_ROOT:=${REPO_ROOT}/evaluator_validation/models}"
mkdir -p "$OUT_ROOT" "$CKPT_ROOT"
echo "[paths] SCRIPT_DIR = $SCRIPT_DIR"
echo "[paths] results    = $OUT_ROOT"
echo "[paths] ckpts      = $CKPT_ROOT"

############################################
# Hyperparameters
############################################
: "${EPOCHS:=20}"
: "${EARLY_STOP_PATIENCE:=2}"
: "${EVAL_N:=100}"
: "${SEED:=42}"

# bs=16, grad_accum=1 as requested. With 2-4 GPUs this gives effective bs=32~64.
: "${BATCH_SIZE:=16}"
: "${GRAD_ACCUM:=1}"
: "${LR:=5e-6}"

NAME="mistral7b"

############################################
# Helper: run one prompt mode
############################################
run_pm () {
    local PM="$1"
    local RUN_TAG="${NAME}_bt_pm${PM}"
    local OUT="$OUT_ROOT/$RUN_TAG"
    local CKPT="$CKPT_ROOT/$RUN_TAG"

    if [ -f "$OUT/summary.txt" ]; then
        echo "[skip] $RUN_TAG already finished (found $OUT/summary.txt)."
        return
    fi
    mkdir -p "$OUT" "$CKPT"

    echo "================================================================"
    echo " >>> Training $RUN_TAG"
    echo "     backbone    = $MODEL_PATH"
    echo "     prompt_mode = $PM"
    echo "     results dir = $OUT"
    echo "     ckpt    dir = $CKPT"
    echo "     bs=$BATCH_SIZE  grad_accum=$GRAD_ACCUM  lr=$LR"
    echo "================================================================"

    python "$PY_DIR/train_reward_model.py" \
        --backbone            "$MODEL_PATH" \
        --out                 "$OUT" \
        --ckpt_dir            "$CKPT" \
        --train_pairs         "$TRAIN_PAIRS" \
        --dev_pairs           "$DEV_PAIRS" \
        --test_pairs          "$TEST_PAIRS" \
        --prompt_mode         "$PM" \
        --epochs              "$EPOCHS" \
        --eval_n              "$EVAL_N" \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        --bf16 \
        --max_grad_norm       1.0 \
        --weight_decay        0.01 \
        --max_len             512 \
        --seed                "$SEED" \
        --log_every           20 \
        --batch_size          "$BATCH_SIZE" \
        --grad_accum          "$GRAD_ACCUM" \
        --lr                  "$LR" \
        2>&1 | tee "$OUT/train.log"

    echo "[done] $RUN_TAG — results: $OUT/summary.txt   |   ckpt: $CKPT"
}

############################################
# Run PM2 (instruction) then PM3 (guidelines)
############################################
run_pm 2
run_pm 3

############################################
# Cross-PM comparison for mistral7b
############################################
echo
echo "================================================================"
echo " Mistral-7B prompt-mode comparison (PM1=raw, PM2=instr, PM3=guide)"
echo "================================================================"
for pm in 1 2 3; do
    f="$OUT_ROOT/${NAME}_bt_pm${pm}/summary.txt"
    if [ -f "$f" ]; then
        echo "----- ${NAME}_bt_pm${pm} -----"
        cat "$f"
        echo
    else
        echo "----- ${NAME}_bt_pm${pm}: summary.txt not found -----"
    fi
done
