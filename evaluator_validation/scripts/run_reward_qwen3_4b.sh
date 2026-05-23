#!/bin/bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Train a Bradley-Terry reward model on top of Qwen3-4B, using the same
# pair dataset / pipeline as run_reward_models.sh. Goal: see whether 4B
# can match or beat the Mistral-7B baseline.
#
# How to launch detached:
#   nohup bash run_reward_qwen3_4b.sh > run_0521_2.log 2>&1 < /dev/null &
#   disown
#   tail -f run_0521_2.log

set -euo pipefail

############################################
# Resolve paths relative to THIS script
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$SCRIPT_DIR"

############################################
# GPU / environment
############################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  NUM_GPUS=$NUM_GPUS"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

############################################
# Data
############################################
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
TRAIN_SRC="${TRAIN_SRC:-$DATA_ROOT/train_level_sft.json}"
DEV_SRC="${DEV_SRC:-$DATA_ROOT/dev_level_sft.json}"
TEST_SRC="${TEST_SRC:-$DATA_ROOT/test_level_sft.json}"

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
# Backbone — Qwen3-4B
############################################
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "[warn] Backbone path not found locally: $MODEL_PATH"
    echo "       (will fall through to HuggingFace Hub if HF_HUB_OFFLINE is unset)"
fi

############################################
# Output roots (same layout as the multi-model script)
############################################
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/rm_ckpt}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/evaluator_validation/models}"
mkdir -p "$OUT_ROOT" "$CKPT_ROOT"
echo "[paths] SCRIPT_DIR = $SCRIPT_DIR"
echo "[paths] results    = $OUT_ROOT"
echo "[paths] ckpts      = $CKPT_ROOT"

############################################
# Hyperparameters
############################################
PROMPT_MODE="${PROMPT_MODE:-1}"
EPOCHS="${EPOCHS:-20}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
EVAL_N="${EVAL_N:-100}"
SEED="${SEED:-42}"

# 4B sits between 3B (bs=4/ga=4/lr=1e-5) and 7B (bs=2/ga=8/lr=5e-6).
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-5}"

NAME="qwen3_4b"
RUN_TAG="${NAME}_bt_pm${PROMPT_MODE}"
OUT="$OUT_ROOT/$RUN_TAG"
CKPT="$CKPT_ROOT/$RUN_TAG"

if [ -f "$OUT/summary.txt" ]; then
    echo "[skip] $NAME already finished (found $OUT/summary.txt)."
    exit 0
fi
mkdir -p "$OUT" "$CKPT"

echo "================================================================"
echo " >>> Training $NAME ($MODEL_PATH)"
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
    --prompt_mode         "$PROMPT_MODE" \
    --epochs              "$EPOCHS" \
    --eval_n              "$EVAL_N" \
    --early_stop_patience "$EARLY_STOP_PATIENCE" \
    --bf16 \
    --max_grad_norm       1.0 \
    --weight_decay        0.01 \
    --max_len             64 \
    --seed                "$SEED" \
    --log_every           20 \
    --batch_size          "$BATCH_SIZE" \
    --grad_accum          "$GRAD_ACCUM" \
    --lr                  "$LR" \
    2>&1 | tee "$OUT/train.log"

echo "[done] $NAME — results: $OUT/summary.txt   |   ckpt: $CKPT"

############################################
# Quick comparison vs the 7B baseline
############################################
echo
echo "================================================================"
echo " Comparison vs mistral7b baseline"
echo "================================================================"
for tag in "qwen3_4b_bt_pm${PROMPT_MODE}" "mistral7b_bt_pm${PROMPT_MODE}"; do
    f="$OUT_ROOT/$tag/summary.txt"
    if [ -f "$f" ]; then
        echo "----- $tag -----"
        cat "$f"
        echo
    else
        echo "----- $tag: summary.txt not found -----"
    fi
done
