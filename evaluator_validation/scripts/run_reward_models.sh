#!/bin/bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Train Bradley-Terry reward models on the WSPairs-derived pair dataset.
#
# This script is *location-independent*: it works whether you keep it under
# the project repo (scripts/evaluator_validation/) or copy it to your GPU NAS
# (e.g. ${REPO_ROOT}/evaluator_validation/). All paths are resolved relative to
# the script itself.
#
# Defaults:
#   - Full fine-tune (not LoRA)
#   - Prompt mode 1 (raw text, no g(·) feature leakage)
#   - 20 epochs max with early-stopping patience=2
#   - Per-epoch dev eval on a 100-pair subsample (fast)
#   - bf16, cosine LR schedule, warmup 10 %, grad-clip 1.0, fused AdamW
#
# Backbones (4 — matched to the paper's 3 controllers + RoBERTa encoder):
#   1) RoBERTa-base                   (~125M, encoder, full FT)
#   2) Llama-3.2-1B-Instruct          (~1B,   causal,  full FT)
#   3) Qwen-2.5-3B-Instruct           (~3B,   causal,  full FT)
#   4) Mistral-7B-Instruct-v0.3       (~7B,   causal,  full FT)
#
# Each backbone:
#   1. Trains (with periodic dev-eval + early stop).
#   2. After training, runs a final FULL-test-set evaluation with the best
#      checkpoint (inside train_reward_model.py → test_results.json).
#
# How to launch detached / nohup:
#   nohup bash run_reward_models.sh > run.log 2>&1 &
#   tail -f run.log

set -euo pipefail

############################################
# Resolve paths relative to THIS script
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$SCRIPT_DIR"   # all python files sit alongside this sh

############################################
# GPU / environment
############################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  NUM_GPUS=$NUM_GPUS"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

############################################
# Data — source corpus splits + built pair files
############################################
# Source corpus JSONs (train/dev/test_level_sft.json). Override DATA_ROOT or
# the individual paths if your data is elsewhere.
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
TRAIN_SRC="${TRAIN_SRC:-$DATA_ROOT/train_level_sft.json}"
DEV_SRC="${DEV_SRC:-$DATA_ROOT/dev_level_sft.json}"
TEST_SRC="${TEST_SRC:-$DATA_ROOT/test_level_sft.json}"

# Built pair files (kept next to this sh).
TRAIN_PAIRS="$SCRIPT_DIR/train_pairs.json"
DEV_PAIRS="$SCRIPT_DIR/dev_pairs.json"
TEST_PAIRS="$SCRIPT_DIR/test_pairs.json"

# Build pair files if missing.
if [ ! -f "$TRAIN_PAIRS" ] || [ ! -f "$DEV_PAIRS" ] || [ ! -f "$TEST_PAIRS" ]; then
    echo "[data] building pair files (one-time) ..."
    python "$PY_DIR/build_pairs_from_splits.py" \
        --train_json "$TRAIN_SRC" \
        --dev_json   "$DEV_SRC" \
        --test_json  "$TEST_SRC" \
        --out_dir    "$SCRIPT_DIR"
fi

############################################
# Backbone paths
############################################
NAS_ROOT="${NAS_ROOT:-${HF_MODELS_DIR:-/path/to/hf_models}}"
ROBERTA_PATH="${ROBERTA_PATH:-$NAS_ROOT/roberta-base}"
LLAMA1B_PATH="${LLAMA1B_PATH:-$NAS_ROOT/Llama-3.2-1B-Instruct}"
QWEN3B_PATH="${QWEN3B_PATH:-$NAS_ROOT/Qwen2.5-3B-Instruct}"
MISTRAL7B_PATH="${MISTRAL7B_PATH:-$NAS_ROOT/Mistral-7B-Instruct-v0.3}"

for p in "$ROBERTA_PATH" "$LLAMA1B_PATH" "$QWEN3B_PATH" "$MISTRAL7B_PATH"; do
    if [ ! -d "$p" ]; then
        echo "[warn] Backbone path not found locally: $p"
        echo "       (will fall through to HuggingFace Hub if HF_HUB_OFFLINE is unset)"
    fi
done

############################################
# Output roots
############################################
# Light-weight results (logs, eval/test JSONs, summary) — small, easy to copy.
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/rm_ckpt}"
# Heavy model checkpoints — separate path so result-copy stays cheap.
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/evaluator_validation/models}"
mkdir -p "$OUT_ROOT" "$CKPT_ROOT"
echo "[paths] SCRIPT_DIR = $SCRIPT_DIR"
echo "[paths] results    = $OUT_ROOT"
echo "[paths] ckpts      = $CKPT_ROOT"

############################################
# Shared hyperparameters (full fine-tune, SFT-style)
############################################
PROMPT_MODE="${PROMPT_MODE:-1}"           # 1=raw, 2=instruction, 3=guidelines
EPOCHS="${EPOCHS:-20}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
EVAL_N="${EVAL_N:-100}"
SEED="${SEED:-42}"

COMMON_ARGS=(
    --train_pairs       "$TRAIN_PAIRS"
    --dev_pairs         "$DEV_PAIRS"
    --test_pairs        "$TEST_PAIRS"
    --prompt_mode       "$PROMPT_MODE"
    --epochs            "$EPOCHS"
    --eval_n            "$EVAL_N"
    --early_stop_patience "$EARLY_STOP_PATIENCE"
    --bf16
    --max_grad_norm     1.0
    --weight_decay      0.01
    --max_len           512
    --seed              "$SEED"
    --log_every         20
)

############################################
# Helper: run one backbone end-to-end
############################################
run_one () {
    local NAME="$1"; local BACKBONE="$2"; shift 2
    local RUN_TAG="${NAME}_bt_pm${PROMPT_MODE}"
    local OUT="$OUT_ROOT/$RUN_TAG"
    local CKPT="$CKPT_ROOT/$RUN_TAG"
    if [ -f "$OUT/summary.txt" ]; then
        echo "[skip] $NAME already finished (found $OUT/summary.txt)."
        return
    fi
    mkdir -p "$OUT" "$CKPT"
    echo "================================================================"
    echo " >>> Training $NAME ($BACKBONE)"
    echo "     results dir = $OUT"
    echo "     ckpt    dir = $CKPT"
    echo "     extra       = $*"
    echo "================================================================"
    python "$PY_DIR/train_reward_model.py" \
        --backbone "$BACKBONE" \
        --out "$OUT" \
        --ckpt_dir "$CKPT" \
        "${COMMON_ARGS[@]}" \
        "$@" 2>&1 | tee "$OUT/train.log"
    echo "[done] $NAME — results: $OUT/summary.txt   |   ckpt: $CKPT"
}

############################################
# Per-backbone configs (small → big LR ↓, BS ↓)
############################################

# 1) RoBERTa-base (~125M, encoder, full FT)
run_one "roberta-base" "$ROBERTA_PATH" \
    --batch_size 16 \
    --grad_accum 1 \
    --lr 2e-5

# 2) Llama-3.2-1B-Instruct (~1B, causal, full FT)
run_one "llama1b" "$LLAMA1B_PATH" \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 1e-5

# 3) Qwen-2.5-3B-Instruct (~3B, causal, full FT)
run_one "qwen3b" "$QWEN3B_PATH" \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 1e-5

# 4) Mistral-7B-Instruct-v0.3 (~7B, causal, full FT — heaviest)
run_one "mistral7b" "$MISTRAL7B_PATH" \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 5e-6

############################################
# Final aggregated summary
############################################
echo
echo "================================================================"
echo " ALL DONE — per-model summaries:"
echo "================================================================"
for d in "$OUT_ROOT"/*/summary.txt; do
    [ -f "$d" ] || continue
    echo "----- $d -----"
    cat "$d" || true
done
echo
echo "All result logs under: $OUT_ROOT"
echo "All model ckpts under: $CKPT_ROOT"
