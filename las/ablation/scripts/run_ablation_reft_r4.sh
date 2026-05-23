#!/bin/bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# =============================================================================
# run_ablation_reft_r4.sh
# =============================================================================
# Re-run the ReFT row of LAS Table 4 with two corrections from the audit:
#
#  1. **REFT_R = 4** (was 2):
#     The previous run kept r=2 while LoRA used r=4 — that's half the
#     expressive capacity and breaks the fair-comparison protocol from
#     §6.1.  Keeping r at 4 (NOT increasing it past LoRA's r) matches
#     the rank used across the rest of the ablation table.
#
#  2. **reft_zero_init = True** (was False):
#     With the legacy xavier_uniform_ init on W, ReFT's delta is non-zero
#     at step 0 because Wh + b - Rh ≠ 0.  The model starts perturbed, so
#     the first epochs are spent un-doing that random perturbation rather
#     than learning the task.  Compare:
#         LAS  : D=0, B=0           ⇒ delta=0 at init
#         LoRA : B=0                ⇒ delta=0 at init
#         IA3  : s=1                ⇒ delta=0 at init
#         ReFT (before): W~Xavier   ⇒ delta ≠ 0 at init   ← bug
#     With --reft_zero_init True the controller initializes W := R, so
#     delta = R^T(Rh + 0 - Rh) = 0 at step 0.  Same starting point as
#     the other variants.
#
#  3. **6% LR warmup**:
#     Even with zero-init, ReFT's two parameters (W and R) interact
#     non-linearly through the orthogonal projection.  A short linear
#     warmup helps the optimizer settle before the cosine decay kicks in.
#     This change is harmless for the other variants but their rows are
#     not re-run here.
#
# Everything else is identical to the ReFT row of run_ablation_qwen.sh:
#   * same LR (1e-3), batch size (2), grad accum (8), epochs (10)
#   * same target set (oral_lh)
#   * same oral layers / head path / dataset
#   * same eval pipeline (eval_ablation.py + eval2.py)
#
# Output:
#   ./../saves/Qwen2.5-3B-Instruct/ablation_reft_r4/
#       affine_reft_params.pt
#       controller_config.json
#       eval_results/generated_predictions.jsonl
#       eval_results/metrics.json
# =============================================================================
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---------------- model & data ----------------
MODEL_NAME="${MODEL_NAME:-Qwen2.5-3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
HEAD_PATH="${HEAD_PATH:-${REPO_ROOT}/mech_analysis/heads_idx/qwen.json}"
LAYERS="${LAYERS:-20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35}"

TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/train_level_sft.json}"
EVAL_FILE="${EVAL_FILE:-${REPO_ROOT}/data/test_level_sft.json}"

# ---------------- GPU ----------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

# ---------------- training cfg ----------------
# Same as the original ReFT row except REFT_R=4 (was 2) and warmup added.
LR=1e-3
BS=2
GA=8
EPOCHS=10
RND_SEED=42
WARMUP_RATIO=0.06     # NEW: short linear warmup; harmless and helps ReFT

# Keep LoRA defaults so the controller treats them as fixed (we only run ReFT).
LORA_R=4
LORA_ALPHA=8
LORA_DROPOUT=0.0

# THE FIX: rank 4 (matches LoRA) + zero-init W=R (matches LAS/LoRA/IA3).
REFT_R=4
REFT_ZERO_INIT=True

# ---------------- output ----------------
TRAIN_SAVE_ROOT="${TRAIN_SAVE_ROOT:-./../saves}"
TAG="reft_r4"
OUTPUT_DIR="${TRAIN_SAVE_ROOT}/${MODEL_NAME}/ablation_${TAG}"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo " ReFT ablation re-run (rank=4 + zero-init + warmup)"
echo "================================================================"
echo "  MODEL_NAME       = $MODEL_NAME"
echo "  REFT_R           = $REFT_R       (was 2, LoRA matches at r=4)"
echo "  REFT_ZERO_INIT   = $REFT_ZERO_INIT  (was False — delta=0 at init now)"
echo "  WARMUP_RATIO     = $WARMUP_RATIO"
echo "  LR / BS / GA / EP= $LR / $BS / $GA / $EPOCHS"
echo "  GPUs             = $CUDA_VISIBLE_DEVICES  (NUM_GPUS=$NUM_GPUS)"
echo "  OUTPUT_DIR       = $OUTPUT_DIR"
echo "================================================================"

# ---------------- Train ----------------
accelerate launch --multi_gpu --num_processes "$NUM_GPUS" train_ablation.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --method reft \
    --target_set oral_lh \
    --head_path "$HEAD_PATH" \
    --direft_layers "$LAYERS" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --reft_r "$REFT_R" \
    --reft_zero_init "$REFT_ZERO_INIT" \
    --rnd_seed "$RND_SEED" \
    --reft_lr "$LR" \
    --oral_token_weight 1.0 \
    --per_device_train_batch_size "$BS" \
    --gradient_accumulation_steps "$GA" \
    --num_train_epochs "$EPOCHS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type linear \
    --logging_steps 2 \
    --save_strategy "no" \
    --bf16 True \
    --remove_unused_columns False \
    --report_to "none"

echo "[reft_r4] training finished"

# ---------------- Eval ----------------
PARAMS_PATH="${OUTPUT_DIR}/affine_reft_params.pt"
CONFIG_PATH="${OUTPUT_DIR}/controller_config.json"
EVAL_DIR="${OUTPUT_DIR}/eval_results"
mkdir -p "$EVAL_DIR"

if [ ! -f "$PARAMS_PATH" ]; then
    echo "[reft_r4] WARNING: params not found, skipping eval"
    exit 1
fi

python eval_ablation.py \
    --model_name_or_path "$MODEL_PATH" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$EVAL_DIR" \
    --per_device_eval_batch_size 16 \
    --max_seq_length 1024 \
    --generation_max_new_tokens 500 \
    --generation_num_beams 1 \
    --generation_do_sample False \
    --generation_temperature 0.7 \
    --generation_top_p 0.9 \
    --generation_top_k 50 \
    --device_map auto \
    --direft_layers "$LAYERS" \
    --head_path "$HEAD_PATH" \
    --affine_reft_params_path "$PARAMS_PATH" \
    --controller_config_path "$CONFIG_PATH" \
    --method reft \
    --target_set oral_lh \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --reft_r "$REFT_R" \
    --reft_zero_init "$REFT_ZERO_INIT" \
    --rnd_seed "$RND_SEED"

echo "[reft_r4] generation finished, results at: ${EVAL_DIR}"

# ---------------- Downstream metrics (L-Acc / BERT) ----------------
# eval2.py lives alongside the original ablation runner; if it's not here,
# adjust the path or comment this out and run it manually.
if [ -f "./eval2.py" ]; then
    python eval2.py "$EVAL_DIR/generated_predictions.jsonl"
elif [ -f "../eval/eval_predictions.py" ]; then
    python ../eval/eval_predictions.py "$EVAL_DIR/generated_predictions.jsonl"
elif [ -f "../cas/eval/eval_predictions.py" ]; then
    python ../cas/eval/eval_predictions.py "$EVAL_DIR/generated_predictions.jsonl"
else
    echo "[reft_r4] downstream metric script not found at ./eval2.py or ../*/eval_predictions.py;"
    echo "          run your usual scorer on $EVAL_DIR/generated_predictions.jsonl manually."
fi

echo
echo "================================================================"
echo " ReFT r=4 + zero-init re-run done."
echo " predictions : $EVAL_DIR/generated_predictions.jsonl"
echo " params      : $PARAMS_PATH"
echo " config      : $CONFIG_PATH"
echo
echo " Next:  compare L-Acc / BERT against the LoRA row (r=4) and the LAS"
echo "        row.  Expected: ReFT should now move from 0.248 closer to"
echo "        LoRA's 0.568 (still likely below LAS 0.584 — ReFT's hard"
echo "        orthonormal constraint is genuinely less expressive than"
echo "        LAS's free element-wise affine, which is the conclusion the"
echo "        ablation table is supposed to make)."
echo "================================================================"
