#!/bin/bash

# Resolve REPO_ROOT relative to this script (override with `export REPO_ROOT=...`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# =============================================================================
# LAS Table-4 ablations on Qwen-2.5-3B-Instruct (WSPairs / dataset=all).
# Runs all 8 variants sequentially: each does train -> generate -> eval2.py.
#
#   variant tag      method          target_set      notes
#   ---------------  --------------  --------------  -------------------------
#   rnd_lh           las             random_lh       Rnd L+H (single seed)
#   no_layers        las             oral_h_only     -Layers (heads only)
#   no_heads         las             oral_l_only    -Heads  (layers only)
#   no_scale         las_no_scale    oral_lh         -Scale (bias only)
#   no_bias          las_no_bias     oral_lh         -Bias  (scale only)
#   lora             lora            oral_lh         rank-4 LoRA
#   ia3              ia3             oral_lh         scale-only IA^3
#   reft             reft            oral_lh         rank-2 LoReFT
# =============================================================================
set -e

# ---------------- model & data ----------------
MODEL_NAME="Qwen2.5-3B-Instruct"
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
HEAD_PATH="${REPO_ROOT}/mech_analysis/heads_idx/qwen.json"
LAYERS="20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35"

TRAIN_FILE="${REPO_ROOT}/data/train_level_sft.json"
EVAL_FILE="${REPO_ROOT}/data/test_level_sft.json"

# ---------------- common training cfg ----------------
#CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}

export CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

LR=1e-3
BS=2
GA=8
EPOCHS=10
RND_SEED=42

# LoRA / ReFT hyperparams (LoRA matches main exp: r=4, alpha=8, dropout=0)
LORA_R=4
LORA_ALPHA=8
LORA_DROPOUT=0.0
REFT_R=2

TRAIN_SAVE_ROOT="./../saves"

# ---------------- variants ----------------
# Format: tag|method|target_set
VARIANTS=(
  "rnd_lh|las|random_lh"
  "no_layers|las|oral_h_only"
  "no_heads|las|oral_l_only"
  "no_scale|las_no_scale|oral_lh"
  "no_bias|las_no_bias|oral_lh"
  "lora|lora|oral_lh"
  "ia3|ia3|oral_lh"
  "reft|reft|oral_lh"
)

for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r TAG METHOD TARGET <<< "$entry"

  OUTPUT_DIR="${TRAIN_SAVE_ROOT}/${MODEL_NAME}/ablation_${TAG}"
  mkdir -p "$OUTPUT_DIR"

  echo "================================================================"
  echo "[ablation] tag=${TAG} | method=${METHOD} | target_set=${TARGET}"
  echo "         output_dir=${OUTPUT_DIR}"
  echo "================================================================"

  # ----------------- Train -----------------
  accelerate launch --multi_gpu --num_processes "$NUM_GPUS" train_ablation.py \
      --model_name_or_path "$MODEL_PATH" \
      --train_file "$TRAIN_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --method "$METHOD" \
      --target_set "$TARGET" \
      --head_path "$HEAD_PATH" \
      --direft_layers "$LAYERS" \
      --lora_r "$LORA_R" \
      --lora_alpha "$LORA_ALPHA" \
      --lora_dropout "$LORA_DROPOUT" \
      --reft_r "$REFT_R" \
      --rnd_seed "$RND_SEED" \
      --reft_lr "$LR" \
      --oral_token_weight 1.0 \
      --per_device_train_batch_size "$BS" \
      --gradient_accumulation_steps "$GA" \
      --num_train_epochs "$EPOCHS" \
      --logging_steps 2 \
      --save_strategy "no" \
      --bf16 True \
      --remove_unused_columns False \
      --report_to "none"

  echo "[ablation] training finished for tag=${TAG}"

  # ----------------- Eval -----------------
  PARAMS_PATH="${OUTPUT_DIR}/affine_reft_params.pt"
  CONFIG_PATH="${OUTPUT_DIR}/controller_config.json"
  EVAL_DIR="${OUTPUT_DIR}/eval_results"
  mkdir -p "$EVAL_DIR"

  if [ ! -f "$PARAMS_PATH" ]; then
    echo "[ablation] WARNING: params not found, skipping eval for ${TAG}"
    continue
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
      --method "$METHOD" \
      --target_set "$TARGET" \
      --lora_r "$LORA_R" \
      --lora_alpha "$LORA_ALPHA" \
      --lora_dropout "$LORA_DROPOUT" \
      --reft_r "$REFT_R" \
      --rnd_seed "$RND_SEED"

  echo "[ablation] generation finished for tag=${TAG}, results at: ${EVAL_DIR}"

  # ----------------- Downstream metrics -----------------
  python eval2.py "$EVAL_DIR/generated_predictions.jsonl"
  echo "[ablation] eval2.py finished for tag=${TAG}"
done

echo "================================================================"
echo "All 8 ablation runs finished."
echo "================================================================"
