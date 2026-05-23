#!/usr/bin/env bash

# rowrownote
# defaultuse GPU 1-7byshard 1500

set -e

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-true}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

model_path=Qwen/Qwen-2.5-3b-Instruct
data_path=../dataset
result_path=./results/train_sft/multi_gpu_qwen3b_1500_attention_reweight
head_path=./results/train_sft/qwen3b_1500/identify_attention_head/attribution_result.json

GPU_IDS="1 2 3 4 5 6 7"
num_chunks=$(echo "$GPU_IDS" | wc -w)
num_samples=1500
reweight_alpha=0.5 # canbymodify alphacanlayersloop

mkdir -p "$result_path/logs"

run_job() {
  local gpu_id=$1
  local chunk_idx=$2
  local alpha=$3
  local out_dir="${result_path}/alpha_${alpha}/chunk_${chunk_idx}"
  mkdir -p "$out_dir"
  echo "Launching chunk ${chunk_idx} on GPU ${gpu_id} with alpha=${alpha} ..."
  CUDA_VISIBLE_DEVICES="$gpu_id" python -u -m oral_attention_reweight \
    --model-path "$model_path" \
    --data_path "$data_path/train_sft.json" \
    --output-path "$out_dir" \
    --answers-file "$out_dir/answers.json" \
    --attention_head_path "$head_path" \
    --head_type hal \
    --top_k 20 \
    --reweight_alpha "$alpha" \
    --temperature 0 \
    --num_samples "$num_samples" \
    --num-chunks "$num_chunks" \
    --chunk-idx "$chunk_idx" \
    --max_new_tokens 50 \
    --seed 678 \
    > "${result_path}/logs/chunk_${chunk_idx}_alpha_${alpha}.log" 2>&1 &
}

# onlyexample alpha=0.5required alpha canlayers for loop
alpha_list="0.5"
for alpha in $alpha_list; do
  chunk_idx=0
  for gpu_id in $GPU_IDS; do
    run_job "$gpu_id" "$chunk_idx" "$alpha"
    chunk_idx=$((chunk_idx + 1))
    sleep 1
  done
  wait
done

echo "All jobs finished. Results at $result_path"
