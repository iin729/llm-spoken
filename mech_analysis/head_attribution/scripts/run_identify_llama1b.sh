#!/usr/bin/env bash

# row GPU 0-7 8 shard
set -e

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-true}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

name=ablation_1k_complex_v2_llama
num_samples=200
model_path=meta-llama/Llama-3.2-1B-Instruct
data_path=../dataset/add_stat
result_path=./results/$name/multi_gpu_llama_${num_samples}
log_dir="$result_path/logs"

# defineuse GPU column
#GPU_IDS="0 1 2 3 4 5 6 7"
#GPU_IDS="0"
GPU_IDS="0 1 2 3"
# to computeshard
num_chunks=$(echo "$GPU_IDS" | wc -w)

mkdir -p "$result_path" "$log_dir"

echo "Total chunks: $num_chunks"

# definefunction
run_job() {
  local gpu_id=$1
  local chunk_idx=$2
  local out_dir="$result_path/chunk_${chunk_idx}"

 # bedefaultinitialize DDP
  local distinct_port=$((29500 + chunk_idx))

  mkdir -p "$out_dir"

  echo "Launching chunk ${chunk_idx} on GPU ${gpu_id} (Port: $distinct_port)..."

 #
 # MASTER_ADDR and MASTER_PORT beif dist.init_process_group()
  CUDA_VISIBLE_DEVICES="$gpu_id" MASTER_ADDR=localhost MASTER_PORT=$distinct_port \
  python -u -m oral_identify_attention_head \
    --model-path "$model_path" \
    --data_path "$data_path/ablation_level_sft.json" \
    --output-path "$out_dir" \
    --answers-file "$out_dir/oral_eval_results.json" \
    --temperature 0 \
    --num-chunks "$num_chunks" \
    --chunk-idx "$chunk_idx" \
    --num_samples "$num_samples" \
    --seed 678 \
    --max_new_tokens 500 \
    --topk 30 \
    > "$log_dir/chunk_${chunk_idx}.log" 2>&1 &
}

# loopstart
chunk_idx=0
for gpu_id in $GPU_IDS; do
  run_job "$gpu_id" "$chunk_idx"
  chunk_idx=$((chunk_idx + 1))

 # waitLoad model CPU/IO
 # ifmemorycanletbe 5-10sifmemoryletbe 30s
  echo "Waiting 10s before launching next job..."
  sleep 10
done

echo "All jobs launched. Waiting for completion..."
wait

echo "------------------------------------------------"
echo "All jobs finished."
echo "Check logs at: $log_dir"
echo "------------------------------------------------"
# whetherfailedlog
grep -i "error" "$log_dir"/*.log | head -n 5