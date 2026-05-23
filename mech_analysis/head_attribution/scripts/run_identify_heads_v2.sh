export HF_HUB_OFFLINE=True

name=ablation_1k_complex_v2
num_samples=1000
#model_path=Qwen/Qwen-2.5-3b-Instruct
model_path=Qwen/Qwen2.5-3B-Instruct
#Qwen/Qwen2.5-1.5B-Instruct
data_path=../dataset/add_stat
result_path=./results/$name/qwen3b_${num_samples}

attribute.sh
# identify oral-style heads
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m oral_identify_attention_head \
    --model-path $model_path \
    --data_path $data_path/ablation_level_sft_complex_v2.json \
    --output-path $result_path/identify_attention_head \
    --answers-file $result_path/oral_eval_results.json \
    --num_samples $num_samples \
    --seed 678 \
    --temperature 0
