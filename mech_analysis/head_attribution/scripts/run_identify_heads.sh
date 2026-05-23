export HF_HUB_OFFLINE=True

dataset=train_sft
num_samples=1500
#model_path=Qwen/Qwen-2.5-3b-Instruct
model_path=Qwen/Qwen2.5-3B-Instruct
#Qwen/Qwen2.5-1.5B-Instruct
data_path=../dataset
result_path=./results/$dataset/qwen3b_${num_samples}

attribute.sh
# identify oral-style heads
CUDA_VISIBLE_DEVICES='0' python -m oral_identify_attention_head \
    --model-path $model_path \
    --data_path $data_path/train_sft.json \
    --output-path $result_path/identify_attention_head \
    --answers-file $result_path/oral_eval_results.json \
    --num_samples $num_samples \
    --seed 678 \
    --temperature 0
