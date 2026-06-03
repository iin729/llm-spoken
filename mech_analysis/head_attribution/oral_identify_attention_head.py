"""
identify_oral_heads.py
======================
Section 3.2 head attribution: per-head zero ablation + Contrastive Influence
Score (CIS, Eq. 1 in the paper).

For each (layer, head) we zero out the head's o_proj contribution while the
model generates, and record the probability gap on oral-marker tokens vs
non-oral content tokens. The 2-sigma outliers form the oral-head set
A_oral, which we save as mech_analysis/heads_idx/<model>.json.
"""

import argparse
import json
import math
import os
import sys
from typing import List, Dict, Any

import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.generation.utils import GenerationMixin as _GM

# Make `from zero_ablation import ...` work when this file is run as a script
# OR via `python -m mech_analysis.head_attribution.identify_oral_heads`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zero_ablation import set_zero_ablation_greedy_search

# =========================================================================== #
# 1. pass viadefineargs (hallucinated_tokens etc.)
# =========================================================================== #
_ORIG_VALIDATE = _GM._validate_model_kwargs


def _patched_validate(self, model_kwargs):
    # # (modified)row model_kwargs afterfunction
    # model_kwargs_copy = model_kwargs.copy()
    #
    # # model.generate defineargs
    # custom_keys = {"hallucinated_tokens", "non_hallucinated_tokens", "influence_score", "tokenizer"}
    # for key in custom_keys:
    #     model_kwargs_copy.pop(key, None)
    # return _ORIG_VALIDATE(self, model_kwargs)
    # 1. createmodify model_kwargs
    # model_kwargs after generate

    # (critical fix)
    # mustcreateotherwise model_kwargs modifyafter zero_ablation_greedy_search tokenizer
    to_validate = model_kwargs.copy()

    # removedefineargs
    custom_keys = {"hallucinated_tokens", "non_hallucinated_tokens", "influence_score", "tokenizer"}
    for key in custom_keys:
        to_validate.pop(key, None)

    # cleanupafterfunction
    return _ORIG_VALIDATE(self, to_validate)


_GM._validate_model_kwargs = _patched_validate

# =========================================================================== #
# 2. oral lexicondefine
# =========================================================================== #
filler_WORDS = ['um', 'uh', 'oh', 'uhm', 'mm', 'ah']
interact_WORDS = ['ok', 'so', 'okay', 'yeah', 'you know', 'well', 'really',
                  'then', 'sure', 'now', 'alright', 'look', 'tell me']
repeat_WORDS = [
    'ok ok', 'i i', 'or or', "i don't think i don't think", "i don't think i", 'for for', 'the the',
    'you you', 'a little bit a little bit', 'a little bit a', 'and and', 'how how', 'that that',
    'a lot of a lot of', 'a lot of a', 'in in', 'it sounds like it sounds like', 'it sounds like it',
    "it's it's", 'really really', 'when you were when you were', 'when you were when', 'any any',
    'good good', 'have you been have you been', 'have you been have', 'in the in the', 'in the in',
    'is is', 'like like', 'so so', 'this this', 'your your'
]
simple_WORDS = ["it's", "i'm", "you're", "don't", "you've", "i've",
                'gonna', "we're", "can't", "i'll", "they're", "doesn't"]
stance_WORDS = ['i think', 'i see', 'i mean', 'i guess']

FILLER_WORDS = filler_WORDS + interact_WORDS + repeat_WORDS + simple_WORDS + stance_WORDS


def build_oral_vocab() -> set:
    vocab = set()
    for word in FILLER_WORDS:
        variants = {word, word.lower(), word.capitalize()}
        for v in variants:
            vocab.add(v)
            vocab.add(" " + v)
    return vocab


ORAL_VOCAB = build_oral_vocab()


# =========================================================================== #
# 3. utility
# =========================================================================== #
def split_list(lst: List[Any], n: int) -> List[List[Any]]:
    chunk_size = math.ceil(len(lst) / n) if n > 0 else len(lst)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: List[Any], n: int, k: int) -> List[Any]:
    chunks = split_list(lst, n)
    if k < 0 or k >= len(chunks):
        return []
    return chunks[k]


def json_custom_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    with open(os.path.expanduser(data_path), "r") as f:
        return json.load(f)


def build_non_oral_tokens(text: str) -> List[str]:
    tokens = set()
    for raw in text.replace("\n", " ").split():
        clean = raw.strip(" ,.!?;:\"'()[]{}")
        if not clean: continue
        variants = {clean, clean.lower(), clean.capitalize()}
        for v in variants:
            tokens.add(v)
            tokens.add(" " + v)
    return list(tokens - ORAL_VOCAB)


def extract_influence_matrices(influences: Dict[str, List[List[Dict[str, Any]]]],
                               layer_num: int,
                               head_num: int) -> List[torch.Tensor]:
    matrices = []
    # influences schema: token_string -> [layer_idx][head_idx] -> dict
    for _, v in influences.items():
        mat = torch.zeros(layer_num, head_num)
        for layer_idx in range(min(layer_num, len(v))):
            for head_idx in range(min(head_num, len(v[layer_idx]))):
                cell = v[layer_idx][head_idx]
                if cell:
                    mat[layer_idx][head_idx] = cell.get("influence", 0.0)
        matrices.append(mat)
    return matrices


def save_heatmap(matrices: List[torch.Tensor], output_path: str, prefix: str, sample_id: str):
    if not matrices: return
    avg_mat = torch.mean(torch.stack(matrices), 0)
    avg_arr = np.array(avg_mat.cpu().tolist())
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(avg_arr, cmap="coolwarm", center=0)
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{prefix}_influences_{sample_id}.png"))
    plt.close()


# =========================================================================== #
# 4. evaluation
# =========================================================================== #
def eval_model(args):
    # 1. Zero Ablation Patch
    set_zero_ablation_greedy_search()

    model_path = os.path.expanduser(args.model_path)

    # fix1Tokenizer config
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.return_dict_in_generate = True
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # fix2
    if hasattr(model, "hf_device_map"):
        main_device = next(model.parameters()).device
    else:
        main_device = model.device

    layer_num = args.layer_num or getattr(model.config, "num_hidden_layers", 32)
    head_num = args.head_num or getattr(model.config, "num_attention_heads", 32)

    # 3. dataloadshard
    data = load_dataset(args.data_path)
    data_len = len(data)
    start = max(args.start_idx, 0)
    end = data_len if args.end_idx is None else min(args.end_idx, data_len)
    if args.num_samples is not None and args.num_samples > 0:
        end = min(start + args.num_samples, end)
    data = data[start:end]
    data = get_chunk(data, args.num_chunks, args.chunk_idx)

    os.makedirs(args.output_path, exist_ok=True)
    if args.answers_file:
        os.makedirs(os.path.dirname(os.path.expanduser(args.answers_file)), exist_ok=True)

    all_answers = []

    # 4. ( chat )
    stop_token_ids = [tokenizer.eos_token_id]
    if "<|im_end|>" in tokenizer.get_vocab():
        stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    # tip
    # SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    SYSTEM_PROMPT = "You are a helpful assistant."


    for local_idx, sample in tqdm(list(enumerate(data)), total=len(data)):
        sample_id = sample.get("id", args.start_idx + local_idx)

        # fix3 apply_chat_template Build prompt
        instruction = sample.get("instruction")
        if not instruction:
            written_text = sample.get("written") or sample.get("text") or ""
            instruction = (
                "Please rewrite the following formal sentence into a casual, spoken style, "
                "inserting fillers like 'um' or 'you know' where natural.\n\n"
                f"Formal: {written_text}\nSpoken:"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction}
        ]

        # generate Prompt (add_generation_prompt=True assistant )
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(main_device)

        #
        non_oral_tokens = build_non_oral_tokens(sample.get("written", ""))

        # with torch.inference_mode():
        # # call Patch generate
        #     output_obj, oral_influences, non_oral_influences = model.generate(
        #         **inputs,
        #         max_new_tokens=args.max_new_tokens,
        #         do_sample=True if args.temperature > 0 else False,
        #         temperature=args.temperature if args.temperature > 0 else None,
        #         top_p=args.top_p,
        #         num_beams=args.num_beams,
        #
        # # fix4pass via eos_token_id
        #         eos_token_id=stop_token_ids,
        #         pad_token_id=tokenizer.pad_token_id,
        #
        # # define Zero Ablation args
        #         hallucinated_tokens=list(ORAL_VOCAB),
        #         non_hallucinated_tokens=non_oral_tokens,
        #         tokenizer=tokenizer,
        #         return_dict_in_generate=True,
        #         influence_score=args.influence_score,
        #     )
        with torch.inference_mode():
            # (fix)defineargs model generate args
            model.hallucinated_tokens = list(ORAL_VOCAB)
            model.non_hallucinated_tokens = non_oral_tokens
            model.influence_score = args.influence_score
            model.zero_ablation_tokenizer = tokenizer #

            # call Patch generate
            # notepass viadefineargs _validate_model_kwargs
            output_obj, oral_influences, non_oral_influences = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                num_beams=args.num_beams,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,

                # args/notealready model
                # hallucinated_tokens=list(ORAL_VOCAB),
                # non_hallucinated_tokens=non_oral_tokens,
                # tokenizer=tokenizer,
                # influence_score=args.influence_score,
            )

        #
        # noteoutput_obj.sequences input promptneeds
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_obj.sequences[0][input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # save
        torch.save(oral_influences, os.path.join(args.output_path, f"hallucination_influences_{sample_id}.pth"))
        torch.save(non_oral_influences, os.path.join(args.output_path, f"non_hallucination_influences_{sample_id}.pth"))

        #
        del oral_influences, non_oral_influences
        torch.cuda.empty_cache()

        # read (read)
        # notecanlastor
        # firstafter
        oral_inf_loaded = torch.load(os.path.join(args.output_path, f"hallucination_influences_{sample_id}.pth"))
        non_oral_inf_loaded = torch.load(
            os.path.join(args.output_path, f"non_hallucination_influences_{sample_id}.pth"))

        oral_mats = extract_influence_matrices(oral_inf_loaded, layer_num, head_num)
        non_oral_mats = extract_influence_matrices(non_oral_inf_loaded, layer_num, head_num)

        save_heatmap(oral_mats, args.output_path, "hallucination", sample_id)
        save_heatmap(non_oral_mats, args.output_path, "non_hallucination", sample_id)

        all_answers.append({
            "id": sample_id,
            "prompt": prompt_text, # chat prompt
            "instruction": instruction,
            "generated": generated_text,
            "written": sample.get("written", ""),
            "spoken_ground_truth": sample.get("spoken", ""),
        })

        if args.answers_file:
            with open(os.path.expanduser(args.answers_file), "w") as f:
                json.dump({"samples": all_answers}, f, default=json_custom_serializer, ensure_ascii=False)

    if args.answers_file:
        with open(os.path.expanduser(args.answers_file), "w") as f:
            json.dump({"samples": all_answers}, f, default=json_custom_serializer, ensure_ascii=False)

    get_constrative_influence(args, layer_num, head_num)


def get_constrative_influence(args, layer_num: int, head_num: int):
    files = os.listdir(args.output_path)
    hallucination_samples = []
    non_hallucination_samples = []

    print("Computing contrastive influence...")
    for file in tqdm(files):
        if not file.endswith(".pth"):
            continue
        full_path = os.path.join(args.output_path, file)
        try:
            data = torch.load(full_path, map_location="cpu") # load CPU
            matrices = extract_influence_matrices(data, layer_num, head_num)
            if file.startswith("hallucination"):
                hallucination_samples += matrices
            elif file.startswith("non_hallucination"):
                non_hallucination_samples += matrices
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not hallucination_samples or not non_hallucination_samples:
        print("Not enough influence data to compute contrastive heatmap.")
        return

    # Stack CPU row
    hallucinated_scores = torch.stack(hallucination_samples).float()
    non_hallucinated_scores = torch.stack(non_hallucination_samples).float()

    difference = torch.mean(hallucinated_scores, 0) - torch.mean(non_hallucinated_scores, 0)

    #
    diff_arr = difference.numpy()
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(diff_arr, cmap="coolwarm", center=0)
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{args.output_path}/constrastive_influences.png")
    plt.close()

    results = {}
    # topk
    _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=True)
    indices = [[int(idx // head_num), int(idx % head_num)] for idx in flat_indices]
    results["hal_heads"] = indices

    # topk
    _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=False)
    indices = [[int(idx // head_num), int(idx % head_num)] for idx in flat_indices]
    results["non_hal_heads"] = indices

    with open(f"{args.output_path}/attribution_result.json", "w") as file:
        json.dump(results, file, default=json_custom_serializer)

    print("Top-5 (layer, head):", results["hal_heads"][:5])
    print("Top-5 / (layer, head):", results["non_hal_heads"][:5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="../dataset/train_sft.json")
    parser.add_argument("--output-path", type=str, default="./results/oral/identify_attention_head")
    parser.add_argument("--answers-file", type=str, default="./results/oral/oral_eval_results.json")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512) #
    parser.add_argument("--seed", type=int, default=678)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--influence_score", type=str, default="prob_diff")
    parser.add_argument("--layer_num", type=int, default=None)
    parser.add_argument("--head_num", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    eval_model(args)


if __name__ == "__main__":
    main()