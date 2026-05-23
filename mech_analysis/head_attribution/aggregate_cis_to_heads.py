import argparse
import os
import tqdm
import torch
from typing import List, Dict, Set
from collections import defaultdict

# ==========================================
# 1. define (unchanged)
# ==========================================
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

ALL_ORAL_WORDS = filler_WORDS + interact_WORDS + repeat_WORDS + simple_WORDS + stance_WORDS

# functionunchanged
def get_vocab_set(phrase_list: List[str]) -> Set[str]:
    vocab = set()
    for phrase in phrase_list:
        words = phrase.split()
        for w in words:
            vocab.add(w.strip().lower())
    return vocab

def get_clean_key(raw_key: str) -> str:
    if "_" in raw_key:
        return raw_key.rsplit("_", 1)[0].lower().strip()
    return raw_key.lower().strip()

def get_num(modelsize="1.5b"):
    configs = {"1.5b": (28, 12), "3b": (36, 16), "7b": (32, 32), "1b": (16, 32)}
    return configs.get(modelsize, (32, 32))

def main(args):

    layer_num, head_num = get_num(args.modelsize)

    # 1. define Group (and)
    vocab_map = {
        "Total (All Oral)": get_vocab_set(ALL_ORAL_WORDS),
        "Repeats": get_vocab_set(repeat_WORDS),
        "Fillers": get_vocab_set(filler_WORDS),
        "Interact": get_vocab_set(interact_WORDS),
        "Stance": get_vocab_set(stance_WORDS),
        "Simple": get_vocab_set(simple_WORDS)
    }

    # === ===
    # Group
    # noneafter
    GLOBAL_VALID_TOKENS = set()
    for s in vocab_map.values():
        GLOBAL_VALID_TOKENS.update(s)

    # key
    for w in ALL_ORAL_WORDS:
        GLOBAL_VALID_TOKENS.add(w.lower())

    print(f"Global Valid Tokens count: {len(GLOBAL_VALID_TOKENS)}")
    # =================================

    hallucination_groups = {k: [] for k in vocab_map.keys()}
    non_hallucination_samples = []

    token_stats = defaultdict(lambda: {'sum': torch.zeros(layer_num, head_num), 'count': 0})

    files = [f for f in os.listdir(args.input_path) if f.endswith(".pth")]
    print(f"Processing {len(files)} files...")

    # debug
    debug_stance_count = 0

    for file in tqdm.tqdm(files):
        full_path = os.path.join(args.input_path, file)
        try:
            data = torch.load(full_path, map_location="cpu")
        except:
            continue

        if "non_hallucination" in file:
            temp_mats = []
            for key, v in data.items():
                mat = torch.zeros(layer_num, head_num)
                has_val = False
                for l in range(min(layer_num, len(v))):
                    for h in range(min(head_num, len(v[l]))):
                        if v[l][h]:
                            mat[l][h] = v[l][h].get("influence", 0.0)
                            has_val = True
                if has_val: temp_mats.append(mat)
            if temp_mats:
                non_hallucination_samples.append(torch.stack(temp_mats).float().mean(0))

        elif "hallucination" in file:
            file_word_mats = {}

            for raw_key, v in data.items():
                clean_word = get_clean_key(raw_key)

                # === after ===
                # “”
                if clean_word not in GLOBAL_VALID_TOKENS:
                    continue
                # =====================

                mat = torch.zeros(layer_num, head_num)
                has_val = False
                for l in range(min(layer_num, len(v))):
                    for h in range(min(head_num, len(v[l]))):
                        if v[l][h]:
                            mat[l][h] = v[l][h].get("influence", 0.0)
                            has_val = True

                if has_val:
                    if clean_word not in file_word_mats: file_word_mats[clean_word] = []
                    file_word_mats[clean_word].append(mat)
                    token_stats[clean_word]['sum'] += mat
                    token_stats[clean_word]['count'] += 1

                    # debug Stance think, mean
                    if clean_word in vocab_map["Stance"]:
                        debug_stance_count += 1

            # ()
            for group_name, vocab_set in vocab_map.items():
                group_mats = []
                for w, mats in file_word_mats.items():
                    if w in vocab_set:
                        group_mats.extend(mats)
                if group_mats:
                    hallucination_groups[group_name].append(torch.stack(group_mats).float().mean(0))

    print(f"\nDebug: Total Stance-related tokens captured: {debug_stance_count}")

    save_data = {
        "hallucination_groups": hallucination_groups,
        "non_hallucination_samples": non_hallucination_samples,
        "token_stats": dict(token_stats),
        "config": {"layer_num": layer_num, "head_num": head_num, "modelsize": args.modelsize}
    }

    torch.save(save_data, args.output_file)
    print(f"Done! Saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to raw .pth folder")
    parser.add_argument("--output-file", type=str, default="summary_data_v2.pt", help="Save path")
    parser.add_argument("--modelsize", type=str, default='3b')

    # debugargs
    debug_args = [
        "--input-path", "./../results/ablation_1k_complex_v2/multi_gpu_qwen3b_1000/identify_attention_head_repair",
        # "--output-path", "./../results/ablation_1k_complex_v2_llama_all/multi_gpu_llama_1000/identify_attention_head",
        # "--output-path", "./../results/ablation_1k_complex_v2_llama_all/multi_gpu_llama_200/identify_attention_head",
        "--input-path", "./../results/ablation_1k_complex_v2_mistral/multi_gpu_mistral_1000/identify_attention_head",
        # "--output-path", "./../results/ablation_1k_complex_v2_mistral/multi_gpu_mistral_200/identify_attention_head",
        # "--input-path","./../results/ablation_1k_complex_v2_mistral_batch/multi_gpu_mistral_200/identify_attention_head",

        # chunk_0
        # "--output-file","./../figs/qwen/summary_data_v2.pt",
        "--output-file", "./../figs/mistral/summary_data_v2_1000.pt",

        # "--output-file", "./../figs/mistral_summary_data_v2.pt",

        # "--modelsize", "3b",
        "--modelsize", "7b",

        # "--sigma", "2.0",
        # "--standardize", "none"
    ]
    args = parser.parse_args(debug_args)
    main(args)