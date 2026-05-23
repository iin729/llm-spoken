# -*- coding: utf-8 -*-
"""
head_reweight.py
================
Validation utility: reweight selected attention heads at inference time to
verify that heads identified by CIS truly mediate orality.

Notes:
1. Uses tokenizer.apply_chat_template (no manual prompt concatenation).
2. Honors model-specific stop tokens (e.g. <|im_end|>).
3. Sets tokenizer.padding_side = "left" for batched generation.
4. Disables no_repeat_ngram_size by default since oral speech naturally repeats.
"""

import argparse
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# -----------------------------
# 1) data sharding utils
# -----------------------------
def split_list(lst: List[Any], n: int) -> List[List[Any]]:
    if n <= 0:
        return [lst]
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: List[Any], n: int, k: int) -> List[Any]:
    chunks = split_list(lst, n)
    if k < 0 or k >= len(chunks):
        return []
    return chunks[k]


# -----------------------------
# 2) dataload
# -----------------------------
def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    with open(os.path.expanduser(data_path), "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# 3) read heads list layer->heads
# -----------------------------
def load_heads(head_path: str, top_k: int, head_type: str) -> List[Tuple[int, int]]:
    with open(os.path.expanduser(head_path), "r", encoding="utf-8") as f:
        data = json.load(f)

    key = "hal_heads" if head_type == "hal" else "non_hal_heads"
    heads = data.get(key, [])

    if top_k is not None and top_k > 0:
        heads = heads[:top_k]

    out: List[Tuple[int, int]] = []
    for x in heads:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            out.append((int(x[0]), int(x[1])))
    return out


def build_layer_head_map(heads: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    layer_map: Dict[int, List[int]] = {}
    for layer, head in heads:
        layer_map.setdefault(layer, []).append(head)
    for k in list(layer_map.keys()):
        layer_map[k] = sorted(set(layer_map[k]))
    return layer_map


# -----------------------------
# 4) module
# -----------------------------
def _get_param_device(model: torch.nn.Module) -> torch.device:
    # first deviceif accelerate device_map='auto'
    # model.device canargs device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _find_attn_o_proj_modules(model: torch.nn.Module) -> List[Tuple[int, torch.nn.Module, str]]:
    patterns = [
        re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.self_attn\.o_proj$"),
        re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.attn\.o_proj$"),
    ]

    found: List[Tuple[int, torch.nn.Module, str]] = []
    for name, module in model.named_modules():
        for pat in patterns:
            m = pat.search(name)
            if m:
                layer_idx = int(m.group(1))
                found.append((layer_idx, module, name))
                break

    found.sort(key=lambda x: x[0])
    return found


# -----------------------------
# 5) Hook register
# -----------------------------
def attach_reweight_hooks(
        model: torch.nn.Module,
        layer_map: Dict[int, List[int]],
        alpha: float,
) -> Tuple[List[torch.utils.hooks.RemovableHandle], Dict[str, Any]]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    config = model.config
    num_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)

    hook_stats: Dict[str, Any] = {
        "total_calls": 0,
        "layer_calls": {},
        "alpha": alpha,
        "layer_map": {k: sorted(set(v)) for k, v in layer_map.items()},
        "enabled_hooks": 0,
    }

    if num_heads <= 0 or hidden_size <= 0:
        return handles, hook_stats
    if hidden_size % num_heads != 0:
        return handles, hook_stats

    head_dim = hidden_size // num_heads
    o_proj_list = _find_attn_o_proj_modules(model)
    if not o_proj_list:
        return handles, hook_stats

    def make_hook(layer_idx: int, head_list: List[int]):
        head_list = sorted(set([h for h in head_list if isinstance(h, int) and 0 <= h < num_heads]))
        if not head_list:
            return None

        cache: Dict[torch.device, torch.Tensor] = {}

        def hook(_module, inputs):
            hook_stats["total_calls"] += 1
            hook_stats["layer_calls"][layer_idx] = hook_stats["layer_calls"].get(layer_idx, 0) + 1

            if not inputs: return
            x = inputs[0]
            if not torch.is_tensor(x) or x.dim() != 3: return

            bsz, seq_len, hsz = x.shape
            if hsz != hidden_size: return

            # reshape
            if not x.is_contiguous():
                x = x.contiguous()

            # [B, S, Hidden] -> [B, S, NumHeads, HeadDim]
            x_view = x.reshape(bsz, seq_len, num_heads, head_dim).clone()

            dev = x_view.device
            head_idx = cache.get(dev)
            if head_idx is None:
                head_idx = torch.tensor(head_list, device=dev, dtype=torch.long)
                cache[dev] = head_idx

            alpha_t = torch.tensor(alpha, device=dev, dtype=x_view.dtype)

            # Head
            x_view[:, :, head_idx, :] *= alpha_t

            # shape
            out = x_view.reshape(bsz, seq_len, hidden_size)
            return (out,)

        return hook

    for layer_idx, module, module_name in o_proj_list:
        target_heads = layer_map.get(layer_idx)
        if not target_heads:
            continue

        hook_fn = make_hook(layer_idx, target_heads)
        if hook_fn is None:
            continue

        handles.append(module.register_forward_pre_hook(hook_fn))
        hook_stats["enabled_hooks"] += 1

    return handles, hook_stats


# -----------------------------
# 6) generateargs
# -----------------------------
def _build_generation_kwargs(args, tokenizer) -> Dict[str, Any]:
    #
    stop_token_ids = [tokenizer.eos_token_id]
    if "<|im_end|>" in tokenizer.get_vocab():
        stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=int(args.max_new_tokens),
        num_beams=int(args.num_beams),
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_token_ids, # fix
    )

    if args.temperature is not None and args.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(args.temperature)
        if args.top_p is not None:
            gen_kwargs["top_p"] = float(args.top_p)
    else:
        gen_kwargs["do_sample"] = False

    # n-gram
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    return gen_kwargs


# -----------------------------
# 7) evaluation
# -----------------------------
def eval_model(args):
    model_path = os.path.expanduser(args.model_path)

    # 1. Tokenizer config
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "left" # generatemust left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. load
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # 3. data
    all_data = load_dataset(args.data_path)
    data_len = len(all_data)
    start = max(int(args.start_idx), 0)
    end = data_len if args.end_idx is None else min(int(args.end_idx), data_len)
    if args.num_samples is not None and int(args.num_samples) > 0:
        end = min(start + int(args.num_samples), end)

    data = all_data[start:end]
    data = get_chunk(data, int(args.num_chunks), int(args.chunk_idx))

    os.makedirs(args.output_path, exist_ok=True)
    answers_path = os.path.expanduser(args.answers_file) if args.answers_file else None
    if answers_path:
        os.makedirs(os.path.dirname(answers_path), exist_ok=True)

    # 4. Hooks
    heads = load_heads(args.attention_head_path, int(args.top_k), args.head_type)
    layer_map = build_layer_head_map(heads)
    handles, hook_stats = attach_reweight_hooks(model, layer_map, float(args.reweight_alpha))

    print(f"[Init] enabled_hooks={hook_stats['enabled_hooks']} alpha={hook_stats['alpha']}")

    gen_kwargs = _build_generation_kwargs(args, tokenizer)
    all_answers: List[Dict[str, Any]] = []

    # defaulttip
    SYSTEM_PROMPT = "You are a helpful assistant."

    try:
        for local_idx, sample in tqdm(list(enumerate(data)), total=len(data)):
            sample_id = sample.get("id", start + local_idx)

            # ================= fixPrompt =================
            instruction = sample.get("instruction")
            if not instruction:
                written = sample.get("written") or sample.get("text") or ""
                instruction = (
                    "Please rewrite the following formal sentence into a casual, spoken style, "
                    "inserting fillers like 'um' or 'you know' where natural.\n\n"
                    f"Formal: {written}\nSpoken:"
                )
            else:
                written = "N/A (Instruction provided)"

            # Chat Template
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction}
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = tokenizer(prompt_text, return_tensors="pt")
            dev = _get_param_device(model)
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            # Hook
            before_calls = hook_stats["total_calls"]
            hook_stats["layer_calls"].clear()

            # generate
            with torch.inference_mode():
                output_obj = model.generate(**inputs, **gen_kwargs)

            #
            seq = output_obj.sequences[0]
            prompt_len = inputs["input_ids"].shape[-1]
            new_ids = seq[prompt_len:] #
            spoken_gen = tokenizer.decode(new_ids, skip_special_tokens=True)

            effective = (hook_stats["total_calls"] > before_calls)
            spoken_gt = sample.get("spoken", "")

            # print3debug
            if local_idx < 3:
                print(f"\n[Sample {sample_id}] Effective: {effective}")
                print(f"Instruction: {instruction[:100]}...")
                print(f"Gen: {spoken_gen}")

            record = {
                "id": sample_id,
                "written": written,
                "spoken_ground_truth": spoken_gt,
                "spoken_generated": spoken_gen,
                "reweight_alpha": float(args.reweight_alpha),
                "head_type": args.head_type,
                "top_k": int(args.top_k),
                "reweight_effective": bool(effective),
                "prompt": prompt_text,
            }
            all_answers.append(record)

            with open(os.path.join(args.output_path, f"sample_{sample_id}.json"), "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)

        if answers_path:
            payload = {
                "meta": {
                    "model_path": args.model_path,
                    "reweight_alpha": float(args.reweight_alpha),
                    "enabled_hooks": hook_stats["enabled_hooks"],
                },
                "samples": all_answers,
            }
            tmp_path = answers_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, answers_path)

    finally:
        for h in handles:
            h.remove()


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="../dataset/train_sft.json")

    #
    parser.add_argument("--output-path", type=str, default="./results/oral/attention_reweight")
    parser.add_argument("--answers-file", type=str, default="./results/oral/attention_reweight/answers.json")

    # heads file
    parser.add_argument(
        "--attention_head_path",
        type=str,
        default="./results/train_sft/qwen3b_1500/identify_attention_head/attribution_result.json",
    )
    parser.add_argument("--head_type", type=str, default="hal", choices=["hal", "non_hal"])
    parser.add_argument("--top_k", type=int, default=20)

    #
    parser.add_argument("--reweight_alpha", type=float, default=0.5)

    # shardsampling
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)

    # generateargs
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    # fixdefault 0
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)

    #
    parser.add_argument("--seed", type=int, default=678)

    args = parser.parse_args()

    set_seed(int(args.seed))
    eval_model(args)


if __name__ == "__main__":
    main()