# -*- coding: utf-8 -*-
"""
Evaluation script for LAS Table-4 ablations.

Mirrors eval_affine_reft.py:
  * loads base model
  * rebuilds AblationController from controller_config.json (or CLI args)
  * loads affine_reft_params.pt
  * runs greedy/sampling generation on the eval_file
  * dumps generated_predictions.jsonl + metrics.json (BLEU/ROUGE)

The downstream LA-Acc/BERT scoring is left to eval2.py (called from the .sh).
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# -----------------------------------------------------------------------------
# torch._dynamo shim — see train_ablation.py for the full rationale.
# torch 2.1.2's onnx exporter is broken (missing `ExportOptions`), and any
# path that imports torch._dynamo crashes.  Even `generate()` can hit it via
# torch._compile decorators on certain attention kernels, so we install the
# shim here too.
# -----------------------------------------------------------------------------
import sys as _sys
import torch  # noqa: E402
try:
    import torch._dynamo  # noqa: F401
except Exception:
    import types as _types

    class _ShimMeta(type):
        def __getattr__(cls, name):
            return cls

    class _NoOp(metaclass=_ShimMeta):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

    class _DynamoShim(_types.ModuleType):
        _KNOWN = {
            "disable": lambda fn=None, *a, **k: (
                fn if fn is not None else (lambda g: g)
            ),
            "is_compiling": lambda: False,
            "is_dynamo_compiling": lambda: False,
        }

        def __getattr__(self, name):
            if name in self._KNOWN:
                return self._KNOWN[name]
            return _NoOp

    _sys.modules["torch._dynamo"] = _DynamoShim("torch._dynamo")

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from tqdm.auto import tqdm
import evaluate

from ablation_controller import AblationController

logger = logging.getLogger(__name__)


# =============================================================================
# Args
# =============================================================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Base model path"})
    torch_dtype: Optional[str] = field(default="auto")


@dataclass
class EvalDataArguments:
    eval_file: str = field(metadata={"help": "Eval JSON file"})
    max_seq_length: int = field(default=1024)
    system_prompt: Optional[str] = field(default=None)


@dataclass
class EvalRunArguments:
    output_dir: str = field(default="./eval_outputs")
    per_device_eval_batch_size: int = field(default=16)
    max_eval_samples: Optional[int] = field(default=None)
    num_workers: int = field(default=4)
    seed: int = field(default=42)

    # Controller reconstruction
    affine_reft_params_path: Optional[str] = field(default=None)
    controller_config_path: Optional[str] = field(default=None)
    head_path: Optional[str] = field(default=None)
    direft_layers: str = field(default="")

    # Allow overriding from CLI if config file is absent (mirrors eval_affine_reft.py)
    method: str = field(default="las")
    target_set: str = field(default="oral_lh")
    lora_r: int = field(default=4)
    lora_alpha: float = field(default=8.0)
    lora_dropout: float = field(default=0.0)
    reft_r: int = field(default=2)
    reft_zero_init: bool = field(default=False)
    rnd_seed: int = field(default=42)

    device_map: Optional[str] = field(default=None)
    device: str = field(default="cuda")
    trust_remote_code: bool = field(default=True)


@dataclass
class GenerationArguments:
    generation_max_new_tokens: int = field(default=512)
    generation_num_beams: int = field(default=1)
    generation_do_sample: bool = field(default=False)
    generation_temperature: float = field(default=0.7)
    generation_top_p: float = field(default=0.9)
    generation_top_k: int = field(default=50)


# =============================================================================
# Data
# =============================================================================
class PromptDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def build_collate_fn(tokenizer, max_seq_length: int):
    def collate(batch):
        prompts = [item["prompt_text"] for item in batch]
        tokenized = tokenizer(
            prompts,
            max_length=max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "prompts": prompts,
            "labels": [item["label"] for item in batch],
        }
    return collate


def infer_model_main_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        return torch.device(list(model.hf_device_map.values())[0])
    return next(model.parameters()).device


# =============================================================================
# Main
# =============================================================================
def main():
    parser = HfArgumentParser((ModelArguments, EvalDataArguments, EvalRunArguments, GenerationArguments))
    model_args, data_args, eval_args, gen_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO)
    set_seed(eval_args.seed)
    os.makedirs(eval_args.output_dir, exist_ok=True)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=eval_args.trust_remote_code,
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Prompts ---
    raw_datasets = load_dataset("json", data_files={"eval": data_args.eval_file})
    eval_dataset = raw_datasets["eval"]
    if eval_args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(eval_args.max_eval_samples, len(eval_dataset))))

    records = []
    logger.info("Formatting prompts...")
    for example in tqdm(eval_dataset, desc="Formatting"):
        instruction = example["instruction"]
        response = example["output"]
        messages = []
        if data_args.system_prompt:
            messages.append({"role": "system", "content": data_args.system_prompt})
        messages.append({"role": "user", "content": instruction})
        try:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_text = instruction
        records.append({"prompt_text": prompt_text, "label": response})

    dataset = PromptDataset(records)
    dataloader = DataLoader(
        dataset,
        batch_size=eval_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=eval_args.num_workers,
        collate_fn=build_collate_fn(tokenizer, data_args.max_seq_length),
    )

    # --- Model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=eval_args.trust_remote_code,
        device_map=eval_args.device_map,
    )

    # --- Controller reconstruction ---
    if eval_args.affine_reft_params_path:
        # Prefer config file (saved at train time) over CLI args
        cfg = None
        if eval_args.controller_config_path and os.path.exists(eval_args.controller_config_path):
            with open(eval_args.controller_config_path, "r") as f:
                cfg = json.load(f)
            logger.info(f"Loaded controller config from {eval_args.controller_config_path}")

        method = cfg["method"] if cfg else eval_args.method
        target_set = cfg["target_set"] if cfg else eval_args.target_set
        lora_r = cfg["lora_r"] if cfg else eval_args.lora_r
        lora_alpha = cfg["lora_alpha"] if cfg else eval_args.lora_alpha
        lora_dropout = cfg["lora_dropout"] if cfg else eval_args.lora_dropout
        reft_r = cfg["reft_r"] if cfg else eval_args.reft_r
        # `reft_zero_init` only exists in configs saved by the patched trainer
        # (else default False -> reproduces legacy behaviour).
        reft_zero_init = (cfg.get("reft_zero_init", False)
                          if cfg else eval_args.reft_zero_init)
        rnd_seed = cfg["rnd_seed"] if cfg else eval_args.rnd_seed

        logger.info(
            f"Rebuilding AblationController: method={method} target_set={target_set} "
            f"lora_r={lora_r} reft_r={reft_r} reft_zero_init={reft_zero_init} "
            f"rnd_seed={rnd_seed}"
        )

        controller = AblationController(
            model=model,
            method=method,
            target_set=target_set,
            oral_layers_str=eval_args.direft_layers,
            head_path=eval_args.head_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            reft_r=reft_r,
            reft_zero_init=reft_zero_init,
            rnd_seed=rnd_seed,
        )

        # Load weights (strip optional reft_controller. prefix as in eval_affine_reft.py)
        state_dict = torch.load(eval_args.affine_reft_params_path, map_location="cpu")
        new_sd = {}
        for k, v in state_dict.items():
            new_sd[k.replace("reft_controller.", "")] = v
        missing, unexpected = controller.load_state_dict(new_sd, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading: {missing[:6]}{'...' if len(missing)>6 else ''}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading: {unexpected[:6]}{'...' if len(unexpected)>6 else ''}")

        model.add_module("reft_controller", controller)
        controller.to(infer_model_main_device(model))
        # Keep ReFT R modules in fp32 regardless of base model dtype (see method docstring).
        controller.keep_reft_R_fp32()
    else:
        logger.info("No ablation params provided; running BASE model.")

    if not eval_args.device_map:
        model.to(eval_args.device)
    model.eval()

    # --- Generation kwargs ---
    stop_token_ids = [tokenizer.eos_token_id]
    if "<|im_end|>" in tokenizer.get_vocab():
        stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    gen_kwargs = {
        "max_new_tokens": gen_args.generation_max_new_tokens,
        "num_beams": gen_args.generation_num_beams,
        "do_sample": gen_args.generation_do_sample,
        "temperature": gen_args.generation_temperature,
        "top_p": gen_args.generation_top_p,
        "top_k": gen_args.generation_top_k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_token_ids,
    }

    # --- Generate ---
    generated_texts, references, prompt_cache = [], [], []
    main_device = infer_model_main_device(model)
    logger.info(f"***** Running ablation eval (Batch={eval_args.per_device_eval_batch_size}) *****")
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Generating"):
            input_ids = batch["input_ids"].to(main_device)
            attention_mask = batch["attention_mask"].to(main_device)
            input_seq_len = input_ids.shape[1]
            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            for idx in range(generated.size(0)):
                seq = generated[idx]
                gen_ids = seq[input_seq_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                generated_texts.append(text)
                references.append(batch["labels"][idx])
                prompt_cache.append(batch["prompts"][idx])

    # --- Metrics (BLEU/ROUGE; downstream eval2.py computes L-Acc / BERT) ---
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    bleu_res = bleu_metric.compute(predictions=generated_texts, references=[[r] for r in references], max_order=4)
    rouge_res = rouge_metric.compute(predictions=generated_texts, references=references, use_stemmer=True)
    metrics = {
        "bleu-4": round(bleu_res["bleu"] * 100, 4),
        "rouge-l": round(rouge_res["rougeL"] * 100, 4),
    }
    logger.info(f"Metrics: {metrics}")

    with open(os.path.join(eval_args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(eval_args.output_dir, "generated_predictions.jsonl"), "w") as f:
        for p, g, r in zip(prompt_cache, generated_texts, references):
            json.dump({"prompt": p, "predict": g, "label": r}, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
