import os
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple

import torch
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

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "model path"})
    torch_dtype: Optional[str] = field(default="auto", metadata={"help": "torch.dtype"})


@dataclass
class EvalDataArguments:
    eval_file: str = field(metadata={"help": "evaluationfile pathJSON"})
    max_seq_length: int = field(default=1024, metadata={"help": "prompt maxlength"})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "tip"})


@dataclass
class EvalRunArguments:
    output_dir: str = field(default="./eval_outputs", metadata={"help": "evaluationresults output dir"})
    per_device_eval_batch_size: int = field(default=16, metadata={"help": " batch size"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": " N rowevaluation"})
    num_workers: int = field(default=4, metadata={"help": "DataLoader worker count"})
    seed: int = field(default=42, metadata={"help": ""})

    # Affine ReFT args
    affine_reft_params_path: Optional[str] = field(default=None, metadata={"help": " affine_reft_params.pt path"})
    head_path: Optional[str] = field(default=None, metadata={"help": " Head path"})
    direft_layers: str = field(default="", metadata={"help": " (e.g. '12,13,14')"})

    device_map: Optional[str] = field(default=None, metadata={"help": "device_map"})
    device: str = field(default="cuda", metadata={"help": "device"})
    trust_remote_code: bool = field(default=True, metadata={"help": "trust_remote_code"})


@dataclass
class GenerationArguments:
    generation_max_new_tokens: int = field(default=512)
    generation_num_beams: int = field(default=1)
    generation_do_sample: bool = field(default=False)
    generation_temperature: float = field(default=0.7)
    generation_top_p: float = field(default=0.9)
    generation_top_k: int = field(default=50)


# =============================================================================
# 2. Affine ReFT Controller (Eval Mode)
# =============================================================================
def _load_target_heads(head_path: Optional[str]) -> Optional[Set[Tuple[int, int]]]:
    if not head_path or not os.path.exists(head_path):
        return None
    with open(head_path, "r") as f:
        data = json.load(f)
    heads = set()
    for item in data.get("hal_heads", []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            heads.add((int(item[0]), int(item[1])))
    return heads if len(heads) > 0 else None


class AffineReFTController(nn.Module):
    """
    Affine ReFT Implementation for Evaluation
    h' = h + D*h + B
    """

    def __init__(self, model: nn.Module, direft_layers: str, head_path: Optional[str] = None):
        super().__init__()
        self.params_dict = nn.ParameterDict()  # Scaling
        self.bias_dict = nn.ParameterDict()  # Bias [New]
        self.hook_handles = []

        # 1.
        self.target_res_layers = set()
        if direft_layers.strip():
            for x in direft_layers.split(","):
                if x.strip().isdigit():
                    self.target_res_layers.add(int(x.strip()))

        # 2. Heads
        target_heads = _load_target_heads(head_path)

        # config
        config = model.config
        if hasattr(model, "model"):
            config = model.model.config

        num_layers = getattr(config, "num_hidden_layers", 32)
        num_q_heads = getattr(config, "num_attention_heads", 32)
        hidden_size = getattr(config, "hidden_size", 4096)
        head_dim = hidden_size // num_q_heads

        logger.info(f"Affine ReFT Eval Config: Layers={num_layers}, ResLayers={self.target_res_layers}")

        # 3. initargsand Hooks
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            # --- A. Setup o_proj Intervention ---
            if hasattr(layer.self_attn, "o_proj"):
                key_proj = f"L{layer_idx}_o_proj"

                mask = torch.zeros(hidden_size)
                if target_heads is not None:
                    active_heads_count = 0
                    for h_q in range(num_q_heads):
                        if (layer_idx, h_q) in target_heads:
                            start = h_q * head_dim
                            end = start + head_dim
                            mask[start:end] = 1.0
                            active_heads_count += 1

                    if active_heads_count > 0:
                        # Scaling D
                        self.params_dict[key_proj] = nn.Parameter(torch.zeros(hidden_size))
                        # Bias B
                        self.bias_dict[key_proj] = nn.Parameter(torch.zeros(hidden_size))

                        self.register_buffer(f"mask_{key_proj}", mask)
                        self._register_proj_hook(layer.self_attn.o_proj, key_proj)

            # --- B. Setup Residual Stream Intervention ---
            if layer_idx in self.target_res_layers:
                key_res = f"L{layer_idx}_residual"
                # Scaling D
                self.params_dict[key_res] = nn.Parameter(torch.zeros(hidden_size))
                # Bias B
                self.bias_dict[key_res] = nn.Parameter(torch.zeros(hidden_size))

                self._register_layer_hook(layer, key_res)

    def _register_proj_hook(self, module: nn.Module, key: str):
        def hook_fn(mod, args, output):
            # output: [Batch, Seq, Hidden]
            D = self.params_dict[key]
            B = self.bias_dict[key]
            mask = getattr(self, f"mask_{key}")

            # Safe Dtype Cast
            target_dtype = output.dtype
            target_device = output.device

            D_cast = D.to(device=target_device, dtype=target_dtype)
            B_cast = B.to(device=target_device, dtype=target_dtype)
            mask_cast = mask.to(device=target_device, dtype=target_dtype)

            # Affine ReFT: h' = h + (D * h + B) * mask
            delta = (output * D_cast + B_cast) * mask_cast
            return output + delta

        self.hook_handles.append(module.register_forward_hook(hook_fn))

    def _register_layer_hook(self, module: nn.Module, key: str):
        def hook_fn(mod, args, output):
            hidden = output[0]
            D = self.params_dict[key]
            B = self.bias_dict[key]

            # Safe Dtype Cast
            target_dtype = hidden.dtype
            target_device = hidden.device

            D_cast = D.to(device=target_device, dtype=target_dtype)
            B_cast = B.to(device=target_device, dtype=target_dtype)

            # Affine ReFT: h' = h + D * h + B
            delta = hidden * D_cast + B_cast

            modified_hidden = hidden + delta

            return (modified_hidden,) + output[1:]

        self.hook_handles.append(module.register_forward_hook(hook_fn))


# =============================================================================
# 3. data
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
    if hasattr(model, "device"): return model.device
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        return torch.device(list(model.hf_device_map.values())[0])
    return next(model.parameters()).device


# =============================================================================
# 4. Main
# =============================================================================
def main():
    parser = HfArgumentParser((ModelArguments, EvalDataArguments, EvalRunArguments, GenerationArguments))
    model_args, data_args, eval_args, gen_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO)
    set_seed(eval_args.seed)
    os.makedirs(eval_args.output_dir, exist_ok=True)

    # 1. load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=eval_args.trust_remote_code,
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. data
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
        except:
            prompt_text = instruction

        records.append({
            "prompt_text": prompt_text,
            "label": response,
        })

    dataset = PromptDataset(records)
    collate_fn = build_collate_fn(tokenizer, data_args.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=eval_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=eval_args.num_workers,
        collate_fn=collate_fn,
    )

    # 3. Load model & Affine ReFT
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=eval_args.trust_remote_code,
        device_map=eval_args.device_map,
    )

    if eval_args.affine_reft_params_path:
        logger.info(f"Restoring Affine ReFT from {eval_args.affine_reft_params_path}")

        # initschema
        reft_controller = AffineReFTController(
            model,
            direft_layers=eval_args.direft_layers,
            head_path=eval_args.head_path
        )

        # loadweights
        state_dict = torch.load(eval_args.affine_reft_params_path, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("reft_controller."):
                new_state_dict[k.replace("reft_controller.", "")] = v
            else:
                new_state_dict[k] = v

        missing, unexpected = reft_controller.load_state_dict(new_state_dict, strict=False)
        if missing: logger.warning(f"Missing keys: {missing}")

        model.add_module("reft_controller", reft_controller)

        # controller move to
        model.reft_controller.to(infer_model_main_device(model))
    else:
        logger.info("No Affine ReFT path provided. Running Base Model.")

    if not eval_args.device_map:
        model.to(eval_args.device)
    model.eval()

    # 4.
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

    generated_texts, references, prompt_cache = [], [], []
    main_device = infer_model_main_device(model)

    logger.info(f"***** Running evaluation (Batch Size={eval_args.per_device_eval_batch_size}) *****")

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Generating"):
            input_ids = batch["input_ids"].to(main_device)
            attention_mask = batch["attention_mask"].to(main_device)

            input_seq_len = input_ids.shape[1]
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            for idx in range(generated.size(0)):
                seq = generated[idx]
                gen_ids = seq[input_seq_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                generated_texts.append(text)
                references.append(batch["labels"][idx])
                prompt_cache.append(batch["prompts"][idx])

    # 5.
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