import os
import json
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
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
from peft import PeftModel

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "model path"})
    torch_dtype: Optional[str] = field(default="auto", metadata={"help": "Load model torch.dtype"})


@dataclass
class EvalDataArguments:
    eval_file: str = field(metadata={"help": "evaluationfile pathJSON"})
    max_seq_length: int = field(default=1024, metadata={"help": "prompt maxlength"})
    # remove template args apply_chat_template read tokenizer_config.json
    system_prompt: Optional[str] = field(default=None, metadata={"help": "tip"})


@dataclass
class EvalRunArguments:
    output_dir: str = field(default="./eval_outputs", metadata={"help": "evaluationresults output dir"})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": " batch size"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": " N rowevaluation"})
    num_workers: int = field(default=4, metadata={"help": "DataLoader worker count"})
    seed: int = field(default=42, metadata={"help": ""})
    lora_adapter_path: Optional[str] = field(default=None, metadata={"help": "LoRA diroptional"})
    device_map: Optional[str] = field(default=None, metadata={"help": " from_pretrained device_map 'auto'"})
    device: str = field(default="cuda", metadata={"help": "device_map 'cuda:0'"})
    trust_remote_code: bool = field(default=True, metadata={"help": "Load model/enable trust_remote_code"})


@dataclass
class GenerationArguments:
    generation_max_new_tokens: int = field(default=512, metadata={"help": "generatemax token"})
    generation_num_beams: int = field(default=1, metadata={"help": "beam search count"})
    generation_do_sample: bool = field(default=False, metadata={"help": "enablesampling"})
    generation_temperature: float = field(default=0.7, metadata={"help": "sampling"})
    generation_top_p: float = field(default=0.9, metadata={"help": "top-p"})
    generation_top_k: int = field(default=50, metadata={"help": "top-k"})


class PromptDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def build_collate_fn(tokenizer, max_seq_length: int):
    def collate(batch):
        # chat_template
        prompts = [item["prompt_text"] for item in batch]

        # row Tokenize and Padding (Left Padding)
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
            "prompts": prompts, # prompt
            "labels": [item["label"] for item in batch],
            "levels": [item["level"] for item in batch],
        }

    return collate


def infer_model_main_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, (list, tuple)):
            first_device = first_device[0]
        if isinstance(first_device, str):
            return torch.device(first_device)
        if isinstance(first_device, torch.device):
            return first_device
    return next(model.parameters()).device


def main():
    parser = HfArgumentParser((ModelArguments, EvalDataArguments, EvalRunArguments, GenerationArguments))
    model_args, data_args, eval_args, gen_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("Process rank: 0, device: %s", eval_args.device)
    set_seed(eval_args.seed)

    os.makedirs(eval_args.output_dir, exist_ok=True)

    # 1. load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=eval_args.trust_remote_code,
        use_fast=False,
    )

    # Padding Side Decoder-only generate
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer pad_token is None, setting to eos_token.")

    # 2. data
    raw_datasets = load_dataset("json", data_files={"eval": data_args.eval_file})
    eval_dataset = raw_datasets["eval"]
    if eval_args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(eval_args.max_eval_samples, len(eval_dataset))))

    records: List[Dict[str, Any]] = []

    logger.info("Formatting prompts using tokenizer.apply_chat_template...")
    for example in tqdm(eval_dataset, desc="Formatting"):
        instruction = example["instruction"]
        response = example["output"]
        spoken_info = example.get("spoken_statics")
        level = None
        if isinstance(spoken_info, dict):
            level = spoken_info.get("level")

        # Messages schema
        messages = []
        if data_args.system_prompt:
            messages.append({"role": "system", "content": data_args.system_prompt})

        messages.append({"role": "user", "content": instruction})

        # generate Prompt
        # add_generation_prompt=True <|im_start|>assistant\n etc.
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # if tokenizer chat_template ()
            logger.error(f"Error applying chat template: {e}")
            raise e

        records.append({
            "prompt_text": prompt_text, # after
            "label": response,
            "level": level,
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

    # 3. Load model
    prep_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=eval_args.trust_remote_code,
        device_map=eval_args.device_map if eval_args.device_map else None,
    )

    # load LoRA
    if eval_args.lora_adapter_path:
        logger.info("Loading LoRA adapter from %s", eval_args.lora_adapter_path)
        # remove config=config argsload
        model = PeftModel.from_pretrained(
            model,
            eval_args.lora_adapter_path,
            is_trainable=False,
        )

    if not eval_args.device_map:
        model.to(eval_args.device)
    model.eval()
    prep_time = time.time() - prep_start

    main_device = infer_model_main_device(model)

    # 4. generateargs (Stop Tokens)
    stop_token_ids = [tokenizer.eos_token_id]
    # <|im_end|> Qwen class
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
        "eos_token_id": stop_token_ids, # pass vialist
    }

    # 5. loop
    generated_texts: List[str] = []
    references: List[str] = []
    prompt_cache: List[str] = []
    level_cache: List[Optional[float]] = []

    start_time = time.time()
    steps = 0

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_args.per_device_eval_batch_size)

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Generating", total=len(dataloader)):
            input_ids = batch["input_ids"].to(main_device)
            attention_mask = batch["attention_mask"].to(main_device)

            # current Batch Input length
            input_seq_len = input_ids.shape[1]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            #
            for idx in range(generated.size(0)):
                seq = generated[idx]

                # input_seq_len attention_mask.sum()
                # can
                gen_ids = seq[input_seq_len:]

                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                generated_texts.append(text)
                references.append(batch["labels"][idx])
                prompt_cache.append(batch["prompts"][idx])
                level_cache.append(batch["levels"][idx])
            steps += 1

    # 6.
    runtime = time.time() - start_time
    samples_per_second = len(dataset) / runtime if runtime > 0 else 0.0

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    # BLEU needs references list of list
    bleu_res = bleu_metric.compute(
        predictions=generated_texts,
        references=[[ref] for ref in references],
        max_order=4,
    )
    rouge_res = rouge_metric.compute(
        predictions=generated_texts,
        references=references,
        use_stemmer=True,
    )

    metrics = {
        "predict_bleu-4": round(bleu_res["bleu"] * 100, 4),
        "predict_rouge-1": round(rouge_res["rouge1"] * 100, 4),
        "predict_rouge-2": round(rouge_res["rouge2"] * 100, 4),
        "predict_rouge-l": round(rouge_res["rougeL"] * 100, 4),
        "predict_runtime": round(runtime, 2),
        "predict_samples_per_second": round(samples_per_second, 4),
        "predict_model_preparation_time": round(prep_time, 4),
    }

    logger.info("***** Predict metrics *****")
    for key, val in metrics.items():
        logger.info("  %s = %s", key, val)

    metrics_path = os.path.join(eval_args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    predictions_path = os.path.join(eval_args.output_dir, "generated_predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for prompt, pred, ref, level in zip(prompt_cache, generated_texts, references, level_cache):
            record = {
                "prompt": prompt,
                "predict": pred,
                "label": ref,
            }
            if level is not None:
                record["level"] = level
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Done. Results saved to %s", eval_args.output_dir)


if __name__ == "__main__":
    main()