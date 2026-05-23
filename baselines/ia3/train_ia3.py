import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from datasets import load_dataset
# [modify1] IA3Config and IA3Model (get_peft_model )
from peft import IA3Config, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)

logger = logging.getLogger(__name__)

# LoRA q,k,v,o,up,down,gate
# IA3 Attention k, v and FFN down_proj ( LLaMA/Qwen )“”defaultconfig
# noteIA3 args hidden_size vectornone LoRA r argsIA3 args LoRA


# =============================================================================
# 1. argsclass
# =============================================================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "model path"})
    torch_dtype: Optional[str] = field(default="auto", metadata={"help": "torch.dtype"})


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "data path"})
    max_seq_length: int = field(default=1024, metadata={"help": "maxlength"})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "tip"})


@dataclass
class AblationArguments(TrainingArguments):
    output_dir: str = field(default="./saves/ia3_ablation", metadata={"help": "output dir"})

    # weightsargs Loss
    oral_token_weight: float = field(default=2.0, metadata={"help": " Loss weights"})

    # [modify2] IA3 args LoRA args
    # IA3 needs r, alpha, dropout
    # Qwen/LLaMA schema IA3 k, v and down_proj (FFN)
    ia3_target_modules: str = field(default="k_proj,v_proj,down_proj", metadata={"help": "IA3 module"})
    ia3_feedforward_modules: str = field(default="down_proj", metadata={"help": "IA3 FFN module"})


# =============================================================================
# 2. Data Collator ()
# =============================================================================
class ResponseAwareCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, oral_vocab, oral_weight=2.0, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.oral_vocab = oral_vocab
        self.oral_weight = oral_weight

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None):
        levels = None
        if len(features) > 0 and "level" in features[0]:
            levels = [f.pop("level") for f in features]

        batch = super().__call__(features, return_tensors=return_tensors)

        if levels is not None:
            batch["level"] = torch.tensor(levels, dtype=torch.float32)

        labels = batch["labels"]
        weights = torch.ones_like(labels, dtype=torch.float32)

        batch_size = labels.shape[0]
        for i in range(batch_size):
            valid_indices = torch.where(labels[i] != -100)[0]
            if len(valid_indices) == 0:
                continue

            for t_idx in valid_indices:
                token_id = labels[i][t_idx].item()
                token_str = self.tokenizer.decode([token_id])

                clean_str = token_str.strip().lower()
                raw_str = token_str.lower()

                if (clean_str in self.oral_vocab) or (raw_str in self.oral_vocab):
                    weights[i][t_idx] = self.oral_weight

        batch["loss_weights"] = weights
        return batch


# =============================================================================
# 3. Weighted Trainer ()
# =============================================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs.pop("level", None)
        inputs.pop("response_mask", None)
        loss_weights = inputs.pop("loss_weights", None)

        outputs = model(**inputs)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if loss_weights is not None:
            shift_weights = loss_weights[..., 1:].contiguous().view(-1)
            loss = loss * shift_weights

        active_elements = (shift_labels.view(-1) != -100)
        if active_elements.sum() > 0:
            loss = loss[active_elements].sum() / active_elements.sum()
        else:
            loss = loss.sum()

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# 4. Main
# =============================================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, AblationArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False
    if training_args.label_names is None: training_args.label_names = ["labels"]
    if "level" not in training_args.label_names: training_args.label_names.append("level")

    logging.basicConfig(level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)
    set_seed(training_args.seed)

    # 1. Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. config IA3 [modify3]
    logger.info("Initializing IA3 adapters...")

    # target_modules and feedforward_modules list
    target_modules_list = training_args.ia3_target_modules.split(',')
    feedforward_modules_list = training_args.ia3_feedforward_modules.split(',')

    peft_config = IA3Config(
        task_type="CAUSAL_LM",
        target_modules=target_modules_list,
        feedforward_modules=feedforward_modules_list,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. data ()
    dataset = load_dataset("json", data_files={"train": data_args.train_file})

    def preprocess_function(examples):
        model_inputs = {"input_ids": [], "labels": [], "level": []}
        instructions = examples['instruction']
        responses = examples['output']
        spoken_statics = examples['spoken_statics']

        for i in range(len(instructions)):
            instr = instructions[i]
            resp = responses[i]
            lvl = spoken_statics[i]['level']

            messages = []
            if data_args.system_prompt:
                messages.append({"role": "system", "content": data_args.system_prompt})
            messages.append({"role": "user", "content": instr})

            prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            messages.append({"role": "assistant", "content": resp})
            full_ids = tokenizer.apply_chat_template(messages, tokenize=True)

            if len(full_ids) > data_args.max_seq_length:
                full_ids = full_ids[:data_args.max_seq_length]

            labels = list(full_ids)
            prompt_len = len(prompt_ids)
            if prompt_len < len(labels):
                labels[:prompt_len] = [-100] * prompt_len
            else:
                labels = [-100] * len(labels)

            model_inputs["input_ids"].append(full_ids)
            model_inputs["labels"].append(labels)
            model_inputs["level"].append(lvl)

        return model_inputs

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Processing dataset using chat template"
    )

    # 4. define
    FILLER_WORDS = [
        'um', 'uh', 'oh', 'uhm', 'mm', 'ah', 'ok', 'so', 'okay', 'yeah', 'you know', 'well', 'really',
        'then', 'sure', 'now', 'alright', 'look', 'tell me', "it's", "i'm", "you're", "don't", "you've",
        "i've", 'gonna', "we're", "can't", "i'll", "they're", "doesn't", 'i think', 'i see', 'i mean', 'i guess'
    ]
    oral_vocab = set()
    for w in FILLER_WORDS:
        oral_vocab.add(w)
        oral_vocab.add(w.lower())
        oral_vocab.add(w.capitalize())
        oral_vocab.add(" " + w)

    collator = ResponseAwareCollator(
        tokenizer,
        oral_vocab=oral_vocab,
        oral_weight=training_args.oral_token_weight,
        pad_to_multiple_of=8
    )

    # 5. init Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    logger.info("*** Starting IA3 Ablation Training ***")
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()