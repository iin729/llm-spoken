# -*- coding: utf-8 -*-
"""
Training script for LAS Table-4 ablations on Qwen-2.5-3B-Instruct.

Drives the unified AblationController (see ablation_controller.py).
Mirrors train_affine_reft.py's data pipeline / Trainer setup but:
  * drops the style-alignment loss (style_loss_weight=0 in main exp anyway)
  * adds --method / --target_set / lora & reft hyperparams
  * saves params.pt + controller_config.json so eval can rebuild the controller
"""

import os
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Optional

# -----------------------------------------------------------------------------
# torch._dynamo shim — torch 2.1.2's onnx submodule is broken
# (`ExportOptions` missing in `torch.onnx._internal.exporter`), and any path
# that imports torch._dynamo (e.g., HF Trainer's optimizer setup via
# torch._compile decorators) crashes the whole training run.  We install a
# catch-all no-op shim before importing torch so any later
# `import torch._dynamo` / `torch._dynamo.<anything>` succeeds with safe
# defaults.  Same trick used in evaluator_validation/train_reward_model.py.
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
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    TrainerCallback,
)

from ablation_controller import AblationController

logger = logging.getLogger(__name__)


# =============================================================================
# Args
# =============================================================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Base model path"})


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Training json file"})
    max_seq_length: int = field(default=1024)
    system_prompt: Optional[str] = field(default=None)


@dataclass
class AblationArguments(TrainingArguments):
    # --- intervention spec ---
    method: str = field(default="las", metadata={"help": "las|las_no_scale|las_no_bias|lora|ia3|reft"})
    target_set: str = field(default="oral_lh", metadata={"help": "oral_lh|random_lh|oral_l_only|oral_h_only"})
    head_path: Optional[str] = field(default=None, metadata={"help": "json with hal_heads list of [layer, head]"})
    direft_layers: str = field(default="", metadata={"help": "Comma-separated oral layer indices."})

    # --- per-method hyperparams ---
    lora_r: int = field(default=4)
    lora_alpha: float = field(default=8.0)
    lora_dropout: float = field(default=0.0)
    reft_r: int = field(default=2)
    reft_zero_init: bool = field(
        default=False,
        metadata={"help": "If True, init ReFT's W = R so delta=0 at start "
                          "(matches LAS/LoRA/IA3 init convention; recommended for r=4)."},
    )
    rnd_seed: int = field(default=42, metadata={"help": "RNG seed for random_lh component sampling"})

    # --- training options inherited from train_affine_reft.py style ---
    reft_lr: float = field(default=1e-3)
    oral_token_weight: float = field(default=1.0)

    # Disable HF checkpointing (avoids huge dirs); we only save controller params at the end.
    save_strategy: str = field(default="no")
    load_best_model_at_end: bool = field(default=False)
    save_every_n_epochs: int = field(default=999, metadata={"help": "Save controller every N epochs"})


# =============================================================================
# Loss-weighted collator (kept identical to train_affine_reft.py for parity)
# =============================================================================
class WeightedCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, oral_vocab: set, oral_weight: float = 1.0, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.oral_vocab = oral_vocab
        self.oral_weight = oral_weight
        self.tokenizer = tokenizer

    def __call__(self, features, return_tensors=None):
        for f in features:
            f.pop("level", None)
        batch = super().__call__(features, return_tensors=return_tensors)
        labels = batch["labels"]
        weights = torch.ones_like(labels, dtype=torch.float32)
        if self.oral_weight != 1.0:
            for i in range(labels.size(0)):
                valid = torch.where(labels[i] != -100)[0]
                for t in valid:
                    token_id = int(labels[i, t].item())
                    token_str = self.tokenizer.decode([token_id])
                    if token_str.strip().lower() in self.oral_vocab:
                        weights[i, t] = float(self.oral_weight)
        batch["loss_weights"] = weights
        return batch


# =============================================================================
# Trainer (CE loss only; style alignment removed)
# =============================================================================
class AblationTrainer(Trainer):
    def __init__(self, controller, reft_lr, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self.reft_lr = float(reft_lr)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs["lr"] = self.reft_lr
        self.optimizer = optimizer_cls(params, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_weights = inputs.pop("loss_weights", None)
        outputs = model(**inputs)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if loss_weights is not None:
            shift_weights = loss_weights[..., 1:].contiguous().view(-1).to(loss.device)
            loss = loss * shift_weights
        active = shift_labels.view(-1) != -100
        ce_loss = loss[active].mean() if active.any() else loss.mean()
        return (ce_loss, outputs) if return_outputs else ce_loss


# =============================================================================
# Save callback (optional per-N-epoch snapshots)
# =============================================================================
class SaveControllerCallback(TrainerCallback):
    def __init__(self, controller, save_freq, output_dir):
        self.controller = controller
        self.save_freq = max(1, int(save_freq))
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        if epoch <= 0:
            return
        if epoch % self.save_freq == 0:
            save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.controller.state_dict(), os.path.join(save_path, "affine_reft_params.pt"))
            with open(os.path.join(save_path, "controller_config.json"), "w") as f:
                json.dump(self.controller.export_config(), f, indent=2)
            logger.info(f"[ckpt] saved controller for epoch {epoch} -> {save_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, AblationArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO)
    set_seed(training_args.seed)

    # --- Model / tokenizer ---
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True, use_fast=False, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze base
    for p in model.parameters():
        p.requires_grad = False

    # --- Controller ---
    controller = AblationController(
        model=model,
        method=training_args.method,
        target_set=training_args.target_set,
        oral_layers_str=training_args.direft_layers,
        head_path=training_args.head_path,
        lora_r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        reft_r=training_args.reft_r,
        reft_zero_init=training_args.reft_zero_init,
        rnd_seed=training_args.rnd_seed,
    )
    model.add_module("reft_controller", controller)
    if training_args.bf16:
        controller.to(dtype=torch.bfloat16)
    # ReFT's orthogonal-parametrized R must stay in fp32 even when the rest is bf16
    # (torch.linalg.householder_product rejects mixed Float/BFloat16). _compute_delta
    # casts R.weight to the activation dtype on the fly, so this does not affect bf16 forward.
    controller.keep_reft_R_fp32()

    # --- Trainable param report ---
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    logger.info(
        f"[params] method={training_args.method} target_set={training_args.target_set} "
        f"| trainable={trainable} | total={total} | trainable%={100*trainable/total:.6f}"
    )

    # --- Dataset ---
    dataset = load_dataset("json", data_files={"train": data_args.train_file})

    def preprocess(examples):
        input_ids_list, labels_list = [], []
        for instr, resp in zip(examples["instruction"], examples["output"]):
            messages = []
            if data_args.system_prompt:
                messages.append({"role": "system", "content": data_args.system_prompt})
            messages.append({"role": "user", "content": instr})
            prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            messages.append({"role": "assistant", "content": resp})
            full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
            if len(full_ids) > data_args.max_seq_length:
                full_ids = full_ids[: data_args.max_seq_length]
            labels = list(full_ids)
            pl = len(prompt_ids)
            if pl < len(labels):
                labels[:pl] = [-100] * pl
            else:
                labels = [-100] * len(labels)
            input_ids_list.append(full_ids)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

    # Filler-word weighting (kept identical to main LAS pipeline)
    FILLER_WORDS = ["um", "uh", "oh", "uhm", "mm", "ah", "ok", "so", "okay", "yeah", "you know", "well", "really"]
    oral_vocab = set()
    for w in FILLER_WORDS:
        oral_vocab.add(w)
        oral_vocab.add(w.lower())
        oral_vocab.add(w.capitalize())
        oral_vocab.add(" " + w)

    collator = WeightedCollator(
        tokenizer=tokenizer,
        oral_vocab=oral_vocab,
        oral_weight=training_args.oral_token_weight,
        pad_to_multiple_of=8,
    )

    # Force off HF default checkpointing
    training_args.save_strategy = "no"
    training_args.save_steps = 0

    trainer = AblationTrainer(
        controller=controller,
        reft_lr=training_args.reft_lr,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.add_callback(SaveControllerCallback(controller, training_args.save_every_n_epochs, training_args.output_dir))

    logger.info(
        f"*** Starting Ablation Training [method={training_args.method} | target_set={training_args.target_set}] ***"
    )
    trainer.train()

    # --- Final save ---
    os.makedirs(training_args.output_dir, exist_ok=True)
    params_path = os.path.join(training_args.output_dir, "affine_reft_params.pt")
    config_path = os.path.join(training_args.output_dir, "controller_config.json")
    torch.save(controller.state_dict(), params_path)
    with open(config_path, "w") as f:
        json.dump(controller.export_config(), f, indent=2)
    logger.info(f"Saved controller params -> {params_path}")
    logger.info(f"Saved controller config -> {config_path}")
    trainer.save_state()


if __name__ == "__main__":
    main()
