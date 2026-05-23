import os
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    TrainerCallback  # (new)
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1) Args
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
class AffineReFTArguments(TrainingArguments):
    vector_path: Optional[str] = field(default=None)
    head_path: Optional[str] = field(default=None)

    reft_lr: float = field(default=5e-3)
    style_loss_weight: float = field(default=1.0)
    oral_token_weight: float = field(default=1.0)

    direft_layers: str = field(default="", metadata={"help": "Layers"})
    target_modules: str = field(default="o_proj")

    # (new) save params every N epochs
    save_every_n_epochs: int = field(default=5, metadata={"help": "Save reft params every N epochs"})
    # =========== edit here ===========
    # Force save_strategy='no' to avoid huge checkpoint folders
    save_strategy: str = field(default="no", metadata={"help": "Disable HF default checkpointing"})
    # also disable load_best_model_at_end (it requires checkpoints)
    load_best_model_at_end: bool = field(default=False)


# =============================================================================
# 2) Controller (same as beforeunchanged)
# =============================================================================
def _load_target_heads(head_path: Optional[str]) -> Optional[Set[Tuple[int, int]]]:
    if not head_path or not os.path.exists(head_path): return None
    with open(head_path, "r") as f:
        data = json.load(f)
    heads = set()
    # for item in data.get("hal_heads", []) + data.get("non_hal_heads", []):
    for item in data.get("hal_heads", []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            heads.add((int(item[0]), int(item[1])))
    return heads if len(heads) > 0 else None


class AffineReFTController(nn.Module):
    """
    Affine ReFT Implementation:
    Intervention: h' = h + D * h + b
    (Scaling + Shift)
    """

    def __init__(self, model: nn.Module, direft_layers: str, head_path: Optional[str] = None):
        super().__init__()
        self.params_dict = nn.ParameterDict()
        self.bias_dict = nn.ParameterDict()
        self.hook_handles = []
        self.current_deltas = {}

        self.target_res_layers = set()
        if direft_layers.strip():
            for x in direft_layers.split(","):
                if x.strip().isdigit(): self.target_res_layers.add(int(x.strip()))

        target_heads = _load_target_heads(head_path)
        config = model.config
        num_layers = getattr(config, "num_hidden_layers", 32)
        num_q_heads = getattr(config, "num_attention_heads", 32)
        hidden_size = getattr(config, "hidden_size", 4096)
        head_dim = hidden_size // num_q_heads

        logger.info(
            f"Affine ReFT Config: Hidden={hidden_size}, Layers={num_layers}, ResLayers={self.target_res_layers}")

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            if hasattr(layer.self_attn, "o_proj"):
                key_proj = f"L{layer_idx}_o_proj"
                mask = torch.zeros(hidden_size)
                if target_heads is not None:
                    active_heads_count = 0
                    for h_q in range(num_q_heads):
                        if (layer_idx, h_q) in target_heads:
                            start = h_q * head_dim;
                            end = start + head_dim
                            mask[start:end] = 1.0;
                            active_heads_count += 1
                    if active_heads_count > 0:
                        self.params_dict[key_proj] = nn.Parameter(torch.zeros(hidden_size))
                        self.bias_dict[key_proj] = nn.Parameter(torch.zeros(hidden_size))
                        self.register_buffer(f"mask_{key_proj}", mask)
                        self._register_proj_hook(layer.self_attn.o_proj, key_proj, layer_idx)

            if layer_idx in self.target_res_layers:
                key_res = f"L{layer_idx}_residual"
                self.params_dict[key_res] = nn.Parameter(torch.zeros(hidden_size))
                self.bias_dict[key_res] = nn.Parameter(torch.zeros(hidden_size))
                self._register_layer_hook(layer, key_res, layer_idx)

    def _register_proj_hook(self, module: nn.Module, key: str, layer_idx: int):
        def hook_fn(mod, args, output):
            try:
                D = self.params_dict[key];
                B = self.bias_dict[key];
                mask = getattr(self, f"mask_{key}")
                target_dtype = output.dtype;
                target_device = output.device
                D_cast = D.to(device=target_device, dtype=target_dtype)
                B_cast = B.to(device=target_device, dtype=target_dtype)
                mask_cast = mask.to(device=target_device, dtype=target_dtype)
                return output + (output * D_cast + B_cast) * mask_cast
            except Exception as e:
                traceback.print_exc(); raise e

        self.hook_handles.append(module.register_forward_hook(hook_fn))

    def _register_layer_hook(self, module: nn.Module, key: str, layer_idx: int):
        def hook_fn(mod, args, output):
            try:
                hidden = output[0]
                D = self.params_dict[key];
                B = self.bias_dict[key]
                target_dtype = hidden.dtype;
                target_device = hidden.device
                D_cast = D.to(device=target_device, dtype=target_dtype)
                B_cast = B.to(device=target_device, dtype=target_dtype)
                delta = hidden * D_cast + B_cast
                if delta.requires_grad: self.current_deltas[layer_idx] = delta[:, -1, :]
                return (hidden + delta,) + output[1:]
            except Exception as e:
                traceback.print_exc(); raise e

        self.hook_handles.append(module.register_forward_hook(hook_fn))

    def clear_deltas(self):
        self.current_deltas = {}


# =============================================================================
# 3) Callback (new)
# =============================================================================
class SaveReFTCallback(TrainerCallback):
    def __init__(self, controller, save_freq, output_dir):
        self.controller = controller
        self.save_freq = save_freq
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        # state.epoch float 1.0, 2.0
        epoch = round(state.epoch)
        if epoch % self.save_freq == 0:
            # create Epoch file
            save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
            os.makedirs(save_path, exist_ok=True)

            # saveargs
            file_path = os.path.join(save_path, "affine_reft_params.pt")
            torch.save(self.controller.state_dict(), file_path)
            logger.info(f"Saved ReFT params for epoch {epoch} to {file_path}")


# =============================================================================
# 4) Trainer & Collator (same as before)
# =============================================================================
class AffineReFTCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, oral_vocab: set, oral_weight: float = 1.0, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.oral_vocab = oral_vocab;
        self.oral_weight = oral_weight;
        self.tokenizer = tokenizer

    def __call__(self, features, return_tensors=None):
        for f in features: f.pop("level", None)
        batch = super().__call__(features, return_tensors=return_tensors)
        labels = batch["labels"];
        weights = torch.ones_like(labels, dtype=torch.float32)
        if self.oral_weight != 1.0:
            for i in range(labels.size(0)):
                valid = torch.where(labels[i] != -100)[0]
                for t in valid:
                    token_id = int(labels[i, t].item());
                    token_str = self.tokenizer.decode([token_id])
                    if token_str.strip().lower() in self.oral_vocab: weights[i, t] = float(self.oral_weight)
        batch["loss_weights"] = weights
        return batch


class AffineReFTTrainer(Trainer):
    def __init__(self, reft_controller, style_vectors, style_loss_weight, reft_lr, **kwargs):
        super().__init__(**kwargs)
        self.reft_controller = reft_controller;
        self.style_vectors = style_vectors
        self.style_loss_weight = float(style_loss_weight);
        self.reft_lr = float(reft_lr)

    def create_optimizer(self):
        if self.optimizer is not None: return self.optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs["lr"] = self.reft_lr
        self.optimizer = optimizer_cls(params, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_weights = inputs.pop("loss_weights", None)
        actual_model = model.module if hasattr(model, "module") else model
        controller = getattr(actual_model, "reft_controller", self.reft_controller)
        controller.clear_deltas()
        outputs = model(**inputs)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous();
        shift_labels = inputs["labels"][..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if loss_weights is not None:
            shift_weights = loss_weights[..., 1:].contiguous().view(-1).to(loss.device)
            loss = loss * shift_weights
        active = shift_labels.view(-1) != -100
        ce_loss = loss[active].mean() if active.any() else loss.mean()

        align_loss = torch.tensor(0.0, device=ce_loss.device)
        if self.style_vectors is not None and self.style_loss_weight > 0:
            captured_deltas = controller.current_deltas
            for layer_idx, vec in self.style_vectors.items():
                if int(layer_idx) in captured_deltas:
                    actual = captured_deltas[int(layer_idx)]
                    target = vec.view(1, -1).to(actual.device, dtype=actual.dtype)
                    align_loss += F.mse_loss(actual, target.expand_as(actual))
        return (ce_loss + self.style_loss_weight * align_loss,
                outputs) if return_outputs else ce_loss + self.style_loss_weight * align_loss


# =============================================================================
# 5) Main
# =============================================================================
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, AffineReFTArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO)
    set_seed(training_args.seed)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False,
                                              padding_side="right")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    for p in model.parameters(): p.requires_grad = False

    reft_controller = AffineReFTController(model=model, direft_layers=training_args.direft_layers,
                                           head_path=training_args.head_path)
    model.add_module("reft_controller", reft_controller)
    if training_args.bf16: reft_controller.to(dtype=torch.bfloat16)

    style_vectors = None
    if training_args.vector_path and os.path.exists(training_args.vector_path):
        raw = torch.load(training_args.vector_path, map_location="cpu")
        style_vectors = {int(k): v for k, v in raw.items()}
        logger.info(f"Loaded style vectors: {len(style_vectors)} layers")

    # ================= Check Params =================
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.6f}"
    )

    dataset = load_dataset("json", data_files={"train": data_args.train_file})

    def preprocess_function(examples):
        input_ids_list, labels_list = [], []
        for instr, resp in zip(examples["instruction"], examples["output"]):
            messages = []
            if data_args.system_prompt: messages.append({"role": "system", "content": data_args.system_prompt})
            messages.append({"role": "user", "content": instr})
            prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            messages.append({"role": "assistant", "content": resp})
            full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
            if len(full_ids) > data_args.max_seq_length: full_ids = full_ids[: data_args.max_seq_length]
            labels = list(full_ids);
            pl = len(prompt_ids)
            if pl < len(labels):
                labels[:pl] = [-100] * pl
            else:
                labels = [-100] * len(labels)
            input_ids_list.append(full_ids);
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    FILLER_WORDS = ["um", "uh", "oh", "uhm", "mm", "ah", "ok", "so", "okay", "yeah", "you know", "well", "really"]
    oral_vocab = set()
    for w in FILLER_WORDS: oral_vocab.add(w); oral_vocab.add(w.lower()); oral_vocab.add(w.capitalize()); oral_vocab.add(
        " " + w)

    collator = AffineReFTCollator(tokenizer=tokenizer, oral_vocab=oral_vocab,
                                  oral_weight=training_args.oral_token_weight, pad_to_multiple_of=8)

    # Force-override to prevent CLI from creating large files
    training_args.save_strategy = "no"
    training_args.save_steps = 0  # ensure no step-based save
    trainer = AffineReFTTrainer(
        reft_controller=reft_controller,
        style_vectors=style_vectors,
        style_loss_weight=training_args.style_loss_weight,
        reft_lr=training_args.reft_lr,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # (new) register callback
    trainer.add_callback(SaveReFTCallback(reft_controller, training_args.save_every_n_epochs, training_args.output_dir))

    logger.info("*** Starting Affine ReFT Training (With Checkpoints) ***")
    trainer.train()

    # final save
    trainer.save_state()
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Save Affine ReFT params
    torch.save(reft_controller.state_dict(), os.path.join(training_args.output_dir, "affine_reft_params.pt"))
    logger.info(f"Saved Affine ReFT parameters to {os.path.join(training_args.output_dir, 'affine_reft_params.pt')}")


if __name__ == "__main__":
    main()