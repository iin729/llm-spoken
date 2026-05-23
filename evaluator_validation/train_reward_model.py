"""Train a Bradley-Terry pairwise reward model.

Loss:
    L = -log σ(r(x_more) - r(x_less))
where r(x) is a scalar produced by a regression head on top of an LM's pooled
hidden state. We use `AutoModelForSequenceClassification(num_labels=1)`, which
works for both encoder LMs (RoBERTa) and causal LMs (Qwen / Llama).

Input format
------------
Each text is wrapped in a chat-style prompt BEFORE tokenization for causal LMs
(via `tokenizer.apply_chat_template`); encoder LMs without a chat template
fall back to the raw text. The wrapper primes the LM to act as an evaluator
of spoken/oral style:

    <system> You are an expert evaluator of spoken/oral style in English text.
    <user>   Rate the spoken/conversational style of the following text.
             Text: {text}

The scalar head reads the final-position hidden state.

Supported backbones (others should also work — pass any HF model name):
    - roberta-base                (~125M, full fine-tune)
    - Qwen/Qwen2.5-1.5B           (~1.5B, LoRA recommended)
    - Qwen/Qwen2.5-3B             (~3B,   LoRA recommended)
    - meta-llama/Llama-3.2-7B     (~7B,   LoRA recommended)

Inputs
------
    --train_pairs scripts/evaluator_validation/train_pairs.json
    --dev_pairs   scripts/evaluator_validation/dev_pairs.json
    --test_pairs  scripts/evaluator_validation/test_pairs.json

Each pair file has records of the form (from build_pairs_from_splits.py):

    { text_a, text_b, gt_label ∈ {A,B,Tie}, level_a, level_b, margin, ... }

Tie rows are skipped at training time by default.

Outputs (under --out)
---------------------
    args.json              training hyperparameters
    eval_results.json      per-epoch dev pair-acc
    test_results.json      final test pair-acc (broken down by margin)
    best/                  best dev-epoch checkpoint (LoRA adapters + head)

Usage examples
--------------

    # RoBERTa-base full fine-tune (encoder; no chat template needed)
    python scripts/evaluator_validation/train_reward_model.py \
        --backbone roberta-base \
        --out scripts/evaluator_validation/rm_ckpt/roberta_bt \
        --epochs 3 --batch_size 16 --lr 2e-5

    # Qwen-2.5-3B with LoRA (causal; uses chat template)
    python scripts/evaluator_validation/train_reward_model.py \
        --backbone Qwen/Qwen2.5-3B \
        --use_lora --lora_r 16 --bf16 \
        --out scripts/evaluator_validation/rm_ckpt/qwen3b_bt \
        --epochs 2 --batch_size 4 --grad_accum 4 --lr 5e-5

    # Llama-3.2-7B with LoRA
    python scripts/evaluator_validation/train_reward_model.py \
        --backbone meta-llama/Llama-3.2-7B \
        --use_lora --lora_r 16 --bf16 \
        --out scripts/evaluator_validation/rm_ckpt/llama7b_bt \
        --epochs 2 --batch_size 2 --grad_accum 8 --lr 5e-5
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import torch

# --------------------------------------------------------------------------
# Workaround for torch-2.1.x installs whose `torch.onnx._internal.exporter`
# is broken (missing `ExportOptions`). The chain is:
#   Optimizer ops / model.save_pretrained / accelerate.is_compiled_module
#   -> torch._dynamo.{graph_break,disable,eval_frame.OptimizedModule,...}
#   -> `import torch._dynamo` -> import torch.onnx -> ImportError.
#
# We install a recursive no-op shim under the name `torch._dynamo` so that
# arbitrary chained access (e.g. `torch._dynamo.eval_frame.OptimizedModule`)
# never throws. Key correctness properties:
#   * `disable(fn)` returns `fn` (so `@disable`-decorated callables still work).
#   * `is_compiling()` returns False.
#   * Any other attribute → a `_NoOp` *class* (not an instance), so:
#       - `isinstance(model, _NoOp)` is False (correct for
#         `accelerate.utils.is_compiled_module(model)`).
#       - chained access like `.eval_frame.OptimizedModule` resolves to
#         `_NoOp` again (via metaclass `__getattr__`).
#       - calling `_NoOp(...)` creates a harmless instance (whose only role
#         is to be callable and return None).
import sys as _sys
try:
    import torch._dynamo  # noqa: F401
except Exception as _e:
    import types as _types

    class _ShimMeta(type):
        """Metaclass so any attribute access on the _NoOp class returns _NoOp
        itself, enabling chains like `_NoOp.eval_frame.OptimizedModule`."""
        def __getattr__(cls, name):
            return cls

    class _NoOp(metaclass=_ShimMeta):
        """A class no real model is an instance of. Callable as a no-op."""
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
            return _NoOp  # return the CLASS — works for both
                          # `isinstance(model, X)` and `X(...)`.

    _sys.modules["torch._dynamo"] = _DynamoShim("torch._dynamo")
    print(f"[warn] torch._dynamo unavailable ({_e!r}); installed recursive "
          f"no-op shim.", file=_sys.stderr)
# --------------------------------------------------------------------------

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# Three prompt-mode presets, selected via --prompt_mode 1/2/3.
#
#   1 (default, "raw")        : just the text, no instruction      — most
#                               independent of g(·); reviewer-proof.
#   2 ("instruction")         : a single neutral one-sentence rating ask;
#                               no feature definitions / no rating scale leak.
#   3 ("guidelines")          : the full Compare-Annotation Guidelines from
#                               the paper Listing 2 (with 6 Rate levels). USE
#                               THIS ONLY FOR ABLATION — it injects g(·)'s
#                               feature lexicon and rating scale into the
#                               reward model, which weakens independence.
PROMPT_PRESETS = {
    "1": {
        "name": "raw",
        "system": "",
        "user_template": "{text}",
    },
    "2": {
        "name": "instruction",
        "system": "",
        "user_template": (
            "Please rate how spoken/conversational the following text sounds.\n\n"
            "Text: {text}"
        ),
    },
    "3": {
        "name": "guidelines",
        "system": "",
        # Full Compare-Annotation Guidelines (Listing 2 in the paper).
        # See LISTING2_GUIDELINES below.
        "user_template": None,  # filled in at module load
    },
}

LISTING2_GUIDELINES = r"""Spoken-vs-Written Compare Annotation Guidelines

Task
Given an English text, rate how spoken it sounds. "More spoken" means more spontaneous, conversational delivery while preserving all facts and intended meaning.

Core signals of "more spoken"
- Structure and density: Shorter clauses, sentence splitting, looser syntax, lower lexical density. Written style favors complex subordination and dense noun phrases.
- Contractions and simpler wording: I'm we're it's; simpler verbs and phrasing over formal alternatives.
- Fillers and discourse markers: um, uh, hmm; well, so, okay, look, anyway, you know. These organize turns and soften entries.
- Stance and hedging: I think, I guess, maybe, kind of, sort of, probably. Shows speaker stance/uncertainty.
- Interaction cues: you know?, right?, yeah?, direct address (you), brief acknowledgments (yeah, mm-hm). Signals audience engagement or turn-yielding.
- Repetition and repairs: small natural repeats, restarts, self corrections (The price is it's about), echoing words for emphasis. Should feel purposeful, not random noise.
- Fragmentation and rhythm: Sentence fragments, trailing off, incremental delivery (Okay, so the plan). Still coherent at the discourse level.
- Prosody-like punctuation: Ellipses and dashes can mark spoken rhythm when used sparingly.

Reference scale (for intuition, not labeling)
- Rate 0 (written): Formal structure, high lexical density, no fillers, no repetition.
- Rate 0.2 (spoken, slight): Light contractions, mild simplification, rare markers.
- Rate 0.4 (spoken, low): Occasional fillers (um/uh/okay), well/so, slight sentence splitting.
- Rate 0.6 (spoken, medium): Clear conversational flow, some hedges (I think/maybe), more fragmentation, small natural repetition.
- Rate 0.8 (spoken, high): Frequent interaction cues (you know/right?), more hedging, noticeable repetition, shorter clauses, still coherent.
- Rate 1 (spoken, high): Strongly spontaneous; frequent fillers, discourse markers, turn-yielding cues; more repetition and repairs; facts preserved.

Annotation note
Do not penalize minor punctuation or casing differences unless they affect meaning or readability. Focus on conversational cues and overall spoken feel while ensuring all information remains intact.

---
Text: {text}
"""
PROMPT_PRESETS["3"]["user_template"] = LISTING2_GUIDELINES

# Back-compat aliases (the script still accepts these flags directly).
DEFAULT_SYSTEM_PROMPT = PROMPT_PRESETS["1"]["system"]
DEFAULT_USER_TEMPLATE = PROMPT_PRESETS["1"]["user_template"]


def resolve_prompt(args):
    """Pick the (system, user_template) pair based on --prompt_mode +
    optional CLI overrides --system_prompt / --user_template."""
    preset = PROMPT_PRESETS.get(str(args.prompt_mode))
    if preset is None:
        raise ValueError(f"--prompt_mode must be one of "
                         f"{list(PROMPT_PRESETS.keys())}, got {args.prompt_mode}")
    system = args.system_prompt if args.system_prompt is not None else preset["system"]
    user = args.user_template if args.user_template is not None else preset["user_template"]
    if "{text}" not in user:
        raise ValueError("user_template must contain `{text}` placeholder.")
    print(f"[prompt] mode={args.prompt_mode} ({preset['name']})")
    return system, user


# ----------------------------------------------------------------------------
# Prompt formatting — chat template for causal LMs, raw text for encoders.
# ----------------------------------------------------------------------------
def make_text_formatter(tokenizer, system_prompt: str, user_template: str):
    """Return a function `format(text)->str` that wraps a single text in the
    appropriate prompt for this tokenizer.

    For causal LMs that expose `apply_chat_template`, we use it with
    add_generation_prompt=True so the scalar head reads the assistant-tag
    hidden state. For encoder LMs without a chat template (RoBERTa, etc.) we
    fall back to raw text.
    """
    has_chat = (
        getattr(tokenizer, "chat_template", None) is not None
        and callable(getattr(tokenizer, "apply_chat_template", None))
    )

    if not has_chat:
        print(f"[fmt] tokenizer has no chat_template — using raw text input.")

        def _fmt_raw(text: str) -> str:
            return text
        return _fmt_raw, False

    print(f"[fmt] tokenizer has chat_template — wrapping each text in chat prompt.")

    def _fmt_chat(text: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user",
                         "content": user_template.format(text=text)})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return _fmt_chat, True


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class PairDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return r["text_more"], r["text_less"]


def load_pairs_for_training(path: str, skip_ties: bool = True) -> List[Dict[str, Any]]:
    """Load a *_pairs.json and turn each row into a (text_more, text_less) pair
    using the row's gt_label as the supervisory signal.
    """
    data = json.loads(Path(path).read_text())
    out = []
    for r in data:
        label = r.get("gt_label")
        if label not in ("A", "B"):
            if skip_ties:
                continue
            label = "A"
        if label == "A":
            more, less = r["text_a"], r["text_b"]
        else:
            more, less = r["text_b"], r["text_a"]
        out.append({
            "text_more": more, "text_less": less,
            "src_label": label, "src_record": r,
        })
    return out


def load_pairs_for_eval(path: str) -> List[Dict[str, Any]]:
    """Load a *_pairs.json as-is for evaluation (keeps text_a / text_b /
    gt_label / margin)."""
    data = json.loads(Path(path).read_text())
    # Evaluator-side we keep the original A/B (no need to skip ties — we just
    # never count them as correct unless both r are equal, which we ignore).
    return data


# ----------------------------------------------------------------------------
# Collator
# ----------------------------------------------------------------------------
@dataclass
class PairCollator:
    tokenizer: Any
    max_len: int
    fmt_fn: Any  # callable: text -> formatted str

    def __call__(self, batch):
        more = [self.fmt_fn(b[0]) for b in batch]
        less = [self.fmt_fn(b[1]) for b in batch]
        enc_more = self.tokenizer(more, padding=True, truncation=True,
                                  max_length=self.max_len, return_tensors="pt")
        enc_less = self.tokenizer(less, padding=True, truncation=True,
                                  max_length=self.max_len, return_tensors="pt")
        return enc_more, enc_less


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def guess_target_modules(model):
    name = type(model).__name__.lower()
    if "roberta" in name or "bert" in name:
        return ["query", "key", "value", "dense"]
    if "qwen" in name or "llama" in name or "mistral" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_model_and_tokenizer(backbone: str, use_lora: bool, lora_r: int,
                              dtype, hf_token: str = None):
    print(f"[model] Loading {backbone}  (use_lora={use_lora}, dtype={dtype})")
    tok = AutoTokenizer.from_pretrained(backbone, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=1, torch_dtype=dtype, token=hf_token,
    )
    model.config.pad_token_id = tok.pad_token_id

    if use_lora:
        if not _HAS_PEFT:
            raise ImportError("`peft` not installed. `pip install peft`.")
        target_modules = guess_target_modules(model)
        print(f"[lora] target_modules = {target_modules}")
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05,
            target_modules=target_modules, bias="none",
            modules_to_save=["score", "classifier"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    return model, tok


# ----------------------------------------------------------------------------
# Forward / loss / eval
# ----------------------------------------------------------------------------
def reward(model, enc, device):
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    return out.logits.squeeze(-1)


def bt_loss(r_more, r_less):
    return -F.logsigmoid(r_more - r_less).mean()


@torch.no_grad()
def evaluate(model, tok, eval_records, fmt_fn, device,
             batch_size: int, max_len: int, name: str = "eval"):
    model.eval()
    by_margin = defaultdict(lambda: [0, 0])  # margin -> [correct, total]
    correct = 0
    total = 0
    n = len(eval_records)
    for i in range(0, n, batch_size):
        batch = eval_records[i:i + batch_size]
        a_texts = [fmt_fn(r["text_a"]) for r in batch]
        b_texts = [fmt_fn(r["text_b"]) for r in batch]
        enc_a = tok(a_texts, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
        enc_b = tok(b_texts, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
        r_a = reward(model, enc_a, device)
        r_b = reward(model, enc_b, device)
        for ra, rb, rec in zip(r_a.tolist(), r_b.tolist(), batch):
            if rec["gt_label"] not in ("A", "B"):
                # Tie GT — ignore (we don't predict ties)
                continue
            pred = "A" if ra > rb else "B"
            ok = (pred == rec["gt_label"])
            by_margin[rec["margin"]][0] += int(ok)
            by_margin[rec["margin"]][1] += 1
            correct += int(ok)
            total += 1
    overall_acc = correct / total if total else float("nan")
    margin_breakdown = {
        int(m): {"acc": (c / t if t else float("nan")),
                 "n": t, "correct": c}
        for m, (c, t) in sorted(by_margin.items())
    }
    return {
        "split": name,
        "n_pairs_used": total,
        "n_pairs_skipped_tie": n - total,
        "pair_acc": overall_acc,
        "by_margin": margin_breakdown,
    }


# ----------------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------------
def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Where the (potentially large) model checkpoint goes. Default = <out>/best.
    # Use --ckpt_dir to redirect checkpoints to a separate disk so light-weight
    # result files (eval_results.json, test_results.json, summary.txt, train.log)
    # can be copied around without dragging multi-GB weights along.
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (out / "best")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(out / "args.json", "w") as f:
        json.dump({**vars(args), "_resolved_ckpt_dir": str(ckpt_dir)}, f, indent=1)
    print(f"[paths] results dir = {out}")
    print(f"[paths] ckpt    dir = {ckpt_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if args.bf16
             else (torch.float16 if args.fp16 else torch.float32))

    # Resolve prompt mode (1/2/3) — CLI --system_prompt / --user_template
    # override the preset if given.
    system_prompt, user_template = resolve_prompt(args)

    # Data
    train_records = load_pairs_for_training(args.train_pairs, skip_ties=True)
    dev_full = load_pairs_for_eval(args.dev_pairs)
    test_records = load_pairs_for_eval(args.test_pairs)
    # Sub-sample dev for per-epoch eval (fast). Tie pairs are kept in pool but
    # `evaluate()` ignores them.
    if args.eval_n is not None and args.eval_n > 0 and args.eval_n < len(dev_full):
        rng = random.Random(args.eval_seed)
        dev_records = rng.sample(dev_full, args.eval_n)
    else:
        dev_records = dev_full
    print(f"[data] train (Tie-filtered)         = {len(train_records)}")
    print(f"[data] dev   (eval_n={args.eval_n}) = {len(dev_records)}  (pool {len(dev_full)})")
    print(f"[data] test  (full)                 = {len(test_records)}")

    # Model + tokenizer
    model, tok = build_model_and_tokenizer(args.backbone, args.use_lora,
                                           args.lora_r, dtype,
                                           hf_token=os.environ.get("HF_TOKEN"))
    fmt_fn, has_chat = make_text_formatter(tok, system_prompt, user_template)
    model.to(device)

    # Preview one formatted training prompt for sanity
    print("\n[fmt] EXAMPLE formatted prompt:")
    print("------------------------------------------------------------")
    print(fmt_fn(train_records[0]["text_more"])[:600])
    print("------------------------------------------------------------\n")

    # Dataloader
    ds_train = PairDataset(train_records)
    coll = PairCollator(tok, args.max_len, fmt_fn)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          collate_fn=coll, num_workers=2)

    # Optim
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr,
                              weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(dl_train) / max(args.grad_accum, 1))
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )
    print(f"[train] steps/epoch={steps_per_epoch} total={total_steps} "
          f"effective_bs={args.batch_size * args.grad_accum}")

    eval_log = []
    best_acc, best_epoch = -1, -1
    epochs_since_best = 0
    global_step = 0
    train_t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        optim.zero_grad()
        ep_t0 = time.time()
        loss_sum, loss_n = 0.0, 0
        for it, (enc_more, enc_less) in enumerate(dl_train):
            r_more = reward(model, enc_more, device)
            r_less = reward(model, enc_less, device)
            loss = bt_loss(r_more, r_less)
            (loss / args.grad_accum).backward()
            if (it + 1) % args.grad_accum == 0 or (it + 1) == len(dl_train):
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1
            loss_sum += loss.item()
            loss_n += 1
            if it % args.log_every == 0:
                print(f"  ep{epoch} it{it}/{len(dl_train)} "
                      f"loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}")
        ep_time = time.time() - ep_t0
        mean_loss = loss_sum / max(loss_n, 1)

        dev_res = evaluate(model, tok, dev_records, fmt_fn, device,
                           batch_size=args.eval_batch_size,
                           max_len=args.max_len, name="dev")
        eval_log.append({"epoch": epoch, "train_loss": mean_loss,
                         "epoch_time_s": round(ep_time, 1), **dev_res})
        with open(out / "eval_results.json", "w") as f:
            json.dump(eval_log, f, indent=1)
        print(f"[epoch {epoch}] loss={mean_loss:.4f}  dev_acc={dev_res['pair_acc']:.4f}  "
              f"({ep_time:.1f}s)")

        if dev_res["pair_acc"] > best_acc + args.early_stop_min_delta:
            best_acc, best_epoch = dev_res["pair_acc"], epoch
            epochs_since_best = 0
            print(f"[save] new best dev_acc={best_acc:.4f} -> {ckpt_dir}")
            model.save_pretrained(str(ckpt_dir))
            tok.save_pretrained(str(ckpt_dir))
            # Also drop a small pointer in the results dir so it's easy to find.
            with open(out / "best_ckpt_path.txt", "w") as f:
                f.write(str(ckpt_dir) + "\n")
        else:
            epochs_since_best += 1
            print(f"[early-stop] no improvement (curr={dev_res['pair_acc']:.4f} "
                  f"vs best={best_acc:.4f}); strikes = {epochs_since_best}/{args.early_stop_patience}")
            if (args.early_stop_patience > 0
                    and epochs_since_best >= args.early_stop_patience):
                print(f"[early-stop] triggered at epoch {epoch} after "
                      f"{epochs_since_best} non-improving evals — stopping.")
                break

    # Final test eval (with best checkpoint loaded back in-memory model)
    print("\n[test] running final test evaluation with current (best-epoch) weights...")
    test_res = evaluate(model, tok, test_records, fmt_fn, device,
                        batch_size=args.eval_batch_size,
                        max_len=args.max_len, name="test")
    print(f"[test] pair_acc = {test_res['pair_acc']:.4f}")
    with open(out / "test_results.json", "w") as f:
        json.dump(test_res, f, indent=1)

    total_time = time.time() - train_t0
    print(f"\n[done] best_dev_acc={best_acc:.4f} (epoch {best_epoch})  "
          f"test_acc={test_res['pair_acc']:.4f}  "
          f"total_time={total_time:.1f}s")
    with open(out / "summary.txt", "w") as f:
        f.write(f"backbone     : {args.backbone}\n")
        f.write(f"prompt_mode  : {args.prompt_mode}\n")
        f.write(f"chat_template: {has_chat}\n")
        f.write(f"use_lora     : {args.use_lora}\n")
        f.write(f"ckpt_dir     : {ckpt_dir}\n")
        f.write(f"train_pairs  : {len(train_records)} (Tie filtered)\n")
        f.write(f"dev_pairs    : {len(dev_records)} (eval_n={args.eval_n})\n")
        f.write(f"test_pairs   : {len(test_records)}\n")
        f.write(f"best_dev_acc : {best_acc:.4f} (epoch {best_epoch})\n")
        f.write(f"test_acc     : {test_res['pair_acc']:.4f}\n")
        f.write(f"total_time   : {total_time:.1f}s\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    help="HF model id, e.g. roberta-base, Qwen/Qwen2.5-3B")
    ap.add_argument("--train_pairs", default="scripts/evaluator_validation/train_pairs.json")
    ap.add_argument("--dev_pairs", default="scripts/evaluator_validation/dev_pairs.json")
    ap.add_argument("--test_pairs", default="scripts/evaluator_validation/test_pairs.json")
    ap.add_argument("--out", required=True,
                    help="Lightweight results dir (logs, eval/test JSONs, summary). "
                         "Keep this on the project disk for easy copy.")
    ap.add_argument("--ckpt_dir", default=None,
                    help="Where to save the best model checkpoint (potentially "
                         "multi-GB). Default = <out>/best. Set this to a "
                         "separate (large) disk so result files stay copyable.")
    # Prompt
    ap.add_argument("--prompt_mode", choices=["1", "2", "3"], default="1",
                    help="1=raw text, 2=one-line instruction, 3=full Listing-2 guidelines.")
    ap.add_argument("--system_prompt", default=None,
                    help="Override preset's system prompt (default: use preset).")
    ap.add_argument("--user_template", default=None,
                    help="Override preset's user template; must contain `{text}`.")
    # Optim
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=512)
    # Precision
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    # LoRA (default OFF = full fine-tune)
    ap.add_argument("--use_lora", action="store_true",
                    help="If set, use LoRA; otherwise full fine-tune.")
    ap.add_argument("--lora_r", type=int, default=16)
    # Eval / early-stop
    ap.add_argument("--eval_n", type=int, default=100,
                    help="Sub-sample size for per-epoch dev eval (None = full dev).")
    ap.add_argument("--eval_seed", type=int, default=7)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--early_stop_patience", type=int, default=2,
                    help="Stop after N consecutive non-improving evals. 0 disables.")
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0,
                    help="Minimum dev acc improvement to count as 'better'.")
    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
