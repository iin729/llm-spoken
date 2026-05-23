"""Score predict/label pairs from a CAS/LAS prediction JSONL with a trained
Bradley-Terry reward model, and compute the paper's $\\Delta R_r$ metric.

Definition (matches Table 1/2 caption in the v9.48 LaTeX):
    For each test sample i we compute
        r_pred_i  = r(predict_i)
        r_label_i = r(label_i)     # gt spoken text
    Then we report:
        deltaR_MAE      = mean(|r_pred_i - r_label_i|)         <-- ΔR in table
        pearson_r       = Pearson(r_pred, r_label)             <-- subscript r
        spearman_r      = Spearman(r_pred, r_label)
        deltaR_MSE      = mean((r_pred_i - r_label_i)^2)
        deltaR_Bias     = mean(r_pred_i - r_label_i)
        mean_r_pred / mean_r_label / std_r_pred / std_r_label  (for sanity)

The script mirrors the EXACT prompt-formatting used at training time
(`train_reward_model.py`): chat-template wrap (system=user_template={text} for
prompt_mode=1, etc.) so the head sees the same input distribution.

Usage
-----
    python score_with_reward_model.py \\
        --rm_ckpt   ${REPO_ROOT}/evaluator_validation/models/mistral7b_bt_pm1 \\
        --pred_jsonl scripts/cas/main/results/Qwen2.5-3B-Instruct_experiment/final_predictions.jsonl \\
        --out_json   ./delta_r/cas_qwen3b.json \\
        --label      "CAS / Qwen-3B"

Notes
-----
* `--predict_key` / `--label_key` default to `predict` / `label`, matching both
  CAS (`final_predictions.jsonl`) and LAS/SFT/LoRA/IA3
  (`generated_predictions.jsonl`).
* `--prompt_mode` is auto-loaded from the RM checkpoint's `args.json` if
  present; passing it on the CLI overrides that.
* The model is loaded in bf16 by default; pass `--no_bf16` to use float32.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# torch._dynamo shim (same as train_reward_model.py — keeps the same broken
# torch 2.1.2 + onnx combo loadable).
# ----------------------------------------------------------------------------
import torch
import sys as _sys

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

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Prompt presets (must match train_reward_model.py exactly)
# ----------------------------------------------------------------------------
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

PROMPT_PRESETS = {
    "1": {"name": "raw",
          "system": "",
          "user_template": "{text}"},
    "2": {"name": "instruction",
          "system": "",
          "user_template": ("Please rate how spoken/conversational the "
                            "following text sounds.\n\nText: {text}")},
    "3": {"name": "guidelines",
          "system": "",
          "user_template": LISTING2_GUIDELINES},
}


def make_text_formatter(tokenizer, system_prompt: str, user_template: str):
    has_chat = (
        getattr(tokenizer, "chat_template", None) is not None
        and callable(getattr(tokenizer, "apply_chat_template", None))
    )
    if not has_chat:
        return (lambda text: text), False

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
# IO
# ----------------------------------------------------------------------------
def load_pred_jsonl(path: Path, predict_key: str, label_key: str):
    texts_pred, texts_label, raw_rows = [], [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = row.get(predict_key)
            l = row.get(label_key)
            if not isinstance(p, str) or not isinstance(l, str):
                continue
            if not p.strip() or not l.strip():
                continue
            texts_pred.append(p)
            texts_label.append(l)
            raw_rows.append(row)
    return texts_pred, texts_label, raw_rows


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------
def rankdata(values):
    n = len(values)
    if n == 0:
        return []
    sorted_pairs = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[sorted_pairs[k][1]] = avg
        i = j
    return ranks


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    d = math.sqrt(vx * vy)
    return float("nan") if d == 0 else cov / d


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    return pearson(rankdata(xs), rankdata(ys))


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
@torch.no_grad()
def score_texts(texts, model, tokenizer, fmt_fn, device, batch_size, max_len):
    scores = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size),
                  desc="scoring", unit="batch", dynamic_ncols=True):
        batch = [fmt_fn(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        # AutoModelForSequenceClassification(num_labels=1) -> [B, 1]
        s = out.logits.squeeze(-1).float().cpu().tolist()
        if isinstance(s, float):
            s = [s]
        scores.extend(s)
    return scores


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm_ckpt", required=True,
                    help="Path to trained BT reward-model directory.")
    ap.add_argument("--pred_jsonl", required=True,
                    help="CAS/LAS predictions jsonl with `predict` & `label`.")
    ap.add_argument("--out_json", required=True,
                    help="Where to write the per-file summary JSON.")
    ap.add_argument("--label", default=None,
                    help="Display label for this run (Method / Model).")

    ap.add_argument("--predict_key", default="predict")
    ap.add_argument("--label_key", default="label")

    ap.add_argument("--prompt_mode", default=None,
                    help="1/2/3. If omitted, read from rm_ckpt/args.json or "
                         "the parent results dir.")
    ap.add_argument("--system_prompt", default=None)
    ap.add_argument("--user_template", default=None)

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.add_argument("--device", default=None,
                    help="cuda / cuda:0 / cpu (auto if omitted).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score at most N samples (debug).")
    args = ap.parse_args()

    rm_ckpt = Path(args.rm_ckpt).expanduser().resolve()
    pred_path = Path(args.pred_jsonl).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rm_ckpt.is_dir():
        print(f"[fatal] rm_ckpt is not a directory: {rm_ckpt}", file=sys.stderr)
        sys.exit(2)
    if not pred_path.is_file():
        print(f"[fatal] pred_jsonl not found: {pred_path}", file=sys.stderr)
        sys.exit(2)

    # ---- prompt mode resolution ---------------------------------------------
    prompt_mode = args.prompt_mode
    if prompt_mode is None:
        # try ckpt_dir/args.json
        for candidate in (rm_ckpt / "args.json",
                          rm_ckpt.parent / "args.json"):
            if candidate.is_file():
                try:
                    rec = json.loads(candidate.read_text())
                    if "prompt_mode" in rec:
                        prompt_mode = str(rec["prompt_mode"])
                        print(f"[prompt_mode] inferred from {candidate}: "
                              f"{prompt_mode}")
                        break
                except Exception:
                    pass
    if prompt_mode is None:
        prompt_mode = "1"
        print("[prompt_mode] not found in checkpoint args.json — defaulting "
              "to 1 (raw).")
    if prompt_mode not in PROMPT_PRESETS:
        raise ValueError(f"--prompt_mode must be 1/2/3, got {prompt_mode}")

    preset = PROMPT_PRESETS[prompt_mode]
    system_prompt = (args.system_prompt
                     if args.system_prompt is not None else preset["system"])
    user_template = (args.user_template
                     if args.user_template is not None else
                     preset["user_template"])
    print(f"[prompt] mode={prompt_mode} ({preset['name']})  "
          f"system={system_prompt[:40]!r}  user_template={user_template[:60]!r}")

    # ---- model load ---------------------------------------------------------
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  bf16={args.bf16}")

    dtype = torch.bfloat16 if (args.bf16 and "cuda" in device) else torch.float32
    print(f"[model] loading reward model from {rm_ckpt}")
    tok = AutoTokenizer.from_pretrained(str(rm_ckpt))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        str(rm_ckpt), num_labels=1, torch_dtype=dtype,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.to(device)

    fmt_fn, has_chat = make_text_formatter(tok, system_prompt, user_template)
    print(f"[fmt] has_chat_template={has_chat}")

    # ---- data ---------------------------------------------------------------
    print(f"[data] loading {pred_path}")
    preds, labels, _rows = load_pred_jsonl(
        pred_path, args.predict_key, args.label_key)
    if args.limit is not None:
        preds = preds[:args.limit]
        labels = labels[:args.limit]
    n = len(preds)
    print(f"[data] {n} valid (predict, label) pairs")
    if n == 0:
        print("[warn] nothing to score; writing empty summary.")
        out_path.write_text(json.dumps({"n": 0, "error": "no_valid_pairs"},
                                       indent=2))
        return

    # ---- score --------------------------------------------------------------
    r_pred = score_texts(preds, model, tok, fmt_fn, device,
                         args.batch_size, args.max_len)
    r_label = score_texts(labels, model, tok, fmt_fn, device,
                          args.batch_size, args.max_len)

    diffs = [p - l for p, l in zip(r_pred, r_label)]
    abs_diffs = [abs(d) for d in diffs]
    sq_diffs = [d * d for d in diffs]

    mean_pred = sum(r_pred) / n
    mean_label = sum(r_label) / n

    def _std(xs, m):
        return math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))

    summary = {
        "label": args.label or pred_path.stem,
        "pred_jsonl": str(pred_path),
        "rm_ckpt": str(rm_ckpt),
        "prompt_mode": prompt_mode,
        "has_chat_template": has_chat,
        "n": n,
        "mean_r_pred": mean_pred,
        "mean_r_label": mean_label,
        "std_r_pred": _std(r_pred, mean_pred),
        "std_r_label": _std(r_label, mean_label),
        "deltaR_MAE": sum(abs_diffs) / n,
        "deltaR_MSE": sum(sq_diffs) / n,
        "deltaR_RMSE": math.sqrt(sum(sq_diffs) / n),
        "deltaR_Bias": sum(diffs) / n,
        "pearson_r": pearson(r_pred, r_label),
        "spearman_r": spearman(r_pred, r_label),
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out_path}")
    print(json.dumps(summary, indent=2))

    # paper-ready one-liner
    print("\n=== paper row ===")
    print(f"  {summary['label']:<32s}  "
          f"ΔR={summary['deltaR_MAE']:.3f}  "
          f"r={summary['pearson_r']:.3f}  "
          f"(n={n})")


if __name__ == "__main__":
    main()
