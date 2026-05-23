# `baselines/` — SFT, LoRA, IA³

These are the four supervised baselines in Table 2 of the paper. All three
load the WSPairs `data/train.json` and share an `output ← spoken` SFT-style
trainer.

| Backbone | Method | Trainable | Notes |
| --- | --- | :---: | --- |
| Any | SFT_ref | 100 % | full fine-tune, used only as a reference ceiling (italicised in Table 2) |
| Any | LoRA r=4 | ~0.03–0.06 % | matched to LAS's scope (`--lora_target_modules down_proj`) |
| Any | IA³ | ~0.006–0.013 % | matched to LAS's scope (`--ia3_target_modules down_proj`) |

Note: matched-scope LoRA / IA³ exists to **isolate the localization signal
from raw capacity**. Unconstrained LoRA / IA³ would effectively become SFT
(reported as `SFT_ref` in Table 2).

```
baselines/
├── sft/{train_full.py, scripts/run_sft.sh}
├── lora/{train_lora.py, scripts/{run_lora.sh, run_lora_ood.sh}}
└── ia3/{train_ia3.py, eval_simple.py, scripts/{run_ia3.sh, run_ia3_ood.sh}}
```

`eval_simple.py` is a tiny generic evaluator used by all three baselines
(it loads a PEFT adapter and generates predictions); it's shared via path,
not as a Python import.

## Run

```bash
# Full SFT (reference ceiling) on Qwen-2.5-3B:
bash sft/scripts/run_sft.sh

# Rank-4 LoRA on down_proj only:
bash lora/scripts/run_lora.sh    MODEL_PATH=Qwen/Qwen2.5-3B-Instruct

# IA³ on down_proj only (with auto-eval at the end):
bash ia3/scripts/run_ia3.sh      MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
```

For OOD (Fig. fig:eff_ood, §5.3), see `run_*_ood.sh` and the OOD note in
[`data/README.md`](../data/README.md).

## Evaluating any checkpoint

After training, use the orality estimator on the produced predictions:

```bash
python -m orality.eval_predictions \
    --pred_file runs/<model>_<method>/eval/predictions.jsonl
```

The output `eval_all_metrics.json` contains the L-Acc / Bias / BERTScore /
ROUGE / BLEU / O-score / Pearson / Spearman / Macro-F1 columns reported in
the paper.
