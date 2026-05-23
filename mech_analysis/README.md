# `mech_analysis/` — where in the LLM does spoken style get rendered?

This module implements the **mechanistic localization** of orality described
in §3 of the paper. Two stages:

1. **Layer-level patching** (`§3.1 / Fig 3 in the paper`) — causal activation
   patching from a Level-5-spoken prompt into a Level-0-written prompt
   identifies the *band of late layers* where orality is rendered. The
   resulting `L_oral` set is hard-coded in
   [`cas/cas_injector.py`](../cas/cas_injector.py) and
   [`las/scripts/run_train_las.sh`](../las/scripts/run_train_las.sh) (e.g.
   layers 20–35 on Qwen-2.5-3B).

2. **Head-level Contrastive Influence Score** (§3.2 / Fig 4 — implemented in
   `head_attribution/`). For each (layer, head), we zero out the head's
   output during generation and measure the probability gap on oral-marker
   tokens vs. non-oral tokens. The 2-sigma outliers form `A_oral`, which we
   cache as JSON in `heads_idx/{qwen,llama,mistral}.json`.

```
mech_analysis/
├── README.md                 # ← this file
├── heads_idx/                # cached A_oral sets used by CAS / LAS
│   ├── qwen.json             #   34 positive heads + 1 negative on Qwen-2.5-3B
│   ├── llama.json            #   25 positive + 7 negative on Llama-3.2-1B
│   └── mistral.json          #   38 positive + 11 negative on Mistral-7B
└── head_attribution/
    ├── README.md
    ├── identify_oral_heads.py    # main per-sample attribution loop
    ├── zero_ablation.py          # custom greedy generator: zero-ablate one head,
    │                             # measure prob diff per token; replaces
    │                             # transformers' greedy_search via monkeypatch
    ├── compute_cis_*.py          # per-vocabulary CIS variants
    │                             # (all_words / fillers / interaction /
    │                             #  stance / repeats / simple)
    ├── aggregate_cis_to_heads.py # CIS scores → A_oral (2-sigma threshold)
    ├── aggregate_influence.py    # tensor utilities shared by the above
    ├── head_reweight.py          # validation: at inference, scale a head's
    │                             # output and check whether outputs become more
    │                             # spoken (sanity check that A_oral is causal)
    ├── oral_vocab.py             # oral-marker vocab used by CIS
    └── scripts/                  # one-shot multi-GPU bash drivers
```

## Recompute `A_oral` from scratch

```bash
# Per-sample zero-ablation pass (multi-GPU; ~1500 samples → ~2h on 4×A100):
bash mech_analysis/head_attribution/scripts/run_identify_qwen3b.sh

# Aggregate per-sample .pth tensors into a single CIS matrix and pick the
# 2-sigma heads:
python -m mech_analysis.head_attribution.aggregate_cis_to_heads \
    --in_dir  mech_analysis/head_attribution/results/qwen3b \
    --out     mech_analysis/heads_idx/qwen.json \
    --modelsize 3b --sigma 2.0
```

Running the same with `--modelsize 1b` / `7b` regenerates `llama.json` /
`mistral.json`. The bundled JSONs already match the paper's numbers, so this
is only needed if you change the oral vocabulary, the influence score type,
or the model.

## What CAS and LAS read from here

- **CAS** reads only the *layer set* `L_oral` (passed as `--selected_layers`).
- **LAS** reads both `L_oral` (passed as `--direft_layers`) and `A_oral`
  (passed as `--head_path heads_idx/<model>.json`).
