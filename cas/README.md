# `cas/` — Calibrated Activation Steering (training-free)

CAS is the training-free knob from §4.1 of the paper. Given a target Orality
Level $t \in \{0,1,\dots,5\}$, CAS injects a fixed direction into the
**localized oral layer set** $\mathcal{L}_{\text{oral}}$, with a coefficient
$\alpha^*(t)$ calibrated on a held-out set.

CAS extends Activation Addition (ActAdd, Turner et al. 2023) along two axes:
**multi-layer injection** (instead of one grid-searched layer), and
**level-conditioned $\alpha$** (instead of a fixed scalar). Ablating either
axis costs 12–25 L-Acc points; see [`ablation/`](ablation/) and Table 3 of
the paper.

```
cas/
├── README.md                         ← this file
├── extract_style_vectors.py          # Stage 1: per-layer mean Level-5 − Level-0 direction
├── cas_injector.py                   # Stages 2–4: calibrate / analyze / predict
├── scripts/
│   ├── run_extract_style_vectors.sh  # sweeps three backbones
│   └── run_cas_pipeline.sh           # one-shot: sweep α, fit α=f(t), generate, evaluate
└── ablation/
    ├── README.md                     # Table 3 ablations
    ├── inject_ablation.py            # full_cas / actadd_single / random_layers / random_vec
    ├── inject_ablation_iso.py        # Iso row: isotonic regression instead of quadratic
    ├── score_minimal.py              # L-Acc / Bias / BERTScore F1 in one pass
    ├── aggregate_results.py / summarize_results.py
    └── scripts/                      # one-shot drivers for each ablation row
```

## Algorithm

For each layer $\ell \in \mathcal{L}_{\text{oral}}$, build the style vector

$$
v_\ell = \mathbb{E}\big[h_\ell(p_5(x)) - h_\ell(p_0(x))\big],\quad
\hat v_\ell = v_\ell / \|v_\ell\|_2
$$

over parallel pairs $(x_w, x_s)$, where $p_t(x)$ is the chat-template prompt
asking for Level-$t$ output. At inference, CAS applies

$$
h'_\ell = h_\ell + \alpha^*(t)\,\hat v_\ell
$$

at every decoding position and every $\ell \in \mathcal{L}_{\text{oral}}$.
$\alpha^*$ is fit as a quadratic $\alpha = w_2 t^2 + w_1 t + w_0$ anchored at
$(t=0, \alpha=0)$, calibrated on the per-level best coefficient
$\arg\min_\alpha |\hat\ell(\alpha) - t|$.

The quadratic *extrapolates smoothly* past the calibration grid boundary,
where the per-level argmin saturates. Replacing it with isotonic regression
(which can't extrapolate) costs **9.7 L-Acc points** (Iso row, Table 3).

## Run

```bash
# Stage 1: extract per-layer style vectors for all backbones (~10 min on 1 GPU per model)
bash scripts/run_extract_style_vectors.sh

# Stages 2–4: calibrate α=f(t), generate with multi-GPU chunking, evaluate
bash scripts/run_cas_pipeline.sh
# → produces, for each backbone:
#     cas/results/<MODEL>_experiment/
#       ├── experiment_results.xlsx       # coeff sweep
#       ├── coeff_function_params.json    # fitted α=f(t) quadratic
#       ├── final_predictions.jsonl
#       └── eval_all_metrics.json         # BLEU / ROUGE / BERTScore / L-Acc / ...
```

Single-stage usage:

```bash
# only sweep α on the calibration set
python cas_injector.py --mode calibrate \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --vector_path  cas/results/Qwen2.5-3B-Instruct_L5_minus_L0/style_vectors_l5_minus_l0.pt \
    --data_path    data/calibration.json \
    --output_dir   cas/results/Qwen2.5-3B-Instruct_experiment \
    --selected_layers 20-35   # L_oral on Qwen-2.5-3B from mech_analysis

# only fit α=f(t) from an existing sweep
python cas_injector.py --mode analyze \
    --output_dir cas/results/Qwen2.5-3B-Instruct_experiment \
    --model_path noop  --vector_path noop  --data_path noop   # ignored in analyze

# only predict on the test set (multi-GPU via --num_chunks / --chunk_idx)
python cas_injector.py --mode predict \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --vector_path  cas/results/Qwen2.5-3B-Instruct_L5_minus_L0/style_vectors_l5_minus_l0.pt \
    --data_path    data/test.json \
    --output_dir   cas/results/Qwen2.5-3B-Instruct_experiment \
    --prediction_file cas/results/Qwen2.5-3B-Instruct_experiment/final_predictions.jsonl \
    --selected_layers 20-35 \
    --num_chunks 4 --chunk_idx 0   # rank 0 of 4
```

## Hyperparameters

| Backbone | $\mathcal{L}_{\text{oral}}$ | $\alpha$ sweep grid | Calibration set |
| --- | --- | --- | --- |
| Qwen-2.5-3B-Instruct | layers 20–35 | $\{0, 0.1, \dots, 4.0\}$ | `data/calibration.json` |
| Llama-3.2-1B-Instruct | layers 5–15 | same | same |
| Mistral-7B-Instruct-v0.3 | layers 15–31 | same | same |

## Optional features

- `--use_dynamic_gating` — replaces the constant $\alpha$ with
  $\alpha \cdot \mathrm{ReLU}(\cos(h_\ell, \hat v_\ell))$, gating the
  intervention to tokens that already lean spoken.
- `--force_coeff X` — skip the quadratic fit and use a fixed $\alpha = X$;
  useful for very small models where the calibration sweep is noisy.
