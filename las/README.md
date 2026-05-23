# `las/` — Learned Activation Steering

LAS is the parameter-efficient learned controller from §4.2 of the paper.
It intervenes on the **same components** identified by mechanistic analysis
(oral layers $\mathcal{L}_{\text{oral}}$ + oral heads $\mathcal{A}_{\text{oral}}$)
with a **per-component element-wise affine**:

$$
\mathbf{h}^{c\prime} = \mathbf{h}^{c} + \mathbf{d}_c \odot \mathbf{h}^{c} + \mathbf{b}_c
$$

with $\mathbf{d}_c, \mathbf{b}_c \in \mathbb{R}^{d}$ initialised to **zero**
(so step 0 is identity). The backbone is frozen.

The total trainable budget is just **78 K / 119 K / 295 K** parameters on
Llama-3.2-1B / Qwen-2.5-3B / Mistral-7B-Instruct-v0.3
(**0.004 – 0.006 %** of the backbone). LAS matches full SFT in-domain and
leads three OOD folds.

```
las/
├── README.md
├── train_las.py             # main training loop (HF Trainer + custom hooks)
├── eval_las.py              # inference + L-Acc / BERTScore / ΔRr eval
├── scripts/
│   └── run_train_las.sh     # multi-backbone driver (BACKBONE=qwen3b|llama1b|mistral7b)
└── ablation/                # Table 4 ablations
    ├── README.md
    ├── ablation_controller.py        # unified hook controller (LAS / -Sc / -Bi / LoRA / IA3 / ReFT)
    ├── train_ablation.py             # drives ablation_controller via HF Trainer
    ├── eval_ablation.py              # inference for any ablation variant
    ├── aggregate_ablation_results.py # gathers L-Acc / Params per row
    └── scripts/
        ├── run_ablation_qwen.sh      # full Table 4 grid on Qwen-3B
        └── run_ablation_reft_r4.sh   # rank-matched ReFT (r=4) row
```

## Intervention targets

LAS attaches hooks at exactly the components identified by `mech_analysis/`:

1. **Hidden states of oral layers** (`--direft_layers`) — forward hook on the
   decoder layer output.
2. **Per-head $o_{\text{proj}}$ outputs** of oral heads (`--head_path`) — a
   mask-and-affine hook on the input of `o_proj` so only the `d_head`-block
   of each oral head is touched. Under GQA (Qwen-3B 16Q/2KV;
   Mistral-7B 32Q/8KV; Llama-1B 32Q/8KV), this leaves K/V shared blocks
   untouched.

## Run

```bash
# Train LAS on all three backbones. To train just one: BACKBONE=qwen3b bash ...
bash scripts/run_train_las.sh
# Outputs:
#   las/runs/Qwen2.5-3B-Instruct_las/{params.pt, controller_config.json, ...}
#   las/runs/Llama-3.2-1B-Instruct_las/...
#   las/runs/Mistral-7B-Instruct-v0.3_las/...

# Evaluate (and score) a trained checkpoint
python eval_las.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --las_ckpt           las/runs/Qwen2.5-3B-Instruct_las \
    --eval_file          data/test.json \
    --output_dir         las/runs/Qwen2.5-3B-Instruct_las/eval

python -m orality.eval_predictions  \
    --pred_file las/runs/Qwen2.5-3B-Instruct_las/eval/predictions.jsonl
```

## Hyperparameters (Table app:training_details in the paper)

| Backbone | $\mathcal{L}_{\text{oral}}$ | $|\mathcal{A}_{\text{oral}}|$ | Trainable | LR | BS × GA | Epochs |
| --- | --- | :---: | :---: | --- | --- | :---: |
| Llama-3.2-1B-Instruct | layers 5–15  | 32 | **78 K (0.006 %)** | 1e-3 | 4 × 4 | 10 |
| Qwen-2.5-3B-Instruct  | layers 20–35 | 35 | **119 K (0.004 %)** | 5e-3 | 2 × 8 | 10 |
| Mistral-7B-Instruct-v0.3 | layers 15–31 | 49 | **295 K (0.004 %)** | 5e-3 | 2 × 8 | 10 |

All runs use bf16, cosine LR schedule with 10 % warmup, and identical
`group_by_length` batching to the SFT baseline.
