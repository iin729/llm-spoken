# LAS ablation (Table 4 in the paper)

Reproduces the eight-column LAS ablation table on Qwen-2.5-3B-Instruct, all
sharing LAS's training budget (LR 1e-3, BS 16, 10 epochs).

| Column | Localization | Parameterization | Description |
| --- | --- | --- | --- |
| **LAS (ours)** | oral L+H | element-wise affine $d \odot h + b$ | full method |
| Rnd | matched-count **random non-oral** L+H | same affine | sanity: localization |
| $-$Lay. | oral H only (no residual hook) | same affine | drop layer-side intervention |
| $-$Hd. | oral L only (no $o_{\text{proj}}$ hook) | same affine | drop head-side intervention |
| $-$Sc. | oral L+H | bias only ($d = 0$) | drop multiplicative half |
| $-$Bi. | oral L+H | scale only ($b = 0$) | drop additive half |
| LoRA | oral L+H (same scope) | rank-4 LoRA on FFN down_proj | adapter comparison |
| IA³ | oral L+H | scalar per channel | adapter comparison |
| ReFT | oral L+H | rank-4 ReFT (zero-init $W = R$) | rank-matched orthogonal alternative |

## Implementation

All variants share a single hook controller, `ablation_controller.py`, which
exposes a `method` and a `target_set` parameter:

```python
AblationController(
    method      = 'las'      | 'las_no_scale' | 'las_no_bias' |
                  'lora'     | 'ia3'          | 'reft',
    target_set  = 'oral_lh'  | 'random_lh'    |
                  'oral_l_only' | 'oral_h_only',
    ...
)
```

This means a *single* training script (`train_ablation.py`) drives every
row of the table.

## Run

```bash
# Full Table 4 grid on Qwen-3B (8 rows × seeds; ~10 h on 4 × A100):
bash scripts/run_ablation_qwen.sh

# Rank-matched ReFT row (r = 4, ~475 K params, zero-init W = R):
bash scripts/run_ablation_reft_r4.sh

# Aggregate everything into a Markdown table:
python aggregate_ablation_results.py --runs_root runs/  --out ablation_table.md
```

## Observed numbers (Qwen-2.5-3B-Instruct, n_test = 1055)

| Column | Params | L-Acc ↑ | BERT ↑ |
| --- | :---: | :---: | :---: |
| **LAS (ours)** | **119 K** | **0.584** | **0.911** |
| Rnd  | 143 K | 0.507 | 0.912 |
| $-$Lay. | 53 K | 0.397 | 0.900 |
| $-$Hd. | 66 K | 0.448 | 0.908 |
| $-$Sc. | 59 K | 0.433 | 0.906 |
| $-$Bi. | 59 K | 0.457 | 0.906 |
| LoRA  | 475 K | 0.568 | 0.909 |
| IA³  | 59 K  | 0.248 | 0.900 |
| ReFT (r=4) | 475 K | 0.248 | 0.900 |

Highlights:
- **Random L+H** at 143 K already costs 7.7 L-Acc — localization helps.
- **LoRA at 4× the parameters** still loses to LAS (0.568 vs 0.584).
- **ReFT collapses to 0.248** regardless of rank because its orthogonal
  rank-r subspace is too rigid for ordinal level control. The paper explains
  this in §5.4 ("ReFT, rank-matched").
