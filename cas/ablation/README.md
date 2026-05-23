# CAS ablation (Table 3 in the paper)

Reproduces the five-column CAS ablation table.

| Column | Vector | Layers | Calibration | Description |
| --- | --- | --- | --- | --- |
| **CAS (ours)** | real $\hat v_\ell$ | $\mathcal{L}_{\text{oral}}$ | quadratic $\alpha=f(t)$ | full method |
| $-$Calib | real $\hat v_\ell$ | $\mathcal{L}_{\text{oral}}$ | best fixed scalar | drop level-conditioned α |
| $-$Multi | real $\hat v_\ell$ | single best layer | quadratic | drop multi-layer injection |
| Iso | real $\hat v_\ell$ | $\mathcal{L}_{\text{oral}}$ | **isotonic regression** | replace quadratic family |
| Rnd L | real $\hat v_\ell$ | matched-count **random non-oral layers** | quadratic | sanity: localization |
| Rnd v | **Gaussian random** $\hat v_\ell$ | $\mathcal{L}_{\text{oral}}$ | quadratic | sanity: direction meaningful |

The doubly-ablated corner ($-$Multi, $-$Calib) reproduces ActAdd
(Turner et al. 2023).

## Run

```bash
# All non-Iso rows (one driver per row to make GPU sharding easy):
bash scripts/run_full_ablation.sh

# Isotonic-regression row:
bash scripts/run_iso_ablation.sh

# Score all outputs into a single LaTeX-ready table:
bash scripts/score_all.sh             # full ablation
bash scripts/score_iso_ablation.sh    # iso row
bash scripts/score_cas_ablation_table.sh  # combined Table 3 with ΔRr column
```

Each row produces `experiment_results.xlsx + coeff_function_params.json +
final_predictions.jsonl + eval_all_metrics.json` in its own subdirectory under
`cas/results/`. `summarize_results.py` joins them into the table.

## Reproducing the paper numbers

We observed the following on Qwen-2.5-3B-Instruct, n_calibration = 1000,
n_test = 1055 (the splits bundled in `data/`):

| Row | $\bar\alpha$ | L-Acc | BERTScore |
| --- | :---: | :---: | :---: |
| **CAS (ours)** | 2.56 | **0.390** | **0.894** |
| $-$Calib | 3.50 | 0.267 | 0.883 |
| $-$Multi | 2.49 | 0.141 | 0.893 |
| Iso | 2.56 | 0.293 | 0.888 |
| Rnd L | 1.65 | 0.204 | 0.894 |
| Rnd v | 3.31 | 0.163 | 0.893 |

Matches Table 3 of the paper to two decimal places.
