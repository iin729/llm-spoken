# Head Attribution via Zero Ablation + Contrastive Influence Score (CIS)

This sub-module is the §3.2 implementation. We isolate **the heads that
mediate orality** without touching the rest of the network.

## Algorithm

For each attention head $a = (\ell, h)$ in a Transformer with $L$ layers
and $H$ heads per layer:

1. Run the model forward on the input prompt.
2. At every decoding step, **zero out the contribution** of head $a$ to that
   layer's $o_{\text{proj}}$ output, then re-forward and record the
   per-token probability drop $\mathcal{I}_a(y_t)$.
3. Aggregate over a small oral-marker vocabulary $\mathcal{V}_{\text{oral}}$
   (fillers, hedges, interaction markers, …) and a complement
   $\mathcal{V}_{\text{non-oral}}$ derived from the written prompt:

   $$
   \mathrm{CIS}_a = \mathbb{E}_{y \in \mathcal{V}_{\text{oral}}}[\mathcal{I}_a(y)]
                  - \mathbb{E}_{y \in \mathcal{V}_{\text{non-oral}}}[\mathcal{I}_a(y)]
   $$

4. Pick the 2-σ outliers:
   $\mathcal{A}_{\text{oral}} = \{a : |\mathrm{CIS}_a - \mu| > 2\sigma\}$.

## File map

| File | What it does |
| --- | --- |
| `zero_ablation.py` | Replaces `transformers`' `greedy_search` with our variant that zero-ablates one head at a time and records the per-token probability difference. Patches `_validate_model_kwargs` to accept extra kwargs. |
| `identify_oral_heads.py` | Per-sample attribution. For each example, builds the oral and non-oral vocabularies, runs the patched generator, and dumps the per-head influence matrix as a `.pth` file. Also draws a quick per-sample heatmap. |
| `oral_vocab.py` | The oral-marker vocabulary used during CIS. |
| `compute_cis_*.py` | CIS variants per feature category (`fillers`, `interaction`, `stance`, `repeats`, `simple`, `all_words`). All use a 3-sigma z-score threshold by default. |
| `aggregate_influence.py` | Shared tensor utilities for loading and stacking the per-sample `.pth` files. |
| `aggregate_cis_to_heads.py` | Aggregates per-sample CIS into a global matrix and writes `heads_idx/<model>.json`. |
| `head_reweight.py` | Validation utility: scale a head's output up/down at inference and check whether outputs become more spoken. Confirms `A_oral` is causal, not merely correlational. |

## Run

```bash
# 1. Per-sample attribution (multi-GPU sharded). One run per backbone:
bash scripts/run_identify_qwen3b.sh        # → results/qwen3b/*.pth
bash scripts/run_identify_llama1b.sh
bash scripts/run_identify_mistral7b.sh

# 2. Aggregate the per-sample matrices into a CIS heatmap and pick A_oral:
python aggregate_cis_to_heads.py \
    --in_dir   results/qwen3b \
    --out      ../heads_idx/qwen.json \
    --modelsize 3b --sigma 2.0

# 3. (Optional) Sanity-check that A_oral really mediates orality:
bash scripts/run_reweight.sh
```

## Results we observed (Qwen-2.5-3B-Instruct, n = 1500)

- 35 / 448 heads exceed |CIS - μ| > 2σ
- 34 positive (mediate oral markers) + 1 negative (suppress them)
- Almost all in layers 20–30, within $\mathcal{L}_{\text{oral}}$ from §3.1
- Top-CIS head is **L20-H8**, which promotes discourse markers
  (*well, so, now*) and stance (*I guess*); see Fig 4 of the paper.

Same sparse-late pattern replicates on Llama-3.2-1B (layers 5–15) and
Mistral-7B-Instruct-v0.3 (layers 15–31).
