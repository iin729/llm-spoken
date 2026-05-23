# `evaluator_validation/` — validate g(·) and train the ΔR<sub>r</sub> reward model

This module backs Appendix B of the paper. Two purposes:

1. **Benchmark g(·) against frontier LLM judges and reward models** on
   200 human-anchor pairs (Listing app:annotation_compare). g(·) wins with
   **0.910 pair-Acc** vs 0.881 (LLM ensemble) and 0.876 (best reward model);
   see §3.4 and Appendix B.
2. **Train an independent Bradley-Terry reward model** $r_\theta$
   (Llama-3.2-7B / Mistral-7B / Qwen-3B etc.) on 10 K LLM-judged pairs, used
   as the orthogonal evaluator producing the ΔR<sub>r</sub> column in
   Tables 1 and 2.

Both axes share two design principles:
- **Same prompt** as the human annotation protocol (Listing app:annotation_compare),
  used verbatim.
- **Tie window** of 0 by default — we score raw forced-choice agreement.

```
evaluator_validation/
├── README.md
├── prompts.py                       # the (single) compare-annotation prompt
├── build_pairs.py                   # build pair pool from a single split
├── build_pairs_from_splits.py       # build pair pool across train/dev/test
├── build_llm_dataset.py             # call an LLM to label 10K pairs
├── eval_llm_judges.py               # Phase-1: evaluate frontier LLM judges
├── run_phase1_concurrent.py         # multi-LLM concurrency driver
├── run_quick_test.py                # 10-pair smoke test
├── train_reward_model.py            # Bradley-Terry RM training (HF Trainer)
├── score_with_reward_model.py       # score a predictions JSONL with a trained RM
├── download_roberta.py              # one-off: fetch roberta-base for the RM tokenizer
└── scripts/
    ├── run_reward_models.sh         # train multiple RMs
    ├── run_reward_qwen3_4b.sh
    ├── run_reward_mistral7b.sh
    ├── score_main_table_delta_r.sh  # ΔRr for Table 2 rows
    └── score_all_delta_r.sh         # ΔRr for everything (Tables 1 + 2 + ablations)
```

## Run

### A) Benchmark g(·) vs LLM judges

```bash
# Smoke test (n=10, one model, cheap):
python run_quick_test.py

# Phase 1: evaluate a list of frontier LLM judges concurrently (default ~$5):
python run_phase1_concurrent.py \
    --pairs pairs_pool.json --n 200 --seed 0 --out_dir phase1/
```

`pairs_pool.json` is included; it has the 200 random pairs anchored to
human pairwise judgments used in the paper.

### B) Build the 10 K LLM-judged dataset

```bash
python build_pairs_from_splits.py \
    --train_file ../data/train.json --dev_file ../data/dev.json --test_file ../data/test.json \
    --n_train 8000 --n_dev 1000 --n_test 1000 \
    --out_dir reward_data/

python build_llm_dataset.py \
    --pairs_in reward_data/train_pairs.json \
    --model gemini-3-flash --out reward_data/train_pairs_labeled.json
```

### C) Train a Bradley-Terry reward model

```bash
bash scripts/run_reward_models.sh                    # iterate over backbones
# OR a single backbone:
MODEL_PATH=Qwen/Qwen3-4B bash scripts/run_reward_qwen3_4b.sh
```

Trained checkpoints land in
`evaluator_validation/models/<model>_bt_<dataset>/`. Two of them go on to
score Table 1 / Table 2 (the Mistral-7B and Qwen-3 4B reward models).

### D) Score paper-table predictions with a trained RM

```bash
bash scripts/score_main_table_delta_r.sh    # ΔRr column of Table 2
bash scripts/score_all_delta_r.sh           # everything (T1 + T2 + ablations)
```

Output: each predictions JSONL is augmented with a per-sample reward score;
the table aggregator computes the MAE between the prediction reward and the
gold-spoken reward (ΔR) and the Pearson r across samples.
