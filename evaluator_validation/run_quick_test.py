"""One-click quick test: 10 pairs × N models × {think=False, think=True}.

Purpose: decide which cheap model (and whether to enable `think`) we want to use
to label the full 1k Phase-2 dataset.

What it does end-to-end:
  1. (Re)builds a 200-pair pool from the WSPairs training set if pairs_pool.json
     does not already exist (or --rebuild_pool is passed).
  2. Samples a fixed-seed 10-pair subset from that pool.
  3. For each (model, think_flag) combination in the user-configurable lists,
     queries the API on those 10 pairs.
  4. Prints a comparison matrix (rows=models, cols=think on/off; cell=acc/lat).
  5. Saves full per-call detail to quick_test_results.json.

Run (from project root):

  python scripts/evaluator_validation/run_quick_test.py
  # or with custom model list:
  python scripts/evaluator_validation/run_quick_test.py \
      --models qwen3.5-flash gpt-4o-mini deepseek-reasoner \
      --think_modes False True \
      --n 10 --seed 0

Notes:
  - `think` is passed straight through to `openai_chat(...)`. If a particular
    model does not support thinking, the run still completes; failures show up
    as HARD FAIL with the error message in the summary.
  - All API calls are sequential (no parallelism) so cost/RPM stays predictable.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from build_pairs import load_train, build_pairs, summarize  # noqa: E402
from eval_llm_judges import eval_one_model  # noqa: E402


DEFAULT_MODELS = [
    "qwen3.5-flash",
    "gpt-4o-mini",
    "gpt-5.4-nano",
    "gemini-3.1-flash-lite",
    "deepseek-v4-flash",  # replaces deepseek-reasoner (got 404 get_instance_failed)
]
# `think=True` by default — OpenAI models silently fall back to no-think
# inside _call_with_think_fallback().
DEFAULT_THINK_MODES = [True]


def ensure_pool(pool_path: Path, train_json: Path, n_pool: int, rebuild: bool, seed: int):
    if pool_path.exists() and not rebuild:
        with open(pool_path) as f:
            pool = json.load(f)
        print(f"[pool] Loaded existing pool ({len(pool)} pairs) from {pool_path}")
        return pool
    print(f"[pool] Building fresh pool of {n_pool} pairs (seed={seed})")
    items = load_train(str(train_json))
    pool = build_pairs(items, n_pairs=n_pool, within_frac=0.25,
                       min_margin=1, seed=seed)
    summarize(pool)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pool_path, "w") as f:
        json.dump(pool, f, indent=1)
    print(f"[pool] Saved -> {pool_path}")
    return pool


def parse_bool(s):
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("1", "true", "t", "yes", "y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="scripts/add_stat/train_level_sft.json")
    ap.add_argument("--pool_path", default="scripts/evaluator_validation/pairs_pool.json")
    ap.add_argument("--n_pool", type=int, default=200,
                    help="Pool size for quick test (cheap; we resample from this).")
    ap.add_argument("--rebuild_pool", action="store_true",
                    help="Force rebuild of pairs_pool.json.")
    ap.add_argument("--n", type=int, default=10, help="Pairs sampled per (model, think).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--think_modes", nargs="+", default=["True"],
                    help="Subset of {False, True}. Default: True only "
                         "(OpenAI models auto-fall-back to no-think).")
    ap.add_argument("--sleep_s", type=float, default=0.0,
                    help="Optional sleep between API calls.")
    ap.add_argument("--out_path", default="scripts/evaluator_validation/quick_test_results.json")
    ap.add_argument("--min_margin", type=int, default=1)
    args = ap.parse_args()

    think_modes = [parse_bool(x) for x in args.think_modes]
    think_modes = sorted(set(think_modes))  # dedupe

    print("=" * 70)
    print(" Quick Test: model × think sweep on a small pair sample")
    print("=" * 70)
    print(f"  models       : {args.models}")
    print(f"  think_modes  : {think_modes}")
    print(f"  n            : {args.n}  (per cell)")
    print(f"  seed         : {args.seed}")
    print()

    pool = ensure_pool(Path(args.pool_path), Path(args.train_json),
                       n_pool=args.n_pool, rebuild=args.rebuild_pool, seed=42)

    import random
    rng = random.Random(args.seed)
    elig = [p for p in pool if p["margin"] >= args.min_margin]
    sample = rng.sample(elig, min(args.n, len(elig)))
    print(f"[sample] Picked {len(sample)} pairs (min_margin={args.min_margin})")
    print(f"[sample] GT distribution: "
          f"{ {k: sum(1 for p in sample if p['gt_label'] == k) for k in ('A','B','Tie')} }")
    print()

    all_runs = []
    grand_t0 = time.time()
    for m in args.models:
        for think in think_modes:
            cell = f"{m} | think={think}"
            print(f"\n>>> RUN: {cell}")
            try:
                res = eval_one_model(m, sample, sleep_s=args.sleep_s, think=think)
            except Exception as e:
                print(f"  HARD FAIL for {cell}: {e!r}")
                res = {"model": m, "think": think, "hard_fail": repr(e)}
            all_runs.append(res)

    elapsed = time.time() - grand_t0
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "think_modes": think_modes,
        "sampled_pairs": sample,
        "runs": all_runs,
        "wall_time_s": round(elapsed, 1),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)

    # -------- Pretty summary matrix --------
    print("\n" + "=" * 70)
    print(" SUMMARY MATRIX (rows=models, cols=think state)")
    print("=" * 70)
    header = f"{'model':<28}"
    for t in think_modes:
        header += f"  {'think=' + str(t):>26}"
    print(header)
    print("-" * len(header))
    for m in args.models:
        line = f"{m:<28}"
        for t in think_modes:
            entry = next((r for r in all_runs
                          if r.get("model") == m and r.get("think") == t), None)
            if entry is None or "hard_fail" in entry:
                cell = "HARD FAIL"
            else:
                acc = entry.get("pair_acc")
                lat = entry.get("mean_latency_s")
                parsed = entry.get("n_parsed", 0)
                npairs = entry.get("n_pairs", args.n)
                pf = entry.get("parse_failures", 0)
                acc_s = "  nan" if acc != acc else f"{acc:5.3f}"
                cell = f"acc={acc_s} parsed={parsed}/{npairs} pf={pf} lat={lat}s"
            line += f"  {cell:>26}"
        print(line)
    print(f"\nTotal wall time: {elapsed:.1f}s")
    print(f"Full details -> {out_path}")
    print()
    print("Pick the best (model, think) combo for Phase-2 1k labeling.")


if __name__ == "__main__":
    main()
