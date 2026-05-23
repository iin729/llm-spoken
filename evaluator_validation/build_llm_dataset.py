"""Phase 2: Label 1k pairs with the best LLM judge(s) to create the reward-model
training dataset.

Reads pairs from pairs_pool.json (built by build_pairs.py). For each pair, queries
each --models entry; if more than one model is provided, it stores all labels and
also computes a majority-vote label.

We DO NOT use g(·) levels as labels here — they remain in the file only for
diagnostics (e.g., how often the LLM agrees with g(·)). The training signal for
the reward model is the LLM label only, keeping it independent of g(·).

Output schema (one JSON file = list of records):
[
  {
    "text_a": str, "text_b": str,
    "level_a": int, "level_b": int,           # diagnostics only
    "gt_label": "A"|"B"|"Tie",                # from g(·), diagnostics only
    "margin": int, "pair_type": "within"|"cross",
    "llm_labels": { "<model>": "A"|"B"|"Tie"|None, ... },
    "llm_raw":    { "<model>": "<raw response>", ... },
    "llm_majority": "A"|"B"|"Tie"|None,       # majority vote across models
    "agrees_with_g": bool                     # llm_majority == gt_label
  },
  ...
]

Usage:
  python scripts/eval/evaluator_validation/build_llm_dataset.py \
      --pairs scripts/eval/evaluator_validation/pairs_pool.json \
      --n 1000 --seed 1 \
      --models gpt-4o-mini deepseek-reasoner \
      --out scripts/eval/evaluator_validation/labeled_1k.json
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from clients.openai_api import openai_chat  # noqa: E402  (kept for symmetry; actual calls go through fallback)
from prompts import build_compare_prompt  # noqa: E402
from eval_llm_judges import parse_response, normalize_label, _call_with_think_fallback  # noqa: E402


def majority(labels):
    """Pick majority vote from a list of {A, B, Tie, None} ignoring Nones.
    Returns None if all are None; returns 'Tie' on a tie among A/B."""
    real = [x for x in labels if x is not None]
    if not real:
        return None
    c = Counter(real)
    best, top = c.most_common(1)[0]
    # If exactly tied between A and B (and no clear majority), return Tie
    if top == c.get("A", 0) and top == c.get("B", 0):
        return "Tie"
    return best


def query_one(model_name, prompt, think: bool = False):
    raw, err = _call_with_think_fallback(prompt, model_name, think)
    obj = parse_response(raw)
    pred = normalize_label(obj.get("more_spoken")) if isinstance(obj, dict) else None
    return raw, pred, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="scripts/eval/evaluator_validation/pairs_pool.json")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--models", nargs="+", required=True,
                    help="LLM model name(s) to query. If >1, also stores majority.")
    ap.add_argument("--out", default="scripts/eval/evaluator_validation/labeled_1k.json")
    ap.add_argument("--min_margin", type=int, default=1,
                    help="Filter pool to pairs with abs(level diff) >= this.")
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true",
                    help="If --out exists, skip pairs already present.")
    ap.add_argument("--think", action="store_true",
                    help="Enable chain-of-thought / thinking mode if the model supports it.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.pairs) as f:
        pool = json.load(f)
    elig = [p for p in pool if p["margin"] >= args.min_margin]
    sample = rng.sample(elig, min(args.n, len(elig)))
    print(f"Sampled {len(sample)} pairs from {len(elig)} eligible (out of {len(pool)} pool).")
    print(f"Models: {args.models}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        # build a key set on (text_a, text_b)
        seen = {(r["text_a"], r["text_b"]) for r in existing}
        sample = [p for p in sample if (p["text_a"], p["text_b"]) not in seen]
        print(f"Resuming: {len(existing)} already labeled, {len(sample)} remaining.")

    out = list(existing)
    t0 = time.time()
    for i, p in enumerate(sample, 1):
        prompt = build_compare_prompt(p["text_a"], p["text_b"])
        labels, raws, errs = {}, {}, {}
        for m in args.models:
            raw, pred, err = query_one(m, prompt, think=args.think)
            labels[m] = pred
            raws[m] = raw
            errs[m] = err
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
        maj = majority([labels[m] for m in args.models])
        rec = {
            **{k: v for k, v in p.items() if k in
               ("text_a", "text_b", "level_a", "level_b", "gt_label", "margin", "pair_type")},
            "llm_labels": labels,
            "llm_raw": raws,
            "llm_errors": errs,
            "llm_majority": maj,
            "agrees_with_g": (maj == p["gt_label"]) if maj is not None else None,
        }
        out.append(rec)
        # periodic save
        if i % 25 == 0 or i == len(sample):
            with open(out_path, "w") as f:
                json.dump(out, f, indent=1)
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1)
            eta = (len(sample) - i) / max(rate, 1e-6)
            print(f"  [{i}/{len(sample)}]  elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
                  f"saved -> {out_path}")

    # Final summary
    n = len(out)
    parsed = sum(1 for r in out if r["llm_majority"] is not None)
    agrees = sum(1 for r in out if r.get("agrees_with_g") is True)
    print("\n========== Phase-2 summary ==========")
    print(f"  Total labeled         : {n}")
    print(f"  With majority label   : {parsed}")
    print(f"  Agree with g(·) (LLM-majority == GT): {agrees}/{parsed} "
          f"= {agrees / max(parsed, 1):.3f}")
    for m in args.models:
        per_model_agree = sum(1 for r in out
                              if r["llm_labels"].get(m) == r["gt_label"])
        per_model_parsed = sum(1 for r in out if r["llm_labels"].get(m) is not None)
        print(f"  {m:<25} per-model agree-with-g: "
              f"{per_model_agree}/{per_model_parsed} = "
              f"{per_model_agree / max(per_model_parsed, 1):.3f}")
    print(f"\n  Output -> {out_path}")


if __name__ == "__main__":
    main()
