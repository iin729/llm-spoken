"""Construct pairwise comparison data from the WSPairs training set.

Pair sources:
  (a) WITHIN-pair: (written, spoken) of the same sample — by corpus construction
      spoken is more spoken than written.
  (b) CROSS-pair: two spoken texts drawn from different samples; gt is decided by
      their g(·)-assigned levels (kmeans cluster).

For each pair we store:
  text_a, text_b       : the two strings to be compared
  level_a, level_b     : g(·) level assigned by EWM+KMeans (0..5)
  gt_label             : "A" if level_a > level_b, "B" if level_b > level_a, "Tie" otherwise
  margin               : abs(level_a - level_b)
  source_a, source_b   : which sample they came from (for traceability)
  pair_type            : "within" or "cross"

Usage:
  python build_pairs.py --train_json scripts/add_stat/train_level_sft.json \
                        --out_json scripts/eval/evaluator_validation/pairs_pool.json \
                        --n_pairs 2000 --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def load_train(path: str):
    with open(path) as f:
        data = json.load(f)
    # Each entry has spoken, written, spoken_statics.level, written_statics.level
    items = []
    for x in data:
        items.append({
            "source": x.get("source"),
            "spoken": x["spoken"],
            "spoken_level": x["spoken_statics"]["level"],
            "written": x["written"],
            "written_level": x["written_statics"]["level"],
        })
    return items


def make_within_pair(item, rng):
    """One (written, spoken) pair with random A/B side assignment."""
    # spoken is always more oral than written (by corpus construction).
    flip = rng.random() < 0.5
    if flip:
        text_a, text_b = item["spoken"], item["written"]
        level_a, level_b = item["spoken_level"], item["written_level"]
    else:
        text_a, text_b = item["written"], item["spoken"]
        level_a, level_b = item["written_level"], item["spoken_level"]
    return _build_record(text_a, text_b, level_a, level_b,
                         source_a=item["source"], source_b=item["source"],
                         pair_type="within")


def make_cross_pair(items, rng, min_margin: int = 1, max_tries: int = 20):
    """Two spoken texts from different samples with abs(level diff) >= min_margin."""
    for _ in range(max_tries):
        a, b = rng.sample(items, 2)
        # Always use the spoken side for cross pairs (more semantic variety).
        la, lb = a["spoken_level"], b["spoken_level"]
        if abs(la - lb) >= min_margin:
            flip = rng.random() < 0.5
            if flip:
                a, b = b, a
                la, lb = lb, la
            return _build_record(a["spoken"], b["spoken"], la, lb,
                                 source_a=a["source"], source_b=b["source"],
                                 pair_type="cross")
    return None  # gave up


def _build_record(text_a, text_b, level_a, level_b, *, source_a, source_b, pair_type):
    if level_a > level_b:
        gt = "A"
    elif level_b > level_a:
        gt = "B"
    else:
        gt = "Tie"
    return {
        "text_a": text_a,
        "text_b": text_b,
        "level_a": int(level_a),
        "level_b": int(level_b),
        "gt_label": gt,
        "margin": abs(int(level_a) - int(level_b)),
        "source_a": source_a,
        "source_b": source_b,
        "pair_type": pair_type,
    }


def build_pairs(items, n_pairs: int, within_frac: float, min_margin: int, seed: int):
    rng = random.Random(seed)
    n_within = int(n_pairs * within_frac)
    n_cross = n_pairs - n_within

    pairs = []
    # Within pairs: sample without replacement
    within_items = rng.sample(items, min(n_within, len(items)))
    for it in within_items:
        pairs.append(make_within_pair(it, rng))

    # Cross pairs: keep trying random pairs of distinct samples
    tries = 0
    while len(pairs) < n_pairs and tries < n_cross * 20:
        p = make_cross_pair(items, rng, min_margin=min_margin)
        if p is not None:
            pairs.append(p)
        tries += 1

    rng.shuffle(pairs)
    return pairs


def summarize(pairs):
    from collections import Counter
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Pair-type: {Counter(p['pair_type'] for p in pairs)}")
    print(f"  GT-label : {Counter(p['gt_label'] for p in pairs)}")
    print(f"  Margin   : {Counter(p['margin'] for p in pairs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="scripts/add_stat/train_level_sft.json")
    ap.add_argument("--out_json", default="scripts/eval/evaluator_validation/pairs_pool.json")
    ap.add_argument("--n_pairs", type=int, default=2000,
                    help="Total pairs to build (we sample subsets downstream).")
    ap.add_argument("--within_frac", type=float, default=0.25,
                    help="Fraction of within-sample (written-vs-spoken) pairs.")
    ap.add_argument("--min_margin", type=int, default=1,
                    help="Min |level_a - level_b| for cross pairs.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = load_train(args.train_json)
    print(f"Loaded {len(items)} training items from {args.train_json}")
    pairs = build_pairs(items, args.n_pairs, args.within_frac, args.min_margin, args.seed)
    print("Built pairs:")
    summarize(pairs)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(pairs, f, indent=1)
    print(f"Saved -> {args.out_json}")


if __name__ == "__main__":
    main()
