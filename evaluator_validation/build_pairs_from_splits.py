"""Construct train_pairs.json / dev_pairs.json / test_pairs.json from the
WSPairs splits.

Pairing rule (per --pair_unit):
  * "text"   (default): each split is flattened to a "text bank" of both spoken
                        AND written sides; every text is paired with 1 random
                        other text from the same bank. → 2 × |split| pairs.
  * "record":           each record contributes ONE text (the spoken side) and
                        is paired with 1 other record from the same split. →
                        |split| pairs.

GT label is derived from g(·) (= EWM+KMeans cluster level):
    A if level_a > level_b ;  B if level_b > level_a ;  Tie if equal.
A/B sides are randomly shuffled to balance position bias.

Usage (from project root):
    python scripts/evaluator_validation/build_pairs_from_splits.py
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def collect_text_bank(items, pair_unit: str):
    bank = []
    if pair_unit == "text":
        for i, x in enumerate(items):
            bank.append({"text": x["spoken"], "level": int(x["spoken_statics"]["level"]),
                         "source": x.get("source"), "side": "spoken", "record_idx": i})
            bank.append({"text": x["written"], "level": int(x["written_statics"]["level"]),
                         "source": x.get("source"), "side": "written", "record_idx": i})
    elif pair_unit == "record":
        # one entry per record (spoken side)
        for i, x in enumerate(items):
            bank.append({"text": x["spoken"], "level": int(x["spoken_statics"]["level"]),
                         "source": x.get("source"), "side": "spoken", "record_idx": i})
    else:
        raise ValueError(f"unknown pair_unit: {pair_unit}")
    return bank


def build_pairs(items, pair_unit: str, seed: int):
    rng = random.Random(seed)
    bank = collect_text_bank(items, pair_unit)

    pairs = []
    for i, a in enumerate(bank):
        # Sample 1 partner from the same bank, excluding self AND same-record
        # entries (so we never pair a record's written with its own spoken).
        for _ in range(50):
            j = rng.randrange(len(bank))
            if j == i:
                continue
            if bank[j].get("record_idx") == a.get("record_idx"):
                continue
            break
        else:
            j = (i + 1) % len(bank)  # extreme fallback
        b = bank[j]
        # Random A/B shuffle to remove position bias
        if rng.random() < 0.5:
            a_pair, b_pair = a, b
        else:
            a_pair, b_pair = b, a

        la, lb = a_pair["level"], b_pair["level"]
        if la > lb:
            gt = "A"
        elif lb > la:
            gt = "B"
        else:
            gt = "Tie"
        pairs.append({
            "text_a": a_pair["text"],
            "text_b": b_pair["text"],
            "level_a": la,
            "level_b": lb,
            "gt_label": gt,
            "margin": abs(la - lb),
            "source_a": a_pair.get("source"),
            "source_b": b_pair.get("source"),
            "side_a": a_pair.get("side"),
            "side_b": b_pair.get("side"),
            "pair_type": "cross",
        })
    return pairs


def summarize(pairs, name: str):
    gt = Counter(p["gt_label"] for p in pairs)
    margin = Counter(p["margin"] for p in pairs)
    side_combo = Counter((p["side_a"], p["side_b"]) for p in pairs)
    print(f"  [{name}] total pairs = {len(pairs)}")
    print(f"      GT label dist : {dict(gt)}")
    print(f"      Margin dist   : {dict(sorted(margin.items()))}")
    print(f"      Side combos   : {dict(side_combo)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="scripts/add_stat/train_level_sft.json")
    ap.add_argument("--dev_json", default="scripts/add_stat/dev_level_sft.json")
    ap.add_argument("--test_json", default="scripts/add_stat/test_level_sft.json")
    ap.add_argument("--out_dir", default="scripts/evaluator_validation")
    ap.add_argument("--pair_unit", choices=["text", "record"], default="text",
                    help="`text` (default): bank includes both spoken+written, 2x pairs. "
                         "`record`: 1 pair per record (spoken side only).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", args.train_json),
        ("dev", args.dev_json),
        ("test", args.test_json),
    ]
    for name, path in splits:
        items = json.loads(Path(path).read_text())
        # Each split gets its own seed offset so within-pair randomization
        # is independent across splits.
        seed = args.seed + {"train": 0, "dev": 1, "test": 2}[name]
        pairs = build_pairs(items, args.pair_unit, seed)
        out_path = out_dir / f"{name}_pairs.json"
        with open(out_path, "w") as f:
            json.dump(pairs, f, indent=1)
        print(f"\n{name}: {len(items)} records -> {out_path}")
        summarize(pairs, name)

    print(f"\n[done] all 3 split pair files saved to {out_dir}")


if __name__ == "__main__":
    main()
