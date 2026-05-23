"""Print first-N raw API responses per (model, think) cell so we can see why
some models had parse_failures=10/10. Run this AFTER run_quick_test.py finishes.

Usage:
  python scripts/evaluator_validation/debug_raw.py                     # default file
  python scripts/evaluator_validation/debug_raw.py --n 2               # 2 examples per cell
  python scripts/evaluator_validation/debug_raw.py --only gpt-4o-mini  # filter model
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default="scripts/evaluator_validation/quick_test_results.json")
    ap.add_argument("--n", type=int, default=2,
                    help="Number of example raw responses to show per (model, think).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="If given, only show these model names.")
    ap.add_argument("--show_all", action="store_true",
                    help="Show ALL cells, not just ones with parse failures.")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    runs = data.get("runs", [])
    for r in runs:
        if "hard_fail" in r:
            print(f"\n=== {r['model']} | think={r.get('think')}  HARD FAIL ===")
            print("  " + r["hard_fail"][:400])
            continue
        model = r["model"]
        think = r.get("think")
        if args.only and model not in args.only:
            continue
        if not args.show_all and r.get("parse_failures", 0) == 0:
            continue  # skip clean cells
        print(f"\n=== {model} | think={think}  "
              f"acc={r.get('pair_acc')} parse_fail={r.get('parse_failures')} ===")
        for row in (r.get("rows") or [])[: args.n]:
            print(f"\n  --- pair #{row['i']}  gt={row['gt']}  pred={row['pred']}  "
                  f"latency={row['latency_s']}s ---")
            if row.get("error"):
                print(f"  ERROR: {row['error'][:300]}")
            raw = row.get("raw")
            if raw is None:
                print("  raw = None  (API returned nothing)")
            else:
                snippet = raw if len(raw) < 800 else raw[:800] + "  ...[truncated]"
                # Print with visible boundaries
                print("  raw <<<")
                print(snippet)
                print("  >>> end raw")


if __name__ == "__main__":
    main()
