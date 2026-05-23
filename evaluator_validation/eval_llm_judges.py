"""Phase 1: Validate candidate LLM judges on a small (default n=10) subset of
pairs, using g(·)-derived GT labels.

For each candidate model we:
  1. Send the Spoken-vs-Written Compare Annotation prompt with the two texts.
  2. Parse the JSON response for `more_spoken` in {A, B, Tie}.
  3. Compute pair accuracy against the g(·)-derived GT (Tie counts only if both
     are predicted Tie).
  4. Also report per-model average latency, parse-failure count, and any errors.

Usage (from project root):
  python scripts/eval/evaluator_validation/eval_llm_judges.py \
      --pairs scripts/eval/evaluator_validation/pairs_pool.json \
      --n 10 --seed 0 --tie_window 0  --out_dir scripts/eval/evaluator_validation/phase1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add repo root to sys.path so `from clients.openai_api import ...` works.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from clients.openai_api import openai_chat  # noqa: E402
from prompts import build_compare_prompt  # noqa: E402  (same dir)


# Cheap candidate models the user listed. Add/remove as needed.
DEFAULT_MODELS = [
    "qwen3.5-flash",
    "gpt-4o-mini",
    "gpt-5.4-nano",
    "gemini-3.1-flash-lite",
    "deepseek-reasoner",
]


def parse_response(raw: str):
    """Robustly extract {"more_spoken": "A"|"B"|"Tie", "reason": "..."} from raw output."""
    if raw is None:
        return None
    s = raw.strip()
    # Strip ```json ... ``` fences if present
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    # Try direct JSON parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fallback: find first balanced { ... }
    try:
        start = s.index("{")
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(s[start : i + 1])
    except Exception:
        return None
    return None


def normalize_label(x):
    """Coerce an LLM reply to one of {'A', 'B', 'Tie'}."""
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("A", "B", "TIE"):
        return s.capitalize() if s == "TIE" else s
    if s.startswith("A"):
        return "A"
    if s.startswith("B"):
        return "B"
    if "TIE" in s or "EQUAL" in s or "SAME" in s:
        return "Tie"
    return None


# Models that we know do NOT accept the `think` kwarg (the wrapper would forward
# it as `enable_thinking` which OpenAI rejects with HTTP 400). Add to this set
# if you discover more.
_THINK_UNSUPPORTED_PREFIXES = ("gpt-", "openai-", "o1-", "o3-", "o4-")


def _model_supports_think(model_name: str) -> bool:
    n = model_name.lower()
    return not any(n.startswith(p) for p in _THINK_UNSUPPORTED_PREFIXES)


def _call_with_think_fallback(prompt: str, model_name: str, think: bool):
    """Call openai_chat; if `think` is rejected, retry once without it."""
    use_think = think and _model_supports_think(model_name)
    try:
        if use_think:
            raw = openai_chat(prompt=prompt, model_name=model_name, think=True)
        else:
            raw = openai_chat(prompt=prompt, model_name=model_name)
        return raw, None
    except Exception as e:
        msg = str(e).lower()
        if use_think and (
            "enable_thinking" in msg
            or "unknown parameter" in msg
            or "unrecognized request argument" in msg
        ):
            # Retry once, dropping the think flag.
            try:
                raw = openai_chat(prompt=prompt, model_name=model_name)
                return raw, f"think_unsupported_fallback ({type(e).__name__})"
            except Exception as e2:
                return None, repr(e2)
        return None, repr(e)


def eval_one_model(model_name: str, pairs, sleep_s: float, think: bool = False):
    correct = 0
    total = 0
    parse_fail = 0
    api_err = 0
    rows = []
    t0 = time.time()
    effective_think = think and _model_supports_think(model_name)
    label = f"{model_name}|think={think}"
    if think and not effective_think:
        label += " (skipped: model rejects think)"
    for i, p in enumerate(pairs, 1):
        prompt = build_compare_prompt(p["text_a"], p["text_b"])
        t_start = time.time()
        raw, err = _call_with_think_fallback(prompt, model_name, think)
        if raw is None and err is not None:
            api_err += 1
        latency = time.time() - t_start
        obj = parse_response(raw)
        pred = normalize_label(obj.get("more_spoken")) if isinstance(obj, dict) else None
        if pred is None:
            parse_fail += 1
        is_correct = pred == p["gt_label"] if pred is not None else False
        if pred is not None:
            total += 1
            if is_correct:
                correct += 1
        rows.append({
            "i": i,
            "gt": p["gt_label"],
            "margin": p["margin"],
            "pair_type": p["pair_type"],
            "pred": pred,
            "raw": raw,
            "latency_s": round(latency, 2),
            "error": err,
        })
        print(f"  [{label}] {i:3d}/{len(pairs)}  gt={p['gt_label']}  pred={pred}  "
              f"{'✓' if is_correct else '✗'}  ({latency:.1f}s)")
        if sleep_s > 0:
            time.sleep(sleep_s)
    acc = correct / total if total else float("nan")
    elapsed = time.time() - t0
    return {
        "model": model_name,
        "think": think,
        "n_pairs": len(pairs),
        "n_parsed": total,
        "n_correct": correct,
        "pair_acc": acc,
        "parse_failures": parse_fail,
        "api_errors": api_err,
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / max(len(rows), 1), 2),
        "wall_time_s": round(elapsed, 1),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="scripts/eval/evaluator_validation/pairs_pool.json")
    ap.add_argument("--n", type=int, default=10, help="Sample size per model.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--sleep_s", type=float, default=0.0,
                    help="Optional sleep between API calls to be polite.")
    ap.add_argument("--out_dir", default="scripts/evaluator_validation/phase1")
    ap.add_argument("--min_margin", type=int, default=1,
                    help="Filter pool to pairs with abs(level diff) >= this.")
    ap.add_argument("--think", action="store_true",
                    help="Enable thinking/CoT mode if the model supports it.")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    with open(args.pairs) as f:
        pool = json.load(f)
    elig = [p for p in pool if p["margin"] >= args.min_margin]
    sample = rng.sample(elig, min(args.n, len(elig)))
    print(f"Sampled {len(sample)} pairs from {len(elig)} eligible "
          f"(margin >= {args.min_margin}) out of {len(pool)} pool.")
    print(f"Models to evaluate: {args.models}")
    print()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for m in args.models:
        print(f"\n=== Evaluating {m} (think={args.think}) ===")
        try:
            res = eval_one_model(m, sample, sleep_s=args.sleep_s, think=args.think)
        except Exception as e:
            print(f"  HARD FAIL for {m}: {e!r}")
            res = {"model": m, "hard_fail": repr(e)}
        # write per-model detail
        detail_path = out_dir / f"detail_{m.replace('/', '_')}.json"
        with open(detail_path, "w") as f:
            json.dump(res, f, indent=1)
        summary.append({
            k: v for k, v in res.items() if k != "rows"
        })
        print(f"  -> saved {detail_path}")
        if "pair_acc" in res:
            print(f"  pair_acc = {res['pair_acc']:.3f}  "
                  f"(parsed {res['n_parsed']}/{res['n_pairs']}, "
                  f"parse_fail={res['parse_failures']}, api_err={res['api_errors']})")

    # write top-level summary
    sum_path = out_dir / "summary.json"
    with open(sum_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary,
                   "sampled_pairs": sample}, f, indent=1)
    print("\n========== SUMMARY ==========")
    print(f"{'model':<25} {'acc':>6} {'parsed':>8} {'parse_fail':>10} {'api_err':>8} {'mean_lat_s':>10}")
    for s in summary:
        if "pair_acc" in s:
            print(f"{s['model']:<25} {s['pair_acc']:>6.3f} {s['n_parsed']:>8} "
                  f"{s['parse_failures']:>10} {s['api_errors']:>8} {s['mean_latency_s']:>10}")
        else:
            print(f"{s['model']:<25} HARD FAIL: {s.get('hard_fail','')[:60]}")
    print(f"\nFull summary -> {sum_path}")


if __name__ == "__main__":
    main()
