"""Phase-1 LLM judge benchmark with multi-thread concurrency and majority-vote.

Differences vs run_quick_test.py:
  * Multi-threaded: --concurrency N parallel API calls (default 50).
  * Per-call JSONL checkpoint (`phase1_per_call.jsonl`) — append-only, thread-safe.
    Every (pair_idx, model) call writes one line with the full raw response, the
    parsed prediction, the error message, and the latency. Safe to resume.
  * Aggregates per-model accuracy + cross-model majority-vote accuracy in the
    final summary, plus 3-way agreement rate.

Files written:
  phase1_per_call.jsonl   — append-only call log (one line per call)
  phase1_summary.json     — final aggregated summary

Usage (one-click):
  python scripts/evaluator_validation/run_phase1_concurrent.py
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from build_pairs import load_train, build_pairs, summarize  # noqa: E402
from prompts import build_compare_prompt  # noqa: E402
from eval_llm_judges import (  # noqa: E402
    _call_with_think_fallback,
    parse_response,
    normalize_label,
    _model_supports_think,
)


DEFAULT_MODELS = [
    "gpt-5.4-nano",
    "gemini-3.1-flash-lite",
    "deepseek-v4-flash",
]

DEFAULT_MODELS = [
    # "gpt-5.4-nano",
    "gemini-3.5-flash",
    # "deepseek-v4-pro",
]


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


def load_done_set(jsonl_path: Path):
    """Read jsonl checkpoint and return set of (pair_idx, model) already finished."""
    if not jsonl_path.exists():
        return set(), 0
    done = set()
    n = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["pair_idx"], r["model"]))
                n += 1
            except Exception:
                pass
    return done, n


def majority_vote(labels):
    """Pick the most-common label among A/B/Tie ignoring None.
    Returns None if no real labels; returns 'Tie' if A and B tie."""
    real = [x for x in labels if x in ("A", "B", "Tie")]
    if not real:
        return None
    c = Counter(real)
    a, b = c.get("A", 0), c.get("B", 0)
    tie = c.get("Tie", 0)
    # If A and B are exactly tied (and >= tie count), call it Tie
    if a == b and a > 0:
        return "Tie"
    return c.most_common(1)[0][0]


def run_one_call(task, think_flag, write_lock, fp, total, counter):
    """Worker: one (pair, model) API call."""
    pair_idx, pair, model = task
    prompt = build_compare_prompt(pair["text_a"], pair["text_b"])
    t0 = time.time()
    raw, err = _call_with_think_fallback(prompt, model, think_flag)
    latency = time.time() - t0
    obj = parse_response(raw)
    pred = normalize_label(obj.get("more_spoken")) if isinstance(obj, dict) else None
    record = {
        "pair_idx": pair_idx,
        "model": model,
        "think_requested": think_flag,
        "think_effective": think_flag and _model_supports_think(model),
        "text_a": pair["text_a"],
        "text_b": pair["text_b"],
        "level_a": pair["level_a"],
        "level_b": pair["level_b"],
        "gt_label": pair["gt_label"],
        "margin": pair["margin"],
        "pair_type": pair["pair_type"],
        "prompt_chars": len(prompt),
        "raw": raw,
        "pred": pred,
        "is_correct": (pred == pair["gt_label"]) if pred is not None else None,
        "error": err,
        "latency_s": round(latency, 3),
        "timestamp": round(time.time(), 1),
    }
    with write_lock:
        fp.write(json.dumps(record) + "\n")
        fp.flush()
        counter[0] += 1
        done = counter[0]
    if done % 25 == 0 or done == total:
        print(f"  [{done}/{total}] last: {model:<24} pair#{pair_idx:<3} "
              f"gt={pair['gt_label']} pred={pred} ({latency:.1f}s)")
    return record


def aggregate(per_call_path: Path, pairs):
    """Read jsonl, return per-pair-per-model dict + per-model + majority stats."""
    by_pair_model = defaultdict(dict)  # pair_idx -> model -> record
    per_model_records = defaultdict(list)
    with open(per_call_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_pair_model[r["pair_idx"]][r["model"]] = r
            per_model_records[r["model"]].append(r)

    # Per-model stats
    per_model = {}
    for m, recs in per_model_records.items():
        n = len(recs)
        parsed = sum(1 for r in recs if r["pred"] is not None)
        correct = sum(1 for r in recs if r.get("is_correct") is True)
        api_err = sum(1 for r in recs if r.get("error"))
        latencies = [r["latency_s"] for r in recs if r["latency_s"] is not None]
        per_model[m] = {
            "n": n,
            "parsed": parsed,
            "parse_failures": n - parsed,
            "correct": correct,
            "pair_acc": correct / parsed if parsed else float("nan"),
            "api_errors": api_err,
            "mean_latency_s": round(sum(latencies) / max(len(latencies), 1), 2),
            "max_latency_s": round(max(latencies), 2) if latencies else None,
        }

    # Majority-vote stats
    maj_total = 0
    maj_correct = 0
    maj_parsed = 0
    three_way_agree = 0  # all three real labels equal
    n_pairs_with_all = 0
    per_pair_majority = []
    for p in pairs:
        idx = p["pair_idx"]
        models_data = by_pair_model.get(idx, {})
        preds = [models_data.get(m, {}).get("pred") for m in DEFAULT_MODELS]
        maj = majority_vote(preds)
        per_pair_majority.append({
            "pair_idx": idx,
            "gt_label": p["gt_label"],
            "preds": dict(zip(DEFAULT_MODELS, preds)),
            "majority": maj,
            "majority_correct": (maj == p["gt_label"]) if maj is not None else None,
        })
        maj_total += 1
        if maj is not None:
            maj_parsed += 1
            if maj == p["gt_label"]:
                maj_correct += 1
        real_preds = [x for x in preds if x is not None]
        if len(real_preds) == len(DEFAULT_MODELS):
            n_pairs_with_all += 1
            if len(set(real_preds)) == 1:
                three_way_agree += 1

    majority_stats = {
        "n_pairs": maj_total,
        "n_with_majority": maj_parsed,
        "n_correct": maj_correct,
        "pair_acc": maj_correct / maj_parsed if maj_parsed else float("nan"),
        "n_pairs_all_models_voted": n_pairs_with_all,
        "three_way_agree_among_voted": three_way_agree,
        "three_way_agree_rate": (three_way_agree / n_pairs_with_all
                                  if n_pairs_with_all else float("nan")),
    }

    return per_model, majority_stats, per_pair_majority


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="scripts/add_stat/train_level_sft.json")
    ap.add_argument("--pool_path", default="scripts/evaluator_validation/pairs_pool.json")
    ap.add_argument("--n_pool", type=int, default=500,
                    help="Pool size if pool needs to be rebuilt.")
    ap.add_argument("--rebuild_pool", action="store_true")
    ap.add_argument("--n", type=int, default=100, help="Pairs to evaluate.")
    ap.add_argument("--seed", type=int, default=1,
                    help="Sampling seed (different from quick_test default).")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--think", action="store_true", default=True,
                    help="Enable think (default True; OpenAI models auto-skip).")
    ap.add_argument("--no_think", dest="think", action="store_false",
                    help="Disable think for all models.")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--out_dir", default="scripts/evaluator_validation/phase1_concurrent")
    ap.add_argument("--min_margin", type=int, default=1)
    args = ap.parse_args()

    # Override the module-level constant if user passed custom models
    # (aggregate() reads DEFAULT_MODELS to decide the per-pair model list).
    globals()["DEFAULT_MODELS"] = list(args.models)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_call_path = out_dir / "phase1_per_call.jsonl"
    summary_path = out_dir / "phase1_summary.json"

    print("=" * 72)
    print(" Phase-1 Concurrent LLM Judge Benchmark")
    print("=" * 72)
    print(f"  models       : {args.models}")
    print(f"  n_pairs      : {args.n}")
    print(f"  think        : {args.think}")
    print(f"  concurrency  : {args.concurrency}")
    print(f"  out_dir      : {out_dir}")
    print()

    # 1) build/load pair pool
    pool = ensure_pool(Path(args.pool_path), Path(args.train_json),
                       n_pool=max(args.n_pool, args.n + 50),
                       rebuild=args.rebuild_pool, seed=42)
    rng = random.Random(args.seed)
    elig = [p for p in pool if p["margin"] >= args.min_margin]
    sample = rng.sample(elig, min(args.n, len(elig)))
    # attach a stable pair_idx so resume is robust
    for i, p in enumerate(sample):
        p["pair_idx"] = i
    print(f"[sample] {len(sample)} pairs, gt dist: "
          f"{ {k: sum(1 for p in sample if p['gt_label'] == k) for k in ('A','B','Tie')} }")

    # also save the exact pair sample for traceability
    with open(out_dir / "sampled_pairs.json", "w") as f:
        json.dump(sample, f, indent=1)

    # 2) figure out remaining tasks (resume-aware)
    done_set, n_done = load_done_set(per_call_path)
    if n_done:
        print(f"[resume] {n_done} prior calls found, resuming.")
    tasks = []
    for p in sample:
        for m in args.models:
            if (p["pair_idx"], m) in done_set:
                continue
            tasks.append((p["pair_idx"], p, m))
    total_calls = len(sample) * len(args.models)
    print(f"[plan] {total_calls} total calls, {len(tasks)} remaining, "
          f"{args.concurrency} concurrent threads.\n")

    # 3) dispatch in a thread pool with append-only jsonl writer
    write_lock = threading.Lock()
    counter = [n_done]  # mutable shared counter
    fp = open(per_call_path, "a")
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool_exec:
            futures = [
                pool_exec.submit(run_one_call, t, args.think,
                                 write_lock, fp, total_calls, counter)
                for t in tasks
            ]
            for _ in as_completed(futures):
                pass  # work already logged inside run_one_call
    finally:
        fp.flush()
        fp.close()
    elapsed = time.time() - t0
    print(f"\n[done] {len(tasks)} new calls in {elapsed:.1f}s "
          f"({len(tasks) / max(elapsed, 1e-6):.1f} calls/s)")

    # 4) aggregate
    per_model, majority_stats, per_pair_maj = aggregate(per_call_path, sample)
    summary = {
        "args": vars(args),
        "models": args.models,
        "n_pairs": len(sample),
        "per_model": per_model,
        "majority": majority_stats,
        "wall_time_s_this_run": round(elapsed, 1),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    # also save per-pair majority detail
    with open(out_dir / "per_pair_majority.json", "w") as f:
        json.dump(per_pair_maj, f, indent=1)

    # 5) pretty print summary
    print("\n" + "=" * 72)
    print(" RESULT SUMMARY")
    print("=" * 72)
    header = f"{'evaluator':<28} {'acc':>7} {'parsed':>10} {'pf':>4} {'api_err':>8} {'mean_lat':>10}"
    print(header)
    print("-" * len(header))
    for m in args.models:
        s = per_model.get(m, {})
        acc = s.get("pair_acc", float("nan"))
        acc_s = f"{acc:.3f}" if acc == acc else "  nan"
        print(f"{m:<28} {acc_s:>7} {s.get('parsed', 0):>4}/{s.get('n', 0):<4} "
              f"{s.get('parse_failures', 0):>4} {s.get('api_errors', 0):>8} "
              f"{s.get('mean_latency_s', 0):>9}s")
    ms = majority_stats
    acc = ms.get("pair_acc", float("nan"))
    acc_s = f"{acc:.3f}" if acc == acc else "  nan"
    print("-" * len(header))
    print(f"{'MAJORITY (3-way vote)':<28} {acc_s:>7} "
          f"{ms.get('n_with_majority', 0):>4}/{ms.get('n_pairs', 0):<4}")
    print(f"{'3-way unanimous rate':<28} "
          f"{ms.get('three_way_agree_rate', float('nan')):.3f}  "
          f"({ms.get('three_way_agree_among_voted', 0)}/"
          f"{ms.get('n_pairs_all_models_voted', 0)} pairs)")
    print()
    print(f"Per-call JSONL  -> {per_call_path}")
    print(f"Summary         -> {summary_path}")
    print(f"Per-pair detail -> {out_dir / 'per_pair_majority.json'}")


if __name__ == "__main__":
    main()
