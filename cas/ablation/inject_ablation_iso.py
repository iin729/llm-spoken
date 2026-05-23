"""
inject_ablation_iso.py
======================

Iso variant of CAS for the paper's Table 3 (CAS-component ablation):

    Iso = L_oral (multi-layer) + real vectors + isotonic regression on
          (t, α*_t)  instead of a quadratic fit.

Everything except the analyze step is identical to `full_cas` from
`inject_ablation.py`, so this row of Table 3 isolates the effect of the
*fit family* alone (quadratic vs. monotone non-parametric).

Three modes (matching inject_ablation.py)
-----------------------------------------
  calibrate : sweep alpha on the calibration set (same grid as full_cas)
  analyze   : per-level best alpha α*_t, then ISOTONIC regression
              (Barlow et al., 1972 PAVA) on {(0,0), (1, α*_1), …, (5, α*_5)}
              constrained to be monotone non-decreasing in t.
              Saves a small monotone lookup table.
  predict   : evaluate the monotone table at integer t (linear interpolation
              for any non-integer t); supports multi-GPU chunking.

Outputs into --output_dir
-------------------------
  experiment_results.xlsx       (calibrate)
  coeff_function_params.json    (analyze; type = "isotonic")
  final_predictions.jsonl       (predict; after manual concat of .partN chunks)

Reuses these helpers from inject_ablation.py (same directory):
  SteeringManager, parse_layer_range, load_vectors,
  build_prompt_messages, calculate_level_only,
  SYSTEM_PROMPT_TEMPLATE
The eval / scoring module discovery + symlink dance in inject_ablation.py
runs at import time and is therefore re-used as-is.
"""

import os
import sys
import json
import math
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reuse helpers from inject_ablation.py (same dir).  Importing this module also
# performs the eval-scoring resolution + oral_level_model.pkl symlink dance,
# so we don't have to repeat that boilerplate here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inject_ablation import (   # noqa: E402
    SteeringManager,
    parse_layer_range,
    load_vectors,
    build_prompt_messages,
    calculate_level_only,
)


# ----------------------------------------------------------------------
# Isotonic regression — prefer sklearn, fall back to a self-contained
# Pool-Adjacent-Violators (PAVA) implementation if sklearn is missing.
# ----------------------------------------------------------------------
def _pava_isotonic(x, y, increasing=True):
    """Monotone non-decreasing (or non-increasing) least-squares fit via
    the Pool-Adjacent-Violators Algorithm.  Returns the fitted values at
    the same x positions, sorted to match the input order.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).copy()
    order = np.argsort(x)
    y_sorted = y[order]
    if not increasing:
        y_sorted = -y_sorted
    blocks = [(float(v), 1) for v in y_sorted]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] <= blocks[i + 1][0]:
            i += 1
            continue
        v0, w0 = blocks[i]
        v1, w1 = blocks[i + 1]
        merged = (v0 * w0 + v1 * w1) / (w0 + w1)
        blocks = blocks[:i] + [(merged, w0 + w1)] + blocks[i + 2:]
        # back-merge until monotone again
        while i > 0 and blocks[i - 1][0] > blocks[i][0]:
            v0, w0 = blocks[i - 1]
            v1, w1 = blocks[i]
            merged = (v0 * w0 + v1 * w1) / (w0 + w1)
            blocks = blocks[:i - 1] + [(merged, w0 + w1)] + blocks[i + 1:]
            i -= 1
    n = len(y_sorted)
    out_sorted = np.zeros(n)
    idx = 0
    for v, w in blocks:
        for _ in range(w):
            out_sorted[idx] = v
            idx += 1
    if not increasing:
        out_sorted = -out_sorted
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def fit_isotonic(X, Y, y_min=0.0, increasing=True):
    """Returns (fitted_Y, model_info_dict).  Prefers sklearn.IsotonicRegression
    if available, otherwise uses the self-contained PAVA implementation.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(y_min=y_min, increasing=increasing,
                                 out_of_bounds="clip")
        Y_fit = iso.fit_transform(X, Y)
        info = {
            "backend": "sklearn",
            "X_thresholds_": [float(v) for v in iso.X_thresholds_.tolist()],
            "y_thresholds_": [float(v) for v in iso.y_thresholds_.tolist()],
        }
    except ImportError:
        Y_fit = _pava_isotonic(X, Y, increasing=increasing)
        if y_min is not None:
            Y_fit = np.maximum(Y_fit, y_min)
        info = {"backend": "pava-fallback"}
    return np.array([float(v) for v in Y_fit.tolist()]), info


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------
def run_calibration(args, model, tokenizer, manager, target_layers):
    print("\n>>> Mode: Calibrate (iso_cas)")
    with open(args.data_path, "r", encoding="utf-8") as f:
        cal_data = json.load(f)[: args.calib_size]

    prompts, labels = [], []
    for item in cal_data:
        msgs = build_prompt_messages(item["written"])
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
        labels.append(item["output"])

    coeffs_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0,
                      1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    print(f"Sweeping alphas: {coeffs_to_test} on layers {sorted(target_layers)}")

    BATCH = 8
    results = []
    for coeff in tqdm(coeffs_to_test, desc="Alpha Sweep"):
        manager.set_config(layers=target_layers, coeff=coeff)
        for i in range(0, len(prompts), BATCH):
            batch_prompts = prompts[i:i + BATCH]
            batch_labels = labels[i:i + BATCH]
            inputs = tokenizer(batch_prompts, return_tensors="pt",
                               padding=True).to(model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=100, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            preds = tokenizer.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )
            for j, pred in enumerate(preds):
                results.append({
                    "experiment": "coeff_sweep", "coeff": coeff,
                    "output": batch_labels[j], "prediction": pred,
                })

    manager.remove_hooks()
    os.makedirs(args.output_dir, exist_ok=True)
    out_xlsx = os.path.join(args.output_dir, "experiment_results.xlsx")
    pd.DataFrame(results).to_excel(out_xlsx, index=False)
    print(f"Calibration sweep saved: {out_xlsx}")


def run_analysis_iso(args):
    print("\n>>> Mode: Analyze (iso_cas) — isotonic regression on (t, α*_t)")
    xlsx = os.path.join(args.output_dir, "experiment_results.xlsx")
    df = pd.read_excel(xlsx)
    sweep = df[df["experiment"] == "coeff_sweep"].copy()

    tqdm.pandas(desc="Calc Levels")
    sweep["Level_pred"] = sweep["prediction"].progress_apply(calculate_level_only)
    agg = sweep.groupby("coeff")["Level_pred"].mean().reset_index()
    print("\nPer-alpha mean predicted level:\n", agg)

    if agg["Level_pred"].isna().all():
        raise RuntimeError(
            "All per-alpha mean predicted levels are NaN. Scoring pipeline "
            "is broken at runtime (stat_metric8 / eval_oscore_level returned "
            "no features). Bailing out — refusing to silently fall back to "
            "alpha = 0 (= no steering)."
        )

    # Same as full_cas: per-target-level best alpha
    best_rows = []
    for t in [1, 2, 3, 4, 5]:
        agg["diff"] = (agg["Level_pred"] - t).abs()
        best_alpha = float(agg.sort_values("diff").iloc[0]["coeff"])
        best_rows.append({"t": t, "best_alpha": best_alpha})
    fit_df = pd.DataFrame(best_rows)

    X = np.array([0] + fit_df["t"].tolist(), dtype=float)
    Y_raw = np.array([0.0] + fit_df["best_alpha"].tolist(), dtype=float)
    print("\nRaw (t, α*_t) anchors before isotonic fit:")
    for x, y in zip(X.tolist(), Y_raw.tolist()):
        print(f"   t={int(x)}  α*_t={y:.4f}")

    Y_iso, info = fit_isotonic(X, Y_raw, y_min=0.0, increasing=True)
    print(f"\nIsotonic backend: {info['backend']}")
    print("Iso-fitted α(t) (monotone non-decreasing):")
    for x, y in zip(X.tolist(), Y_iso.tolist()):
        print(f"   t={int(x)}  α(t)={y:.4f}")

    params = {
        "type": "isotonic",
        "anchors_t": [int(x) for x in X.tolist()],
        "anchors_alpha_raw":    [float(v) for v in Y_raw.tolist()],
        "anchors_alpha_fitted": [float(v) for v in Y_iso.tolist()],
        "increasing": True,
        "y_min": 0.0,
        "iso_info": info,
    }
    params_path = os.path.join(args.output_dir, "coeff_function_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved {params_path}")


def iso_lookup_alpha(params, t):
    """Lookup α(t) from the saved monotone anchor table.

    If t is an integer anchor, returns the exact monotone-fitted value.
    Otherwise linearly interpolates between adjacent anchors and clamps to
    the table's [first, last] range.  Output is always ≥ y_min (= 0.0) by
    construction of the isotonic fit.
    """
    xs = params["anchors_t"]                  # e.g. [0,1,2,3,4,5]
    ys = params["anchors_alpha_fitted"]
    if t <= xs[0]:
        return float(ys[0])
    if t >= xs[-1]:
        return float(ys[-1])
    for i in range(len(xs) - 1):
        if xs[i] <= t <= xs[i + 1]:
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            if x1 == x0:
                return float(y0)
            frac = (t - x0) / (x1 - x0)
            return float(y0 + frac * (y1 - y0))
    return float(ys[-1])


def run_prediction(args, model, tokenizer, manager, target_layers):
    print(f"\n>>> Mode: Predict (iso_cas, chunk={args.chunk_idx}/{args.num_chunks})")
    params_path = os.path.join(args.output_dir, "coeff_function_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Missing {params_path}. Run analyze first.")
    with open(params_path, "r") as f:
        params = json.load(f)
    if params.get("type") != "isotonic":
        raise ValueError(
            f"Expected isotonic params but found type='{params.get('type')}'. "
            f"Did you accidentally point at a poly2 / fixed_alpha output dir?"
        )
    print(f"Loaded iso params: anchors_t={params['anchors_t']}  "
          f"anchors_alpha_fitted={params['anchors_alpha_fitted']}")

    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    chunk_size = math.ceil(len(all_data) / args.num_chunks)
    start = args.chunk_idx * chunk_size
    end = min(start + chunk_size, len(all_data))
    test_data = all_data[start:end]

    results = []
    for item in tqdm(test_data, desc=f"Chunk {args.chunk_idx}"):
        target_lvl = item.get("level", 3)
        if "spoken_statics" in item:
            target_lvl = item["spoken_statics"].get("level", target_lvl)

        coeff = iso_lookup_alpha(params, float(target_lvl))
        coeff = max(0.0, min(coeff, args.alpha_cap))

        manager.set_config(layers=target_layers, coeff=coeff)
        msgs = build_prompt_messages(item["written"])
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        pred = tokenizer.decode(
            outputs[:, input_len:][0], skip_special_tokens=True
        )
        results.append({
            "instruction": text, "predict": pred, "label": item["output"],
            "target_level": target_lvl, "applied_coeff": coeff,
            "variant": "iso_cas",
        })

    manager.remove_hooks()
    out_part = f"{args.prediction_file}.part{args.chunk_idx}"
    with open(out_part, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {out_part}  "
          f"(chunk {args.chunk_idx}/{args.num_chunks}, {len(results)} rows)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Iso variant of CAS — isotonic regression on (t, α*_t)."
    )
    p.add_argument("--mode", required=True,
                   choices=["calibrate", "analyze", "predict"])

    p.add_argument("--model_path", required=True)
    p.add_argument("--vector_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--prediction_file", default="final_predictions.jsonl")

    p.add_argument("--oral_layers", default="12-28",
                   help="L_oral range (same as full_cas). "
                        "e.g. 12-28 (Qwen-3B), 5-10 (Llama-1B), 15-25 (Mistral-7B).")

    p.add_argument("--calib_size", type=int, default=100)
    p.add_argument("--num_chunks", type=int, default=1)
    p.add_argument("--chunk_idx", type=int, default=0)
    p.add_argument("--alpha_cap", type=float, default=4.0,
                   help="Upper bound on the applied steering coefficient at "
                        "inference. The isotonic fit is unconstrained on the "
                        "upper side; this clamp matches the full_cas runner "
                        "(use --alpha_cap 1.5 for Mistral safety).")
    args = p.parse_args()

    # analyze mode: no model load needed
    if args.mode == "analyze":
        run_analysis_iso(args)
        return

    # Load model + tokenizer
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16
    )

    real_vecs = load_vectors(args.vector_path)
    target_layers = parse_layer_range(args.oral_layers)
    print(f"[iso_cas] target_layers = {sorted(target_layers)}")

    manager = SteeringManager(model, real_vecs)

    if args.mode == "calibrate":
        run_calibration(args, model, tokenizer, manager, target_layers)
    elif args.mode == "predict":
        run_prediction(args, model, tokenizer, manager, target_layers)


if __name__ == "__main__":
    main()
