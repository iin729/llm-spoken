"""
inject_ablation.py
==================

Unified mode-driven injector for the CAS component ablation + ActAdd baseline.
Self-contained: subsumes the calibration / polynomial-fit / chunked-prediction
flow of unified_injector.py and adds variant-specific layer/vector overrides.

Six supported variants
----------------------
  full_cas       : L_oral (multi-layer) + calibrated alpha*(t)             [= our CAS]
  no_calib       : L_oral (multi-layer) + best fixed scalar alpha          [- calibration]
  no_multilayer  : {best single layer}   + calibrated alpha*(t)            [- multi-layer]
  actadd         : {best single layer}   + best fixed scalar alpha         [Turner et al. 2023]
  random_layers  : random non-oral layer set + calibrated alpha*(t)        [sanity 1]
  random_vec     : L_oral + random Gaussian per-layer vectors + calibrated [sanity 2]

Three modes
-----------
  calibrate : sweep alpha on the calibration set with the variant's layer/vector config
  analyze   : fit alpha*(t) polynomial OR pick the best fixed alpha (variant-dependent)
  predict   : run on the test set with the variant's alpha policy; supports multi-GPU chunking

Outputs into --output_dir
-------------------------
  experiment_results.xlsx        (calibrate)
  coeff_function_params.json     (analyze; type "poly2" or "fixed_alpha")
  final_predictions.jsonl        (predict; after manual concat of .partN chunks)
"""

import os
import sys
import json
import math
import argparse
import random
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add repo root to sys.path so `from orality import ...` works regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# IMPORTANT: do NOT silently swallow ImportError here. If scoring is missing,
# the analyze step would produce NaN -> alpha = 0 -> no steering. Fail loud.
try:
    from orality import eval_oscore_and_level, stat_all_metrics
except ImportError as e:
    raise ImportError(
        f"Failed to import the `orality` package (provides eval_oscore_and_level "
        f"and stat_all_metrics). Make sure the repo root is on PYTHONPATH and "
        f"`pip install -r requirements.txt` succeeded.\n"
        f"Original error: {e}"
    ) from e


# ----------------------------------------------------------------------
# 0. Prompt template (must match training/inference; aligned with WSPairs)
# ----------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """### Instruction
You are an expert conversationalist. Your task is to convert formal written language into spontaneous spoken language at a specified **Orality Level (0-5)** with **Semantic Preservation**. Please directly output the rewritten spoken text without any explanations or formatting symbols.

### Orality Level Definitions
You must adjust the density of orality words and phrases, repetition, and sentence fragmentation based on the target level:

- Level 0 (written): Formal structure. High lexical density. No fillers. No extra interaction. No repetition.
- Level 1 (spoken, slight): Light contraction and mild simplification. Rare discourse markers. No repetition.
- Level 2 (spoken, low): Add occasional fillers (um/uh/okay) and light turn-organization (well/so). Slight sentence splitting.
- Level 3 (spoken, medium): Clear conversational flow. Some stance/hedges (I think/maybe). More fragmentation. Small, natural repetitions allowed.
- Level 4 (spoken, high): Frequent interaction cues (you know/right?). More hedging. Noticeable repetition and shorter clauses, but still coherent.
- Level 5 (spoken, very high): Strongly spontaneous. Frequent fillers + discourse markers + turn-yielding cues. More repetition and repairs. Still preserve every fact.

### Semantic Preservation

- Preserve Core Information: Do not add, invent, change, or omit substantive content (e.g., entities, numbers, conditions, stances).
- Maintain Logical Integrity: Ensure the core meaning and intent are clear. While sentence structure should be simplified, the original logical order and core meaning must be maintained.

### Let's begin
Please directly output the one spoken text at a specified Orality Level.
"""


# (scoring deps already imported above; they are mandatory, not optional)


# ----------------------------------------------------------------------
# 1. Variant catalogue
# ----------------------------------------------------------------------
VARIANTS = {
    "full_cas":      {"layers": "oral",   "vectors": "real",   "alpha": "calib"},
    "no_calib":      {"layers": "oral",   "vectors": "real",   "alpha": "fixed"},
    "no_multilayer": {"layers": "single", "vectors": "real",   "alpha": "calib"},
    "actadd":        {"layers": "single", "vectors": "real",   "alpha": "fixed"},
    "random_layers": {"layers": "random", "vectors": "real",   "alpha": "calib"},
    "random_vec":    {"layers": "oral",   "vectors": "random", "alpha": "calib"},
}


def variant_config(variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Pick from {list(VARIANTS)}.")
    return VARIANTS[variant]


# ----------------------------------------------------------------------
# 2. Steering manager
# ----------------------------------------------------------------------
class SteeringManager:
    def __init__(self, model, vectors):
        self.model = model
        self.vectors = vectors           # {layer_idx -> tensor}
        self.target_layers = []
        self.coeff = 0.0
        self.hooks = []

    def set_config(self, layers, coeff):
        self.remove_hooks()
        self.target_layers = layers
        self.coeff = coeff
        if len(layers) > 0 and coeff != 0:
            self._register_hooks()

    def _register_hooks(self):
        for layer_idx in self.target_layers:
            if layer_idx not in self.vectors:
                continue
            layer_module = self.model.model.layers[layer_idx]

            def hook_fn(module, args, output, idx=layer_idx):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                vec = self.vectors[idx].to(hidden.device).to(hidden.dtype)
                hidden.add_(self.coeff * vec.view(1, 1, -1))
                return output

            handle = layer_module.register_forward_hook(hook_fn)
            self.hooks.append(handle)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ----------------------------------------------------------------------
# 3. Helpers
# ----------------------------------------------------------------------
def parse_layer_range(s: str):
    """`12-28` or `12,15,20` or `20` -> list[int]."""
    if not s:
        return []
    s = str(s).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    return [int(s)]


def load_vectors(path):
    print(f"Loading vectors from: {path}")
    raw_vectors = torch.load(path)
    vecs = {}
    for k, v in raw_vectors.items():
        try:
            if isinstance(k, int):
                idx = k
            elif isinstance(k, str):
                idx = int(k.replace("layer_", "").replace("layers.", ""))
            else:
                continue
            vec = v.float()
            vec = vec / (vec.norm() + 1e-6)   # L2-normalize per layer
            vecs[idx] = vec
        except Exception:
            continue
    return vecs


def build_random_vectors(real_vecs, seed: int):
    """Per-layer Gaussian vectors matched in shape and L2-norm to the real ones."""
    g = torch.Generator().manual_seed(seed)
    rand_vecs = {}
    for idx, v in real_vecs.items():
        rv = torch.randn(v.shape, generator=g, dtype=torch.float32)
        rv = rv / (rv.norm() + 1e-6)
        rand_vecs[idx] = rv
    return rand_vecs


def resolve_target_layers(variant: str, oral_layers, best_single_layer, all_model_layers,
                          random_layer_set=None, seed: int = 0):
    cfg = variant_config(variant)
    mode = cfg["layers"]
    if mode == "oral":
        return list(oral_layers)
    if mode == "single":
        if best_single_layer is None:
            raise ValueError("--best_single_layer required for variants that use a single layer.")
        return [int(best_single_layer)]
    if mode == "random":
        if random_layer_set is not None:
            return list(random_layer_set)
        # Sample matched-count non-oral layers
        random.seed(seed)
        oral_set = set(oral_layers)
        non_oral = [l for l in all_model_layers if l not in oral_set]
        return random.sample(non_oral, min(len(oral_layers), len(non_oral)))
    raise ValueError(f"Unknown layer mode: {mode}")


def resolve_vectors(variant: str, real_vecs, seed: int = 0):
    cfg = variant_config(variant)
    if cfg["vectors"] == "real":
        return real_vecs
    if cfg["vectors"] == "random":
        return build_random_vectors(real_vecs, seed=seed)
    raise ValueError(f"Unknown vector mode: {cfg['vectors']}")


def build_prompt_messages(written_text):
    user_content = f"**Written Language Level 0:**\n{written_text}\n\n**Spoken Language Level 0:**\n"
    return [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": user_content},
    ]


def calculate_level_only(text):
    """Predict orality level (int) for a single text. Returns NaN only when
    the text itself is empty/None and stat_all_metrics legitimately yields
    no features. Module-level import already guarantees scoring deps exist."""
    if text is None or (isinstance(text, str) and not text.strip()):
        return float("nan")
    stats = stat_all_metrics(str(text))
    if not stats:
        return float("nan")
    _, level = eval_oscore_and_level(stats)
    return int(level)


# ----------------------------------------------------------------------
# 4. Mode: calibrate
# ----------------------------------------------------------------------
def run_calibration(args, model, tokenizer, manager, target_layers):
    print(f"\n>>> Mode: Calibrate (variant={args.variant})")
    with open(args.data_path, "r", encoding="utf-8") as f:
        cal_data = json.load(f)[: args.calib_size]

    prompts, labels = [], []
    for item in cal_data:
        msgs = build_prompt_messages(item["written"])
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        labels.append(item["output"])

    coeffs_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    print(f"Sweeping alphas: {coeffs_to_test} on layers {sorted(target_layers)}")

    BATCH = 8
    results = []
    for coeff in tqdm(coeffs_to_test, desc="Alpha Sweep"):
        manager.set_config(layers=target_layers, coeff=coeff)
        for i in range(0, len(prompts), BATCH):
            batch_prompts = prompts[i: i + BATCH]
            batch_labels = labels[i: i + BATCH]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=100, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            preds = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
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


# ----------------------------------------------------------------------
# 5. Mode: analyze
#    - For calibrated variants: fit alpha*(t) quadratic anchored at (0,0).
#    - For fixed-alpha variants: pick best fixed alpha (closest mean predicted level
#      to the diagonal midpoint t=3, OR maximize average alignment, configurable).
# ----------------------------------------------------------------------
def run_analysis(args):
    print(f"\n>>> Mode: Analyze (variant={args.variant})")
    cfg = variant_config(args.variant)
    df = pd.read_excel(os.path.join(args.output_dir, "experiment_results.xlsx"))
    sweep = df[df["experiment"] == "coeff_sweep"].copy()

    tqdm.pandas(desc="Calc Levels")
    sweep["Level_pred"] = sweep["prediction"].progress_apply(calculate_level_only)
    agg = sweep.groupby("coeff")["Level_pred"].mean().reset_index()
    print("\nPer-alpha mean predicted level:\n", agg)

    # Sanity check: if every alpha's mean predicted level is NaN, scoring is
    # broken at runtime (even though import succeeded). Bail out so we don't
    # silently fall back to alpha = 0 (= no steering).
    if agg["Level_pred"].isna().all():
        raise RuntimeError(
            "All per-alpha mean predicted levels are NaN. This usually means "
            "the scoring pipeline (stat_metric8 / eval_oscore_level) returned "
            "no features for every prediction. Verify spacy model / vocabulary "
            "loaded correctly; check stat_all_metrics(text) on a sample prediction."
        )

    params_path = os.path.join(args.output_dir, "coeff_function_params.json")

    if cfg["alpha"] == "calib":
        # Anchored quadratic fit through {(0,0)} U {(t, best_alpha_t)}_{t=1..5}
        best_rows = []
        for t in [1, 2, 3, 4, 5]:
            agg["diff"] = (agg["Level_pred"] - t).abs()
            best_alpha = float(agg.sort_values("diff").iloc[0]["coeff"])
            best_rows.append({"t": t, "best_alpha": best_alpha})
        fit_df = pd.DataFrame(best_rows)
        X = np.array([0] + fit_df["t"].tolist())
        Y = np.array([0] + fit_df["best_alpha"].tolist())
        z = np.polyfit(X, Y, 2)
        p2, p1, p0 = float(z[0]), float(z[1]), float(z[2])
        params = {"p2": p2, "p1": p1, "p0": p0, "type": "poly2"}
        print(f"Fitted poly2: alpha(t) = {p2:.4f} t^2 + {p1:.4f} t + {p0:.4f}")
    else:
        # Fixed alpha: pick the alpha whose mean predicted level is closest to 3
        # (= midpoint of {1..5}); this is the "best single scalar" interpretation.
        agg["diff_to_mid"] = (agg["Level_pred"] - args.fixed_alpha_target).abs()
        best_row = agg.sort_values("diff_to_mid").iloc[0]
        fixed_alpha = float(best_row["coeff"])
        params = {"fixed_alpha": fixed_alpha, "type": "fixed_alpha",
                  "selected_for_target_level": args.fixed_alpha_target}
        print(f"Selected fixed alpha = {fixed_alpha:.3f} "
              f"(mean predicted level at this alpha = {best_row['Level_pred']:.2f}; "
              f"target_level for selection = {args.fixed_alpha_target})")

    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved {params_path}")


# ----------------------------------------------------------------------
# 6. Mode: predict
# ----------------------------------------------------------------------
def run_prediction(args, model, tokenizer, manager, target_layers):
    print(f"\n>>> Mode: Predict (variant={args.variant}, chunk={args.chunk_idx}/{args.num_chunks})")
    params_path = os.path.join(args.output_dir, "coeff_function_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Missing {params_path}. Run analyze first.")
    with open(params_path, "r") as f:
        params = json.load(f)
    print(f"Loaded params: {params}")

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

        # Coefficient policy
        if params.get("type") == "fixed_alpha":
            coeff = float(params["fixed_alpha"])
        else:
            p2 = float(params.get("p2", 0.0))
            p1 = float(params.get("p1", 0.5))
            p0 = float(params.get("p0", 0.0))
            coeff = p2 * (target_lvl ** 2) + p1 * target_lvl + p0
        coeff = max(0.0, min(coeff, args.alpha_cap))

        manager.set_config(layers=target_layers, coeff=coeff)
        msgs = build_prompt_messages(item["written"])
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        pred = tokenizer.decode(outputs[:, input_len:][0], skip_special_tokens=True)
        results.append({
            "instruction": text, "predict": pred, "label": item["output"],
            "target_level": target_lvl, "applied_coeff": coeff,
            "variant": args.variant,
        })

    manager.remove_hooks()
    out_part = f"{args.prediction_file}.part{args.chunk_idx}"
    with open(out_part, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {out_part}  (chunk {args.chunk_idx}/{args.num_chunks}, {len(results)} rows)")


# ----------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--mode", required=True, choices=["calibrate", "analyze", "predict"])

    p.add_argument("--model_path", required=True)
    p.add_argument("--vector_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--prediction_file", default="final_predictions.jsonl")

    # Layer configuration
    p.add_argument("--oral_layers", default="12-28",
                   help="L_oral range, e.g. 12-28 (Qwen-3B), 5-10 (Llama-1B), 15-25 (Mistral-7B).")
    p.add_argument("--best_single_layer", type=int, default=None,
                   help="Single best layer (required for actadd and no_multilayer).")
    p.add_argument("--all_model_layers", default="0-35",
                   help="Total layer range of the backbone, used for sampling random non-oral set.")
    p.add_argument("--random_layer_seed", type=int, default=0,
                   help="Seed for random_layers / random_vec sampling (for reproducibility).")

    # Calibration / chunking
    p.add_argument("--calib_size", type=int, default=100,
                   help="Number of calibration-set samples used in the alpha sweep.")
    p.add_argument("--num_chunks", type=int, default=1)
    p.add_argument("--chunk_idx", type=int, default=0)
    p.add_argument("--fixed_alpha_target", type=float, default=3.0,
                   help="Target level used to pick the best fixed alpha "
                        "for variants no_calib / actadd.")
    p.add_argument("--alpha_cap", type=float, default=4.0,
                   help="Upper bound on the applied steering coefficient at inference. "
                        "Calibration polynomial is fit unconstrained, but the actual "
                        "applied alpha is min(poly(t), alpha_cap). Lower this when a "
                        "backbone is alpha-sensitive, e.g. --alpha_cap 1.5 for Mistral.")
    args = p.parse_args()

    # analyze mode: no model load needed
    if args.mode == "analyze":
        run_analysis(args)
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

    # Vectors (real, possibly replaced by random for the random_vec variant)
    real_vecs = load_vectors(args.vector_path)
    vecs = resolve_vectors(args.variant, real_vecs, seed=args.random_layer_seed)

    # Layers (oral, single-best, or random non-oral)
    oral_layers = parse_layer_range(args.oral_layers)
    all_layers = parse_layer_range(args.all_model_layers)
    target_layers = resolve_target_layers(
        args.variant, oral_layers, args.best_single_layer,
        all_layers, seed=args.random_layer_seed,
    )
    print(f"[{args.variant}] target_layers = {sorted(target_layers)}")

    manager = SteeringManager(model, vecs)

    if args.mode == "calibrate":
        run_calibration(args, model, tokenizer, manager, target_layers)
    elif args.mode == "predict":
        run_prediction(args, model, tokenizer, manager, target_layers)


if __name__ == "__main__":
    main()
