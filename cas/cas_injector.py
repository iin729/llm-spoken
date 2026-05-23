import os
import sys
import json
import torch
import math
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add repo root to sys.path so `from orality import ...` works regardless of cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ==============================================================================
# 1. PROMPT TEMPLATE (dataset Example)
# ==============================================================================

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

# Orality estimator g(.). Wrapped in try/except so the script can still load
# the model and run prediction even if optional eval deps are missing
# (calibration / level analysis will then be skipped).
try:
    from orality import eval_oscore_and_level, stat_all_metrics
except ImportError:
    print("[Warning] orality package not importable. Calibration analysis will be skipped.")


# ==============================================================================
# 2. Steering Manager
# ==============================================================================
class SteeringManager:
    def __init__(self, model, vectors):
        self.model = model
        self.vectors = vectors
        self.target_layers = []
        self.coeff = 0.0
        self.hooks = []
        self.use_dynamic_gating = False

    def set_config(self, layers, coeff, use_dynamic_gating=False):
        self.remove_hooks()
        self.target_layers = layers
        self.coeff = coeff
        self.use_dynamic_gating = use_dynamic_gating
        if len(layers) > 0 and coeff != 0:
            self.register_hooks()

    def register_hooks(self):
        for layer_idx in self.target_layers:
            if layer_idx not in self.vectors: continue
            layer_module = self.model.model.layers[layer_idx]

            def hook_fn(module, args, output, idx=layer_idx):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                # vectormove to
                vec = self.vectors[idx].to(hidden.device).to(hidden.dtype)

                if self.use_dynamic_gating:
                    # : ReLU(cos_sim) * coeff
                    sim = torch.nn.functional.cosine_similarity(hidden, vec.view(1, 1, -1), dim=-1).unsqueeze(-1)
                    dynamic_scale = torch.nn.functional.relu(sim)
                    intervention = self.coeff * dynamic_scale * vec.view(1, 1, -1)
                    hidden.add_(intervention)
                else:
                    # note
                    hidden.add_(self.coeff * vec.view(1, 1, -1))
                return output

            self.hooks.append(layer_module.register_forward_hook(hook_fn))

    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []


# ==============================================================================
# 3. Helpers
# ==============================================================================
def build_prompt_messages(written_text):
    """
     Prompt
     Level Level 0vector (Level N - Level 0) row
    """
    user_content = f"**Written Language Level 0:**\n{written_text}\n\n**Spoken Language Level 0:**\n"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": user_content}
    ]
    return messages, user_content


def parse_layers(s):
    if not s: return []
    s = str(s).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip() != ""]
    return [int(s)]


def load_vectors(path):
    print(f"Loading vectors from: {path}")
    raw_vectors = torch.load(path)
    processed_vectors = {}
    for k, v in raw_vectors.items():
        try:
            if isinstance(k, int):
                idx = k
            elif isinstance(k, str):
                idx = int(k.replace("layer_", "").replace("layers.", ""))
            else:
                continue
            vec = v.float()
            #
            vec = vec / (vec.norm() + 1e-6)
            processed_vectors[idx] = vec
        except:
            continue
    return processed_vectors


def calculate_level_only(text):
    """ Calibration stageevaluationgenerateetc."""
    stats = stat_all_metrics(str(text))
    if stats:
        _, level = eval_oscore_and_level(stats)
        return int(level)



# ==============================================================================
# 4. Modes (Calibration, Analysis, Prediction)
# ==============================================================================

def run_calibration(args, model, tokenizer, manager):
    print("\n>>> Mode: Calibration")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)[:100] # 100row

    prompts = []
    original_outputs = []
    for item in val_data:
        msgs, _ = build_prompt_messages(item['written'])
        # must apply_chat_template System Prompt
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        original_outputs.append(item['output'])

    results = []
    target_layers = parse_layers(args.selected_layers)

    #
    coeffs_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    print(f"Sweeping coeffs: {coeffs_to_test} on Layers {args.selected_layers}")

    # OOMMistral etc.mustbatch
    BATCH_SIZE = 8

    for coeff in tqdm(coeffs_to_test, desc="Coeff Sweep"):
        manager.set_config(layers=target_layers, coeff=coeff, use_dynamic_gating=False)

        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i: i + BATCH_SIZE]
            batch_labels = original_outputs[i: i + BATCH_SIZE]

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            input_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id)

            predictions = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

            for j, pred in enumerate(predictions):
                results.append({
                    "experiment": "coeff_sweep",
                    "coeff": coeff,
                    "output": batch_labels[j],
                    "prediction": pred
                })

    manager.remove_hooks()
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(results).to_excel(os.path.join(args.output_dir, "experiment_results.xlsx"), index=False)


def run_analysis(args):
    print("\n>>> Mode: Analysis (Polynomial Fitting)")
    input_file = os.path.join(args.output_dir, "experiment_results.xlsx")
    df = pd.read_excel(input_file)
    coeff_df = df[df['experiment'] == 'coeff_sweep'].copy()

    # Level
    tqdm.pandas(desc="Calc Levels")
    coeff_df['Level_pred'] = coeff_df['prediction'].progress_apply(calculate_level_only)

    # Level
    agg = coeff_df.groupby('coeff')['Level_pred'].mean().reset_index()
    print("\nCalibration Data:\n", agg)

    best_coeffs = []
    # etc. (1-5)
    for tgt_lvl in [1, 2, 3, 4, 5]:
        agg['diff'] = abs(agg['Level_pred'] - tgt_lvl)
        # diff minrow
        best_row = agg.sort_values('diff').iloc[0]
        best_coeffs.append({'Target_Level': tgt_lvl, 'Best_Coeff': best_row['coeff']})

    fit_data = pd.DataFrame(best_coeffs)

    # Level 0 -> Coeff 0
    X = np.array([0] + fit_data['Target_Level'].tolist())
    Y = np.array([0] + fit_data['Best_Coeff'].tolist())

    # modify 2 (Quadratic Fit): y = p2*x^2 + p1*x + p0
    z = np.polyfit(X, Y, 2)
    p2, p1, p0 = z[0], z[1], z[2]

    print(f"Fitted Polynomial: y = {p2:.4f}x^2 + {p1:.4f}x + {p0:.4f}")

    # saveargs
    with open(os.path.join(args.output_dir, "coeff_function_params.json"), 'w') as f:
        json.dump({"p2": p2, "p1": p1, "p0": p0, "type": "poly2"}, f, indent=2)


def run_prediction(args, model, tokenizer, manager):
    print(f"\n>>> Mode: Prediction (Chunk {args.chunk_idx})")

    # 1. args
    # default (iffile)
    params = {"p2": 0.0, "p1": 0.5, "p0": 0.0, "type": "poly2"}

    params_path = os.path.join(args.output_dir, "coeff_function_params.json")
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)

    print(f"Loaded Params: {params}")

    # 2. load data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    chunk_size = math.ceil(len(all_data) / args.num_chunks)
    start = args.chunk_idx * chunk_size
    end = min(start + chunk_size, len(all_data))
    test_data = all_data[start: end]

    results = []
    target_layers = parse_layers(args.selected_layers)

    for item in tqdm(test_data):
        target_lvl = item.get('level', 3)
        if 'spoken_statics' in item: target_lvl = item['spoken_statics'].get('level', target_lvl)

        #
        if args.force_coeff is not None:
            # ifforce ( Llama-1B force 0.35)
            coeff = args.force_coeff
        else:
            #
            if params.get("type") == "poly2":
                p2, p1, p0 = params['p2'], params['p1'], params['p0']
                coeff = p2 * (target_lvl ** 2) + p1 * target_lvl + p0
            else:
                # args (slope, intercept)
                slope = params.get('slope', 0.5)
                intercept = params.get('intercept', 0.0)
                coeff = slope * target_lvl + intercept

        #
        coeff = max(0.0, min(coeff, 4.0))

        #
        manager.set_config(layers=target_layers, coeff=coeff, use_dynamic_gating=args.use_dynamic_gating)

        msgs, _ = build_prompt_messages(item['written'])
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        pred = tokenizer.decode(outputs[:, input_len:][0], skip_special_tokens=True)
        results.append({
            "instruction": text, "predict": pred, "label": item['output'],
            "target_level": target_lvl, "applied_coeff": coeff
        })

    manager.remove_hooks()
    with open(f"{args.prediction_file}.part{args.chunk_idx}", 'w', encoding='utf-8') as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ==============================================================================
# 5. Main Entry
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["calibrate", "analyze", "predict"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vector_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--prediction_file", type=str, default="predictions.jsonl")
    parser.add_argument("--selected_layers", type=str, default="12-28")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)

    # addedargs
    parser.add_argument("--use_dynamic_gating", action="store_true", help="Enable dynamic gating")
    parser.add_argument("--force_coeff", type=float, default=None, help="Force a fixed coefficient (overrides fitting)")

    args = parser.parse_args()

    # Analyze needsLoad model
    if args.mode == "analyze":
        run_analysis(args)
        return

    print(f"Loading Model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
    vectors = load_vectors(args.vector_path)
    manager = SteeringManager(model, vectors)

    if args.mode == "calibrate":
        run_calibration(args, model, tokenizer, manager)
    elif args.mode == "predict":
        run_prediction(args, model, tokenizer, manager)


if __name__ == "__main__":
    main()