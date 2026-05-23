import argparse
import json
import os
import sys
from typing import List, Dict, Any, Tuple, Set

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import tqdm
from matplotlib.patches import Rectangle

# matplotlib optional
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# addeddefineneeds
# ==========================================
# FILLER_WORDS = ['um', 'uh', 'oh', 'uhm', 'mm', 'ah']
FILLER_WORDS = ['ok', 'so', 'okay', 'yeah', 
                'you know', 'well', 'really',
                'then', 'sure','now', 'alright', 'look', 'tell me',
                ]


def get_allowed_vocab(phrase_list: List[str]) -> Set[str]:
    """
    listset
    """
    vocab = set()
    for phrase in phrase_list:
        #
        words = phrase.split()
        for w in words:
            # can
            vocab.add(w.strip().lower())
    return vocab


def json_custom_serializer(obj):
    """JSON tool numpy value array"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


def extract_influence_matrices(influences: Dict[str, List[List[Dict[str, Any]]]],
                               layer_num: int,
                               head_num: int,
                               allowed_vocab: Set[str] = None) -> List[torch.Tensor]:
    """
    valuematrixlist
    """
    matrices = []
    for key, v in influences.items():
        # Key "Word_Index" "Did_0", "ok_5"
        if allowed_vocab is not None:
            if "_" in key:
                word_part = key.rsplit("_", 1)[0]
            else:
                word_part = key

            if word_part.lower() not in allowed_vocab:
                continue
            else:
                pass  # print(key)

        mat = torch.zeros(layer_num, head_num)
        for layer_idx in range(min(layer_num, len(v))):
            for head_idx in range(min(head_num, len(v[layer_idx]))):
                cell = v[layer_idx][head_idx]
                if cell:
                    mat[layer_idx][head_idx] = cell.get("influence", 0.0)
        matrices.append(mat)
    return matrices


def _standardize_for_plot(mat: torch.Tensor, mode: str) -> torch.Tensor:
    """can"""
    if mode == "none":
        return mat
    if mode == "global":
        scale = mat.abs().amax()
        return mat / (scale + 1e-12)
    if mode == "zscore-layer":
        mean = mat.mean(dim=1, keepdim=True)
        std = mat.std(dim=1, keepdim=True)
        return (mat - mean) / (std + 1e-12)
    return mat


# ==========================================
# addedZ-Score TopK function
# ==========================================
def get_topk_by_zscore(difference_tensor: torch.Tensor, sigma: float) -> Tuple[int, int, Dict]:
    """
     Z-Score (Mean +/- Sigma * Std) Head count
    return: (k_pos, k_neg, stats)
    """
    flat_diff = difference_tensor.flatten()
    mean_val = flat_diff.mean()
    std_val = flat_diff.std()

    threshold_pos = mean_val + sigma * std_val
    threshold_neg = mean_val - sigma * std_val

    # count
    # mean + sigma * std
    k_pos = (flat_diff > threshold_pos).sum().item()

    # mean - sigma * std
    k_neg = (flat_diff < threshold_neg).sum().item()

    stats = {
        "mean": mean_val.item(),
        "std": std_val.item(),
        "threshold_pos": threshold_pos.item(),
        "threshold_neg": threshold_neg.item(),
        "k_pos_found": k_pos,
        "k_neg_found": k_neg
    }

    # 1 data
    if flat_diff.numel() > 0:
        k_pos = max(1, k_pos) if k_pos > 0 else 0
        k_neg = max(1, k_neg) if k_neg > 0 else 0

    return int(k_pos), int(k_neg), stats


def auto_determine_topk_asymmetric(difference_tensor: torch.Tensor, ratio: float = 0.8) -> Tuple[int, int, Dict]:
    """
    and Top K
    """
    flat_diff = difference_tensor.flatten()
    stats = {"pos": {}, "neg": {}}

    # === 1. (Positive) ===
    pos_vals = flat_diff[flat_diff > 0]
    k_pos = 0
    if pos_vals.numel() > 0:
        sorted_pos, _ = torch.sort(pos_vals, descending=True) #
        cumsum_pos = torch.cumsum(sorted_pos, dim=0)
        total_pos = cumsum_pos[-1].item()

        indices = torch.where(cumsum_pos / (total_pos + 1e-12) >= ratio)[0]
        if len(indices) > 0:
            k_pos = int(indices[0].item()) + 1
        else:
            k_pos = pos_vals.numel()

        stats["pos"]["total_mass"] = total_pos
        stats["pos"]["covered_mass"] = cumsum_pos[k_pos - 1].item()
        stats["pos"]["values"] = sorted_pos[:k_pos].tolist()
    else:
        stats["pos"] = {"total_mass": 0, "covered_mass": 0, "values": []}

    # === 2. (Negative) ===
    neg_vals = flat_diff[flat_diff < 0].abs()
    k_neg = 0
    if neg_vals.numel() > 0:
        sorted_neg, _ = torch.sort(neg_vals, descending=True)
        cumsum_neg = torch.cumsum(sorted_neg, dim=0)
        total_neg = cumsum_neg[-1].item()

        indices = torch.where(cumsum_neg / (total_neg + 1e-12) >= ratio)[0]
        if len(indices) > 0:
            k_neg = int(indices[0].item()) + 1
        else:
            k_neg = neg_vals.numel()

        stats["neg"]["total_mass"] = total_neg
        stats["neg"]["covered_mass"] = cumsum_neg[k_neg - 1].item()
        stats["neg"]["values"] = sorted_neg[:k_neg].tolist()
    else:
        stats["neg"] = {"total_mass": 0, "covered_mass": 0, "values": []}

    k_pos = max(1, k_pos) if pos_vals.numel() > 0 else 0
    k_neg = max(1, k_neg) if neg_vals.numel() > 0 else 0

    return k_pos, k_neg, stats


def _draw_topk_boxes(ax, diff: torch.Tensor, head_num: int,
                     k_pos: int, k_neg: int,
                     color_pos="blue", color_neg="red", lw=1.5):
    """
     Top-K /
    """
    flat = diff.flatten()

    # Top-K max
    if k_pos > 0:
        real_k_pos = min(k_pos, flat.numel())
        _, flat_idx_pos = torch.topk(flat, k=real_k_pos, largest=True)
        for idx in flat_idx_pos.tolist():
            r = idx // head_num
            c = idx % head_num
            ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=color_pos, linewidth=lw))

    # Top-K min
    if k_neg > 0:
        real_k_neg = min(k_neg, flat.numel())
        _, flat_idx_neg = torch.topk(flat, k=real_k_neg, largest=False)
        for idx in flat_idx_neg.tolist():
            r = idx // head_num
            c = idx % head_num
            ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=color_neg, linewidth=lw))


def print_detailed_stats(stats, k_pos, k_neg, ratio):
    """print"""
    print("\n" + "=" * 60)
    print(f"  [Auto-TopK Analysis] Target Cumulative Ratio: {ratio * 100}%")
    print("=" * 60)

    p = stats["pos"]
    if p.get("total_mass", 0) > 0:
        actual_ratio = p["covered_mass"] / p["total_mass"] * 100
        print(f"\n>>> Positive Influence:")
        print(f"    - Total Influence Mass:   {p['total_mass']:.4f}")
        print(f"    - Selected Top K:         {k_pos}")
        print(f"    - Covered Mass:           {p['covered_mass']:.4f} ({actual_ratio:.2f}%)")
    else:
        print("\n>>> Positive Influence: None detected.")

    n = stats["neg"]
    if n.get("total_mass", 0) > 0:
        actual_ratio = n["covered_mass"] / n["total_mass"] * 100
        print(f"\n>>> Negative Influence:")
        print(f"    - Total Influence Mass:   {n['total_mass']:.4f}")
        print(f"    - Selected Top K:         {k_neg}")
        print(f"    - Covered Mass:           {n['covered_mass']:.4f} ({actual_ratio:.2f}%)")
    else:
        print("\n>>> Negative Influence: None detected.")
    print("=" * 60 + "\n")


def get_constrative_influence(args, layer_num: int, head_num: int):
    # path
    if not os.path.exists(args.output_path):
        print(f"Error: Output path does not exist: {args.output_path}")
        return

    #
    allowed_vocab = get_allowed_vocab(FILLER_WORDS)
    print(f"Focusing ONLY on interaction words vocabulary. Total unique tokens: {len(allowed_vocab)}")

    files = os.listdir(args.output_path)
    hallucination_samples = []
    non_hallucination_samples = []

    print("Loading samples...")
    for file in tqdm.tqdm(files[:]):
        if not file.endswith(".pth"):
            continue
        full_path = os.path.join(args.output_path, file)
        try:
            data = torch.load(full_path, map_location="cpu")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        if "non_hallucination" in file:
            matrices = extract_influence_matrices(data, layer_num, head_num, allowed_vocab=None)
        else:
            matrices = extract_influence_matrices(data, layer_num, head_num, allowed_vocab=allowed_vocab)

        if len(matrices) == 0:
            continue

        sample_mean = torch.stack(matrices).float().mean(0)

        if "non_hallucination" in file:
            non_hallucination_samples.append(sample_mean)
        elif "hallucination" in file:
            hallucination_samples.append(sample_mean)

    print(
        f"Loaded: {len(hallucination_samples)} hallucination samples, {len(non_hallucination_samples)} non-hallucination samples.")

    if not hallucination_samples or not non_hallucination_samples:
        print("Not enough data to compute contrastive heatmap.")
        return

    #
    hallucinated_scores = torch.stack(hallucination_samples).float().mean(0)
    non_hallucinated_scores = torch.stack(non_hallucination_samples).float().mean(0)

    # Log
    if args.use_log_pro:
        eps = 1e-12
        hallucinated_log_scores = torch.log(hallucinated_scores.clamp_min(eps))
        non_hallucinated_log_scores = torch.log(non_hallucinated_scores.clamp_min(eps))
        difference = hallucinated_log_scores - non_hallucinated_log_scores
        use_log_pro = "logp"
    else:
        difference = hallucinated_scores - non_hallucinated_scores
        use_log_pro = "nologp"

    print(f"Difference Matrix Range: [{float(difference.min()):.6f}, {float(difference.max()):.6f}]")

    # ================= modify Top K =================
    # first: Sigma > Auto-TopK > Fixed TopK
    if args.sigma > 0:
        print(f"Mode: Z-Score Threshold (Sigma = {args.sigma})")
        k_pos, k_neg, stats = get_topk_by_zscore(difference, args.sigma)
        print(f"  -> Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"  -> Pos Threshold: {stats['threshold_pos']:.4f} (Count: {k_pos})")
        print(f"  -> Neg Threshold: {stats['threshold_neg']:.4f} (Count: {k_neg})")
        suffix_mode = f"sigma{args.sigma}"

    elif args.auto_topk:
        print(f"Mode: Auto-TopK (Cumulative Ratio = {args.ratio})")
        k_pos, k_neg, stats = auto_determine_topk_asymmetric(difference, ratio=args.ratio)
        print_detailed_stats(stats, k_pos, k_neg, args.ratio)
        suffix_mode = f"auto{int(args.ratio * 100)}"
    else:
        print(f"Mode: Fixed TopK ({args.topk})")
        k_pos = args.topk
        k_neg = args.topk
        suffix_mode = f"top{args.topk}"
    # ======================================================

    #
    diff_for_plot = _standardize_for_plot(difference, getattr(args, "standardize", "none"))
    diff_arr = diff_for_plot.cpu().numpy()
    vmax = float(np.nanmax(np.abs(diff_arr)))
    if vmax <= 1e-12: vmax = 1.0
    vmin = -vmax

    #
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        diff_arr,
        cmap="coolwarm",
        center=0.0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": f"Contrastive influence (H - NH), std={getattr(args, 'standardize', 'none')}"}
    )
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    ax.set_title(f"Analysis (Pos K={k_pos}, Neg K={k_neg}) [Repeat Words Only]")

    #
    _draw_topk_boxes(ax, diff_for_plot, head_num=head_num,
                     k_pos=k_pos, k_neg=k_neg,
                     color_pos="red", color_neg="blue", lw=1.5)

    suffix = f"v7_num{len(non_hallucination_samples)}_diff_{use_log_pro}_sigma{args.sigma}_std{args.standardize}_{suffix_mode}"

    save_path = f"{args.output_path}/constrastive_influences_{suffix}_interactionword.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Heatmap saved to: {save_path}")
    plt.close()

    #
    results = {}

    # Heads
    if k_pos > 0:
        _, flat_indices = torch.topk(difference.flatten(), k_pos, largest=True)
        results["hal_heads"] = [[int(idx // head_num), int(idx % head_num)] for idx in flat_indices]
    else:
        results["hal_heads"] = []

    # Heads
    if k_neg > 0:
        _, flat_indices = torch.topk(difference.flatten(), k_neg, largest=False)
        results["non_hal_heads"] = [[int(idx // head_num), int(idx % head_num)] for idx in flat_indices]
    else:
        results["non_hal_heads"] = []

    # data
    results["meta"] = {
        "mode": "zscore" if args.sigma > 0 else ("auto_asymmetric" if args.auto_topk else "fixed"),
        "sigma": args.sigma if args.sigma > 0 else None,
        "ratio": args.ratio if args.auto_topk and args.sigma <= 0 else None,
        "k_pos": k_pos,
        "k_neg": k_neg,
        "filter": "interaction_words_only"
    }

    json_path = f"{args.output_path}/attribution_result_{suffix}_interactionword.json"
    with open(json_path, "w") as file:
        json.dump(results, file, default=json_custom_serializer, indent=2)
    print(f"JSON results saved to: {json_path}")


def get_num(modelsize="1.5b"):
    configs = {
        "1.5b": (28, 12),
        "3b": (36, 16),
        "7b": (28, 28)
    }
    if modelsize in configs:
        return configs[modelsize]
    raise Exception(f"Unknown model size: {modelsize}")


def main(debug_args):
    parser = argparse.ArgumentParser(description="LLM Attention Head Attribution Analysis")

    # pathargs
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to the directory containing .pth influence files")

    # args
    parser.add_argument("--modelsize", type=str, default='3b', choices=['1.5b', '3b', '7b'])

    # log
    parser.add_argument("--use_log_pro", type=bool, default=False, help="use_log_pro.")

    # === modifyadded Sigma args ===
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="If > 0, use Z-Score (mean +/- sigma*std) to determine Top-K heads. Overrides auto_topk and topk.")

    # Top-K args
    parser.add_argument("--auto_topk", action="store_true",
                        help="If set, automatically calculate Top-K based on cumulative ratio.")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="Cumulative contribution ratio (0.0 - 1.0). Default 0.8.")

    # Top-K args
    parser.add_argument("--topk", type=int, default=30,
                        help="Fixed Top-K number if auto_topk is not enabled and sigma is 0.")

    # canargs
    parser.add_argument("--standardize", type=str, choices=["none", "global", "zscore-layer"],
                        default="zscore-layer", help="Standardization method for plotting only.")
    if debug_args:
        args = parser.parse_args(debug_args)
    else:
        args = parser.parse_args()

    print("Running with arguments:")
    print(json.dumps(vars(args), indent=2))
    print("-" * 30)

    print(f"Start Analysis for Model: {args.modelsize}")
    layer_num, head_num = get_num(args.modelsize)
    get_constrative_influence(args, layer_num, head_num)


if __name__ == "__main__":
    # example Sigma = 3.0 row
    # "--output-path", "./../results/ablation_1k_complex_v2/multi_gpu_qwen3b_1000/identify_attention_head",
    debug_args = [
        "--output-path","./../results/ablation_1k_complex_v2/multi_gpu_qwen3b_1000/identify_attention_head_repair",
        "--modelsize", "3b",
        "--use_log_pro", False,
        "--sigma", "2.0", # Sigmafirsthighest
        "--standardize", "none"
    ]

    main(debug_args)