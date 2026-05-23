import argparse
import json
import os
from typing import List, Dict, Any
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import tqdm
from matplotlib.patches import Rectangle


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
                               head_num: int) -> List[torch.Tensor]:
    """
    dict token Layer x Head listtensorlist
     influence valueshape: (layer_num, head_num)
    first 0 minmodify
    """
    matrices = []
    for _, v in influences.items():
        mat = torch.zeros(layer_num, head_num)
        for layer_idx in range(min(layer_num, len(v))):
            for head_idx in range(min(head_num, len(v[layer_idx]))):
                cell = v[layer_idx][head_idx]
                if cell:
                    mat[layer_idx][head_idx] = cell.get("influence", 0.0)
        matrices.append(mat)
    return matrices


def _standardize_for_plot(mat: torch.Tensor, mode: str) -> torch.Tensor:
    """
    can
    mode:
      - none:
      - global: max
      - zscore-layer: (z-score)
    """
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


def _draw_topk_boxes(ax, diff: torch.Tensor, head_num: int, topk: int,
                     color_pos="blue", color_neg="red", lw=1.5):
    """
     Top-K /
     difference hal_heads/non_hal_heads
    """
    flat = diff.flatten()

    # Top-K max“/”
    k_pos = min(topk, flat.numel())
    _, flat_idx_pos = torch.topk(flat, k=k_pos, largest=True)
    for idx in flat_idx_pos.tolist():
        r = idx // head_num
        c = idx % head_num
        ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=color_pos, linewidth=lw))

    # Top-K min“/”
    k_neg = min(topk, flat.numel())
    _, flat_idx_neg = torch.topk(flat, k=k_neg, largest=False)
    for idx in flat_idx_neg.tolist():
        r = idx // head_num
        c = idx % head_num
        ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=color_neg, linewidth=lw))


def get_constrative_influence(args, layer_num: int, head_num: int):
    """
    
    - vs (constrastive_influences.png)
    - max topk (hal_heads) min topk (non_hal_heads)

    already
    1) vmin=-vmax, vmax=vmax+ coolwarm + center=0
    2) first“”“”
    3) optional--standardize: none/global/zscore-layer
    4) Top-K ==
    """
    files = os.listdir(args.output_path)
    hallucination_samples = []
    non_hallucination_samples = []

    for file in tqdm.tqdm(files):
        if not file.endswith(".pth"):
            continue
        full_path = os.path.join(args.output_path, file)
        data = torch.load(full_path, map_location="cpu")

        # file token (L,H) matrix
        matrices = extract_influence_matrices(data, layer_num, head_num)
        if len(matrices) == 0:
            continue

        # 2
        sample_mean = torch.stack(matrices).float().mean(0)

        if file.startswith("hallucination"):
            hallucination_samples.append(sample_mean)
        elif file.startswith("non_hallucination"):
            non_hallucination_samples.append(sample_mean)

    print(f"len(hallucination_samples):{len(hallucination_samples)}")
    print(f"len(non_hallucination_samples):{len(non_hallucination_samples)}")

    if not hallucination_samples or not non_hallucination_samples:
        print("Not enough influence data to compute contrastive heatmap.")
        return

    #
    hallucinated_scores = torch.stack(hallucination_samples).float().mean(0)
    non_hallucinated_scores = torch.stack(non_hallucination_samples).float().mean(0)

    # H - NH
    difference = hallucinated_scores - non_hallucinated_scores
    print(f"Difference min/max: {float(difference.min()):.6f} / {float(difference.max()):.6f}")

    # 3optional
    diff_for_plot = _standardize_for_plot(difference, getattr(args, "standardize", "none"))

    # 1
    diff_arr = diff_for_plot.cpu().numpy()
    vmax = float(np.nanmax(np.abs(diff_arr)))
    if vmax <= 1e-12:
        vmax = 1.0 # 0 exception
    vmin = -vmax

    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        diff_arr,
        cmap="coolwarm",
        center=0.0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": f"Contrastive influence (H - NH), standardize={getattr(args, 'standardize', 'none')}"}
    )
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)

    # 4 Top-K difference
    # _draw_topk_boxes(ax, difference, head_num=head_num, topk=args.topk,
    #                  color_pos="red", color_neg="blue", lw=1.5)
    _draw_topk_boxes(ax, diff_for_plot, head_num=head_num, topk=args.topk,
                     color_pos="red", color_neg="blue", lw=1.5)

    plt.tight_layout()
    plt.savefig(f"{args.output_path}/constrastive_influences_v3_zscore-layer.png")
    plt.close()

    # Top-K
    results = {}
    _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=True)
    indices = [[int(idx // head_num), int(idx % head_num)] for idx in flat_indices]
    results["hal_heads"] = indices

    _, flat_indices = torch.topk(difference.flatten(), args.topk, largest=False)
    indices = [[int(idx // head_num), int(idx % head_num)] for idx in flat_indices]
    results["non_hal_heads"] = indices

    with open(f"{args.output_path}/attribution_result_v3_zscore-layer.json", "w") as file:
        json.dump(results, file, default=json_custom_serializer)

    # print 5 /
    hal_preview = results["hal_heads"][:5]
    non_hal_preview = results["non_hal_heads"][:5]
    print("Top-5 (layer, head):", hal_preview)
    print("Top-5 / (layer, head):", non_hal_preview)


def get_num(modelsize="1.5b"):
    qwen_1d5b = {
      "architectures": [
        "Qwen2ForCausalLM"
      ],
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151645,
      "hidden_act": "silu",
      "hidden_size": 1536,
      "initializer_range": 0.02,
      "intermediate_size": 8960,
      "max_position_embeddings": 32768,
      "max_window_layers": 21,
      "model_type": "qwen2",
      "num_attention_heads": 12,
      "num_hidden_layers": 28,
      "num_key_value_heads": 2,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000.0,
      "sliding_window": 32768,
      "tie_word_embeddings": True,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.43.1",
      "use_cache": True,
      "use_sliding_window": False,
      "vocab_size": 151936
    }
    qwen_3b = {
        "architectures": [
            "Qwen2ForCausalLM"
        ],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 32768,
        "max_window_layers": 70,
        "model_type": "qwen2",
        "num_attention_heads": 16,
        "num_hidden_layers": 36,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "sliding_window": 32768,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.43.1",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 151936
    }
    qwen_7b = {
      "architectures": [
        "Qwen2ForCausalLM"
      ],
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151645,
      "hidden_act": "silu",
      "hidden_size": 3584,
      "initializer_range": 0.02,
      "intermediate_size": 18944,
      "max_position_embeddings": 32768,
      "max_window_layers": 28,
      "model_type": "qwen2",
      "num_attention_heads": 28,
      "num_hidden_layers": 28,
      "num_key_value_heads": 4,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000.0,
      "sliding_window": 131072,
      "tie_word_embeddings": False,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.43.1",
      "use_cache": True,
      "use_sliding_window": False,
      "vocab_size": 152064
    }

    if modelsize == '1.5b':
        return qwen_1d5b.get("num_hidden_layers"), qwen_1d5b.get("num_attention_heads")
    if modelsize == '3b':
        return qwen_3b.get("num_hidden_layers"), qwen_3b.get("num_attention_heads")
    if modelsize == '7b':
        return qwen_7b.get("num_hidden_layers"), qwen_7b.get("num_attention_heads")
    raise Exception('')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default="./../results/train_sft/qwen3b_1500/identify_attention_head")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--modelsize", type=str, default='3b')
    # addedoptionalcan
    parser.add_argument("--standardize", type=str, choices=["none", "global", "zscore-layer"],
                        default="zscore-layer", help="")
    args = parser.parse_args()
    layer_num, head_num = get_num(args.modelsize)
    get_constrative_influence(args, layer_num, head_num)


if __name__ == "__main__":
    main()
