# -*- coding: utf-8 -*-
"""
Unified controller for LAS Table-4 ablations on Qwen-2.5-3B-Instruct.

Methods (replace LAS's element-wise affine with a different parameterization
at the SAME hook locations):
    las            : delta = D*h + B                          (baseline, ours)
    las_no_scale   : delta = B                                (-Scale: bias only)
    las_no_bias    : delta = D*h                              (-Bias:  scale only)
    lora           : delta = (h @ A^T) @ B^T * (alpha/r)      (rank-4 LoRA)
    ia3            : delta = (s - 1) * h                      (scale-only, init s=1)
    reft           : delta = R^T (W h + b - R h)              (LoReFT, rank-2)

Target sets (which components to intervene on):
    oral_lh        : oral layers + oral heads (LAS default)
    random_lh      : matched-count random NON-oral layers/heads (Rnd L+H)
    oral_l_only    : oral layers only (-Heads ablation; no o_proj hook)
    oral_h_only    : oral heads only (-Layers ablation; no residual hook)

Hooks share infrastructure with train_affine_reft.py:
    * o_proj forward hook for head-level intervention (with head_dim mask)
    * decoder-layer forward hook for residual stream intervention
"""

import os
import json
import logging
import math
import random
from typing import Optional, Set, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# helpers
# =============================================================================
def load_oral_heads(head_path: Optional[str]) -> Set[Tuple[int, int]]:
    if not head_path or not os.path.exists(head_path):
        return set()
    with open(head_path, "r") as f:
        data = json.load(f)
    heads = set()
    for item in data.get("hal_heads", []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            heads.add((int(item[0]), int(item[1])))
    return heads


def parse_layers_str(s: Optional[str]) -> Set[int]:
    layers: Set[int] = set()
    if s and s.strip():
        for x in s.split(","):
            x = x.strip()
            if x.isdigit():
                layers.add(int(x))
    return layers


def sample_random_layers(oral_layers: Set[int], num_layers: int, seed: int) -> Set[int]:
    """Match the count of oral_layers from NON-oral layers.
    If non-oral is insufficient, supplement from oral (so count is preserved)."""
    rng = random.Random(seed)
    n_target = len(oral_layers)
    if n_target == 0:
        return set()
    all_layers = list(range(num_layers))
    non_oral = [l for l in all_layers if l not in oral_layers]
    if len(non_oral) >= n_target:
        return set(rng.sample(non_oral, n_target))
    chosen = list(non_oral)
    remaining = n_target - len(non_oral)
    chosen.extend(rng.sample(sorted(oral_layers), min(remaining, len(oral_layers))))
    return set(chosen)


def sample_random_heads(
    oral_heads: Set[Tuple[int, int]], num_layers: int, num_heads: int, seed: int
) -> Set[Tuple[int, int]]:
    """Match the count of oral_heads from NON-oral (layer, head) pairs.
    If non-oral is insufficient, supplement from oral pairs."""
    rng = random.Random(seed + 1)
    n_target = len(oral_heads)
    if n_target == 0:
        return set()
    all_pairs = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    oral_set = set((int(a), int(b)) for a, b in oral_heads)
    non_oral = [p for p in all_pairs if p not in oral_set]
    if len(non_oral) >= n_target:
        return set(rng.sample(non_oral, n_target))
    chosen = list(non_oral)
    remaining = n_target - len(non_oral)
    chosen.extend(rng.sample(sorted(oral_set), min(remaining, len(oral_set))))
    return set(chosen)


# =============================================================================
# controller
# =============================================================================
class AblationController(nn.Module):
    SUPPORTED_METHODS = {"las", "las_no_scale", "las_no_bias", "lora", "ia3", "reft"}
    SUPPORTED_TARGETS = {"oral_lh", "random_lh", "oral_l_only", "oral_h_only"}

    def __init__(
        self,
        model: nn.Module,
        method: str,
        target_set: str,
        oral_layers_str: str,
        head_path: Optional[str],
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        reft_r: int = 2,
        reft_zero_init: bool = False,
        rnd_seed: int = 42,
    ):
        super().__init__()
        assert method in self.SUPPORTED_METHODS, f"Unknown method: {method}"
        assert target_set in self.SUPPORTED_TARGETS, f"Unknown target_set: {target_set}"

        self.method = method
        self.target_set = target_set
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.reft_r = int(reft_r)
        # When True, initialize ReFT's learned W to a copy of the orthonormal R
        # so the start-of-training delta = R^T (Wh + b - Rh) = R^T(Rh + 0 - Rh)
        # is exactly zero. This matches the "delta = 0 at init" convention used
        # by LAS (D=0, B=0), LoRA (B=0), and IA^3 (s=1); see ReFT-ablation audit.
        self.reft_zero_init = bool(reft_zero_init)
        self.rnd_seed = int(rnd_seed)

        # Parameters / submodules
        self.params_dict = nn.ParameterDict()
        self.reft_R = nn.ModuleDict()  # holds orthogonally-parametrized Linear(d, r)
        self.hook_handles: List[Any] = []

        # Optional dropout (only meaningful for LoRA; kept zero by default per main exp)
        self.lora_drop = nn.Dropout(p=self.lora_dropout) if self.lora_dropout > 0 else nn.Identity()

        # Model topology
        config = model.config
        num_layers = getattr(config, "num_hidden_layers", 32)
        num_q_heads = getattr(config, "num_attention_heads", 32)
        hidden_size = getattr(config, "hidden_size", 4096)
        head_dim = hidden_size // num_q_heads

        oral_layers = parse_layers_str(oral_layers_str)
        oral_heads = load_oral_heads(head_path)

        if target_set == "oral_lh":
            self.target_res_layers = set(oral_layers)
            self.target_heads = set(oral_heads)
        elif target_set == "random_lh":
            self.target_res_layers = sample_random_layers(oral_layers, num_layers, self.rnd_seed)
            self.target_heads = sample_random_heads(oral_heads, num_layers, num_q_heads, self.rnd_seed)
        elif target_set == "oral_l_only":  # -Heads
            self.target_res_layers = set(oral_layers)
            self.target_heads = set()
        elif target_set == "oral_h_only":  # -Layers
            self.target_res_layers = set()
            self.target_heads = set(oral_heads)
        else:
            raise ValueError(target_set)

        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim

        logger.info(
            f"[AblationController] method={method} | target_set={target_set} | "
            f"hidden={hidden_size} | layers={num_layers} | q_heads/layer={num_q_heads}"
        )
        logger.info(f"  target_res_layers = {sorted(self.target_res_layers)}")
        logger.info(f"  target_heads ({len(self.target_heads)}) = {sorted(self.target_heads)}")

        # Register params & hooks per layer
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            # ---- head-level: o_proj output ----
            if hasattr(layer.self_attn, "o_proj"):
                mask = torch.zeros(hidden_size)
                active = 0
                for h_q in range(num_q_heads):
                    if (layer_idx, h_q) in self.target_heads:
                        mask[h_q * head_dim : (h_q + 1) * head_dim] = 1.0
                        active += 1
                if active > 0:
                    key = f"L{layer_idx}_o_proj"
                    self._init_params(key)
                    self.register_buffer(f"mask_{key}", mask)
                    self._register_hook(layer.self_attn.o_proj, key, is_layer=False)

            # ---- layer-level: decoder-layer output ----
            if layer_idx in self.target_res_layers:
                key = f"L{layer_idx}_residual"
                self._init_params(key)
                self._register_hook(layer, key, is_layer=True)

    # -------------------------------------------------------------------------
    # param init dispatched by method
    # -------------------------------------------------------------------------
    def _init_params(self, key: str):
        d = self.hidden_size
        m = self.method
        if m == "las":
            self.params_dict[f"{key}__D"] = nn.Parameter(torch.zeros(d))
            self.params_dict[f"{key}__B"] = nn.Parameter(torch.zeros(d))
        elif m == "las_no_scale":
            self.params_dict[f"{key}__B"] = nn.Parameter(torch.zeros(d))
        elif m == "las_no_bias":
            self.params_dict[f"{key}__D"] = nn.Parameter(torch.zeros(d))
        elif m == "lora":
            r = self.lora_r
            A = torch.empty(r, d)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))  # standard LoRA init for A
            self.params_dict[f"{key}__loraA"] = nn.Parameter(A)
            self.params_dict[f"{key}__loraB"] = nn.Parameter(torch.zeros(d, r))  # B = 0 -> delta=0 at init
        elif m == "ia3":
            # IA3 multiplicative scaling, init=1 so delta=0 at start
            self.params_dict[f"{key}__ia3"] = nn.Parameter(torch.ones(d))
        elif m == "reft":
            r = self.reft_r
            # R: orthonormal projection, parametrized via torch.nn.utils.parametrizations.orthogonal
            lin = nn.Linear(d, r, bias=False)
            torch.nn.utils.parametrizations.orthogonal(lin)
            self.reft_R[key] = lin
            # W: full linear with two init modes.
            #   xavier_uniform_ (default, legacy): non-zero delta at init — the
            #     first epochs are spent un-doing a random perturbation rather
            #     than learning the task.
            #   W := R.weight (reft_zero_init=True, recommended for r=4 runs):
            #     gives delta = R^T(Wh + b - Rh) = 0 at step 0, matching how
            #     LAS / LoRA / IA^3 start from the unperturbed base model.
            if self.reft_zero_init:
                # lin.weight reads through the orthogonal parametrization, so
                # this is the actual orthonormal r×d matrix used at forward.
                W = lin.weight.detach().clone().to(torch.float32)
            else:
                W = torch.empty(r, d)
                nn.init.xavier_uniform_(W)
            self.params_dict[f"{key}__reftW"] = nn.Parameter(W)
            self.params_dict[f"{key}__reftb"] = nn.Parameter(torch.zeros(r))
        else:
            raise ValueError(m)

    # -------------------------------------------------------------------------
    # delta computation dispatched by method
    # -------------------------------------------------------------------------
    def _compute_delta(self, key: str, h: torch.Tensor) -> torch.Tensor:
        """Compute additive delta of shape h.shape; caller multiplies by mask if needed."""
        dev = h.device
        dt = h.dtype
        m = self.method

        if m == "las":
            D = self.params_dict[f"{key}__D"].to(device=dev, dtype=dt)
            B = self.params_dict[f"{key}__B"].to(device=dev, dtype=dt)
            return h * D + B

        if m == "las_no_scale":
            B = self.params_dict[f"{key}__B"].to(device=dev, dtype=dt)
            return B.expand_as(h)

        if m == "las_no_bias":
            D = self.params_dict[f"{key}__D"].to(device=dev, dtype=dt)
            return h * D

        if m == "lora":
            A = self.params_dict[f"{key}__loraA"].to(device=dev, dtype=dt)  # [r, d]
            B = self.params_dict[f"{key}__loraB"].to(device=dev, dtype=dt)  # [d, r]
            scaling = self.lora_alpha / self.lora_r
            h_in = self.lora_drop(h)
            tmp = F.linear(h_in, A)       # h_in @ A^T : [..., r]
            return F.linear(tmp, B) * scaling  # tmp @ B^T : [..., d]

        if m == "ia3":
            s = self.params_dict[f"{key}__ia3"].to(device=dev, dtype=dt)  # init=1
            return h * (s - 1.0)  # so h + delta = s * h

        if m == "reft":
            R_mod = self.reft_R[key]
            R_w = R_mod.weight.to(device=dev, dtype=dt)         # [r, d], orthonormal rows
            W = self.params_dict[f"{key}__reftW"].to(device=dev, dtype=dt)  # [r, d]
            b = self.params_dict[f"{key}__reftb"].to(device=dev, dtype=dt)  # [r]
            Rh = F.linear(h, R_w)            # [..., r]
            Wh = F.linear(h, W) + b          # [..., r]
            diff = Wh - Rh                   # [..., r]
            return diff @ R_w                # [..., d]  (== R^T diff for row-orthonormal R)

        raise ValueError(m)

    # -------------------------------------------------------------------------
    # forward-hook registration
    # -------------------------------------------------------------------------
    def _register_hook(self, module: nn.Module, key: str, is_layer: bool):
        if is_layer:
            def hook_fn(mod, args, output):
                hidden = output[0]
                delta = self._compute_delta(key, hidden)
                return (hidden + delta,) + output[1:]
        else:
            def hook_fn(mod, args, output):
                mask = getattr(self, f"mask_{key}").to(device=output.device, dtype=output.dtype)
                delta = self._compute_delta(key, output) * mask
                return output + delta
        self.hook_handles.append(module.register_forward_hook(hook_fn))

    # -------------------------------------------------------------------------
    # checkpointing helpers
    # -------------------------------------------------------------------------
    def export_config(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "target_set": self.target_set,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "head_dim": self.head_dim,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "reft_r": self.reft_r,
            "reft_zero_init": self.reft_zero_init,
            "rnd_seed": self.rnd_seed,
            "target_res_layers": sorted(self.target_res_layers),
            "target_heads": sorted([list(p) for p in self.target_heads]),
        }

    def trainable_param_count(self) -> int:
        n = 0
        for p in self.parameters():
            if p.requires_grad:
                n += p.numel()
        return n

    def keep_reft_R_fp32(self):
        """Keep the orthogonal-parametrized R modules in fp32.

        Rationale: torch.nn.utils.parametrizations.orthogonal uses
        torch.linalg.householder_product internally, which raises
            "tau dtype Float does not match input dtype BFloat16"
        if the underlying weight is cast to bf16 (the tau buffer is precomputed
        and stays fp32). Keeping R in fp32 sidesteps this and also improves
        numerical stability of the orthogonal projection. _compute_delta()
        already casts R.weight to the activation dtype at call time, so the
        rest of the bf16 forward path is unaffected.

        Call this AFTER `controller.to(dtype=torch.bfloat16)`.
        """
        for m in self.reft_R.values():
            m.float()
