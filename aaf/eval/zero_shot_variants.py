"""Track-B inner-loop variants for zero-shot adaptation (Chunk 3.6).

The six variants probe whether the zero-shot failure of Chunk 3.5 is fixable
purely by changing how we optimise z_star, without retraining the model:

    B1  baseline                 (8 obs, 2K iters, random init)            — control
    B2  more observations        (32 obs, 2K iters, random init)
    B3  longer adaptation        (8 obs, 10K iters, random init)
    B4  multi-restart            (8 obs, 2K iters, 10 random inits, keep best)
    B5  nearest-train init       (8 obs, 2K iters, init from nearest train room)
    B6  simplex of train latents (8 obs, 2K iters, z = softmax(logits) @ Z_train)

``variant_kwargs(variant)`` returns the kwargs dict consumed by
``aaf.eval.zero_shot.zero_shot_adapt``. ``SimplexLatent`` is the parameterised
module used by B6.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplexLatent(nn.Module):
    """Latent parameterised as a convex combination of trained room latents.

    z_star = softmax(logits) @ Z_train   ∈ ℝ^{latent_dim}

    Z_train is fixed (registered as a buffer); ``logits`` is the optimisation
    parameter (one scalar per training room). At init, all logits are zero so
    z_star starts at the uniform mean of training latents — a neutral interior
    point of the convex hull.
    """

    def __init__(self, Z_train: torch.Tensor):
        super().__init__()
        if Z_train.dim() != 2:
            raise ValueError(f"Z_train must be 2-D [n_train, latent_dim], got {Z_train.shape}")
        self.register_buffer("Z_train", Z_train.detach().clone())
        # logits MUST live on the same device as Z_train, otherwise the forward
        # `w @ self.Z_train` raises (softmax(logits) ends up on whichever device
        # logits is on, and that has to match Z_train).
        self.logits = nn.Parameter(
            torch.zeros(Z_train.size(0), dtype=Z_train.dtype, device=Z_train.device)
        )

    @property
    def latent_dim(self) -> int:
        return int(self.Z_train.size(1))

    @property
    def n_train(self) -> int:
        return int(self.Z_train.size(0))

    def weights(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=0)

    def forward(self) -> torch.Tensor:
        w = self.weights()                            # [n_train]
        return w @ self.Z_train                       # [latent_dim]


def variant_kwargs(variant: str) -> dict[str, Any]:
    """Map a variant ID (B1-B6) to the kwargs consumed by ``zero_shot_adapt``.

    The base config (B1) matches the existing Chunk-3 zero-shot behaviour
    exactly; the others change one knob each.
    """
    base = dict(
        n_obs_receivers=8,
        n_adapt_iters=2000,
        lr=1e-2,
        init_strategy="random",
        n_restarts=1,
        random_seed=0,
    )
    if variant == "B1":
        return base
    if variant == "B2":
        return {**base, "n_obs_receivers": 32}
    if variant == "B3":
        return {**base, "n_adapt_iters": 10000}
    if variant == "B4":
        # n_restarts=5 (not 10) because the 90-min scavenger wall fits ~5×10min
        # restarts comfortably; 10 restarts would TIMEOUT and discard ALL work
        # (the function writes the winner only after the full restart loop).
        # 5 is still a meaningful test of multi-basin behaviour.
        return {**base, "n_restarts": 5}
    if variant == "B5":
        return {**base, "init_strategy": "nearest_train"}
    if variant == "B6":
        return {**base, "init_strategy": "simplex"}
    raise ValueError(f"unknown variant {variant!r}; must be one of B1..B6")


VARIANT_DESCRIPTIONS: dict[str, str] = {
    "B1": "baseline (8 obs, 2K iters, random init)",
    "B2": "32 observed receivers (was 8)",
    "B3": "10K inner iters (was 2K)",
    "B4": "10 random restarts, keep best obs LSD",
    "B5": "init z_star from nearest-L training latent",
    "B6": "z_star = softmax(logits) @ Z_train (simplex)",
}

ALL_VARIANTS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5", "B6")
