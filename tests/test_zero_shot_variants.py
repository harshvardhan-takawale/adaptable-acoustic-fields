"""Inner-loop variants (Chunk 3.6 Track B): SimplexLatent + variant_kwargs."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aaf.eval.zero_shot_variants import (
    ALL_VARIANTS,
    SimplexLatent,
    VARIANT_DESCRIPTIONS,
    variant_kwargs,
)


def test_simplex_latent_outputs_convex_combination():
    n_train, d = 7, 8
    Z = torch.randn(n_train, d)
    sx = SimplexLatent(Z)
    z = sx()
    assert z.shape == (d,)
    w = sx.weights().detach()
    assert w.shape == (n_train,)
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)


def test_simplex_init_at_uniform_mean():
    n_train, d = 5, 4
    Z = torch.arange(n_train * d, dtype=torch.float32).reshape(n_train, d)
    sx = SimplexLatent(Z)
    z = sx().detach()
    expected = Z.mean(dim=0)
    assert torch.allclose(z, expected, atol=1e-5), f"got {z}, expected {expected}"


def test_simplex_gradient_flows_to_logits():
    Z = torch.randn(7, 8)
    sx = SimplexLatent(Z)
    z = sx()
    loss = (z ** 2).sum()
    loss.backward()
    assert sx.logits.grad is not None
    assert torch.any(sx.logits.grad != 0)
    # Z_train is registered as a buffer, not a parameter -> no grad attr.
    assert not isinstance(sx.Z_train, torch.nn.Parameter)


def test_variant_kwargs_dispatch_table():
    for v in ALL_VARIANTS:
        kw = variant_kwargs(v)
        assert isinstance(kw, dict)
        # Must always provide all six fields zero_shot_adapt expects.
        for k in ("n_obs_receivers", "n_adapt_iters", "lr",
                  "init_strategy", "n_restarts", "random_seed"):
            assert k in kw, f"variant {v} missing {k}"
        assert v in VARIANT_DESCRIPTIONS

    # Spot-check the per-variant differences from baseline B1.
    base = variant_kwargs("B1")
    assert variant_kwargs("B2")["n_obs_receivers"] == 32
    assert variant_kwargs("B3")["n_adapt_iters"] == 10000
    assert variant_kwargs("B4")["n_restarts"] == 5
    assert variant_kwargs("B5")["init_strategy"] == "nearest_train"
    assert variant_kwargs("B6")["init_strategy"] == "simplex"

    # Everything else should match base for the single-knob variants.
    for v, k in [("B2", "n_obs_receivers"), ("B3", "n_adapt_iters"),
                 ("B4", "n_restarts"), ("B5", "init_strategy"),
                 ("B6", "init_strategy")]:
        for other in base:
            if other == k:
                continue
            assert variant_kwargs(v)[other] == base[other], \
                f"{v} unexpectedly changed {other}"


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        variant_kwargs("B99")


def test_select_obs_indices_n8_matches_chunk3():
    from aaf.eval.zero_shot import OBS_INDICES, select_obs_indices

    out = select_obs_indices(8, total=64)
    assert np.array_equal(out, OBS_INDICES)


def test_select_obs_indices_n32_is_checkerboard():
    from aaf.eval.zero_shot import select_obs_indices

    out = select_obs_indices(32, total=64)
    assert out.shape == (32,)
    # Checkerboard property: (iy + ix) is always even.
    for idx in out:
        iy, ix = idx // 8, idx % 8
        assert (iy + ix) % 2 == 0
    # Disjoint with the complement (32 white squares).
    complement = np.array([i for i in range(64) if i not in set(out.tolist())])
    assert complement.shape == (32,)
    for idx in complement:
        iy, ix = idx // 8, idx % 8
        assert (iy + ix) % 2 == 1
