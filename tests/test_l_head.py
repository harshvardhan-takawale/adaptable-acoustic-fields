"""L-head: shape, gradient flow, optional-disabling."""
from __future__ import annotations

import pytest
import torch


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="tcnn requires CUDA"
)


@cuda_required
def test_predict_L_returns_tensor_with_lhead_enabled():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=3, latent_dim=8, n_freq_bins=129, l_head_enabled=True
    ).cuda()
    z = torch.randn(4, 8, device="cuda")
    L_pred = model.predict_L(z)
    assert L_pred is not None
    assert L_pred.shape == (4,)
    assert L_pred.dtype == torch.float32


@cuda_required
def test_predict_L_returns_none_with_lhead_disabled():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=3, latent_dim=8, n_freq_bins=129, l_head_enabled=False
    ).cuda()
    assert model.l_head is None
    z = torch.randn(4, 8, device="cuda")
    assert model.predict_L(z) is None


@cuda_required
def test_lhead_loss_backprops_to_lhead_and_latents():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=4, latent_dim=8, n_freq_bins=129, l_head_enabled=True
    ).cuda()
    room_ids = torch.tensor([0, 1, 2, 3], device="cuda")
    z = model.get_latent(room_ids)                 # [4, 8]
    L_true = torch.tensor([3.0, 4.0, 5.0, 6.0], device="cuda")
    loss = (model.predict_L(z) - L_true).abs().mean()
    loss.backward()
    # Both the L-head MLP and the latent embedding receive gradient.
    lhead_grad = model.l_head[0].weight.grad
    assert lhead_grad is not None and torch.any(lhead_grad != 0)
    latent_grad = model.latents.weight.grad
    assert latent_grad is not None and torch.any(latent_grad != 0)


@cuda_required
def test_predict_L_handles_1d_input():
    """A single latent (latent_dim,) should auto-broadcast to a [1] result."""
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=2, latent_dim=8, n_freq_bins=129, l_head_enabled=True
    ).cuda()
    z = torch.randn(8, device="cuda")
    L_pred = model.predict_L(z)
    assert L_pred.shape == (1,)


@cuda_required
def test_linear_l_head_arch():
    """l_head_arch='linear' wires a single nn.Linear(d, 1), no nonlinearity.

    Verifies the Chunk-3.5+ addendum branch (R6-R8): a linear L-head forces z_s
    to be linearly readable as L, the strongest inductive bias toward a 1-D manifold.
    """
    import torch.nn as nn
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=3, latent_dim=8, n_freq_bins=129,
        l_head_enabled=True, l_head_arch="linear",
    ).cuda()
    assert isinstance(model.l_head, nn.Linear), (
        f"expected nn.Linear, got {type(model.l_head).__name__}"
    )
    assert model.l_head.in_features == 8
    assert model.l_head.out_features == 1

    z = torch.randn(4, 8, device="cuda")
    L_pred = model.predict_L(z)
    assert L_pred.shape == (4,)

    # Default arch is mlp_32 — a Sequential (existing behaviour for R0-R5).
    model_mlp = INR2D_AutoDecoder(
        n_rooms=3, latent_dim=8, n_freq_bins=129, l_head_enabled=True,
    ).cuda()
    assert isinstance(model_mlp.l_head, nn.Sequential)


@cuda_required
def test_unknown_l_head_arch_raises():
    from aaf.models.inr_2d import INR2D_AutoDecoder
    with pytest.raises(ValueError, match="Unknown l_head_arch"):
        INR2D_AutoDecoder(
            n_rooms=2, latent_dim=4, n_freq_bins=129,
            l_head_enabled=True, l_head_arch="quadratic",
        )
