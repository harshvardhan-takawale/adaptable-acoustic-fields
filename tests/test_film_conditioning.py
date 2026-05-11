"""FiLM conditioning (Chunk 3.6 C1): construction, shapes, gradient flow."""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="tcnn requires CUDA"
)


@cuda_required
def test_film_module_present_and_reduces_tcnn_input_dim():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    latent_dim = 8
    model = INR2D_AutoDecoder(
        n_rooms=3, latent_dim=latent_dim, n_freq_bins=129,
        conditioning_type="film",
    ).cuda()
    assert model.film_sigma is not None
    assert model.film_signal is not None
    # Each FiLM Linear emits 2 * F (gamma | beta).
    sigma_F = model._sigma_feat_dim
    signal_F = model._signal_feat_dim
    assert model.film_sigma.out_features == 2 * sigma_F
    assert model.film_signal.out_features == 2 * signal_F
    # tcnn MLPs were built smaller (no z_s in the cat).
    assert model._model_encoder_sigma.n_input_dims == sigma_F
    assert model._model_signal.n_input_dims == signal_F


@cuda_required
def test_concat_path_is_default_and_unchanged():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(n_rooms=3, latent_dim=8, n_freq_bins=129).cuda()
    # Default conditioning_type is concat.
    assert model.conditioning_type == "concat"
    assert model.film_sigma is None and model.film_signal is None
    # Concat MLP input dims include the latent.
    assert model._model_encoder_sigma.n_input_dims == model._sigma_feat_dim + 8
    assert model._model_signal.n_input_dims == model._signal_feat_dim + 8


@cuda_required
def test_film_init_is_identity_at_construction():
    """Untrained FiLM should produce gamma=1, beta=0 -> sigma_input == sigma_feat."""
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=3, latent_dim=8, n_freq_bins=129, conditioning_type="film",
    ).cuda()
    z = torch.randn(8, device="cuda")
    sigma_F = model._sigma_feat_dim
    gb = model.film_sigma(z)
    gamma, beta = gb.chunk(2, dim=-1)
    # Weights are zero -> gamma == bias_gamma = 1, beta == bias_beta = 0.
    assert torch.allclose(gamma, torch.ones(sigma_F, device="cuda"))
    assert torch.allclose(beta, torch.zeros(sigma_F, device="cuda"))


@cuda_required
def test_film_forward_runs_and_grads_flow_to_film():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=3, latent_dim=8, n_freq_bins=129, conditioning_type="film",
    ).cuda()
    B, N = 2, 16
    pts = torch.rand(B, N, 2, device="cuda") * 2 - 1
    view = torch.rand(B, N, 2, device="cuda") * 2 - 1
    tx = torch.rand(B, N, 2, device="cuda") * 2 - 1
    z_s = model.get_latent(torch.tensor([0, 1], device="cuda"))
    attn, signal = model(pts, view, tx, z_s=z_s)
    assert attn.shape == (B, N, 129)
    assert signal.shape == (B, N, 129)
    loss = signal.real.abs().mean() + attn.real.abs().mean()
    loss.backward()
    # FiLM generators should receive non-zero gradient.
    assert model.film_sigma.weight.grad is not None
    assert torch.any(model.film_sigma.weight.grad != 0)
    assert model.film_signal.weight.grad is not None
    assert torch.any(model.film_signal.weight.grad != 0)


@cuda_required
def test_unknown_conditioning_type_raises():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    with pytest.raises(ValueError):
        INR2D_AutoDecoder(
            n_rooms=3, latent_dim=8, n_freq_bins=129,
            conditioning_type="bogus",
        )
