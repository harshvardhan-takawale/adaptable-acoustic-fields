"""Latent jitter (Chunk 3.6 C2): training perturbs z_s, eval doesn't."""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="tcnn requires CUDA"
)


@cuda_required
def test_jitter_active_in_training_mode():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=4, latent_dim=8, n_freq_bins=129,
        latent_jitter_sigma=0.5,
    ).cuda()
    model.train()
    rid = torch.tensor([0], device="cuda")
    z1 = model.get_latent(rid).detach().clone()
    z2 = model.get_latent(rid).detach().clone()
    assert not torch.allclose(z1, z2), "jitter should produce different z each call"


@cuda_required
def test_jitter_off_in_eval_mode():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=4, latent_dim=8, n_freq_bins=129,
        latent_jitter_sigma=0.5,
    ).cuda()
    model.eval()
    rid = torch.tensor([0], device="cuda")
    z1 = model.get_latent(rid).detach().clone()
    z2 = model.get_latent(rid).detach().clone()
    assert torch.allclose(z1, z2), "eval mode should be deterministic"


@cuda_required
def test_jitter_disabled_when_sigma_zero():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    model = INR2D_AutoDecoder(
        n_rooms=4, latent_dim=8, n_freq_bins=129,
        latent_jitter_sigma=0.0,
    ).cuda()
    model.train()
    rid = torch.tensor([0], device="cuda")
    z1 = model.get_latent(rid).detach().clone()
    z2 = model.get_latent(rid).detach().clone()
    assert torch.allclose(z1, z2), "sigma=0 must be deterministic even in train mode"


@cuda_required
def test_negative_jitter_sigma_raises():
    from aaf.models.inr_2d import INR2D_AutoDecoder

    with pytest.raises(ValueError):
        INR2D_AutoDecoder(
            n_rooms=4, latent_dim=8, n_freq_bins=129,
            latent_jitter_sigma=-0.1,
        )
