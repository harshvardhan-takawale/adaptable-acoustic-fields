"""Tests for INR3D_Single — forward shape, RFFT symmetry, gradient flow."""
import math

import pytest
import torch

# tcnn requires CUDA at instantiation; skip these tests on CPU-only nodes.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="INR3D_Single needs tcnn (CUDA).",
)


def _make_model(n_freq_bins=257):
    from aaf.models.inr_3d import INR3D_Single
    # Small HashGrid to keep memory tiny in tests.
    hg = {
        "otype": "HashGrid",
        "n_levels": 4,
        "n_features_per_level": 2,
        "log2_hashmap_size": 12,
        "base_resolution": 8,
        "per_level_scale": 1.5,
    }
    return INR3D_Single(n_freq_bins=n_freq_bins, hash_grid_config=hg)


def test_inr3d_forward_shape():
    m = _make_model(n_freq_bins=257).cuda()
    B, N = 2, 16
    pts = torch.rand(B, N, 3, device="cuda") * 2 - 1
    view = torch.rand(B, N, 3, device="cuda") * 2 - 1
    tx = torch.rand(B, N, 3, device="cuda") * 2 - 1
    attn, signal = m(pts, view, tx)
    assert attn.shape == (B, N, 257)
    assert signal.shape == (B, N, 257)
    assert attn.is_complex() and signal.is_complex()


def test_inr3d_sigma_nonnegative():
    """σ (real part of attn) must be ≥ 0 because of softplus."""
    m = _make_model(n_freq_bins=257).cuda()
    pts = torch.rand(1, 32, 3, device="cuda") * 2 - 1
    view = torch.rand(1, 32, 3, device="cuda") * 2 - 1
    tx = torch.rand(1, 32, 3, device="cuda") * 2 - 1
    attn, _ = m(pts, view, tx)
    assert (attn.real >= 0).all().item()


def test_inr3d_rfft_symmetry_dc_and_nyquist():
    """DC and Nyquist bins of `signal` must have imag = 0."""
    m = _make_model(n_freq_bins=257).cuda()  # n_time = 2*256 = 512 (even)
    pts = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    view = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    tx = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    _, signal = m(pts, view, tx)
    assert torch.allclose(
        signal[..., 0].imag, torch.zeros_like(signal[..., 0].imag), atol=1e-6
    )
    assert torch.allclose(
        signal[..., -1].imag, torch.zeros_like(signal[..., -1].imag), atol=1e-6
    )


def test_inr3d_gradient_flow():
    """Loss.backward() must populate gradients on at least one tcnn param."""
    m = _make_model(n_freq_bins=129).cuda()
    pts = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    view = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    tx = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    attn, signal = m(pts, view, tx)
    # Build a meaningful loss.
    loss = signal.abs().mean() + attn.abs().mean()
    loss.backward()
    has_any_grad = False
    for p in m.parameters():
        if p.grad is not None and torch.isfinite(p.grad).any():
            has_any_grad = True
            break
    assert has_any_grad, "no parameter received gradient"


def test_inr3d_input_validation():
    m = _make_model(n_freq_bins=129).cuda()
    bad_pts = torch.rand(1, 8, 2, device="cuda") * 2 - 1
    view = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    tx = torch.rand(1, 8, 3, device="cuda") * 2 - 1
    with pytest.raises(ValueError):
        m(bad_pts, view, tx)
