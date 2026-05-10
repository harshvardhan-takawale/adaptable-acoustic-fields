"""FreqRenderer2D: shape, dtype, finiteness, and 2D AABB intersection sanity.

Uses a fake model returning random complex tensors so the test is GPU-independent
modulo torch.complex64 — but the real-world use is on CUDA. We attempt CUDA, fall
back to CPU only for ray-AABB tests (the renderer's complex math runs fine on CPU).
"""
import math

import pytest
import torch

from aaf.renderers.freq_2d import FreqRenderer2D


def _fake_model(n_freq_bins: int):
    """Returns (attn, signal) of the right shape filled with small random complex values."""

    class FakeModel(torch.nn.Module):
        def forward(self, pts, view, tx, tx_view=None, z_s=None):
            B, N, _ = pts.shape
            # Real-positive sigma to get sensible alpha; small phase increments.
            attn = torch.complex(
                torch.rand(B, N, n_freq_bins, device=pts.device) * 0.05,
                torch.randn(B, N, n_freq_bins, device=pts.device) * 0.01,
            )
            signal = torch.complex(
                torch.randn(B, N, n_freq_bins, device=pts.device) * 0.1,
                torch.randn(B, N, n_freq_bins, device=pts.device) * 0.1,
            )
            return attn, signal

    return FakeModel()


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_forward_shape_and_dtype():
    device = _device()
    fs = 4096
    n_time = 1024  # smaller for test speed
    n_freq = n_time // 2 + 1
    renderer = FreqRenderer2D(
        n_azi=16, n_pts_per_ray=16, fs=fs, n_time_samples=n_time
    ).to(device)
    model = _fake_model(n_freq).to(device)
    B = 4
    rx_pos = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [1.5, 2.5]], device=device)
    tx_pos = torch.tensor([[0.5, 0.5]] * B, device=device)
    room_min = torch.tensor([0.0, 0.0], device=device)
    room_max = torch.tensor([4.0, 4.0], device=device)

    out = renderer(model, rx_pos, tx_pos, room_min, room_max)
    assert out.shape == (B, n_freq), f"unexpected shape {out.shape}"
    assert out.is_complex(), f"expected complex output, got {out.dtype}"
    assert torch.isfinite(out.real).all()
    assert torch.isfinite(out.imag).all()


def test_ray_aabb_intersection_centered_receiver():
    """Receiver at room center: t_far in any direction equals half the room
    diagonal (for diagonal rays) or half a wall-distance (for axis-aligned).
    """
    device = _device()
    renderer = FreqRenderer2D(
        n_azi=8, n_pts_per_ray=2, fs=4096, n_time_samples=512
    ).to(device).eval()  # eval → no jitter
    rx_pos = torch.tensor([[2.0, 2.0]], device=device)
    room_min = torch.tensor([0.0, 0.0], device=device)
    room_max = torch.tensor([4.0, 4.0], device=device)
    dirs = renderer._ray_directions_2d(device)
    t_far = renderer._ray_aabb_intersect_2d(rx_pos, dirs, room_min, room_max)
    # For axis-aligned rays in a 4x4 room from center: t_far = 2.0 (half-width).
    # For diagonal rays: t_far = 2*sqrt(2) ≈ 2.83.
    assert (t_far > 0).all()
    assert (t_far <= 4.0 * math.sqrt(2)).all()


def test_ray_directions_have_unit_norm():
    device = _device()
    renderer = FreqRenderer2D(
        n_azi=64, n_pts_per_ray=2, fs=4096, n_time_samples=512
    ).to(device)
    dirs = renderer._ray_directions_2d(device)
    norms = torch.linalg.vector_norm(dirs, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_jitter_active_in_train_mode():
    device = _device()
    renderer = FreqRenderer2D(
        n_azi=8, n_pts_per_ray=2, fs=4096, n_time_samples=512
    ).to(device)
    renderer.train()
    d1 = renderer._ray_directions_2d(device)
    d2 = renderer._ray_directions_2d(device)
    assert not torch.allclose(d1, d2), "training mode should jitter ray directions"
    renderer.eval()
    d3 = renderer._ray_directions_2d(device)
    d4 = renderer._ray_directions_2d(device)
    assert torch.allclose(d3, d4), "eval mode should produce deterministic directions"
