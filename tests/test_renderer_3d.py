"""Tests for FreqRenderer3D — ray sampling, 3D AABB, smoke forward."""
import math

import numpy as np
import pytest
import torch

from aaf.renderers.freq_3d import FreqRenderer3D


@pytest.fixture
def renderer():
    return FreqRenderer3D(
        n_azi=4, n_ele=4, n_pts_per_ray=8, near=1e-3,
        fs=1024, n_time_samples=2048, use_geometric_attn=False,
    )


def test_renderer_3d_ray_count(renderer):
    assert renderer.n_rays == 4 * 4 + 2


def test_renderer_3d_ray_directions_unit_norm(renderer):
    renderer.eval()
    dirs = renderer._ray_directions_3d(device=torch.device("cpu"))
    assert dirs.shape == (renderer.n_rays, 3)
    norms = dirs.norm(dim=-1)
    # All should be unit vectors (within float tolerance).
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_renderer_3d_aabb_in_unit_cube(renderer):
    """Receiver at the centre of the unit cube: each ray's t_far must equal
    the box's intersection with that direction (at most √3/2 along diagonal)."""
    rx = torch.tensor([[0.5, 0.5, 0.5]])
    dirs = renderer._ray_directions_3d(device=torch.device("cpu"))
    room_min = torch.tensor([0.0, 0.0, 0.0])
    room_max = torch.tensor([1.0, 1.0, 1.0])
    t_far = renderer._ray_aabb_intersect_3d(rx, dirs, room_min, room_max)
    assert t_far.shape == (1, renderer.n_rays)
    # Lower bound: cube half-side (0.5) for the axis-aligned rays; upper bound
    # diagonal half (√3/2 ≈ 0.866).
    assert (t_far >= 0.5).all(), f"some t_far < 0.5: {t_far.min()}"
    assert (t_far <= math.sqrt(3) / 2 + 1e-4).all(), (
        f"some t_far > diag/2: {t_far.max()}"
    )


def test_renderer_3d_jitter_in_train_mode_only(renderer):
    """In .train() the ray grid jitters; in .eval() it's deterministic."""
    torch.manual_seed(0)
    renderer.eval()
    d1 = renderer._ray_directions_3d(device=torch.device("cpu"))
    d2 = renderer._ray_directions_3d(device=torch.device("cpu"))
    assert torch.allclose(d1, d2), "eval-mode directions must be deterministic"

    renderer.train()
    torch.manual_seed(0)
    d3 = renderer._ray_directions_3d(device=torch.device("cpu"))
    torch.manual_seed(1)
    d4 = renderer._ray_directions_3d(device=torch.device("cpu"))
    # Different seeds → different jitter → different directions.
    assert not torch.allclose(d3, d4), "training-mode directions must jitter"


class _FakeModel(torch.nn.Module):
    """Model whose output is constant complex per (B, N, F)."""

    def __init__(self, n_freq_bins: int, attn_val=0.1, sig_val=(1.0, 0.0)):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.attn_val = attn_val
        self.sig_val = sig_val

    def forward(self, pts, view, tx, tx_view=None, z_s=None):
        B, N = pts.shape[:2]
        attn = torch.full(
            (B, N, self.n_freq_bins), self.attn_val,
            device=pts.device, dtype=torch.float32,
        )
        attn_c = torch.complex(attn, torch.zeros_like(attn))
        signal_re = torch.full(
            (B, N, self.n_freq_bins), self.sig_val[0],
            device=pts.device, dtype=torch.float32,
        )
        signal_im = torch.full(
            (B, N, self.n_freq_bins), self.sig_val[1],
            device=pts.device, dtype=torch.float32,
        )
        # Force RFFT symmetry on the imag DC + Nyquist
        signal_im = signal_im.clone()
        signal_im[..., 0] = 0
        signal_im[..., -1] = 0
        signal_c = torch.complex(signal_re, signal_im)
        return attn_c, signal_c


def test_renderer_3d_forward_shape_and_finite(renderer):
    model = _FakeModel(n_freq_bins=renderer.n_freq_bins)
    rx = torch.tensor([[0.5, 0.5, 0.5], [0.6, 0.4, 0.7]])
    tx = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    rm = torch.tensor([0.0, 0.0, 0.0])
    rM = torch.tensor([1.0, 1.0, 1.0])
    H = renderer(model, rx, tx, rm, rM)
    assert H.shape == (2, renderer.n_freq_bins)
    assert H.is_complex()
    assert torch.isfinite(H.real).all() and torch.isfinite(H.imag).all()
    # Non-trivial: at least some non-zero output.
    assert H.abs().max() > 0


def test_renderer_3d_input_shape_validation(renderer):
    model = _FakeModel(n_freq_bins=renderer.n_freq_bins)
    rm = torch.tensor([0.0, 0.0, 0.0])
    rM = torch.tensor([1.0, 1.0, 1.0])
    # 2D rx_pos → must raise
    with pytest.raises(ValueError):
        renderer(model, torch.tensor([[0.5, 0.5]]),
                 torch.tensor([[0.1, 0.1]]), rm, rM)
