"""P3-2 band-limited protocol (0-300 Hz) — pure CPU.

Same slice-mask convention as P3-1 (tests/test_band_mask.py), re-asserted for the 2D
trainer, plus the band-limited-RIR symmetry that P3-2's decay metrics depend on: the
>300 Hz mask must be applied IDENTICALLY to prediction and target, or the decay
comparison measures the mask instead of the room.

Does NOT import the trainer or the model (they drag in tinycudann/CUDA); it reproduces the
one-line slice so the invariant is tested independently of the GPU stack.
"""
from __future__ import annotations

import numpy as np
import torch

from aaf.eval.band_limited import band_indices
from aaf.eval.modal_decay import band_limited_rir

FS, N_FREQ, N_TIME = 4096, 4097, 8192


def _band_loss(H_pred, H_target, band):
    """Mirror multi_room_2d_mat._losses: 4 terms over the sliced tensors."""
    lo, hi = band
    hp, ht = H_pred[..., lo:hi], H_target[..., lo:hi]
    eps = 1e-6
    return ((hp.real - ht.real).abs().mean()
            + (hp.imag - ht.imag).abs().mean()
            + (torch.log10(hp.abs() + eps) - torch.log10(ht.abs() + eps)).abs().mean()
            + (1.0 - torch.cos(hp.angle() - ht.angle())).mean())


def test_band_indices_for_the_p3_2_protocol():
    assert band_indices(FS, N_FREQ, 0.0, 300.0) == (0, 601)   # df=0.5 Hz -> bins 0..600
    assert band_indices(FS, N_FREQ, 0.0, 250.0) == (0, 501)


def test_out_of_band_gradient_exactly_zero():
    torch.manual_seed(0)
    band = band_indices(FS, N_FREQ, 0.0, 300.0)
    H_pred = (torch.randn(8, N_FREQ) + 1j * torch.randn(8, N_FREQ)).requires_grad_(True)
    H_target = torch.randn(8, N_FREQ) + 1j * torch.randn(8, N_FREQ)
    _band_loss(H_pred, H_target, band).backward()
    g = H_pred.grad
    assert torch.count_nonzero(g[..., band[1]:]) == 0
    assert torch.count_nonzero(g[..., :band[1]]) > 0


def test_loss_ignores_out_of_band_target():
    torch.manual_seed(1)
    band = band_indices(FS, N_FREQ, 0.0, 300.0)
    H_pred = torch.randn(4, N_FREQ) + 1j * torch.randn(4, N_FREQ)
    H_target = torch.randn(4, N_FREQ) + 1j * torch.randn(4, N_FREQ)
    l0 = float(_band_loss(H_pred, H_target, band))
    Ht2 = H_target.clone()
    Ht2[..., band[1]:] = torch.randn_like(Ht2[..., band[1]:])
    assert abs(l0 - float(_band_loss(H_pred, Ht2, band))) < 1e-6


def test_band_limited_rir_mask_is_symmetric_and_real():
    """The identical-mask property that makes pred-vs-target decay comparable."""
    rng = np.random.default_rng(0)
    H = rng.normal(size=(3, N_FREQ)) + 1j * rng.normal(size=(3, N_FREQ))
    r = band_limited_rir(H, FS, N_TIME, 0.0, 300.0)
    assert r.shape == (3, N_TIME)
    assert np.isrealobj(r) or np.allclose(np.imag(r), 0.0)

    # identical input -> identical output (no side-dependent state)
    assert np.allclose(r, band_limited_rir(H.copy(), FS, N_TIME, 0.0, 300.0))

    # scrambling >300 Hz must not change the band-limited RIR at all
    H2 = H.copy()
    _, hi = band_indices(FS, N_FREQ, 0.0, 300.0)
    H2[..., hi:] = rng.normal(size=H2[..., hi:].shape) + 1j * rng.normal(size=H2[..., hi:].shape)
    assert np.allclose(r, band_limited_rir(H2, FS, N_TIME, 0.0, 300.0), atol=1e-12)

    # and the round trip really is band-limited
    R = np.fft.rfft(r, n=N_TIME, axis=-1)
    assert np.max(np.abs(R[..., hi:])) < 1e-8


def test_band_limited_rir_does_not_mutate_input():
    rng = np.random.default_rng(3)
    H = rng.normal(size=(2, N_FREQ)) + 1j * rng.normal(size=(2, N_FREQ))
    H_ref = H.copy()
    band_limited_rir(H, FS, N_TIME, 0.0, 300.0)
    assert np.array_equal(H, H_ref), "band_limited_rir must not modify its argument"
