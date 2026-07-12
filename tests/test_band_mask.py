"""P3-1 band-limited loss mask — pure CPU.

Replicates the exact slice-masked L1 the trainer applies (multi_room_3d._losses with
band=(0,601)) and asserts the two load-bearing invariants:
  1. gradients are EXACTLY zero for out-of-band bins (>300 Hz);
  2. the loss value is unchanged when out-of-band target bins are randomized.
Does NOT import the trainer/model (they drag in tinycudann/CUDA) — it reproduces the
one-line slice so the invariant is tested independently of the GPU stack.
"""
import torch

from aaf.eval.band_limited import band_indices


def _band_l1(H_pred, H_target, band):
    lo, hi = band
    hp, ht = H_pred[..., lo:hi], H_target[..., lo:hi]
    # mirror _losses' real+imag+log_amp+phase over the sliced tensors
    eps = 1e-6
    l_real = (hp.real - ht.real).abs().mean()
    l_imag = (hp.imag - ht.imag).abs().mean()
    l_amp = (torch.log10(hp.abs() + eps) - torch.log10(ht.abs() + eps)).abs().mean()
    l_phase = (1.0 - torch.cos(hp.angle() - ht.angle())).mean()
    return l_real + l_imag + l_amp + l_phase


def test_band_indices_300hz():
    assert band_indices(4096, 4097, 0.0, 300.0) == (0, 601)   # Δf=0.5 → bins 0..600
    assert band_indices(4096, 4097, 0.0, 250.0) == (0, 501)


def test_out_of_band_gradient_exactly_zero():
    torch.manual_seed(0)
    n_freq = 4097
    band = band_indices(4096, n_freq, 0.0, 300.0)             # (0, 601)
    H_pred = (torch.randn(8, n_freq) + 1j * torch.randn(8, n_freq)).requires_grad_(True)
    H_target = torch.randn(8, n_freq) + 1j * torch.randn(8, n_freq)
    loss = _band_l1(H_pred, H_target, band)
    loss.backward()
    g = H_pred.grad
    assert torch.count_nonzero(g[..., band[1]:]) == 0         # bins >300 Hz: exactly zero
    assert torch.count_nonzero(g[..., :band[1]]) > 0          # in-band: nonzero


def test_loss_ignores_out_of_band_target():
    torch.manual_seed(1)
    n_freq = 4097
    band = band_indices(4096, n_freq, 0.0, 300.0)
    H_pred = torch.randn(4, n_freq) + 1j * torch.randn(4, n_freq)
    H_target = torch.randn(4, n_freq) + 1j * torch.randn(4, n_freq)
    l0 = float(_band_l1(H_pred, H_target, band))
    Ht2 = H_target.clone()
    Ht2[..., band[1]:] = torch.randn_like(Ht2[..., band[1]:])  # scramble out-of-band only
    l1 = float(_band_l1(H_pred, Ht2, band))
    assert abs(l0 - l1) < 1e-6
