"""Chunked-receiver gradient accumulation (Chunk 3.7 I3): must match full-batch grad."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


def _spec_loss(H_pred: torch.Tensor, H_target: torch.Tensor,
               weights=(1.0, 1.0, 1.0, 0.1)) -> torch.Tensor:
    """Replicates the per-loss-term reduction inside zero_shot_adapt's inner loop."""
    w_r, w_i, w_a, w_p = weights
    eps = 1e-6
    return (
        w_r * torch.nn.functional.l1_loss(H_pred.real, H_target.real)
        + w_i * torch.nn.functional.l1_loss(H_pred.imag, H_target.imag)
        + w_a * torch.nn.functional.l1_loss(
            torch.log10(H_pred.abs() + eps),
            torch.log10(H_target.abs() + eps),
        )
        + w_p * (1.0 - torch.cos(H_pred.angle() - H_target.angle())).mean()
    )


def test_chunked_grad_matches_full_batch():
    """Build a tiny synthetic predictor `f(z, r) = (W r) * z`, run a forward+backward
    on N=32 receivers in (a) full-batch mode and (b) chunked mode with chunk_size=8.
    The accumulated gradient ∂L/∂z must match the full-batch gradient within fp32 noise.
    """
    torch.manual_seed(0)
    n_obs = 32
    n_freq = 17
    z_dim = 4

    z_init = torch.randn(z_dim, dtype=torch.float64)
    R = torch.randn(n_obs, z_dim, dtype=torch.float64)            # per-receiver mixing
    W_pred = torch.randn(n_freq, dtype=torch.float64)             # spectral kernel
    H_target = torch.randn(n_obs, n_freq, dtype=torch.complex128)

    def render(z):
        # f_i(z) = (R[i] · z) * W_pred  → real spectrum; cast to complex with zero imag.
        coeff = (R @ z).unsqueeze(-1)                              # [n_obs, 1]
        return torch.complex(coeff * W_pred.unsqueeze(0), torch.zeros_like(coeff * W_pred.unsqueeze(0)))

    # --- full-batch gradient ---
    z_full = z_init.clone().requires_grad_(True)
    H_pred_full = render(z_full)
    loss_full = _spec_loss(H_pred_full, H_target)
    loss_full.backward()
    grad_full = z_full.grad.detach().clone()

    # --- chunked gradient ---
    z_chunk = z_init.clone().requires_grad_(True)
    chunk_size = 8
    for c0 in range(0, n_obs, chunk_size):
        c1 = min(c0 + chunk_size, n_obs)
        H_pred_c = render(z_chunk)[c0:c1]
        weight = (c1 - c0) / n_obs
        loss_c = weight * _spec_loss(H_pred_c, H_target[c0:c1])
        loss_c.backward()
    grad_chunk = z_chunk.grad.detach().clone()

    # The two gradients should agree closely. Reduction by L1.mean() inside
    # _spec_loss is computed over (chunk_n × n_freq) elements per chunk and over
    # (n_obs × n_freq) for full-batch; the chunk weighting scales the partial
    # gradient so the sum equals the full-batch reduction.
    rel_err = (grad_chunk - grad_full).abs().max() / (grad_full.abs().max() + 1e-12)
    assert rel_err < 1e-9, f"chunked grad differs from full by rel_err={float(rel_err)}"


def test_chunked_grad_with_uneven_last_chunk():
    """If n_obs is NOT divisible by chunk_size, the last chunk is smaller. The
    weighting `weight = (c1 - c0) / n_obs` ensures the accumulated gradient
    still matches full-batch."""
    torch.manual_seed(1)
    n_obs = 30      # 30 % 8 = 6 → last chunk has 6 receivers
    n_freq = 13
    z_dim = 3

    z_init = torch.randn(z_dim, dtype=torch.float64)
    R = torch.randn(n_obs, z_dim, dtype=torch.float64)
    W_pred = torch.randn(n_freq, dtype=torch.float64)
    H_target = torch.randn(n_obs, n_freq, dtype=torch.complex128)

    def render(z):
        coeff = (R @ z).unsqueeze(-1)
        return torch.complex(coeff * W_pred.unsqueeze(0),
                             torch.zeros_like(coeff * W_pred.unsqueeze(0)))

    z_full = z_init.clone().requires_grad_(True)
    _spec_loss(render(z_full), H_target).backward()
    grad_full = z_full.grad.detach().clone()

    z_chunk = z_init.clone().requires_grad_(True)
    chunk_size = 8
    for c0 in range(0, n_obs, chunk_size):
        c1 = min(c0 + chunk_size, n_obs)
        H_pred_c = render(z_chunk)[c0:c1]
        weight = (c1 - c0) / n_obs
        (weight * _spec_loss(H_pred_c, H_target[c0:c1])).backward()
    grad_chunk = z_chunk.grad.detach().clone()

    rel_err = (grad_chunk - grad_full).abs().max() / (grad_full.abs().max() + 1e-12)
    assert rel_err < 1e-9, f"chunked grad (uneven) differs by rel_err={float(rel_err)}"
