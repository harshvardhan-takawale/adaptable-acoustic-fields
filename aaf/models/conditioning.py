"""P3-1 conditioning-vector builders for the edit-mechanism head-to-head.

Three arms differ ONLY in the conditioning path fed to the shared FiLM generator:
  - ``latent``       : the Phase-2 per-room auto-decoder latent (handled by the model).
  - ``geom_fourier`` : Arm G — 48-d Fourier features of normalized (L,W,H).
  - ``eigen``        : Arm G+ — 64-d sorted eigenfrequency vector (+ a per-bin resonance
                       map R applied at the signal-branch output; see INR3D_AutoDecoder).

This module is deliberately **tcnn-free** (imports only torch / math / the analytic modal
enumerator) so it stays importable — and unit-testable — on CPU-only nodes where tinycudann
cannot load. Do NOT import ``aaf.models.inr_3d`` here.

Normalization box (matches sample_rooms_3d.DEFAULT_RANGES): L∈[3,6], W∈[3,5], H∈[2.5,4].
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from aaf.sim.analytical_modal_3d import eigenfrequencies_3d, C_DEFAULT

# ----------------------------------------------------------------------
# Arm G — Fourier features of (L, W, H)
# ----------------------------------------------------------------------
_N_FOURIER_K = 8                    # k = 0..7 → 2^k frequencies
FOURIER_DIM = 3 * 2 * _N_FOURIER_K  # 3 dims × {sin, cos} × 8 = 48


def _normalize_geom(L: float, W: float, H: float, device, dtype) -> torch.Tensor:
    """g = ((L−3)/3, (W−3)/2, (H−2.5)/1.5) ∈ [0,1]^3 over the room box."""
    return torch.tensor(
        [(L - 3.0) / 3.0, (W - 3.0) / 2.0, (H - 2.5) / 1.5],
        device=device, dtype=dtype,
    )


def fourier_features(L: float, W: float, H: float, device=None,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """48-d Fourier features. Layout: per-dimension block of
    ``[sin(2^0 π g_i) .. sin(2^7 π g_i), cos(2^0 π g_i) .. cos(2^7 π g_i)]`` for i in (L,W,H).
    All values ∈ [-1, 1]."""
    g = _normalize_geom(L, W, H, device, dtype)                    # [3]
    freqs = (2.0 ** torch.arange(_N_FOURIER_K, device=device, dtype=dtype)) * math.pi  # [8]
    ang = g[:, None] * freqs[None, :]                              # [3, 8]
    feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)      # [3, 16]
    return feat.reshape(-1)                                        # [48]


# ----------------------------------------------------------------------
# Arm G+ — analytic eigenstructure
# ----------------------------------------------------------------------
def eigen_features(L: float, W: float, H: float, n: int = 64, f_ref: float = 300.0,
                   c: float = C_DEFAULT, f_max: float = 2000.0, device=None,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sorted vector of the lowest ``n`` distinct nonzero eigenfrequencies / ``f_ref``.

    Enumerate to a generous ``f_max`` (2000 Hz) so ≥64 distinct modes always exist even
    for the smallest room, then take the lowest ``n``. Entries may exceed 1.0 (that is fine).
    """
    modes = [e.f for e in eigenfrequencies_3d(L, W, H, c=c, f_max=f_max) if e.f > 0.0]
    if len(modes) < n:
        raise ValueError(
            f"only {len(modes)} distinct eigenfreqs < {f_max} Hz for room "
            f"({L},{W},{H}); need {n}. Raise f_max."
        )
    return torch.tensor(modes[:n], device=device, dtype=dtype) / f_ref  # [n]


def resonance_map(L: float, W: float, H: float, n_bins: int = 601, df: float = 0.5,
                  sigma_hz: float = 2.0, f_cap: float = 310.0, c: float = C_DEFAULT,
                  device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Per-bin resonance map R[b] = Σ_n exp(−(f_b − f_n)² / 2σ²) over distinct eigenfreqs
    f_n ≤ ``f_cap``, evaluated on the supervised bins f_b = b·df (b=0..n_bins−1), then
    max-normalized to 1. σ=2 Hz ≈ the modal peak width at α=0.15.
    """
    f_b = torch.arange(n_bins, device=device, dtype=dtype) * df            # [n_bins]  (0..300 Hz)
    f_n = [e.f for e in eigenfrequencies_3d(L, W, H, c=c, f_max=f_cap) if 0.0 < e.f <= f_cap]
    if not f_n:
        return torch.zeros(n_bins, device=device, dtype=dtype)
    f_n_t = torch.tensor(f_n, device=device, dtype=dtype)                  # [M]
    diff = f_b[:, None] - f_n_t[None, :]                                   # [n_bins, M]
    R = torch.exp(-(diff ** 2) / (2.0 * sigma_hz ** 2)).sum(dim=1)         # [n_bins]
    return R / R.max().clamp(min=1e-8)


# ----------------------------------------------------------------------
# Shared dispatch — used by BOTH the trainer and the eval
# ----------------------------------------------------------------------
def build_cond_vector(cond_source: str, L: float, W: float, H: float, device=None, *,
                      model=None, room_ids: Optional[torch.Tensor] = None,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the per-room conditioning vector fed to the FiLM generator.

    ``latent``       → delegate to ``model.get_latent(room_ids)`` (jitter applied inside
                       when model.training); returns [B, latent_dim].
    ``geom_fourier`` → ``fourier_features`` [48].
    ``eigen``        → ``eigen_features`` [64].
    (The eigen arm's per-bin resonance map R is set separately via ``model.set_resonance``.)
    """
    if cond_source == "latent":
        if model is None or room_ids is None:
            raise ValueError("latent arm requires model + room_ids")
        return model.get_latent(room_ids)
    if cond_source == "geom_fourier":
        return fourier_features(L, W, H, device=device, dtype=dtype)
    if cond_source == "eigen":
        return eigen_features(L, W, H, device=device, dtype=dtype)
    raise ValueError(f"unknown cond_source {cond_source!r}")
