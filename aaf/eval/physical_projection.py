"""Render-time PHYSICAL PROJECTION (P4-6 default, D160; validated in D152(d5)).

The least-squares projection of a field sampled at in-room grid nodes onto fields band-limited to
|k| <= 1.5 k0 (k0 = 2 pi f / c), solved by CG on a zero-padded grid. It respects the room boundary
and passes |k| <= 1.5 k0 at unity: it keeps a physical plane wave to <= 5e-4 at every frequency
25-300 Hz (test), leaves FDTD's own fields at d1 0.99-1.00, and removes the supra-physical speckle
(D152). It is POST-PROCESSING: every report shows raw and projected metrics side by side.

Bins below 25 Hz are returned unprojected (at 1.5 k0 the band holds almost no spatial degrees of
freedom there, and the near-DC term is spatially uniform by construction of the DC head).
"""
from __future__ import annotations

import math

import numpy as np

C = 343.0
DF = 0.5
CUT = 1.5
LO_BIN = 50                     # 25 Hz: bins below are left raw
CG_ITERS = 300


def k_grid(nx, ny, dx, dy, pad=4):
    """|k| on the pad-x zero-padded (n x n) grid for nodes spaced dx, dy."""
    n = pad * max(nx, ny)
    kx = 2 * np.pi * np.fft.fftfreq(n, dx)
    ky = 2 * np.pi * np.fft.fftfreq(n, dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return np.hypot(KX, KY)


def _torch():
    import torch
    return torch, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _embed(vals, keep, n):
    """[B, n_in] values at keep nodes -> [B, n, n] on the padded grid (zeros elsewhere)."""
    torch, dev = _torch()
    g = torch.zeros((vals.shape[0], n, n), dtype=torch.complex128, device=dev)
    ix, iy = np.nonzero(keep)
    g[:, ix, iy] = torch.as_tensor(np.asarray(vals), dtype=torch.complex128, device=dev)
    return g


def project_lowpass(vals, keep, K, kc, iters=CG_ITERS):
    """Least-squares projection of in-room samples onto fields band-limited to |k| <= kc.

    Solves min_g ||M g - f||^2 over g with FFT support in |K| <= kc (one kc per row), by CG on the
    normal equations P M P g = P M f (P = band-limit projector, M = in-room mask), from g = 0.
    Returns the fitted field at the keep nodes, [B, n_in]."""
    torch, dev = _torch()
    n = K.shape[0]
    band = torch.as_tensor(K[None] <= np.asarray(kc)[:, None, None], device=dev)
    m = torch.zeros((n, n), dtype=torch.float64, device=dev)
    ix, iy = np.nonzero(keep)
    m[ix, iy] = 1.0

    def P(u):
        return torch.fft.ifft2(torch.fft.fft2(u) * band)

    def A(u):
        return P(m * P(u))

    b = P(_embed(vals, keep, n))
    x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = (r.abs() ** 2).sum((1, 2))
    tol = 1e-24 * rs                                  # stop a row once |r| / |b| < 1e-12
    for _ in range(iters):
        live = rs > tol
        if not bool(live.any()):
            break
        Ap = A(p)
        pAp = (p.conj() * Ap).sum((1, 2)).real
        # a converged row (few band-limited dof at low f) has pAp ~ rounding; freeze it there
        ok = live & (pAp > 0)
        alpha = torch.where(ok, rs / torch.where(ok, pAp, torch.ones_like(pAp)), torch.zeros_like(rs))
        x = x + alpha[:, None, None] * p
        r = r - alpha[:, None, None] * Ap
        rs_new = (r.abs() ** 2).sum((1, 2))
        beta = torch.where(ok, rs_new / torch.where(ok, rs, torch.ones_like(rs)), torch.zeros_like(rs))
        p = torch.where(ok[:, None, None], r + beta[:, None, None] * p, p)
        rs = torch.where(ok, rs_new, rs)
    return x[:, ix, iy].cpu().numpy()



def project_field(F, keep, dx, dy, pad=4, df=DF, lo_bin=LO_BIN, chunk=128):
    """Project every bin >= lo_bin of F [n_rx, n_bins] (receivers in `keep`'s raster order, i.e.
    np.nonzero(keep)) onto |k| <= 1.5 k0. Returns a complex64 array of F's shape."""
    keep = np.asarray(keep, bool)
    K = k_grid(keep.shape[0], keep.shape[1], dx, dy, pad)
    out = np.array(F, dtype=np.complex64, copy=True)
    bins = np.arange(lo_bin, F.shape[1])
    kc = CUT * 2 * math.pi * bins * df / C
    for s in range(0, len(bins), chunk):
        sl = slice(s, s + chunk)
        out[:, bins[sl]] = project_lowpass(F[:, bins[sl]].T, keep, K, kc[sl]).T.astype(np.complex64)
    return out
