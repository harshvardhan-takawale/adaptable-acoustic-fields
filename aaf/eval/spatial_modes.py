"""Spatial mode analysis helpers (Chunk 3.7 V0 critical path).

Extract pressure fields from saved zero-shot predictions at specific
frequencies, compare to analytical 2D mode shapes, and quantify alignment
(spatial Pearson correlation, node match, shape-fit error).

Receiver-grid convention (matches scripts/build_datasets.py:60-63):
    8×8 grid stored row-major with the OUTER loop over y and the INNER loop
    over x. So flat index ``i ∈ [0, 63]`` maps to ``(iy, ix) = (i // 8, i % 8)``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from aaf.sim.analytical_modal_2d import _pair_shape, eigenfrequencies_2d


N_GRID_DEFAULT = 8
MARGIN_DEFAULT = 0.5


def bin_index_for_freq(f_hz: float, fs: float, n_freq_bins: int) -> int:
    """Return the rfft bin index closest to ``f_hz``. Clamped to [0, n_freq_bins-1]."""
    n_time = 2 * (n_freq_bins - 1)
    idx = int(round(float(f_hz) * n_time / float(fs)))
    return max(0, min(n_freq_bins - 1, idx))


def extract_pressure_field(
    H_64xF: np.ndarray, f_hz: float, fs: float, n_freq_bins: int,
    n_grid: int = N_GRID_DEFAULT,
) -> np.ndarray:
    """Pressure at frequency ``f_hz`` reshaped onto the (iy, ix) 8×8 grid.

    ``H_64xF`` is the saved zero-shot prediction or ISM ground truth, shape
    ``[64, n_freq_bins]`` complex. Returns ``[n_grid, n_grid]`` complex.
    """
    if H_64xF.ndim != 2 or H_64xF.shape[0] != n_grid * n_grid:
        raise ValueError(
            f"expected H shape [{n_grid*n_grid}, n_freq_bins], got {H_64xF.shape}"
        )
    if H_64xF.shape[1] != n_freq_bins:
        raise ValueError(
            f"H_64xF last dim {H_64xF.shape[1]} != n_freq_bins {n_freq_bins}"
        )
    bin_idx = bin_index_for_freq(f_hz, fs, n_freq_bins)
    p_flat = H_64xF[:, bin_idx]                      # [64] complex
    # build_datasets.py uses `for y in ys: for x in xs`, so flat index i has
    # iy = i // n_grid (outer) and ix = i % n_grid (inner).
    return p_flat.reshape(n_grid, n_grid)


def receiver_grid_xy(
    L: float, W: float, n_grid: int = N_GRID_DEFAULT, margin: float = MARGIN_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, Y)`` arrays of shape ``[n_grid, n_grid]`` matching the receiver
    layout from ``scripts/build_datasets.py`` (``X[iy, ix]``, ``Y[iy, ix]``).
    """
    xs = np.linspace(margin, L - margin, n_grid)
    ys = np.linspace(margin, W - margin, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")        # X varies with ix, Y with iy
    return X, Y


def analytical_mode_shape(
    n_x: int, n_y: int, L: float, W: float,
    n_grid: int = N_GRID_DEFAULT, margin: float = MARGIN_DEFAULT,
) -> np.ndarray:
    """``cos(n_x π x / L) · cos(n_y π y / W)`` evaluated on the receiver grid.

    Returns shape ``[n_grid, n_grid]`` real. Reuses
    ``aaf.sim.analytical_modal_2d._pair_shape``.
    """
    X, Y = receiver_grid_xy(L=L, W=W, n_grid=n_grid, margin=margin)
    return _pair_shape(int(n_x), int(n_y), X, Y, L=L, W=W)


def spatial_correlation_complex(P_pred: np.ndarray, P_ism: np.ndarray) -> float:
    """Magnitude of complex Pearson correlation between two complex 2D fields.

    Returns a value in ``[0, 1]``. We use ``|<a, b>| / (||a|| ||b||)`` on
    mean-subtracted complex vectors so the metric is invariant to global phase
    and magnitude scaling (which the model is free to choose).
    """
    a = P_pred.ravel() - P_pred.mean()
    b = P_ism.ravel() - P_ism.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.abs(np.vdot(a, b)) / denom)      # vdot = sum(a.conj() * b)


def node_match_score(
    P_pred: np.ndarray, P_ism: np.ndarray, threshold_db: float = -20.0,
) -> float:
    """Fraction of ISM-node positions where predicted magnitude is also below threshold.

    A position is an "ISM node" if ``|P_ism|`` there is at least ``-threshold_db``
    below ``max |P_ism|``. The score is the fraction of those positions where
    ``|P_pred|`` is also at least that far below ``max |P_pred|``. If ISM has
    no nodes (i.e., the mode is uniform), returns ``nan``.
    """
    eps = 1e-12
    ism_mag = np.abs(P_ism).ravel()
    pred_mag = np.abs(P_pred).ravel()
    ism_max = float(ism_mag.max() + eps)
    pred_max = float(pred_mag.max() + eps)
    threshold_lin = 10 ** (float(threshold_db) / 20.0)
    ism_node_mask = ism_mag <= threshold_lin * ism_max
    if not ism_node_mask.any():
        return float("nan")
    pred_below = pred_mag <= threshold_lin * pred_max
    return float(np.mean(pred_below[ism_node_mask]))


def mode_shape_fit_error(
    P: np.ndarray, n_x: int, n_y: int, L: float, W: float,
    n_grid: int = N_GRID_DEFAULT, margin: float = MARGIN_DEFAULT,
) -> dict:
    """Least-squares fit of ``α · cos(n_xπx/L) cos(n_yπy/W)`` to ``|P|`` on the grid.

    Returns ``{'alpha': complex, 'residual_rms': float, 'snr_db': float}``.
    ``snr_db`` = ``20·log10(||α·Φ|| / ||P - α·Φ||)`` — large = good fit.
    """
    Phi = analytical_mode_shape(n_x, n_y, L=L, W=W, n_grid=n_grid, margin=margin)
    Phi_flat = Phi.ravel().astype(np.float64)
    P_flat = P.ravel().astype(np.complex128)
    # Least-squares scalar α minimising ||P - α Φ||^2 (complex P, real Φ).
    phi_norm_sq = float(Phi_flat @ Phi_flat) + 1e-12
    alpha = complex((P_flat * Phi_flat).sum() / phi_norm_sq)
    fit = alpha * Phi_flat
    residual = P_flat - fit
    res_rms = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
    fit_rms = float(np.sqrt(np.mean(np.abs(fit) ** 2)))
    snr_db = float(20.0 * np.log10((fit_rms + 1e-12) / (res_rms + 1e-12)))
    return {"alpha": alpha, "residual_rms": res_rms, "snr_db": snr_db}


def pick_first_modes(
    L: float, W: float, n_modes: int = 6, f_min: float = 1.0, f_max: float = 150.0,
    c: float = 343.0,
) -> list[tuple[int, int, float]]:
    """First ``n_modes`` distinct eigenfrequencies in ``(f_min, f_max)`` Hz.

    Returns list of ``(n_x, n_y, f)`` tuples. For degenerate frequencies (e.g.,
    L≈W with multiplicity > 1), the lexicographically smallest ``(n_x, n_y)``
    pair is taken — this gives an unambiguous mode shape to compare against.
    """
    eigs = eigenfrequencies_2d(L=L, W=W, c=c, f_max=f_max)
    out: list[tuple[int, int, float]] = []
    for e in eigs:
        if e.f < f_min or e.f > f_max:
            continue
        n_x, n_y = sorted(e.pairs)[0]                # smallest (n_x, n_y) pair
        if n_x == 0 and n_y == 0:
            continue                                  # skip DC
        out.append((n_x, n_y, float(e.f)))
        if len(out) >= n_modes:
            break
    return out
