"""Band-limited LSD metrics.

Used by Track A of Chunk 3.6 to recompute zero-shot metrics on physically
meaningful frequency bands (modal regime 0-250 Hz, transition 250-500 Hz,
diffuse 500-2000 Hz) given saved zero-shot predictions. Also wired into
``aaf.eval.zero_shot`` so every future zero-shot run reports band metrics
alongside full-band LSD.

LSD formula matches the existing eval (mean over receivers AND freq bins of
``|20 * log10(|H_pred|/|H_target|)|``).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def band_indices(fs: float, n_freq_bins: int, lo_hz: float, hi_hz: float) -> tuple[int, int]:
    """Inclusive ``(lo_idx, hi_idx)`` such that ``f_axis[lo_idx:hi_idx]`` covers ``[lo_hz, hi_hz]``.

    ``f_axis = arange(n_freq_bins) * (fs / (2 * (n_freq_bins - 1)))`` matches the
    rfft layout used by FreqRenderer2D (``fs`` is the time-domain sample rate
    and ``n_freq_bins = n_time_samples // 2 + 1``).
    """
    if n_freq_bins <= 1:
        raise ValueError(f"n_freq_bins must be > 1, got {n_freq_bins}")
    n_time = 2 * (n_freq_bins - 1)
    df = float(fs) / float(n_time)
    lo_idx = max(0, int(round(float(lo_hz) / df)))
    hi_idx = min(n_freq_bins, int(round(float(hi_hz) / df)) + 1)
    if hi_idx <= lo_idx:
        raise ValueError(
            f"empty band [{lo_hz}, {hi_hz}] Hz at fs={fs}, n_freq_bins={n_freq_bins} "
            f"(lo_idx={lo_idx}, hi_idx={hi_idx})"
        )
    return lo_idx, hi_idx


def _lsd_db(H_pred: np.ndarray, H_target: np.ndarray, eps: float = 1e-8) -> float:
    """Mean ``|20*log10(|H_pred|/|H_target|)|`` over all elements."""
    num = np.maximum(np.abs(H_pred), eps)
    den = np.maximum(np.abs(H_target), eps)
    return float(np.mean(np.abs(20.0 * np.log10(num / den))))


def compute_band_limited_metrics(
    H_pred: np.ndarray,
    H_target: np.ndarray,
    fs: float,
    n_freq_bins: int,
    bands: Sequence[tuple[float, float]],
) -> dict:
    """Per-band LSD over the freq axis.

    Args:
        H_pred, H_target: complex arrays of identical shape, last dim is frequency
            and equals ``n_freq_bins``. Other dims are receivers / batches and are
            collapsed by the mean.
        fs: time-domain sample rate (Hz).
        n_freq_bins: must equal ``H_pred.shape[-1]``.
        bands: iterable of ``(lo_hz, hi_hz)`` tuples.

    Returns:
        ``{'lsd_band_<lo>_<hi>_db': float, 'lsd_band_<lo>_<hi>_n_bins': int, ...}``
        for each ``(lo, hi)`` band. Mean is computed across receivers and the bins
        within the band.
    """
    if H_pred.shape != H_target.shape:
        raise ValueError(
            f"H_pred {H_pred.shape} and H_target {H_target.shape} must match"
        )
    if H_pred.shape[-1] != n_freq_bins:
        raise ValueError(
            f"H_pred last-dim {H_pred.shape[-1]} != n_freq_bins {n_freq_bins}"
        )
    out: dict = {}
    for lo, hi in bands:
        lo_i, hi_i = band_indices(fs, n_freq_bins, lo, hi)
        sl = (..., slice(lo_i, hi_i))
        out[f"lsd_band_{int(lo)}_{int(hi)}_db"] = _lsd_db(H_pred[sl], H_target[sl])
        out[f"lsd_band_{int(lo)}_{int(hi)}_n_bins"] = hi_i - lo_i
    return out


DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 250.0),
    (250.0, 500.0),
    (500.0, 2000.0),
    (0.0, 2000.0),
)
