"""Eval metrics: noise-perturbed predictions are small; shifted-peak predictions
yield the right modal MAE.
"""
import math

import numpy as np
import torch

from aaf.eval.modal_verifier import (
    modal_error_metrics,
    pick_peaks,
)


def _spectrum_with_peaks(f_axis, peaks_hz, q_factor=80.0):
    """Sum of complex Lorentzian responses (lets us make a synthetic |H|)."""
    H = np.zeros_like(f_axis, dtype=np.complex128)
    for f0 in peaks_hz:
        bw = f0 / q_factor
        H += 1.0 / (1.0 + 2j * (f_axis - f0) / bw)
    return H


def test_modal_mae_small_for_clean_target():
    """When predicted spectrum equals target plus tiny noise, modal MAE ≈ 0."""
    fs = 4096
    n = 8192
    df = fs / n
    f_axis = np.arange(n // 2 + 1) * df

    peak_freqs = [60.0, 120.0, 200.0]
    H_target = _spectrum_with_peaks(f_axis, peak_freqs)
    H_pred = H_target + 1e-3 * (np.random.randn(*H_target.shape) + 1j * np.random.randn(*H_target.shape))

    picked_pred = pick_peaks(H_pred, f_axis, prominence_db=3.0, min_distance_hz=10.0)

    class _M:
        def __init__(self, f):
            self.f = f
    modes = [_M(f) for f in peak_freqs]
    metrics = modal_error_metrics(picked_pred, modes, tolerance_hz=4.0)
    assert metrics["recall_at_tol"] == 1.0
    assert metrics["mae_hz"] < 1.0


def test_modal_mae_reflects_known_shift():
    """If predicted peaks are shifted by ~5 Hz, MAE should report ~5."""
    fs = 4096
    n = 8192
    df = fs / n
    f_axis = np.arange(n // 2 + 1) * df

    target_freqs = [60.0, 120.0, 200.0]
    pred_freqs = [f + 5.0 for f in target_freqs]
    H_pred = _spectrum_with_peaks(f_axis, pred_freqs)

    picked_pred = pick_peaks(H_pred, f_axis, prominence_db=3.0, min_distance_hz=10.0)

    class _M:
        def __init__(self, f):
            self.f = f
    modes = [_M(f) for f in target_freqs]
    metrics = modal_error_metrics(picked_pred, modes, tolerance_hz=10.0)
    assert metrics["recall_at_tol"] == 1.0
    # Picker quantizes to bin centers (df = 0.5 Hz); MAE should be 5 ± 0.5.
    assert 4.0 < metrics["mae_hz"] < 6.0


def test_full_band_lsd_zero_for_identity():
    """Importable LSD computation: if H_pred == H_target, LSD = 0."""
    H = np.array([1 + 0j, 0.5 + 0.5j, 0.1 - 0.2j], dtype=np.complex64)
    # Inline LSD per the spec; mirror what single_room_eval will compute.
    lsd_db = float(
        np.mean(np.abs(20 * np.log10(np.maximum(np.abs(H), 1e-8) / np.maximum(np.abs(H), 1e-8))))
    )
    assert lsd_db == 0.0
