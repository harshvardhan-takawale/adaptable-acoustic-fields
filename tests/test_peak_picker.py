"""Synthesize a known sum of damped sinusoids, verify pick_peaks recovers them."""
import numpy as np
import pytest

from aaf.eval.modal_verifier import pick_peaks


def _spectrum_from_lorentzians(f_axis, peaks_hz, amps, q_factor=200.0):
    """Sum of Lorentzian magnitudes at the requested centre frequencies."""
    H = np.zeros_like(f_axis, dtype=np.complex128)
    for f0, A in zip(peaks_hz, amps):
        bw = f0 / q_factor
        H += A / (1 + 2j * (f_axis - f0) / bw)
    return H


def test_picks_4_known_peaks_within_1hz():
    fs = 4096
    n = 2048
    df = fs / n
    f_axis = np.arange(n // 2 + 1) * df

    peaks_hz = [100.0, 200.0, 300.0, 400.0]
    amps = [1.0, 0.8, 0.6, 0.4]
    H = _spectrum_from_lorentzians(f_axis, peaks_hz, amps, q_factor=300.0)

    picked = pick_peaks(H, f_axis, prominence_db=3.0, min_distance_hz=10.0)
    picked_f = sorted(p.f for p in picked)

    assert len(picked) == len(peaks_hz), f"expected {len(peaks_hz)} peaks, got {len(picked)}"
    for p_target, p_actual in zip(peaks_hz, picked_f):
        assert abs(p_target - p_actual) < 1.0, f"peak {p_target}: got {p_actual}, off by {abs(p_target-p_actual)}"


def test_min_distance_suppresses_close_peaks():
    """Two peaks 5 Hz apart with min_distance_hz=10 should be merged into 1."""
    fs = 4096
    n = 2048
    df = fs / n
    f_axis = np.arange(n // 2 + 1) * df

    peaks_hz = [200.0, 205.0]
    amps = [1.0, 1.0]
    H = _spectrum_from_lorentzians(f_axis, peaks_hz, amps, q_factor=500.0)

    picked = pick_peaks(H, f_axis, prominence_db=3.0, min_distance_hz=10.0)
    assert len(picked) == 1, f"expected 1 peak, got {len(picked)}"


def test_q_factor_estimate_in_range():
    """Moderately-Q peak (Q=20) should give an estimate in the right ballpark.

    With fs=4096, n=8192 (Δf=0.5 Hz) and a Q=20 Lorentzian at 500 Hz the
    half-power bandwidth is 25 Hz, well-resolved by 0.5 Hz spacing. Picker
    estimate should fall in [10, 60].
    """
    fs = 4096
    n = 8192
    df = fs / n
    f_axis = np.arange(n // 2 + 1) * df

    H = _spectrum_from_lorentzians(f_axis, [500.0], [1.0], q_factor=20.0)
    picked = pick_peaks(H, f_axis, prominence_db=3.0, min_distance_hz=10.0)

    assert len(picked) == 1
    q = picked[0].q_factor
    assert q is not None, "expected a non-None q_factor"
    assert 10 < q < 60, f"q_factor {q} not in (10, 60) for Q=20 Lorentzian"


def test_invalid_axis_raises():
    H = np.zeros(100, dtype=np.complex128)
    f = np.arange(50)  # mismatched length
    with pytest.raises(ValueError):
        pick_peaks(H, f, prominence_db=3.0, min_distance_hz=10.0)
