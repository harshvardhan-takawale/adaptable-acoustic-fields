"""Controlled tests for match_peaks_to_modes / modal_error_metrics."""
from dataclasses import dataclass

import math

from aaf.eval.modal_verifier import (
    Peak,
    match_peaks_to_modes,
    modal_error_metrics,
)


@dataclass
class _M:
    """Tiny mode stub with just .f."""
    f: float
    n_x: int = 0
    n_y: int = 0


def _peak(f, mag_db=0.0):
    return Peak(f=f, magnitude_db=mag_db, prominence_db=10.0, q_factor=100.0)


def test_perfect_match():
    modes = [_M(100.0), _M(200.0), _M(300.0)]
    picks = [_peak(100.0), _peak(200.0), _peak(300.0)]
    out = match_peaks_to_modes(picks, modes, tolerance_hz=4.0)
    assert len(out["matches"]) == 3
    assert out["spurious_peak_idx"] == []
    assert out["missed_mode_idx"] == []


def test_one_spurious_zero_missed():
    """6 picks, 5 modes, 1 spurious pick well outside any mode tol."""
    modes = [_M(100.0), _M(200.0), _M(300.0), _M(400.0), _M(500.0)]
    picks = [
        _peak(100.5),
        _peak(199.0),
        _peak(301.0),
        _peak(401.5),
        _peak(499.0),
        _peak(750.0),  # spurious
    ]
    metrics = modal_error_metrics(picks, modes, tolerance_hz=4.0)
    assert metrics["recall_at_tol"] == 1.0
    assert metrics["n_spurious"] == 1
    assert metrics["n_matched"] == 5
    # MAE of the matched peaks is mean(|0.5, 1, 1, 1.5, 1|) = 1.0
    assert math.isclose(metrics["mae_hz"], 1.0, abs_tol=1e-9)


def test_one_missed():
    """Mode at 250 has no nearby pick; should appear in missed list."""
    modes = [_M(100.0), _M(250.0), _M(400.0)]
    picks = [_peak(101.0), _peak(401.0)]  # no pick near 250
    out = match_peaks_to_modes(picks, modes, tolerance_hz=4.0)
    assert len(out["matches"]) == 2
    assert out["missed_mode_idx"] == [1]


def test_per_band_breakdown():
    """6 modes (low: 0-5, mid: 6-15), all matched perfectly."""
    modes = [_M(100.0 * i) for i in range(1, 17)]  # 16 modes; ordinal 0..15
    picks = [_peak(m.f) for m in modes]
    metrics = modal_error_metrics(picks, modes, tolerance_hz=4.0)

    assert metrics["per_mode_breakdown"]["low"]["n_modes"] == 6  # 0..5
    assert metrics["per_mode_breakdown"]["mid"]["n_modes"] == 10  # 6..15
    assert metrics["per_mode_breakdown"]["high"]["n_modes"] == 0
    assert metrics["per_mode_breakdown"]["low"]["recall"] == 1.0
    assert metrics["per_mode_breakdown"]["mid"]["recall"] == 1.0


def test_tolerance_pct_kicks_in_at_high_freq():
    """At 1500 Hz, tolerance_pct=0.02 → 30 Hz tolerance overrides 4 Hz hz floor."""
    modes = [_M(1500.0)]
    picks = [_peak(1520.0)]  # 20 Hz off; > 4 Hz, < 30 Hz
    out = match_peaks_to_modes(picks, modes, tolerance_hz=4.0, tolerance_pct=0.02)
    assert len(out["matches"]) == 1


def test_empty_inputs():
    assert match_peaks_to_modes([], [_M(100.0)]) == {
        "matches": [],
        "spurious_peak_idx": [],
        "missed_mode_idx": [0],
    }
    assert match_peaks_to_modes([_peak(100.0)], []) == {
        "matches": [],
        "spurious_peak_idx": [0],
        "missed_mode_idx": [],
    }
