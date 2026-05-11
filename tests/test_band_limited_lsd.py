"""Band-limited LSD metric: indices, attribution, edge cases."""
from __future__ import annotations

import numpy as np
import pytest

from aaf.eval.band_limited import (
    DEFAULT_BANDS,
    band_indices,
    compute_band_limited_metrics,
)


def test_band_indices_modal():
    # fs=4096, n_time=8192 -> df = 0.5 Hz, n_freq_bins=4097
    lo, hi = band_indices(fs=4096.0, n_freq_bins=4097, lo_hz=0.0, hi_hz=250.0)
    assert lo == 0
    assert hi == 501  # 250 / 0.5 = 500 -> +1 inclusive


def test_band_indices_clamps_to_grid():
    lo, hi = band_indices(fs=4096.0, n_freq_bins=4097, lo_hz=0.0, hi_hz=10000.0)
    assert lo == 0
    assert hi == 4097


def test_band_indices_empty_band_raises():
    with pytest.raises(ValueError):
        band_indices(fs=4096.0, n_freq_bins=4097, lo_hz=500.0, hi_hz=400.0)


def test_lsd_attribution_to_correct_band():
    """If we add a known magnitude offset only inside one band, the LSD for
    that band must show it and the other bands must stay near zero."""
    fs = 4096.0
    n_freq = 4097
    n_recv = 4
    rng = np.random.default_rng(0)
    H_target = rng.normal(size=(n_recv, n_freq)) + 1j * rng.normal(size=(n_recv, n_freq))
    H_target = H_target.astype(np.complex64)
    # Avoid near-zero magnitudes that blow up LSD.
    H_target = H_target / (np.abs(H_target) + 1e-3) * 1.0

    # Put a 6 dB magnitude offset in the 250-500 Hz band only.
    lo_i, hi_i = band_indices(fs, n_freq, 250.0, 500.0)
    H_pred = H_target.copy()
    H_pred[..., lo_i:hi_i] *= 10.0 ** (6.0 / 20.0)  # +6 dB

    out = compute_band_limited_metrics(H_pred, H_target, fs, n_freq, DEFAULT_BANDS)
    # The 250-500 Hz band should show ~6 dB.
    assert out["lsd_band_250_500_db"] == pytest.approx(6.0, abs=0.05)
    # 0-250 Hz and 500-2000 Hz should be ~0.
    assert out["lsd_band_0_250_db"] == pytest.approx(0.0, abs=0.05)
    assert out["lsd_band_500_2000_db"] == pytest.approx(0.0, abs=0.05)
    # Full-band LSD is the average over [0, 2000] = 4001 bins; 500 of them
    # are 6 dB and the rest are 0 dB -> mean ~ 6 * 500/4001 ≈ 0.75 dB.
    expected_full = 6.0 * 500 / 4001
    assert out["lsd_band_0_2000_db"] == pytest.approx(expected_full, abs=0.05)


def test_compute_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_band_limited_metrics(
            np.zeros((4, 100), dtype=np.complex64),
            np.zeros((4, 99), dtype=np.complex64),
            fs=4096.0, n_freq_bins=100, bands=[(0.0, 100.0)],
        )


def test_compute_metrics_records_n_bins():
    out = compute_band_limited_metrics(
        np.ones((2, 4097), dtype=np.complex64),
        np.ones((2, 4097), dtype=np.complex64),
        fs=4096.0, n_freq_bins=4097, bands=[(0.0, 250.0)],
    )
    assert out["lsd_band_0_250_n_bins"] == 501
    assert out["lsd_band_0_250_db"] == pytest.approx(0.0, abs=1e-6)
