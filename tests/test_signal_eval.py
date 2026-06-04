"""Tests for the signal-level eval suite.

For each Layer-1 function, verify that:
  - Identical inputs → ideal score (correlation 1.0, LSD 0.0, EDC delta 0.0).
  - Perturbed inputs → score moves the expected direction.
"""
import numpy as np
import pytest

from aaf.eval.signal_level import (
    compute_signal_metrics,
    early_late_corr,
    edc_db,
    edc_error,
    envelope_corr,
    magnitude_correlation,
    per_band_lsd,
    phase_correlation_mag_weighted,
    rir_pearson,
)


FS = 4096
N_TIME = 4096                  # smaller in tests for speed
N_FREQ = N_TIME // 2 + 1


def _synthesize_rir(n_rx=4, n_time=N_TIME, t60_samples=2000, seed=0):
    """Decaying-exponential white-noise RIRs (Schroeder-style)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    env = np.exp(-3.0 * np.log(10.0) * t / t60_samples)        # decays to -60 dB
    noise = rng.standard_normal((n_rx, n_time))
    rir = (env * noise).astype(np.float32)
    return rir


def _to_H(rir):
    return np.fft.rfft(rir, axis=-1).astype(np.complex64)


def test_identical_signals_give_unit_correlations():
    rir = _synthesize_rir(n_rx=4)
    H = _to_H(rir)
    assert np.isclose(magnitude_correlation(H, H), 1.0, atol=1e-5)
    assert np.isclose(phase_correlation_mag_weighted(H, H), 1.0, atol=1e-5)
    assert np.isclose(rir_pearson(rir, rir), 1.0, atol=1e-5)
    assert np.isclose(envelope_corr(rir, rir), 1.0, atol=1e-5)
    e, l = early_late_corr(rir, rir, fs=FS)
    assert np.isclose(e, 1.0, atol=1e-5)
    assert np.isclose(l, 1.0, atol=1e-5)


def test_identical_signals_lsd_zero():
    rir = _synthesize_rir(n_rx=2)
    H = _to_H(rir)
    bands = ((0.0, 250.0), (250.0, 500.0))
    d = per_band_lsd(H, H, fs=FS, n_freq_bins=N_FREQ, bands=bands)
    assert abs(d["lsd_band_0_250_db"]) < 1e-3
    assert abs(d["lsd_band_250_500_db"]) < 1e-3


def test_phase_shift_drops_phase_keeps_magnitude():
    """If we add a constant phase term to all of H_pred, |H| is unchanged but
    phase agreement drops; magnitude_correlation should remain ≈ 1.0 and
    phase_correlation_mag_weighted should drop."""
    rir = _synthesize_rir(n_rx=2)
    H = _to_H(rir)
    # Apply a global half-cycle shift to predicted.
    H_pred = H * np.exp(1j * np.pi / 2)
    assert np.isclose(magnitude_correlation(H_pred, H), 1.0, atol=1e-5)
    assert phase_correlation_mag_weighted(H_pred, H) < 0.1


def test_noisy_perturbation_lowers_all_correlations():
    """Adding strong white noise to rir_pred should lower every correlation."""
    rir = _synthesize_rir(n_rx=2, seed=0)
    rir_noisy = rir + 5.0 * _synthesize_rir(n_rx=2, seed=1)
    H = _to_H(rir)
    H_noisy = _to_H(rir_noisy)
    assert magnitude_correlation(H_noisy, H) < 0.99
    assert rir_pearson(rir_noisy, rir) < 0.99
    assert envelope_corr(rir_noisy, rir) < 0.99


def test_edc_monotone_decreasing():
    """The Schroeder integration is always non-increasing in time."""
    rir = _synthesize_rir(n_rx=1)
    edc = edc_db(rir, fs=FS)
    assert edc.shape == rir.shape
    diff = np.diff(edc[0])
    # The reverse cumulative sum cannot increase (energy only removed).
    assert (diff <= 1e-6).all(), (
        f"EDC not monotone: max increase={float(diff.max()):.4f}"
    )


def test_edc_error_zero_for_identical_signals():
    rir = _synthesize_rir(n_rx=2)
    err = edc_error(rir, rir, fs=FS)
    assert abs(err["edc_max_db"]) < 1e-3
    assert abs(err["edc_rmse_db"]) < 1e-3


def test_compute_signal_metrics_returns_expected_keys():
    rir = _synthesize_rir(n_rx=4)
    H = _to_H(rir)
    out = compute_signal_metrics(
        H, H, fs=FS, n_time_samples=N_TIME,
        rir_pred=rir, rir_target=rir,
    )
    for k in ("mag_corr", "phase_corr_mw", "rir_pearson",
              "edc_max_db", "edc_rmse_db", "early_corr", "late_corr",
              "envelope_corr"):
        assert k in out, f"missing {k}"
    # Identical signals → all corrs close to 1.
    assert out["mag_corr"] > 0.99
    assert out["rir_pearson"] > 0.99
