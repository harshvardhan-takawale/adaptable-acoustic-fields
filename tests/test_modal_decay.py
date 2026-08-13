"""Band-limited decay + per-mode decay rate. Pure CPU, no pyroomacoustics.

Validated against CAUSAL synthetic signals whose decay rate is known exactly.

Deliberately NOT validated against ``analytical_modal_2d.modal_rir_2d``: its Lorentzians
are symmetric in k, so its inverse FFT is non-causal (energy wraps to the end of the
buffer and the Schroeder EDC stays at 0 dB until t ~ N/fs). It is a frequency-domain
reference only -- see tests/test_modal_bandwidth.py, which uses it for bandwidth.
"""
from __future__ import annotations

import numpy as np
import pytest

from aaf.eval.modal_decay import (
    band_limited_edc,
    band_limited_rir,
    modal_decay_rate,
    t20_band,
    truncation_knee,
)

FS, N = 4096, 8192
T = np.arange(N) / FS


def _causal_mode(f0: float, gamma: float, spreading: bool = False, t_on: float = 0.005):
    """Causal decaying sinusoid; optionally with the ISM's 1/d geometric spreading."""
    env = np.exp(-gamma * T)
    if spreading:
        env = env / np.maximum(T, t_on)
        env[T < t_on] = 0.0
    sig = env * np.cos(2 * np.pi * f0 * T)
    return np.fft.rfft(sig, n=N)


def test_pure_exponential_recovered_without_spread_comp():
    gamma, f0 = 12.0, 76.0
    g, info = modal_decay_rate(_causal_mode(f0, gamma), f0, FS, N,
                               gamma_prior=gamma, spread_comp=False)
    assert g == pytest.approx(gamma, rel=0.08), (g, info)
    assert info["r2"] > 0.99


def test_spreading_envelope_needs_compensation():
    """With 1/d spreading, compensation recovers gamma and its absence biases HIGH."""
    gamma, f0 = 12.0, 76.0
    spec = _causal_mode(f0, gamma, spreading=True)
    g_on, info = modal_decay_rate(spec, f0, FS, N, gamma_prior=gamma, spread_comp=True)
    assert g_on == pytest.approx(gamma, rel=0.10), (g_on, info)
    g_off, _ = modal_decay_rate(spec, f0, FS, N, gamma_prior=gamma, spread_comp=False)
    assert g_off > g_on * 1.1, (g_off, g_on)


@pytest.mark.parametrize("gamma", [6.0, 12.0, 26.0])
def test_recovers_a_range_of_rates(gamma):
    g, _ = modal_decay_rate(_causal_mode(76.0, gamma), 76.0, FS, N,
                            gamma_prior=gamma, spread_comp=False)
    assert g == pytest.approx(gamma, rel=0.12)


def test_window_halfwidth_sensitivity_is_bounded():
    """Estimate must be stable across sensible windows (too narrow -> window ringing)."""
    gamma, f0 = 12.0, 76.0
    spec = _causal_mode(f0, gamma)
    rates = [modal_decay_rate(spec, f0, FS, N, gamma_prior=gamma, spread_comp=False,
                              halfwidth_hz=hw)[0] for hw in (20.0, 30.0, 40.0)]
    rates = [r for r in rates if np.isfinite(r)]
    assert len(rates) >= 2
    assert (max(rates) - min(rates)) / np.mean(rates) < 0.15


def _decaying_band_noise(gamma: float, n_rx: int = 8, seed: int = 0) -> np.ndarray:
    """Band-limited noise SHAPED by an exponential envelope.

    Order matters: band-limiting a fast-decaying burst makes the brick-wall filter's own
    ringing dominate the tail (apparent T60 then grows with gamma, which is backwards).
    Band-limit first, then impose the decay, so the envelope is what the estimator sees.
    """
    rng = np.random.default_rng(seed)
    noise = band_limited_rir(np.fft.rfft(rng.normal(size=(n_rx, N)), n=N, axis=-1),
                             FS, N, 0.0, 300.0)
    return noise * np.exp(-gamma * T)[None, :]


def test_t20_band_matches_the_known_decay_rate():
    """Amplitude rate gamma => EDC slope -20*gamma/ln10 dB/s => T60 = 6.91/gamma."""
    gamma = 12.0
    out = t20_band(_decaying_band_noise(gamma), FS, is_spectrum=False)
    assert out["n_valid"] >= 6, out["n_valid"]
    t60 = float(np.nanmedian(out["t60_from_t20"]))
    assert t60 == pytest.approx(6.91 / gamma, rel=0.20), (t60, 6.91 / gamma)


def test_more_absorption_decays_faster():
    t60 = {
        g: float(np.nanmedian(
            t20_band(_decaying_band_noise(g, seed=1), FS, is_spectrum=False)["t60_from_t20"]))
        for g in (4.0, 12.0, 30.0)
    }
    assert t60[4.0] > t60[12.0] > t60[30.0], t60
    for g, v in t60.items():
        assert v == pytest.approx(6.91 / g, rel=0.25), (g, v)


def test_t20_reports_per_receiver_and_validity_separately():
    rng = np.random.default_rng(2)
    rir = rng.normal(size=(64, N)) * np.exp(-12.0 * T)[None, :]
    out = t20_band(rir, FS, is_spectrum=False)
    assert out["t20"].shape[0] == 64, "T20 must be per-receiver, not pre-averaged"
    assert out["n_valid"] == int(np.sum(np.isfinite(out["t20"])))
    assert 0.0 <= out["frac_valid"] <= 1.0
    silent = t20_band(np.zeros((4, N)), FS, is_spectrum=False)
    assert silent["n_valid"] == 0, "silence must report n_valid=0, not a number"


def test_edc_is_monotone_and_starts_at_zero():
    rng = np.random.default_rng(3)
    rir = rng.normal(size=(4, N)) * np.exp(-12.0 * T)[None, :]
    edc = band_limited_edc(rir, FS)
    assert edc.shape == rir.shape
    assert np.allclose(edc[:, 0], 0.0, atol=1e-6)
    assert np.all(np.diff(edc, axis=-1) <= 1e-6), "Schroeder EDC must be non-increasing"


def test_truncation_knee_detects_a_hard_stop():
    rng = np.random.default_rng(4)
    rir = np.zeros((1, N))
    stop = 2000
    rir[0, :stop] = rng.normal(size=stop) * np.exp(-np.arange(stop) / 400.0)
    knee = truncation_knee(rir, FS)
    assert np.isfinite(knee[0])
    assert knee[0] == pytest.approx(stop / FS, abs=0.05)


def test_knee_guard_keeps_the_fit_off_the_truncated_tail():
    """A response that stops early must not have its T20 fitted across the cliff."""
    rng = np.random.default_rng(5)
    gamma, stop = 8.0, 3000
    rir = np.zeros((4, N))
    rir[:, :stop] = rng.normal(size=(4, stop)) * np.exp(-gamma * T[:stop])[None, :]
    out = t20_band(rir, FS, is_spectrum=False)
    if out["n_valid"]:
        t60 = float(np.nanmedian(out["t60_from_t20"]))
        assert t60 == pytest.approx(6.91 / gamma, rel=0.30), t60
