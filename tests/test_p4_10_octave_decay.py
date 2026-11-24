"""P4-10 A7: octave-band decay metrics on synthetic decays of known T60 (behavioural, not a checkpoint test)."""
import numpy as np
import pytest

from aaf.eval.octave_decay import BANDS, FS, NT, decay_metrics, valid_mask


def _field(band, t60, n_modes=40, seed=0):
    """causal impulse response with modes in a band, all decaying with T60 (amplitude gamma = 6.908 / T60)"""
    rng = np.random.default_rng(seed)
    g = 6.908 / t60
    lo, hi = BANDS[band]
    t = np.arange(NT) / FS
    h = np.zeros(NT)
    f = np.linspace(lo + 5, min(hi, 295) - 5, n_modes)
    for k in range(n_modes):
        h += rng.normal() * np.exp(-g * t) * np.cos(2 * np.pi * f[k] * t + rng.uniform(0, 2 * np.pi))
    H = np.fft.rfft(np.concatenate([h, np.zeros(0)]), n=NT)
    return np.stack([np.fft.rfft(np.pad(np.roll(h, 0), (0, 0)))])[:, :601]


def test_t60_recovered():
    """the median over 12 independent mode sets (a room's receivers) recovers T60; EDT is noisier but in range"""
    for band, t60 in (("63", 0.8), ("125", 0.6), ("250", 0.5)):
        T, E = [], []
        for s in range(12):
            m = decay_metrics(_field(band, t60, seed=s), band)
            T.append(m["T30"][0])
            E.append(m["EDT"][0])
        assert abs(np.nanmedian(T) / t60 - 1) < 0.10, (band, np.nanmedian(T))
        assert abs(np.nanmedian(E) / t60 - 1) < 0.35, (band, np.nanmedian(E))


def test_slow_decay_is_flagged_invalid():
    """T60 2.5 s: the EDC does not reach -35 dB inside the 1.5 s gate -> T30 must be rejected, not extrapolated"""
    H = _field("125", 2.5)
    m = decay_metrics(H, "125")
    assert not valid_mask(m["t35"])[0]


def test_c50_orders_with_clarity():
    H_fast, H_slow = _field("125", 0.3), _field("125", 1.2)
    assert decay_metrics(H_fast, "125")["C50"][0] > decay_metrics(H_slow, "125")["C50"][0] + 2


def test_floor_rule_drops_a_model_with_a_late_floor_and_keeps_a_clean_decay():
    """D202: a clean decay keeps its receiver; the same decay plus a strong flat floor is dropped, a weak one is kept (the rule
    is a threshold on the floor's share of the EDC at the -25 dB point), and EDT is read above the floor either way"""
    from aaf.eval.octave_decay import floor_mask
    rng = np.random.default_rng(1)
    t = np.arange(NT) / FS
    h = np.exp(-6.908 / 0.6 * t) * np.sin(2 * np.pi * 125 * t)                # T60 0.6 s, 125 Hz
    H0 = np.fft.rfft(h)[:601][None]
    m0 = decay_metrics(H0, "125")
    noise = rng.normal(size=NT) * np.sqrt((h ** 2).mean())
    strong = decay_metrics(np.fft.rfft(h + 0.10 * noise)[:601][None], "125")
    weak = decay_metrics(np.fft.rfft(h + 0.01 * noise)[:601][None], "125")
    assert floor_mask(m0)[0] and floor_mask(weak)[0] and not floor_mask(strong)[0]
    assert m0["floor_db"][0] < weak["floor_db"][0] < strong["floor_db"][0]
    assert abs(weak["EDT"][0] / m0["EDT"][0] - 1) < 0.05                     # EDT is read over 0 to -10 dB, above the floor
