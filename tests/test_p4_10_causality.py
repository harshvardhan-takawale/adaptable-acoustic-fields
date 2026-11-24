"""P4-10 P-2 (D219): the causality measure behaves on synthetic impulse responses (Rule 8; numpy only)."""
from __future__ import annotations

import numpy as np

from scripts.p4_10_p2_audit import FS600, NT, UP, early_excess, ir_bandlimited, onset_offset

N = NT * UP
FS = FS600 * UP
DIST = np.linspace(3.0, 5.0, 60)


def _spectrum(h):
    """[R, N] pulse-centred time signal -> the stored 601-bin convention WITH the pulse delay (so `ir_bandlimited` must undo it)"""
    f = np.arange(601) * 0.5
    X = np.fft.rfft(h, axis=-1)[:, :601] / UP
    return X * np.exp(-2j * np.pi * f * 0.04)


def _causal(decay=0.15, floor=0.0, pre=0.0, seed=0):
    """a direct arrival (one sample) at d / 343 plus a decaying noise tail; pre = energy before the direct arrival"""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    h = np.zeros((len(DIST), N))
    for i, d in enumerate(DIST):
        td = d / 343.0
        h[i] = 0.1 * rng.standard_normal(N) * (t >= td) * np.exp(-(t - td) / decay) + floor * rng.standard_normal(N)
        h[i, int(round(td * FS))] += 40.0
        h[i] += pre * rng.standard_normal(N) * ((t > td - 0.02) & (t < td - 0.006))
    return h

def test_pulse_delay_is_removed_and_onset_is_at_t_direct():
    H = _spectrum(_causal())
    h = ir_bandlimited(H)
    off = onset_offset(h, DIST)
    assert np.all(np.abs(off) < 0.004), off                # band-limited (25-300 Hz): the 10% onset is within a few ms of the direct arrival
    h2 = ir_bandlimited(H, t0=0.0)                          # with the delay left in, the onset is 40 ms away
    assert not np.allclose(h, h2)


def test_causal_response_has_small_excess():
    h = ir_bandlimited(_spectrum(_causal()))
    ex, di, _ = early_excess(h, DIST)
    assert abs(ex.sum() / di.sum()) < 0.05, ex.sum() / di.sum()


def test_late_floor_alone_is_subtracted():
    """a stationary floor adds energy to the window; the subtraction removes most of it"""
    h0, h1 = ir_bandlimited(_spectrum(_causal())), ir_bandlimited(_spectrum(_causal(floor=1.2)))
    raw = lambda h: sum((h[i, : int((DIST[i] / 343 - 0.005) * FS)] ** 2).sum() for i in range(len(h)))  # noqa: E731
    e0, d0, _ = early_excess(h0, DIST)
    e1, d1, _ = early_excess(h1, DIST)
    added_raw = raw(h1) / d1.sum() - raw(h0) / d0.sum()
    added_excess = e1.sum() / d1.sum() - e0.sum() / d0.sum()
    assert added_raw > 0.02                                  # the floor is really there
    assert abs(added_excess) < 0.35 * added_raw, (added_raw, added_excess)

def test_energy_before_the_direct_sound_raises_the_measure():
    a = ir_bandlimited(_spectrum(_causal()))
    b = ir_bandlimited(_spectrum(_causal(pre=2.0)))
    ea, da, _ = early_excess(a, DIST)
    eb, db, _ = early_excess(b, DIST)
    assert eb.sum() / db.sum() > ea.sum() / da.sum() + 0.01, (ea.sum() / da.sum(), eb.sum() / db.sum())


def test_a_no_op_measure_would_fail():
    h = ir_bandlimited(_spectrum(_causal(pre=2.0)))
    ex, di, _ = early_excess(h, DIST)
    assert ex.sum() / di.sum() > 0.02                       # if the window were empty (a vacuous measure) this would be exactly zero
