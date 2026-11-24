"""P4-10 V7 (D211, D216): ISO 3382-1 style noise compensation (Lundeby truncation plus the energy of the fitted exponential after the truncation point) of a squared impulse-response segment.
Pure numpy; tested in tests/test_p4_10_lundeby.py."""
from __future__ import annotations

import numpy as np

W0, WMIN, WMAX, NOISE_FRAC, RANGE_DB, MIN_PN_DB, T_MIN, MAX_IT = 0.030, 0.010, 0.050, 0.10, 20.0, 20.0, 0.10, 5


def _blocks(e, fs, w):
    k = max(int(round(w * fs)), 1)
    m = len(e) // k
    E = e[:m * k].reshape(m, k).mean(1)
    return (np.arange(m) + 0.5) * k / fs, 10 * np.log10(np.maximum(E, 1e-300))


def lundeby(h, fs):
    """h: onset-shifted segment of one receiver-band. Returns (n_trunc, C, info) or None if it cannot be compensated (D216)."""
    n = len(h)
    e = np.asarray(h, float) ** 2
    nl = max(int(round(NOISE_FRAC * n)), 1)
    N = float(e[-nl:].mean())
    if not N > 0:
        N = 1e-300
    t, L = _blocks(e, fs, W0)
    p = int(np.argmax(L))
    Nd = 10 * np.log10(N)
    if L[p] - Nd < MIN_PN_DB:
        return None

    def line(t, L, lo, hi):
        """least-squares line over the blocks after the peak with lo <= level <= hi (the first block below lo ends the range)"""
        q = int(np.argmax(L))
        end = q
        while end < len(L) and L[end] > lo:
            end += 1
        idx = np.arange(q, end)
        idx = idx[L[idx] <= hi] if hi is not None else idx
        if len(idx) < 3:
            idx = np.arange(q, end)
        if len(idx) < 3:
            return None
        b, a = np.polyfit(t[idx], L[idx], 1)
        return (a, b) if b < 0 else None
    ln = line(t, L, Nd + 10.0, None)
    if ln is None:
        return None
    a, b = ln
    tc = (Nd - a) / b
    for _ in range(MAX_IT):
        w = float(np.clip((10.0 / -b) / 5.0, WMIN, WMAX))
        t, L = _blocks(e, fs, w)
        i0 = int(min(max((tc + 5.0 / -b) * fs, 0), n - nl))
        N = float(e[i0:].mean())
        Nd = 10 * np.log10(max(N, 1e-300))
        if L.max() - Nd < MIN_PN_DB:
            return None
        ln = line(t, L, Nd + 10.0, Nd + 10.0 + RANGE_DB)
        if ln is None:
            return None
        a, b = ln
        new = (Nd - a) / b
        done = abs(new - tc) < 0.005
        tc = new
        if done:
            break
    if tc < T_MIN:
        return None
    nt = int(np.clip(round(tc * fs), 1, n))
    lvl = a + b * (nt / fs)
    r = 10 ** (b / (10.0 * fs))
    C = 10 ** (lvl / 10.0) / (1.0 - r)
    return nt, float(C), {"tc": float(tc), "noise_db": float(Nd), "peak_to_noise_db": float(L.max() - Nd), "slope_db_per_s": float(b)}


def edc_db(h, fs, comp=True):
    """energy decay curve (dB re its own first sample) of one segment: raw (backward integral over the gate) or compensated. None if it cannot be compensated."""
    e = np.asarray(h, float) ** 2
    if not comp:
        c = np.cumsum(e[::-1])[::-1]
        return 10 * np.log10(np.maximum(c / max(c[0], 1e-300), 1e-12)), None
    r = lundeby(h, fs)
    if r is None:
        return None, None
    nt, C, info = r
    c = np.full(len(e), np.nan)
    c[:nt] = np.cumsum(e[:nt][::-1])[::-1] + C
    # after the truncation point the curve is the fitted exponential itself (not used by the fits unless they reach it)
    b = info["slope_db_per_s"]
    c[nt:] = C * 10 ** (b * (np.arange(nt, len(e)) - nt) / (10.0 * fs))
    return 10 * np.log10(np.maximum(c / max(c[0], 1e-300), 1e-12)), info


def fit(L, fs, a, b, scale):
    """octave_decay.py's fit: slope between the first samples at or below a and b dB, x scale to 60 dB"""
    if not (L <= b).any():
        return np.nan
    i0, i1 = int(np.argmax(L <= a)), int(np.argmax(L <= b))
    if i1 <= i0 + 3:
        return np.nan
    A = np.polyfit(np.arange(i0, i1) / fs, L[i0:i1], 1)
    return -60.0 / A[0] * scale if A[0] < 0 else np.nan


def times(L, fs, limit=None):
    """EDT, T20, T30; with `limit` (truncation sample) a metric whose range ends after it is kept (the curve continues as the fitted exponential)"""
    return {"EDT": fit(L, fs, 0.0, -10.0, 1.0), "T20": fit(L, fs, -5.0, -25.0, 1.0), "T30": fit(L, fs, -5.0, -35.0, 1.0)}
