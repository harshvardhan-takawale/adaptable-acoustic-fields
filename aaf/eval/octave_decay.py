"""P4-10 (D195, D197 A7): octave-band decay metrics (T30, EDT, C50) from band-limited complex spectra.

Input H [R, 601]: bins 0-600 at 0.5 Hz (a 2 s record). Per octave band (63 / 125 / 250 Hz; the 250 Hz band is clipped at
300 Hz) the spectrum gets a 10 Hz raised-cosine taper at each edge, is inverse transformed at 600 Hz sample rate, and the
decay is read from the energy decay curve (reverse-integrated within a 1.5 s gate after the onset). The ONSET is the first
sample whose running |h| reaches 10% of the maximum within the first 0.4 s (the source pulse has a 0.04 s delay).
T30: least-squares slope of the EDC from -5 to -35 dB (x 2 for a 60 dB extrapolation); EDT: 0 to -10 dB (x 6); C50: early
(0-50 ms) over late energy within the gate, in dB. Validity is judged on the TRUTH (A7): the truth's EDC must reach -35 dB
at least 0.2 s before the gate ends, else the receiver-band is NaN for every field.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

BANDS = {"63": (44.0, 88.0), "125": (88.0, 177.0), "250": (177.0, 300.0)}
FS, NT = 600.0, 1200                                        # 2 s at 600 Hz
GATE_S, EDGE_HZ = 1.5, 10.0


def _taper(lo, hi):
    f = np.arange(601) * 0.5
    w = np.zeros(601)
    inside = (f >= lo) & (f <= hi)
    up = np.clip((f - (lo - EDGE_HZ)) / EDGE_HZ, 0, 1)
    dn = np.clip(((hi + EDGE_HZ) - f) / EDGE_HZ, 0, 1)
    w = np.minimum(up, dn)
    return np.sin(0.5 * np.pi * w) ** 2 * ((f >= lo - EDGE_HZ) & (f <= hi + EDGE_HZ)) if inside.any() else w


def octave_ir(H: np.ndarray, band: str) -> np.ndarray:
    """[R, 601] complex bins -> [R, NT] band-limited impulse responses"""
    lo, hi = BANDS[band]
    return np.fft.irfft(np.asarray(H) * _taper(lo, hi), n=NT, axis=-1)


def _onset(h: np.ndarray) -> np.ndarray:
    """per row: index of the first sample with |h| >= 0.1 max within the first 0.4 s of the record"""
    m = np.abs(h)
    thr = 0.1 * m[:, : int(0.4 * FS)].max(1, keepdims=True)
    return np.argmax(m >= thr, axis=1)


def _edc(seg: np.ndarray) -> np.ndarray:
    e = np.cumsum(seg[:, ::-1] ** 2, axis=1)[:, ::-1]
    return 10 * np.log10(np.maximum(e / np.maximum(e[:, :1], 1e-300), 1e-12))


def decay_metrics(H: np.ndarray, band: str) -> Dict[str, np.ndarray]:
    """{T30, EDT, C50, t35} per receiver (T30 in s; EDT in s; C50 in dB; t35 = time at which the EDC reaches -35 dB)."""
    h = octave_ir(H, band)
    R = len(h)
    n_gate = int(GATE_S * FS)
    on = _onset(h)
    seg = np.zeros((R, n_gate))
    for r in range(R):
        s = h[r, on[r]:on[r] + n_gate]
        seg[r, :len(s)] = s
    L = _edc(seg)
    t = np.arange(n_gate) / FS

    def fit(a, b, scale):
        out = np.full(R, np.nan)
        for r in range(R):
            i0 = np.argmax(L[r] <= a)
            i1 = np.argmax(L[r] <= b)
            if L[r, i1] > b or i1 <= i0 + 3:
                continue
            A = np.polyfit(t[i0:i1], L[r, i0:i1], 1)
            if A[0] < 0:
                out[r] = -60.0 / A[0] * scale
        return out
    i35 = np.array([np.argmax(L[r] <= -35.0) if (L[r] <= -35.0).any() else n_gate for r in range(R)])
    t35 = np.where(i35 < n_gate, t[np.minimum(i35, n_gate - 1)], np.nan)
    n50 = int(0.05 * FS)
    early, late = (seg[:, :n50] ** 2).sum(1), (seg[:, n50:] ** 2).sum(1)
    return {"T30": fit(-5.0, -35.0, 1.0), "EDT": fit(0.0, -10.0, 1.0), "T20": fit(-5.0, -25.0, 1.0),
            "C50": 10 * np.log10(np.maximum(early, 1e-30) / np.maximum(late, 1e-30)), "t35": t35}


def valid_mask(t35: np.ndarray) -> np.ndarray:
    """receivers whose truth EDC reaches -35 dB at least 0.2 s before the gate ends (A7)"""
    return np.isfinite(t35) & (t35 <= GATE_S - 0.2)
