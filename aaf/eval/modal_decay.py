"""Band-limited decay metrics and per-mode decay rates.

Material editing IS a change of decay, so P3-2 needs decay measured *in band* and *per
mode*. Neither exists in the repo: ``signal_level.edc_db`` is broadband,
``signal_level._t_reverse`` is private, averages over receivers before returning, and
returns ``nan`` indistinguishably from "no valid receiver".

Two estimators here:

``t20_band`` -- in-band T20/T30 with per-receiver output and a **truncation-knee guard**.
At ``max_order=60`` the image expansion stops contributing at ~0.8 s while the record is
2.0 s, so the energy-decay curve has a hard knee; fitting across it measures the
truncation, not the room. The fit is therefore restricted to the pre-knee region and the
knee time is reported.

``modal_decay_rate`` -- the theory-exact per-mode rate. Two traps, both measured:

* pyroomacoustics applies **1/d geometric spreading** even in 2D, so a mode's envelope
  decays like ``e^{-gamma t}/t``, not ``e^{-gamma t}``. Fitting without compensation gave
  14.7 1/s where theory says 6.19. ``spread_comp=True`` multiplies by t before fitting.
* Too narrow a spectral window makes the estimate saturate at the WINDOW's ringing rate
  (halfwidth 15 Hz returned 3.5 1/s where theory said 26.0; halfwidth 35 Hz returned
  23.5). The default halfwidth scales with the expected bandwidth.

With both handled, measured vs ISM-ray predicted agree to ~10% across materials
(e.g. mode (1,0) under west->M3: 23.5 measured vs 26.0 predicted).

``spread_comp`` must MATCH THE GENERATOR:

* pyroomacoustics ISM (what P3-2 trains and evaluates on) applies 1/d -> ``True``.
* a pure exponential envelope (a synthetic test signal) -> ``False``.

CAUTION -- do not use ``analytical_modal_2d.modal_rir_2d`` as a time-domain reference.
Its ``H_complex`` is a sum of Lorentzians that are symmetric in k, so the inverse FFT is
NOT causal: the response wraps to the end of the buffer and its Schroeder EDC sits at 0 dB
until t ~ N/fs. It remains an excellent FREQUENCY-domain reference (peak positions and
-3 dB widths, where gamma is known in closed form) and is used as such by
``tests/test_modal_bandwidth.py`` -- but every decay estimator here is validated against
causal synthetic signals and real ISM output instead.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from aaf.eval.band_limited import band_indices

EPS = 1e-20


def band_limited_rir(H: np.ndarray, fs: float, n_time: int, f_lo: float = 0.0,
                     f_hi: float = 300.0) -> np.ndarray:
    """Zero every bin outside [f_lo, f_hi] (identical mask on both sides), irfft.

    Same convention as ``aaf.eval.p3_1_eval.band_limited_rir`` -- the mask must be applied
    identically to prediction and target or the comparison is meaningless.
    """
    lo, hi = band_indices(fs, H.shape[-1], f_lo, f_hi)
    Hf = np.array(H, copy=True)
    Hf[..., :lo] = 0.0
    Hf[..., hi:] = 0.0
    Hf[..., 0] = Hf[..., 0].real
    return np.fft.irfft(Hf, n=n_time, axis=-1)


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def band_limited_edc(rir: np.ndarray, fs: float) -> np.ndarray:
    """Schroeder backward-integrated energy decay curve, dB, per receiver."""
    r = _as_2d(rir).astype(np.float64)
    rev = np.cumsum((r * r)[..., ::-1], axis=-1)[..., ::-1]
    return 10.0 * np.log10(rev / (rev[..., :1] + EPS) + EPS)


def truncation_knee(rir: np.ndarray, fs: float, *, win_ms: float = 10.0,
                    rel_floor: float = 1e-12) -> np.ndarray:
    """First time (s) at which a short forward window holds essentially no energy.

    Detects where the image-source expansion stopped contributing. Returns one time per
    receiver (``inf`` if the response never dies inside the record).
    """
    r = _as_2d(rir).astype(np.float64)
    n_win = max(1, int(round(win_ms * 1e-3 * fs)))
    p2 = r * r
    csum = np.cumsum(p2, axis=-1)
    win = csum[..., n_win:] - csum[..., :-n_win]           # energy in each window
    peak = np.max(p2, axis=-1, keepdims=True) * n_win + EPS
    dead = win < (rel_floor * peak)
    out = np.full(r.shape[0], np.inf)
    for i in range(r.shape[0]):
        idx = np.flatnonzero(dead[i])
        if idx.size:
            out[i] = idx[0] / float(fs)
    return out


def t20_band(
    H_or_rir: np.ndarray,
    fs: float,
    *,
    n_time: Optional[int] = None,
    f_lo: float = 0.0,
    f_hi: float = 300.0,
    ref_db: float = -5.0,
    drop_db: float = 20.0,
    min_fit_span_db: float = 15.0,
    is_spectrum: bool = True,
) -> dict:
    """In-band reverberation time, per receiver.

    Returns ``{t20, t60_from_t20, n_valid, frac_valid, knee_s, ref_db, drop_db}`` where
    ``t20`` is a per-receiver array with ``nan`` where the decay never spans the required
    range before the truncation knee. Reporting ``n_valid`` separately is the point: a
    receiver-averaged ``nan`` cannot be distinguished from "nothing was measurable".
    """
    if is_spectrum:
        if n_time is None:
            n_time = 2 * (np.asarray(H_or_rir).shape[-1] - 1)
        rir = band_limited_rir(np.asarray(H_or_rir), fs, n_time, f_lo, f_hi)
    else:
        rir = np.asarray(H_or_rir, dtype=float)
    rir = _as_2d(rir)

    edc = band_limited_edc(rir, fs)
    knee = truncation_knee(rir, fs)
    t = np.arange(rir.shape[-1]) / float(fs)

    t20 = np.full(rir.shape[0], np.nan)
    for i in range(rir.shape[0]):
        curve = edc[i]
        stop = curve.size if not np.isfinite(knee[i]) else max(2, int(knee[i] * fs))
        curve, tt = curve[:stop], t[:stop]
        if curve.size < 8:
            continue
        lo_t = np.flatnonzero(curve <= ref_db)
        hi_t = np.flatnonzero(curve <= ref_db - drop_db)
        if lo_t.size == 0 or hi_t.size == 0:
            continue
        i0, i1 = int(lo_t[0]), int(hi_t[0])
        if i1 - i0 < 4 or (curve[i0] - curve[i1]) < min_fit_span_db:
            continue
        slope = np.polyfit(tt[i0:i1 + 1], curve[i0:i1 + 1], 1)[0]   # dB/s, negative
        if slope >= 0:
            continue
        t20[i] = float(drop_db / (-slope))

    n_valid = int(np.sum(np.isfinite(t20)))
    return {
        "t20": t20,
        "t60_from_t20": t20 * 3.0,
        "n_valid": n_valid,
        "frac_valid": n_valid / float(t20.size),
        "knee_s": knee,
        "ref_db": ref_db,
        "drop_db": drop_db,
    }


def modal_decay_rate(
    a_mode: np.ndarray,
    f_mode: float,
    fs: float,
    n_time: int,
    *,
    halfwidth_hz: Optional[float] = None,
    gamma_prior: Optional[float] = None,
    spread_comp: bool = True,
    iters: int = 6,
    t_min_s: float = 0.03,
    t_max_s: float = 0.55,
) -> Tuple[float, dict]:
    """Decay rate gamma (1/s) of one mode, from its projected spectrum ``a_mode`` [n_freq].

    Windows the mode's own resonance, transforms to a complex analytic envelope, undoes the
    1/d spreading, and fits ``20 log10(|s(t)| * t)`` over an iteratively refined interval.

    ``halfwidth_hz`` defaults to ``clip(6 * gamma_prior / pi, 15, 40)``; pass
    ``gamma_prior`` from ``analytical_modal_2d.modal_damping_2d`` when available.
    """
    a_mode = np.asarray(a_mode)
    n_freq = a_mode.shape[-1]
    df = fs / float(n_time)

    if halfwidth_hz is None:
        g0 = 12.0 if gamma_prior is None else float(gamma_prior)
        halfwidth_hz = float(np.clip(6.0 * g0 / np.pi, 15.0, 40.0))

    lo = max(0, int(round((f_mode - halfwidth_hz) / df)))
    hi = min(n_freq, int(round((f_mode + halfwidth_hz) / df)) + 1)
    if hi - lo < 8:
        return float("nan"), {"reason": "window too narrow", "halfwidth_hz": halfwidth_hz}

    seg = np.zeros(n_freq, dtype=complex)
    m = hi - lo
    # Tukey(0.5): taper the window edges so the truncation does not ring into the envelope.
    w = np.ones(m)
    taper = max(1, int(0.25 * m))
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper) / taper))
    w[:taper] = ramp
    w[-taper:] = ramp[::-1]
    seg[lo:hi] = a_mode[lo:hi] * w

    # One-sided spectrum x2 -> complex analytic signal (envelope without half-cycle ripple).
    s = np.fft.ifft(np.concatenate([2.0 * seg, np.zeros(n_time - n_freq, dtype=complex)]))
    env = np.abs(s[:n_time])
    t = np.arange(n_time) / float(fs)

    y_lin = env * t if spread_comp else env
    with np.errstate(divide="ignore"):
        y = 20.0 * np.log10(np.maximum(y_lin, 1e-30))

    gamma = 12.0 if gamma_prior is None else float(gamma_prior)
    info: dict = {"halfwidth_hz": halfwidth_hz, "spread_comp": spread_comp}
    for it in range(max(1, iters)):
        t0 = max(t_min_s, 2.0 / max(gamma, 1e-6))
        t1 = min(t0 + 3.0 / max(gamma, 1e-6), t_max_s)
        i0, i1 = int(t0 * fs), int(t1 * fs)
        if i1 - i0 < 16:
            info.update({"reason": "fit window too short", "t0": t0, "t1": t1})
            return float("nan"), info
        tt, yy = t[i0:i1], y[i0:i1]
        good = np.isfinite(yy)
        if good.sum() < 16:
            info.update({"reason": "no finite samples"})
            return float("nan"), info
        slope, intercept = np.polyfit(tt[good], yy[good], 1)
        if slope >= 0:
            info.update({"reason": "non-decaying fit", "slope_db_s": float(slope)})
            return float("nan"), info
        gamma_new = float(-slope * np.log(10.0) / 20.0)          # dB/s -> 1/s
        pred = slope * tt[good] + intercept
        ss_res = float(np.sum((yy[good] - pred) ** 2))
        ss_tot = float(np.sum((yy[good] - yy[good].mean()) ** 2)) + EPS
        info.update({"r2": 1.0 - ss_res / ss_tot, "t0": t0, "t1": t1, "n_iter": it + 1})
        if abs(gamma_new - gamma) / max(gamma, 1e-6) < 0.01:
            gamma = gamma_new
            break
        gamma = gamma_new
    return float(gamma), info
