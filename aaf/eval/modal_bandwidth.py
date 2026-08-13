"""Robust per-mode peak level and -3 dB bandwidth.

Bandwidth is THE headline observable for P3-2: making a wall absorbent broadens the mode
families that reflect off it, and the measured wall-selectivity is ~50:1 on bandwidth
versus only ~4:1 on peak level (D47). So the estimator has to be trustworthy at the few-Hz
scale -- at fs=4096 / N=8192 the bin spacing is 0.5 Hz and a baseline modal bandwidth is
only ~7 bins wide.

Why this is not ``modal_verifier.pick_peaks``: that function computes a -3 dB width inline
but (a) **discards it**, keeping only ``q_factor``, (b) walks outward with **no distance
cap**, so on real 2D ground truth it strides across neighbouring modes and returns widths
of 13-172 Hz, (c) does no sub-bin interpolation, quantizing every width to a multiple of
0.5 Hz, and (d) fabricates ``bw = df`` when the walk collapses, turning an unresolvable
peak into a confident number. Each of those corrupts exactly the quantity we are measuring,
so P3-2 uses this module instead. ``pick_peaks`` is left untouched for P2/P3-1
reproducibility.

Fixes here: capped walk, parabolic sub-bin peak, linearly interpolated -3 dB crossings, an
explicit resolvability floor, and ``nan`` + a flag instead of a fabricated width.

Validation (see tests/test_modal_bandwidth.py): for the analytic modal model, every mode's
damping is known in closed form, so ``bandwidth == gamma / pi`` exactly -- e.g. L=4.5,
W=4.0, alpha=0.15 gives gamma=12.148 1/s and BW=3.867 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Flags
OK = "ok"
CAP_LEFT = "cap_left"
CAP_RIGHT = "cap_right"
EDGE = "edge"
FLOOR = "floor"
NO_PEAK = "no_peak"

DEFAULT_CAP_HZ = 20.0
DEFAULT_SEARCH_HZ = 2.0
DEFAULT_MIN_BINS = 2.0


@dataclass
class ModalPeak:
    """Measurement of one mode's spectral peak."""

    n_x: int
    n_y: int
    family: str
    f_mode: float          # analytic eigenfrequency (Hz)
    f_peak: float          # sub-bin interpolated peak location (Hz), nan if not found
    level_db: float        # sub-bin interpolated peak level (dB), nan if not found
    bw_3db_hz: float       # nan unless bw_valid
    bw_valid: bool
    bw_flag: str
    q_factor: float        # nan unless bw_valid
    cap_hz: float

    def to_dict(self) -> dict:
        return asdict(self)


def _to_db(mag: np.ndarray, floor_db: float = -200.0) -> np.ndarray:
    m = np.abs(np.asarray(mag, dtype=float))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(m, 1e-30))
    return np.maximum(db, floor_db)


def _parabolic_vertex(y_lo: float, y_mid: float, y_hi: float) -> Tuple[float, float]:
    """Vertex of the parabola through (-1,y_lo), (0,y_mid), (1,y_hi).

    Returns ``(delta_bins, level_offset)`` where the peak sits at ``i + delta_bins`` and
    the interpolated level is ``y_mid + level_offset``. Degenerate curvature -> (0, 0).
    """
    denom = y_lo - 2.0 * y_mid + y_hi
    if abs(denom) < 1e-12:
        return 0.0, 0.0
    delta = 0.5 * (y_lo - y_hi) / denom
    if not np.isfinite(delta) or abs(delta) > 1.0:
        return 0.0, 0.0
    return float(delta), float(-0.25 * (y_lo - y_hi) * delta)


def _interp_crossing(f0: float, db0: float, f1: float, db1: float, target: float) -> float:
    """Linear-in-dB interpolation of the frequency where the curve crosses ``target``."""
    if db0 == db1:
        return f1
    t = (db0 - target) / (db0 - db1)
    return float(f0 + t * (f1 - f0))


def peak_level_and_bw(
    mag: np.ndarray,
    f_axis: np.ndarray,
    f_mode: float,
    *,
    search_hz: float = DEFAULT_SEARCH_HZ,
    cap_hz: float = DEFAULT_CAP_HZ,
    drop_db: float = 3.0,
    min_bins: float = DEFAULT_MIN_BINS,
) -> Tuple[float, float, float, str]:
    """Measure one modal peak. Returns ``(bw_hz, level_db, f_peak_hz, flag)``.

    ``bw_hz`` is ``nan`` whenever the width is not trustworthy; ``flag`` says why. Never
    returns a fabricated width.

    ``cap_hz`` bounds the outward -3 dB walk on EACH side. Set it from the distance to the
    nearest other mode (see :func:`caps_from_mode_spacing`) so the walk cannot stride into
    a neighbouring resonance.
    """
    mag = np.asarray(mag, dtype=float)
    f_axis = np.asarray(f_axis, dtype=float)
    if mag.ndim != 1 or mag.shape != f_axis.shape:
        raise ValueError(f"mag {mag.shape} and f_axis {f_axis.shape} must be 1-D and equal")
    if mag.size < 3:
        return float("nan"), float("nan"), float("nan"), NO_PEAK

    db = _to_db(mag)
    df = float(f_axis[1] - f_axis[0])

    # --- locate the peak bin within +/- search_hz of the analytic frequency -------------
    lo = int(np.searchsorted(f_axis, f_mode - search_hz, side="left"))
    hi = int(np.searchsorted(f_axis, f_mode + search_hz, side="right"))
    lo, hi = max(lo, 0), min(hi, db.size)
    if hi - lo < 1:
        return float("nan"), float("nan"), float("nan"), NO_PEAK
    i = int(lo + np.argmax(db[lo:hi]))
    if i <= 0 or i >= db.size - 1:
        return float("nan"), float("nan"), float("nan"), EDGE

    # --- sub-bin peak (parabolic in dB) -------------------------------------------------
    delta, lvl_off = _parabolic_vertex(db[i - 1], db[i], db[i + 1])
    f_peak = float(f_axis[i] + delta * df)
    level_db = float(db[i] + lvl_off)
    target = level_db - drop_db

    max_steps = max(1, int(np.ceil(cap_hz / df)))

    # --- walk left ----------------------------------------------------------------------
    j = i
    while j > 0 and db[j] > target:
        if i - j >= max_steps:
            return float("nan"), level_db, f_peak, CAP_LEFT
        j -= 1
    if db[j] > target:                       # ran off the start of the array
        return float("nan"), level_db, f_peak, EDGE
    f_left = _interp_crossing(f_axis[j], db[j], f_axis[j + 1], db[j + 1], target) if j < i \
        else f_axis[j]

    # --- walk right ---------------------------------------------------------------------
    k = i
    while k < db.size - 1 and db[k] > target:
        if k - i >= max_steps:
            return float("nan"), level_db, f_peak, CAP_RIGHT
        k += 1
    if db[k] > target:                       # ran off the end
        return float("nan"), level_db, f_peak, EDGE
    f_right = _interp_crossing(f_axis[k], db[k], f_axis[k - 1], db[k - 1], target) if k > i \
        else f_axis[k]

    bw = float(f_right - f_left)
    if not np.isfinite(bw) or bw <= 0.0 or bw < min_bins * df:
        # Unresolvable at this record length -- report nan, do NOT clamp to one bin.
        return float("nan"), level_db, f_peak, FLOOR
    return bw, level_db, f_peak, OK


def caps_from_predicted_bw(
    bw_pred_hz: Sequence[float],
    *,
    frac: float = 3.0,
    cap_min_hz: float = 4.0,
    cap_max_hz: float = 40.0,
) -> List[float]:
    """Per-mode walk cap scaled to the width we EXPECT (``frac`` x predicted -3 dB BW).

    This is the right rule for modal-projected spectra, where neighbouring modes have
    already been suppressed spatially so the only thing the cap must do is stop a runaway
    walk. Use it whenever a predicted bandwidth is available (from
    ``analytical_modal_2d.modal_damping_2d`` -> ``damping_to_bandwidth_hz``): absorbent
    configs have genuinely wide modes (M3 x-axial is ~13.8 Hz) and a spacing-based cap
    would reject exactly the configs the chunk is about.
    """
    return [
        float(np.clip(frac * float(b), cap_min_hz, cap_max_hz)) if np.isfinite(b)
        else float(cap_max_hz)
        for b in bw_pred_hz
    ]


def caps_from_mode_spacing(
    f_modes: Sequence[float],
    *,
    frac: float = 0.45,
    cap_max_hz: float = DEFAULT_CAP_HZ,
    cap_min_hz: float = 1.0,
) -> List[float]:
    """Per-mode walk cap = ``frac`` x distance to the nearest OTHER mode, clipped.

    For measuring a **single-receiver** spectrum, where neighbouring modes are still
    present and the walk must be kept inside the mode's own resonance. Note this rule is
    intentionally conservative and will return ``nan`` (flag ``cap_*``) for strongly
    damped, overlapping modes -- prefer :func:`caps_from_predicted_bw` on projected
    spectra.
    """
    f = np.asarray(list(f_modes), dtype=float)
    caps: List[float] = []
    for idx in range(f.size):
        others = np.delete(f, idx)
        d = float(np.min(np.abs(others - f[idx]))) if others.size else cap_max_hz
        caps.append(float(np.clip(frac * d, cap_min_hz, cap_max_hz)))
    return caps


def measure_modes(
    spectra: np.ndarray,
    f_axis: np.ndarray,
    modes: Sequence,
    *,
    caps: Optional[Sequence[float]] = None,
    search_hz: float = DEFAULT_SEARCH_HZ,
    drop_db: float = 3.0,
    min_bins: float = DEFAULT_MIN_BINS,
) -> List[ModalPeak]:
    """Measure every mode.

    ``spectra`` is ``[n_modes, n_freq]`` (one modal-projected spectrum per mode, from
    :mod:`aaf.eval.modal_projection`) or ``[n_freq]`` (a single receiver spectrum reused
    for every mode). ``modes`` items need ``.n_x``, ``.n_y``, ``.f``, ``.family``.

    ``caps`` defaults to a uniform :data:`DEFAULT_CAP_HZ`, which suits projected spectra.
    For single-receiver spectra pass :func:`caps_from_mode_spacing`; when a predicted
    bandwidth is available prefer :func:`caps_from_predicted_bw`.
    """
    spectra = np.asarray(spectra)
    if caps is None:
        caps = [DEFAULT_CAP_HZ] * len(list(modes))
    out: List[ModalPeak] = []
    for idx, m in enumerate(modes):
        mag = spectra if spectra.ndim == 1 else spectra[idx]
        bw, lvl, fpk, flag = peak_level_and_bw(
            np.abs(mag), f_axis, float(m.f),
            search_hz=search_hz, cap_hz=float(caps[idx]),
            drop_db=drop_db, min_bins=min_bins,
        )
        valid = bool(np.isfinite(bw))
        out.append(
            ModalPeak(
                n_x=int(m.n_x), n_y=int(m.n_y), family=str(m.family),
                f_mode=float(m.f), f_peak=fpk, level_db=lvl,
                bw_3db_hz=bw, bw_valid=valid, bw_flag=flag,
                q_factor=(float(m.f) / bw if valid and bw > 0 else float("nan")),
                cap_hz=float(caps[idx]),
            )
        )
    return out
