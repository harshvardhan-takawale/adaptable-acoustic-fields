"""Modal verifier: pick peaks in |H(f)|, match them to analytical eigenfreqs,
report MAE / recall / spurious counts. Reused for every model in Chunks 2-5.

Pipeline
--------
    H_complex, f_axis  ──►  pick_peaks  ──►  list[Peak]
                                              │
                       analytical Modes ──────┼──► match_peaks_to_modes
                                              │      └─► matched / spurious / missed
                                              ▼
                                        modal_error_metrics
                                              │
                                              ▼
                              {mae_hz, recall_at_tol, n_spurious, ...}

Notes
-----
- Picking is on log-magnitude (dB) so prominence threshold is meaningful
  across frequency-dependent floor levels.
- Matching is a Hungarian assignment with a tolerance window
  ``max(tolerance_hz, tolerance_pct * f_mode)`` so low-frequency modes get a
  Hz floor and high-frequency modes get a percentage band.
- Stratification: low / mid / high mode index based on ordinal position in
  the analytical list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


@dataclass
class Peak:
    f: float
    magnitude_db: float
    prominence_db: float
    q_factor: Optional[float]


@dataclass
class Match:
    mode_idx: int
    peak_idx: int
    f_mode: float
    f_peak: float
    delta_hz: float
    delta_pct: float


def pick_peaks(
    H_complex: np.ndarray,
    f_axis: np.ndarray,
    prominence_db: float = 3.0,
    min_distance_hz: float = 10.0,
    db_floor: float = -120.0,
) -> list[Peak]:
    """Pick peaks in 20·log10(|H|) using scipy.signal.find_peaks.

    Args:
        H_complex: 1-D array of complex H(f) values.
        f_axis: 1-D array of frequencies (Hz). Same length as H_complex.
        prominence_db: minimum prominence above local floor (dB).
        min_distance_hz: minimum spacing between adjacent peaks (Hz).
        db_floor: clamp |H| below this to db_floor when computing log-mag.

    Returns: list of Peak (sorted ascending in f).
    """
    if H_complex.shape != f_axis.shape:
        raise ValueError(
            f"H_complex shape {H_complex.shape} must match f_axis {f_axis.shape}"
        )
    if len(f_axis) < 2:
        raise ValueError("f_axis must have at least 2 samples.")

    df = float(f_axis[1] - f_axis[0])
    if df <= 0:
        raise ValueError(f"f_axis must be strictly increasing, got df={df}")

    mag = np.abs(H_complex).astype(np.float64)
    # Clamp tiny values so log doesn't blow up.
    mag_db = 20.0 * np.log10(np.maximum(mag, 10 ** (db_floor / 20.0)))

    distance_samples = max(1, int(round(min_distance_hz / df)))
    peak_indices, props = find_peaks(
        mag_db, prominence=prominence_db, distance=distance_samples
    )

    peaks: list[Peak] = []
    for idx, prom in zip(peak_indices, props["prominences"]):
        f = float(f_axis[idx])
        m_db = float(mag_db[idx])
        # -3 dB bandwidth → Q estimate. Walk left/right while above peak - 3 dB.
        target = m_db - 3.0
        left = idx
        while left > 0 and mag_db[left - 1] >= target:
            left -= 1
        right = idx
        while right < len(mag_db) - 1 and mag_db[right + 1] >= target:
            right += 1
        bw_hz = float(f_axis[right] - f_axis[left])
        # Sub-resolution peaks (BW < Δf): fall back to one bin so Q stays finite.
        # The estimate is then an upper bound on Q.
        if bw_hz <= 0:
            bw_hz = df
        q_factor = f / bw_hz if f > 0 else None
        peaks.append(
            Peak(f=f, magnitude_db=m_db, prominence_db=float(prom), q_factor=q_factor)
        )
    return peaks


def _tolerance(f_mode: float, tolerance_hz: float, tolerance_pct: float) -> float:
    return max(tolerance_hz, tolerance_pct * f_mode)


def match_peaks_to_modes(
    picked: list[Peak],
    analytical_modes: list,  # list of Mode-like objects with .f attribute
    tolerance_hz: float = 4.0,
    tolerance_pct: float = 0.02,
) -> dict:
    """Hungarian assignment between picked peaks and analytical modes.

    Each pairing must satisfy |f_peak - f_mode| <= tol(f_mode); otherwise the
    pairing's cost is +∞ and it won't be selected.

    Returns:
        {
          "matches": list[Match],
          "spurious_peak_idx": list[int],          # picks that didn't match
          "missed_mode_idx":  list[int],           # analytical modes unmatched
        }

    Notes:
        - When the picked peak set and analytical mode set have different sizes,
          the assignment is rectangular; only valid pairings within tolerance
          are accepted.
    """
    n_picks = len(picked)
    n_modes = len(analytical_modes)

    if n_picks == 0 or n_modes == 0:
        return {
            "matches": [],
            "spurious_peak_idx": list(range(n_picks)),
            "missed_mode_idx": list(range(n_modes)),
        }

    big = 1e9
    # Cost matrix: rows = analytical modes, cols = picked peaks.
    cost = np.full((n_modes, n_picks), big, dtype=np.float64)
    for i, mode in enumerate(analytical_modes):
        tol = _tolerance(mode.f, tolerance_hz, tolerance_pct)
        for j, peak in enumerate(picked):
            d = abs(peak.f - mode.f)
            if d <= tol:
                cost[i, j] = d
    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[Match] = []
    matched_peaks: set[int] = set()
    matched_modes: set[int] = set()
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] >= big:
            continue
        matches.append(
            Match(
                mode_idx=int(i),
                peak_idx=int(j),
                f_mode=float(analytical_modes[i].f),
                f_peak=float(picked[j].f),
                delta_hz=float(picked[j].f - analytical_modes[i].f),
                delta_pct=float(
                    (picked[j].f - analytical_modes[i].f) / max(analytical_modes[i].f, 1e-9)
                ),
            )
        )
        matched_peaks.add(int(j))
        matched_modes.add(int(i))

    spurious_peak_idx = [j for j in range(n_picks) if j not in matched_peaks]
    missed_mode_idx = [i for i in range(n_modes) if i not in matched_modes]
    return {
        "matches": matches,
        "spurious_peak_idx": spurious_peak_idx,
        "missed_mode_idx": missed_mode_idx,
    }


def modal_error_metrics(
    picked: list[Peak],
    analytical_modes: list,
    tolerance_hz: float = 4.0,
    tolerance_pct: float = 0.02,
) -> dict:
    """Aggregate modal error metrics, including stratified per-band breakdown.

    Returns:
        {
          "mae_hz": float | nan,
          "mae_pct": float | nan,
          "recall_at_tol": float in [0, 1],
          "n_picked": int,
          "n_analytical": int,
          "n_matched": int,
          "n_spurious": int,
          "per_mode_breakdown": {
              "low":  {"mae_hz", "recall", "n_modes", "n_matched"},   # ordinal ≤ 5
              "mid":  same                                             # 6 .. 15
              "high": same                                             # 16+
          },
        }
    """
    out = match_peaks_to_modes(picked, analytical_modes, tolerance_hz, tolerance_pct)
    matches = out["matches"]

    n_picked = len(picked)
    n_analytical = len(analytical_modes)
    n_matched = len(matches)
    n_spurious = len(out["spurious_peak_idx"])
    recall = n_matched / max(n_analytical, 1)

    if n_matched == 0:
        mae_hz = float("nan")
        mae_pct = float("nan")
    else:
        mae_hz = float(np.mean([abs(m.delta_hz) for m in matches]))
        mae_pct = float(np.mean([abs(m.delta_pct) for m in matches]))

    bands = {
        "low": (0, 5),
        "mid": (6, 15),
        "high": (16, 10**9),
    }
    per_band: dict[str, dict] = {}
    for name, (lo, hi) in bands.items():
        mode_idxs = [i for i in range(n_analytical) if lo <= i <= hi]
        band_matches = [m for m in matches if lo <= m.mode_idx <= hi]
        n_modes_band = len(mode_idxs)
        n_matched_band = len(band_matches)
        if n_matched_band == 0:
            band_mae = float("nan")
        else:
            band_mae = float(np.mean([abs(m.delta_hz) for m in band_matches]))
        per_band[name] = {
            "mae_hz": band_mae,
            "recall": (n_matched_band / max(n_modes_band, 1)),
            "n_modes": n_modes_band,
            "n_matched": n_matched_band,
        }

    return {
        "mae_hz": mae_hz,
        "mae_pct": mae_pct,
        "recall_at_tol": recall,
        "n_picked": n_picked,
        "n_analytical": n_analytical,
        "n_matched": n_matched,
        "n_spurious": n_spurious,
        "per_mode_breakdown": per_band,
    }


def plot_modal_overlay(
    H_complex: np.ndarray,
    f_axis: np.ndarray,
    analytical_modes: list,
    picked_peaks: list[Peak],
    ax,
    title: str = "",
    f_min: float = 0.0,
    f_max: float = 2000.0,
    db_floor: float = -100.0,
):
    """Overlay |H(f)| in dB with vertical lines at analytical f_m and markers
    at picked peaks. Different colours: matched (green), spurious (red),
    missed (orange — bottom of axis). For use in subplot grids.
    """
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H_complex), 10 ** (db_floor / 20.0)))
    mask = (f_axis >= f_min) & (f_axis <= f_max)

    ax.plot(f_axis[mask], mag_db[mask], color="steelblue", lw=0.8)

    out = match_peaks_to_modes(picked_peaks, analytical_modes)
    matched_peak_idx = {m.peak_idx for m in out["matches"]}
    matched_mode_idx = {m.mode_idx for m in out["matches"]}

    for m in analytical_modes:
        if f_min <= m.f <= f_max:
            color = "tab:green" if any(mt.f_mode == m.f for mt in out["matches"]) else "tab:orange"
            ax.axvline(m.f, ymin=0, ymax=0.05, color=color, lw=1.0, alpha=0.7)

    for j, p in enumerate(picked_peaks):
        if not (f_min <= p.f <= f_max):
            continue
        color = "tab:green" if j in matched_peak_idx else "tab:red"
        marker = "o" if j in matched_peak_idx else "x"
        ax.plot(p.f, p.magnitude_db, color=color, marker=marker, markersize=6, lw=0)

    ax.set_xlim(f_min, f_max)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|H| (dB)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
