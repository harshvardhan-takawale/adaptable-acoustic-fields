"""P3-2b headline estimator: the predicted d(bandwidth)/d(m) slope against ISM-ray theory.

``edit_bw_slope`` (predicted delta regressed on ground-truth delta) answers "does the model
move in the right direction, on average, over a pool of modes". It cannot answer "does the
model implement the LAW", because a model that memorised two alpha values and interpolated
badly between them can still post a decent pooled slope. The law is a statement about a
derivative: under the ISM-ray damping the -3 dB width of a mode is EXACTLY affine in the
material coordinate ``m = -ln(1 - alpha)`` of the walls its family bounces between. So fit
that derivative directly, per (geometry, wall, family), and compare it with a number that
was computed before the model existed.

*** THE kappa CORRECTION -- the single highest-risk step in this file ***

The measurement chain does not return the Lorentzian width. ``measure_modes`` returns a
-3 dB width read off a projected, windowed, finite-length spectrum, and the P3-2 physics
gate (T5) calibrated it against theory as::

    BW_measured = 0.30238 + 1.66076 * (gamma / pi)

The intercept is an estimator offset shared by the edited and baseline measurement, so it
cancels in the paired delta. **The slope does not cancel.** A measured delta-bandwidth is
therefore ``kappa`` times the raw Lorentzian prediction::

    a_theory = kappa * c / (4 * pi * D)          kappa = 1.6607564051417665 (FROZEN)
    D = L for a west/east edit measured on the x-axial family
    D = W for a south/north edit measured on the y-axial family
    a_theory = 0 for the orthogonal axial family (ISM has no grazing-incidence absorption)

At L = 4.5 that is 1.66076 * 6.06557 = 10.073 Hz per unit m, not 6.066. Verified against
ground truth on the P3-2 data: the west->M2 GT slope measures 5.355 Hz where the kappa-scaled
theory predicts 5.345 (0.2% error) and the raw Lorentzian predicts 3.219 (-40%). Scoring a
model against the RAW value inflates rho by 1/kappa: a model reproducing the physics
perfectly would score rho = 0.602 and be recorded as a failure. ``rho_vs_raw_theory`` is
reported alongside purely so a reader can see that the correction was applied and how large
it is -- it is never the acceptance number.

Derivation of the raw slope, for the record. For an x-axial mode, ``modal_damping_2d``
(model="ism_ray") gives gamma = c * (m_west + m_east) / (4L) since the pressure reflection
coefficient is R = sqrt(1-alpha) and -ln R = m/2, and the -3 dB width is gamma/pi. Hence
d(BW_raw)/d(m_west) = c/(4 pi L), independent of mode index -- which is also why the
within-family mode-to-mode spread (control C2) is pure estimator noise.

Fit protocol (all four choices exist to stop a broken model from scoring well):

* **One point per (config, family), not per mode.** Modes within an axial family share a
  damping rate exactly, so treating them as independent observations would shrink the CI by
  sqrt(n) on nothing. Each point is the mean over that family's paired-valid modes and is
  weighted by how many contributed.
* **The baseline is a point** at ``d_m = 0``. It is the anchor every delta is taken against
  and it is what makes 5 alpha points available on a test geometry (baseline + the four
  presets 0.05 / 0.30 / 0.50 / 0.70). It is (0, 0) by construction, which pins the intercept.
* **Free intercept.** A model with a constant bandwidth bias in the paired delta should lose
  points on E_BW, not on the slope; forcing b = 0 would mix the two failure modes.
* **frac_modes_dropped is reported and gates the verdict.** A model whose predicted peaks are
  so mangled that ``measure_modes`` rejects them produces few, cherry-picked survivors -- and
  those survivors can fit a beautiful line. An arm must not pass by being unmeasurable
  exactly where it is being tested.

Slab vs non-slab grouping: cells are grouped by whether the edited WALL carries a held-out
m-slab (west, north) or not (east, south). Both groups fit the same 5-point alpha ladder;
they differ only in whether the model ever saw material values in that neighbourhood on
that wall during training. ``slab_local`` additionally reports the local ratio at the slab
point itself (predicted delta / theory delta at alpha = 0.50 on west, 0.70 on north).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aaf.data.mat_configs_cont import HOLDOUT_SLABS
from aaf.eval.modal_projection import TANGENTIAL, X_AXIAL, Y_AXIAL
from aaf.walls import WALL_AXIS

# Frozen from outputs/p3_2/gate/gate.json -> T5_calibration.ism_ray.slope. Hard-coded rather
# than re-read so an edited gate file cannot silently move every published rho.
KAPPA = 1.6607564051417665
C_SOUND = 343.0

MIN_ALPHA_POINTS = 5
MIN_D_M_SPAN = 0.5
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260813


def a_theory_hz_per_m(L: float, W: float, wall: str, family: str,
                      kappa: float = KAPPA, c: float = C_SOUND) -> float:
    """d(measured -3 dB bandwidth) / d(m_wall) in Hz, for modes of ``family``.

    kappa-scaled: this is the slope of a MEASURED delta, which is what the estimator
    returns. Divide by ``kappa`` for the raw Lorentzian value.
    """
    own = X_AXIAL if WALL_AXIS[wall] == "x" else Y_AXIAL
    if family == own:
        d = float(L) if own == X_AXIAL else float(W)
        return float(kappa) * float(c) / (4.0 * math.pi * d)
    if family in (X_AXIAL, Y_AXIAL):
        return 0.0
    # Tangential modes damp at c*(cos_x*kappa_x + cos_y*kappa_y), so the slope depends on
    # (n_x, n_y) and no single number describes the family. Excluded from rho by design.
    return float("nan")


def own_family(wall: str) -> str:
    return X_AXIAL if WALL_AXIS[wall] == "x" else Y_AXIAL


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Weighted least squares for ``y = a x + b``. Returns (a, b)."""
    sw = np.sqrt(np.asarray(w, dtype=float))
    design = np.stack([np.asarray(x, dtype=float), np.ones_like(x, dtype=float)], axis=1)
    sol, *_ = np.linalg.lstsq(design * sw[:, None], np.asarray(y, dtype=float) * sw,
                              rcond=None)
    return float(sol[0]), float(sol[1])


def fit_cell(points: Sequence[dict], L: float, W: float, wall: str, family: str,
             kappa: float = KAPPA) -> dict:
    """Fit one (geometry, wall, family) cell.

    ``points``: one dict per alpha value, with keys ``d_m``, ``d_bw_pred``, ``d_bw_gt``,
    ``n_modes`` (paired-valid), ``n_modes_candidate``, ``alpha``, ``in_slab``.
    """
    a_th = a_theory_hz_per_m(L, W, wall, family, kappa=kappa)
    a_th_raw = a_theory_hz_per_m(L, W, wall, family, kappa=1.0)
    cand = int(sum(int(p["n_modes_candidate"]) for p in points))
    valid_pts = [p for p in points if int(p["n_modes"]) > 0]
    n_valid = int(sum(int(p["n_modes"]) for p in valid_pts))
    frac_dropped = float(1.0 - n_valid / cand) if cand else float("nan")

    out = {
        "L": float(L), "W": float(W), "wall": wall, "family": family,
        "own_family": bool(family == own_family(wall)),
        "wall_has_slab": bool(wall in HOLDOUT_SLABS),
        "n_alpha_points": len(valid_pts),
        "n_modes_valid": n_valid,
        "n_modes_candidate": cand,
        "frac_modes_dropped": frac_dropped,
        "a_theory": a_th, "a_theory_raw": a_th_raw,
        "a_fit": float("nan"), "b_fit": float("nan"), "a_fit_gt": float("nan"),
        "rho": float("nan"), "rho_gt": float("nan"), "rho_vs_raw_theory": float("nan"),
        "d_m_span": float("nan"), "fitted": False,
        "reject_reason": None,
        "slab_point": None,
    }
    if not valid_pts:
        out["reject_reason"] = "no paired-valid modes"
        return out

    x = np.array([p["d_m"] for p in valid_pts], dtype=float)
    span = float(x.max() - x.min())
    out["d_m_span"] = span
    if len(valid_pts) < MIN_ALPHA_POINTS:
        out["reject_reason"] = "only {} alpha points (< {})".format(
            len(valid_pts), MIN_ALPHA_POINTS)
        return out
    if span < MIN_D_M_SPAN:
        out["reject_reason"] = "d_m span {:.3f} < {}".format(span, MIN_D_M_SPAN)
        return out

    w = np.array([p["n_modes"] for p in valid_pts], dtype=float)
    yp = np.array([p["d_bw_pred"] for p in valid_pts], dtype=float)
    yg = np.array([p["d_bw_gt"] for p in valid_pts], dtype=float)
    a_p, b_p = _wls(x, yp, w)
    a_g, _ = _wls(x, yg, w)
    out.update({"a_fit": a_p, "b_fit": b_p, "a_fit_gt": a_g, "fitted": True})
    if np.isfinite(a_th) and abs(a_th) > 1e-12:
        out["rho"] = float(a_p / a_th)
        out["rho_gt"] = float(a_g / a_th)
    if np.isfinite(a_th_raw) and abs(a_th_raw) > 1e-12:
        out["rho_vs_raw_theory"] = float(a_p / a_th_raw)

    # Local behaviour AT the held-out value, which the 5-point fit only partly reflects.
    slab_pts = [p for p in valid_pts if p.get("in_slab")]
    if slab_pts and np.isfinite(a_th) and abs(a_th) > 1e-12:
        p0 = slab_pts[0]
        denom = a_th * float(p0["d_m"])
        out["slab_point"] = {
            "alpha": float(p0["alpha"]), "d_m": float(p0["d_m"]),
            "d_bw_pred": float(p0["d_bw_pred"]), "d_bw_gt": float(p0["d_bw_gt"]),
            "theory_d_bw": float(denom),
            "ratio_pred_over_theory": float(p0["d_bw_pred"] / denom)
            if abs(denom) > 1e-12 else float("nan"),
            "ratio_gt_over_theory": float(p0["d_bw_gt"] / denom)
            if abs(denom) > 1e-12 else float("nan"),
            "n_modes": int(p0["n_modes"]),
        }
    return out


def _bootstrap_median_ci(values_by_geom: Dict[Tuple[float, float], List[float]],
                         n_boot: int = N_BOOTSTRAP,
                         seed: int = BOOTSTRAP_SEED) -> Tuple[float, List[float]]:
    """Median and a 95% percentile CI, resampling GEOMETRIES (not cells).

    Cells from one geometry share a room, a receiver grid and a mode set, so they are not
    independent; resampling cells would report a CI several times too narrow.
    """
    keys = [k for k, v in values_by_geom.items() if v]
    pooled = [x for k in keys for x in values_by_geom[k]]
    if not pooled:
        return float("nan"), [float("nan"), float("nan")]
    med = float(np.median(pooled))
    if len(keys) < 2:
        return med, [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = []
    idx = np.arange(len(keys))
    for _ in range(int(n_boot)):
        pick = rng.choice(idx, size=len(keys), replace=True)
        vals = [x for i in pick for x in values_by_geom[keys[i]]]
        if vals:
            draws.append(float(np.median(vals)))
    if not draws:
        return med, [float("nan"), float("nan")]
    return med, [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _aggregate(cells: Sequence[dict], n_boot: int = N_BOOTSTRAP) -> dict:
    fitted = [c for c in cells if c["fitted"] and np.isfinite(c["rho"])]
    by_geom: Dict[Tuple[float, float], List[float]] = {}
    for c in fitted:
        by_geom.setdefault((c["L"], c["W"]), []).append(float(c["rho"]))
    med, ci = _bootstrap_median_ci(by_geom, n_boot=n_boot)
    cand = sum(int(c["n_modes_candidate"]) for c in cells)
    valid = sum(int(c["n_modes_valid"]) for c in cells)
    ratios = [c["slab_point"]["ratio_pred_over_theory"] for c in cells
              if c.get("slab_point") and np.isfinite(
                  c["slab_point"]["ratio_pred_over_theory"])]
    out = {
        "rho_median": med,
        "rho_ci95": ci,
        "a_fit_median": float(np.median([c["a_fit"] for c in fitted])) if fitted
        else float("nan"),
        "a_theory_median": float(np.median([c["a_theory"] for c in fitted])) if fitted
        else float("nan"),
        "a_fit_gt_median": float(np.median([c["a_fit_gt"] for c in fitted])) if fitted
        else float("nan"),
        "rho_gt_median": float(np.median([c["rho_gt"] for c in fitted
                                          if np.isfinite(c["rho_gt"])])) if fitted
        else float("nan"),
        "n_cells": len(fitted),
        "n_cells_attempted": len(cells),
        "n_geometries": len(by_geom),
        "frac_modes_dropped": float(1.0 - valid / cand) if cand else float("nan"),
    }
    if ratios:
        out["slab_point_ratio_median"] = float(np.median(ratios))
        out["slab_point_n"] = len(ratios)
    return out


def slope_fit(cells: Sequence[dict], n_boot: int = N_BOOTSTRAP) -> dict:
    """Aggregate per-cell fits into the ``slope_fit`` block of summary.json."""
    own = [c for c in cells if c["own_family"]]
    orth = [c for c in cells if (not c["own_family"]) and c["family"] != TANGENTIAL]
    agg = {
        "own_family": {
            "all": _aggregate(own, n_boot),
            "non_slab": _aggregate([c for c in own if not c["wall_has_slab"]], n_boot),
            "slab_local": _aggregate([c for c in own if c["wall_has_slab"]], n_boot),
        },
        "orthogonal_family": {
            # a_theory is exactly 0 here, so rho is undefined; the fitted slope IS the
            # leakage in Hz per unit m -- how much of a west edit the model puts on the
            # y-axial modes. GT is ~0 for the ISM simulator.
            "a_fit_median": float(np.median([c["a_fit"] for c in orth if c["fitted"]]))
            if any(c["fitted"] for c in orth) else float("nan"),
            "a_fit_gt_median": float(np.median([c["a_fit_gt"] for c in orth if c["fitted"]]))
            if any(c["fitted"] for c in orth) else float("nan"),
            "n_cells": int(sum(1 for c in orth if c["fitted"])),
            "frac_modes_dropped": _aggregate(orth, n_boot=1)["frac_modes_dropped"],
        },
    }
    raw = [c["rho_vs_raw_theory"] for c in own
           if c["fitted"] and np.isfinite(c["rho_vs_raw_theory"])]
    return {
        "aggregate": agg,
        "per_cell": list(cells),
        "kappa": KAPPA,
        "rho_vs_raw_theory_median": float(np.median(raw)) if raw else float("nan"),
        "protocol": {
            "min_alpha_points": MIN_ALPHA_POINTS,
            "min_d_m_span": MIN_D_M_SPAN,
            "weighting": "n paired-valid modes per alpha point",
            "intercept": "free",
            "baseline_included_as_anchor": True,
            "bootstrap": "percentile CI over geometries, n={}".format(n_boot),
            "a_theory": "kappa * c / (4 pi D); D = L for west/east on x_axial, "
                        "W for south/north on y_axial; 0 for the orthogonal axial family",
            "groups": {
                "all": "every own-family cell",
                "non_slab": "cells on walls with no held-out slab (east, south)",
                "slab_local": "cells on walls carrying a held-out slab (west, north); "
                              "slab_point_ratio_median is the local pred/theory ratio at "
                              "the held-out alpha itself",
            },
        },
    }
