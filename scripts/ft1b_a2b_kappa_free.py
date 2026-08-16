"""A2b (rebuilt): validate the FDTD boundary admittance WITHOUT the estimator's kappa.

The first attempt compared measured -3 dB widths across the two solvers and reported a 38.68%
discrepancy. That number was circular. The repo's width estimator carries
``BW = 0.302 + 1.6608*(gamma/pi)``, a calibration fitted on ISM data in the P3-2 gate, and FDTD
data does not carry it (FT-A measured ``kappa_fdtd = 1.0208``). So the comparison conflated the
question we care about -- *is the FDTD boundary admittance right* -- with a question we already
know the answer to -- *does a width convention transfer between solvers*. Measured: ISM tracked
the kappa-calibrated value (14.62 vs 14.07 predicted) while FDTD tracked the raw one (9.50 vs
8.29), and ISM/FDTD = 1.539 sat next to kappa_ism/kappa_fdtd = 1.627.

This rebuild is kappa-free BY CONSTRUCTION, not by adjustment. Three tests:

**T1 modal frequencies** -- FDTD vs ISM vs analytic f(n,m). Pure eigenvalues; no width
estimator anywhere. Necessary but NOT sufficient: it validates geometry and discretization,
and is silent on the boundary admittance.

**T2 EDC decay rate (PRIMARY)** -- isolate a mode spatially by modal projection, band-pass it,
Schroeder-integrate backwards, and fit the decay slope in dB/s. Time-domain, no peak-picking,
no -3 dB convention, no kappa. This is the direct physical observable the admittance controls.
Criterion: agreement within 10% per mode.

**T3 ratio test (robustness)** -- form (alpha=0.70)/(alpha=0.15) *within* each solver, then
compare those ratios across solvers. Any multiplicative estimator bias cancels exactly, so T3
survives even if T2 carries an absolute offset.

Reading the outcome:
  T2 and T3 both pass -> the boundary calibration is validated independently of the estimator.
  T2 fails while T3 passes -> a systematic SCALE error in the alpha->admittance conversion.
      Report it and fix the mapping; do not proceed to FT-B.

Only modes well isolated in BOTH solvers are used, and the script reports which. Per P3-2b,
~45% of modes drop out on hard splits, so a handful of clean modes is worth more than an
aggregate over everything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

import aaf.sim.fdtd_2d as F
from aaf.eval.modal_projection import enumerate_modes, mode_shape_matrix
from aaf.sim.analytical_modal_2d import damping_to_bandwidth_hz, modal_damping_2d

C = 343.0
L, W = 4.5, 4.0
DX = 0.05
FDTD_FS, FDTD_N = 12288.0, 24576
ISM_FS, ISM_N = 4096.0, 8192
MAX_ORDER = 60
F_MAX = 200.0

# Isolation is judged EMPIRICALLY, by whether the Schroeder decay is actually single-slope in
# both solvers, not by frequency separation. A frequency-separation test is the wrong
# instrument here: at alpha=0.70 the modal bandwidth is ~6-7 Hz while (1,0) and (0,1) sit
# 4.8 Hz apart, so a 3x-width rule isolates ZERO of 24 modes. What separates modes in this
# pipeline is the MODAL PROJECTION -- (n_x, n_y) shapes are spatially orthogonal even when
# they overlap spectrally -- and the honest check that it worked is a clean single-slope EDC.
EDC_R2_MIN = 0.98
EDC_DB_RANGE = (-5.0, -25.0)
TOL_T1 = 0.01               # 1% on frequency
TOL_T2 = 0.10               # 10% on decay rate
TOL_T3 = 0.10               # 10% on the within-solver ratio


def _receivers(nx: int = 8, ny: int = 8) -> np.ndarray:
    xs = np.linspace(L / (2 * nx), L - L / (2 * nx), nx)
    ys = np.linspace(W / (2 * ny), W - W / (2 * ny), ny)
    return np.array([[x, y] for x in xs for y in ys])


def _modal_ir(H: np.ndarray, phi: np.ndarray, idx: int, n_time: int) -> np.ndarray:
    """Spatially isolate one mode by projection, then return its impulse response."""
    spec = np.linalg.pinv(phi) @ H
    return np.fft.irfft(spec[idx], n=n_time)


def _edc_slope_db_per_s(h: np.ndarray, fs: float, f0: float, bw_hint: float) -> Optional[dict]:
    """Schroeder backward integration -> decay slope in dB/s.

    Band-passed around the mode first so the fit sees one decay, not a sum of them. The
    Schroeder integral is monotone by construction, which is what makes the slope fit stable
    without any peak-picking.
    """
    lo = max(1.0, f0 - max(3.0 * bw_hint, 4.0))
    hi = min(fs / 2.0 - 1.0, f0 + max(3.0 * bw_hint, 4.0))
    if hi <= lo:
        return None
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, h)
    e = y ** 2
    edc = np.cumsum(e[::-1])[::-1]
    if edc[0] <= 0:
        return None
    edc_db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-300))
    t = np.arange(edc_db.size) / fs
    i0 = int(np.argmax(edc_db <= EDC_DB_RANGE[0]))
    i1 = int(np.argmax(edc_db <= EDC_DB_RANGE[1]))
    if i1 <= i0 + 8:
        return None
    A = np.polyfit(t[i0:i1], edc_db[i0:i1], 1)
    pred = np.polyval(A, t[i0:i1])
    ss_res = float(np.sum((edc_db[i0:i1] - pred) ** 2))
    ss_tot = float(np.sum((edc_db[i0:i1] - np.mean(edc_db[i0:i1])) ** 2))
    return {"slope_db_per_s": float(A[0]),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n_samples": int(i1 - i0),
            "band_hz": [float(lo), float(hi)]}


def _peak_freq(H: np.ndarray, phi: np.ndarray, idx: int, freqs: np.ndarray,
               f0: float, search: float = 3.0) -> float:
    spec = np.abs(np.linalg.pinv(phi) @ H)[idx]
    lo = int(np.searchsorted(freqs, f0 - search))
    hi = int(np.searchsorted(freqs, f0 + search))
    if hi <= lo:
        return float("nan")
    k = lo + int(np.argmax(spec[lo:hi]))
    if 0 < k < spec.size - 1:                       # parabolic sub-bin refinement
        y0, y1, y2 = spec[k - 1], spec[k], spec[k + 1]
        den = y0 - 2 * y1 + y2
        d = 0.0 if den == 0 else 0.5 * (y0 - y2) / den
    else:
        d = 0.0
    return float(freqs[k] + d * (freqs[1] - freqs[0]))


def run_case(alphas, rx, modes, phi) -> dict:
    from aaf.sim.ism_2d import simulate_room_2d
    src = (0.37 * L, 0.29 * W)
    fd = F.simulate(L, W, alphas, src=src, rx=rx, dx=DX, fs=FDTD_FS, n=FDTD_N, c=C)
    rx_snap = np.asarray(fd["meta"]["rx_pos_snapped"], dtype=float)
    phi_s = mode_shape_matrix(modes, rx_snap, L, W)
    ism = simulate_room_2d(dict(L=L, W=W, source_pos=np.asarray(src, float),
                                receiver_pos=rx_snap, alphas=tuple(float(x) for x in alphas),
                                fs=ISM_FS, n_time_samples=ISM_N, max_order=MAX_ORDER))
    return {"fdtd": fd, "ism": ism, "phi": phi_s, "rx_snap": rx_snap}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ft1b/a2b_kappa_free.json")
    a = ap.parse_args()

    rx = _receivers()
    modes = [m for m in enumerate_modes(L, W, f_max=F_MAX)
             if not (m.n_x == 0 and m.n_y == 0)]
    phi0 = mode_shape_matrix(modes, rx, L, W)

    cases = {"a015": (0.15,) * 4, "a070": (0.70, 0.15, 0.15, 0.15)}
    sims = {k: run_case(v, rx, modes, phi0) for k, v in cases.items()}

    iso: List[int] = list(range(len(modes)))
    iso_detail = []

    rows = []
    for i in iso:
        m = modes[i]
        rec = {"mode": [m.n_x, m.n_y], "f_analytic_hz": float(m.f),
               "family": ("x_axial" if m.n_y == 0 else
                          "y_axial" if m.n_x == 0 else "tangential")}
        for ck, s in sims.items():
            bw_hint = damping_to_bandwidth_hz(modal_damping_2d(
                L, W, list(cases[ck]), m.n_x, m.n_y, model="kuttruff"))
            fdec = _edc_slope_db_per_s(
                _modal_ir(s["fdtd"]["H_complex"], s["phi"], i, FDTD_N), FDTD_FS, m.f, bw_hint)
            idec = _edc_slope_db_per_s(
                _modal_ir(np.asarray(s["ism"]["H_complex"]), s["phi"], i, ISM_N),
                ISM_FS, m.f, bw_hint)
            rec[ck] = {
                "fdtd_f_hz": _peak_freq(s["fdtd"]["H_complex"], s["phi"], i,
                                        np.asarray(s["fdtd"]["freqs"], float), m.f),
                "ism_f_hz": _peak_freq(np.asarray(s["ism"]["H_complex"]), s["phi"], i,
                                       np.arange(ISM_N // 2 + 1) * (ISM_FS / ISM_N), m.f),
                "fdtd_edc": fdec, "ism_edc": idec,
            }
        r2s = [rec[ck][k]["r2"] for ck in cases for k in ("fdtd_edc", "ism_edc")
               if rec[ck][k]]
        n_fits = sum(1 for ck in cases for k in ("fdtd_edc", "ism_edc") if rec[ck][k])
        rec["clean"] = bool(n_fits == 4 and r2s and min(r2s) >= EDC_R2_MIN)
        rec["min_edc_r2"] = float(min(r2s)) if r2s else float("nan")
        iso_detail.append({"mode": rec["mode"], "f_hz": rec["f_analytic_hz"],
                           "min_edc_r2": rec["min_edc_r2"], "clean": rec["clean"]})
        rows.append(rec)

    rows_all = rows
    rows = [r for r in rows if r["clean"]]

    # ---- T1 frequencies
    t1 = []
    for r in rows:
        for ck in cases:
            for solver in ("fdtd", "ism"):
                f = r[ck]["{}_f_hz".format(solver)]
                if np.isfinite(f):
                    t1.append(abs(f - r["f_analytic_hz"]) / r["f_analytic_hz"])
    t1_worst = float(max(t1)) if t1 else float("nan")

    # ---- T2 decay rate (primary)
    t2 = []
    for r in rows:
        for ck in cases:
            fd, im = r[ck]["fdtd_edc"], r[ck]["ism_edc"]
            if fd and im and im["slope_db_per_s"] != 0:
                rel = abs(fd["slope_db_per_s"] - im["slope_db_per_s"]) / abs(im["slope_db_per_s"])
                t2.append({"mode": r["mode"], "case": ck, "rel": float(rel),
                           "fdtd_db_s": fd["slope_db_per_s"], "ism_db_s": im["slope_db_per_s"],
                           "fdtd_r2": fd["r2"], "ism_r2": im["r2"]})
    t2_worst = float(max((x["rel"] for x in t2), default=float("nan")))

    # ---- T3 within-solver ratio (robustness)
    t3 = []
    for r in rows:
        f0, f7 = r["a015"]["fdtd_edc"], r["a070"]["fdtd_edc"]
        i0, i7 = r["a015"]["ism_edc"], r["a070"]["ism_edc"]
        if not (f0 and f7 and i0 and i7):
            continue
        if f0["slope_db_per_s"] == 0 or i0["slope_db_per_s"] == 0:
            continue
        rf = f7["slope_db_per_s"] / f0["slope_db_per_s"]
        ri = i7["slope_db_per_s"] / i0["slope_db_per_s"]
        t3.append({"mode": r["mode"], "fdtd_ratio": float(rf), "ism_ratio": float(ri),
                   "rel": float(abs(rf - ri) / abs(ri))})
    t3_worst = float(max((x["rel"] for x in t3), default=float("nan")))

    p1 = np.isfinite(t1_worst) and t1_worst <= TOL_T1
    p2 = bool(t2) and t2_worst <= TOL_T2
    p3 = bool(t3) and t3_worst <= TOL_T3
    if p2 and p3:
        verdict = "PASS -- boundary admittance validated independently of the estimator"
    elif p3 and not p2:
        verdict = ("SCALE ERROR -- T3 passes but T2 fails: the alpha->admittance mapping has a "
                   "systematic scale error. Fix the conversion; do NOT proceed to FT-B.")
    else:
        verdict = "FAIL -- escalate"

    out = {
        "gate": "A2b (kappa-free rebuild)",
        "supersedes": "outputs/ft1b/a2b_cross_solver.json (INCONCLUSIVE: circular via kappa)",
        "why_kappa_free": ("kappa was fitted on ISM data in the P3-2 gate, so scoring "
                           "FDTD-vs-ISM through a kappa-scaled width estimator conflates the "
                           "boundary admittance with the width convention. T2 and T3 use no "
                           "width estimator at all."),
        "isolation": {"criterion": "single-slope Schroeder EDC, r^2 >= {} in BOTH solvers "
                                   "and BOTH alpha cases".format(EDC_R2_MIN),
                      "why_not_frequency_separation": (
                          "at alpha=0.70 the modal bandwidth is ~6-7 Hz while (1,0) and (0,1) "
                          "are 4.8 Hz apart, so a 3x-width separation rule isolates 0 of 24 "
                          "modes; the modal projection is what separates them here"),
                      "n_clean": len(rows), "n_modes": len(modes),
                      "modes": iso_detail},
        "T1_modal_frequency": {"pass": bool(p1), "worst_rel": t1_worst, "tol": TOL_T1,
                               "note": "necessary, not sufficient: silent on the admittance"},
        "T2_edc_decay_rate": {"pass": bool(p2), "worst_rel": t2_worst, "tol": TOL_T2,
                              "primary": True, "per_mode": t2},
        "T3_within_solver_ratio": {"pass": bool(p3), "worst_rel": t3_worst, "tol": TOL_T3,
                                   "note": ("alpha=0.70 / alpha=0.15 inside each solver, then "
                                            "compared -- any multiplicative bias cancels"),
                                   "per_mode": t3},
        "verdict": verdict,
        "rows": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)

    print("clean (single-slope EDC in both solvers) modes: {} of {}".format(len(rows), len(modes)))
    print("  " + ", ".join("{}".format(tuple(r["mode"])) for r in rows))
    print("\nT2 (primary) -- EDC decay rate, dB/s:")
    print("  {:>8s} {:>6s} {:>11s} {:>11s} {:>9s}".format("mode", "case", "fdtd", "ism", "rel"))
    for x in t2:
        print("  {:>8s} {:>6s} {:11.3f} {:11.3f} {:+8.2%}".format(
            str(tuple(x["mode"])), x["case"], x["fdtd_db_s"], x["ism_db_s"], x["rel"]))
    print("\nT3 -- within-solver ratio a070/a015:")
    for x in t3:
        print("  {:>8s} fdtd {:7.4f}  ism {:7.4f}  rel {:+8.2%}".format(
            str(tuple(x["mode"])), x["fdtd_ratio"], x["ism_ratio"], x["rel"]))
    print("\nT1 worst {:.3%} (tol {:.0%})   T2 worst {:.2%} (tol {:.0%})   "
          "T3 worst {:.2%} (tol {:.0%})".format(t1_worst, TOL_T1, t2_worst, TOL_T2,
                                                t3_worst, TOL_T3))
    print("VERDICT: {}".format(verdict))
    print("-> {}".format(a.out))
    return 0 if p2 and p3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
