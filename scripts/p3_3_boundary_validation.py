"""P3-3 Part 0: validate the FDTD boundary against ANALYTIC theory, not against ISM.

The earlier A2b compared FDTD to ISM and reported a 38.68% "failure". That framing was wrong
twice over: it was circular through the estimator's ISM-fitted kappa, and its physical
prediction (FDTD damps LESS at grazing, from alpha(theta) -> 0) does not apply to modal
damping at all.

The correction. alpha(theta) -> 0 is the reflection coefficient of a TRAVELLING PLANE WAVE at
grazing. A room mode has a pressure ANTINODE at every wall, and a locally-reacting boundary
dissipates in proportion to |p|^2 Re(beta), so a mode uniform along x still loses energy at the
x-normal walls. In 2D the west wall contributes ~ alpha_w/L to a (n,0) mode and ~ alpha_w/(2L)
to a (0,m) mode -- a factor of 2, not zero. ISM's ray picture gives (0,m) modes essentially
zero west-wall damping. So FDTD damping MORE than ISM at grazing is the expected result, and it
is the origin of the 2:1-vs-29:1 selectivity gap recorded in D48.

  0a  GATE: measured FDTD modal damping vs the analytic locally-reacting formula, within 15%
      for both (n,0) and (0,m) at alpha = 0.15 and 0.70. ISM is NOT a reference here.
  0b  MEASUREMENT (no pass/fail): the west-edit selectivity ratio in each solver. Theory says
      ~2 for a wave solver and >>2 for a ray solver; confirming that turns D48's scoping
      caveat into an independently measured two-solver result.
  0c  Fix the clean-mode filter, which admitted no normal-incidence mode at all, and
      cross-check on a high-aspect room where the modes are well separated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import math

import numpy as np
from scipy.signal import butter, sosfiltfilt

import aaf.sim.fdtd_2d as F
from aaf.eval.modal_projection import enumerate_modes, mode_shape_matrix
from aaf.sim.analytical_modal_2d import damping_to_bandwidth_hz, modal_damping_2d

C = 343.0
DX = 0.05
FS, N = 12288.0, 24576
ISM_FS, ISM_N, MAX_ORDER = 4096.0, 8192, 60
F_MAX = 200.0
TOL_0A = 0.15
EDC_DB = (-5.0, -25.0)


# ----------------------------------------------------------------- derivation 1 (independent)
def _xi(a):
    r = math.sqrt(1.0 - float(a))
    return (1.0 - r) / (1.0 + r)


def gamma_exact(L, W, alphas, n_x, n_y, c=C):
    """EXACT locally-reacting modal damping. Derivation 1 of 2 (D54).

    Per round trip along x the pressure amplitude is multiplied by R_w R_e, so the decay rate
    is -(c/2L) ln(R_w R_e) = (c/L)[artanh(xi_w) + artanh(xi_e)] using -ln R = 2 artanh(xi) for
    R = (1-xi)/(1+xi). A mode uniform along x sees each x-wall at half weight (its |p|^2
    average over the transverse coordinate is 1 rather than 1/2 relative to the stored energy),
    which is the eps(k) = 1 vs 2 split:

        gamma = (c/L)[artanh xi_w + artanh xi_e] * eps(n_x)/2
              + (c/W)[artanh xi_s + artanh xi_n] * eps(n_y)/2

    WHY NOT the first-order form (D54 requires naming the limit and its error). The repo's
    modal_damping_2d(kuttruff) is gamma = (c/8)[(a_w+a_e) eps/L + (a_s+a_n) eps/W], which is
    this expression with artanh(xi) -> xi AND xi -> alpha/4. Both hold only as alpha -> 0.
    At alpha = 0.70 the true xi is 0.2922 while alpha/4 = 0.1750 -- a 67% underestimate of the
    quantity that sets the damping -- and artanh adds a further +3.0%. Measured consequence on
    6.0 x 3.0, mode (n,0), alpha_w = 0.70:
        first-order target 16.435 | exact 24.175 | FDTD measured 23.945
        gap +45.7% (first-order)  ->  -1.0% (exact); the correction closes 97% of it.
    At alpha = 0.15 the first-order error is only +0.06%, which is why the uniform case looked
    fine (+7%) while the absorbing case did not (+46%). An admittance SCALE error would have
    corrupted both equally; it did not.
    """
    eps = lambda k: 1.0 if k == 0 else 2.0
    a_w, a_e, a_s, a_n = [float(x) for x in alphas]
    return ((c / L) * (math.atanh(_xi(a_w)) + math.atanh(_xi(a_e))) * eps(n_x) / 2.0
            + (c / W) * (math.atanh(_xi(a_s)) + math.atanh(_xi(a_n))) * eps(n_y) / 2.0)


def gamma_first_order(L, W, alphas, n_x, n_y, c=C):
    """The small-alpha limit, kept ONLY to report its error at the operating point."""
    eps = lambda k: 1.0 if k == 0 else 2.0
    a_w, a_e, a_s, a_n = [float(x) for x in alphas]
    return (c / 8.0) * ((a_w + a_e) * eps(n_x) / L + (a_s + a_n) * eps(n_y) / W)


def _receivers(L, W, nx=8, ny=8):
    return np.array([[x, y]
                     for x in np.linspace(L / (2 * nx), L - L / (2 * nx), nx)
                     for y in np.linspace(W / (2 * ny), W - W / (2 * ny), ny)])


def _edc_slope(h, fs, f0, bw_hint):
    lo = max(1.0, f0 - max(3.0 * bw_hint, 4.0))
    hi = min(fs / 2 - 1.0, f0 + max(3.0 * bw_hint, 4.0))
    if hi <= lo:
        return None
    y = sosfiltfilt(butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos"), h)
    edc = np.cumsum((y ** 2)[::-1])[::-1]
    if edc[0] <= 0:
        return None
    db = 10 * np.log10(np.maximum(edc / edc[0], 1e-300))
    t = np.arange(db.size) / fs
    i0, i1 = int(np.argmax(db <= EDC_DB[0])), int(np.argmax(db <= EDC_DB[1]))
    if i1 <= i0 + 8:
        return None
    A = np.polyfit(t[i0:i1], db[i0:i1], 1)
    pr = np.polyval(A, t[i0:i1])
    ss = float(np.sum((db[i0:i1] - pr) ** 2)); tt = float(np.sum((db[i0:i1] - db[i0:i1].mean()) ** 2))
    return {"slope_db_per_s": float(A[0]), "gamma": float(-A[0] * np.log(10) / 20.0),
            "r2": float(1 - ss / tt) if tt > 0 else float("nan")}


def measure_room(L, W, alphas, label):
    from aaf.sim.ism_2d import simulate_room_2d
    src = (0.37 * L, 0.29 * W)
    rx = _receivers(L, W)
    fd = F.simulate(L, W, alphas, src=src, rx=rx, dx=DX, fs=FS, n=N, c=C)
    rxs = np.asarray(fd["meta"]["rx_pos_snapped"], float)
    modes = [m for m in enumerate_modes(L, W, f_max=F_MAX) if not (m.n_x == 0 and m.n_y == 0)]
    phi = mode_shape_matrix(modes, rxs, L, W)
    ism = simulate_room_2d(dict(L=L, W=W, source_pos=np.asarray(src, float),
                                receiver_pos=rxs, alphas=tuple(float(x) for x in alphas),
                                fs=ISM_FS, n_time_samples=ISM_N, max_order=MAX_ORDER))
    pf = np.linalg.pinv(phi) @ fd["H_complex"]
    pi = np.linalg.pinv(phi) @ np.asarray(ism["H_complex"])
    rows = []
    for k, m in enumerate(modes):
        g1 = gamma_exact(L, W, alphas, m.n_x, m.n_y)
        g2 = gamma_first_order(L, W, alphas, m.n_x, m.n_y)
        g_repo = modal_damping_2d(L, W, list(alphas), m.n_x, m.n_y, model="kuttruff")
        bw = damping_to_bandwidth_hz(g1)
        fe = _edc_slope(np.fft.irfft(pf[k], n=N), FS, m.f, bw)
        ie = _edc_slope(np.fft.irfft(pi[k], n=ISM_N), ISM_FS, m.f, bw)
        rows.append({
            "mode": [m.n_x, m.n_y], "f_hz": float(m.f),
            "family": ("x_axial" if m.n_y == 0 else "y_axial" if m.n_x == 0 else "tangential"),
            "gamma_theory_exact": float(g1),
            "gamma_theory_first_order": float(g2),
            "gamma_repo_kuttruff": float(g_repo),
            "first_order_rel_error": float((g2 - g1) / g1),
            "derivations_agree": bool(abs(g2 - g_repo) <= 1e-9 * max(1.0, abs(g2))),
            "fdtd": fe, "ism": ie,
            "fdtd_rel_vs_theory": (None if not fe else float((fe["gamma"] - g1) / g1)),
            "ism_rel_vs_theory": (None if not ie else float((ie["gamma"] - g1) / g1)),
        })
    return {"case": label, "L": L, "W": W, "alphas": list(alphas), "modes": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ft1b/a2b_grazing_diagnostic.json")
    a = ap.parse_args()

    cases = []
    for (L, W) in ((4.5, 4.0), (6.0, 3.0)):
        for tag, al in (("a015", (0.15,) * 4), ("a070_west", (0.70, 0.15, 0.15, 0.15))):
            cases.append(measure_room(L, W, al, "{:.1f}x{:.1f}_{}".format(L, W, tag)))

    # derivation cross-check (D54)
    disagree = [r for c in cases for r in c["modes"] if not r["derivations_agree"]]

    # 0a gate: FDTD vs analytic, both families, both alphas
    judged = [(c, r) for c in cases for r in c["modes"]
              if r["fdtd"] and r["family"] in ("x_axial", "y_axial")
              and r["fdtd"]["r2"] >= 0.95]
    worst = max((abs(r["fdtd_rel_vs_theory"]) for _, r in judged), default=float("nan"))
    fam = {}
    for _, r in judged:
        fam.setdefault(r["family"], []).append(abs(r["fdtd_rel_vs_theory"]))
    p0a = bool(judged) and worst <= TOL_0A and len(fam) == 2

    # 0b measurement: west-edit selectivity delta(n,0)/delta(0,m) per solver
    sel = {}
    for c in cases:
        if "a070_west" not in c["case"]:
            continue
        base = next(x for x in cases if x["L"] == c["L"] and "a015" in x["case"])
        for solver in ("fdtd", "ism"):
            dx_ax, dy_ax = [], []
            for r, rb in zip(c["modes"], base["modes"]):
                if not (r[solver] and rb[solver]):
                    continue
                d = r[solver]["gamma"] - rb[solver]["gamma"]
                if r["family"] == "x_axial":
                    dx_ax.append(d)
                elif r["family"] == "y_axial":
                    dy_ax.append(d)
            if dx_ax and dy_ax and np.median(dy_ax) != 0:
                sel.setdefault(c["case"], {})[solver] = float(
                    np.median(dx_ax) / np.median(dy_ax))

    out = {
        "part": "P3-3 Part 0",
        "0a_gate": {
            "pass": p0a, "tolerance": TOL_0A, "worst_rel": worst,
            "n_modes_judged": len(judged),
            "families_present": sorted(fam),
            "per_family_worst": {k: max(v) for k, v in fam.items()},
            "reference": "analytic locally-reacting modal damping (NOT ISM)",
        },
        "derivation_cross_check_D54": {
            "method_1": "exact -ln(R) = 2 artanh(xi) round-trip decay, derived in this file",
            "method_2": "the first-order limit reproduces modal_damping_2d(kuttruff) exactly",
            "limit_stated_D54": "first-order uses artanh(xi)->xi and xi->alpha/4; error +0.06% at alpha=0.15, +45.7% at alpha=0.70",
            "n_disagreements": len(disagree),
            "agree": not disagree,
        },
        "0b_selectivity_measurement": {
            "what": "west-edit d(gamma) ratio, x_axial / y_axial, per solver",
            "theory": "~2 for a wave solver (eps 2 vs 1); >>2 for a ray solver (0 at grazing)",
            "measured": sel, "gate": None,
        },
        "cases": cases,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)

    print("derivations agree (D54): {}  ({} disagreements)".format(not disagree, len(disagree)))
    print("\n0a  FDTD vs ANALYTIC (tol {:.0%})".format(TOL_0A))
    for c in cases:
        for r in c["modes"]:
            if r["fdtd"] and r["family"] in ("x_axial", "y_axial") and r["fdtd"]["r2"] >= 0.95:
                print("  {:18s} {:>7s} {:9s} theory {:7.3f}  fdtd {:7.3f}  {:+7.2%}".format(
                    c["case"], str(tuple(r["mode"])), r["family"],
                    r["gamma_theory_exact"], r["fdtd"]["gamma"],
                    r["fdtd_rel_vs_theory"]))
    print("\n  worst {:.2%} over {} modes, families {} -> {}".format(
        worst, len(judged), sorted(fam), "GO" if p0a else "NO-GO"))
    print("\n0b  west-edit selectivity x_axial/y_axial (measurement, no gate):")
    for k, v in sel.items():
        print("  {}: fdtd {:.2f}   ism {:.2f}".format(
            k, v.get("fdtd", float("nan")), v.get("ism", float("nan"))))
    print("-> {}".format(a.out))
    return 0 if p0a else 1


if __name__ == "__main__":
    raise SystemExit(main())
