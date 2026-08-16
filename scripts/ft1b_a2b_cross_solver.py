"""A2b-fix: a GENUINE two-solver cross-validation of the FDTD boundary calibration.

FT-A's A2b claimed to cross-validate FDTD against ISM. It did not: no image-source simulation
was ever run. `BW_ISM` was the *analytic* `ism_ray` formula, so A2b was an algebraic
restatement of A2 and validated nothing independent. (That was an error in the task spec, not
in its execution.)

This is the real thing: pyroomacoustics ShoeBox 2D at max_order=60 -- the exact simulator that
built the entire P3-2/P3-2b/P3-2c corpus -- against the FDTD solver, mode by mode, on the same
room with the same receivers.

Why it is the load-bearing gate. Every other FT-A gate compares FDTD against a formula. If the
alpha -> impedance mapping `xi = (1-sqrt(1-a))/(1+sqrt(1-a))` were mis-calibrated, the solver
would still be stable, still conserve energy, still show the right mode frequencies and
shapes, and still track the Kuttruff law's *shape* -- it would simply absorb the wrong amount,
and every gate scored against that same law would move together and agree. Only a second,
independently-implemented simulator can catch it.

Scope, stated because it bounds the conclusion: ISM and a locally-reacting wall genuinely
disagree by 30-44% on uniform alpha (grazing-incidence absorption, D48/D54), so this is NOT a
test that the two solvers agree in general. It is run in the regime where they must agree --
low-order axial modes, where incidence on the absorbing wall is near-normal -- and the
criterion is applied there.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.eval.modal_bandwidth import caps_from_predicted_bw, measure_modes
from aaf.eval.modal_projection import enumerate_modes, mode_shape_matrix
from aaf.sim.analytical_modal_2d import damping_to_bandwidth_hz, modal_damping_2d

C = 343.0
FS = 12288.0
N = 24576
DX = 0.05
MAX_ORDER = 60
BAND_HI = 200.0          # projection cap: cond(Phi) is only conditioned below this
ISM_FS = 4096.0          # the P3-2 corpus protocol; fs/n = 0.5 Hz, matching the FDTD grid
ISM_N = 8192
TOL = 0.10


def _receivers(L: float, W: float, nx: int = 8, ny: int = 8) -> np.ndarray:
    xs = np.linspace(L / (2 * nx), L - L / (2 * nx), nx)
    ys = np.linspace(W / (2 * ny), W - W / (2 * ny), ny)
    return np.array([[x, y] for x in xs for y in ys])


def run_ism(L: float, W: float, alphas, src, rx, n: int = N, fs: float = FS) -> np.ndarray:
    """pyroomacoustics ShoeBox 2D via the project's OWN helper.

    Deliberately ``aaf.sim.ism_2d.simulate_room_2d`` rather than a fresh pra call: that helper
    built every room in the P3-2/P3-2b/P3-2c corpus, so the fractional-delay handling, the 1/d
    convention, the truncation policy and max_order are byte-identical to the ground truth the
    model is trained against. A hand-rolled pra call could differ in any of those and the
    cross-validation would then be testing my second implementation, not the corpus.
    """
    from aaf.sim.ism_2d import simulate_room_2d
    out = simulate_room_2d(dict(
        L=float(L), W=float(W),
        source_pos=np.asarray(src, dtype=float),
        receiver_pos=np.asarray(rx, dtype=float),
        alphas=tuple(float(x) for x in alphas),
        fs=float(fs), n_time_samples=int(n), max_order=MAX_ORDER))
    return np.asarray(out["H_complex"])


def _measure(H, freqs, modes, phi, bw_pred):
    """The REPO estimator on the modally-projected spectrum -- identical for both streams.

    The first attempt at this gate used a raw receiver-averaged power spectrum with no walk
    caps. It reported FDTD (1,0) = 8.77 Hz where FT-A's validated path measures 3.08 Hz, and
    ISM bandwidths of 6114 Hz inside a 0-300 Hz band (the -3 dB walk simply ran off the array
    because neighbouring modes were never suppressed). Projection is what isolates a mode;
    caps are what stop the walk. Measuring the two solvers with DIFFERENT estimators, or with
    an unvalidated one, would make this gate a test of my harness rather than of the physics
    (the D49 rule: prediction and reference share one estimator).
    """
    spec = np.linalg.pinv(phi) @ H
    return measure_modes(np.abs(spec), freqs, modes, caps=caps_from_predicted_bw(bw_pred))


def compare(L: float, W: float, alphas, label: str) -> dict:
    src = (0.37 * L, 0.29 * W)
    rx = _receivers(L, W)

    t0 = time.perf_counter()
    fd = F.simulate(L, W, alphas, src=src, rx=rx, dx=DX, fs=FS, n=N, c=C)
    t_fdtd = time.perf_counter() - t0
    # Both solvers are evaluated at the FDTD's SNAPPED receiver positions, so the projection
    # basis is literally the same matrix for both and no interpolation enters the comparison.
    rx_snap = np.asarray(fd["meta"]["rx_pos_snapped"], dtype=float)

    t0 = time.perf_counter()
    H_ism = run_ism(L, W, alphas, src, rx_snap, n=ISM_N, fs=ISM_FS)
    t_ism = time.perf_counter() - t0

    modes = [m for m in enumerate_modes(L, W, f_max=BAND_HI) if not (m.n_x == 0 and m.n_y == 0)]
    phi = mode_shape_matrix(modes, rx_snap, L, W)
    cond = float(np.linalg.cond(phi))
    bw_pred = [damping_to_bandwidth_hz(
        modal_damping_2d(L, W, list(alphas), m.n_x, m.n_y, model="kuttruff")) for m in modes]

    f_fd = np.asarray(fd["freqs"], dtype=float)
    df_ism = ISM_FS / ISM_N
    f_ism = np.arange(H_ism.shape[-1], dtype=float) * df_ism
    assert abs((f_fd[1] - f_fd[0]) - df_ism) < 1e-12, "bin spacings must match"

    pk_fd = _measure(fd["H_complex"], f_fd, modes, phi, bw_pred)
    pk_is = _measure(H_ism, f_ism, modes, phi, bw_pred)

    rows = []
    for k, m in enumerate(modes):
        a_bw, b_bw = pk_fd[k].bw_3db_hz, pk_is[k].bw_3db_hz
        rel = (None if not (np.isfinite(a_bw) and np.isfinite(b_bw) and b_bw > 0)
               else float((a_bw - b_bw) / b_bw))
        rows.append({
            "mode": [m.n_x, m.n_y], "f_analytic_hz": float(m.f),
            "family": ("x_axial" if m.n_y == 0 else "y_axial" if m.n_x == 0 else "tangential"),
            "fdtd_bw_hz": float(a_bw), "ism_bw_hz": float(b_bw),
            "fdtd_flag": pk_fd[k].bw_flag, "ism_flag": pk_is[k].bw_flag,
            "fdtd_valid": bool(pk_fd[k].bw_valid), "ism_valid": bool(pk_is[k].bw_valid),
            "rel_diff_fdtd_vs_ism": rel,
            "kuttruff_bw_hz": float(bw_pred[k]),
            "ism_ray_bw_hz": float(damping_to_bandwidth_hz(
                modal_damping_2d(L, W, list(alphas), m.n_x, m.n_y, model="ism_ray"))),
        })
    return {"case": label, "L": L, "W": W, "alphas": list(alphas), "src": list(src),
            "n_rx": len(rx), "cond_phi": cond,
            "seconds": {"fdtd": t_fdtd, "ism": t_ism}, "modes": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ft1b/a2b_cross_solver.json")
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args()

    L, W = 4.5, 4.0
    # Low-order AXIAL modes only for the criterion: these are the ones whose incidence on the
    # absorbing wall is near-normal, so ISM and a locally-reacting wall must agree. Tangential
    # modes are reported but excluded -- that is where grazing incidence makes the two laws
    # legitimately differ by 30-44%, which is physics, not calibration error (D48/D54).
    cases = [
        ("uniform_a015", (0.15,) * 4),
        ("west_a070", (0.70, 0.15, 0.15, 0.15)),
    ]
    results = [compare(L, W, al, name) for name, al in cases]

    judged = [r for res in results for r in res["modes"]
              if r["family"] != "tangential" and r["rel_diff_fdtd_vs_ism"] is not None]
    worst = max((abs(r["rel_diff_fdtd_vs_ism"]) for r in judged), default=float("nan"))
    ok = bool(judged) and worst <= a.tol

    out = {
        "gate": "A2b-fix",
        "what": ("simulated pyroomacoustics ShoeBox 2D ISM (max_order=60) vs simulated FDTD, "
                 "per mode, same room and receivers"),
        "why": ("FT-A's A2b ran no ISM simulation -- BW_ISM was the analytic ism_ray formula, "
                "so it restated A2. This is the only gate that independently validates the "
                "alpha->impedance calibration; a mis-calibrated xi would keep every "
                "formula-scored gate self-consistent and wrong."),
        "criterion": ("axial modes only, |FDTD - ISM| / ISM <= {:.0%}; tangential reported but "
                      "excluded because grazing-incidence absorption makes the two laws "
                      "differ by 30-44% there BY PHYSICS (D48/D54)".format(a.tol)),
        "tolerance": a.tol,
        "n_axial_modes_judged": len(judged),
        "worst_rel_diff_axial": worst,
        "pass": ok,
        "verdict": ("PASS" if ok else
                    "FAIL -- STOP AND ESCALATE: a boundary-calibration error would invalidate "
                    "every topological-edit dataset built on this solver"),
        "cases": results,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)

    for res in results:
        print("--- {}  (fdtd {:.1f}s, ism {:.1f}s)".format(
            res["case"], res["seconds"]["fdtd"], res["seconds"]["ism"]))
        print("     {:>7s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
            "mode", "f_an", "fdtd_bw", "ism_bw", "rel_diff", "kuttruff"))
        for r in res["modes"]:
            rel = "  --  " if r["rel_diff_fdtd_vs_ism"] is None else "{:+7.2%}".format(
                r["rel_diff_fdtd_vs_ism"])
            mark = "" if r["family"] != "tangential" else "  (excl: tangential)"
            print("     {:>7s} {:9.3f} {:9.4f} {:9.4f} {:>9s} {:9.4f}{}".format(
                str(tuple(r["mode"])), r["f_analytic_hz"], r["fdtd_bw_hz"] or float("nan"),
                r["ism_bw_hz"] or float("nan"), rel, r["kuttruff_bw_hz"], mark))
    print("\nworst axial |rel diff| = {:.2%}  (tol {:.0%})  -> {}".format(
        worst, a.tol, out["verdict"]))
    print("-> {}".format(a.out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
