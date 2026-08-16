"""FT-C: does WHERE an absorber sits carry signal beyond how much of it there is?

One room (4.5 x 4.0), west wall segmented into 8 strips of 0.5 m. Baseline alpha = 0.15
everywhere; one contiguous patch at alpha = 0.70 is swept over extent x position.

The decisive question is not "does the patch change the field" -- of course it does -- but
whether position adds anything **beyond area-weighted mean alpha**. If it does not, this axis
collapses into P3-2b's whole-wall material axis and there is no chunk here.

Three design choices that decide whether the test can answer that:

* **Everything is measured within FDTD, and every observable is a PAIRED delta against the
  same room's uniform-0.15 baseline.** The repo's -3 dB estimator carries a kappa calibrated
  on ISM data (A2b), so absolute widths are not portable between solvers -- but a paired
  delta inside one solver cancels the width convention entirely. The mean-alpha prediction is
  therefore fitted from these configs themselves, not imported from the ISM-derived P3-2b
  law.
* **The residual is restricted to n_y >= 1.** For n_y = 0 the mode is uniform along the west
  wall (cos^2(0) = 1), so its mode-weighted mean absorption IS the area-weighted mean, by
  algebra. Those modes carry exactly zero position information and including them dilutes the
  residual toward a guaranteed NO-GO.
* **2-segment is the headline.** A 1-segment patch was pre-computed at ~4.7x the floor
  against a 5x threshold -- underpowered before it is run. 4-segment patches alias against
  n_y = 4 and are reported as a null, not as evidence.

The floor is measured, not assumed: replicate runs that differ only in source position give
the within-solver reproducibility of each per-mode delta.
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
# fs must scale with 1/dx or the CFL bound is violated (dx=0.02 at fs=12288 gives
# lam = 1.396 against a bound of 1.0 -- caught by the A0a assertion). These pairs hold lambda
# at EXACTLY the frozen dx=0.05/fs=12288 value (0.55827), so the dispersion characterization
# carries over unchanged, while n = 2*fs keeps T = 2.000 s and df = 0.5 Hz exactly.
FS_FOR_DX = {0.05: 12288.0, 0.02: 30720.0, 0.01: 61440.0}
DX = 0.02
FS = FS_FOR_DX[DX]
N = int(2 * FS)
L, W = 4.5, 4.0
N_SEG = 8
SEG = W / N_SEG                      # 0.5 m
A_BASE, A_PATCH = 0.15, 0.70
F_MAX = 200.0
RES_FLOOR_MULT = 5.0
ANTINODE_R_MIN = 0.70


def _receivers(nx: int = 8, ny: int = 8) -> np.ndarray:
    xs = np.linspace(L / (2 * nx), L - L / (2 * nx), nx)
    ys = np.linspace(W / (2 * ny), W - W / (2 * ny), ny)
    return np.array([[x, y] for x in xs for y in ys])


def _basis(rx: np.ndarray):
    modes = [m for m in enumerate_modes(L, W, f_max=F_MAX)
             if not (m.n_x == 0 and m.n_y == 0)]
    phi = mode_shape_matrix(modes, rx, L, W)
    return modes, phi, float(np.linalg.cond(phi))


def _measure(H, freqs, modes, phi, bw_pred):
    spec = np.linalg.pinv(phi) @ H
    return measure_modes(np.abs(spec), freqs, modes, caps=caps_from_predicted_bw(bw_pred))


def _run(extra_walls, src, rx, modes, phi, bw_pred):
    out = F.simulate(L, W, (A_BASE,) * 4, src=src, rx=rx, dx=DX, fs=FS, n=N, c=C,
                     extra_walls=extra_walls)
    pk = _measure(out["H_complex"], np.asarray(out["freqs"], float), modes, phi, bw_pred)
    bw = np.array([p.bw_3db_hz if p.bw_valid else np.nan for p in pk])
    return bw, out


def mean_alpha(lo: float, hi: float) -> float:
    """Area-weighted mean absorption of the west wall with the patch on [lo, hi]."""
    frac = (hi - lo) / W
    return A_BASE + frac * (A_PATCH - A_BASE)


def modal_mean_alpha(lo: float, hi: float, n_y: int) -> float:
    """cos^2-weighted mean absorption seen by mode n_y on the west wall.

    weight(y) = cos^2(n_y pi y / W), normalized. For n_y = 0 this reduces to the
    area-weighted mean exactly -- which is precisely why those modes are excluded from the
    position residual.
    """
    y = np.linspace(0.0, W, 2001)
    w = np.cos(n_y * np.pi * y / W) ** 2
    a = np.where((y >= lo) & (y <= hi), A_PATCH, A_BASE)
    return float(np.trapz(w * a, y) / np.trapz(w, y))


def patch_pressure_sq(lo: float, hi: float, n_y: int) -> float:
    """Mean pressure^2 of the mode over the patch region (analytic cosine shapes)."""
    y = np.linspace(lo, hi, 501)
    return float(np.mean(np.cos(n_y * np.pi * y / W) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ft1b/patch_sweep.json")
    a = ap.parse_args()
    t_start = time.perf_counter()

    rx = _receivers()
    modes, phi, cond = _basis(rx)
    bw_pred = [damping_to_bandwidth_hz(
        modal_damping_2d(L, W, [A_BASE] * 4, m.n_x, m.n_y, model="kuttruff")) for m in modes]
    src = (0.37 * L, 0.29 * W)

    # ---- baseline + replicate floor (same room, different source -> should be identical)
    bw_base, _ = _run(None, src, rx, modes, phi, bw_pred)
    bw_base_rep, _ = _run(None, (0.61 * L, 0.44 * W), rx, modes, phi, bw_pred)
    floor_per_mode = np.abs(bw_base - bw_base_rep)
    floor = float(np.nanmedian(floor_per_mode))

    # ---- sweep: extent in segments x every valid offset
    configs = []
    for n_seg in (1, 2, 4):
        for k in range(N_SEG - n_seg + 1):
            lo, hi = k * SEG, (k + n_seg) * SEG
            configs.append({"n_seg": n_seg, "lo": lo, "hi": hi})
    configs.append({"n_seg": N_SEG, "lo": 0.0, "hi": W, "whole_wall": True})

    rows = []
    for cfg in configs:
        spec = {"type": "patch", "wall": "west", "span": (cfg["lo"], cfg["hi"]),
                "alpha": A_PATCH}
        bw, out = _run([spec], src, rx, modes, phi, bw_pred)
        realized = out["meta"]["extra_walls"][0]
        d_bw = bw - bw_base
        rows.append({
            **cfg,
            "width_requested_m": realized["width_requested_m"],
            "width_realized_m": realized["width_realized_m"],
            "mean_alpha": mean_alpha(cfg["lo"], cfg["hi"]),
            "d_bw": [None if not np.isfinite(v) else float(v) for v in d_bw],
        })
        print("  patch {:.2f}-{:.2f} m ({} seg, realized {:.3f} m): mean_alpha {:.4f}".format(
            cfg["lo"], cfg["hi"], cfg["n_seg"], realized["width_realized_m"],
            mean_alpha(cfg["lo"], cfg["hi"])), flush=True)

    # ---- the decisive control, per mode with n_y >= 1
    ny1 = [i for i, m in enumerate(modes) if m.n_y >= 1]
    analysis = {}
    for tag, sel in (("1seg", 1), ("2seg", 2), ("4seg", 4)):
        sub = [r for r in rows if r["n_seg"] == sel]
        res_all, anti_x, anti_y = [], [], []
        for i in ny1:
            x = np.array([r["mean_alpha"] for r in sub], float)
            y = np.array([np.nan if r["d_bw"][i] is None else r["d_bw"][i] for r in sub], float)
            ok = np.isfinite(y)
            if ok.sum() < 3:
                continue
            # Within a fixed extent, mean_alpha is CONSTANT across positions -- so any spread
            # in d_bw at fixed extent is position information by construction. The residual is
            # therefore the deviation from that extent's own mean.
            res = y[ok] - np.nanmean(y[ok])
            res_all.extend(np.abs(res).tolist())
            for r, v in zip([s for s, o in zip(sub, ok) if o], y[ok]):
                anti_x.append(patch_pressure_sq(r["lo"], r["hi"], modes[i].n_y))
                anti_y.append(float(v))
        med_res = float(np.median(res_all)) if res_all else float("nan")
        r_anti = (float(np.corrcoef(anti_x, anti_y)[0, 1])
                  if len(anti_x) > 2 and np.std(anti_x) > 0 else float("nan"))
        analysis[tag] = {
            "n_positions": len(sub), "n_modes_ny_ge_1": len(ny1),
            "median_abs_position_residual_hz": med_res,
            "residual_over_floor": (med_res / floor) if floor > 0 else float("nan"),
            "antinode_pearson_r": r_anti,
        }

    head = analysis["2seg"]
    go = (np.isfinite(head["residual_over_floor"])
          and head["residual_over_floor"] >= RES_FLOOR_MULT
          and np.isfinite(head["antinode_pearson_r"])
          and head["antinode_pearson_r"] >= ANTINODE_R_MIN)

    out = {
        "gate": "FT-C",
        "question": "does absorber POSITION carry signal beyond area-weighted mean alpha?",
        "room": {"L": L, "W": W, "dx": DX, "n_segments": N_SEG, "segment_m": SEG},
        "alpha": {"baseline": A_BASE, "patch": A_PATCH},
        "cond_phi": cond,
        "n_modes": len(modes),
        "n_modes_ny_ge_1": len(ny1),
        "floor_hz": floor,
        "floor_method": ("median |d bandwidth| between two runs of the SAME room differing "
                         "only in source position -- a measured within-solver replicate "
                         "floor, not an assumed one"),
        "why_ny_ge_1": ("for n_y = 0 the mode is uniform along the west wall, so its "
                        "cos^2-weighted mean absorption equals the area-weighted mean BY "
                        "ALGEBRA; those modes carry zero position information and would "
                        "dilute the residual toward a guaranteed NO-GO"),
        "headline": "2seg",
        "thresholds": {"residual_over_floor": RES_FLOOR_MULT,
                       "antinode_pearson_r": ANTINODE_R_MIN},
        "analysis": analysis,
        "verdict": ("GO" if go else "NO-GO"),
        "verdict_note": ("4seg aliases against n_y = 4 and is reported as a NULL, not as "
                         "evidence; 1seg was pre-computed at ~4.7x floor against a 5x "
                         "threshold, i.e. underpowered before it was run"),
        "configs": rows,
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)

    print("\nfloor = {:.4f} Hz (measured replicate)".format(floor))
    print("{:6s} {:>10s} {:>16s} {:>12s}".format("extent", "n_pos", "resid/floor", "antinode r"))
    for k, v in analysis.items():
        print("{:6s} {:10d} {:16.2f} {:12.3f}".format(
            k, v["n_positions"], v["residual_over_floor"], v["antinode_pearson_r"]))
    print("\nFT-C verdict: {}  (headline 2seg: residual >= {}x floor and r >= {})".format(
        out["verdict"], RES_FLOOR_MULT, ANTINODE_R_MIN))
    print("-> {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
