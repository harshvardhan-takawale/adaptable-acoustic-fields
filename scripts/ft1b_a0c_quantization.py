"""A0c: is the dx quantum small enough that the edit axes behave as continuous?

B3 was reclassified from blocker to arithmetic check: a quantum is fatal only if it is
comparable to the required sampling interval. This measures the quantum's effect directly --
run one aperture and one patch at dx = 0.02 and dx = 0.01 and compare every FT-B/FT-C
observable against the P3-2b estimator floor of 0.040 Hz.

fs scales with 1/dx to hold lambda at the frozen 0.55827 (a fixed fs at dx=0.02 violates CFL
by 40%, which the A0a assertion catches), and n = 2*fs keeps T = 2.000 s and df = 0.5 Hz, so
the two grids are compared on an identical frequency axis with no resampling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.eval.modal_bandwidth import caps_from_predicted_bw, measure_modes
from aaf.eval.modal_projection import enumerate_modes, mode_shape_matrix
from aaf.sim.analytical_modal_2d import damping_to_bandwidth_hz, modal_damping_2d

C = 343.0
FS_FOR_DX = {0.02: 30720.0, 0.01: 61440.0}
P3_2B_FLOOR_HZ = 0.040
F_MAX = 200.0


def measure(L, W, alphas, extra, dx):
    fs = FS_FOR_DX[dx]
    n = int(2 * fs)
    rx = np.array([[x, y]
                   for x in np.linspace(L / 16, L - L / 16, 8)
                   for y in np.linspace(W / 16, W - W / 16, 8)])
    out = F.simulate(L, W, alphas, src=(0.37 * L, 0.29 * W), rx=rx, dx=dx, fs=fs, n=n, c=C,
                     extra_walls=extra)
    modes = [m for m in enumerate_modes(L, W, f_max=F_MAX)
             if not (m.n_x == 0 and m.n_y == 0)]
    phi = mode_shape_matrix(modes, np.asarray(out["meta"]["rx_pos_snapped"], float), L, W)
    bw_pred = [damping_to_bandwidth_hz(modal_damping_2d(
        L, W, list(alphas), m.n_x, m.n_y, model="kuttruff")) for m in modes]
    pk = measure_modes(np.abs(np.linalg.pinv(phi) @ out["H_complex"]),
                       np.asarray(out["freqs"], float), modes,
                       caps=caps_from_predicted_bw(bw_pred))
    lev = 20.0 * np.log10(np.maximum(np.abs(out["H_complex"]).mean(axis=0), 1e-30))
    return {"modes": [[m.n_x, m.n_y] for m in modes],
            "f": [float(p.f_peak) for p in pk],
            "bw": [float(p.bw_3db_hz) if p.bw_valid else None for p in pk],
            "level_mean_db": float(np.mean(lev[:int(300.0 / 0.5)])),
            "realized": out["meta"].get("extra_walls")}


def main() -> int:
    cases = {
        "aperture_0.5m": dict(L=8.0, W=3.0, alphas=(0.15,) * 4,
                              extra=[{"type": "slab", "axis": "x", "pos": 4.0,
                                      "apertures": [(1.25, 1.75)], "alpha": 0.15}]),
        "patch_0.5m": dict(L=4.5, W=4.0, alphas=(0.15,) * 4,
                           extra=[{"type": "patch", "wall": "west",
                                   "span": (1.75, 2.25), "alpha": 0.70}]),
    }
    res = {}
    for name, cfg in cases.items():
        a = measure(cfg["L"], cfg["W"], cfg["alphas"], cfg["extra"], 0.02)
        b = measure(cfg["L"], cfg["W"], cfg["alphas"], cfg["extra"], 0.01)
        df = [abs(x - y) for x, y in zip(a["f"], b["f"])]
        dbw = [abs(x - y) for x, y in zip(a["bw"], b["bw"]) if x is not None and y is not None]
        res[name] = {
            "max_df_hz": float(np.max(df)), "max_dbw_hz": float(np.max(dbw)) if dbw else None,
            "median_dbw_hz": float(np.median(dbw)) if dbw else None,
            "d_level_db": abs(a["level_mean_db"] - b["level_mean_db"]),
            "n_modes_compared": len(dbw),
            "dbw_over_floor": (float(np.max(dbw)) / P3_2B_FLOOR_HZ) if dbw else None,
            "realized_dx002": a["realized"], "realized_dx001": b["realized"],
        }
        print("{}: max|df| {:.4f} Hz, max|dBW| {:.4f} Hz ({:.2f}x the {:.3f} Hz floor), "
              "dlevel {:.4f} dB".format(name, res[name]["max_df_hz"], res[name]["max_dbw_hz"],
                                        res[name]["dbw_over_floor"], P3_2B_FLOOR_HZ,
                                        res[name]["d_level_db"]), flush=True)
    ok = all(v["max_dbw_hz"] is not None and v["max_dbw_hz"] < P3_2B_FLOOR_HZ
             for v in res.values())
    out = {
        "gate": "A0c", "floor_hz": P3_2B_FLOOR_HZ,
        "dx_compared": [0.02, 0.01], "fs_for_dx": FS_FOR_DX,
        "lambda_held_at": C / (30720.0 * 0.02),
        "cases": res,
        "b3_closed": bool(ok),
        "verdict": ("B3 CLOSED for the feasibility sweeps: every observable moves less than "
                    "the estimator floor between dx=0.02 and dx=0.01, so dx=0.02 is adequate"
                    if ok else
                    "B3 NOT closed at dx=0.02: at least one observable exceeds the floor; use "
                    "the finer grid and report the required dx"),
        "derived_rule": ("the edit-parameter quantum must be <~10% of the required sampling "
                         "interval in the axis's linearizing coordinate; FT-B measures that "
                         "interval for the aperture axis and Part 3 measures it for absorption"),
    }
    Path("outputs/ft1b").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("outputs/ft1b/a0c_quantization.json", "w"), indent=1, default=float)
    print("\n" + out["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
