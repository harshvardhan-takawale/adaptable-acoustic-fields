"""Arm C v2, stage 1: per-mode accuracy across the whole modal hierarchy.

v1 reported spatial Pearson at the first THREE modes and drew the fundamental. That is the
easiest part of the band, so it cannot say where reconstruction actually holds. This walks
every analytic mode up to 200 Hz on the cached 64x64 fields and reports each one.

It runs entirely off the v1 `.npz` dumps -- those store the full (4096, 601) complex spectra to
300 Hz, so every mode below 200 Hz is already in hand. No GPU, no re-simulation, no checkpoint.

Two guards make the mode LABELS trustworthy rather than nominal, and both are recorded per mode
rather than asserted once in prose:

  * **isolation** -- a "mode" is only a mode if nothing else sits inside its linewidth. Kuttruff
    linewidth here is ~2 Hz; every mode carries its nearest-neighbour spacing and a flag.
  * **resolvability** -- the n_x <= floor(L / 2 dx_rx) bound that bit the 8x8 grid (D61e). At
    64x64 it passes with enormous margin, which is worth showing rather than assuming.

Both dB and LINEAR spatial Pearson are reported for every mode. They tell different stories --
the dB form compresses dynamic range and rewards getting the nodal pattern right, the linear
form weights the loud regions -- and reporting only one would repeat a mistake this project has
already made three times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aaf.eval.modal_projection import enumerate_modes
from aaf.eval.p3_2_eval import _pearson
from aaf.sim.analytical_modal_2d import damping_to_bandwidth_hz, modal_damping_2d

F_MAX_HZ = 200.0
DF_HZ = 4096.0 / 8192
ALPHA_BASE = 0.15
GEOMS = ("median", "small", "large")
SCEN = ("a_baseline", "b_east_curtain", "c_north_absorber", "d_two_wall")
PRIMARY = "median"
GOOD = 0.85


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def screen_geometry(tag, fdir):
    """Full per-mode table for one geometry. Baseline is the screen; all scenarios recorded."""
    dat = {s: np.load(Path(fdir) / "{}_{}.npz".format(tag, s)) for s in SCEN}
    d0 = dat["a_baseline"]
    L, W = float(d0["L"]), float(d0["W"])
    rx = d0["rx"]
    n = int(round(np.sqrt(rx.shape[0])))

    # receiver spacing -> the Nyquist-style resolvability bound (D61e's rule)
    ux = np.unique(np.round(rx[:, 0], 6))
    uy = np.unique(np.round(rx[:, 1], 6))
    d_x, d_y = float(np.diff(ux)[0]), float(np.diff(uy)[0])
    lim = {"n_x_max": int(np.floor(L / (2.0 * d_x))), "n_y_max": int(np.floor(W / (2.0 * d_y))),
           "rx_spacing_m": [d_x, d_y]}

    modes = enumerate_modes(L, W, f_max=F_MAX_HZ)
    fs_all = np.array([m.f for m in modes], float)
    rows = []
    for i, m in enumerate(modes):
        b = int(round(m.f / DF_HZ))
        lw = damping_to_bandwidth_hz(
            modal_damping_2d(L, W, [ALPHA_BASE] * 4, m.n_x, m.n_y, model="kuttruff"))
        gap = float(np.min(np.abs(np.delete(fs_all, i) - m.f))) if len(modes) > 1 else np.inf
        r = {"n_x": m.n_x, "n_y": m.n_y, "f_hz": float(m.f), "bin": b, "family": m.family,
             "linewidth_hz": float(lw), "nn_spacing_hz": gap,
             "isolated": bool(gap > lw),
             "resolvable": bool(m.n_x <= lim["n_x_max"] and m.n_y <= lim["n_y_max"]),
             "both_indices_nonzero": bool(m.n_x > 0 and m.n_y > 0),
             "pearson_db": {}, "pearson_lin": {}}
        for s in SCEN:
            p, g = dat[s]["pred"][:, b], dat[s]["gt"][:, b]
            r["pearson_db"][s] = _pearson(_db(p), _db(g))
            r["pearson_lin"][s] = _pearson(np.abs(p), np.abs(g))
        rows.append(r)

    base = np.array([r["pearson_db"]["a_baseline"] for r in rows], float)
    above60 = [r for r in rows if r["f_hz"] > 60.0]
    return {
        "geometry": tag, "L": L, "W": W, "n_rx": int(rx.shape[0]), "grid": [n, n],
        "resolvability": lim, "f_max_hz": F_MAX_HZ, "df_hz": DF_HZ, "n_modes": len(rows),
        "screen_scenario": "a_baseline",
        "baseline_db_min": float(base.min()), "baseline_db_mean": float(base.mean()),
        "baseline_db_max": float(base.max()),
        "n_ge_070": int((base >= 0.70).sum()), "n_ge_085": int((base >= GOOD).sum()),
        "f_highest_ge_085": float(max([r["f_hz"] for r in rows
                                       if r["pearson_db"]["a_baseline"] >= GOOD], default=np.nan)),
        "n_above_60hz": len(above60),
        "n_above_60hz_ge_085": int(sum(r["pearson_db"]["a_baseline"] >= GOOD for r in above60)),
        "all_isolated": bool(all(r["isolated"] for r in rows)),
        "all_resolvable": bool(all(r["resolvable"] for r in rows)),
        "modes": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="outputs/armC_demo/fields")
    ap.add_argument("--out", default="outputs/armC_demo/v2/mode_screen.json")
    a = ap.parse_args()

    res = {g: screen_geometry(g, a.fields) for g in GEOMS}
    p = res[PRIMARY]
    print("PRIMARY geometry '{}'  {:.2f} x {:.2f} m | {} modes <= {:.0f} Hz | grid {}x{}".format(
        PRIMARY, p["L"], p["W"], p["n_modes"], F_MAX_HZ, *p["grid"]))
    print("resolvable to n_x<={} n_y<={} (rx spacing {:.4f} x {:.4f} m) | all isolated: {}".format(
        p["resolvability"]["n_x_max"], p["resolvability"]["n_y_max"],
        *p["resolvability"]["rx_spacing_m"], p["all_isolated"]))
    print("\n  mode      f_Hz  bin |   R_dB    R_lin | lw_Hz  nn_Hz  iso 2D")
    for r in p["modes"]:
        print("  ({},{})  {:8.2f} {:4d} | {:+.3f}  {:+.3f} | {:5.2f} {:6.2f}   {}  {}".format(
            r["n_x"], r["n_y"], r["f_hz"], r["bin"], r["pearson_db"]["a_baseline"],
            r["pearson_lin"]["a_baseline"], r["linewidth_hz"], r["nn_spacing_hz"],
            "Y" if r["isolated"] else "n", "Y" if r["both_indices_nonzero"] else "-"))
    print("\nbaseline dB Pearson: min {:+.3f} mean {:+.3f} max {:+.3f}".format(
        p["baseline_db_min"], p["baseline_db_mean"], p["baseline_db_max"]))
    print(">=0.70: {}/{} | >=0.85: {}/{} (highest {:.1f} Hz)".format(
        p["n_ge_070"], p["n_modes"], p["n_ge_085"], p["n_modes"], p["f_highest_ge_085"]))
    print("ABORT CHECK -- modes >60 Hz clearing 0.85: {} of {} -> {}".format(
        p["n_above_60hz_ge_085"], p["n_above_60hz"],
        "PROCEED" if p["n_above_60hz_ge_085"] > 0 else "STOP AND REPORT"))
    for g in GEOMS:
        if g != PRIMARY:
            q = res[g]
            print("  [secondary] {:6s} {:.2f}x{:.2f}: {} modes, mean {:+.3f}, "
                  ">=0.85 {}/{}".format(g, q["L"], q["W"], q["n_modes"],
                                        q["baseline_db_mean"], q["n_ge_085"], q["n_modes"]))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"primary": PRIMARY, "abort_rule": "report if NO mode above 60 Hz reaches 0.85",
               "proceed": bool(p["n_above_60hz_ge_085"] > 0), "good_threshold": GOOD,
               "geometries": res}, open(a.out, "w"), indent=1, default=float)
    print("\n-> {}".format(a.out))
    return 0 if p["n_above_60hz_ge_085"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
