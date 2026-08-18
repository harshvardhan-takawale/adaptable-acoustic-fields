"""FT-B criterion (iii): effect size vs a MEASURED floor.

FT-B returned GO on (i) smoothness and (ii) linearizing coordinate, but (iii) was never
computed, so that GO has been resting on two of three criteria. The gap was not laziness in the
sweep: (i) and (ii) are shape properties of the response curve, while (iii) needs a NOISE FLOOR
for the observable, and the inter-room level difference is a LEVEL in dB -- the project's
familiar 0.040 Hz estimator floor is a BANDWIDTH floor and does not transfer.

So the floor is measured here the same way FT-C measured its own: replicate runs of the SAME
configuration differing only in source position. Whatever spread that produces is the
reproducibility of the observable, and the effect size is the sweep's total response range
divided by it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import aaf.sim.fdtd_2d as F
from scripts.p3_3fast_ftb import band_level_ratio  # reuse FT-B's own definition

C = 343.0
L, W = 8.0, 4.0
DX = 0.01
FS, N = 61440.0, 122880          # holds lambda at the frozen 0.55827; T=2.000 s, df=0.5 Hz
X0, A_REF = 4.0, 0.5


def receivers(nx=16, ny=8, margin=0.3):
    xs = np.linspace(margin, L - margin, nx)
    ys = np.linspace(margin, W - margin, ny)
    return np.array([[x, y] for x in xs for y in ys])


def run(src):
    walls = [{"type": "slab", "axis": "x", "pos": X0,
              "apertures": [(W / 2 - A_REF / 2, W / 2 + A_REF / 2)], "alpha": 0.15}]
    return F.simulate(L, W, (0.15,) * 4, src=src, rx=receivers(), dx=DX, fs=FS, n=N, c=C,
                      extra_walls=walls)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/p3_3fast/trackB/effect_size.json")
    a = ap.parse_args()
    # Replicates differ ONLY in source position, both inside room A.
    srcs = [(0.5, 0.5), (0.7, 1.3), (0.4, 2.6)]
    lds = []
    for s in srcs:
        o = run(s)
        rx = np.asarray(o["meta"]["rx_pos_snapped"], float)
        H = np.abs(np.asarray(o["H_complex"]))
        A = rx[:, 0] < X0 - DX
        B = rx[:, 0] > X0 + DX
        hi = int(round(300.0 / (FS / N))) + 1
        ld = 20.0 * np.log10(max(H[B, :hi].mean(), 1e-30) / max(H[A, :hi].mean(), 1e-30))
        lds.append(float(ld))
        print("  src {} -> level difference {:+.4f} dB".format(s, ld), flush=True)

    lds = np.array(lds)
    floor = float(lds.std(ddof=1))
    rng = 11.157706571919036          # FT-B's measured total range over a > 0
    ratio = rng / floor if floor > 0 else float("inf")
    out = {
        "criterion": "FT-B (iii) effect size vs floor",
        "floor_method": ("replicate runs of the SAME config (a=0.5) differing only in source "
                         "position -- the same instrument FT-C used, because the 0.040 Hz "
                         "project floor is a BANDWIDTH floor and does not transfer to a level"),
        "n_replicates": len(srcs), "source_positions": [list(s) for s in srcs],
        "level_difference_db": lds.tolist(),
        "floor_db_sd": floor,
        "floor_db_range": float(lds.max() - lds.min()),
        "sweep_total_range_db": rng,
        "effect_size_over_floor": ratio,
        "threshold": 10.0,
        "pass": bool(ratio >= 10.0),
        "verdict": ("GO" if ratio >= 10.0 else "NO-GO"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("\nfloor (sd over {} replicates) = {:.4f} dB".format(len(srcs), floor))
    print("sweep range {:.3f} dB -> effect size {:.1f}x floor (threshold 10x) -> {}".format(
        rng, ratio, out["verdict"]))
    print("-> {}".format(a.out))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
