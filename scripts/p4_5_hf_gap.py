"""P4-5: the on-minus-off lattice gap ABOVE Gate 2's band, on all 15 frozen shapes.

Gate 2's gap (D102, D105) is measured on six mode bins at ~26-81 Hz. The 3-shape A2 split (D99)
covers 25-300 Hz but on three rooms only. This takes the dense 15-shape renders already cached by
`p4_5_gate2_dense.py`, computes the per-bin dB spatial Pearson ON and OFF the training lattice for
every bin of each band, averages over the band, and reports the paired gap with a 10k bootstrap
over shapes (seed 20261013). Used to check whether an arm trained OFF the lattice (B6) shows any
residual lattice preference at high frequency, against arms on the lattice with and without one.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.p4_5_gate2_dense import on_lattice

CACHE = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
BANDS = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
NB, SEED = 10000, 20261013


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def colr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", default="outputs/p4_5/gate2_dense/HF_GAP.json")
    a = ap.parse_args()
    rows = []
    print("{:16s} | ".format("checkpoint") + " | ".join("{:^34s}".format(b + " Hz: off / gap [95% CI]") for b in BANDS))
    for n in a.names.split(","):
        z = np.load(CACHE.format(n), allow_pickle=False)
        on_r = {b: [] for b in BANDS}; off_r = {b: [] for b in BANDS}
        i = 0
        while "P%d" % i in z.files:
            P, T, rx = db(z["P%d" % i][:, :601]), db(z["T%d" % i][:, :601]), z["rx%d" % i]
            m = on_lattice(rx)
            for b, (lo, hi) in BANDS.items():
                on_r[b].append(float(np.nanmean(colr(P[m, lo:hi], T[m, lo:hi]))))
                off_r[b].append(float(np.nanmean(colr(P[~m, lo:hi], T[~m, lo:hi]))))
            i += 1
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, i, (NB, i))
        r = {"name": n, "n_shapes": i}
        line = "{:16s} | ".format(n)
        for b in BANDS:
            on, off = np.array(on_r[b]), np.array(off_r[b])
            g = on - off
            ci = [float(v) for v in np.percentile(g[idx].mean(1), [2.5, 97.5])]
            r[b] = {"off": float(off.mean()), "on": float(on.mean()), "gap": float(g.mean()), "gap_ci95": ci}
            line += "{:+.3f} / {:+.3f} [{:+.3f}, {:+.3f}]    | ".format(off.mean(), g.mean(), *ci)
        rows.append(r)
        print(line)
    json.dump({"method": "per-bin dB spatial Pearson, band mean, 15 dense shapes, paired shape bootstrap",
               "rows": rows}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
