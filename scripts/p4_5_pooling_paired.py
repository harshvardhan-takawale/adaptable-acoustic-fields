"""P4-5 (manager item 6, D124): D105(d)'s paired pooling bootstrap, re-run UNDER B1's LOSS.

D105(d) compared Arm D (masked_mean) with C92 (attention) on the identical 92-shape corpus under the
BASELINE loss: off-lattice dense, D - C92 = +0.12 / +0.32 / +0.24 at 25-100 / 100-200 / 200-300 Hz.
The manager's question: is that "mean pooling is better", or "the baseline loss made attention
pooling memorise"? Under B1's loss at 250 shapes the pair is B8 (masked_mean + B1 weights) and B1
(attn_residual_extent + B1 weights), identical otherwise. Same statistic as D105(d): per frozen
shape, the band mean of the per-bin OFF-lattice dB spatial Pearson; paired difference; 10k bootstrap
over the 15 shapes (seed 20261013). The baseline-loss pair is re-run for reference (Arm D - C92, D105(d) itself). B8 - B1
isolates POOLING under B1's loss; B8 - B7 isolates the LOSS under mean pooling.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.p4_5_gate2_dense import on_lattice

G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
BANDS = {"25-100": (50, 200), "100-200": (200, 400), "200-300": (400, 601)}
NB, SEED = 10000, 20261013


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def colr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def per_shape(name):
    z = np.load(G2.format(name), allow_pickle=False)
    out = {b: [] for b in BANDS}
    i = 0
    while "P%d" % i in z.files:
        P, T, rx = db(z["P%d" % i][:, :601]), db(z["T%d" % i][:, :601]), z["rx%d" % i]
        off = ~on_lattice(rx)
        r = colr(P[off], T[off])
        for b, (lo, hi) in BANDS.items():
            out[b].append(float(np.nanmean(r[lo:hi])))
        i += 1
    return {b: np.array(v) for b, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="A:B,A:B,... (A minus B; ':' because names contain '-')")
    ap.add_argument("--out", default="outputs/p4_5/gate2_dense/POOLING_PAIRED_B1LOSS.json")
    a = ap.parse_args()
    rep = {"method": "D105(d) statistic: per-shape band-mean off-lattice dB Pearson, paired, 10k shape bootstrap", "pairs": {}}
    cache = {}
    for pr in a.pairs.split(","):
        A, B = pr.split(":")
        for n in (A, B):
            if n not in cache:
                cache[n] = per_shape(n)
        row = {}
        for b in BANDS:
            d = cache[A][b] - cache[B][b]
            rng = np.random.default_rng(SEED)
            bs = d[rng.integers(0, len(d), (NB, len(d)))].mean(1)
            row[b] = {"diff": float(d.mean()), "ci95": [float(v) for v in np.percentile(bs, [2.5, 97.5])],
                      "n_shapes_A_better": int((d > 0).sum()), "n_shapes": int(len(d))}
        rep["pairs"]["{} minus {}".format(A, B)] = row
        print("{:24s} ".format(A + " - " + B) + " | ".join("{}: {:+.3f} [{:+.3f}, {:+.3f}] ({}/{})".format(
            b, r["diff"], *r["ci95"], r["n_shapes_A_better"], r["n_shapes"]) for b, r in row.items()))
    json.dump(rep, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
