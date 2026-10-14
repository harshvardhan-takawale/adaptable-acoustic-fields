"""P4-5 (manager, D119 items 1-2): the two metrics an arm must carry before it enters a headline.

1. BAND LSD BELOW 100 Hz, AS A FIELD. D106 inverted the shape-vs-material parity claim on exactly
   this number (shape 3.41-3.89 dB off-lattice vs material 1.7-2.3 dB). Definition identical to
   `p4_5_lattice_diag.py` (the source of D106's 3.41 / 3.89): mean |dB(P) - dB(T)| over bins
   [0, 200) (0-99.5 Hz) and over receivers, here OFF the training lattice, on all 15 frozen test
   shapes at dense receivers (D102's data), per shape, then averaged; 95% CI by a 10k bootstrap
   over shapes (seed 20261013). The 3-shape A2 value D106 quoted is printed beside it.

2. FULL-BAND RIR PEARSON. D96 named it the first metric to show damage if an arm hurts the time
   response; D70 is the precedent (DC-masked loss: 0.9989 -> 0.0630). The modal RIR excludes DC,
   so it cannot see that failure. Here: irfft of ALL 601 bins (DC included, n = 1200), Pearson
   per receiver against the FDTD's, averaged over every OFF-lattice dense receiver of each shape,
   then over shapes, with the same bootstrap. The receiver-0 "Gate-2 style" number from each
   arm's eval JSON (800-receiver data, D70's exact metric) is printed beside it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.p4_5_gate2_dense import on_lattice

CACHE = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
NB, SEED = 10000, 20261013
LO = (0, 200)


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def rir_r(P, T):
    n = 2 * (P.shape[1] - 1)
    a, b = np.fft.irfft(P, n=n, axis=1), np.fft.irfft(T, n=n, axis=1)
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def per_shape(name):
    z = np.load(CACHE.format(name), allow_pickle=False)
    lsd, rir = [], []
    i = 0
    while "P%d" % i in z.files:
        P, T, rx = z["P%d" % i][:, :601], z["T%d" % i][:, :601], z["rx%d" % i]
        off = ~on_lattice(rx)
        lsd.append(float(np.mean(np.abs(db(P[off, LO[0]:LO[1]]) - db(T[off, LO[0]:LO[1]])))))
        rir.append(float(np.nanmean(rir_r(P[off], T[off]))))
        i += 1
    return np.array(lsd), np.array(rir)


def boot(x):
    rng = np.random.default_rng(SEED)
    b = x[rng.integers(0, len(x), (NB, len(x)))].mean(1)
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="gate2_dense cache names")
    ap.add_argument("--evals", default="", help="matching p4_5_eval names (for the A2 3-shape LSD and the rx-0 RIR), '-' to skip")
    ap.add_argument("--out", default="outputs/p4_5/gate2_dense/RISK_METRICS.json")
    a = ap.parse_args()
    names = a.names.split(",")
    evals = a.evals.split(",") if a.evals else ["-"] * len(names)
    rows = []
    print("{:22s} | {:>6s} {:>17s} {:>7s} | {:>7s} {:>17s} {:>9s}".format(
        "checkpoint", "LSD<100", "95% CI (15 shapes)", "3-shape", "fullRIR", "95% CI", "rx0 G2"))
    for n, e in zip(names, evals):
        lsd, rir = per_shape(n)
        r = {"name": n, "n_shapes": int(len(lsd)), "lsd_0_100_off": float(lsd.mean()), "lsd_ci95": boot(lsd),
             "rir_fullband_off": float(rir.mean()), "rir_ci95": boot(rir),
             "lsd_per_shape": lsd.tolist(), "rir_per_shape": rir.tolist()}
        if e != "-":
            lat = json.load(open("outputs/p4_5/diagnostics/lattice_{}.json".format(e)))
            ev = json.load(open("outputs/p4_5/eval/{}.json".format(e)))
            r["lsd_0_100_off_3shape_A2"] = lat["off_lattice"]["lsd_db"]["0-100"]
            r["rir_gate2_style_rx0"] = ev["rir_pearson_gate2_style"]
            r["rir_modal_20_300"] = ev["rir_pearson_modal_20_300"]
        rows.append(r)
        print("{:22s} | {:6.2f}  [{:5.2f}, {:5.2f}]  {:>7s} | {:7.4f} [{:.4f}, {:.4f}] {:>9s}".format(
            n, r["lsd_0_100_off"], *r["lsd_ci95"],
            "{:.2f}".format(r["lsd_0_100_off_3shape_A2"]) if "lsd_0_100_off_3shape_A2" in r else "-",
            r["rir_fullband_off"], *r["rir_ci95"],
            "{:.4f}".format(r["rir_gate2_style_rx0"]) if "rir_gate2_style_rx0" in r else "-"))
    json.dump({"method": __doc__.strip().splitlines()[0], "lsd_band_bins": list(LO), "rows": rows},
              open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
