"""P4-5 (manager item 0, D129): the per-frequency off-lattice curve OUT TO 300 Hz, with the top-of-band
value published, and the 200-300 Hz band LSD beside it.

Two protocols, both OFF the training lattice:
  * A2 (3 frozen dense shapes, `outputs/p4_5/eval/cache/render_<name>.npz`, via
    `p4_5_figures.split_curves`) -- the protocol behind every "r at 200 Hz" quoted so far
    (B1 @30K 0.451). Pre-registered marks are the mean over +-2.5 Hz (11 bins).
  * 15 shapes (Gate-2's frozen set at dense receivers, `outputs/p4_5/gate2_dense/cache/`), per-bin
    dB spatial Pearson per shape then averaged, 95% CI by a 10k bootstrap over shapes.
Marks every 25 Hz from 25 to 300 Hz. **TOP OF BAND = 295-300 Hz (bins 590-600, one-sided)**
because there is no data above 300 Hz; it is labelled as such wherever it appears (D129).
Band LSD 200-300 Hz off-lattice, 15 shapes, is the mean |dB error| over bins 400-600 (D126).

`--check` first reproduces the A2 200 Hz marks published in `fig_hf_curve_*.json` exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.p4_5_figures import split_curves
from scripts.p4_5_gate2_dense import on_lattice

DF, HALF, NB, SEED = 0.5, 5, 10000, 20261013
MARKS = list(range(25, 301, 25))
G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
A2 = "outputs/p4_5/eval/cache/render_{}.npz"


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def colr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def window(hz):
    b = int(round(hz / DF))
    return (b - 2 * HALF, b + 1) if hz == 300 else (b - HALF, b + HALF + 1)     # 300 Hz: one-sided


def a2_marks(name):
    _on, off = split_curves(A2.format(name))
    out = {str(hz): float(np.nanmean(off[slice(*window(hz))])) for hz in MARKS}
    out["band_200_300_mean_r"] = float(np.nanmean(off[400:601]))
    return out, off


def g15(name):
    z = np.load(G2.format(name), allow_pickle=False)
    per, lsd = [], []
    i = 0
    while "P%d" % i in z.files:
        P, T, rx = db(z["P%d" % i][:, :601]), db(z["T%d" % i][:, :601]), z["rx%d" % i]
        off = ~on_lattice(rx)
        per.append(colr(P[off], T[off]))
        lsd.append(float(np.mean(np.abs(P[off, 400:601] - T[off, 400:601]))))
        i += 1
    per = np.array(per); lsd = np.array(lsd)
    rng = np.random.default_rng(SEED); idx = rng.integers(0, i, (NB, i))
    out = {}
    for hz in MARKS:
        v = np.nanmean(per[:, slice(*window(hz))], axis=1)
        out[str(hz)] = [float(v.mean()), *map(float, np.percentile(v[idx].mean(1), [2.5, 97.5]))]
    out["lsd_200_300"] = [float(lsd.mean()), *map(float, np.percentile(lsd[idx].mean(1), [2.5, 97.5]))]
    out["per_bin_mean_r"] = np.nanmean(per, axis=0).tolist()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="name for both caches (A2 'render_<n>', 15-shape '<n2>'): A2NAME:G2NAME,...")
    ap.add_argument("--out", default="outputs/p4_5/FREQ_CURVE_60K.json")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        pub = json.load(open("outputs/p4_5/fig_hf_curve_30K.json"))
        k = [x for x in pub if x.startswith("B1  ")][0]
        got, _ = a2_marks("B1_30K")
        ok = abs(got["200"] - pub[k]["200Hz"][0]) < 1e-12
        print("check A2 200 Hz mark vs fig_hf_curve_30K.json (B1 @30K): {} vs {} -> {}".format(
            got["200"], pub[k]["200Hz"][0], "EXACT" if ok else "MISMATCH"))
        if not ok:
            return 3
    rep = {"method": __doc__.strip().splitlines()[0], "top_of_band": "295-300 Hz, bins 590-600, one-sided",
           "marks_hz": MARKS, "rows": {}}
    hdr = "{:16s} | ".format("arm (off-lattice)") + " ".join("{:>6d}".format(h) for h in MARKS) + " | {:>6s} | LSD 200-300 [CI]".format("200-300r")
    print(hdr)
    for pair in a.names.split(","):
        a2n, g2n = pair.split(":")
        r2, curve = a2_marks(a2n)
        r15 = g15(g2n)
        rep["rows"][a2n] = {"A2_3shape": r2, "G15": {k: v for k, v in r15.items() if k != "per_bin_mean_r"},
                            "A2_per_bin_off": [None if not np.isfinite(x) else float(x) for x in curve],
                            "G15_per_bin_off": r15["per_bin_mean_r"]}
        print("{:16s} | ".format(a2n + " A2") + " ".join("{:+.3f}".format(r2[str(h)]) for h in MARKS) +
              " | {:+.3f} |".format(r2["band_200_300_mean_r"]))
        print("{:16s} | ".format("   15-shape") + " ".join("{:+.3f}".format(r15[str(h)][0]) for h in MARKS) +
              " |        | {:.2f} [{:.2f}, {:.2f}]".format(*r15["lsd_200_300"]))
    json.dump(rep, open(a.out, "w"), indent=1)
    figure(rep, Path(a.out).with_suffix(".png"))
    return 0


def figure(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = np.arange(601) * DF
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), dpi=150, sharey=True)
    for name, r in rep["rows"].items():
        for k, (key, title) in enumerate((("A2_per_bin_off", "A2: 3 frozen dense shapes"),
                                           ("G15_per_bin_off", "15 frozen shapes, dense receivers"))):
            y = np.array([np.nan if v is None else v for v in r[key]], float)
            ys = np.convolve(np.nan_to_num(y), np.ones(11) / 11, "same")
            ax[k].plot(f[50:], ys[50:], lw=2.2 if name.startswith(("B1_", "B4_", "B5_")) else 1.4, label=name)
            ax[k].set_title(title + " -- OFF the training lattice, +-2.5 Hz smoothing", loc="left", fontsize=10)
    for a_ in ax:
        a_.axvspan(295, 300, color="0.9"); a_.grid(alpha=0.25); a_.set_xlim(25, 300); a_.set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("dB spatial Pearson, off-lattice"); ax[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Per-frequency field accuracy to 300 Hz (grey: top of band, 295-300 Hz, one-sided)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, facecolor="white")


if __name__ == "__main__":
    raise SystemExit(main())
