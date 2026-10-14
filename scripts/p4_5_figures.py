"""P4-5 headline figure: spatial Pearson vs frequency, every arm and the references, one axis.

AFTER D99 THE HEADLINE IS THE OFF-LATTICE CURVE. Every P4 room puts its receivers on one absolute
lattice (0.12 + 0.06 k) and the attention-pooling arms memorised it, so a curve over all dense
receivers mixes the model-as-a-field (~89% of points) with the model-at-memorised-positions
(~11%). The top panel is OFF-lattice only -- the honest measure of a continuous field. The
bottom panel is ON-lattice only, where Gate 2 and every 800-receiver number were measured, so the
gap between the panels is the memorisation, frequency by frequency.

Curves are computed per bin from each evaluation's render cache (`p4_5_eval.py`), per room, then
averaged over the 3 frozen dense test shapes, then smoothed with the same +-2.5 Hz window the
reported marks use. Full 0-300 Hz, nothing cropped, no frequency selected. A3 found ground truth
valid across the whole band, and the figure says so rather than omitting the line silently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.build_p4_5_rooms import test_shapes_for_a2
from scripts.p4_5_eval import DENSE_DIR, _db, _pearson_cols

DF, HALF = 0.5, 5
# (name prefix, label, colour, line width, line style)
CURVES = [("baseline", "C250 baseline (attention)", "0.15", 3.0, "-"),
          ("P4-3_D", "Arm D: mean pooling, 92 shapes (reference)", "#7A1F5C", 2.4, "--"),
          ("P4-4_C92", "C92: attention, same 92 shapes (reference)", "0.55", 1.8, "--"),
          ("B1", "B1  1/RMS spec weights", "#0072B2", 1.6, "-"),
          ("B1b", "B1b gradient-calibrated weights", "#56B4E9", 1.6, "-"),
          ("B2", "B2  coarse-to-fine levels", "#E69F00", 1.6, "-"),
          ("B3", "B3  frequency curriculum", "#009E73", 1.6, "-"),
          ("B4", "B4  4x receivers (same lattice)", "#D55E00", 1.6, "-"),
          ("B5", "B5  DC compliance head", "#CC79A7", 1.6, "-"),
          ("B6", "B6  off-lattice receivers", "#F0E442", 2.0, "-"),
          ("B7", "B7  mean pooling, 250 shapes", "#8B0000", 2.4, "-")]


def _smooth(r):
    out = np.full_like(r, np.nan)
    for i in range(len(r)):
        s = r[max(0, i - HALF):i + HALF + 1]
        s = s[np.isfinite(s)]
        out[i] = s.mean() if len(s) else np.nan
    return out


def _onlat(rx):
    k = (rx - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def split_curves(cache):
    """Per-bin dB spatial Pearson on and off the lattice, averaged over the 3 dense shapes."""
    z = np.load(cache, allow_pickle=False)
    on, off = [], []
    for i, (_tag, c) in enumerate(test_shapes_for_a2()):
        with h5py.File(Path(DENSE_DIR) / c.filename) as f:
            rx = np.asarray(json.loads(f.attrs["receiver_pos"]))
        P, T = z["dP%d" % i], z["dT%d" % i]
        m = _onlat(rx)
        on.append(_pearson_cols(_db(P[m]), _db(T[m])))
        off.append(_pearson_cols(_db(P[~m]), _db(T[~m])))
    return np.nanmean(np.stack(on), 0), np.nanmean(np.stack(off), 0)


def pick(ev, prefix, prefer):
    """`<prefix>_<prefer>` if evaluated, else the latest iteration available -- labelled."""
    want = ev / "cache" / "render_{}_{}.npz".format(prefix, prefer)
    if want.exists():
        return want, prefer
    c = sorted(ev.glob("cache/render_{}_*K.npz".format(prefix)),
               key=lambda p: int(p.stem.split("_")[-1][:-1]))
    return (c[-1], c[-1].stem.split("_")[-1]) if c else (None, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="outputs/p4_5/eval")
    ap.add_argument("--a3", default="outputs/p4_5/diagnostics/a3_dx_convergence.json")
    ap.add_argument("--at", default="60K")
    ap.add_argument("--out", default="outputs/p4_5/fig_hf_curve.png")
    a = ap.parse_args()
    ev = Path(a.eval)
    lim = json.load(open(a.a3)).get("gt_validity_limit_hz") if Path(a.a3).exists() else None
    f = np.arange(601) * DF
    fig, ax = plt.subplots(2, 1, figsize=(15, 12), dpi=170, sharex=True,
                           gridspec_kw={"height_ratios": [3, 2]})
    marks = {}
    for prefix, lab, col, lw, ls in CURVES:
        cache, tag = pick(ev, prefix, a.at)
        if cache is None:
            continue
        on, off = split_curves(cache)
        ax[0].plot(f, _smooth(off), color=col, lw=lw, ls=ls, label="{} @ {}".format(lab, tag))
        ax[1].plot(f, _smooth(on), color=col, lw=lw * 0.8, ls=ls)
        marks[lab + " @ " + tag] = {
            band: (float(np.nanmean(off[b0:b1])), float(np.nanmean(on[b0:b1])))
            for band, (b0, b1) in (("25-100", (50, 200)), ("100-200", (200, 400)),
                                   ("200-300", (400, 601)))}
        # the pre-registered point values (plan, A2): mean over +-2.5 Hz, so no lucky bin is picked
        for hz in (50, 100, 150, 200, 250):
            b = int(round(hz / DF))
            marks[lab + " @ " + tag]["{}Hz".format(hz)] = (float(np.nanmean(off[b - HALF:b + HALF + 1])),
                                                          float(np.nanmean(on[b - HALF:b + HALF + 1])))
    for a_ in ax:
        a_.axvspan(0, 25, color="0.93", zorder=0)
        a_.grid(alpha=0.25)
        a_.set_xlim(0, 300)
        if lim:
            a_.axvline(lim, color="k", ls=":", lw=1.5)
    ax[0].set_ylabel("spatial Pearson (dB), OFF the training lattice\n= the model as a continuous "
                     "field (headline)")
    ax[1].set_ylabel("ON the training lattice\n= where Gate 2 was measured")
    ax[1].set_xlabel("frequency (Hz)")
    gt = ("A3: ground truth valid to {:.0f} Hz (dotted)".format(lim) if lim else
          "A3: FDTD ground truth valid across the whole band (dx 0.02 vs 0.01, r >= 0.99 to 300 Hz)")
    ax[0].set_title("Does the high-frequency tail lift?  3 frozen test shapes, dense 64x64, every "
                    "0.5 Hz bin, +-2.5 Hz smoothing\n" + gt, loc="left", fontsize=11)
    ax[0].legend(fontsize=8.5, loc="upper right", ncol=2)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="white")
    json.dump(marks, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    print("wrote", a.out)
    cols = ["25-100", "100-200", "200-300", "50Hz", "100Hz", "150Hz", "200Hz", "250Hz"]
    print("{:52s} ".format("curve  (off-lattice / on-lattice)") + " ".join("{:>16s}".format(c) for c in cols))
    for k, v in marks.items():
        print("{:52s} ".format(k) + " ".join("{:+.3f} / {:+.3f}".format(*v[c]) for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
