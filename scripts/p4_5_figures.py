"""P4-5 headline figure: spatial Pearson vs frequency, baseline + every arm, one axis.

Reads whatever `outputs/p4_5/eval/<name>.json` files exist, so it can be re-run as each
evaluation lands; an arm with no JSON yet is simply absent and the legend says which iteration
each curve is. The full 25-300 Hz range is always drawn -- nothing cropped, no frequency selected.
The A3 ground-truth-validity limit is drawn as a vertical line; when A3 found none (valid to
300 Hz), the figure says so rather than omitting the line silently.

Curves are the 3-shape DENSE mean, smoothed with the same +-2.5 Hz window the reported marks
use; the raw per-bin values are in the JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DF = 0.5
HALF = 5
ARMS = [("baseline", "C250 baseline", "0.15", 3.0),
        ("B1", "B1  1/RMS spec weights", "#0072B2", 1.8),
        ("B1b", "B1b gradient-calibrated weights", "#56B4E9", 1.8),
        ("B2", "B2  coarse-to-fine levels", "#E69F00", 1.8),
        ("B3", "B3  frequency curriculum", "#009E73", 1.8),
        ("B4", "B4  4x receivers", "#D55E00", 1.8)]


def _smooth(r):
    r = np.asarray(r, float)
    out = np.full_like(r, np.nan)
    for i in range(len(r)):
        s = r[max(0, i - HALF):i + HALF + 1]
        s = s[np.isfinite(s)]
        out[i] = s.mean() if len(s) else np.nan
    return out


def _latest(ev, arm, prefer):
    """The requested iteration if present, else the latest available, labelled honestly."""
    cands = sorted(ev.glob("{}_*K.json".format(arm)),
                   key=lambda p: int(p.stem.split("_")[-1][:-1]))
    want = ev / "{}_{}.json".format(arm, prefer)
    if want.exists():
        return want
    return cands[-1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="outputs/p4_5/eval")
    ap.add_argument("--a3", default="outputs/p4_5/diagnostics/a3_dx_convergence.json")
    ap.add_argument("--at", default="60K")
    ap.add_argument("--out", default="outputs/p4_5/fig_hf_curve.png")
    a = ap.parse_args()
    ev = Path(a.eval)
    a3 = json.load(open(a.a3)) if Path(a.a3).exists() else {}
    lim = a3.get("gt_validity_limit_hz")

    fig, ax = plt.subplots(2, 1, figsize=(14, 11), dpi=170, sharex=True,
                           gridspec_kw={"height_ratios": [3, 2]})
    f = None
    table = []
    for key, lab, col, lw in ARMS:
        p = _latest(ev, key, "60K") if key != "baseline" else ev / "baseline_60K.json"
        if p is None or not p.exists():
            continue
        j = json.load(open(p))
        f = np.asarray(j["freqs_hz"])
        tag = p.stem.split("_")[-1]
        ax[0].plot(f, _smooth(j["dense_mean_r_db"]), color=col, lw=lw,
                   label="{} @ {}".format(lab, tag))
        ax[1].plot(f, _smooth(j["sparse_mean_r_db"]), color=col, lw=lw * 0.8)
        table.append((lab, tag, j["dense_mean_marks"], j["band_lsd_db"],
                      j["rir_pearson_gate2_style"], j["rir_pearson_modal_20_300"]))
    if f is None:
        raise SystemExit("no evaluation JSONs yet")
    for a_ in ax:
        a_.axvspan(0, 25, color="0.93", zorder=0)
        a_.grid(alpha=0.25)
        if lim:
            a_.axvline(lim, color="k", ls=":", lw=1.5)
        a_.set_xlim(0, 300)
    ax[0].set_ylabel("spatial Pearson (dB)\n3 frozen test shapes, dense 64x64")
    ax[1].set_ylabel("spatial Pearson (dB)\n15 frozen test shapes, 800 rx")
    ax[1].set_xlabel("frequency (Hz)")
    gt = ("A3: ground truth valid to {:.0f} Hz (dotted)".format(lim) if lim else
          "A3: ground truth valid across the whole band (dx 0.02 vs 0.01, r >= 0.99 to 300 Hz)")
    ax[0].set_title("Does the high-frequency tail lift?  Spatial accuracy vs frequency, every "
                    "0.5 Hz bin from 25 to 300 Hz, +-2.5 Hz smoothing\n" + gt, loc="left",
                    fontsize=11)
    ax[0].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="white")
    print("wrote", a.out)
    print("{:34s} {:>4s}  {}   LSD 0-100/100-200/200-300   RIR g2 / modal".format(
        "arm", "iter", "  ".join("{:>6s}".format(k + "Hz") for k in table[0][2])))
    for lab, tag, mk, lsd, g2, mod in table:
        print("{:34s} {:>4s}  {}   {:.2f} / {:.2f} / {:.2f}   {:+.4f} / {:+.4f}".format(
            lab, tag, "  ".join("{:+.4f}".format(v) for v in mk.values()),
            lsd["0-100"], lsd["100-200"], lsd["200-300"], g2, mod))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
