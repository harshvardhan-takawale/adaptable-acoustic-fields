"""P4-5: does the model UNDER-PREDICT high-frequency energy, or get the energy right and the
pattern wrong? Pre-registered as D98 before this was run.

Two quantities per bin, from a `p4_5_eval.py` render cache (no GPU, no re-rendering):

  dE  = 10 log10( mean_rx |P|^2 / mean_rx |T|^2 )     -- energy: is the power there at all?
  k   = std_rx(dB|P|) / std_rx(dB|T|)                  -- contrast: is the spatial variation?

and the outcome, by D98's fixed thresholds over 100-300 Hz:
  A  dE <= -3 dB                 degenerate in ENERGY        -> B1 is the right fix
  B  |dE| < 3 dB and k < 0.7     degenerate in CONTRAST      -> right level, washed-out field
  C  |dE| < 3 dB and k >= 0.7    resolution / conditioning   -> right energy and structure,
                                                                 wrong placement: B2 matters more
Each is computed per shape and then averaged over shapes, so one loud room cannot dominate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DF = 0.5
BANDS = {"0-25": (0, 50), "25-50": (50, 100), "50-100": (100, 200), "100-150": (200, 300),
         "150-200": (300, 400), "200-250": (400, 500), "250-300": (500, 601)}
TEST = (200, 601)                                    # 100-300 Hz, D98's scored band
CTRL = (50, 200)                                     # 25-100 Hz, the control


def per_bin(P, T):
    eP, eT = (np.abs(P) ** 2).mean(0), (np.abs(T) ** 2).mean(0)
    dP, dT = 20 * np.log10(np.maximum(np.abs(P), 1e-30)), 20 * np.log10(np.maximum(np.abs(T), 1e-30))
    return {"dE": 10 * np.log10(eP / eT),
            "mag_ratio_db": 20 * np.log10(np.abs(P).mean(0) / np.abs(T).mean(0)),
            "mean_db_err": (dP - dT).mean(0),
            "k": dP.std(0) / np.maximum(dT.std(0), 1e-12),
            "stdP_db": dP.std(0), "stdT_db": dT.std(0)}


def outcome(dE, k):
    if dE <= -3.0:
        return "A"
    if dE >= 3.0:
        return "over"
    return "B" if k < 0.7 else "C"


def summarise(rooms):
    rows = [per_bin(P, T) for P, T in rooms]
    avg = {q: np.mean([r[q] for r in rows], axis=0) for q in rows[0]}
    out = {"bands": {}}
    for name, (a, b) in BANDS.items():
        out["bands"][name] = {q: float(np.mean(avg[q][a:b])) for q in ("dE", "mag_ratio_db",
                                                                       "mean_db_err", "k")}
    for tag, (a, b) in (("scored_100_300", TEST), ("control_25_100", CTRL)):
        dE, k = float(np.mean(avg["dE"][a:b])), float(np.mean(avg["k"][a:b]))
        out[tag] = {"dE_db": dE, "k": k, "outcome": outcome(dE, k),
                    "per_shape": [{"dE_db": float(np.mean(r["dE"][a:b])),
                                   "k": float(np.mean(r["k"][a:b])),
                                   "outcome": outcome(float(np.mean(r["dE"][a:b])),
                                                      float(np.mean(r["k"][a:b])))}
                                  for r in rows]}
    out["per_bin"] = {q: avg[q].tolist() for q in avg}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/p4_5/eval/cache/render_baseline_60K.npz")
    ap.add_argument("--name", default="baseline_60K")
    ap.add_argument("--out", default="outputs/p4_5/diagnostics")
    a = ap.parse_args()
    z = np.load(a.cache, allow_pickle=False)
    dense = [(z["dP%d" % i], z["dT%d" % i]) for i in range(3)]
    sparse = [(z["sP%d" % i], z["sT%d" % i]) for i in range(15)]
    rep = {"name": a.name, "cache": a.cache, "pre_registration": "D98",
           "dense_3_shapes": summarise(dense), "sparse_15_shapes": summarise(sparse)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out / "energy_{}.json".format(a.name), "w"), indent=1)

    for key in ("dense_3_shapes", "sparse_15_shapes"):
        s = rep[key]
        print("== {} ==".format(key))
        print("  band(Hz)   dE dB   |mean|P|/|T| dB   mean dB err    contrast k")
        for b, v in s["bands"].items():
            print("  {:8s} {:+7.2f}      {:+7.2f}         {:+7.2f}        {:5.3f}".format(
                b, v["dE"], v["mag_ratio_db"], v["mean_db_err"], v["k"]))
        for tag in ("control_25_100", "scored_100_300"):
            t = s[tag]
            print("  {:15s} dE {:+.2f} dB  k {:.3f}  -> OUTCOME {}   per shape: {}".format(
                tag, t["dE_db"], t["k"], t["outcome"],
                ", ".join("{}({:+.1f},{:.2f})".format(p["outcome"], p["dE_db"], p["k"])
                          for p in t["per_shape"])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = np.arange(601) * DF
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), dpi=160, sharex=True)
    for key, ls, lab in (("dense_3_shapes", "-", "dense, 3 shapes"),
                         ("sparse_15_shapes", "--", "sparse 800 rx, 15 shapes")):
        pb = rep[key]["per_bin"]
        ax[0].plot(f, pb["dE"], ls=ls, color="#0072B2", lw=1.4, label=lab)
        ax[1].plot(f, pb["k"], ls=ls, color="#D55E00", lw=1.4, label=lab)
    for v in (-3, 0, 3):
        ax[0].axhline(v, color="0.5", lw=0.8, ls=":" if v else "-")
    ax[1].axhline(0.7, color="0.5", lw=0.8, ls=":")
    ax[1].axhline(1.0, color="0.5", lw=0.8)
    for a_ in ax:
        a_.axvspan(100, 300, color="#FFF2CC", zorder=0)
        a_.grid(alpha=0.25)
        a_.legend(fontsize=9)
    ax[0].set_ylabel("energy ratio dE (dB)\npredicted / FDTD")
    ax[1].set_ylabel("spatial contrast k\nstd(dB|P|) / std(dB|T|)")
    ax[1].set_xlabel("frequency (Hz)")
    s = rep["dense_3_shapes"]["scored_100_300"]
    fig.suptitle("Energy vs pattern, C250 @ 60K (D98). Scored band 100-300 Hz shaded.  Dense: "
                 "dE {:+.2f} dB, k {:.2f} -> outcome {}".format(s["dE_db"], s["k"],
                                                                 s["outcome"]), fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "energy_{}.png".format(a.name), facecolor="white")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
