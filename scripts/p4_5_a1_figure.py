"""P4-5 A1 figure + the scored verdict against D92, from the two per-state A1 JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DF = 0.5
TERMS = ("L_spec_real", "L_spec_imag", "L_amp", "L_phase")
COL = {"L_spec_real": "#0072B2", "L_spec_imag": "#56B4E9", "L_amp": "#D55E00",
       "L_phase": "#009E73"}


def _to_bands(per_bin, edges):
    per_bin = np.asarray(per_bin)
    out = []
    for lo, hi in edges:
        b0, b1 = int(round(lo / DF)), int(round(hi / DF))
        out.append(per_bin[b0:min(b1, len(per_bin))].sum())
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/p4_5/diagnostics")
    a = ap.parse_args()
    d = Path(a.dir)
    S = {}
    mean_H = None
    for st in ("C250_60K", "fresh_init"):
        p = d / "a1_{}.json".format(st)
        if p.exists():
            j = json.load(open(p))
            S[st] = j["states"][st]
            mean_H = np.asarray(j["mean_abs_H_per_bin"])
            pred = j["prediction"]
    if "C250_60K" not in S:
        raise SystemExit("a1_C250_60K.json missing")
    m = S["C250_60K"]
    edges = m["band_edges_hz"]
    mids = np.array([(lo + hi) / 2 for lo, hi in edges])
    width = 9.0
    f = np.arange(len(mean_H)) * DF

    fig, ax = plt.subplots(4, 1, figsize=(13, 15), dpi=150)
    # (a) the roll-off the hypothesis rests on
    ax[0].semilogy(f, mean_H, color="0.2")
    ax[0].set_ylabel("mean |H| (training corpus)")
    ax[0].set_title("(a) target magnitude: the roll-off (DC compliance term at bin 0)", loc="left")

    # (b) the three definitions of 'share', C250 @ 60K
    lv = _to_bands(m["loss_value_share_total"], edges)
    pj = np.asarray(m["param_proj_share_total"])
    nm = np.asarray(m["param_norm_share_total"])
    ax[1].bar(mids - width / 3, lv, width / 3, color="0.6", label="loss VALUE share")
    ax[1].bar(mids, pj, width / 3, color="#0072B2", label="parameter-gradient PROJECTION share "
                                                          "(scored)")
    ax[1].bar(mids + width / 3, nm, width / 3, color="#E69F00", label="parameter-gradient NORM "
                                                                      "share")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("share per 10 Hz band")
    sc = m["scored"]
    ax[1].set_title("(b) C250 @ 60K -- share at >= 100 Hz: projection {:.3f} +- {:.3f} | norm "
                    "{:.3f} | loss value {:.3f}   (prediction: projection < {:.2f})".format(
                        sc["share_above_100hz_param_proj"], sc["se"],
                        sc["share_above_100hz_param_norm"], sc["share_above_100hz_loss_value"],
                        pred["max_share"]), loc="left", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper right")

    # (c) per term, projection, C250
    bottom_p, bottom_n = np.zeros(len(edges)), np.zeros(len(edges))
    for t in TERMS:
        v = np.asarray(m["param_proj_share"][t])
        pos, neg = np.clip(v, 0, None), np.clip(v, None, 0)
        ax[2].bar(mids, pos, width, bottom=bottom_p, color=COL[t], label=t)
        ax[2].bar(mids, neg, width, bottom=bottom_n, color=COL[t])
        bottom_p += pos
        bottom_n += neg
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_ylabel("projection share")
    ax[2].set_title("(c) C250 @ 60K -- projection share by loss term (negative = pulls against "
                    "the net step)", loc="left")
    ax[2].legend(fontsize=8, ncol=4)

    # (d) fresh init vs trained
    if "fresh_init" in S:
        fi = S["fresh_init"]
        ax[3].bar(mids - width / 4, np.asarray(fi["param_proj_share_total"]), width / 2,
                  color="#CC79A7", label="fresh seed-0 model (iteration 0): share >= 100 Hz "
                                         "{:.3f}".format(
                                             fi["scored"]["share_above_100hz_param_proj"]))
        ax[3].bar(mids + width / 4, pj, width / 2, color="#0072B2",
                  label="C250 @ 60K: share >= 100 Hz {:.3f}".format(
                      sc["share_above_100hz_param_proj"]))
        ax[3].axhline(0, color="k", lw=0.6)
        ax[3].legend(fontsize=9)
    ax[3].set_ylabel("projection share")
    ax[3].set_title("(d) the balance at initialisation vs at convergence", loc="left")
    for a_ in ax:
        a_.axvline(100, color="#D55E00", ls="--", lw=1.2)
        a_.set_xlim(0, 300)
        a_.grid(alpha=0.25)
    ax[3].set_xlabel("frequency (Hz)")
    fig.suptitle("A1 -- where the training gradient comes from (p4_4_C250; D92 pre-registration)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(d / "a1_bin_gradient.png", facecolor="white")

    verdict = {"prediction": pred["text"], "scored_on": "C250 @ 60K, parameter-gradient "
               "projection share, bins >= 100 Hz",
               "measured": sc["share_above_100hz_param_proj"], "se": sc["se"],
               "holds": sc["prediction_holds"],
               "also": {"param_norm_share": sc["share_above_100hz_param_norm"],
                        "loss_value_share": sc["share_above_100hz_loss_value"]},
               "cumulative_proj": m["cum_param_proj_total"],
               "cumulative_norm": m["cum_param_norm_total"],
               "cumulative_loss_value": m["cum_loss_value_total"],
               "projection_closure": m["projection_closure"]}
    if "fresh_init" in S:
        verdict["fresh_init"] = S["fresh_init"]["scored"]
    json.dump(verdict, open(d / "a1_verdict.json", "w"), indent=1)
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
