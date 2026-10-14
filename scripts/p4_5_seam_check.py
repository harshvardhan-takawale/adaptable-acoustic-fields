"""P4-5 (D144): does PB3's band split leave a SEAM?

Statistic: mean |dB(P_b+1) - dB(P_b)| over adjacent-bin pairs and off-lattice receivers, inside
each crossfade window (163-183 Hz around the 173 Hz edge, 235-255 Hz around 245 Hz), against the
same statistic over the rest of 100-300 Hz. Ground truth's own ratio is the reference (spectra
have structure; a ratio near GT's means no added seam). The manager's rule: the split costs
something if the seam statistic exceeds 2x the in-band value at either edge (or full-band RIR drops).
15 frozen shapes, dense receivers, off-lattice; from the dense Gate-2 render caches.
"""
import argparse
import json

import numpy as np

from scripts.p4_5_gate2_dense import on_lattice

G2 = "outputs/p4_5/gate2_dense/cache/{}__track_p4_5_testdense.npz"
WIN = {"173Hz": (326, 366), "245Hz": (470, 510)}       # bins: edge +-10 Hz
REST = (200, 601)


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def stat(D, lo, hi):
    return float(np.mean(np.abs(np.diff(D[:, lo:hi + 1], axis=1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", default="outputs/p4_5/diagnostics/seam_check.json")
    a = ap.parse_args()
    rep = {}
    for n in a.names.split(","):
        z = np.load(G2.format(n), allow_pickle=False)
        rows = {"pred": {k: [] for k in list(WIN) + ["rest"]}, "gt": {k: [] for k in list(WIN) + ["rest"]}}
        i = 0
        while "P%d" % i in z.files:
            off = ~on_lattice(z["rx%d" % i])
            for key, H in (("pred", z["P%d" % i]), ("gt", z["T%d" % i])):
                D = db(H[off, :601])
                mask = np.ones(601, bool); mask[:REST[0]] = False
                for w, (lo, hi) in WIN.items():
                    rows[key][w].append(stat(D, lo, hi)); mask[lo:hi + 1] = False
                idx = np.flatnonzero(mask[:-1] & mask[1:])
                rows[key]["rest"].append(float(np.mean(np.abs(D[:, idx + 1] - D[:, idx]))))
            i += 1
        r = {key: {k: float(np.mean(v)) for k, v in d.items()} for key, d in rows.items()}
        for key in ("pred", "gt"):
            for w in WIN:
                r[key]["ratio_" + w] = r[key][w] / r[key]["rest"]
        r["seam_flag_2x"] = bool(any(r["pred"]["ratio_" + w] > 2.0 for w in WIN))
        rep[n] = r
        print("{:10s} pred |dB step|: 173 win {:.3f}, 245 win {:.3f}, rest {:.3f} -> ratios {:.2f} / {:.2f}   (GT ratios {:.2f} / {:.2f})  2x-flag {}".format(
            n, r["pred"]["173Hz"], r["pred"]["245Hz"], r["pred"]["rest"], r["pred"]["ratio_173Hz"], r["pred"]["ratio_245Hz"],
            r["gt"]["ratio_173Hz"], r["gt"]["ratio_245Hz"], r["seam_flag_2x"]))
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
