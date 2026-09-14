"""P4-4 Task 2: the data-scaling curve, and the architecture ablation that anchors it.

Two questions, and the design separates them on purpose:

  * **SCALE** -- C92 / C250 / C500 are the same architecture on strictly NESTED corpora
    (92 subset 250 subset 500) with the 15 test shapes byte-identical across all three. So the
    only thing varying along that row is how much data the model saw.
  * **ARCHITECTURE** -- C92 and P4-3's Arm D are the SAME 92-shape corpus and the same 15 test
    shapes, differing only in pooling (`attn_residual_extent` vs `masked_mean`). One run buys
    both the scaling anchor and the ablation.

A NOTE ON WHICH NUMBERS ARE COMPARABLE, because two of them are not.
`val_lsd_db` is measured on held-out RECEIVERS of the TRAINING corpus, so it differs between
C92, C250 and C500 by construction -- a bigger corpus means a different, generally harder val
set, and `val_max_configs` differs too (30 vs 40). Reading the scaling curve off val LSD would
be reading an artifact. Gate 2 and the sweep are computed on the 15 FROZEN test shapes and the
20 fixed sweep depths, identical for every arm, and those are the numbers the curve uses.
"""
from __future__ import annotations

import json
import os

ARMS = [
    # label,                 run dir,                                    sweep dir,                       corpus
    ("BASE mean_unmasked",  "outputs/p4_2/stage2/p4_2_s2_mean_unmasked", "outputs/p4_3/sweep/baseline_mean_unmasked", 60),
    ("P4-3 X1 attn_resid",  "outputs/p4_3/p4_3_X1_attn_residual",        "outputs/p4_3/sweep/X1_attn_residual",       60),
    ("P4-3 D  masked_mean", "outputs/p4_3/p4_3_D_data",                  "outputs/p4_3/sweep/D_data",                 92),
    ("P4-4 C92  attn+ext",  "outputs/p4_4/p4_4_C92",                     "outputs/p4_4/sweep/C92",                    92),
    ("P4-4 C250 attn+ext",  "outputs/p4_4/p4_4_C250",                    "outputs/p4_4/sweep/C250",                  250),
    ("P4-4 C500 attn+ext",  "outputs/p4_4/p4_4_C500",                    "outputs/p4_4/sweep/C500",                  500),
]
TARGET_R, TARGET_LSD = 0.88, 3.0


def _gate(run):
    for c in (run + "_eval_fixed", run + "_eval"):
        f = os.path.join(c, "GATE2.json")
        if os.path.exists(f):
            return json.load(open(f))
    return None


def _sweep(d):
    f = os.path.join(d, "sweep_metrics.json")
    return json.load(open(f)) if os.path.exists(f) else None


def _valfinal(run):
    f = os.path.join(run, "scalars.json")
    if not os.path.exists(f):
        return float("nan")
    d = json.load(open(f))
    recs = d if isinstance(d, list) else (d.get("val") or [])
    v = [float(r.get("lsd_db", r.get("val_lsd_db", 0))) for r in recs
         if isinstance(r, dict) and ("lsd_db" in r or "val_lsd_db" in r)]
    return v[-1] if v else float("nan")


def main() -> int:
    rows = []
    for lab, run, swd, n in ARMS:
        g, s = _gate(run), _sweep(swd)
        R = [p["spatial_pearson"] for p in s["per_depth"]] if s else []
        rows.append({
            "label": lab, "n_corpus": n, "gate": g, "sweep": s,
            "in_slab": g["in_slab"]["spatial_pearson"] if g else float("nan"),
            "out_slab": g["out_slab"]["spatial_pearson"] if g else float("nan"),
            "lsd": g["in_slab"]["band_lsd_db"] if g else float("nan"),
            "rir": g["in_slab"]["rir"] if g else float("nan"),
            "sigma": g["sigma_summary"]["mean_solid_over_air"] if g else float("nan"),
            "passed": g["passed"] if g else None,
            "u_ampl": (max(R) - min(R)) if R else float("nan"),
            "sweep_mean": (sum(R) / len(R)) if R else float("nan"),
            "val_lsd": _valfinal(run),
        })

    print("=" * 108)
    # The `gate` column is the FROZEN P4-2 criterion (in-slab >= 0.80), carried unchanged so the
    # number stays comparable across P4-2, P4-3 and P4-4. P4-4's own target (0.88) is stricter
    # and is reported separately at the bottom -- conflating the two would let an arm that merely
    # clears the old bar read as having met this chunk's goal.
    print("GATE 2 -- 15 FROZEN test shapes, identical for every arm "
          "(gate column = the frozen 0.80 criterion)")
    print("=" * 108)
    print("{:20s} {:>7s} {:>10s} {:>10s} {:>8s} {:>8s} {:>8s} {:>9s} {:>7s}".format(
        "arm", "corpus", "in-slab R", "out-slab", "LSD dB", "RIR r", "sigma", "sweep mn", "gate"))
    for r in rows:
        if r["gate"] is None:
            print("{:20s} {:>7d}  (pending)".format(r["label"], r["n_corpus"])); continue
        print("{:20s} {:>7d} {:>10.4f} {:>10.4f} {:>8.2f} {:>8.4f} {:>8.4f} {:>9.4f} {:>7s}".format(
            r["label"], r["n_corpus"], r["in_slab"], r["out_slab"], r["lsd"], r["rir"],
            r["sigma"], r["sweep_mean"], "PASS" if r["passed"] else "FAIL"))

    print("\n" + "=" * 108)
    print("THE DATA-SCALING CURVE -- same architecture, strictly nested corpora, same test set")
    print("=" * 108)
    sc = [r for r in rows if r["label"].startswith("P4-4") and r["gate"]]
    if len(sc) >= 2:
        print("{:>8s} {:>11s} {:>10s} {:>10s} {:>11s}".format(
            "corpus", "in-slab R", "LSD dB", "U ampl", "sweep mean"))
        for r in sc:
            print("{:>8d} {:>11.4f} {:>10.2f} {:>10.4f} {:>11.4f}".format(
                r["n_corpus"], r["in_slab"], r["lsd"], r["u_ampl"], r["sweep_mean"]))
        a, b = sc[0], sc[-1]
        dn = b["n_corpus"] / max(a["n_corpus"], 1)
        print("\n  {}x more data: in-slab {:+.4f}, LSD {:+.2f} dB, U amplitude {:+.4f}".format(
            round(dn, 1), b["in_slab"] - a["in_slab"], b["lsd"] - a["lsd"],
            b["u_ampl"] - a["u_ampl"]))
        if len(sc) >= 3:
            d1 = sc[1]["in_slab"] - sc[0]["in_slab"]
            d2 = sc[2]["in_slab"] - sc[1]["in_slab"]
            print("  92->250 {:+.4f}   250->500 {:+.4f}   -> the curve is {}".format(
                d1, d2, "STILL CLIMBING" if d2 > 0.25 * abs(d1) and d2 > 0 else
                        "FLATTENING" if d2 > 0 else "SATURATED / REVERSING"))
    else:
        print("  (pending)")

    print("\n" + "=" * 108)
    print("THE ARCHITECTURE ABLATION -- C92 vs P4-3 D: identical 92-shape corpus, identical")
    print("test set, differing ONLY in pooling")
    print("=" * 108)
    d = next((r for r in rows if r["label"].startswith("P4-3 D")), None)
    c = next((r for r in rows if "C92" in r["label"]), None)
    if d and c and d["gate"] and c["gate"]:
        print("  {:22s} in-slab {:+.4f}  LSD {:5.2f}  sweep mean {:+.4f}  val LSD {:.3f}".format(
            "masked_mean (D)", d["in_slab"], d["lsd"], d["sweep_mean"], d["val_lsd"]))
        print("  {:22s} in-slab {:+.4f}  LSD {:5.2f}  sweep mean {:+.4f}  val LSD {:.3f}".format(
            "attn_residual_extent", c["in_slab"], c["lsd"], c["sweep_mean"], c["val_lsd"]))
        print("  {:22s} in-slab {:+.4f}  LSD {:+5.2f}  sweep mean {:+.4f}".format(
            "DELTA (new - old)", c["in_slab"] - d["in_slab"], c["lsd"] - d["lsd"],
            c["sweep_mean"] - d["sweep_mean"]))
    else:
        print("  (pending)")

    print("\n" + "=" * 108)
    print("TARGETS: in-slab >= {:.2f} and LSD < {:.1f} dB".format(TARGET_R, TARGET_LSD))
    print("=" * 108)
    for r in rows:
        if not r["gate"]:
            continue
        hit_r = r["in_slab"] >= TARGET_R
        hit_l = r["lsd"] < TARGET_LSD
        print("  {:20s} R {:.4f} {}   LSD {:.2f} {}".format(
            r["label"], r["in_slab"], "MET " if hit_r else "miss",
            r["lsd"], "MET" if hit_l else "miss"))
    print("\nNOTE: val_lsd_db is measured on held-out receivers of each arm's OWN training")
    print("corpus, so it is NOT comparable across corpus sizes. Everything above is on the 15")
    print("frozen test shapes / 20 fixed sweep depths, which are identical for every arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
