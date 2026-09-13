"""P4-3: assemble every arm against the baseline, with the pre-registered readouts first.

The chunk pre-registered its interpretation before any arm ran:
  * X raises sigma meaningfully above 1.0 AND flattens the U -> diagnosis confirmed, adopt.
  * S improves fields while sigma rises only because it was forced -> mechanism INJECTED, and
    that must be said, not glossed.
  * D alone flattens the U with sigma ~ 1.0 -> the shallow-notch weakness was a data problem
    and the mechanism question is separable from it.

So sigma and the U amplitude are printed before anything else, and the U amplitude is defined
once here (max - min over the 20 swept depths) rather than eyeballed off a figure.
"""
from __future__ import annotations

import json
import os

ARMS = [
    ("BASELINE mean_unmasked", "outputs/p4_2/stage2/p4_2_s2_mean_unmasked",
     "outputs/p4_3/sweep/baseline_mean_unmasked"),
    ("X1 attn_residual", "outputs/p4_3/p4_3_X1_attn_residual", "outputs/p4_3/sweep/X1_attn_residual"),
    ("X2 attn",          "outputs/p4_3/p4_3_X2_attn",          "outputs/p4_3/sweep/X2_attn"),
    ("S  sigma-sup",     "outputs/p4_3/p4_3_S_sigma",          "outputs/p4_3/sweep/S_sigma"),
    ("D  data(92)",      "outputs/p4_3/p4_3_D_data",           "outputs/p4_3/sweep/D_data"),
]


def _gate(run):
    for cand in (run + "_eval_fixed", run + "_eval",
                 os.path.join(os.path.dirname(run), os.path.basename(run) + "_eval")):
        f = os.path.join(cand, "GATE2.json")
        if os.path.exists(f):
            return json.load(open(f))
    return None


def _sweep(d):
    f = os.path.join(d, "sweep_metrics.json")
    return json.load(open(f)) if os.path.exists(f) else None


def main() -> int:
    print("=" * 104)
    print("P4-3 -- the pre-registered readouts first: SIGMA (mechanism) and the U AMPLITUDE")
    print("=" * 104)
    print("{:24s} {:>10s} {:>10s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "arm", "sigma", "U ampl", "U min", "U max", "in-slab", "out-slab"))
    rows = []
    for name, run, swd in ARMS:
        g, s = _gate(run), _sweep(swd)
        sig = g["sigma_summary"]["mean_solid_over_air"] if g else float("nan")
        if s:
            R = [p["spatial_pearson"] for p in s["per_depth"]]
            amp, lo, hi = max(R) - min(R), min(R), max(R)
            isl, osl = s["in_slab_mean_pearson"], s["out_slab_mean_pearson"]
        else:
            amp = lo = hi = isl = osl = float("nan")
        rows.append((name, g, s, sig, amp))
        print("{:24s} {:>10.4f} {:>10.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f}".format(
            name, sig, amp, lo, hi, isl, osl))

    print("\n" + "=" * 104)
    print("GATE 2 metrics (15 frozen test shapes, identical for every arm)")
    print("=" * 104)
    print("{:24s} {:>11s} {:>11s} {:>9s} {:>9s} {:>10s} {:>7s}".format(
        "arm", "in-slab R", "out-slab R", "LSD dB", "RIR r", "NLOS def", "gate"))
    for name, g, s, sig, amp in rows:
        if not g:
            print("{:24s} {:>11s}".format(name, "(pending)")); continue
        print("{:24s} {:>11.4f} {:>11.4f} {:>9.2f} {:>9.4f} {:>10.4f} {:>7s}".format(
            name, g["in_slab"]["spatial_pearson"], g["out_slab"]["spatial_pearson"],
            g["in_slab"]["band_lsd_db"], g["in_slab"]["rir"],
            g["criteria"]["nlos_deficit"]["value"], "PASS" if g["passed"] else "FAIL"))

    base = rows[0]
    print("\n" + "=" * 104)
    print("DELTAS vs baseline  (positive in-slab = better; negative U ampl = FLATTER, the goal)")
    print("=" * 104)
    for name, g, s, sig, amp in rows[1:]:
        if not (g and base[1] and s and base[2]):
            print("  {:22s} (pending)".format(name)); continue
        print("  {:22s} in-slab {:+.4f}   U ampl {:+.4f}   sigma {:+.4f}   LSD {:+.2f}".format(
            name, g["in_slab"]["spatial_pearson"] - base[1]["in_slab"]["spatial_pearson"],
            amp - base[4], sig - base[3],
            g["in_slab"]["band_lsd_db"] - base[1]["in_slab"]["band_lsd_db"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
