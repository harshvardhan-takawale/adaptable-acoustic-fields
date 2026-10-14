"""P4-5 (D105, D112(d)): bootstrap CIs for the dense Gate-2 re-score of any checkpoint -- the
off-lattice LEVEL and the on-minus-off lattice GAP.

Method (D105, unchanged): across-test-shape bootstrap, stratified by slab (in-slab: its 6 shapes,
out-of-slab: its 9), percentile 95% CI, 10,000 resamples, seed 20261013, a FRESH generator per
checkpoint so a result does not depend on which other checkpoints were scored in the same call.
The GAP is PAIRED: one resample of shape indices serves both the on- and off-lattice means, so
shape-to-shape difficulty cancels instead of widening the interval.

Verdict rule (D105): a level CI containing 0.80 reads AT THRESHOLD, whichever side the point is.

`--check` reproduces D105's Arm D off-lattice row of BOOTSTRAP_CI.json (scored first there, from a
generator in the same initial state) to the last digit before anything new is reported.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

NB, SEED = 10000, 20261013
G2 = "outputs/p4_5/gate2_dense/{}__track_p4_5_testdense.json"


def per_shape(name, sub):
    ps = json.load(open(G2.format(name)))["subsets"][sub]["per_shape"]
    ins = np.array([r["spatial_pearson"] for r in ps if r["in_holdout"]])
    out = np.array([r["spatial_pearson"] for r in ps if not r["in_holdout"]])
    assert len(ins) == 6 and len(out) == 9, (name, sub, len(ins), len(out))
    return ins, out


def verdict(ci):
    return "PASS" if ci[0] > 0.80 else ("FAIL" if ci[1] < 0.80 else "AT THRESHOLD")


def score(name):
    rng = np.random.default_rng(SEED)
    off_i, off_o = per_shape(name, "off_lattice")
    on_i, on_o = per_shape(name, "on_lattice")
    ii = rng.integers(0, 6, (NB, 6)); io = rng.integers(0, 9, (NB, 9))
    lvl_i, lvl_o = off_i[ii].mean(1), off_o[io].mean(1)
    gap_i, gap_o = (on_i - off_i)[ii].mean(1), (on_o - off_o)[io].mean(1)
    pc = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731
    return {"checkpoint": name,
            "off_in_slab": float(off_i.mean()), "off_in_ci95": pc(lvl_i),
            "off_in_verdict": verdict(pc(lvl_i)), "p_off_in_below_080": float((lvl_i < 0.80).mean()),
            "off_out_slab": float(off_o.mean()), "off_out_ci95": pc(lvl_o),
            "gap_in": float((on_i - off_i).mean()), "gap_in_ci95": pc(gap_i),
            "gap_out": float((on_o - off_o).mean()), "gap_out_ci95": pc(gap_o)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        ref = [r for r in json.load(open("outputs/p4_5/gate2_dense/BOOTSTRAP_CI.json"))["rows"]
               if r["checkpoint"] == "P4-3_D" and r["subset"] == "off_lattice"][0]
        got = score("P4-3_D")
        ok = got["off_in_ci95"] == ref["in_ci95"] and got["off_out_ci95"] == ref["out_ci95"]
        print("check vs BOOTSTRAP_CI.json (P4-3_D off-lattice): {}".format(
            "EXACT" if ok else "MISMATCH {} vs {}".format(got["off_in_ci95"], ref["in_ci95"])))
        if not ok:
            return 3
    rows = [score(n) for n in a.names.split(",")]
    print("{:14s} | {:>7s} {:>17s} {:>13s} | {:>7s} {:>17s} | {:>7s} {:>17s} | {:>7s} {:>17s}".format(
        "checkpoint", "OFF in", "95% CI", "verdict", "OFF out", "95% CI", "GAP in", "95% CI", "GAP out", "95% CI"))
    for r in rows:
        print("{:14s} | {:7.4f} [{:.4f}, {:.4f}] {:>13s} | {:7.4f} [{:.4f}, {:.4f}] | {:+7.4f} [{:+.4f}, {:+.4f}] | "
              "{:+7.4f} [{:+.4f}, {:+.4f}]".format(
                  r["checkpoint"], r["off_in_slab"], *r["off_in_ci95"], r["off_in_verdict"], r["off_out_slab"],
                  *r["off_out_ci95"], r["gap_in"], *r["gap_in_ci95"], r["gap_out"], *r["gap_out_ci95"]))
    if a.out:
        json.dump({"method": "slab-stratified shape bootstrap, 10k, seed {} fresh per checkpoint; gap paired".format(SEED),
                   "rows": rows}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
