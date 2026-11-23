"""P4-9 X3 (D191): moving the source -- the pre-registered reads, written before any X3 number exists.

Inputs: S-3's per-room JSONs (`scripts/p4_9_src2_eval.py`): outputs/p4_9/src2/<NAME>.json (unseen 16) and
outputs/p4_9/src2_test/<NAME>.json (in-family 15); raw off-lattice r per band; projected beside.
  X3a (descriptive)  G2 @60K vs AFAM @60K (identical but for D4 flips + reciprocity swaps): each one's r at the original
                     and the second source, the drop, and AFAM - G2 at the second source (paired per room).
  X3b                ARM vs CTL @150K at the second source, paired per room, bootstrap 10k (seed 20261013):
                     S1 (primary) in-family 15, 25-100 Hz: >= +0.10, CI lo > 0;  S2 unseen 16, 25-100 Hz: >= +0.05, CI lo > 0;
                     C2 (no harm at the original source, standard report): >= -0.02 on HF and each UF band, no CI
                     entirely below -0.02. Reference levels (Rule 12): each item's original-source r, the 'ignore the
                     move' field (original-source truth scored against second-source truth), m1 where present.
  PYTHONPATH=$PWD python scripts/p4_9_x3_read.py [--arm RCP_150K] -> outputs/p4_9/x3_read.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

B = ("25-100", "100-200", "200-300")
DIRS = {"in_family15": Path("outputs/p4_9/src2_test"), "unseen16": Path("outputs/p4_9/src2")}
SEED, NB = 20261013, 10000


def _boot(v):
    v = np.asarray(v, float)
    idx = np.random.default_rng(SEED).integers(0, len(v), (NB, len(v)))
    m = v[idx].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _rooms(sname, name, v="raw"):
    p = DIRS[sname] / "{}.json".format(name)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {r["room"]: {"s1": r["s1"][v], "s2": r["s2"][v], "ign": r.get("ignore_move", {}).get(v)} for r in d["rooms"]}


def _paired(a, b, key, band):
    common = sorted(set(a) & set(b))
    return _boot([a[r][key][band] - b[r][key][band] for r in common]), len(common)


def levels(sname, name):
    d = _rooms(sname, name)
    if d is None:
        return None
    out = {}
    for b in B:
        s1 = [x["s1"][b] for x in d.values()]
        s2 = [x["s2"][b] for x in d.values()]
        out[b] = {"src1": float(np.mean(s1)), "src2": float(np.mean(s2)),
                  "drop": _boot(np.array(s2) - np.array(s1))}
        ign = [x["ign"][b] for x in d.values() if x["ign"] is not None]
        if ign:
            out[b]["ignore_move"] = float(np.mean(ign))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="RCP_150K")
    ap.add_argument("--ctl", default="CTL_150K")
    a = ap.parse_args()
    rep = {"levels": {}, "X3a": {}, "X3b": {}}
    names = ["R0_120K", "S2_120K", a.ctl, a.arm, "G2_60K", "AFAM_60K", "m1"]
    for s in DIRS:
        rep["levels"][s] = {n: levels(s, n) for n in names if levels(s, n) is not None}
    # X3a
    for s in DIRS:
        g, f = _rooms(s, "G2_60K"), _rooms(s, "AFAM_60K")
        if g is None or f is None:
            continue
        rep["X3a"][s] = {b: {"AFAM_minus_G2_src2": _paired(f, g, "s2", b)[0],
                             "AFAM_minus_G2_src1": _paired(f, g, "s1", b)[0]} for b in B}
    # X3b
    for s, (bar, key) in (("in_family15", (0.10, "S1")), ("unseen16", (0.05, "S2"))):
        x, c = _rooms(s, a.arm), _rooms(s, a.ctl)
        if x is None or c is None:
            continue
        per = {b: {"src2": _paired(x, c, "s2", b)[0], "src1": _paired(x, c, "s1", b)[0]} for b in B}
        d = per["25-100"]["src2"]
        rep["X3b"][key] = {"set": s, "bar": bar, "by_band": per, "holds": bool(d[0] >= bar and d[1] > 0)}
    try:
        from scripts.p4_8_primary_table import load, paired
        xa, xc = load(a.arm), load(a.ctl)
        if xa is not None and xc is not None:
            d = paired(xa, xc)
            keys = ["HF"] + ["UF_" + b for b in B]
            rep["X3b"]["C2"] = {"deltas": {k: d[k] for k in keys},
                                "holds": bool(all(d[k][0] >= -0.02 and d[k][2] >= -0.02 for k in keys)),
                                "levels": {n: {"HF": v["test_r"]["200-300"], "UF": v["uf"]}
                                           for n, v in ((a.arm, xa), (a.ctl, xc))}}
    except Exception as e:                          # the standard report may not exist yet
        rep["X3b"]["C2"] = {"pending": repr(e)}
    Path("outputs/p4_9/x3_read.json").write_text(json.dumps(rep, indent=1))

    L = ["# P4-9 X3 (D191): moving the source", "",
         "Raw off-lattice r per band (S-3's scorer), mean over rooms; original source -> second source; drop = second - "
         "original [95% CI], paired over rooms. 'ignore the move' = the original-source TRUTH scored against the "
         "second-source truth (a field that does not move at all).", ""]
    for s in DIRS:
        L += ["## {}".format(s), "", "| item | 25-100 Hz | 100-200 Hz | 200-300 Hz |", "|---|---|---|---|"]
        for n, lv in rep["levels"][s].items():
            L.append("| {} | {} |".format(n, " | ".join("{:.3f} -> {:.3f}; {:+.3f} [{:+.3f}, {:+.3f}]".format(
                lv[b]["src1"], lv[b]["src2"], *lv[b]["drop"]) for b in B)))
        ign = next((lv for lv in rep["levels"][s].values() if "ignore_move" in lv[B[0]]), None)
        if ign:
            L.append("| ignore the move | {} |".format(" | ".join("{:.3f}".format(ign[b]["ignore_move"]) for b in B)))
        L.append("")
        if s in rep["X3a"]:
            L.append("X3a, AFAM - G2 (paired): " + "; ".join("{} Hz second source {:+.3f} [{:+.3f}, {:+.3f}], original "
                                                           "{:+.3f} [{:+.3f}, {:+.3f}]".format(
                b, *rep["X3a"][s][b]["AFAM_minus_G2_src2"], *rep["X3a"][s][b]["AFAM_minus_G2_src1"]) for b in B))
            L.append("")
    for key in ("S1", "S2"):
        if key in rep["X3b"]:
            e = rep["X3b"][key]
            L.append("**{} ({}, {} - {} at the second source, 25-100 Hz, bar +{:.2f}): {}** -- ".format(
                key, e["set"], a.arm, a.ctl, e["bar"], "HOLDS" if e["holds"] else "does not hold") + "; ".join(
                "{} Hz {:+.3f} [{:+.3f}, {:+.3f}]".format(b, *e["by_band"][b]["src2"]) for b in B))
    c2 = rep["X3b"].get("C2", {})
    if "holds" in c2:
        L.append("**C2 (no harm at the original source): {}** -- ".format("HOLDS" if c2["holds"] else "FAILS") + "; ".join(
            "{} {:+.3f} [{:+.3f}, {:+.3f}]".format(k, *v) for k, v in c2["deltas"].items()))
    Path("outputs/p4_9/x3_read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
