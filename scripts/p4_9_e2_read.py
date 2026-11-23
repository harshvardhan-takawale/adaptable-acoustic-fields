"""P4-9 I11: the pre-registered 2b reads for DIR and DIR-G at 150K -- E2 (D181, as fixed by D188), C2 and C3 (D180).

Written before any 150K number exists.
  E2 (primary, D188)  per band, the arm's figure is the better of its three directional readouts (per-ray D, raw-field FD
                      gradient, projected-field spectral gradient) and CTL's bar is the better of its two omni readouts
                      (raw FD, projected); "better" = higher in-family band mean. Paired per-room difference over the 15
                      in-family shapes, bootstrap 10k (seed 20261013). PASS: >= +0.05 with CI lo > 0 in >= 2 of 3 bands.
  E2 (literal)        the same, but CTL's bar is its raw-field FD gradient only. Reported beside; not the decision.
  C2 (no harm)        arm - CTL >= -0.02 on HF and on each UF band (point estimate), and no CI entirely below -0.02.
  C3 (exploratory)    arm - CTL on UF 25-100 Hz >= +0.03, CI lo > 0.
  Unseen directional numbers are reported beside E2 (c2), with the same selection.
Inputs: outputs/p4_9/directional_zero_shot.json (readouts "<ARM>_150K <readout>"); the standard reports via
p4_8_primary_table.load. -> outputs/p4_9/e2_read.{json,md}
  PYTHONPATH=$PWD python scripts/p4_9_e2_read.py --arms DIR_150K,DIRG_150K --ctl CTL_150K
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.p4_8_primary_table import boot, load, paired

B = ("25-100", "100-200", "200-300")
ARM_READOUTS = ("per-ray", "omni-FD", "proj-FD")
CTL_READOUTS = ("omni-FD", "proj-FD")


def _per_room(rows, name, readout, b, group):
    return np.array([r["readouts"]["{} {}".format(name, readout)]["re_" + b] for r in rows if r["group"] == group])


def _best(rows, name, readouts, b):
    return max(readouts, key=lambda rd: _per_room(rows, name, rd, b, "test").mean())


def e2(rows, arm, ctl, literal):
    out = {"bands": {}}
    n_pass = 0
    for b in B:
        ra = _best(rows, arm, ARM_READOUTS, b)
        rc = "omni-FD" if literal else _best(rows, ctl, CTL_READOUTS, b)
        d = boot(_per_room(rows, arm, ra, b, "test") - _per_room(rows, ctl, rc, b, "test"))
        du = boot(_per_room(rows, arm, ra, b, "unseen") - _per_room(rows, ctl, rc, b, "unseen"))
        ok = d[0] >= 0.05 and d[1] > 0
        n_pass += ok
        out["bands"][b] = {"arm_readout": ra, "ctl_readout": rc,
                           "arm_level": float(_per_room(rows, arm, ra, b, "test").mean()),
                           "ctl_level": float(_per_room(rows, ctl, rc, b, "test").mean()),
                           "delta_in_family": d, "delta_unseen": du, "band_passes": bool(ok)}
    out["n_bands_pass"] = int(n_pass)
    out["passes"] = bool(n_pass >= 2)
    return out


def c2_c3(arm, ctl):
    a, c = load(arm), load(ctl)
    d = paired(a, c)
    keys = ["HF"] + ["UF_" + b for b in B]
    c2 = {k: d[k] for k in keys}
    c2_ok = all(d[k][0] >= -0.02 and d[k][2] >= -0.02 for k in keys)
    c3 = d["UF_25-100"]
    return {"C2": {"deltas": c2, "holds": bool(c2_ok)},
            "C3": {"delta": c3, "holds": bool(c3[0] >= 0.03 and c3[1] > 0)},
            "levels": {n: {"HF": x["test_r"]["200-300"], "UF": x["uf"]} for n, x in ((arm, a), (ctl, c))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="DIR_150K,DIRG_150K")
    ap.add_argument("--ctl", default="CTL_150K")
    a = ap.parse_args()
    rows = json.load(open("outputs/p4_9/directional_zero_shot.json"))["rows"]
    have = set(rows[0]["readouts"])
    rep = {"ctl": a.ctl, "arms": {}}
    md = ["# P4-9 I11: DIR / DIR-G at 150K -- E2 (D188), C2, C3", "",
          "Directional r = Re rho vs FDTD grad p / (jk), receivers >= 0.5 m from the source; in-family 15 shapes decide, "
          "unseen 16 reported beside. Paired per-room bootstrap (10k, seed 20261013).", ""]
    for arm in [x for x in a.arms.split(",") if x]:
        if "{} per-ray".format(arm) not in have:
            md += ["**{}**: no directional render yet.".format(arm), ""]
            continue
        r = {"E2": e2(rows, arm, a.ctl, literal=False), "E2_literal": e2(rows, arm, a.ctl, literal=True)}
        r.update(c2_c3(arm, a.ctl))
        rep["arms"][arm] = r
        md += ["## {} vs {}".format(arm, a.ctl), "",
               "| band | arm readout (level) | CTL bar (level) | delta in-family [95% CI] | delta unseen [95% CI] | band |",
               "|---|---|---|---|---|---|"]
        for b in B:
            e = r["E2"]["bands"][b]
            md.append("| {} Hz | {} ({:.3f}) | {} ({:.3f}) | {:+.3f} [{:+.3f}, {:+.3f}] | {:+.3f} [{:+.3f}, {:+.3f}] | {} |".format(
                b, e["arm_readout"], e["arm_level"], e["ctl_readout"], e["ctl_level"], *e["delta_in_family"],
                *e["delta_unseen"], "pass" if e["band_passes"] else "--"))
        lit = r["E2_literal"]
        md += ["", "**E2 (D188, primary): {}** ({} of 3 bands). Literal reading (CTL bar = raw-FD only, labelled): {} "
               "({} of 3).".format("PASSES" if r["E2"]["passes"] else "FAILS", r["E2"]["n_bands_pass"],
                                   "passes" if lit["passes"] else "fails", lit["n_bands_pass"]), ""]
        c2 = r["C2"]["deltas"]
        md += ["**C2 (no harm): {}** -- {}.".format("HOLDS" if r["C2"]["holds"] else "FAILS", "; ".join(
            "{} {:+.3f} [{:+.3f}, {:+.3f}]".format(k, *v) for k, v in c2.items())),
               "**C3 (exploratory): {}** -- UF 25-100 Hz {:+.3f} [{:+.3f}, {:+.3f}].".format(
                   "HOLDS" if r["C3"]["holds"] else "does not hold", *r["C3"]["delta"]),
               "Levels (Rule 12): " + "; ".join("{} HF {:.3f}, UF {}".format(
                   n, v["HF"], " / ".join("{:.3f}".format(v["UF"][b]) for b in B)) for n, v in r["levels"].items()), ""]
    Path("outputs/p4_9/e2_read.json").write_text(json.dumps(rep, indent=1))
    Path("outputs/p4_9/e2_read.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
