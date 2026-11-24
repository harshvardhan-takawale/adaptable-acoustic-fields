"""P4-10 R1 verdicts at a final checkpoint (D195 2a, D201 Q27 / Q28): RS1, RS2, RS3, RC1, the extension rule and the Q28 fallback.
Everything is computed from scored JSON; nothing is typed.  python scripts/p4_10_verdict_r1.py --name R1_150K [--ext-ref R1_140K] -> outputs/p4_10/verdict_<name>.{json,md}
RS1 strict: every one of the six in-family gaps (two sources x three bands; room mean of static r minus moved r, off-lattice) <= 0.10.
RS2: unseen moved-source r >= ignore-the-move r + 0.05 in every band at both sources. RS3: swapped render r >= normal r - 0.05 (in-family; outputs/p4_9/recip_gap).
RC1: HF (test 200-300) and each UF band >= CTL @150K - 0.02 (paired deltas and CIs beside). The extension rule (D201): worst of the six <= 0.15 AND mean of the six fell by >= 0.02
from the reference (lite set, the same 8 rooms at both checkpoints). Q28 fallback: lite third-source r below its +15K value by > 0.03 in any band, or HF cost vs CTL > 0.04 -> R0 stays the base.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.p4_10_scorecard import SC, boot, cells, cid

B = ("25-100", "100-200", "200-300")
MAGS = ("second", "third")


def source_rows(name, lite, group):
    out = {m: {} for m in MAGS}
    for c in cells(["Source"], lite):
        f = SC / name / (cid(c) + ".json")
        if c["group"] == group and c["mag"] in MAGS and f.exists():
            r = json.loads(f.read_text())
            out[c["mag"]][c["room"]] = {"gap": [r["orig_r"][b] - r["edited"]["r"][b] for b in B], "r": [r["edited"]["r"][b] for b in B],
                                        "static": [r["orig_r"][b] for b in B], "ign": [r["ignore"]["r"][b] for b in B]}
    return out


def six(name, lite, group="test"):
    s = source_rows(name, lite, group)
    res = {}
    for m in MAGS:
        rooms = sorted(s[m])
        res[m] = {b: boot([s[m][r]["gap"][i] for r in rooms]) for i, b in enumerate(B)}
        res[m]["n_rooms"] = len(rooms)
        res[m]["r"] = [float(np.mean([s[m][r]["r"][i] for r in rooms])) for i in range(3)]
        res[m]["static"] = [float(np.mean([s[m][r]["static"][i] for r in rooms])) for i in range(3)]
        res[m]["ign"] = [float(np.mean([s[m][r]["ign"][i] for r in rooms])) for i in range(3)]
        res[m]["rs2"] = {b: boot([s[m][r]["r"][i] - s[m][r]["ign"][i] for r in rooms]) for i, b in enumerate(B)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="R1_150K")
    ap.add_argument("--ext-ref", default="R1_140K")
    a = ap.parse_args()
    rep = {"name": a.name}
    full_in = six(a.name, False, "test")
    full_un = six(a.name, False, "unseen")
    rep["RS1"] = {"in_family_full": full_in, "holds": bool(all(full_in[m][b][0] <= 0.10 for m in MAGS for b in B))}
    lite = {n: six(n, True, "test") for n in (a.name, a.ext_ref)}
    g = lambda n: [lite[n][m][b][0] for m in MAGS for b in B]  # noqa: E731
    worst, mean, mean0 = max(g(a.name)), float(np.mean(g(a.name))), float(np.mean(g(a.ext_ref)))
    rep["extension"] = {"worst_lite": worst, "mean_lite": mean, "mean_lite_at_ref": mean0, "fell": mean0 - mean,
                        "triggered": bool(worst <= 0.15 and (mean0 - mean) >= 0.02)}
    rep["RS2"] = {"unseen_full": full_un, "holds": bool(all(full_un[m]["rs2"][b][0] >= 0.05 for m in MAGS for b in B))}
    p = Path("outputs/p4_9/recip_gap/{}.json".format(a.name))
    if p.exists():
        s = json.loads(p.read_text())["summary"]
        rep["RS3"] = {g_: {b: {"normal": s[g_]["normal"][b], "swapped": s[g_]["swapped"][b], "ok": bool(s[g_]["swapped"][b] >= s[g_]["normal"][b] - 0.05)} for b in B} for g_ in s}
        rep["RS3"]["holds_in_family"] = bool(all(v["ok"] for v in rep["RS3"]["in_family15"].values()))
    try:
        from scripts.p4_8_primary_table import load, paired
        x, c = load(a.name), load("CTL_150K")
        d = paired(x, c)
        rep["RC1"] = {"HF": {"r1": x["test_r"]["200-300"], "ctl": c["test_r"]["200-300"], "delta": d["HF"]},
                      "UF": {b: {"r1": x["uf"][b], "ctl": c["uf"][b], "delta": d["UF_" + b]} for b in B}}
        rep["RC1"]["holds"] = bool(x["test_r"]["200-300"] >= c["test_r"]["200-300"] - 0.02 and all(x["uf"][b] >= c["uf"][b] - 0.02 for b in B))
        hf_cost = c["test_r"]["200-300"] - x["test_r"]["200-300"]
        rep["Q28"] = {"HF_cost": hf_cost}
    except Exception as e:
        rep["RC1"] = {"pending": repr(e)}
    # Q28 fallback on the lite set
    try:
        l15 = six("R1_135K", True, "test")["third"]["r"]
        now = lite[a.name]["third"]["r"]
        rep.setdefault("Q28", {})["third_r_lite_vs_15K"] = {"now": now, "at_15K": l15, "drop": [float(x - y) for x, y in zip(l15, now)]}
        if "HF_cost" in rep["Q28"]:
            rep["Q28"]["fallback_R0"] = bool(any(d > 0.03 for d in rep["Q28"]["third_r_lite_vs_15K"]["drop"]) or rep["Q28"]["HF_cost"] > 0.04)
    except Exception as e:
        rep.setdefault("Q28", {})["error"] = repr(e)
    out = Path("outputs/p4_10")
    (out / "verdict_{}.json".format(a.name)).write_text(json.dumps(rep, indent=1))
    f = lambda v: "{:.3f} [{:.3f}, {:.3f}]".format(*v)  # noqa: E731
    L = ["# R1 verdicts at " + a.name, "",
         "RS1 (strict, every in-family gap <= 0.10; 15 rooms, off-lattice, paired bootstrap 10k): **{}**".format("HOLDS" if rep["RS1"]["holds"] else "FAILS")]
    for m in MAGS:
        L.append("- {} source gaps (25-100 / 100-200 / 200-300 Hz): {}".format(m, "; ".join(f(full_in[m][b]) for b in B)))
    L.append("Extension rule (lite): worst {:.3f} (<= 0.15?), mean {:.4f} against {:.4f} at {} (fell {:+.4f}, needs >= 0.02) -> {}".format(
        rep["extension"]["worst_lite"], rep["extension"]["mean_lite"], rep["extension"]["mean_lite_at_ref"], a.ext_ref, rep["extension"]["fell"],
        "TRIGGERED" if rep["extension"]["triggered"] else "not triggered"))
    L.append("RS2 (unseen, moved r >= ignore + 0.05 in every band, both sources): **{}**".format("HOLDS" if rep["RS2"]["holds"] else "FAILS"))
    for m in MAGS:
        L.append("- {}: moved r minus ignore: {}".format(m, "; ".join(f(full_un[m]["rs2"][b]) for b in B)))
    if "RS3" in rep:
        L.append("RS3 (swapped >= normal - 0.05, in family): **{}**".format("HOLDS" if rep["RS3"]["holds_in_family"] else "FAILS"))
        for g_ in ("in_family15", "unseen16"):
            if g_ in rep["RS3"]:
                L.append("- {}: normal / swapped {}".format(g_, "; ".join("{:.3f}/{:.3f}".format(rep["RS3"][g_][b]["normal"], rep["RS3"][g_][b]["swapped"]) for b in B)))
    if "holds" in rep.get("RC1", {}):
        L.append("RC1 (HF and each UF band >= CTL @150K - 0.02): **{}**; HF {:.3f} vs {:.3f}, delta {:+.3f} [{:+.3f}, {:+.3f}]; UF {}".format(
            "HOLDS" if rep["RC1"]["holds"] else "FAILS", rep["RC1"]["HF"]["r1"], rep["RC1"]["HF"]["ctl"], *rep["RC1"]["HF"]["delta"],
            "; ".join("{} {:.3f} vs {:.3f}".format(b, rep["RC1"]["UF"][b]["r1"], rep["RC1"]["UF"][b]["ctl"]) for b in B)))
    L.append("Q28 fallback to R0: {}".format(rep["Q28"].get("fallback_R0", "pending")))
    (out / "verdict_{}.md".format(a.name)).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
