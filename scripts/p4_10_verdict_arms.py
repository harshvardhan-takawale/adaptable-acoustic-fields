"""P4-10 verdicts for the arms read against R1 (D195 2b / 2c, D201, D202). Everything is computed from scored JSON; nothing is typed.
  python scripts/p4_10_verdict_arms.py --arm R1S --name R1S_150K --ref R1_150K --ext-ref R1S_140K --ext-ref-ref R1_140K  -> outputs/p4_10/verdict_<name>.{json,md}
R1-S: SS1 = paired R1-S - R1 stretch r_delta (Stretch-U and Stretch-A pooled per room, full in-family cells, 15 rooms) >= +0.05 with CI > 0 in >= 2 of 3 bands; SC1 = HF and each UF band >= R1 - 0.02
(read at the same step, with the paired CI); the extension rule (D201 Q27, lite 8 rooms): the second-best band's difference >= 0.00 AND the band-mean difference rose by >= 0.02 from +20K to +30K.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.p4_10_scorecard import SC, boot, cells, cid

B = ("25-100", "100-200", "200-300")


def stretch_rooms(name, lite, group):
    """per-room mean r_delta over the Stretch-U and Stretch-A cells of the group"""
    t = {}
    for fam in ("StretchU", "StretchA"):
        for c in cells([fam], lite):
            f = SC / name / (cid(c) + ".json")
            if c["group"] == group and f.exists():
                t.setdefault(c["room"], []).append([json.loads(f.read_text())["r_delta"][b] for b in B])
    return {r: np.nanmean(np.array(v, float), 0) for r, v in t.items()}


def stretch_delta(name, ref, lite, group):
    a, c = stretch_rooms(name, lite, group), stretch_rooms(ref, lite, group)
    rooms = sorted(set(a) & set(c))
    d = np.array([a[r] - c[r] for r in rooms])
    return {"n_rooms": len(rooms), "level": [float(np.mean([a[r][i] for r in rooms])) for i in range(3)],
            "ref_level": [float(np.mean([c[r][i] for r in rooms])) for i in range(3)], "delta": {b: boot(d[:, i]) for i, b in enumerate(B)}}


def r1s(a):
    rep = {"arm": a.name, "ref": a.ref}
    full = {g: stretch_delta(a.name, a.ref, False, g) for g in ("test", "unseen")}
    rep["SS1"] = {"in_family_full": full["test"], "unseen_full": full["unseen"]}
    n_ok = sum(full["test"]["delta"][b][0] >= 0.05 and full["test"]["delta"][b][1] > 0 for b in B)
    rep["SS1"]["bands_met"], rep["SS1"]["holds"] = int(n_ok), bool(n_ok >= 2)
    from scripts.p4_8_primary_table import load, paired
    x, c = load(a.name), load(a.ref)
    if x is not None and c is not None:
        d = paired(x, c)
        rep["SC1"] = {"HF": {"arm": x["test_r"]["200-300"], "ref": c["test_r"]["200-300"], "delta": d["HF"]},
                      "UF": {b: {"arm": x["uf"][b], "ref": c["uf"][b], "delta": d["UF_" + b]} for b in B}}
        rep["SC1"]["holds"] = bool(x["test_r"]["200-300"] >= c["test_r"]["200-300"] - 0.02 and all(x["uf"][b] >= c["uf"][b] - 0.02 for b in B))
    else:
        rep["SC1"] = {"pending": True}
    now = stretch_delta(a.name, a.ref, True, "test")
    before = stretch_delta(a.ext_ref, a.ext_ref_ref, True, "test")
    dn, db_ = [now["delta"][b][0] for b in B], [before["delta"][b][0] for b in B]
    second = sorted(dn, reverse=True)[1]
    rise = float(np.mean(dn) - np.mean(db_))
    rep["extension"] = {"lite_delta_now": dn, "lite_delta_before": db_, "second_best_now": float(second), "band_mean_now": float(np.mean(dn)),
                        "band_mean_before": float(np.mean(db_)), "rise": rise, "triggered": bool(second >= 0.0 and rise >= 0.02)}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R1S")
    ap.add_argument("--name", default="R1S_150K")
    ap.add_argument("--ref", default="R1_150K")
    ap.add_argument("--ext-ref", default="R1S_140K")
    ap.add_argument("--ext-ref-ref", default="R1_140K")
    a = ap.parse_args()
    rep = r1s(a)
    out = Path("outputs/p4_10")
    (out / "verdict_{}.json".format(a.name)).write_text(json.dumps(rep, indent=1))
    f = lambda v: "{:+.3f} [{:+.3f}, {:+.3f}]".format(*v)  # noqa: E731
    L = ["# R1-S verdicts at {} (reference {})".format(a.name, a.ref), ""]
    for g, lab in (("test", "in-family (SS1; full set)"), ("unseen", "unseen (descriptive)")):
        e = rep["SS1"]["in_family_full" if g == "test" else "unseen_full"]
        L.append("- {}: {} rooms; r_delta {} vs {}; delta {}".format(lab, e["n_rooms"], " / ".join("{:.3f}".format(x) for x in e["level"]),
                                                                         " / ".join("{:.3f}".format(x) for x in e["ref_level"]), " / ".join(f(e["delta"][b]) for b in B)))
    L.append("SS1 (>= +0.05 with CI > 0 in >= 2 of 3 bands): **{}** ({} bands met)".format("HOLDS" if rep["SS1"]["holds"] else "FAILS", rep["SS1"]["bands_met"]))
    if "holds" in rep["SC1"]:
        L.append("SC1 (HF and UF >= R1 - 0.02): **{}**; HF {:.3f} vs {:.3f} {}".format("HOLDS" if rep["SC1"]["holds"] else "FAILS", rep["SC1"]["HF"]["arm"], rep["SC1"]["HF"]["ref"], f(rep["SC1"]["HF"]["delta"])))
    x = rep["extension"]
    L.append("Extension (lite): second-best band {:+.4f} (>= 0.00?), band mean {:+.4f} against {:+.4f} at {} (rise {:+.4f}, needs >= 0.02) -> **{}**".format(
        x["second_best_now"], x["band_mean_now"], x["band_mean_before"], a.ext_ref, x["rise"], "TRIGGERED" if x["triggered"] else "not triggered"))
    (out / "verdict_{}.md".format(a.name)).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
