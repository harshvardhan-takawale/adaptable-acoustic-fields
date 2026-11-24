"""P4-10 R1-M verdicts (D195 2c, re-registered by D201 / D202; extension reading D208). Everything is computed from scored JSON; nothing is typed.
  python scripts/p4_10_verdict_r1m.py --name R1M_150K --ref R1_150K --ext-ref R1M_140K  -> outputs/p4_10/verdict_<name>.{json,md}
MM1: EDT edit slope (pooled over the Material cells and the 63 / 125 / 250 Hz octaves, in family, FULL set of 60 cells) in [0.8, 1.2] with Pearson >= 0.8. MM2: edited |EDT error| <= own unedited + 5 points; the absolute
levels and JNDs beside it (5% ISO 3382-1, about 18% measured; D206). MC1: HF and each UF band >= R1 @150K - 0.02 at alpha 0.15. Extension (D208, lite cells): slope >= 0.75 AND Pearson >= 0.75 AND slope rose by >= 0.02
from +20K (baseline 0.53 / 0.77). R1-M extends only if R1 is continued too, and R1 has not been.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.p4_10_scorecard import material_mm

B = ("25-100", "100-200", "200-300")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="R1M_150K")
    ap.add_argument("--ref", default="R1_150K")
    ap.add_argument("--ext-ref", default="R1M_140K")
    ap.add_argument("--lite-mm1", action="store_true", help="dry run on lite cells only")
    a = ap.parse_args()
    rep = {"name": a.name, "ref": a.ref}
    full = {g: material_mm(a.name, a.lite_mm1, g) for g in ("test", "unseen")}
    for g in full:
        e = full[g]["EDT"]
        full[g] = {"slope": e["slope"], "pearson": e["pearson"], "n_cell_bands": e["n"], "bands": {b: (v["EDT"] or {}) for b, v in full[g]["bands"].items()}, "T20": full[g]["T20"],
                   "level_change": full[g]["level_change"], "edt_err_edit": full[g]["edt_err_edit_med"], "edt_err_orig": full[g]["edt_err_orig_med"], "MM2_margin": full[g]["MM2_margin_pts"],
                   "n_cells": full[g]["n_cells"], "n_missing": full[g]["n_missing_decay"], "dropped_cell_bands": full[g]["n_dropped_cells_bands"],
                   "t20_rx_bands_dropped": "{} of {}".format(full[g]["n_t20_receiver_bands_dropped"], full[g]["n_receiver_bands_valid"])}
    rep["MM1"] = {"in_family_full": full["test"], "unseen_full": full["unseen"], "holds": bool(0.8 <= full["test"]["slope"] <= 1.2 and full["test"]["pearson"] >= 0.8)}
    rep["MM2"] = {"edited": full["test"]["edt_err_edit"], "unedited": full["test"]["edt_err_orig"], "margin_pts": 100 * full["test"]["MM2_margin"], "holds": bool(full["test"]["MM2_margin"] <= 0.05),
                  "unseen": {"edited": full["unseen"]["edt_err_edit"], "unedited": full["unseen"]["edt_err_orig"], "margin_pts": 100 * full["unseen"]["MM2_margin"]}}
    from scripts.p4_8_primary_table import load, paired
    x, c = load(a.name), load(a.ref)
    if x is not None and c is not None:
        d = paired(x, c)
        rep["MC1"] = {"HF": {"arm": x["test_r"]["200-300"], "ref": c["test_r"]["200-300"], "delta": d["HF"]}, "UF": {b: {"arm": x["uf"][b], "ref": c["uf"][b], "delta": d["UF_" + b]} for b in B}}
        rep["MC1"]["holds"] = bool(x["test_r"]["200-300"] >= c["test_r"]["200-300"] - 0.02 and all(x["uf"][b] >= c["uf"][b] - 0.02 for b in B))
    else:
        rep["MC1"] = {"pending": True}
    lite_now, lite_before = material_mm(a.name, True, "test")["EDT"], material_mm(a.ext_ref, True, "test")["EDT"]
    rep["extension"] = {"lite_now": {"slope": lite_now["slope"], "pearson": lite_now["pearson"]}, "lite_before": {"slope": lite_before["slope"], "pearson": lite_before["pearson"]},
                        "rise": lite_now["slope"] - lite_before["slope"],
                        "triggered_by_reading": bool(0.75 <= lite_now["slope"] <= 1.25 and lite_now["pearson"] >= 0.75 and (lite_now["slope"] - lite_before["slope"]) >= 0.02)}
    out = Path("outputs/p4_10")
    (out / "verdict_{}.json".format(a.name)).write_text(json.dumps(rep, indent=1))
    L = ["# R1-M verdicts at {} (reference {})".format(a.name, a.ref), "",
         "MM1 (EDT slope in [0.8, 1.2], Pearson >= 0.8; in family, {} cells, {} cell-bands): **{}** -- slope {:.2f}, Pearson {:.2f}".format(
             rep["MM1"]["in_family_full"]["n_cells"], rep["MM1"]["in_family_full"]["n_cell_bands"], "HOLDS" if rep["MM1"]["holds"] else "FAILS", full["test"]["slope"], full["test"]["pearson"])]
    for g, lab in (("test", "in-family"), ("unseen", "unseen")):
        e = full[g]
        L.append("- {}: {} cells; slope per octave {}; level-change slope {}; T20 rx-bands dropped {}; cell-bands dropped {}".format(
            lab, e["n_cells"], " / ".join("{} {:.2f}".format(b, (v or {}).get("slope", float("nan"))) for b, v in e["bands"].items()), "{:.2f}".format(e["level_change"]["slope"]) if e["level_change"] else "n/a", e["t20_rx_bands_dropped"], e["dropped_cell_bands"]))
    m = rep["MM2"]
    L.append("MM2 (edited |EDT error| <= own unedited + 5 points; absolute levels and JNDs printed): **{}** -- in family edited {:.1%} vs unedited {:.1%} (margin {:+.1f} points); unseen {:.1%} vs {:.1%} ({:+.1f}); JNDs 5% (ISO 3382-1), about 18% measured".format(
        "HOLDS" if m["holds"] else "FAILS", m["edited"], m["unedited"], m["margin_pts"], m["unseen"]["edited"], m["unseen"]["unedited"], m["unseen"]["margin_pts"]))
    if "holds" in rep["MC1"]:
        q = rep["MC1"]
        L.append("MC1 (HF and each UF band >= R1 @150K - 0.02): **{}**; HF {:.3f} vs {:.3f} delta {:+.3f} [{:+.3f}, {:+.3f}]; UF {}".format(
            "HOLDS" if q["holds"] else "FAILS", q["HF"]["arm"], q["HF"]["ref"], *q["HF"]["delta"], "; ".join("{} {:.3f} vs {:.3f}".format(b, q["UF"][b]["arm"], q["UF"][b]["ref"]) for b in B)))
    ex = rep["extension"]
    L.append("Extension reading (D208): lite slope {:.2f} Pearson {:.2f} against {:.2f} / {:.2f} at {}; rise {:+.2f} -> {}".format(
        ex["lite_now"]["slope"], ex["lite_now"]["pearson"], ex["lite_before"]["slope"], ex["lite_before"]["pearson"], a.ext_ref, ex["rise"], "MET (manager to confirm before anything is extended)" if ex["triggered_by_reading"] else "not met"))
    (out / "verdict_{}.md".format(a.name)).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
