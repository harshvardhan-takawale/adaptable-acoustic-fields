"""P4-10 R1-SM reads (D211 / D212): R1-SM @STEP against R1-S, R1-M and R1 at the SAME step (lite cells). Descriptive before +30K; the four bars (CS1, CM1, CSrc1, CN1) are judged only on the full set at +30K.
  python scripts/p4_10_sm_read.py --step 125    -> outputs/p4_10/progress/R1SM_<step>K.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.p4_10_board import B, stretch_primary
from scripts.p4_10_scorecard import SC, boot, cells, cid, material_slope


def rd(name, fam, mag=None, g="test"):
    """{room: {band: r}} of the lite cells of a family (edited-field r)"""
    out = {}
    for c in cells([fam], True):
        if c["group"] == g and (mag is None or c["mag"] == mag):
            f = SC / name / (cid(c) + ".json")
            if f.exists():
                d = json.loads(f.read_text())
                out.setdefault(c["room"], []).append([d["edited"]["r"][b] for b in B])
    return {r: np.nanmean(v, 0) for r, v in out.items()}


def paired(a, b):
    rooms = sorted(set(a) & set(b))
    d = np.array([a[r] - b[r] for r in rooms])
    return {"n_rooms": len(rooms), "arm": [float(np.mean([a[r][i] for r in rooms])) for i in range(3)], "ref": [float(np.mean([b[r][i] for r in rooms])) for i in range(3)], "delta": {q: boot(d[:, i]) for i, q in enumerate(B)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=125)
    k = ap.parse_args().step
    n = "R1SM_{}K".format(k)
    rep = {"name": n}
    for g in ("test", "unseen"):
        rep["stretch_vs_R1S|" + g] = stretch_primary(n, "R1S_{}K".format(k)).get(g)
        rep["stretch_vs_R1|" + g] = stretch_primary(n, "R1_{}K".format(k)).get(g)
        for m in ("R1SM", "R1M", "R1"):
            s = material_slope("{}_{}K".format(m, k), True, g)
            rep["material|{}|{}".format(m, g)] = {"slope": s["pooled"].get("slope"), "pearson": s["pooled"].get("pearson"), "n": s["pooled"].get("n")}
    rep["third_source_vs_R1"] = paired(rd(n, "Source", "third"), rd("R1_{}K".format(k), "Source", "third"))
    rep["second_source_vs_R1"] = paired(rd(n, "Source", "second"), rd("R1_{}K".format(k), "Source", "second"))
    rep["material_edited_r"] = {m: (lambda d: [float(np.mean([v[i] for v in d.values()])) for i in range(3)] if d else None)(rd("{}_{}K".format(m, k), "Material")) for m in ("R1SM", "R1M", "R1")}
    out = Path("outputs/p4_10/progress/R1SM_{}K".format(k))
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    f = lambda x: "{:+.3f} [{:+.3f}, {:+.3f}]".format(*x)  # noqa: E731
    t = lambda x: " / ".join("{:.3f}".format(v) for v in x)  # noqa: E731
    L = ["# R1-SM @{}K against its parents at the same step (lite cells, descriptive)".format(k), "", "| what | rooms | R1-SM | reference | difference 25-100 / 100-200 / 200-300 [CI] |", "|---|---|---|---|---|"]
    for g in ("test", "unseen"):
        for ref in ("R1S", "R1"):
            e = rep["stretch_vs_{}|{}".format(ref, g)]
            if e:
                L.append("| stretch r_delta, {} | {} | {} | {} {} | {} |".format(g, e["n_rooms"], t([e["level"][b] for b in B]), ref, t([e["ref_level"][b] for b in B]), " / ".join(f(e["delta"][b]) for b in B)))
    for lab, key in (("third source r, in family", "third_source_vs_R1"), ("second source r, in family", "second_source_vs_R1")):
        e = rep[key]
        L.append("| {} | {} | {} | R1 {} | {} |".format(lab, e["n_rooms"], t(e["arm"]), t(e["ref"]), " / ".join(f(e["delta"][b]) for b in B)))
    for g in ("test", "unseen"):
        sl = {m: ("{:.2f} ({:.2f})".format(rep["material|{}|{}".format(m, g)]["slope"], rep["material|{}|{}".format(m, g)]["pearson"]) if rep["material|{}|{}".format(m, g)]["slope"] is not None else "not scored at this step") for m in ("R1SM", "R1M", "R1")}
        L.append("| material EDT slope (Pearson), {} | - | {} | R1-M {}; R1 {} | - |".format(g, sl["R1SM"], sl["R1M"], sl["R1"]))
    L.append("| material edited r, in family | - | {} | R1-M {}; R1 {} | - |".format(*(t(rep["material_edited_r"][m]) if rep["material_edited_r"][m] else "n/a" for m in ("R1SM", "R1M", "R1"))))
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
