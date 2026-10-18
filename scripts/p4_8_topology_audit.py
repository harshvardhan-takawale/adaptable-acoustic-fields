"""P4-8 Step 0a (D174(c)): which topologies occur in each training corpus, read from each room's polygon.

The m2 leak-preflight classifier (`scripts/p4_7_m2_leak_preflight.classify`: 1 cm raster of bbox minus polygon,
solid components labelled by the bbox sides they touch) generalised to counts:
  rect | L (1 corner) | T-like (1 side notch, any side) | DN (2 diagonal corners) | U (2 corners on one wall)
  | LT (1 corner + 1 side notch) | TT (2 side notches) | C3 (3 corners) | other.
Also restates P4-7's "GX +0.110 out of family" (D171(b)) per family. Writes outputs/p4_8/topology_audit.{json,md}.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.p4_7_m2_leak_preflight import classify

CORPORA = {
    "C250": ("configs/sweeps_2d_mat/p4_4_shapes_250.json", "shape"),
    "C500": ("configs/sweeps_2d_mat/p4_4_shapes_500.json", "shape"),
    "CX500": ("configs/sweeps_2d_mat/p4_7_cx500_manifest.json", "rows"),
    "CX1000": ("configs/sweeps_2d_mat/p4_8_cx1000_manifest.json", "rows"),
    "family (FAM)": ("configs/sweeps_2d_mat/p4_4_family_manifest.json", "family"),
    "family holdU (G2)": ("configs/sweeps_2d_mat/p4_6_family_holdU_manifest.json", "family"),
}
OUT = Path("outputs/p4_8")


def label(parts):
    pos = [p[0] for p in parts]
    corners = [p for p in pos if len(p) == 2]
    sides = [p for p in pos if p.startswith("mid")]
    if len(corners) + len(sides) != len(pos):
        return "other"
    c, s = len(corners), len(sides)
    if c == 0 and s == 0:
        return "rect"
    if (c, s) == (1, 0):
        return "L"
    if (c, s) == (0, 1):
        return "T-like"
    if (c, s) == (2, 0):
        a, b = sorted(corners)
        return "U" if (a[0] == b[0] or a[1] == b[1]) else "DN"
    if (c, s) == (1, 1):
        return "LT"
    if (c, s) == (0, 2):
        return "TT"
    if (c, s) == (3, 0):
        return "C3"
    return "other({}c{}s)".format(c, s)


def rooms(path, kind):
    m = json.loads(Path(path).read_text())
    if kind == "rows":
        return [(r.get("split", "train"), r["verts"], r["L"], r["W"]) for r in m["configs"]]
    if kind == "shape":
        from aaf.data.shape_configs import configs_from_rows
    else:
        from aaf.data.multi_notch import configs_from_rows
    return [(c.split, c.verts, c.L, c.W) for c in configs_from_rows(m["configs"])]


def main():
    rep = {"corpora": {}}
    for name, (path, kind) in CORPORA.items():
        if not Path(path).exists():
            continue
        cnt = {}
        for split, v, L, W in rooms(path, kind):
            _cls, parts = classify(v, L, W)
            cnt.setdefault(split, Counter())[label(parts)] += 1
        rep["corpora"][name] = {s: dict(c) for s, c in cnt.items()}
        print(name, {s: dict(c) for s, c in cnt.items()}, flush=True)
    # P4-7's GX "+0.110 out of family" per family (D171(b)), with the in-distribution caveat
    fo = {n: json.load(open("outputs/p4_6/family_oof/{}.json".format(n)))["per_family"] for n in ("GX_60K", "G1_60K")}
    B = ["25-100", "100-200", "200-300"]
    rep["gx_oof_restated"] = {f: {"GX_60K": [fo["GX_60K"][f]["raw"][b] for b in B],
                                  "G1_60K": [fo["G1_60K"][f]["raw"][b] for b in B],
                                  "in_GX_training": f in ("T", "DN", "L", "rect")} for f in ("T", "DN", "U")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "topology_audit.json").write_text(json.dumps(rep, indent=1))
    L = ["# P4-8 Step 0a -- topology audit (D174(c))", "",
         "Classified from each room's polygon (m2 classifier, generalised). Counts per split.", "",
         "| corpus | split | " + " | ".join(["rect", "L", "T-like", "DN", "U", "LT", "TT", "C3"]) + " | other |",
         "|---|---|" + "---|" * 9]
    for name, sp in rep["corpora"].items():
        for s, c in sp.items():
            other = sum(v for k, v in c.items() if k.startswith("other"))
            L.append("| {} | {} | ".format(name, s) + " | ".join(str(c.get(k, 0)) for k in
                     ["rect", "L", "T-like", "DN", "U", "LT", "TT", "C3"]) + " | {} |".format(other))
    L += ["", "**GX's P4-7 '+0.110 out of family' restated per family** (25-100 / 100-200 / 200-300 Hz, off-lattice r):", "",
          "| family | GX @60K | G1 @60K | GX - G1 | in GX's training corpus? |", "|---|---|---|---|---|"]
    for f, d in rep["gx_oof_restated"].items():
        L.append("| {} | {} | {} | {} | {} |".format(f, "/".join("%.3f" % x for x in d["GX_60K"]),
                 "/".join("%.3f" % x for x in d["G1_60K"]),
                 "/".join("%+.3f" % (a - b) for a, b in zip(d["GX_60K"], d["G1_60K"])),
                 "**yes** (CX500 has this topology)" if d["in_GX_training"] else "no (truly unseen)"))
    (OUT / "topology_audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-6:]))


if __name__ == "__main__":
    main()
