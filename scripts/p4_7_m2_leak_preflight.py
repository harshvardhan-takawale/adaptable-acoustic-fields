"""P4-7 m2 (D169): leakage pre-flight for the configuration-diversity arms, BEFORE launch.

The manager's rule: classify the topology of every D4 image of every training room of both arms'
corpora; none may fall in the held-out U family's configuration and parameter ranges, and any element
that leaks is dropped. Topology is read from the IMAGE POLYGON itself (not from the notch metadata the
map was derived from), so a bug in `d4_polygon` cannot hide a leak:

  * rasterise bbox minus polygon at 1 cm cell centres; label the solid components (4-connected);
  * each component is a corner notch (touches two adjacent bbox sides) or a side notch (one side);
  * classes: rect | L (one corner) | T (one side notch) | DN (two diagonal corners) |
    U (two corners on the SAME wall: the held-out family, in any of its 4 orientations) | other.

A U-class image is then checked against the U family's parameter ranges (L, W, each notch's d / w as
drawn by `multi_notch._draw_family`, and the stem rule). The held-out d_hat slab is checked separately
for the transposing elements (`transpose_holdout_leaks`). Writes outputs/p4_7/m2_leak_preflight.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

from aaf.data.augment_2d import D4_ELEMS, TRANSPOSING, d4_dims, d4_polygon, transpose_holdout_leaks
from aaf.sim.polygon_geom import polygon_contains

ARMS = {
    "A-fam": "configs/sweeps_2d_mat/p4_6_family_holdU_manifest.json",
    "A-500": "configs/sweeps_2d_mat/p4_4_shapes_500.json",
}
CELL = 0.01
OUT = Path("outputs/p4_7/m2_leak_preflight.json")


def _configs(manifest):
    m = json.loads(Path(manifest).read_text())
    rows = m["configs"]
    if "notches" in rows[0]:
        from aaf.data.multi_notch import configs_from_rows
    else:
        from aaf.data.shape_configs import configs_from_rows
    return m, rows, configs_from_rows(rows, split="train")


def classify(verts, L, W):
    """(class, [(position, d_extent, w_extent)]) read from the polygon alone."""
    nx, ny = int(round(L / CELL)), int(round(W / CELL))
    xs = (np.arange(nx) + 0.5) * L / nx
    ys = (np.arange(ny) + 0.5) * W / ny
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    solid = ~polygon_contains(verts, np.stack([X, Y], -1))
    lab, n = ndimage.label(solid)
    parts = []
    for k in range(1, n + 1):
        ii, jj = np.nonzero(lab == k)
        touch = {"W": ii.min() == 0, "E": ii.max() == nx - 1, "S": jj.min() == 0, "N": jj.max() == ny - 1}
        t = sorted(s for s, v in touch.items() if v)
        wx = (ii.max() - ii.min() + 1) * L / nx
        wy = (jj.max() - jj.min() + 1) * W / ny
        if len(t) == 2 and t in (["N", "W"], ["E", "N"], ["S", "W"], ["E", "S"]):
            pos = {("N", "W"): "NW", ("E", "N"): "NE", ("S", "W"): "SW", ("E", "S"): "SE"}[tuple(t)]
            d, w = wy, wx                              # depth along y, width along x (corner notch)
        elif len(t) == 1:
            pos = "mid" + t[0]
            d, w = (wy, wx) if t[0] in ("N", "S") else (wx, wy)
        else:
            pos = "?" + "".join(t)
            d, w = wy, wx
        parts.append((pos, round(float(d), 3), round(float(w), 3)))
    pos = sorted(p[0] for p in parts)
    if not pos:
        cls = "rect"
    elif len(pos) == 1:
        cls = "L" if len(pos[0]) == 2 else ("T" if pos[0].startswith("mid") else "other")
    elif len(pos) == 2 and all(len(p) == 2 for p in pos):
        a, b = pos
        same_wall = a[0] == b[0] or a[1] == b[1]       # NW/NE share N, NW/SW share W, ...
        cls = "U" if same_wall else "DN"
    else:
        cls = "other"
    return cls, parts


def in_u_ranges(parts, Lg, Wg, u_rng):
    """Is a U-class image inside the U family's drawn parameter ranges (in the image's own frame,
    where the shared wall is rotated to N for comparison)?"""
    (p1, d1, w1), (p2, d2, w2) = parts
    shared = "N" if (p1[0] == p2[0] == "N") else "S" if (p1[0] == p2[0] == "S") else \
        "W" if (p1[1] == p2[1] == "W") else "E"
    if shared in ("W", "E"):                          # rotate so the shared wall is N
        Lg, Wg = Wg, Lg
        d1, w1, d2, w2 = w1, d1, w2, d2
    ok = (u_rng["L"][0] - 1e-6 <= Lg <= u_rng["L"][1] + 1e-6 and u_rng["W"][0] - 1e-6 <= Wg <= u_rng["W"][1] + 1e-6)
    for d, w in ((d1, w1), (d2, w2)):
        ok &= u_rng["d"][0] - 0.02 <= d <= u_rng["d"][1] + 0.02 and u_rng["w"][0] - 0.02 <= w <= u_rng["w"][1] + 0.02
    ok &= (Lg - w1 - w2) >= 0.8 - 1e-6
    return bool(ok)


def main():
    fam_m = json.loads(Path(ARMS["A-fam"]).read_text())
    U = [r for r in fam_m["configs"] if r["kind"] == "U"]
    u_rng = {"L": (min(r["L"] for r in U), max(r["L"] for r in U)),
             "W": (min(r["W"] for r in U), max(r["W"] for r in U)),
             "d": (min(n["d"] for r in U for n in r["notches"]), max(n["d"] for r in U for n in r["notches"])),
             "w": (min(n["w"] for r in U for n in r["notches"]), max(n["w"] for r in U for n in r["notches"]))}
    rep = {"u_family_ranges": u_rng, "cell_m": CELL, "arms": {}}
    for arm, man in ARMS.items():
        m, rows, cfgs = _configs(man)
        slab = m.get("d_hat_holdout")
        slab_leaks = {g: (len(transpose_holdout_leaks(cfgs, tuple(slab))) if (slab and g in TRANSPOSING) else 0)
                      for g in D4_ELEMS}
        per = {}
        native_mismatch = 0
        for g in D4_ELEMS:
            cnt, u_cls, u_in = Counter(), 0, 0
            for c in cfgs:
                vg, _ag, _ = d4_polygon(g, c.verts, c.edge_alphas, c.L, c.W)
                Lg, Wg = d4_dims(g, c.L, c.W)
                cls, parts = classify(vg, Lg, Wg)
                cnt[cls] += 1
                if g == "e" and cls != {"rect": "rect", "L": "L", "T": "T", "DN": "DN", "U": "U"}.get(
                        getattr(c, "kind", "rect" if not c.verts or len(c.verts) == 4 else "L"), cls):
                    native_mismatch += 1
                if cls == "U":
                    u_cls += 1
                    u_in += in_u_ranges(parts, Lg, Wg, u_rng)
            per[g] = {"classes": dict(cnt), "U_topology": u_cls, "U_in_param_ranges": u_in,
                      "d_hat_slab_leaks": slab_leaks[g]}
            print(arm, g, dict(cnt), "U:", u_cls, "in-range:", u_in, "slab:", slab_leaks[g], flush=True)
        rep["arms"][arm] = {"manifest": man, "n_train": len(cfgs), "d_hat_holdout": slab,
                            "identity_class_mismatches": native_mismatch, "per_element": per,
                            "leaking_elements": [g for g in D4_ELEMS
                                                 if per[g]["U_topology"] or per[g]["d_hat_slab_leaks"]]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
