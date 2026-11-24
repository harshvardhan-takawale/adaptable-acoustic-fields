"""P4-10 source-clearance diagnostic (D201, D209): does wall clearance explain why R1's second source trails the third? In the 8 in-family lite rooms, a source on the line y = 1.6 at 0.5, 0.75, 1.0 and 1.5 m from
the east wall (x = L - c) -> 32 (room, clearance) pairs; pairs whose point is outside the polygon, or closer than 0.3 m to a wall, are skipped and logged. For each: dense FDTD truth (the room's own dense receivers), R1 @150K's
render at the moved source; r per band (frozen scorer, off-lattice) vs truth; reported beside the static r at the original source and the gap. Also the clearance histogram of the MS training sources (nearest-wall distance; 4 slots x 1000 rooms).
  --plan                      list the pairs (and the skips) -> configs/p4_10/src_clearance.json
  --fdtd --idx I              CPU: truth of pair I -> outputs/p4_10/srcclear/truth/<key>.npz
  --render --idx I            GPU: R1 render of pair I
  --score                     CPU: outputs/p4_10/srcclear/report.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.p4_10_scorecard import BAND, Renderer, band_r, boot, cells, on_lattice, stored

CLEAR = (0.5, 0.75, 1.0, 1.5)
Y = 1.6
OUT = Path("outputs/p4_10/srcclear")
PLAN = Path("configs/p4_10/src_clearance.json")
CK = "outputs/p4_10/arms/R1/ckpt_iter0150000.pt"
MIN_CLEAR = 0.3


def edge_dist(p, verts):
    v = np.asarray(verts, float)
    a, b = v, np.roll(v, -1, 0)
    ab = b - a
    t = np.clip(((np.asarray(p) - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
    return float(np.linalg.norm(np.asarray(p) - (a + t[:, None] * ab), axis=-1).min())


def plan():
    from aaf.sim.polygon_geom import polygon_contains
    rooms, pairs, skips = {}, [], []
    for c in cells(["Source"], True):
        if c["group"] == "test":
            rooms.setdefault(c["room"], c)
    for r, c in rooms.items():
        V = [tuple(map(float, v)) for v in c["orig_verts"]]
        L, W = c["orig_L"], c["orig_W"]
        for cl in CLEAR:
            p = (round(L - cl, 6), Y)
            ok = bool(np.asarray(polygon_contains(V, np.array([p])))[0]) and edge_dist(p, V) >= MIN_CLEAR
            (pairs if ok else skips).append({"room": r, "clear": cl, "src": p, "edge_dist": edge_dist(p, V), "L": L, "W": W, "verts": V, "filename": c["filename"], "data": c["data"]})
    PLAN.write_text(json.dumps({"pairs": pairs, "skips": skips}, indent=1))
    print("pairs", len(pairs), "skips", [(s["room"][:18], s["clear"]) for s in skips])


def load_pair(i):
    return json.loads(PLAN.read_text())["pairs"][i]


def key(d):
    return "{}_c{:.2f}".format(d["room"], d["clear"])


def fdtd(i):
    from scripts.p4_10_truth import record, snap, solver_room
    d = load_pair(i)
    c = next(x for x in cells(["Source"], True) if x["room"] == d["room"])
    rx = stored(c)["rx"]
    H = record(d["L"], d["W"], d["src"], snap(rx), solver_room(d["L"], d["W"], d["verts"], [0.15] * len(d["verts"])))
    OUT.joinpath("truth").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "truth" / (key(d) + ".npz"), rx=snap(rx), H=H, src=np.array(d["src"]))
    print("truth", key(d), len(rx), "rx", flush=True)


def render(i):
    d = load_pair(i)
    z = np.load(OUT / "truth" / (key(d) + ".npz"))
    P = Renderer("R1_150K", CK).render(d["verts"], d["L"], d["W"], [0.15] * len(d["verts"]), d["src"], z["rx"])
    OUT.joinpath("pred").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "pred" / (key(d) + ".npz"), P=P)
    print("render", key(d), len(P), flush=True)


def score():
    from aaf.data.cx_rooms import edge_distance  # noqa: F401  (kept for provenance: the MS rule's distance)
    import json as j
    p = json.loads(PLAN.read_text())
    rows = []
    for d in p["pairs"]:
        f, g = OUT / "truth" / (key(d) + ".npz"), OUT / "pred" / (key(d) + ".npz")
        if not f.exists() or not g.exists():
            continue
        z, o = np.load(f), np.load(g)
        sel = ~on_lattice(z["rx"])
        c = next(x for x in cells(["Source"], True) if x["room"] == d["room"])
        from scripts.p4_10_scorecard import PRED
        op = np.load(PRED / "R1_150K" / "orig" / (d["room"] + ".npz"))
        base = stored(c)
        so = ~on_lattice(base["rx"])
        stat = band_r(op["P"][so], base["H"][so])
        rows.append({"room": d["room"], "clear": d["clear"], "edge_dist": d["edge_dist"], "r": band_r(o["P"][sel], z["H"][sel]), "static": stat})
    # MS clearance histogram
    m = json.load(open("configs/sweeps_2d_mat/p4_10_ms_manifest.json"))
    verts_of = {r["filename"]: r["verts"] for r in json.load(open("configs/sweeps_2d_mat/p4_7_c500_cx500_manifest.json"))["configs"]}
    d_all, d_slot0 = [], []
    for c in m["configs"]:
        V = [tuple(map(float, v)) for v in verts_of[c["filename"]]]
        d_slot0.append(edge_dist(c["slot0"], V))
        for q in [c["slot0"]] + list(c["new_sources"]):
            d_all.append(edge_dist(q, V))
    d_all = np.array(d_all)
    edges = [0, 0.5, 0.75, 1.0, 1.5, 2.0, 10]
    hist = [int(((d_all >= a) & (d_all < b)).sum()) for a, b in zip(edges[:-1], edges[1:])]
    rep = {"rows": rows, "by_clear": {}, "ms_hist": {"edges": edges, "counts": hist, "n": int(len(d_all)), "slot0_mean": float(np.mean(d_slot0)), "slot0_hist": [int(((np.array(d_slot0) >= a) & (np.array(d_slot0) < b)).sum()) for a, b in zip(edges[:-1], edges[1:])], "mean": float(d_all.mean()), "median": float(np.median(d_all))}}
    L = ["# P4-10 source-clearance diagnostic (R1 @150K, 8 in-family lite rooms, source on y = 1.6 at a set distance from the east wall)", "",
         "| clearance (m) | rooms | r (25-100 / 100-200 / 200-300 Hz) | static r at the original source | gap (static - moved) |", "|---|---|---|---|---|"]
    for cl in CLEAR:
        sel = [r for r in rows if r["clear"] == cl]
        if not sel:
            continue
        e = {"n": len(sel), "r": {b: boot([r["r"][b] for r in sel]) for b in BAND}, "gap": {b: boot([r["static"][b] - r["r"][b] for r in sel]) for b in BAND}, "static": {b: float(np.mean([r["static"][b] for r in sel])) for b in BAND}}
        rep["by_clear"][str(cl)] = e
        f = lambda d, i=0: " / ".join("{:.3f}".format(d[b][i]) for b in BAND)  # noqa: E731
        L.append("| {:.2f} | {} | {} | {} | {} |".format(cl, e["n"], f(e["r"]), " / ".join("{:.3f}".format(e["static"][b]) for b in BAND), " / ".join("{:+.3f} [{:+.3f}, {:+.3f}]".format(*e["gap"][b]) for b in BAND)))
    L += ["", "MS training sources' nearest-wall distance ({} sources): counts per bin {}: {}; mean {:.2f} m, median {:.2f} m. (The MS rule requires >= 0.5 m, >= 1.0 m for at least one of slots 1-3.)".format(
        rep["ms_hist"]["n"], rep["ms_hist"]["edges"], hist, rep["ms_hist"]["mean"], rep["ms_hist"]["median"]), "",
          "Skipped pairs (point outside the polygon or < {} m from a wall): {}".format(MIN_CLEAR, [(s["room"][:20], s["clear"]) for s in p["skips"]])]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--plan", "--fdtd", "--render", "--score"):
        ap.add_argument(k, action="store_true")
    ap.add_argument("--idx", type=int, default=0)
    a = ap.parse_args()
    if a.plan:
        plan()
    elif a.fdtd:
        fdtd(a.idx)
    elif a.render:
        render(a.idx)
    elif a.score:
        score()
