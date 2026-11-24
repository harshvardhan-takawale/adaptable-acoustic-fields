"""P4-10 E-E (D211): does moved-source accuracy follow the local density of training sources? 8 in-family lite rooms x 20 test positions (uniform over the interior, >= 0.5 m from every wall, seeded 20261128, >= 0.3 m apart, off the training-source nodes).
  --plan            configs/p4_10/ee_plan.json
  --fdtd --idx I    CPU: dense truth of position I -> scratch md/ee/truth/<key>.npz
  --render --idx I  GPU: R1 -> scratch md/ee/pred/<key>.npz
  --score           CPU: outputs/p4_10/ee/report.{json,md}: R1's r per band (frozen scorer, off-lattice) against the count of MS training sources within 0.3 / 0.5 / 1.0 m (absolute coordinates, all 1000 rooms x 4 sources), Spearman per band, plus the room as a covariate (within-room Spearman).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PLAN = Path("configs/p4_10/ee_plan.json")
ROOT = Path("/fs/nexus-scratch/htakawal/aaf_p410/ee")
OUT = Path("outputs/p4_10/ee")
CK = "outputs/p4_10/arms/R1/ckpt_iter0150000.pt"
SEED, N = 20261128, 20


def plan():
    from aaf.data.cx_rooms import edge_distance
    from aaf.sim.polygon_geom import polygon_contains
    from scripts.p4_10_scorecard import cells
    rng = np.random.default_rng(SEED)
    rooms, rows = {}, []
    for c in cells(["Source"], True):
        if c["group"] == "test":
            rooms.setdefault(c["room"], c)
    for r, c in sorted(rooms.items()):
        V = [tuple(map(float, v)) for v in c["orig_verts"]]
        pts = []
        while len(pts) < N:
            p = np.round(np.rint(np.array([rng.uniform(0.5, c["orig_L"] - 0.5), rng.uniform(0.5, c["orig_W"] - 0.5)]) / 0.02) * 0.02, 6)
            if not bool(polygon_contains(V, p[None])[0]) or edge_distance(V, p[None])[0] < 0.5 or np.hypot(*(p - [0.5, 1.6])) < 0.3:
                continue
            if any(np.hypot(*(p - q)) < 0.3 for q in pts):
                continue
            pts.append(p)
        rows += [{"room": r, "i": i, "src": [float(x) for x in p], "L": c["orig_L"], "W": c["orig_W"], "verts": [list(v) for v in V], "filename": c["filename"], "data": c["data"]} for i, p in enumerate(pts)]
    PLAN.write_text(json.dumps({"seed": SEED, "rows": rows}, indent=0))
    print(len(rows), "positions")


def key(d):
    return "{}_{:02d}".format(d["room"], d["i"])


def rows():
    return json.loads(PLAN.read_text())["rows"]


def rx_of(d):
    from scripts.p4_10_scorecard import cells, stored
    c = next(x for x in cells(["Source"], True) if x["room"] == d["room"])
    return stored(c)["rx"]


def fdtd(i):
    from scripts.p4_10_md import save
    from scripts.p4_10_truth import record, snap, solver_room
    d = rows()[i]
    dst = ROOT / "truth" / (key(d) + ".npz")
    if dst.exists():
        return
    rx = snap(rx_of(d))
    H = record(d["L"], d["W"], d["src"], rx, solver_room(d["L"], d["W"], [tuple(v) for v in d["verts"]], [0.15] * len(d["verts"])))
    save(dst, rx=rx, H=H, src=np.array(d["src"]))
    print("truth", key(d), flush=True)


def render(i):
    from scripts.p4_10_md import save
    from scripts.p4_10_scorecard import Renderer
    from scripts.p4_10_truth import snap
    rd = None
    rr = rows()
    for j in range(i * 20, min((i + 1) * 20, len(rr))):                # one task = the 20 positions of one room
        d = rr[j]
        dst = ROOT / "pred" / (key(d) + ".npz")
        if dst.exists():
            continue
        rd = rd or Renderer("R1_150K", CK)
        P = rd.render(d["verts"], d["L"], d["W"], [0.15] * len(d["verts"]), d["src"], snap(rx_of(d)))
        save(dst, P=P)
        print("render", key(d), flush=True)


def ms_sources():
    m = json.load(open("configs/sweeps_2d_mat/p4_10_ms_manifest.json"))
    return np.array([q for c in m["configs"] for q in [c["slot0"]] + list(c["new_sources"])], float)


def score():
    from scipy.stats import spearmanr
    from scripts.p4_10_scorecard import BAND, band_r, on_lattice
    S = ms_sources()
    R = []
    for d in rows():
        f, g = ROOT / "truth" / (key(d) + ".npz"), ROOT / "pred" / (key(d) + ".npz")
        if not (f.exists() and g.exists()):
            continue
        z, p = np.load(f), np.load(g)
        s = ~on_lattice(z["rx"])
        dist = np.hypot(*(S - np.array(d["src"])).T)
        R.append({"room": d["room"], "src": d["src"], "r": band_r(p["P"][s], z["H"][s]), "n03": int((dist <= 0.3).sum()), "n05": int((dist <= 0.5).sum()), "n10": int((dist <= 1.0).sum())})
    rep = {"n_positions": len(R), "n_ms_sources": len(S), "rows": R, "spearman": {}, "within_room_mean_spearman": {}}
    for b in BAND:
        y = np.array([x["r"][b] for x in R])
        for k in ("n03", "n05", "n10"):
            x = np.array([q[k] for q in R])
            rep["spearman"]["{}|{}".format(b, k)] = [float(v) for v in spearmanr(x, y)]
            wr = []
            for rm in sorted({q["room"] for q in R}):
                sel = [i for i, q in enumerate(R) if q["room"] == rm]
                if len(set(x[sel])) > 2:
                    wr.append(spearmanr(x[sel], y[sel])[0])
            rep["within_room_mean_spearman"]["{}|{}".format(b, k)] = [float(np.nanmean(wr)), len(wr)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    L = ["# P4-10 E-E: does R1's moved-source accuracy follow training-source density?", "", "{} positions (8 in-family lite rooms x 20), MS training sources {} (1000 rooms x 4, absolute coordinates).".format(len(R), len(S)), "",
         "| band | sources within 0.3 m: Spearman (p) | within 0.5 m | within 1.0 m | mean within-room Spearman (0.3 / 0.5 / 1.0 m) |", "|---|---|---|---|---|"]
    for b in BAND:
        L.append("| {} | {} | {} | {} | {} |".format(b, *["{:+.2f} ({:.3f})".format(*rep["spearman"]["{}|{}".format(b, k)]) for k in ("n03", "n05", "n10")], " / ".join("{:+.2f}".format(rep["within_room_mean_spearman"]["{}|{}".format(b, k)][0]) for k in ("n03", "n05", "n10"))))
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
