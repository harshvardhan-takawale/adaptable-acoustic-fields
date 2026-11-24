"""P4-10 meeting day (D211, D213): shared truth and renders for V1-V5. Display and scoring are separate: the GRID outputs draw maps, the DENSE outputs (the frozen 64 x 64 rule, off-lattice) give every printed number.
  --plan                      configs/p4_10/md_cases.json (frozen case list; sha printed)
  --fdtd --idx I              CPU: case I -> scratch md/truth/<key>.npz : dense H (601 bins), path H (601 bins), grid H at 50 / 100 / 125 Hz on every 2 cm air node
  --render --idx I            GPU: render job I = (case, model) -> scratch md/pred/<model>__<key>.npz : dense, path (601 bins), grid at 50 / 100 / 125 Hz on every second node (4 cm)
  --jobs                      number of render jobs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

PLAN = Path("configs/p4_10/md_cases.json")
ROOT = Path("/fs/nexus-scratch/htakawal/aaf_p410/md")
SHOW = "L5.36_W4.34_d1.88_w1.40"
MED125 = "L6.06_W4.52_d1.08_w2.44"          # median room on R1's third-source r at 100-200 Hz (the showcase ranks 3 of 15 there)
UNSEEN = "C3L6.54_W4.82_NWd0.80w2.82_NEd1.84w1.88_SEd0.60w1.66"   # 16 rooms: the LOWER of the two middle rooms on R1's static r (mean of 3 bands)
CK = {"R0_120K": "outputs/p4_8/arms/R0/ckpt_iter0120000.pt", "R1_150K": "outputs/p4_10/arms/R1/ckpt_iter0150000.pt",
      "R1S_150K": "outputs/p4_10/arms/R1S/ckpt_iter0150000.pt", "R1M_150K": "outputs/p4_10/arms/R1M/ckpt_iter0150000.pt"}
BINS = (100, 200, 250)                      # 50, 100, 125 Hz at 0.5 Hz
SCALES = (0.90, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.10)
STRIP_S = (0.90, 0.95, 1.0, 1.05, 1.10)
POS12 = tuple(i / 11 for i in range(12))
STRIP_P = (0, 2, 4, 7, 9, 11)
A_ALL = (0.10, 0.20, 0.30, 0.40, 0.50)      # 0.15 = the original room
A_LONG = (0.30, 0.40, 0.50)


def rooms():
    from scripts.p4_10_edits import base_rooms
    return {Path(r["filename"]).stem: r for r in base_rooms()}


def man(room, fam, mag):
    M = json.loads(Path("configs/p4_10/edit_manifest.json").read_text())["cells"]
    return next(c for c in M if c["room"] == room and c["family"] == fam and c["mag"] == mag)


def plan():
    from aaf.data.cx_rooms import q
    from aaf.sim.polygon_geom import polygon_contains
    from scripts.p4_10_edits import verts_of
    R = rooms()
    out = []

    def geo(room, s):
        r = R[room]
        L, W = q(r["L"] * s), q(r["W"] * s)
        N = [dict(n, w=q(n["w"] * s), d=q(n["d"] * s)) for n in r["notches"]]
        return L, W, verts_of(L, W, N)

    def add(key, room, L, W, V, alphas, src, models, grid, path, **kw):
        assert bool(polygon_contains(V, np.array([src]))[0]), key
        out.append(dict(key=key, room=room, group=R[room]["group"], filename=R[room]["filename"], data=R[room]["data"], L=L, W=W, verts=[list(v) for v in V], alphas=list(alphas),
                        src=[float(x) for x in src], models=list(models), grid=grid, path=path, **kw))
    S0 = (0.5, 1.6)
    for s in SCALES:                                                         # V1 (R1-S), V4 (path), V5 (grid, R1 at s = 1)
        L, W, V = geo(SHOW, s)
        m = ["R1S_150K"] + (["R1_150K"] if s in (1.0, 1.10) else []) + (["R0_120K", "R1M_150K"] if s == 1.0 else [])
        add("show_s{:.3f}".format(s), SHOW, L, W, V, [0.15] * len(V), S0, m, True, True, s=s)
    for s in STRIP_S:
        L, W, V = geo(UNSEEN, s)
        if s != 1.0 and __import__("scripts.p4_10_edits", fromlist=["valid"]).valid(L, W, [dict(n, w=q(n["w"] * s), d=q(n["d"] * s)) for n in R[UNSEEN]["notches"]], S0, True, R[UNSEEN]["notches"]):
            print("unseen backup scale invalid", s)
            continue
        add("unseen_s{:.3f}".format(s), UNSEEN, L, W, V, [0.15] * len(V), S0, ["R1S_150K"] + (["R1_150K", "R0_120K"] if s == 1.0 else []), True, False, s=s)
    for tag, room, idx in (("show", SHOW, range(12)), ("med", MED125, STRIP_P), ("unseen", UNSEEN, STRIP_P)):   # V2 (R1, R0)
        L, W, V = geo(room, 1.0)
        t3 = np.array(man(room, "Source", "third")["src"])
        for i in idx:
            p = np.array(S0) + (t3 - np.array(S0)) * POS12[i]
            p = np.round(np.rint(p / 0.02) * 0.02, 6)
            if i == 0 and tag != "med":
                continue                                                     # = <tag>_s1.000
            add("{}_p{:02d}".format(tag, i), room, L, W, V, [0.15] * len(V), p, ["R1_150K", "R0_120K"], True, False, pos=i, u=POS12[i])
    L, W, V = geo(SHOW, 1.0)                                                 # V3 (R1, R1-M): decay only, no grid
    lng = man(SHOW, "Material", "long0.50")["edge_alphas"]
    k = int(np.argmax(lng))
    for a in A_ALL:
        add("show_all{:.2f}".format(a), SHOW, L, W, V, [a] * len(V), S0, ["R1_150K", "R1M_150K"], False, False, alpha=a, edit="all")
    for a in A_LONG:
        e = [0.15] * len(V)
        e[k] = a
        add("show_long{:.2f}".format(a), SHOW, L, W, V, e, S0, ["R1_150K", "R1M_150K"], False, False, alpha=a, edit="long", edge=k)
    txt = json.dumps({"cases": out, "bins": BINS, "scales": SCALES, "strip_scales": STRIP_S, "strip_pos": STRIP_P}, indent=1)
    PLAN.write_text(txt)
    print(len(out), "cases", sum(len(c["models"]) for c in out), "render jobs; sha256", hashlib.sha256(txt.encode()).hexdigest()[:16])


def cases():
    return json.loads(PLAN.read_text())["cases"]


def path_points(L):
    """V4: y = 1.0 m, x from 0.5 to L - 0.5, 100 points on 2 cm nodes (>= 0.5 m from every wall, 0.6 m from the source)"""
    x = np.linspace(0.5, L - 0.5, 100)
    return np.round(np.rint(np.stack([x, np.full(100, 1.0)], 1) / 0.02) * 0.02, 6)


def grid_nodes(c, step=1):
    from aaf.sim.polygon_geom import polygon_contains
    nx, ny = int(round(c["L"] / 0.02)) + 1, int(round(c["W"] / 0.02)) + 1
    ii, jj = np.meshgrid(np.arange(0, nx, step), np.arange(0, ny, step), indexing="ij")
    P = np.round(np.stack([ii.ravel(), jj.ravel()], 1) * 0.02, 6)
    ok = np.asarray(polygon_contains([tuple(v) for v in c["verts"]], P))
    return P[ok], np.stack([ii.ravel()[ok], jj.ravel()[ok]], 1)


def dense_rx(c):
    from scripts.p4_10_truth import dense_points, snap
    d, _ = dense_points(c["L"], c["W"], [tuple(v) for v in c["verts"]])
    return snap(d)


def save(dst, **kw):
    import os
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / "{}.{}.tmp.npz".format(dst.stem, os.getpid())
    np.savez_compressed(tmp, **kw)
    os.replace(tmp, dst)


def fdtd(i):
    from scripts.p4_10_truth import record, solver_room
    c = cases()[i]
    dst = ROOT / "truth" / (c["key"] + ".npz")
    if dst.exists():
        print("skip", c["key"])
        return
    t0 = time.time()
    V = [tuple(v) for v in c["verts"]]
    extra = solver_room(c["L"], c["W"], V, c["alphas"])
    rx = dense_rx(c)
    pp = path_points(c["L"]) if c["path"] else np.zeros((0, 2))
    H = record(c["L"], c["W"], c["src"], np.concatenate([rx, pp]), extra)
    out = {"rx": rx, "Hd": H[:len(rx)], "path": pp, "Hp": H[len(rx):]}
    print(c["key"], "dense + path", len(rx), len(pp), "{:.0f}s".format(time.time() - t0), flush=True)
    if c["grid"]:
        P, ij = grid_nodes(c)
        G = []
        for s in range(0, len(P), 4900):
            G.append(record(c["L"], c["W"], c["src"], P[s:s + 4900], extra)[:, list(BINS)])
        out.update(gxy=P.astype(np.float32), gij=ij.astype(np.int16), Hg=np.concatenate(G))
    save(dst, key=c["key"], secs=time.time() - t0, **out)
    print("truth", c["key"], "grid", len(out.get("gxy", [])), "{:.0f}s".format(time.time() - t0), flush=True)


def jobs():
    return [(c, m) for c in cases() for m in c["models"]]


def render(i):
    from scripts.p4_10_scorecard import Renderer
    c, m = jobs()[i]
    dst = ROOT / "pred" / "{}__{}.npz".format(m, c["key"])
    if dst.exists():
        print("skip", dst.name)
        return
    t0 = time.time()
    rd = Renderer(m, CK[m])
    f = lambda pts: rd.render(c["verts"], c["L"], c["W"], c["alphas"], c["src"], pts)  # noqa: E731
    out = {"Pd": f(dense_rx(c))}
    if c["path"]:
        out["Pp"] = f(path_points(c["L"]))
    if c["grid"]:
        P, ij = grid_nodes(c, 2)
        out.update(gxy=P.astype(np.float32), gij=ij.astype(np.int16), Pg=f(P)[:, list(BINS)])
    save(dst, **out)
    print("rendered", dst.name, "{:.0f}s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--plan", "--fdtd", "--render", "--jobs"):
        ap.add_argument(k, action="store_true")
    ap.add_argument("--idx", type=int, default=0)
    a = ap.parse_args()
    if a.plan:
        plan()
    elif a.fdtd:
        fdtd(a.idx)
    elif a.render:
        render(a.idx)
    elif a.jobs:
        print(len(jobs()))
