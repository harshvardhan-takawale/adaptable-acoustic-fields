"""P4-10 motion figure M4 (D221): capture then move the source, in the LOWER-median unseen room by R1-M zero-shot static r at 25-100 Hz (seed 0 = first draw).
  --pick          print / store the room (configs/p4_10/m4_room.json)
  --fdtd --idx I  CPU: grid truth at 50 Hz, 0 = original source, 1 = third position -> scratch md/truth/m4_<orig|third>.npz
  --gpu           fit the capture (E-B's code, seed 0) and render the 4 cm grid at 50 Hz: zero-shot original, captured original, zero-shot third, captured then moved -> scratch md/pred/m4.npz
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from scripts.p4_10_md import BINS, ROOT, grid_nodes, save

PICK = Path("configs/p4_10/m4_room.json")
B50 = 100


def pick():
    z = {}
    for f in sorted(glob.glob("/fs/nexus-scratch/htakawal/aaf_p410/eb/score/*.json")):
        d = json.loads(Path(f).read_text())
        z[d["room"]] = float(np.mean([s["static"]["zero"]["25-100"] for s in d["seeds"]]))
    order = sorted(z, key=lambda r: z[r])
    room = order[len(order) // 2 - 1]                              # 16 rooms: the LOWER of the two middle ones
    PICK.write_text(json.dumps({"room": room, "zero_shot_static_r_25-100": z[room], "rank_from_lowest": order.index(room) + 1, "of": len(order), "all": z}, indent=1))
    print(room, z[room], order.index(room) + 1, "of", len(order))


def spec():
    from scripts.p4_10_scorecard import cells
    room = json.loads(PICK.read_text())["room"]
    c = next(x for x in cells() if x["group"] == "unseen" and x["room"] == room and x["family"] == "Source" and x["mag"] == "third")
    return room, c


def geom(c, src):
    return {"L": c["orig_L"], "W": c["orig_W"], "verts": [list(v) for v in c["orig_verts"]], "alphas": [0.15] * len(c["orig_verts"]), "src": list(src)}


def fdtd(i):
    from scripts.p4_10_truth import record, solver_room
    room, c = spec()
    tag, src = ("orig", (0.5, 1.6)) if i == 0 else ("third", tuple(c["src"]))
    g = geom(c, src)
    dst = ROOT / "truth" / "m4_{}.npz".format(tag)
    if dst.exists():
        print("skip", dst)
        return
    V = [tuple(v) for v in g["verts"]]
    extra = solver_room(g["L"], g["W"], V, g["alphas"])
    P, ij = grid_nodes(g, 1)
    G = [record(g["L"], g["W"], g["src"], P[s:s + 4900], extra)[:, [B50]] for s in range(0, len(P), 4900)]
    save(dst, gxy=P.astype(np.float32), gij=ij.astype(np.int16), Hg=np.concatenate(G), src=np.array(g["src"]), room=room)
    print("truth", tag, len(P), flush=True)


def gpu():
    if (ROOT / "pred" / "m4.npz").exists():
        print("m4 exists")
        return
    import scripts.p4_9_grounding as G
    from scripts.p4_10_eb_capture import EBNet, K, stored
    from scripts.p4_10_scorecard import cells
    room, c = spec()
    rr = next(x for x in G.eval_rooms() if x["group"] == "unseen" and Path(x["path"]).stem == room)
    r = G.load(rr["path"])
    base = stored(c)
    sp = G.split(r)
    cal = json.loads(Path("outputs/p4_9/step1/calibration.json").read_text())["chosen"]
    net = EBNet()
    idx = G.draw(r, sp, K, 0)
    zo = net.cond(r["verts"], r["L"], r["W"])
    E0 = net.emb(zo)
    snap = net.snapshot()
    g = geom(c, (0.5, 1.6))
    P, ij = grid_nodes(g, 2)
    off = base["rx"]
    b = [B50]
    out = {"gxy": P.astype(np.float32), "gij": ij.astype(np.int16), "measured": idx}
    for tag, src, dense in (("orig", (0.5, 1.6), off), ("third", tuple(c["src"]), off)):
        out["zero_grid_" + tag] = net.field(zo, r["L"], r["W"], P, src=src)[:, b]
        out["zero_dense_" + tag] = net.field(zo, r["L"], r["W"], dense, src=src)
    Efit = net.fit(zo, r["L"], r["W"], r["rx"][idx], r["T"][idx], "M2", cal["M2"]["lr"], cal["M2"]["lam"], cal["M2"]["steps"], seed=0)
    for tag, src in (("orig", (0.5, 1.6)), ("third", tuple(c["src"]))):
        out["cap_grid_" + tag] = net.field(zo, r["L"], r["W"], P, E=Efit, src=src)[:, b]
        out["cap_dense_" + tag] = net.field(zo, r["L"], r["W"], off, E=Efit, src=src)
    net.restore(snap)
    # KRR on the 32 microphones at the ORIGINAL source (it cannot know the move): dense and grid, 50 Hz
    Kd = G.krr(r["rx"][idx], r["T"][idx], off, cal["M5"]["lam"])
    Kg = G.krr(r["rx"][idx], r["T"][idx], P, cal["M5"]["lam"])
    out.update(krr_dense=Kd.astype(np.complex64)[:, b], krr_grid=Kg.astype(np.complex64)[:, b])
    save(ROOT / "pred" / "m4.npz", **out)
    print("m4 rendered", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--pick", "--fdtd", "--gpu"):
        ap.add_argument(k, action="store_true")
    ap.add_argument("--idx", type=int, default=0)
    a = ap.parse_args()
    if a.pick:
        pick()
    elif a.fdtd:
        fdtd(a.idx)
    elif a.gpu:
        gpu()
