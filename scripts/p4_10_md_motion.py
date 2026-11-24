"""P4-10 meeting day, motion figures (D221): the extra renders and the walk truth. Display / scoring rules as D213.
  --jobs                    number of render jobs; --idx I --n N: jobs I, I+N, ... (so a few array tasks share them)
  --fdtd-walk               CPU: truth of V4's walk with the source at the third position -> scratch md/truth/walk_p11.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from scripts.p4_10_md import CK, ROOT, SHOW, BINS, STRIP_P, cases, dense_rx, grid_nodes, path_points, save

CASE = {c["key"]: c for c in cases()}
CK = dict(CK, RCP_150K="outputs/p4_9/arms/RCP/ckpt_iter0150000.pt")
S0 = (0.5, 1.6)


def jobs():
    out = [("normal", "RCP_150K", "show_s1.000" if i == 0 else "show_p{:02d}".format(i)) for i in STRIP_P]
    out += [("swap", m, "show_s1.000") for m in ("R0_120K", "R1_150K")]
    out += [("path", m, "show_p11") for m in ("R1_150K", "R0_120K")]
    return out


def render_swapped(rd, c, pts):
    """source at each pts[i], listener fixed at the original source; same output as the normal map at pts"""
    import os

    from aaf.models.conditioning_2d import build_cond_vector_2d
    t = rd.torch
    chunk = int(os.environ.get("P410_CHUNK", "16"))
    V = [tuple(map(float, v)) for v in c["verts"]]
    z = build_cond_vector_2d("geom_token", c["L"], c["W"], list(c["alphas"]), verts=V, device=rd.dev)
    lo, hi = t.zeros(2, device=rd.dev), t.tensor([c["L"], c["W"]], device=rd.dev, dtype=t.float32)
    P = t.tensor(np.asarray(pts), dtype=t.float32, device=rd.dev)
    L0 = t.tensor([list(S0)], device=rd.dev, dtype=t.float32)
    out = []
    with t.no_grad():
        for s0 in range(0, len(P), chunk):
            s = P[s0:s0 + chunk]
            out.append(rd.renderer(rd.model, L0.expand(len(s), -1), s, lo, hi, z_s=z.unsqueeze(0).expand(len(s), -1), n_out=601).cpu().numpy())
    return np.concatenate(out).astype(np.complex64)


def run(j):
    from scripts.p4_10_scorecard import Renderer
    kind, m, key = jobs()[j]
    c = CASE[key]
    dst = ROOT / "pred" / {"normal": "{}__{}.npz", "swap": "{}__swap_{}.npz", "path": "{}__path_{}.npz"}[kind].format(m, key)
    if dst.exists():
        print("skip", dst.name, flush=True)
        return
    t0 = time.time()
    rd = Renderer(m, CK[m])
    if kind == "normal":
        f = lambda pts: rd.render(c["verts"], c["L"], c["W"], c["alphas"], c["src"], pts)  # noqa: E731
        P, ij = grid_nodes(c, 2)
        save(dst, Pd=f(dense_rx(c)), gxy=P.astype(np.float32), gij=ij.astype(np.int16), Pg=f(P)[:, list(BINS)])
    elif kind == "swap":
        P, ij = grid_nodes(c, 2)
        save(dst, Pd=render_swapped(rd, c, dense_rx(c)), gxy=P.astype(np.float32), gij=ij.astype(np.int16), Pg=render_swapped(rd, c, P)[:, list(BINS)])
    else:
        save(dst, Pp=rd.render(c["verts"], c["L"], c["W"], c["alphas"], c["src"], path_points(c["L"])))
    print("rendered", dst.name, "{:.0f}s".format(time.time() - t0), flush=True)


def fdtd_walk():
    from scripts.p4_10_truth import record, solver_room
    c = CASE["show_p11"]
    dst = ROOT / "truth" / "walk_p11.npz"
    if dst.exists():
        return
    pp = path_points(c["L"])
    H = record(c["L"], c["W"], c["src"], pp, solver_room(c["L"], c["W"], [tuple(v) for v in c["verts"]], c["alphas"]))
    save(dst, path=pp, Hp=H)
    print("walk truth", len(pp), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--jobs", "--fdtd-walk"):
        ap.add_argument(k, action="store_true")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    if a.jobs:
        print(len(jobs()))
    elif a.fdtd_walk:
        fdtd_walk()
    else:
        for j in range(a.idx, len(jobs()), a.n):
            run(j)
