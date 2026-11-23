"""P4-10: FDTD truth for the edit manifest (D195 0a, D197 A5/A6, D198).

Per cell: the edited room on its own dense lattice (the frozen sets' rule: 64 x 64 over the bounding box, 0.12 m from
walls, > 0.15 m from reflex corners, snapped to the 2 cm nodes) AND the common receivers (the original room's off-lattice
dense receivers inside the edited room, >= 0.1 m from its walls) -> one npz per cell on scratch. Source cells keep the
original receivers; 'second' sources reuse the existing src2 file. Material cells use the per-edge absorption spec (A5,
D199) with the original receivers.
  --gate              parity: (i) the dense rule reproduces each stored room's receivers; (ii) an identity cell re-run
                      reproduces the stored H, error normalised by the BAND's own norm (bins >= 50), 3 rooms
  --idx I --chunk N   build cells I*N .. of the selected families  [--families StretchU+StretchA+Wall+Notch+Source]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

MAN = Path("configs/p4_10/edit_manifest.json")
OUT = Path("/fs/nexus-scratch/htakawal/aaf_p410/truth")
FS, N, C, NB, DX, MAXN = 30720.0, 61440, 343.0, 601, 0.02, 4900
WALL, CORNER, GRID = 0.12, 0.15, 64


def cid(c):
    return hashlib.sha1(c["id"].encode()).hexdigest()[:16]


def on_lattice(rx):
    k = (np.asarray(rx) - 0.12) / 0.06
    return np.all(np.abs(k - np.round(k)) < 1e-6, axis=1)


def dense_points(L, W, verts):
    """the frozen sets' dense receiver rule -> points kept and the 64 x 64 keep mask"""
    from aaf.data.cx_rooms import reflex_vertices
    from aaf.sim.polygon_geom import polygon_contains
    X, Y = np.meshgrid(np.linspace(WALL, L - WALL, GRID), np.linspace(WALL, W - WALL, GRID), indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], -1)
    keep = polygon_contains(verts, P)
    for (x0, y0), (x1, y1) in zip(verts, list(verts[1:]) + [verts[0]]):
        if abs(x1 - x0) < 1e-9:
            keep &= ~((np.abs(P[:, 0] - x0) < WALL) & (P[:, 1] >= min(y0, y1) - WALL) & (P[:, 1] <= max(y0, y1) + WALL))
        else:
            keep &= ~((np.abs(P[:, 1] - y0) < WALL) & (P[:, 0] >= min(x0, x1) - WALL) & (P[:, 0] <= max(x0, x1) + WALL))
    for cx, cy in reflex_vertices(verts):
        keep &= np.hypot(P[:, 0] - cx, P[:, 1] - cy) > CORNER
    return P[keep], keep.reshape(GRID, GRID)


def stored(c):
    with h5py.File(Path(c["data"]) / c["filename"], "r") as f:
        return {"H": np.asarray(f["ism/H_complex"])[:, :NB], "rx": np.asarray(json.loads(f.attrs["receiver_pos"]), float),
                "src": tuple(float(x) for x in json.loads(f.attrs["source_pos"]))}


def solver_room(L, W, verts, edge_alphas):
    """geometry and extra_walls for a polygon; non-0.15 absorption goes through the per-edge spec (A5, D199)."""
    from aaf.sim import fdtd_2d as F
    from aaf.sim.polygon_geom import polygon_contains
    nx, dxx = F._fit_axis(L, DX, "L")
    ny, dxy = F._fit_axis(W, DX, "W")
    assert abs(dxx - DX) < 1e-12 and abs(dxy - DX) < 1e-12, (L, W, dxx, dxy)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    P = np.stack([ii * dxx, jj * dxy], -1)
    inside = polygon_contains(verts, P)
    extra = [{"type": "mask", "solid": ~inside, "alpha": 0.15}] if (~inside).any() else []
    if any(abs(a - 0.15) > 1e-12 for a in edge_alphas):
        extra = extra + [{"type": "edges", "verts": [tuple(v) for v in verts], "alphas": [float(a) for a in edge_alphas]}]
    geom = F.build_geometry(L, W, [0.15] * 4, dx=DX, extra_walls=extra)
    bad = int((geom.air != inside).sum())
    assert bad == 0, bad
    return extra or None


def record(L, W, src, pts, extra):
    """H at node positions `pts` (N x 2, already on nodes), chunked to <= MAXN nodes per run."""
    from aaf.sim import fdtd_2d as F
    H = np.zeros((len(pts), NB), np.complex64)
    for s in range(0, len(pts), MAXN):
        res = F.simulate(L, W, [0.15] * 4, tuple(src), pts[s:s + MAXN], dx=DX, fs=FS, n=N, c=C, extra_walls=extra)
        assert np.abs(np.asarray(res["meta"]["rx_pos_snapped"]) - pts[s:s + MAXN]).max() < 1e-9
        H[s:s + MAXN] = np.asarray(res["H_complex"])[:, :NB].astype(np.complex64)
        del res
    return H


def snap(p):
    return np.round(np.rint(p / DX) * DX, 6)


def rel_band(a, b):
    return float(np.linalg.norm((a - b)[:, 50:]) / np.linalg.norm(b[:, 50:]))


def points(c):
    """(dense receivers, keep mask, common receivers, their indices in the original room's receivers) for a cell"""
    from aaf.data.cx_rooms import edge_distance
    from aaf.sim.polygon_geom import polygon_contains
    base = stored(c)
    V, L, W = [tuple(v) for v in c["verts"]], c["L"], c["W"]
    off = np.nonzero(~on_lattice(base["rx"]))[0]
    if c["family"] in ("Source", "Material"):
        return base["rx"], None, base["rx"][off], off
    d, keep = dense_points(L, W, V)
    p = base["rx"][off]
    ok = polygon_contains(V, p) & (edge_distance(V, p) >= 0.1 - 1e-9)
    return snap(d), keep, p[ok], off[ok]


def build(c):
    dst = OUT / (cid(c) + ".npz")
    if dst.exists():
        return "skip"
    rx, keep, common, cidx = points(c)
    t0 = time.time()
    if c["family"] == "Source" and c["mag"] == "second":
        with h5py.File(Path({"test": "data/track_p4_9_test_src2", "unseen": "data/track_p4_9_unseen_src2"}[c["group"]])
                       / c["filename"], "r") as f:
            H = np.asarray(f["ism/H_complex"])[:, :NB].astype(np.complex64)
            s = tuple(float(x) for x in json.loads(f.attrs["source_pos"]))
            assert np.allclose(s, c["src"], atol=1e-6) and np.allclose(json.loads(f.attrs["receiver_pos"]), rx)
        Hd, Hc = H, H[cidx]
    else:
        extra = solver_room(c["L"], c["W"], [tuple(v) for v in c["verts"]], c["edge_alphas"])
        Hd = record(c["L"], c["W"], c["src"], rx, extra)
        Hc = Hd[cidx] if c["family"] in ("Source", "Material") else record(c["L"], c["W"], c["src"], snap(common), extra)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, id=c["id"], Hd=Hd, rx=rx, keep=(keep if keep is not None else np.zeros((0, 0), bool)),
                        Hc=Hc, common=common, common_idx=cidx, secs=time.time() - t0)
    tmp.rename(dst)
    return "{} dense {} common {} {:.0f}s".format(c["id"], len(rx), len(common), time.time() - t0)


def gate():
    M = json.loads(MAN.read_text())["cells"]
    bad, worst = 0, 0.0
    seen = {}
    for c in M:
        seen.setdefault((c["data"], c["filename"]), c)
    for c in seen.values():
        base = stored(c)
        d, keep = dense_points(c["orig_L"], c["orig_W"], [tuple(v) for v in c["orig_verts"]])
        d = snap(d)
        bad += int(d.shape != base["rx"].shape or not np.allclose(d, base["rx"]))
    print("dense rule vs stored receivers: {} of {} rooms differ".format(bad, len(seen)), flush=True)
    pick = [c for c in M if c["family"] == "Source"][:3:1]
    from itertools import islice
    for c in list(islice(seen.values(), 0, 31, 12)):
        base = stored(c)
        sel = np.linspace(0, len(base["rx"]) - 1, 200).round().astype(int)
        extra = solver_room(c["orig_L"], c["orig_W"], [tuple(v) for v in c["orig_verts"]], [0.15] * len(c["orig_verts"]))
        H = record(c["orig_L"], c["orig_W"], base["src"], base["rx"][sel], extra)
        e = rel_band(H, base["H"][sel])
        worst = max(worst, e)
        print(c["filename"], "identity re-run, rel L2 over 25-300 Hz:", "{:.2e}".format(e), flush=True)
    print("GATE", "PASS" if bad == 0 and worst < 1e-4 else "FAIL", bad, worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--families", default="StretchU+StretchA+Wall+Notch+Source")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    if a.gate:
        gate()
        return
    fam = a.families.split("+")
    M = [c for c in json.loads(MAN.read_text())["cells"] if c["family"] in fam]
    if a.plan:
        print(len(M), "cells ->", (len(M) - 1) // a.chunk + 1, "tasks")
        return
    if a.idx is not None:
        for c in M[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
            print(build(c), flush=True)


if __name__ == "__main__":
    main()
