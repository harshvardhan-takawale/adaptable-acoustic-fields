"""P4-9 modal solves (D180 0d, Step 1 M3/M4, 2c S-a; D181 A2, A7): m1's solver (`scripts/p4_7_m1_modal.py`, imported,
not re-implemented) on the 31 dense rooms, with EXACT and APPROXIMATE geometry.

Variants (one SLURM array task per (room, variant)):
  exact     the room's own polygon (the FDTD's)
  snap10    every vertex snapped to the 10 cm grid (D180 0d(ii)); collinear / repeated points merged
  bbox      the bounding rectangle only (0d(iii))
  notch_rm  the smallest notch (area d x w) removed -- the 16 unseen rooms only (0d(iv), D181 A7: for the 15 L-shaped
            test rooms it IS the bounding rectangle)
Per variant: m1's operator on the variant polygon (the solver's discrete Laplacian, image-node Neumann walls, mu-weighted
symmetrisation), every eigenpair to 600 Hz (m1's primary set), eta_n at the corpus beta (alpha 0.15; beta scales eta
linearly, D181 A2), psi at the room's dense receivers (nearest air node of the variant grid; the snap distance is
recorded), at the source, and -- unseen rooms -- at the S-3 second source; the zero mode's exact discrete record td0
(m1's `formula_zero_exact` ingredient). Eigen-solve wall time is recorded (0a: the 600 Hz solve).
  python scripts/p4_9_modal.py --list              # the (room, variant) task list
  python scripts/p4_9_modal.py --idx I             # one task -> outputs/p4_9/modal/<room>__<variant>.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from scripts.p4_7_m1_modal import F_PRIMARY, eigen, operator, pulse, td_synthesis

OUT = Path("outputs/p4_9/modal")
GOUT = Path("data/track_p4_9_modal_grid")      # large: mode shapes on every air node (gitignored data/)
GRID = False
UNSEEN_MAN = "configs/sweeps_2d_mat/p4_8_unseen_manifest.json"
VARIANTS = ("exact", "snap10", "bbox", "notch_rm")


def tasks():
    from scripts.p4_9_fdtd_stencils import dense_rooms
    out = []
    for k, (group, fam, path) in enumerate(dense_rooms()):
        for v in VARIANTS:
            if v == "notch_rm" and group != "unseen":
                continue
            out.append((k, group, fam, path, v))
    return out


def _clean(verts, tol=1e-9):
    """Drop repeated points and collinear middle points (snapping can create both)."""
    v = [tuple(p) for p in verts]
    changed = True
    while changed and len(v) > 3:
        changed = False
        n = len(v)
        for i in range(n):
            a, b, c = v[i - 1], v[i], v[(i + 1) % n]
            same = abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol
            col = abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) < tol
            if same or col:
                v.pop(i)
                changed = True
                break
    return v


def variant_verts(verts, v, group, fname):
    L, W = max(p[0] for p in verts), max(p[1] for p in verts)
    if v == "exact":
        return list(verts)
    if v == "bbox":
        return [(0.0, 0.0), (L, 0.0), (L, W), (0.0, W)]
    if v == "snap10":
        return _clean([(round(x / 0.1) * 0.1, round(y / 0.1) * 0.1) for x, y in verts])
    if v == "notch_rm":
        from aaf.data.multi_notch import Notch, verts_for_notches
        row = next(r for r in json.load(open(UNSEEN_MAN))["configs"] if r["filename"] == fname)
        ns = [Notch(n["side"], n["d"], n["w"], x0=n.get("x0", 0.0)) for n in row["notches"]]
        drop = min(range(len(ns)), key=lambda i: ns[i].d * ns[i].w)
        keep = [n for i, n in enumerate(ns) if i != drop]
        out = verts_for_notches(row["L"], row["W"], keep)
        assert np.allclose(sorted(map(tuple, verts_for_notches(row["L"], row["W"], ns))), sorted(map(tuple, verts)))
        return out
    raise ValueError(v)


def run(idx):
    from aaf.sim.polygon_geom import polygon_contains
    k, group, fam, path, v = tasks()[idx]
    with h5py.File(path) as f:
        rx = np.asarray(json.loads(f.attrs["receiver_pos"]), float)
        src = np.asarray(json.loads(f.attrs["source_pos"]), float)
        verts = [tuple(map(float, p)) for p in json.loads(f.attrs["verts"])]
    vv = variant_verts(verts, v, group, path.name)
    L, W = max(p[0] for p in vv), max(p[1] for p in vv)
    assert polygon_contains(vv, rx).all() and polygon_contains(vv, src[None]).all(), "receivers / source outside"
    cfg = SimpleNamespace(L=L, W=W, verts=vv, label="{}__{}".format(path.stem, v))
    op = operator(cfg)
    vals, psi, t_eig, n_req = eigen(op, F_PRIMARY)

    def node_of(pts):
        i = np.rint(pts[:, 0] / op["dxx"]).astype(int)
        j = np.rint(pts[:, 1] / op["dxy"]).astype(int)
        i, j = np.clip(i, 0, op["idx"].shape[0] - 1), np.clip(j, 0, op["idx"].shape[1] - 1)
        q = op["idx"][i, j]
        assert (q >= 0).all(), "a point snaps to a solid node"
        snap = np.hypot(i * op["dxx"] - pts[:, 0], j * op["dxy"] - pts[:, 1])
        return q, float(snap.max())

    jr, snap_r = node_of(rx)
    js, snap_s = node_of(src[None])
    eta = ((op["mu"] * op["D"])[:, None] * psi ** 2).sum(0) * op["dA"]
    s, S = pulse()
    g0 = op["mu"][js[0]] * psi[js[0], :1] * op["dA"]
    td0 = td_synthesis(vals[:1], eta[:1], g0, s)[0]                         # the zero mode's exact record
    extra = {}
    if group == "unseen":
        from scripts.p4_9_build_extra_rooms import src2_for
        from scripts.p4_9_fdtd_stencils import geometry, load_room
        r = load_room(path)
        geom, _x, dxx, dxy = geometry(r)
        s2 = np.asarray(src2_for(r, geom, dxx, dxy))
        j2, snap_2 = node_of(s2[None])
        g20 = op["mu"][j2[0]] * psi[j2[0], :1] * op["dA"]
        extra = {"psi_s2": psi[j2[0]], "src2": s2, "td0_s2": td_synthesis(vals[:1], eta[:1], g20, s)[0],
                 "snap_s2": snap_2}
    f_n = 343.0 * np.sqrt(vals) / (2 * np.pi)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"room": path.name, "group": group, "family": fam, "variant": v, "verts": vv, "L": L, "W": W,
            "n_modes_600": int(len(vals)), "eig_s": round(t_eig, 2), "eig_requested": n_req, "n_air": op["n_air"],
            "dxx": op["dxx"], "dxy": op["dxy"], "dA": op["dA"], "area": op["area_q"], "perimeter": op["perimeter"],
            "beta": op["beta"], "asymmetry": op["asym"], "max_rx_snap_m": snap_r, "src_snap_m": snap_s,
            "solved": time.strftime("%Y-%m-%d %H:%M")}
    if GRID:
        # Step 1 realism (D181 E8 / S-2): mode shapes on the full air grid (float32) for measurement positions that
        # are off the dense receivers; air node q sits at (I[q] dxx, J[q] dxy).
        I, J = np.nonzero(op["idx"] >= 0)
        assert (op["idx"][I, J] == np.arange(len(I))).all()
        GOUT.mkdir(parents=True, exist_ok=True)
        np.save(GOUT / "{}__{}__psi.npy".format(path.stem, v), psi.astype(np.float32))
        np.savez_compressed(GOUT / "{}__{}__grid.npz".format(path.stem, v), idx=op["idx"], dxx=op["dxx"],
                            dxy=op["dxy"], k2=vals, eta=eta)
        print("grid", psi.shape, flush=True)
        return
    np.savez_compressed(OUT / "{}__{}.npz".format(path.stem, v), k2=vals, f_n=f_n, eta=eta,
                        psi_r=psi[jr].astype(np.float64), psi_s=psi[js[0]], td0=td0, S=S, rx=rx, src=src,
                        meta=json.dumps(meta), **extra)
    print(json.dumps({k2: meta[k2] for k2 in ("room", "variant", "n_modes_600", "eig_s", "max_rx_snap_m")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--grid", action="store_true", help="store psi on the full air grid (exact/snap10/bbox only)")
    a = ap.parse_args()
    global GRID
    GRID = a.grid
    if a.list:
        t = tasks()
        for i, (k, g, fam, p, v) in enumerate(t):
            print(i, k, g, fam, p.name, v)
        print("{} tasks -> --array=0-{}".format(len(t), len(t) - 1))
        return
    run(a.idx)


if __name__ == "__main__":
    main()
