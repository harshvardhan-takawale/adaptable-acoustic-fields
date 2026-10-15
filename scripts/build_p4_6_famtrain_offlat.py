"""P4-6 (D160(f)): OFF-LATTICE receivers in FAMILY TRAINING rooms, so g2 (PB3, U held out) has a
non-circular training-room fit (D125/D135) -- the family corpus only has receivers on the shared
0.12 + 0.06 k lattice.

Rooms: g2's training split (`p4_6_family_holdU_manifest.json`: rect / L / T / DN, 240 rooms) sorted
by filename, every 12th -> 20 rooms, 5 per family (each family has 60). Fixed by rule.
Receivers: 800 per room drawn uniformly from EVERY in-room 0.02 m FDTD node, with the dense family
lattice's clearances (0.12 m from every wall including notch faces, CORNER_CLEAR from reflex
corners), seed [RX_SEED_OFFLAT, shape_id] -- B6's construction (D99) on the multi-notch geometry of
`build_p4_5_rooms.build_famdense`. About 1 in 9 lands on the old lattice; the fit uses only the rest.
  python scripts/build_p4_6_famtrain_offlat.py --plan
  python scripts/build_p4_6_famtrain_offlat.py --idx 0 --chunk 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.shape_configs import ALPHA, SRC
from aaf.sim.polygon_geom import line_of_sight, polygon_contains
from scripts.build_p4_5_rooms import N_RX, RX_SEED_OFFLAT, _simulate

MANIFEST = "configs/sweeps_2d_mat/p4_6_family_holdU_manifest.json"
OUT = Path("data/track_p4_6_famtrain_offlat")


def rooms():
    from aaf.data.multi_notch import configs_from_rows
    cs = sorted(configs_from_rows(json.load(open(MANIFEST))["configs"], split="train"), key=lambda c: c.filename)
    return cs[::12]


def build(cfg, force=False):
    from aaf.data.multi_notch import reflex_corners, solid_blocks
    from scripts.build_p4_3_demo_rooms import CORNER_CLEAR, WALL_CLEAR
    path, done = OUT / cfg.filename, OUT / (cfg.filename + ".done")
    if done.exists() and path.exists() and not force:
        return "skip"
    dx = 0.02
    nx, dxx = F._fit_axis(cfg.L, dx, "L")
    ny, dxy = F._fit_axis(cfg.W, dx, "W")
    m = solid_blocks(nx, ny, dxx, dxy, cfg.L, cfg.W, cfg.notches)
    extra = [{"type": "mask", "solid": m, "alpha": ALPHA}] if m.any() else None
    geom = F.build_geometry(cfg.L, cfg.W, [ALPHA] * 4, dx=dx, extra_walls=extra)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    want = polygon_contains(cfg.verts, np.stack([ii * dxx, jj * dxy], axis=-1))
    bad = int((geom.air != want).sum())
    if bad:
        raise AssertionError("{}: air mask disagrees with polygon at {} nodes".format(cfg.label, bad))
    P = np.stack([ii.ravel() * dxx, jj.ravel() * dxy], axis=-1)
    keep = geom.air.ravel() & polygon_contains(cfg.verts, P)
    V = cfg.verts
    for (x0, y0), (x1, y1) in zip(V, V[1:] + V[:1]):
        if abs(x1 - x0) < 1e-9:
            keep &= ~((np.abs(P[:, 0] - x0) < WALL_CLEAR) & (P[:, 1] >= min(y0, y1) - WALL_CLEAR)
                      & (P[:, 1] <= max(y0, y1) + WALL_CLEAR))
        else:
            keep &= ~((np.abs(P[:, 1] - y0) < WALL_CLEAR) & (P[:, 0] >= min(x0, x1) - WALL_CLEAR)
                      & (P[:, 0] <= max(x0, x1) + WALL_CLEAR))
    for cx, cy in reflex_corners(cfg.L, cfg.W, cfg.notches):
        keep &= np.hypot(P[:, 0] - cx, P[:, 1] - cy) > CORNER_CLEAR
    pool = P[keep]
    rng = np.random.default_rng([RX_SEED_OFFLAT, int(cfg.shape_id)])
    rx = pool[np.sort(rng.choice(len(pool), size=N_RX, replace=False))]
    H, rx_snap, res, fs, n, loop_s = _simulate(cfg, rx, dx, geom, extra)
    assert H.shape[0] == N_RX and np.abs(rx_snap - rx).max() < 1e-9
    k = (rx_snap - 0.12) / 0.06
    on = float(np.all(np.abs(k - np.round(k)) < 1e-6, axis=1).mean())
    los = line_of_sight(cfg.verts, SRC, rx_snap)
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("ism/H_complex", data=H, compression="gzip", compression_opts=4)
        a = f.attrs
        a["source_pos"] = json.dumps(list(SRC))
        a["receiver_pos"] = json.dumps(rx_snap.tolist())
        a["line_of_sight"] = json.dumps([bool(x) for x in los])
        a["verts"] = json.dumps([list(v) for v in cfg.verts])
        a["L"], a["W"], a["kind"], a["label"] = cfg.L, cfg.W, cfg.kind, cfg.label
        a["shape_id"] = int(cfg.shape_id)
        a["dx"], a["fs"], a["n_time_samples"], a["loop_s"] = dx, fs, n, round(loop_s, 1)
        a["rx_construction"] = ("800 drawn uniformly from every in-room 0.02 m node (famdense clearances), "
                                "seed [{}, shape_id] (P4-6, D160(f))".format(RX_SEED_OFFLAT))
        a["rx_candidate_pool"], a["frac_on_old_lattice"] = int(len(pool)), on
    done.touch()
    print("built {:9s} {:52s} pool {:6d} on-old-lattice {:.3f} {:.0f}s".format(
        cfg.kind, cfg.label, len(pool), on, loop_s), flush=True)
    return "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    work = rooms()
    if a.plan or a.idx is None:
        print(json.dumps({"n": len(work), "rooms": [(c.kind, c.filename) for c in work]}, indent=1))
        return
    for c in work[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
        print(build(c), c.filename, flush=True)


if __name__ == "__main__":
    main()
