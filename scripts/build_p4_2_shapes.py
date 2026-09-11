"""P4-2 Stage 2: FDTD ground truth for the notch family. 75 rooms, SLURM-array friendly.

Generalizes `build_p4_1_lroom.py` from one L-room to the whole family. Two things it keeps
verbatim because they earned their place:

  * the polygon is cut with **`mask`**, never `wall_segments` (D65), and
  * every room is reconstructed analytically from its vertex list and asserted **node-for-node**
    against `geom.air`. That assert fired on P4-1's first run (37500 nodes = exactly twice the
    block, the signature of a mask in the wrong corner) and caught a real error. Here the notch
    indices `i_w = round(w/dx)` and `j_d = round((W-d)/dx)` VARY PER ROOM, so it is not a
    one-time check -- it runs for all 75.

EXACTLY 800 RECEIVERS PER ROOM, and why
---------------------------------------
The trainer stacks targets with `np.stack(H_list)`, which requires every config to have the
SAME receiver count -- a ragged corpus is a hard error, not a degradation. Room areas here span
2.4x (20.7-35.7 m2), so a fixed-step grid would give different counts. Fixing the COUNT instead
means the receiver DENSITY varies by that factor across the family; that is a real limitation
and is recorded rather than hidden. Evaluation builds its own denser per-room grid and is not
bound by the trainer's stacking.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

import aaf.sim.fdtd_2d as F
from aaf.data.shape_configs import ALPHA, SRC, configs_from_rows
from aaf.sim.polygon_geom import (
    assert_extents_match_polygon, line_of_sight, notch_solid_mask, polygon_contains,
)

C = 343.0
DX = 0.02
FS_FOR_DX = {0.05: 12288.0, 0.02: 30720.0, 0.01: 61440.0}
FS = FS_FOR_DX[DX]
N = int(2 * FS)                      # T = 2.000 s, df = 0.5 Hz
BAND_HI_HZ = 300.0
MANIFEST = "configs/sweeps_2d_mat/p4_2_shapes_manifest.json"

N_RX = 800                           # EXACTLY, for every room -- see the module docstring
RX_WALL_CLEAR = 0.12
RX_CORNER_CLEAR = 0.15               # clear of the r^(2/3) reflex singularity
RX_SEED = 20260913


def receivers(cfg, geom, dxx, dxy):
    """Exactly N_RX interior points, seeded so a rebuild is identical."""
    V = cfg.verts
    step = 0.06
    xs = np.arange(RX_WALL_CLEAR, cfg.L, step)
    ys = np.arange(RX_WALL_CLEAR, cfg.W, step)
    P = np.array([[x, y] for x in xs for y in ys], dtype=float)
    P = P[polygon_contains(V, P)]
    # clear of every wall
    keep = np.ones(len(P), dtype=bool)
    for (x0, y0), (x1, y1) in zip(V, V[1:] + V[:1]):
        if abs(x1 - x0) < 1e-9:                                   # vertical wall
            keep &= ~((np.abs(P[:, 0] - x0) < RX_WALL_CLEAR)
                      & (P[:, 1] >= min(y0, y1) - RX_WALL_CLEAR)
                      & (P[:, 1] <= max(y0, y1) + RX_WALL_CLEAR))
        else:                                                     # horizontal wall
            keep &= ~((np.abs(P[:, 1] - y0) < RX_WALL_CLEAR)
                      & (P[:, 0] >= min(x0, x1) - RX_WALL_CLEAR)
                      & (P[:, 0] <= max(x0, x1) + RX_WALL_CLEAR))
    P = P[keep]
    if not cfg.is_rect:                                           # the reflex corner
        P = P[np.hypot(P[:, 0] - cfg.w, P[:, 1] - (cfg.W - cfg.d)) > RX_CORNER_CLEAR]
    # final authority: the solver's own air mask
    i = np.rint(P[:, 0] / dxx).astype(int)
    j = np.rint(P[:, 1] / dxy).astype(int)
    P = P[geom.air[i, j]]
    if len(P) < N_RX:
        raise RuntimeError("{}: only {} candidate receivers, need {}".format(
            cfg.label, len(P), N_RX))
    rng = np.random.default_rng([RX_SEED, cfg.shape_id])
    return P[np.sort(rng.choice(len(P), size=N_RX, replace=False))]


def build_one(cfg, out_dir: Path, force: bool = False) -> str:
    out_path = out_dir / cfg.filename
    done = out_dir / (cfg.filename + ".done")
    if done.exists() and out_path.exists() and not force:
        return "skip"

    nx, dxx = F._fit_axis(cfg.L, DX, "L")
    ny, dxy = F._fit_axis(cfg.W, DX, "W")
    m = notch_solid_mask(nx, ny, dxx, dxy, cfg.W, cfg.d, cfg.w)
    extra = [{"type": "mask", "solid": m, "alpha": ALPHA}] if m.any() else None
    geom = F.build_geometry(cfg.L, cfg.W, [ALPHA] * 4, dx=DX, extra_walls=extra)

    # node-for-node cross-check against the analytic polygon -- mandatory PER ROOM here
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    want = polygon_contains(cfg.verts, np.stack([ii * dxx, jj * dxy], axis=-1))
    bad = int((geom.air != want).sum())
    if bad:
        raise AssertionError(
            "{}: solver air mask disagrees with the polygon at {} nodes -- check the notch "
            "indices".format(cfg.label, bad))

    rx = receivers(cfg, geom, dxx, dxy)
    los = line_of_sight(cfg.verts, SRC, rx)
    t0 = time.time()
    res = F.simulate(cfg.L, cfg.W, [ALPHA] * 4, SRC, rx, dx=DX, fs=FS, n=N, c=C,
                     extra_walls=extra)
    # D65: any reported wall extent must match THIS polygon's wall, not the bounding rectangle
    assert_extents_match_polygon(res["meta"].get("extra_walls", []), cfg.verts)
    hi = int(round(BAND_HI_HZ / (FS / N))) + 1
    H = np.asarray(res["H_complex"])[:, :hi].astype(np.complex64)
    loop_s = time.time() - t0

    with h5py.File(out_path, "w") as f:
        f.create_dataset("ism/H_complex", data=H, compression="gzip", compression_opts=4)
        a = f.attrs
        a["source_pos"] = json.dumps(list(SRC))
        a["receiver_pos"] = json.dumps(np.asarray(res["meta"]["rx_pos_snapped"]).tolist())
        a["line_of_sight"] = json.dumps([bool(x) for x in los])
        a["split"], a["kind"], a["label"] = cfg.split, cfg.kind, cfg.label
        a["alphas"] = json.dumps(list(cfg.alphas))
        a["edge_alphas"] = json.dumps(list(cfg.edge_alphas))
        a["verts"] = json.dumps([list(v) for v in cfg.verts])
        a["L"], a["W"], a["d"], a["w"] = cfg.L, cfg.W, cfg.d, cfg.w
        a["d_hat"], a["w_hat"] = cfg.d_hat, cfg.w_hat
        a["shape_id"] = cfg.shape_id
        a["notch_solid_nodes"] = int(m.sum())
        a["n_air_nodes"] = int(geom.air.sum())
        a["nlos_fraction"] = float(1.0 - los.mean())
        a["solver"] = "fdtd_2d_slf_kw"
        a["dx"], a["dx_x"], a["dx_y"] = DX, dxx, dxy
        a["fs"], a["n_time_samples"] = FS, N
        a["band_hi_hz"], a["n_freq_bins"], a["n_rx"] = BAND_HI_HZ, H.shape[1], int(len(rx))
        a["loop_s"] = round(loop_s, 1)
        a["extra_walls"] = json.dumps(res["meta"].get("extra_walls", []))
    done.touch()
    print("built {:34s} d_hat {:.3f} NLOS {:5.1%} {:.0f}s".format(
        cfg.label, cfg.d_hat, 1 - los.mean(), loop_s), flush=True)
    return "built"


def worklist(manifest: str):
    """The FULL stable list, never filtered on .done -- filtering shrinks the list mid-build and
    races the array index against the config mapping, which silently left 79 of 479 P3-2c
    configs unbuilt."""
    rows = json.load(open(manifest))["configs"]
    cfgs = configs_from_rows(rows)
    seen, out = set(), []
    for c in cfgs:
        if c.filename not in seen:
            seen.add(c.filename)
            out.append(c)
    return sorted(out, key=lambda c: c.filename)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out-dir", default="data/track_p4_2_shapes")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    work = worklist(a.manifest)
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    if a.plan or a.idx is None:
        n_pend = sum(1 for c in work if not (d / (c.filename + ".done")).exists())
        print(json.dumps({"n_total": len(work), "n_pending": n_pend, "chunk": a.chunk,
                          "array_range": "0-{}".format((len(work) - 1) // a.chunk)}))
        return 0
    for c in work[a.idx * a.chunk:(a.idx + 1) * a.chunk]:
        print(build_one(c, d, force=a.force), c.filename, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
